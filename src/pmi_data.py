import os
import os.path as osp
import pickle
import shutil
from typing import Any, Callable, List, Optional
from .manuall_split import ManualLinkSplit
from copy import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce
from torch_geometric.transforms import RandomLinkSplit
from lightning.pytorch import LightningDataModule
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
)
from torch_geometric.loader import LinkNeighborLoader


class ProteinPeptideInteraction(InMemoryDataset):
    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        model_name=None,
        split_method=True,
        pooling_method="mean",
        random_feat=False,
    ):
        self.model_name = model_name
        self.split_method = split_method
        self.random_feat = random_feat
        self.pooling_method = pooling_method
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def split_methods(self):
        return {
            "a": "pep_cls_net_cluster_0107_05.csv",
            "b": "pep_cls_net_cluster_wide_0107_04pep.csv",
            "c": "pep_cls_net_cluster_wide_0107_04pair.csv",
        }

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "prot_esm",
            "pep_esm_finetune",  # 4000,16500,19500,21500
            "prot_vocab.csv",
            "pep_vocab.csv",
            "protein_protein_link.csv",
            "protein_peptide_link.csv",
            "split.csv",
        ]

    @property
    def processed_file_names(self) -> str:
        file_name = "data"

        # add split method
        file_name += "_split_{}".format(self.split_method)

        if self.use_ppi:
            file_name += "_useppi"
        else:
            file_name += "_noppi"

        file_name += "_{0}.pt".format(self.pooling_method)
        return file_name

    def process(self):
        import numpy as np
        import pandas as pd

        (
            protein_feat_path,
            peptide_feat_path,
            protein_vocab,
            peptide_vocab,
            protein_protein_link_path,
            protein_peptide_link_path,
            split_path,
        ) = self.raw_paths
        # load peptide and protein data
        if self.pep_feat_src != "finetune":
            peptide_feat_path.replace("_finetune", "")

        with open(
            protein_feat_path + "_{0}.pickle".format(self.pooling_method),
            "rb",
        ) as fin:
            protein_feat = pickle.load(fin)

        with open(
            peptide_feat_path + "_{0}.pickle".format(self.pooling_method),
            "rb",
        ) as fin:
            peptide_feat = pickle.load(fin)

        # load vocab data
        # using pretrained embedding
        prot_vocab_df = pd.read_csv(protein_vocab)
        pep_vocab_df = pd.read_csv(peptide_vocab)

        protein_feat_x = np.zeros((len(prot_vocab_df), 1280))
        peptide_feat_x = np.zeros((len(pep_vocab_df), 1280))

        for _, row in prot_vocab_df.iterrows():
            protein_feat_x[row["prot_idx"]] = protein_feat[row["prot_seq"]]

        for _, row in pep_vocab_df.iterrows():
            peptide_feat_x[row["pep_idx"]] = peptide_feat[row["pep_seq"]]

        data = HeteroData()
        data["prot"].node_id = torch.arange(len(prot_vocab_df))
        data["pep"].node_id = torch.arange(
            len(pep_vocab_df)
        )  # Add the node features and edge indices

        # protein_feat_x = sk_normalize(protein_feat_x, norm="l1", axis=0)
        # peptide_feat_x = sk_normalize(peptide_feat_x, norm="l1", axis=0)

        # add protein data
        if self.random_feat:
            data["prot"].x = torch.rand(len(prot_vocab_df), 1280).to(
                torch.float
            )
        else:
            data["prot"].x = torch.from_numpy(protein_feat_x).to(torch.float)
        # data["prot"].x = torch.zeros(len(prot_vocab_df), 1280).to(torch.float)

        # add peptide data
        if self.random_feat:
            data["pep"].x = torch.rand(len(pep_vocab_df), 1280).to(torch.float)
        else:
            data["pep"].x = torch.from_numpy(peptide_feat_x).to(torch.float)
        # data["pep"].x = torch.zeros(len(pep_vocab_df), 1280).to(torch.float)
        data.num_pep_nodes = len(pep_vocab_df)
        data.num_prot_nodes = len(prot_vocab_df)

        # one-hot for embedding learning
        # data["prot"].x = torch.arange(0, len(prot_vocab_df))
        # data["pep"].x = torch.arange(0, len(pep_vocab_df))

        # load connections,splited
        # edges_df = pd.read_csv(
        #     "./data/yipin_protein_peptide/raw/pep_cls_net_cluster_0107_04pair.csv",
        # )
        edges_df = pd.read_csv(
            os.path.join(
                self.root, "raw", self.split_methods[self.split_method]
            )
        )
        edges = np.array(edges_df.loc[:, ["pep_idx", "prot_idx"]]).transpose()
        # src = edges_df["pep_idx"].tolist()
        # dst = edges_df["prot_idx"].tolist()
        #     edges_df["Fold{}_split".format(self.fold_id)] == "test"
        # )
        data["pep", "bind", "prot"].edge_index = torch.tensor(edges)
        data["pep", "bind", "prot"].edge_label = torch.zeros(
            len(edges_df), dtype=torch.long
        )

        data.perm = np.hstack(
            [
                np.argwhere(
                    (edges_df["Fold{}_split".format(self.fold_id)] == "train")
                ).flatten(),
                np.argwhere(
                    (edges_df["Fold{}_split".format(self.fold_id)] == "test")
                ).flatten(),
            ]
        )
        data.num_val = (
            np.argwhere(
                (edges_df["Fold{}_split".format(self.fold_id)] == "test")
            )
            .flatten()
            .shape[0]
        )

        # add protein protein link
        if self.use_ppi:
            edges_df = pd.read_csv(
                os.path.join(
                    self.root,
                    "raw",
                    "STRING_physical_positive_bothin_pairs.csv",
                )
            )
            edges = np.array(
                edges_df.loc[:, ["Prot_A_idx", "Prot_B_idx"]]
            ).transpose()
            _edge_index, _edge_label = coalesce(
                torch.tensor(edges),
                torch.zeros(len(edges_df), dtype=torch.long),
                reduce="mean",
            )

            data["prot", "bind", "prot"].edge_index = _edge_index
            data["prot", "bind", "prot"].edge_label = _edge_label

        # data["prot", "bind", "prot"] = data["prot", "bind", "prot"].coalesce()

        # data.train_mask = train_mask
        # data.val_mask = val_mask

        # data = data.to_homogeneous()
        # edge_index_new = torch.zeros_like(data.edge_index)
        # edge_index_new[0] = data.edge_index[1]
        # edge_index_new[1] = data.edge_index[0]
        # data.edge_index = edge_index_new

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def get_train_val_data(self, data):
        train_data = self._split_by_mask(data, data.train_mask)
        val_data = self._split_by_mask(data, data.val_mask)
        return train_data, val_data


class PLProteinPeptideInteraction(LightningDataModule):
    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        model_name=None,
        split_method="a",
        use_ppi=True,
        pooling_method="mean",
        remove_cache=True,
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        num_neighbors=[20, 10],
        batch_size=64,
        fold_id=0,
        pep_feat="",
        random_feat=False,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.model_name = model_name
        self.split_method = split_method
        self.use_ppi = use_ppi
        self.pooling_method = pooling_method
        self.remove_cache = remove_cache
        self.num_val = num_val
        self.num_test = num_test
        self.disjoint_train_ratio = disjoint_train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.add_negative_train_samples = add_negative_train_samples
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.fold_id = fold_id
        self.pep_feat = pep_feat
        self.random_feat = random_feat

        if remove_cache:
            cache_dir = os.path.join(self.root, "processed")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
        dataset = ProteinPeptideInteraction(
            root=self.root,
            pre_transform=self.pre_transform,
            model_name=self.model_name,
            split_method=self.split_method,
            use_ppi=self.use_ppi,
            pooling_method=self.pooling_method,
            fold_id=self.fold_id,
            pep_feat_src=pep_feat,
            random_feat=self.random_feat,
        )

        self.data = dataset[0]
        self.num_prot = self.data["prot"].num_nodes
        self.num_pep = self.data["pep"].num_nodes

        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        self.data = T.ToUndirected(reduce="mean")(self.data)
        # self.data = T.ToUndirected()(self.data)
        # self.train_data, self.val_data = dataset.get_train_val_data(self.data)

        # train_transform = T.RandomLinkSplit(
        #     num_val=0,
        #     num_test=0,
        #     disjoint_train_ratio=self.disjoint_train_ratio,
        #     # neg_sampling_ratio=self.neg_sampling_ratio,
        #     add_negative_train_samples=False,
        #     edge_types=("pep", "bind", "prot"),
        #     rev_edge_types=("prot", "rev_bind", "pep"),
        # )
        # self.train_data, _, _ = train_transform(self.train_data)

        # val_transform = T.RandomLinkSplit(
        #     num_val=0.0,
        #     num_test=0.0,
        #     disjoint_train_ratio=0.0,
        #     neg_sampling_ratio=self.neg_sampling_ratio,
        #     add_negative_train_samples=True,
        #     edge_types=("pep", "bind", "prot"),
        #     rev_edge_types=("prot", "rev_bind", "pep"),
        # )
        # self.val_data, _, _ = val_transform(self.val_data)
        # self.test_data = copy(self.val_data)

        transform = ManualLinkSplit(
            perm=self.data.perm,
            num_val=self.data.num_val,
            disjoint_train_ratio=self.disjoint_train_ratio,
            neg_sampling_ratio=self.neg_sampling_ratio,
            is_undirected=True,
            add_negative_train_samples=False,
            edge_types=("pep", "bind", "prot"),
            rev_edge_types=("prot", "rev_bind", "pep"),
        )
        self.train_data, self.val_data = transform(self.data)

        self.test_data = copy(self.val_data)

        # self.gen_splits()

    # def gen_splits(
    #     self,
    # ):
    #     self.train_data = self.gen_single_split(
    #         copy(self.data), self.data.train_mask
    #     )
    #     self.val_data = self.gen_single_split(
    #         copy(self.data), self.data.val_mask
    #     )
    #     self.test_data = self.gen_single_split(
    #         copy(self.data), self.data.test_mask
    #     )

    # def gen_single_split(self, in_data, mask):
    #     in_data["pep", "bind", "prot"].edge_index = self.data[
    #         "pep", "bind", "prot"
    #     ].edge_index[:, mask]
    #     in_data["pep", "bind", "prot"].edge_label_index = self.data[
    #         "pep", "bind", "prot"
    #     ].edge_index[:, mask]
    #     in_data["pep", "bind", "prot"].edge_label = self.data[
    #         "pep", "bind", "prot"
    #     ].edge_label[mask]
    #     in_data["prot", "rev_bind", "pep"].edge_index = self.data[
    #         "prot", "rev_bind", "pep"
    #     ].edge_index[:, mask]
    #     in_data["prot", "rev_bind", "pep"].edge_label = self.data[
    #         "prot", "rev_bind", "pep"
    #     ].edge_label[mask]
    #     return in_data

    def train_dataloader(self):
        # Define seed edges:
        edge_label_index = self.train_data[
            "pep", "bind", "prot"
        ].edge_label_index
        edge_label = self.train_data["pep", "bind", "prot"].edge_label
        return LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=self.num_neighbors,
            neg_sampling_ratio=self.neg_sampling_ratio,
            edge_label_index=(("pep", "bind", "prot"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        edge_label_index = self.val_data["pep", "bind", "prot"].edge_label_index

        edge_label = self.val_data["pep", "bind", "prot"].edge_label
        return LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(("pep", "bind", "prot"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
        )

    def test_dataloader(self):
        edge_label_index = self.test_data[
            "pep", "bind", "prot"
        ].edge_label_index

        edge_label = self.test_data["pep", "bind", "prot"].edge_label
        return LinkNeighborLoader(
            data=self.test_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(("pep", "bind", "prot"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
        )
