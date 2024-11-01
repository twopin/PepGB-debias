import os
import os.path as osp
import pickle
from copy import copy
import shutil
from typing import Any, Callable, List, Optional

import torch
import torch_geometric.transforms as T
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
        random_split=True,
        use_ppi=True,
        pooling_method="mean",
    ):
        self.model_name = model_name
        self.random_split = random_split
        self.use_ppi = use_ppi
        self.pooling_method = pooling_method
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "prot_esm",
            "pep_esm",
            "prot_vocab.csv",
            "pep_vocab.csv",
            "protein_protein_link.csv",
            "protein_peptide_link.csv",
            "split.csv",
        ]

    @property
    def processed_file_names(self) -> str:
        file_name = "data"
        if self.random_split:
            file_name += "_randomsplit"
        else:
            file_name += "_customsplit"

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
        data["prot"].x = torch.from_numpy(protein_feat_x).to(torch.float)

        # add peptide data
        data["pep"].x = torch.from_numpy(peptide_feat_x).to(torch.float)

        # one-hot for embedding learning
        # data["prot"].x = torch.arange(0, len(prot_vocab_df))
        # data["pep"].x = torch.arange(0, len(pep_vocab_df))

        # load connections no split
        edges_df = pd.read_csv(
            "./data/yipin_protein_peptide/raw/pep_cls_net.csv", sep="\t"
        )
        pep_prot_edges = [
            eval(_edge) for _edge in edges_df["pep_src_prot_dst"].tolist()
        ]
        pep_prot_edges = torch.tensor(np.transpose(pep_prot_edges))
        data["pep", "bind", "prot"].edge_index = pep_prot_edges
        data["pep", "bind", "prot"].edge_label = torch.zeros(
            len(edges_df), dtype=torch.long
        )

        # load connections,splited
        # edges_df = pd.read_csv(
        #     "./data/yipin_protein_peptide/raw/pep_cls_net_cluster_0107_05.csv",
        # )
        # src = edges_df["pep_idx"].tolist()
        # dst = edges_df["prot_idx"].tolist()
        # train_mask = torch.tensor(edges_df["Fold1_split"] == "train")
        # val_mask = torch.tensor(edges_df["Fold1_split"] == "test")
        # test_mask = torch.tensor(edges_df["Fold1_split"] == "test")
        # data["pep", "bind", "prot"].edge_index = torch.tensor([src, dst])
        # data["pep", "bind", "prot"].edge_label = torch.zeros(
        #     len(edges_df), dtype=torch.long
        # )
        # data.train_mask = train_mask
        # data.val_mask = val_mask
        # data.test_mask = test_mask

        # data = data.to_homogeneous()
        # edge_index_new = torch.zeros_like(data.edge_index)
        # edge_index_new[0] = data.edge_index[1]
        # edge_index_new[1] = data.edge_index[0]
        # data.edge_index = edge_index_new

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class PLProteinPeptideInteraction(LightningDataModule):
    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        model_name=None,
        random_split=True,
        use_ppi=True,
        pooling_method="mean",
        remove_cache=True,
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        num_neighbors=[20, 10],
        batch_size=64,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.model_name = model_name
        self.random_split = random_split
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
        if remove_cache:
            cache_dir = os.path.join(self.root, "processed")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
        dataset = ProteinPeptideInteraction(
            self.root,
            self.pre_transform,
            self.model_name,
            self.random_split,
            self.use_ppi,
            self.pooling_method,
        )

        self.data = dataset[0]
        self.num_prot = self.data["prot"].num_nodes
        self.num_pep = self.data["pep"].num_nodes

        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        self.data = T.ToUndirected()(self.data)
        # self.train_data, self.val_data, self.test_data = self.gen_splits()
        # self.gen_splits()

        transform = T.RandomLinkSplit(
            num_val=self.num_val,
            num_test=self.num_test,
            disjoint_train_ratio=self.disjoint_train_ratio,
            neg_sampling_ratio=self.neg_sampling_ratio,
            add_negative_train_samples=self.add_negative_train_samples,
            edge_types=("pep", "bind", "prot"),
            rev_edge_types=("prot", "rev_bind", "pep"),
        )
        self.train_data, self.val_data, self.test_data = transform(self.data)
        with open("tmp/data_back.pickle", "wb") as fout:
            pickle.dump(self.test_data, fout)

    def gen_splits(
        self,
    ):
        self.train_data = self.gen_single_split(
            copy(self.data), self.data.train_mask
        )
        self.val_data = self.gen_single_split(
            copy(self.data), self.data.val_mask
        )
        self.test_data = self.gen_single_split(
            copy(self.data), self.data.test_mask
        )

    def gen_single_split(self, in_data, mask):
        in_data["pep", "bind", "prot"].edge_index = self.data[
            "pep", "bind", "prot"
        ].edge_index[:, mask]
        in_data["pep", "bind", "prot"].edge_label_index = self.data[
            "pep", "bind", "prot"
        ].edge_index[:, mask]
        in_data["pep", "bind", "prot"].edge_label = self.data[
            "pep", "bind", "prot"
        ].edge_label[mask]
        in_data["prot", "rev_bind", "pep"].edge_index = self.data[
            "prot", "rev_bind", "pep"
        ].edge_index[:, mask]
        in_data["prot", "rev_bind", "pep"].edge_label = self.data[
            "prot", "rev_bind", "pep"
        ].edge_label[mask]
        return in_data

    def train_dataloader(self):
        # Define seed edges:
        edge_label_index = self.train_data[
            "pep", "bind", "prot"
        ].edge_label_index
        edge_label = self.train_data["pep", "bind", "prot"].edge_label
        return LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=self.num_neighbors,
            # disjoint=self.disjoint_train_ratio,
            neg_sampling_ratio=self.neg_sampling_ratio,
            edge_label_index=(
                ("pep", "bind", "prot"),
                edge_label_index,
            ),
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        edge_label_index = self.val_data["pep", "bind", "prot"].edge_label_index

        edge_label = self.val_data["pep", "bind", "prot"].edge_label
        print(edge_label)
        return LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(("pep", "bind", "prot"), edge_label_index),
            neg_sampling_ratio=self.neg_sampling_ratio,
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
            edge_label_index=(
                ("pep", "bind", "prot"),
                edge_label_index,
            ),
            edge_label=edge_label,
            neg_sampling_ratio=self.neg_sampling_ratio,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
        )
