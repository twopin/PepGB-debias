import argparse
import os

import mlflow
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import coalesce, negative_sampling
from torch.optim.lr_scheduler import CosineAnnealingLR


from src.model import GNN
from src.data import PLPMI


def find_intersection(A, B):
    A = A.transpose()
    B = B.transpose()

    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    return np.array([x for x in aset & bset])


def get_args():
    parser = argparse.ArgumentParser("train pmi")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="./data/yipin_protein_peptide/",
        help="path to save logs",
    )
    parser.add_argument(
        "--log_root_dir",
        type=str,
        default="./tmp/training_logs/pmi",
        help="path to save logs",
    )

    parser.add_argument(
        "--split_method",
        type=str,
        default="random",
        choices=["random", "similarity", "1", "2"],
        help="methods to split data",
    )
    parser.add_argument("--use_random_feat", action="store_true")
    parser.add_argument(
        "--input_feat_dim",
        type=int,
        default=1280,
        help="dimension of raw input feat",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="dimension for hidden layers",
    )
    parser.add_argument(
        "--num_gnn_layers",
        type=int,
        default=1,
        help="num of of gnn layers",
    )
    parser.add_argument(
        "--dropout_ratio",
        type=float,
        default=0.5,
        help="value for dropout",
    )
    parser.add_argument(
        "--disjoint_train_ratio",
        type=float,
        default=0.3,
        help="value for edges used only for message passing",
    )
    parser.add_argument(
        "--neg_sampling_ratio",
        type=float,
        default=5.0,
        help="sampling ratio for negtive samples",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio of validation data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="batch size for dataloader",
    )
    parser.add_argument(
        "--num_neighbors",
        nargs="+",
        default=[10],
        help="num of neighbors for sampleing edges",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="value of learning rate",
    )
    parser.add_argument(
        "--epoch", type=int, default=20, help="epoch for training"
    )
    parser.add_argument(
        "--gnn_method",
        type=str,
        default="gat_conv",
        help="type of gnn layer",
        choices=["sage_conv", "gat_conv", "gin_conv"],
    )
    parser.add_argument("--ala_test", action="store_true")
    parser.add_argument("--add_skip_connection", action="store_true")
    parser.add_argument("--add_self_loops", action="store_true")
    parser.add_argument(
        "--dropout_method",
        type=str,
        default="DropMessage",
        choices=["Dropout", "DropEdge", "DropNode" "DropMessage"],
        help="methods for apply dropout",
    )
    parser.add_argument(
        "--esm_feat_epoch",
        type=int,
        default=21500,
        help="epoch for saving checkpoint when finetuning esm",
    )
    return parser.parse_args()


def extract_data(df, feat_dict, stage="train"):
    stage_df = df[df["flag"] == stage]
    pep_list = stage_df["pep_seq1"].unique().tolist()
    ordered_pep_dict = {_pep_str: i for i, _pep_str in enumerate(pep_list)}
    ordered_pep_list = list(ordered_pep_dict.keys())
    # new id for edges
    stage_df["pep_seq1_idx_ordered"] = stage_df["pep_seq1"].apply(
        lambda x: ordered_pep_dict[x]
    )
    stage_df["pep_seq2_idx_ordered"] = stage_df["pep_seq2"].apply(
        lambda x: ordered_pep_dict[x]
    )

    # stack feat array
    feat_x = np.vstack([feat_dict[_pep] for _pep in ordered_pep_list])
    feat_x = torch.from_numpy(feat_x).float()

    # load postive edge only
    if stage == "train":
        stage_df = stage_df[stage_df["label"] == 1]

    edges = torch.tensor(
        np.array(
            stage_df.loc[:, ["pep_seq1_idx_ordered", "pep_seq2_idx_ordered"]]
        ).transpose(),
    )
    if stage == "train":
        edge_label = torch.zeros(len(stage_df), dtype=torch.long)
    else:
        edge_label = torch.tensor(stage_df["label"].tolist(), dtype=torch.long)
    return Data(x=feat_x, edge_index=edges, edge_label=edge_label)


def add_neg_test_edge(orig_edges, neg_sampling_ratio):
    test_edges_offset = orig_edges - 80
    num_edges = test_edges_offset.shape[1]
    test_neg_edges = negative_sampling(
        test_edges_offset, num_neg_samples=num_edges * neg_sampling_ratio
    )
    return test_neg_edges


def load_data_0112(
    data_root,
    esm_feat_epoch,
    use_random_feat=False,
    ala_id=1,
    disjoint_train_ratio=0.3,
    val_ratio=0.2,
    neg_sampling_ratio=5.0,
):
    # edges
    edges_df = pd.read_csv(
        os.path.join(data_root, "new_data_0112/PMI_all_edges_0112.csv")
    )
    edges_df = edges_df[edges_df["label"] == 1]
    train_edges_df = edges_df[edges_df["flag"] == "assay1"]
    train_edges_df["phase"] = ["train"] * len(train_edges_df)
    train_nodes = set(
        list(train_edges_df["pep_idx1"].unique())
        + list(train_edges_df["pep_idx2"].unique())
    )
    num_train_nodes = len(train_nodes)
    if ala_id == 1:
        test_edges_df = edges_df[edges_df["flag"] == "assay_2"]
        test_nodes = set(
            list(test_edges_df["pep_idx1"].unique())
            + list(test_edges_df["pep_idx2"].unique())
        )
        num_test_nodes = len(test_nodes)
        test_nodes = test_nodes.difference(train_nodes)
        test_edges_df["phase"] = ["test"] * len(test_edges_df)
    else:
        test_edges_df = edges_df[edges_df["flag"] == "assay_3"]
        test_nodes = set(
            list(test_edges_df["pep_idx1"].unique())
            + list(test_edges_df["pep_idx2"].unique())
        )
        num_test_nodes = len(test_nodes)
        test_nodes = test_nodes.difference(train_nodes)
        test_edges_df["phase"] = ["test"] * len(test_edges_df)
    edges_df = pd.concat([train_edges_df, test_edges_df])
    all_nodes = list(train_nodes) + list(test_nodes)

    nodes_mapping = {_node: i for i, _node in enumerate(all_nodes)}
    edges_df["pep_idx1_ordered"] = edges_df["pep_idx1"].apply(
        lambda x: nodes_mapping[x]
    )
    edges_df["pep_idx2_ordered"] = edges_df["pep_idx2"].apply(
        lambda x: nodes_mapping[x]
    )

    nodes_id_to_seq_orig = {}
    for _, row in edges_df.iterrows():
        if row["pep_idx1_ordered"] not in nodes_id_to_seq_orig.keys():
            nodes_id_to_seq_orig[row["pep_idx1_ordered"]] = row["pep_seq1"]
        if row["pep_idx2_ordered"] not in nodes_id_to_seq_orig.keys():
            nodes_id_to_seq_orig[row["pep_idx2_ordered"]] = row["pep_seq2"]
    nodes_id_to_seq = {
        i: nodes_id_to_seq_orig[i] for i in range(len(nodes_id_to_seq_orig))
    }
    nodes_ordered = list(nodes_id_to_seq.values())
    pep_list = list(
        set(edges_df["pep_seq1"].tolist() + edges_df["pep_seq2"].tolist())
    )

    # load esm feats
    if not use_random_feat:
        with open(
            os.path.join(
                data_root,
                "new_data_0112",
                "new_feature_0112",
                "pep_PMI_0112_finetune_len_{0}_mean.pickle".format(
                    esm_feat_epoch
                ),
            ),
            "rb",
        ) as fin:
            feat_dict = pickle.load(fin)
    else:
        feat_dict = {_pep: np.random.random(1280) for _pep in pep_list}
    # feats for training
    train_feats = torch.from_numpy(
        np.vstack(
            [feat_dict[_pep] for _pep in nodes_ordered[: len(train_nodes)]]
        )
    ).float()
    test_feats = torch.from_numpy(
        np.vstack(
            [feat_dict[_pep] for _pep in nodes_ordered[len(train_nodes) :]]
        )
    ).float()
    # return edges_df, feats

    # # load train edges
    # train_edges_df = pd.read_csv(
    #     os.path.join(
    #         data_root, "PMI_0111/PMI_rank_train_0111_f{}.csv".format(fold_id)
    #     )
    # )
    train_edges_df = edges_df[edges_df["phase"] == "train"]
    train_edges = torch.from_numpy(
        np.array(
            train_edges_df.loc[:, ["pep_idx1_ordered", "pep_idx2_ordered"]]
        ).transpose()
    )
    train_data_orig = Data(
        x=train_feats,
        edge_index=train_edges,
        edge_label=torch.zeros(len(train_edges_df), dtype=torch.long),
    )
    train_split_transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=0.0,
        is_undirected=False,
        disjoint_train_ratio=disjoint_train_ratio,
        add_negative_train_samples=False,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    train_data, val_data, _ = train_split_transform(train_data_orig)

    # # load test edges
    test_edges_df = edges_df[edges_df["phase"] == "test"]
    test_edges = torch.from_numpy(
        np.array(
            test_edges_df.loc[:, ["pep_idx1_ordered", "pep_idx2_ordered"]]
        ).transpose()
    )

    neg_edges_labeled, neg_edges_mp = neg_sampling_for_ala(
        train_edges,
        test_edges,
        neg_sampling_ratio,
        neg_sampling_ratio * 2,
        num_train_nodes,
        num_test_nodes,
    )
    test_edge_label_index = torch.cat([test_edges, neg_edges_labeled], dim=-1)
    test_edge_label = torch.cat(
        [
            torch.ones(test_edges.shape[1], dtype=torch.long),
            torch.zeros(neg_edges_labeled.shape[1], dtype=torch.long),
        ],
        dim=-1,
    )
    test_data = Data(
        x=torch.cat([train_feats, test_feats], dim=0),
        edge_index=torch.cat([train_edges, neg_edges_mp], dim=-1),
        edge_label=test_edge_label,
        edge_label_index=test_edge_label_index,
    )
    with open("test_orig_data_ala_2.pickle", "wb") as fout:
        pickle.dump(
            {
                "edges_df": test_edges_df,
                "test_edge_label": test_edge_label.numpy(),
                "test_edge_label_index": test_edge_label_index.numpy(),
            },
            fout,
        )

    # test_neg_edges = add_neg_test_edge(test_edges, neg_sampling_ratio)
    # num_test_edges_neg = test_neg_edges.shape[1]
    # # load edges in training
    # # test_data = Data(
    # #     x=feats,
    # #     edge_index=train_edges,
    # #     edge_label=torch.cat(
    # #         [
    # #             torch.ones(num_test_edges_pos, dtype=torch.long),
    # #             torch.zeros(num_test_edges_neg, dtype=torch.long),
    # #         ],
    # #         dim=0,
    # #     ),
    # #     edge_label_index=torch.cat([test_edges, test_neg_edges], dim=-1),
    # # )

    # test_data = Data(
    #     x=feats,
    #     edge_index=train_edges,
    #     edge_label=torch.zeros(num_test_edges_pos, dtype=torch.long),
    #     edge_label_index=test_edges,
    # )

    # return train_data, val_data, edges_all
    return train_data, val_data, test_data


def load_data_with_fold(
    data_root,
    fold_id,
    esm_feat_epoch,
    use_random_feat=False,
    disjoint_train_ratio=0.3,
    val_ratio=0.2,
    neg_sampling_ratio=5.0,
):
    # load_vocab
    vocab_df = vocab_df = pd.read_csv(
        os.path.join(
            data_root, "PMI_0111/PMI_random_vocab_0111_f{}.csv".format(fold_id)
        )
    )
    pep_list = vocab_df["pep_seq"].tolist()

    # load esm feats
    if not use_random_feat:
        with open(
            os.path.join(
                data_root,
                "pepPMI_feat",
                "pepPMI_esm_finetune_len_{0}_mean.pickle".format(
                    esm_feat_epoch
                ),
            ),
            "rb",
        ) as fin:
            feat_dict = pickle.load(fin)
    else:
        feat_dict = {_pep: np.random.random(1280) for _pep in pep_list}
    feats = torch.from_numpy(
        np.vstack([feat_dict[_pep] for _pep in pep_list])
    ).float()

    # load train edges
    train_edges_df = pd.read_csv(
        os.path.join(
            data_root, "PMI_0111/PMI_rank_train_0111_f{}.csv".format(fold_id)
        )
    )
    train_edges = torch.from_numpy(
        np.array(train_edges_df.loc[:, ["pep_idx1", "pep_idx2"]]).transpose()
    )
    train_data_orig = Data(
        x=feats,
        edge_index=train_edges,
        edge_label=torch.zeros(len(train_edges_df), dtype=torch.long),
    )

    train_split_transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=0.0,
        is_undirected=False,
        disjoint_train_ratio=disjoint_train_ratio,
        add_negative_train_samples=False,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    train_data, val_data, _ = train_split_transform(train_data_orig)

    # load test edges
    test_edges_df = pd.read_csv(
        os.path.join(
            data_root, "PMI_0111/PMI_rank_test_0111_f{}.csv".format(fold_id)
        )
    )
    num_test_edges_pos = len(test_edges_df)
    test_edges = torch.from_numpy(
        np.array(test_edges_df.loc[:, ["pep_idx1", "pep_idx2"]]).transpose()
    )
    test_neg_edges = add_neg_test_edge(test_edges, neg_sampling_ratio)
    num_test_edges_neg = test_neg_edges.shape[1]

    # load edges in training
    # test_data = Data(
    #     x=feats,
    #     edge_index=train_edges,
    #     edge_label=torch.cat(
    #         [
    #             torch.ones(num_test_edges_pos, dtype=torch.long),
    #             torch.zeros(num_test_edges_neg, dtype=torch.long),
    #         ],
    #         dim=0,
    #     ),
    #     edge_label_index=torch.cat([test_edges, test_neg_edges], dim=-1),
    # )

    test_data = Data(
        x=feats,
        edge_index=train_edges,
        edge_label=torch.zeros(num_test_edges_pos, dtype=torch.long),
        edge_label_index=test_edges,
    )

    return train_data, val_data, test_data


def load_data(
    data_root,
    split_method,
    esm_feat_epoch,
    use_random_feat,
    disjoint_train_ratio,
):
    """loda edges, split and esm feat

    Args:
        data_root (str): path to data root
        split_method (str): split method, random split or by similarity
        esm_feat_epoch (int): epoch of finetuning chekcpoint
        use_random_feat (bool): whether to use random feat
        disjoint_train_ratio (float): ratios of edges used for message passing
        only

    """

    # mapping from split method to edge file
    splits_dict = {
        "random": "pmi_random_wide_pairs.csv",
        "similarity": "pmi_sim_wide_pairs.csv",
    }
    # read edges df
    edges_df = pd.read_csv(os.path.join(data_root, splits_dict[split_method]))

    # set id of pep from index 0
    pep_list = edges_df["pep_seq1"].unique().tolist()
    if not use_random_feat:
        with open(
            os.path.join(
                data_root,
                "pepPMI_feat",
                "pepPMI_esm_finetune_len_{0}_mean.pickle".format(
                    esm_feat_epoch
                ),
            ),
            "rb",
        ) as fin:
            feat_dict = pickle.load(fin)
    else:
        feat_dict = {_pep: np.random.random(1280) for _pep in pep_list}

    train_data_orig = extract_data(edges_df, feat_dict, "train")
    test_data = extract_data(edges_df, feat_dict, "test")
    split_transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.0,
        is_undirected=False,
        disjoint_train_ratio=disjoint_train_ratio,
        add_negative_train_samples=False,
    )

    train_data, val_data, _ = split_transform(train_data_orig)
    return train_data, val_data, test_data


def neg_sampling_for_ala(
    train_edges,
    test_edges,
    neg_sampling_ratio,
    neg_mp_ratio,
    num_train_nodes,
    num_test_nodes,
):
    edges_all = torch.cat([train_edges, test_edges], dim=-1)
    num_pos_edges = test_edges.shape[1]
    exp_ratio = num_test_nodes / (num_train_nodes + num_test_nodes)
    num_edges_to_sample = int(
        num_pos_edges * (neg_sampling_ratio + neg_mp_ratio) + 100
    )
    num_gen = int(
        num_pos_edges * (neg_sampling_ratio + neg_mp_ratio) / exp_ratio
    )
    neg_labeled = int(num_pos_edges * neg_sampling_ratio)
    neg_edges_orig = negative_sampling(edges_all, num_neg_samples=num_gen)
    sample_done = False
    while not sample_done:
        neg_edges = torch.cat(
            [
                neg_edges_orig[:, neg_edges_orig[0] > num_train_nodes],
                neg_edges_orig[:, neg_edges_orig[1] > num_train_nodes],
            ],
            dim=1,
        )
        if neg_edges.shape[1] > num_edges_to_sample:
            neg_edges_labeled = neg_edges[:, :neg_labeled]
            neg_edges_mp = neg_edges[:, neg_labeled:num_edges_to_sample]
            sample_done = True
        else:
            neg_edges_orig = negative_sampling(
                edges_all, num_neg_samples=num_gen
            )
    return neg_edges_labeled, neg_edges_mp


def load_data_feats_ala(
    data_root,
    esm_feat_epoch,
    split_method,
    disjoint_train_ratio,
    val_ratio,
    neg_sampling_ratio,
):
    """loda edges, split and esm feat

    Args:
        data_root (str): path to data root
        esm_feat_epoch (int): epoch of finetuning chekcpoint
        split_method (int): id of test data

    """

    split_method = int(split_method)
    # mapping from split method to edge file
    test_data_dict = {
        1: "PMI_1_ALAscanning_includingPMI.csv",
        2: "PMI_2_ALAscanning_includingPMI.csv",
    }

    # read edges df
    test_edges_df = pd.read_csv(
        os.path.join(data_root, test_data_dict[split_method])
    )
    num_peps_test = len(test_edges_df["pep_seq1"].unique())

    train_edges_df = pd.read_csv(
        os.path.join(data_root, "PMI_0111/PMI_rank_all_0111.csv")
    )
    train_id_to_pep_dict_orig = {}
    for _, row in train_edges_df.iterrows():
        pep_id = eval(row["edges"])[0]
        if pep_id not in train_id_to_pep_dict_orig.keys():
            train_id_to_pep_dict_orig[pep_id] = row["pep_seq1"]
    train_id_to_pep_dict = {
        k: train_id_to_pep_dict_orig[k]
        for k in range(len(train_id_to_pep_dict_orig))
    }

    # load train feats
    with open(
        os.path.join(
            data_root,
            "pepPMI_feat",
            "pepPMI_esm_finetune_len_{0}_mean.pickle".format(esm_feat_epoch),
        ),
        "rb",
    ) as fin:
        train_feat_dict = pickle.load(fin)

    # laod test feats
    with open(
        os.path.join(
            data_root,
            "pepPMI_test_ALA_feat",
            "pepPMI_ala{0}_esm_finetune_len_{1}_mean.pickle".format(
                split_method, esm_feat_epoch
            ),
        ),
        "rb",
    ) as fin:
        test_feat_dict = pickle.load(fin)

    pos_edges = train_edges_df[train_edges_df["label"] == 1]
    num_peps_train = len(train_id_to_pep_dict)
    train_feats = torch.from_numpy(
        np.vstack(
            [
                train_feat_dict[train_id_to_pep_dict[_id]]
                for _id in range(len(train_id_to_pep_dict))
            ]
        )
    )
    train_edge_index = torch.from_numpy(
        np.array([eval(_edge) for _edge in pos_edges["edges"]]).transpose()
    )
    test_id_to_pep_dict = {}
    for _, row in test_edges_df.iterrows():
        pep_id = eval(row["pep_idx_pair"])[0]
        if pep_id not in test_id_to_pep_dict.keys():
            test_id_to_pep_dict[pep_id] = row["pep_seq1"]

    test_feats = torch.from_numpy(
        np.vstack(
            [
                test_feat_dict[test_id_to_pep_dict[_id]]
                for _id in range(num_peps_test)
            ]
        )
    )
    test_edges_df = test_edges_df[test_edges_df["rank_y_cls2"] == 1]
    test_edges = torch.from_numpy(
        np.stack(
            [eval(_row["pep_idx_pair"]) for _, _row in test_edges_df.iterrows()]
        ).transpose()
    )

    feats_combined = torch.cat([train_feats, test_feats], dim=0)

    num_train_edges = train_edge_index.shape[1]
    train_data_orig = Data(
        x=feats_combined,
        edge_index=train_edge_index,
        edge_label=torch.zeros(num_train_edges, dtype=torch.long),
    )

    # test_edge_labels = torch.from_numpy(np.array(test_edges_df["rank_y_cls2"]))
    test_edge_labels = torch.zeros(len(test_edges_df), dtype=torch.long)
    # shift index of orig test edges
    test_edges = test_edges + num_peps_train
    neg_edges_labels, neg_edges_mp = neg_sampling_for_ala(
        test_edges, neg_sampling_ratio, neg_sampling_ratio
    )

    test_edge_index = torch.cat([train_edge_index, neg_edges_mp], axis=-1)
    test_labels = torch.cat([torch.zeros(len(test_edges_df))])

    # _edge_index, _edge_label = coalesce(
    #     test_edges,
    #     test_edge_labels,
    #     reduce="mean",
    # )
    # test_data = Data(
    #     x=torch.from_numpy(test_feats),
    #     edge_index=_edge_index,
    #     edge_label=_edge_label,
    # )

    test_data = Data(
        x=feats_combined,
        edge_index=train_edge_index,
        edge_label=test_edge_labels,
        edge_label_index=test_edges,
    )

    split_transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=0.0,
        is_undirected=False,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=False,
    )

    train_data, val_data, _ = split_transform(train_data_orig)
    return train_data, val_data, test_data


class MLPWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPWithAttention, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(1)  # Reshape for attention mechanism
        x, _ = self.attention(x, x, x)  # Apply attention
        x = F.dropout(x.squeeze(), p=0.5, training=self.training)

        x = self.fc2(x)
        x = x.unsqueeze(1)  # Reshape for attention mechanism
        x, _ = self.attention(x, x, x)  # Apply attention
        # x = x.squeeze()
        x = F.dropout(x.squeeze(), p=0.5, training=self.training)

        x = self.fc3(x)
        return x


class Net(torch.nn.Module):
    def __init__(
        self,
        input_feat_dim,
        hidden_dim,
        gnn_method,
        droput_ratio,
        dropout_method,
        add_skip_connection,
        add_self_loops,
        num_gnn_layers,
    ):
        super().__init__()
        self.gnn = GNN(
            input_feat_dim,
            hidden_dim,
            gnn_method,
            droput_ratio,
            dropout_method,
            add_skip_connection,
            num_gnn_layers,
            add_self_loops,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(droput_ratio),
            nn.Linear(hidden_dim, 1),
        )
        # self.mlp = MLPWithAttention(512 * 2, 512, 1)

    def graph_forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        return x

    def predict(self, x, edge_label_index):
        h_src = x[edge_label_index[0]]
        h_dst = x[edge_label_index[1]]
        feat = torch.cat([h_src, h_dst], dim=-1)
        return self.mlp(feat).squeeze()


def train_model(model, optimizer, criterion, train_data, neg_sampling_ratio):
    train_loader = LinkNeighborLoader(
        data=train_data,
        neg_sampling_ratio=neg_sampling_ratio,
        num_neighbors=[10],
        batch_size=12,
    )
    model.train()
    epoch_loss = []
    for _batch_data in train_loader:
        # optimizer.zero_grad()
        # z = model(_batch_data.x, _batch_data.edge_index)

        # h_src = z[_batch_data.edge_label_index[0]]
        # h_dst = z[_batch_data.edge_label_index[1]]
        # pred = (h_src * h_dst).sum(dim=-1)

        z = model.graph_forward(_batch_data)
        pred = model.predict(z, _batch_data.edge_label_index)
        # print(_batch_data.edge_label)
        # print(_batch_data.edge_label.shape)

        loss = criterion(pred, _batch_data.edge_label.cuda())
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss.data.cpu().numpy()))
    return np.mean(epoch_loss)


@torch.no_grad()
def val_model(model, data):
    model.eval()
    z = model.graph_forward(data)
    pred = model.predict(z, data.edge_label_index)
    # if stage == "val":
    #     pred = model.predict(z, data.edge_label_index)
    # else:
    #     pred = model.predict(z, data.edge_index)
    pred = torch.sigmoid(pred)

    auc_v = (
        roc_auc_score(data.edge_label.cpu().numpy(), pred.cpu().numpy()) * 100
    )
    precision, recall, _ = precision_recall_curve(
        data.edge_label.cpu().numpy(), pred.cpu().numpy()
    )

    aupr = auc(recall, precision) * 100
    return auc_v, aupr


@torch.no_grad()
def test_model(model, data):
    model.eval()
    z = model.graph_forward(data)
    pred = model.predict(z, data.edge_label_index)
    # if stage == "val":
    #     pred = model.predict(z, data.edge_label_index)
    # else:
    #     pred = model.predict(z, data.edge_index)
    pred = torch.sigmoid(pred)

    auc_v = (
        roc_auc_score(data.edge_label.cpu().numpy(), pred.cpu().numpy()) * 100
    )
    precision, recall, _ = precision_recall_curve(
        data.edge_label.cpu().numpy(), pred.cpu().numpy()
    )

    aupr = auc(recall, precision) * 100
    with open("test_res_ala_2.pickle", "wb") as fout:
        pickle.dump(
            {"gt": data.edge_label.cpu().numpy(), "pred": pred.cpu().numpy()},
            fout,
        )
    return auc_v, aupr


# @torch.no_grad()
# def test_model(model, data, neg_sampling_ratio):
#     model.eval()
#     z = model.graph_forward(data)
#     pred = model.predict(z, data.edge_label_index)
#     # if stage == "val":
#     #     pred = model.predict(z, data.edge_label_index)
#     # else:
#     #     pred = model.predict(z, data.edge_index)
#     pred = torch.sigmoid(pred)

#     test_loader = LinkNeighborLoader(
#         test_data,
#         num_neighbors=[10],
#         neg_sampling_ratio=neg_sampling_ratio,
#         edge_label=test_data.edge_label,
#         edge_label_index=test_data.edge_label_index,
#         batch_size=24,
#     )
#     y_true = []
#     y_pred = []
#     for test_batch in test_loader:
#         z = model.graph_forward(test_batch)
#         pred = model.predict(z, test_batch.edge_label_index)
#         pred = torch.sigmoid(pred)
#         y_true += test_batch.edge_label.tolist()
#         y_pred += pred.cpu().numpy().tolist()

#     # auc_v = (
#     #     roc_auc_score(data.edge_label.cpu().numpy(), pred.cpu().numpy()) * 100
#     # )
#     # precision, recall, _ = precision_recall_curve(
#     #     data.edge_label.cpu().numpy(), pred.cpu().numpy()
#     # )
#     auc_v = roc_auc_score(y_true, y_pred) * 100
#     precision, recall, _ = precision_recall_curve(y_true, y_pred)

#     aupr = auc(recall, precision) * 100
#     return auc_v, aupr


if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.ala_test:
        exp_name = "pmi_ala_0112"
    else:
        exp_name = "pmi_no_ala"

    mlflow.set_tracking_uri("http://192.168.1.237:45001")
    mlflow.set_experiment(exp_name)  # set the experiment
    mlflow.pytorch.autolog()

    device = torch.device("cuda")

    test_aucs = []
    test_auprs = []
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        for fold_id in range(1, 2):
            model = Net(
                args.input_feat_dim,
                args.hidden_dim,
                args.gnn_method,
                args.dropout_ratio,
                args.dropout_method,
                args.add_skip_connection,
                args.add_self_loops,
                args.num_gnn_layers,
            )

            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            model.cuda()

            if not args.ala_test:
                train_data, val_data, test_data = load_data_with_fold(
                    args.data_root_dir,
                    fold_id,
                    args.esm_feat_epoch,
                    args.use_random_feat,
                    args.disjoint_train_ratio,
                    args.val_ratio,
                    args.neg_sampling_ratio,
                )
            else:
                # train_data, val_data, test_data = load_data_feats_ala(
                #     args.data_root_dir,
                #     args.esm_feat_epoch,
                #     args.split_method,
                #     args.disjoint_train_ratio,
                #     args.val_ratio,
                #     args.neg_sampling_ratio,
                # )
                train_data, val_data, test_data = load_data_0112(
                    data_root=args.data_root_dir,
                    esm_feat_epoch=args.esm_feat_epoch,
                    use_random_feat=args.use_random_feat,
                    ala_id=int(args.split_method),
                    disjoint_train_ratio=args.disjoint_train_ratio,
                    val_ratio=args.val_ratio,
                    neg_sampling_ratio=args.neg_sampling_ratio,
                )

            train_data = train_data.to(device)
            val_data = val_data.to(device)
            test_data = test_data.to(device)
            for _epoch in range(args.epoch):
                loss = train_model(
                    model,
                    optimizer,
                    criterion,
                    train_data,
                    args.neg_sampling_ratio,
                )
                val_auc, val_aupr = val_model(model, val_data)
                test_auc, test_aupr = test_model(model, test_data)
                mlflow.log_metrics(
                    {
                        "loss-fold-{0}".format(fold_id): loss,
                        "val_auc-fold-{0}".format(fold_id): val_auc,
                        "val_aupr-fold-{0}".format(fold_id): val_aupr,
                        "test_auc-fold-{0}".format(fold_id): test_auc,
                        "test_aupr-fold-{0}".format(fold_id): test_aupr,
                    }
                )
            test_aucs.append(test_auc)
            test_auprs.append(test_aupr)
        mlflow.log_metrics(
            {
                "test_auc_mean": np.mean(test_aucs),
                "test_auc_std": np.std(test_aucs),
                "test_aupr_mean": np.mean(test_auprs),
                "test_aupr_std": np.std(test_auprs),
            }
        )

    # if not args.ala_test:
    #     train_data, val_data, test_data = load_data(
    #         args.data_root_dir,
    #         args.split_method,
    #         args.esm_feat_epoch,
    #         args.use_random_feat,
    #         args.disjoint_train_ratio,
    #     )
    #     train_data = train_data.to(device)
    #     val_data = val_data.to(device)
    #     test_data = test_data.to(device)

    #     for _epoch in range(100):
    #         loss = train_model(model, optimizer, criterion, train_data)
    #         val_auc, val_aupr = test_model(model, val_data, stage="val")
    #         test_auc, test_aupr = test_model(model, test_data, stage="test")
    #         print(
    #             "epoch-{0}\tloss:{1:.4f}\tval_auc:{2:.2f}\tval_aupr:{3:.2f}\ttest_auc:{4:.2f}\ttest_aupr:{5:.2f}".format(
    #                 _epoch, loss, val_auc, val_aupr, test_auc, test_aupr
    #             )
    #         )
    # else:
    #     train_data, test_data = load_data_feats_ala(
    #         args.data_root_dir,
    #         args.esm_feat_epoch,
    #         args.split_method,
    #         args.disjoint_train_ratio,
    #     )
    #     train_data = train_data.to(device)
    #     test_data = test_data.to(device)

    #     for _epoch in range(100):
    #         loss = train_model(model, optimizer, criterion, train_data)
    #         test_auc, test_aupr = test_model(model, test_data, stage="test")
    #         print(
    #             "epoch-{0}\tloss:{1:.4f}\ttest_auc:{2:.2f}\ttest_aupr:{3:.2f}".format(
    #                 _epoch, loss, test_auc, test_aupr
    #             )
    #         )
