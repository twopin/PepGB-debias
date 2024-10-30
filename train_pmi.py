import argparse
import os
# import mlflow
import pickle

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling

from src.model import GNN


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
        default="./tmp/training_logs",
        help="path to save logs",
    )

    parser.add_argument(
        "--split_method",
        type=str,
        default="random",
        choices=["random", "similarity"],
        help="methods to split data",
    )
    parser.add_argument("--use_random_feat", action="store_true")
    parser.add_argument(
        "--esm_feat_epoch",
        type=int,
        default=21500,
        help="epoch for saving checkpoint when finetuning esm",
    )
    return parser.parse_args()


def load_data(data_root, split_method, esm_feat_epoch, use_random_feat):
    """loda edges, split and esm feat

    Args:
        data_root (str): path to data root
        split_method (str): split method, random split or by similarity
        esm_feat_epoch (int): epoch of finetuning chekcpoint
        use_random_feat (bool): whether to use random feat

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
    ordered_pep_dict = {_pep_str: i for i, _pep_str in enumerate(pep_list)}
    ordered_pep_list = list(ordered_pep_dict.keys())

    # new id for edges
    edges_df["pep_seq1_idx_ordered"] = edges_df["pep_seq1"].apply(
        lambda x: ordered_pep_dict[x]
    )
    edges_df["pep_seq2_idx_ordered"] = edges_df["pep_seq2"].apply(
        lambda x: ordered_pep_dict[x]
    )

    if not use_random_feat:
        with open(
            os.path.join(
                data_root,
                "pepPMI_feat",
                "pepPMI_esm_finetune_{0}_mean.pickle".format(esm_feat_epoch),
            ),
            "rb",
        ) as fin:
            feat_dict = pickle.load(fin)
    else:
        feat_dict = {_pep: np.random.random(1280) for _pep in pep_list}

    # stack feat array
    feat_x = np.vstack([feat_dict[_pep] for _pep in ordered_pep_list])
    feat_x = torch.from_numpy(feat_x).float()
    edges = torch.tensor(
        np.array(
            edges_df.loc[:, ["pep_seq1_idx_ordered", "pep_seq2_idx_ordered"]]
        ).transpose(),
    )

    edge_pos_train_mask = np.array(
        edges_df[
            (edges_df["label"] == 1) & (edges_df["flag"] == "train")
        ].index.tolist()
    )
    # edge_neg_train_mask = np.array(
    #     edges_df[
    #         (edges_df["label"] == 0) & (edges_df["flag"] == "train")
    #     ].index.tolist()
    # )
    edge_test_mask = np.array(
        edges_df[edges_df["flag"] == "test"].index.tolist()
    )
    test_edge_label = edges_df[edges_df["flag"] == "test"]["label"].tolist()
    train_data_orig = Data(
        x=feat_x,
        edge_index=edges[:, edge_pos_train_mask],
        edge_label=torch.zeros(len(edge_pos_train_mask), dtype=torch.long),
    )
    test_data = Data(
        x=feat_x,
        edge_index=edges[:, edge_test_mask],
        edge_label=torch.tensor(test_edge_label, dtype=torch.long),
    )
    split_transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.0,
        is_undirected=False,
        add_negative_train_samples=False,
    )

    split_done = False
    while not split_done:
        train_data, val_data, _ = split_transform(train_data_orig)
        overlap_size = find_intersection(
            val_data.edge_label_index.numpy(), test_data.edge_index.numpy()
        ).shape[0]
        # if overlap_size == 0:
        if overlap_size < 3:
            split_done = True

    return train_data, val_data, test_data


def train_model(model, optimizer, criterion, train_data):
    model.train()
    optimizer.zero_grad()
    num_train_edge = train_data.edge_index.shape[1]
    z = model(train_data.x, train_data.edge_index)

    # add negtive sample
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=80,
        num_neg_samples=num_train_edge,
    )
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat(
        [
            torch.ones(num_train_edge),
            torch.zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )
    h_src = z[edge_label_index[0]]
    h_dst = z[edge_label_index[1]]
    pred = (h_src * h_dst).sum(dim=-1)

    loss = criterion(pred, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test_model(model, data, stage):
    model.eval()
    z = model(data.x, data.edge_index)
    if stage == "val":
        h_src = z[data.edge_label_index[0]]
        h_dst = z[data.edge_label_index[1]]
    else:
        h_src = z[data.edge_index[0]]
        h_dst = z[data.edge_index[1]]
    pred = (h_src * h_dst).sum(dim=-1)
    auc_v = (
        roc_auc_score(data.edge_label.cpu().numpy(), pred.cpu().numpy()) * 100
    )
    precision, recall, _ = precision_recall_curve(
        data.edge_label.cpu().numpy(), pred.cpu().numpy()
    )

    aupr = auc(recall, precision) * 100
    return auc_v, aupr


if __name__ == "__main__":
    args = get_args()
    train_data, val_data, test_data = load_data(
        args.data_root_dir,
        args.split_method,
        args.esm_feat_epoch,
        args.use_random_feat,
    )

    model = GNN(1280, 512, "gat_conv", 0.3, "DropMessage", True, 3)
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _epoch in range(100):
        loss = train_model(model, optimizer, criterion, train_data)
        val_auc, val_aupr = test_model(model, val_data, stage="val")
        test_auc, test_aupr = test_model(model, test_data, stage="test")
        print(
            "epoch-{0}\tloss:{1:.2f}\tval_auc:{2:.2f}\tval_aupr:{3:.2f}\ttest_auc:{4:.2f}\ttest_aupr:{5:.2f}".format(
                _epoch, loss, val_auc, val_aupr, test_auc, test_aupr
            )
        )
