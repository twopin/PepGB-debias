import argparse
import os
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch_geometric.data import Data
from sklearn import metrics
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import coalesce, negative_sampling
from torch.optim.lr_scheduler import CosineAnnealingLR
from train_pmi_split import Net
from src.model_diPepGB import GNN
from src.data import PLPMI


def cal_metric(labels,preds):
    fpr1, tpr1, thresholds = metrics.roc_curve(labels, preds)
    roc_auc1 = metrics.auc(fpr1, tpr1)  # the value of roc_auc1

    precision1, recall1, _ = metrics.precision_recall_curve(labels, preds)
    aupr1 = metrics.auc(recall1, precision1)  # the value of roc_auc1

    return roc_auc1,aupr1,precision1,recall1


input_feat_dim = 1280
hidden_dim = 512
num_gnn_layers = 1
dropout_ratio = 0.5
# val_ratio = 0.2

lr = 1e-4
epoch_num = 30
gnn_method = "gat_conv"
add_skip_connection = True
add_self_loops = True
dropout_method = "DropMessage"
esm_feat_epoch = 21500
device = torch.device("cuda")
data_root = "../pep_data/PMI_0111/PMI_random_vocab_0111_f4.csv"
train_root = "../pep_data/PMI_0111/PMI_rank_train_0111_all_f4.csv"
test_root = "../pep_data/PMI_0111/PMI_rank_test_0111_all_f4.csv"


def load_data(data_root, train_root, val_root, esm_feat_epoch):
    """loda edges, split and esm feat
    Args:
        data_root (str): path to data root
        train_root (str): path to training data root
        val_root (str): path to validation data root
        esm_feat_epoch (int): epoch of finetuning chekcpoint
    """

    # read edges df
    df_vocab = pd.read_csv(data_root)
    pep_list = edges_df["pep_seq1"].unique().tolist()

    with open(
        os.path.join(
            "pepPMI_feat",
            "pepPMI_esm_finetune_{0}_mean.pickle".format(esm_feat_epoch),
        ),
        "rb",
    ) as fin:
        feat_dict = pickle.load(fin)

    feats = torch.from_numpy(
            np.vstack([feat_dict[_pep] for _pep in pep_list])
        ).float()

    train_edges_df_all = pd.read_csv(train_root)
    train_edge_label_all = torch.tensor(train_edges_df_all.label, dtype=torch.long)

    val_edges_df_all = pd.read_csv(val_root)
    val_edge_label_all = torch.tensor(val_edges_df_all.label, dtype=torch.long)

    whole_edges_df_all = pd.concat([train_edges_df_all,val_edges_df_all],ignore_index=True)

    train_edges_all = torch.from_numpy(
        np.array(train_edges_df_all.loc[:, ["pep_idx1", "pep_idx2"]]).transpose()
    )

    val_edges_all = torch.from_numpy(
            np.array(val_edges_df_all.loc[:, ["pep_idx1", "pep_idx2"]]).transpose()
        )

    whole_edges_df_all = torch.from_numpy(
            np.array(whole_edges_df_all.loc[:, ["pep_idx1", "pep_idx2"]]).transpose()
        )


    train_data = Data(
            x=feats,
            edge_index=whole_edges_df_all,
            edge_label=train_edge_label_all, 
            edge_label_index=train_edges_all)
    val_data = Data(
            x=feats,
            edge_index=whole_edges_df_all,
            edge_label=val_edge_label_all, 
            edge_label_index=val_edges_all)

    train_data = train_data.to('cuda:0', non_blocking=True)
    val_data = val_data.to('cuda:0', non_blocking=True)


    return train_data, val_data


train_data, val_data = load_data(data_root, train_root, val_root, esm_feat_epoch)

model = Net(input_feat_dim,hidden_dim,gnn_method,
                dropout_ratio,dropout_method,
                add_skip_connection,add_self_loops,
                num_gnn_layers,)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()
model.cuda()

def train_model(model, optimizer, criterion, train_data):
    model.train()
    optimizer.zero_grad()
    # z = model(train_data.x, train_data.edge_index)

    # h_src = z[train_data.edge_label_index[0]]
    # h_dst = z[train_data.edge_label_index[1]]
    # pred = torch.cat([h_src, h_dst], dim=-1)

    z = model.graph_forward(train_data)
    pred = model.predict(z, train_data.edge_label_index)
    # print(_batch_data.edge_label)
    # print(_batch_data.edge_label.shape)

    loss = criterion(pred, train_data.edge_label.cuda().float())
    loss.backward()
    optimizer.step()
    return loss

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
    return auc_v, aupr, pred

best_val_auc = 0
save_log='./pmi_test_results/pmi_diPepGB_'
for epoch in range(1, 301):
    loss = train_model(model, optimizer, criterion, train_data)
    val_auc, val_aupr, pred = val_model(model, test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {val_auc:.4f},' )
    # test_edge_ar = test_data.edge_index.cpu().numpy()
    # test_pred_ar = pred.cpu().numpy()
    # with open(save_log+f"e{epoch}_{val_auc:.4f}.pickle", "wb") as fout:
    #         pickle.dump(
    #             {"test_edge_ar": test_edge_ar, "test_pred_ar": test_pred_ar},
    #             fout,
    #         )
