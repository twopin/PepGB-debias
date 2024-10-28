import argparse
import os
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
from train_pmi_split import Net
from src.model import GNN
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


model = Net(input_feat_dim,hidden_dim,gnn_method,
                dropout_ratio,dropout_method,
                add_skip_connection,add_self_loops,
                num_gnn_layers,)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()
model.cuda()


checkpoint = torch.load('./model_diPepGB.pt', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



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
    print(pred.shape,np.mean(pred.cpu().numpy()),np.quantile(pred.cpu().numpy(),0.75))

    auc_v = (
        roc_auc_score(data.edge_label.cpu().numpy(), pred.cpu().numpy()) * 100
    )
    precision, recall, _ = precision_recall_curve(
        data.edge_label.cpu().numpy(), pred.cpu().numpy()
    )

    aupr = auc(recall, precision) * 100
    return auc_v, aupr, pred


