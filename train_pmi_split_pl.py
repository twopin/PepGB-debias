import argparse
import os

# import mlflow
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
from lightning.pytorch import LightningModule, Trainer
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR


from src.model import GNN
from src.data import PLPMI
from torch_geometric.data import Batch
from libauc.optimizers import PESG
from libauc.losses import AUCMLoss
from torchmetrics import AUROC, ROC
import mlflow


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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--epoch", type=int, default=20, help="epoch for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="value of learning rate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.7,
        help="value for balance losses",
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

    def forward(self, data):
        x = self.graph_forward(data)
        return self.predict(x)

    def graph_forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        return x

    def predict(self, x, edge_label_index):
        h_src = x[edge_label_index[0]]
        h_dst = x[edge_label_index[1]]
        feat = torch.cat([h_src, h_dst], dim=-1)
        return self.mlp(feat).squeeze()


class PLPMIModel(LightningModule):
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
        beta,
        optimizer,
        lr,
        device,
        num_training_steps,
    ):
        super().__init__()
        self.model = Net(
            input_feat_dim,
            hidden_dim,
            gnn_method,
            droput_ratio,
            dropout_method,
            add_skip_connection,
            add_self_loops,
            num_gnn_layers,
        )
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")
        self.val_aupr = ROC(task="binary")
        self.test_aupr = ROC(task="binary")
        self.auc_loss = AUCMLoss(device=device)
        self.beta = beta
        self.optimizer_type = optimizer
        self.lr = lr
        self.add_skip_connection = add_skip_connection
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.num_training_steps = num_training_steps

        self.metric_cache = {
            "val": {"gt": [], "pred": []},
            "test": {"gt": [], "pred": []},
        }

    def forward(self, batch: Batch):
        return self.model(batch)

    def on_validation_epoch_start(
        self,
    ):
        # clear cache
        self.metric_cache["val"] = {"gt": [], "pred": []}

    def on_test_epoch_start(
        self,
    ):
        # clear cache
        self.metric_cache["test"] = {"gt": [], "pred": []}

    def common_step(self, batch: Batch, stage: str):
        batch_size = batch.edge_label.shape[0]
        z = self.model.graph_forward(batch)
        y_pred = self.model.predict(z, batch.edge_label_index)
        y = batch.edge_label
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        # loss = self.beta * F.binary_cross_entropy_with_logits(
        #     y_pred, y.float()
        # ) + (1 - self.beta) * self.auc_loss(torch.sigmoid(y_pred), y.float())
        # loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        loss = self.criterion(y_pred, y.float())

        self.log(
            "{}_loss".format(stage),
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "lr",
            lr,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        if stage in ["val", "test"]:
            self.metric_cache[stage]["gt"].append(y)
            self.metric_cache[stage]["pred"].append(torch.sigmoid(y_pred))
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self.common_step(batch, "train")

    def validation_step(self, batch: Batch, batch_idx: int):
        return self.common_step(batch, "val")

    def test_step(self, batch: Batch, batch_idx: int):
        return self.common_step(batch, "test")

    def on_validation_epoch_end(self):
        pred = torch.cat(self.metric_cache["val"]["pred"], dim=0)
        gt = torch.cat(self.metric_cache["val"]["gt"], dim=0)

        auc_v = self.val_auc(pred, gt) * 100
        precision, recall, _ = precision_recall_curve(
            gt.cpu().numpy(), pred.cpu().numpy()
        )
        aupr = auc(recall, precision) * 100

        self.log(
            "val_auc",
            auc_v,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_aupr",
            aupr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_end(self):
        pred = torch.cat(self.metric_cache["test"]["pred"], dim=0)
        gt = torch.cat(self.metric_cache["test"]["gt"], dim=0)
        auc_v = self.test_auc(pred, gt) * 100
        precision, recall, _ = precision_recall_curve(
            gt.cpu().numpy(), pred.cpu().numpy()
        )
        aupr = auc(recall, precision) * 100

        self.log(
            "test_auc",
            auc_v,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_aupr",
            aupr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return auc, aupr

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = CosineAnnealingLR(
                optimizer, self.num_training_steps, 1e-6
            )
        else:
            optimizer = PESG(self.model.parameters(), lr=self.lr)
            scheduler = CosineAnnealingLR(
                optimizer, self.num_training_steps, 1e-6
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train_model(model, optimizer, criterion, train_data):
    train_loader = LinkNeighborLoader(
        data=train_data,
        neg_sampling_ratio=10,
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
def test_model(model, data, stage):
    model.eval()
    z = model.graph_forward(data)
    if stage == "val":
        pred = model.predict(z, data.edge_label_index)
    else:
        pred = model.predict(z, data.edge_index)
    pred = torch.sigmoid(pred)
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

    mlflow.set_tracking_uri("http://192.168.1.237:45001")
    mlflow.set_experiment(
        "pmi_debug".format(args.split_method)
    )  # set the experiment
    mlflow.pytorch.autolog()

    device = torch.device("cuda:{}".format(args.device))
    # model.cuda()
    data_module = PLPMI(
        args.data_root_dir,
        args.esm_feat_epoch,
        args.split_method,
        1280,
        args.disjoint_train_ratio,
        args.neg_sampling_ratio,
        args.num_neighbors,
        args.use_random_feat,
        args.batch_size,
        args.ala_test,
    )
    num_training_steps = len(data_module.train_dataloader()) * args.epoch
    model = PLPMIModel(
        args.input_feat_dim,
        args.hidden_dim,
        args.gnn_method,
        args.dropout_ratio,
        args.dropout_method,
        args.add_skip_connection,
        args.add_self_loops,
        args.num_gnn_layers,
        args.beta,
        "adam",
        args.lr,
        device,
        num_training_steps,
    )
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        trainer = Trainer(
            devices=[args.device], max_epochs=args.epoch, log_every_n_steps=1
        )
        trainer.fit(model, data_module)
        # trainer.test(model, data_module)
        test_auc, test_aupr = test_model(
            model.model, data_module.test_data, "test"
        )
        mlflow.log_metrics({"test_auc": test_auc, "test_aupr": test_aupr})

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
