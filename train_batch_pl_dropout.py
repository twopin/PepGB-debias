import shutil
import numpy as np
import os
from src.data import PLProteinPeptideInteraction

# from src.model import PPIHetero
from src.dropout_model.model import PPIHetero
from torchmetrics import AUROC, ROC
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, List, Tuple
from torch import Tensor
import torch_geometric.transforms as T
from argparse import ArgumentParser

from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.data import Batch
import torch.nn.functional as F
import torch
from lightning.pytorch import (
    LightningModule,
    Trainer,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch_geometric.data.lightning import LightningLinkData
import mlflow
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG


class PLPPIHetero(LightningModule):
    def __init__(
        self,
        hidden_channels,
        num_pep_nodes,
        num_prot_nodes,
        metadata,
        gnn_method,
        dropout_ratio,
        fold_idx,
        feat_src,
        beta,
        device,
        optimizer,
        lr,
    ):
        super().__init__()
        self.model = PPIHetero(
            hidden_channels,
            num_pep_nodes,
            num_prot_nodes,
            metadata,
            gnn_method,
            dropout_ratio,
            feat_src,
        )
        self.fold_idx = fold_idx
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")
        self.val_aupr = ROC(task="binary")
        self.test_aupr = ROC(task="binary")
        self.auc_loss = AUCMLoss(device=device)
        self.beta = beta
        self.optimizer = optimizer
        self.lr = lr

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
        batch_size = batch.x_dict["prot"].shape[0]
        y_pred = self.model(batch)
        y = batch["pep", "bind", "prot"].edge_label
        # if stage == "train":
        #     print(torch.count_nonzero(y))
        loss = self.beta * F.binary_cross_entropy_with_logits(
            y_pred, y.float()
        ) + (1 - self.beta) * self.auc_loss(torch.sigmoid(y_pred), y.float())
        self.log(
            "{}_loss_fold_{}".format(stage, self.fold_idx),
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        if stage in ["val", "test"]:
            self.metric_cache[stage]["gt"].append(y)
            self.metric_cache[stage]["pred"].append(y_pred)
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
            "val_auc_fold_{}".format(self.fold_idx),
            auc_v,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_aupr_fold_{}".format(self.fold_idx),
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
            "test_auc_{}".format(self.fold_idx),
            auc_v,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_aupr_{}".format(self.fold_idx),
            aupr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return auc, aupr

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            return PESG(self.model.parameters(), lr=self.lr)


def main(args):
    model = None

    mlflow.set_tracking_uri("http://192.168.1.237:45001")
    mlflow.set_experiment(
        "ppi_graph_dropout_no_protein_inner_{}".format(args.split_method)
    )  # set the experiment
    mlflow.pytorch.autolog()

    device = torch.device("cuda:{}".format(args.device))
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "disjoint_train_ratio": args.disjoint_train_ratio,
                "dropout_ratio": args.dropout_ratio,
                "gnn_method": args.gnn_method,
                "neg_sampling_ratio": args.neg_sampling_ratio,
                "feat_src": args.feat_src,
                "pep_feat": args.pep_feat,
                "hidden_channels": args.hidden_channels,
                "epoch": args.epoch,
                "beta": args.beta,
            }
        )
        aucs = []
        auprs = []
        for i in range(5):
            # for i in range(1, 2):
            data_module = PLProteinPeptideInteraction(
                args.data_root_dir,
                remove_cache=True,
                batch_size=args.batch_size,
                disjoint_train_ratio=args.disjoint_train_ratio,
                neg_sampling_ratio=args.neg_sampling_ratio,
                pooling_method=args.pooling_method,
                split_method=args.split_method,
                fold_id=i + 1,
            )
            num_prot = data_module.data["prot"]["node_id"].shape[0]
            num_pep = data_module.data["pep"]["node_id"].shape[0]
            metadata = data_module.data.metadata()
            # checkpoint = ModelCheckpoint(
            #     monitor="val_auc",
            #     save_top_k=2,
            #     mode="max",
            #     filename="{epoch:03d}-{val_auc:.2f}",
            # )
            if model is not None:
                del model
            model = PLPPIHetero(
                args.hidden_channels,
                num_pep,
                num_prot,
                metadata,
                args.gnn_method,
                args.dropout_ratio,
                i,
                args.feat_src,
                args.beta,
                device,
                args.optimizer,
                args.lr,
            )
            trainer = Trainer(
                devices=[args.device],
                max_epochs=args.epoch,
                log_every_n_steps=1,
                # callbacks=[checkpoint],
                default_root_dir=args.log_root_dir,
            )
            # checkpoint.dir_path = os.path.join(
            #     checkpoint.dirpath, "fold_{}".format(i + 1)
            # )
            trainer.fit(model, data_module)
            res = trainer.test(model, data_module)[0]
            aucs.append(res["test_auc_{}".format(i)])
            auprs.append(res["test_aupr_{}".format(i)])
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        aupr_mean = np.mean(auprs)
        aupr_std = np.std(auprs)
        mlflow.log_metric("test_auc_mean", auc_mean)
        mlflow.log_metric("test_auc_std", auc_std)
        mlflow.log_metric("test_aupr_mean", aupr_mean)
        mlflow.log_metric("test_aupr_std", aupr_std)

    # trainer.fit(model, data_module.train_dataloader())
    # trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--split_method", type=str, default="a", choices=["a", "b", "c"]
    )
    parser.add_argument(
        "--pooling_method", type=str, default="mean", choices=["mean", "max"]
    )
    parser.add_argument(
        "--pep_feat", type=str, default="finetune", choices=["esm", "finetune"]
    )

    parser.add_argument(
        "--feat_src", type=str, default="esm", choices=["esm", "emb", "esm_emb"]
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "pesg"]
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disjoint_train_ratio", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--neg_sampling_ratio", type=float, default=5.0)
    parser.add_argument("--dropout_ratio", type=float, default=0.3)
    parser.add_argument(
        "--gnn_method",
        type=str,
        default="GAT",
        choices=["GCN", "GAT", "APPNP"],
    )
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

    args = parser.parse_args()
    main(args)
