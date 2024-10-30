import os
import pickle
import shutil
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import add_self_loops
from torchmetrics import AUROC, ROC

from src.data import PLProteinPeptideInteraction
from src.model import PPIHetero


class PLPPIHetero(LightningModule):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        metadata,
        gnn_method,
        dropout_ratio,
        fold_idx,
        feat_src,
        beta,
        device,
        optimizer,
        lr,
        dropping_method,
        repeat_id,
        add_skip_connection,
        num_gnn_layers,
    ):
        super().__init__()
        self.model = PPIHetero(
            in_channels,
            hidden_channels,
            metadata,
            gnn_method,
            dropout_ratio,
            feat_src,
            dropping_method,
            add_skip_connection,
            num_gnn_layers,
        )
        self.fold_idx = fold_idx
        self.repeat_id = repeat_id
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")
        self.val_aupr = ROC(task="binary")
        self.test_aupr = ROC(task="binary")
        self.auc_loss = AUCMLoss(device=device)
        self.beta = beta
        self.optimizer = optimizer
        self.lr = lr
        self.add_skip_connection = add_skip_connection

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
        # add self loop edge
        # new_edge_index,_ = add_self_loops(batch["pep","bind","prot"]["edge_index"])
        # batch["pep","bind","prot"]["edge_index"] = new_edge_index

        # new_edge_index,_ = add_self_loops(batch["prot","rev_bind","pep"]["edge_index"])
        # batch["prot","rev_bind","pep"]["edge_index"] = new_edge_index

        y_pred = self.model(batch)
        # with open(
        #     "./tmp/debug_data/{0}_data.pickle".format(stage), "wb"
        # ) as fout:
        #     pickle.dump(batch, fout)
        y = batch["pep", "bind", "prot"].edge_label
        # if stage == "train":
        #     print(torch.count_nonzero(y))
        loss = self.beta * F.binary_cross_entropy_with_logits(
            y_pred, y.float()
        ) + (1 - self.beta) * self.auc_loss(torch.sigmoid(y_pred), y.float())
        self.log(
            "{}_loss_fold_{}_repeat".format(
                stage, self.fold_idx, self.repeat_id
            ),
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
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
            "val_auc_fold_{}_repeat_{}".format(self.fold_idx, self.repeat_id),
            auc_v,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_aupr_fold_{}_repeat_{}".format(self.fold_idx, self.repeat_id),
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
        # with open("./tmp/debug_data/test_out.pickle", "wb") as fout:
        #     pickle.dump({"gt": gt, "pred": pred}, fout)

        self.log(
            "test_auc_fold_{}_repeat_{}".format(self.fold_idx, self.repeat_id),
            auc_v,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_aupr_fold_{}_repeat_{}".format(self.fold_idx, self.repeat_id),
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


@torch.no_grad()
def test_model(model, data):
    pred = torch.sigmoid(model(data))
    gt = data["pep", "bind", "prot"].edge_label.numpy()
    # if stage == "val":
    #     pred = model.predict(z, data.edge_label_index)
    # else:
    #     pred = model.predict(z, data.edge_index)
    pred = torch.sigmoid(pred).cpu().numpy()

    auc_v = roc_auc_score(gt, pred) * 100
    precision, recall, _ = precision_recall_curve(gt, pred)

    aupr = auc(recall, precision) * 100
    # with open("test_res_ala_2.pickle", "wb") as fout:
    #     pickle.dump(
    #         {"gt": data.edge_label.cpu().numpy(), "pred": pred.cpu().numpy()},
    #         fout,
    #     )
    return auc_v, aupr


def main(args):
    # print(vars(args))
    # exit(0)
    model = None

    mlflow.set_tracking_uri("http://10.28.0.57:45000")
    mlflow.set_experiment(
        "ppi_idp_test".format(args.split_method)
    )  # set the experiment
    mlflow.pytorch.autolog()

    logdir_logged = False
    device = torch.device("cuda:{}".format(args.device))
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        # mlflow.log_params(
        #     {
        #         "disjoint_train_ratio": args.disjoint_train_ratio,
        #         "dropout_ratio": args.dropout_ratio,
        #         "gnn_method": args.gnn_method,
        #         "neg_sampling_ratio": args.neg_sampling_ratio,
        #         "feat_src": args.feat_src,
        #         "pep_feat": args.pep_feat,
        #         "hidden_channels": args.hidden_channels,
        #         "in_channels": args.in_channels,
        #         "repeat_times": args.repeat_times,
        #         "epoch": args.epoch,
        #         "beta": args.beta,
        #         "dropping_methods": args.dropping_method,
        #         "use_ppi": args.use_ppi,
        #     }
        # )
        repeat_aucs = []
        repeat_auc_stds = []
        repeat_auprs = []
        repeat_aupr_stds = []
        # for _repeat_id in range(args.repeat_times):
        for _repeat_id in range(1):
            fold_aucs = []
            fold_auprs = []
            # for i in range(5):
            for i in range(1, 2):
                data_module = PLProteinPeptideInteraction(
                    root=args.data_root_dir,
                    remove_cache=True,
                    batch_size=args.batch_size,
                    use_ppi=args.use_ppi,
                    disjoint_train_ratio=args.disjoint_train_ratio,
                    neg_sampling_ratio=args.neg_sampling_ratio,
                    pooling_method=args.pooling_method,
                    split_method=args.split_method,
                    fold_id=i + 1,
                    pep_feat=args.pep_feat,
                    random_feat=args.random_feat,
                    idp_test=args.idp_test,
                )
                # num_prot = data_module.data["prot"]["node_id"].shape[0]
                # num_pep = data_module.data["pep"]["node_id"].shape[0]
                metadata = data_module.train_data.metadata()
                # checkpoint = ModelCheckpoint(
                #     monitor="val_auc",
                #     save_top_k=2,
                #     mode="max",
                #     filename="{epoch:03d}-{val_auc:.2f}",
                # )
                if model is not None:
                    del model
                model = PLPPIHetero(
                    args.in_channels,
                    args.hidden_channels,
                    metadata,
                    args.gnn_method,
                    args.dropout_ratio,
                    i,
                    args.feat_src,
                    args.beta,
                    device,
                    args.optimizer,
                    args.lr,
                    args.dropping_method,
                    _repeat_id,
                    args.add_skip_connection,
                    args.gnn_layers,
                )
                trainer = Trainer(
                    devices=[args.device],
                    max_epochs=args.epoch,
                    log_every_n_steps=1,
                    # callbacks=[checkpoint],
                    default_root_dir=args.log_root_dir,
                )
                if not logdir_logged:
                    mlflow.log_param(
                        "run_dir_fold_{}_repeat_{}".format(i, _repeat_id),
                        trainer.log_dir,
                    )
                    logdir_logged = True
                # checkpoint.dir_path = os.path.join(
                #     checkpoint.dirpath, "fold_{}".format(i + 1)
                # )
                trainer.fit(model, data_module)
                auc, aupr = test_model(model.model, data_module.test_data)
                # res = trainer.test(model, data_module)[0]

                # fold_aucs.append(
                #     res["test_auc_fold_{}_repeat_{}".format(i, _repeat_id)]
                # )
                # fold_auprs.append(
                #     res["test_aupr_fold_{}_repeat_{}".format(i, _repeat_id)]
                # )
            # auc_mean = np.mean(fold_aucs)
            # auc_std = np.std(fold_aucs)
            # aupr_mean = np.mean(fold_auprs)
            # aupr_std = np.std(fold_auprs)

            # repeat_aucs.append(auc_mean)
            # repeat_auc_stds.append(auc_std)
            # repeat_auprs.append(aupr_mean)
            # repeat_aupr_stds.append(aupr_std)

            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_aupr", aupr)
            # mlflow.log_metric(
        #         "test_aupr_mean_repeat_{}".format(_repeat_id), aupr_mean
        #     )
        #     mlflow.log_metric(
        #         "test_aupr_std_repeat_{}".format(_repeat_id), aupr_std
        #     )
        # auc_mean = np.mean(fold_aucs)
        # auc_std = np.std(fold_aucs)
        # aupr_mean = np.mean(fold_auprs)
        # aupr_std = np.std(fold_auprs)
        # mlflow.log_metric("test_auc_mean", auc_mean)
        # mlflow.log_metric("test_auc_std", auc_std)
        # mlflow.log_metric("test_aupr_mean", aupr_mean)
        # mlflow.log_metric("test_aupr_std", aupr_std)

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
    parser.add_argument("--use_ppi", action="store_true")
    parser.add_argument(
        "--dropping_method",
        type=str,
        default="Dropout",
        choices=["Dropout", "DropNode", "DropEdge", "DropMessage"],
    )
    parser.add_argument(
        "--pep_feat", type=str, default="finetune", choices=["esm", "finetune"]
    )

    parser.add_argument(
        "--feat_src", type=str, default="esm", choices=["esm", "emb", "esm_emb"]
    )
    parser.add_argument("--random_feat", action="store_true")
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "pesg"]
    )
    parser.add_argument("--add_skip_connection", action="store_true")
    parser.add_argument("--idp_test", action="store_true")
    parser.add_argument("--in_channels", type=int, default=256)
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=5,
        help="times to repeat run crossfold",
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
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument(
        "--gnn_method",
        type=str,
        default="gat_conv",
        choices=["sage_conv", "gat_conv", "gin_conv"],
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