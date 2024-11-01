import shutil
from src.data import PLProteinPeptideInteraction, ProteinPeptideInteraction
from src.model import PPIHetero
from torchmetrics import AUROC, ROC
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, List, Tuple
from torch import Tensor
import torch_geometric.transforms as T

from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.data import Batch
import torch.nn.functional as F
import torch
from lightning.pytorch import (
    LightningModule,
    Trainer,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.data.lightning import LightningLinkData


class PLPPIHetero(LightningModule):
    def __init__(
        self,
        hidden_channels,
        num_pep_nodes,
        num_prot_nodes,
        metadata,
    ):
        super().__init__()
        self.model = PPIHetero(
            hidden_channels, num_pep_nodes, num_prot_nodes, metadata
        )
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")
        self.val_aupr = ROC(task="binary")
        self.test_aupr = ROC(task="binary")

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
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.log(
            "{}_loss".format(stage),
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


if __name__ == "__main__":
    # data_module = PLProteinPeptideInteraction(
    #     "./data/yipin_protein_peptide/",
    #     remove_cache=True,
    #     batch_size=128,
    #     neg_sampling_ratio=10,
    # )

    shutil.rmtree("./data/yipin_protein_peptide/processed/")
    ppi = ProteinPeptideInteraction("./data/yipin_protein_peptide/")
    data = ppi[0]
    data = T.ToUndirected()(data)

    data_module = LightningLinkData(
        data=data,
        input_train_edges=(
            ("pep", "bind", "prot"),
            data["pep", "bind", "prot"]["edge_index"][:, data.train_mask],
        ),
        input_train_labels=data["pep", "bind", "prot"]["edge_label"][
            data.train_mask
        ],
        input_val_edges=(
            ("pep", "bind", "prot"),
            data["pep", "bind", "prot"]["edge_index"][:, data.test_mask],
        ),
        input_val_labels=data["pep", "bind", "prot"]["edge_label"][
            data.test_mask
        ],
        input_test_edges=(
            ("pep", "bind", "prot"),
            data["pep", "bind", "prot"]["edge_index"][:, data.test_mask],
        ),
        input_test_labels=data["pep", "bind", "prot"]["edge_label"][
            data.test_mask
        ],
        num_neighbors={
            ("pep", "bind", "prot"): [10, 10],
            ("prot", "rev_bind", "pep"): [10, 10],
        },
        neg_sampling_ratio=2,
        batch_size=128,
        num_workers=2,
    )

    # data_module.prepare_data()
    # data_module.setup()
    num_prot = data["prot"]["node_id"].shape[0]
    num_pep = data["pep"]["node_id"].shape[0]
    checkpoint = ModelCheckpoint(
        monitor="val_auc",
        save_top_k=4,
        mode="max",
        filename="{epoch:03d}-{val_auc:.2f}",
    )
    model = PLPPIHetero(256, num_pep, num_prot, data.metadata())
    trainer = Trainer(
        devices=[2],
        max_epochs=5,
        log_every_n_steps=1,
        callbacks=[checkpoint],
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # trainer.fit(model, data_module.train_dataloader())
    # trainer.test(model, data_module.test_dataloader())
