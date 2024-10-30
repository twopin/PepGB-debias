import argparse
import os
import shutil

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score

from src.data import ProteinPeptideInteraction
from src.model import Model


def train(model, optimizer, train_data, criterion):
    model.train()
    optimizer.zero_grad()
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data["pep", "prot"].edge_label_index,
    )
    target = train_data["pep", "prot"].edge_label

    target = target.float()
    # loss = criterion(pred, target.float())

    # pred = torch.sigmoid(pred)
    # loss = criterion(pred.view(-1), target)
    # target = target.view(target.size(0), -1).float()
    # print(pred.shape)
    # print(target.shape)

    loss = criterion(pred.view(-1), target)

    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(
        data.x_dict,
        data.edge_index_dict,
        data["pep", "prot"].edge_label_index,
    ).sigmoid()
    target = data["pep", "prot"].edge_label
    # print(target.cpu().numpy())
    # print(pred.cpu().numpy())
    # print("-------------")
    auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_weighted_loss",
        action="store_true",
        help="Whether to use weighted MSE loss.",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if os.path.exists("./data/yipin_protein_peptide/processed"):
        shutil.rmtree("./data/yipin_protein_peptide/processed")
    dataset = ProteinPeptideInteraction(
        "./data/yipin_protein_peptide/",
    )
    data = dataset[0].to(device)
    # num_src = data.x_dict["pep"].shape[0]
    # num_dst = data.x_dict["prot"].shape[0]

    # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
    data = T.ToUndirected()(data)
    del data["prot", "rev_bind", "pep"].edge_label

    # Perform a link-level split into training, validation, and test edges:
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=[("pep", "bind", "prot")],
        rev_edge_types=[("prot", "rev_bind", "pep")],
    )(data)
    # trai_data = train_data.coalesce()
    # val_data = val_data.coalesce()
    # test_data = test_data.coalesce()

    # creat model
    model = Model(
        in_channels=1280,
        hidden_channels=256,
        metadata=train_data.metadata(),
        src="pep",
        dst="prot",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.BCELoss()

    for epoch in range(1, 10000):
        loss = train(model, optimizer, train_data, criterion)
        train_auc = test(model, train_data)
        val_auc = test(model, val_data)
        test_auc = test(model, test_data)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_auc:.4f}, "
            f"Val: {val_auc:.4f}, Test: {test_auc:.4f}"
        )
