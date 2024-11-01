from src.data import ProteinPeptideInteraction
from src.model import Model
import argparse
import torch_geometric.transforms as T
import torch

import torch.nn.functional as F


def weighted_mse_loss(pred, target, weight=None):
    weight = 1.0 if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data["pep", "prot"].edge_label_index,
    )
    target = train_data["pep", "prot"].edge_label
    loss = weighted_mse_loss(pred, target, weight)
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
    )
    pred = pred.clamp(min=0, max=5)
    target = data["pep", "prot"].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


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

    dataset = ProteinPeptideInteraction("./data/yipin_protein_peptide/")
    data = dataset[0].to(device)

    # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
    data = T.ToUndirected()(data)

    # Perform a link-level split into training, validation, and test edges:
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[("pep", "bind", "prot")],
        rev_edge_types=[("prot", "rev_bind", "pep")],
    )(data)

    # We have an unbalanced dataset with many labels for rating 3 and 4, and very
    # few for 0 and 1. Therefore we use a weighted MSE loss.
    if args.use_weighted_loss:
        weight = torch.bincount(train_data["pep", "prot"].edge_label)
        weight = weight.max() / weight
    else:
        weight = None
    model = Model(hidden_channels=32, metadata=train_data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 301):
        loss = train(model, optimizer, train_data)
        train_rmse = test(model, train_data)
        val_rmse = test(model, val_data)
        test_rmse = test(model, test_data)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
            f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
        )
