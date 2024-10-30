import os
import shutil

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero

from src.data import ProteinPeptideInteraction

if os.path.exists("./data/yipin_protein_peptide/processed"):
    shutil.rmtree("./data/yipin_protein_peptide/processed")
dataset = ProteinPeptideInteraction(
    "./data/yipin_protein_peptide/",
)
data = dataset[0]
num_prot = data["prot"].num_nodes
num_pep = data["pep"].num_nodes

# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)


transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("pep", "bind", "prot"),
    rev_edge_types=("prot", "rev_bind", "pep"),
)
train_data, val_data, test_data = transform(data)

# Define seed edges:
edge_label_index = train_data["pep", "bind", "prot"].edge_label_index
edge_label = train_data["pep", "bind", "prot"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("pep", "bind", "prot"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(
        self, x_pep: Tensor, x_prot: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_pep = x_pep[edge_label_index[0]]
        edge_feat_prot = x_prot[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_pep * edge_feat_prot).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_pep_nodes, num_prot_nodes):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for peps and prots:
        self.prot_lin = torch.nn.Linear(1280, hidden_channels)
        self.pep_lin = torch.nn.Linear(1280, hidden_channels)
        self.pep_emb = torch.nn.Embedding(num_pep_nodes, hidden_channels)
        self.prot_emb = torch.nn.Embedding(num_prot_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        # x_dict = {
        #     "pep": self.pep_lin(data["pep"].x)
        #     + self.pep_emb(data["pep"].node_id),
        #     "prot": self.prot_lin(data["prot"].x)
        #     + self.prot_emb(data["prot"].node_id),
        # }

        # only embedding
        # x_dict = {
        #     "pep": self.pep_emb(data["pep"].node_id),
        #     "prot": self.prot_emb(data["prot"].node_id),
        # }

        # orig feature
        x_dict = {
            "pep": self.pep_lin(data["pep"].x),
            "prot": self.prot_lin(data["prot"].x),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["pep"],
            x_dict["prot"],
            data["pep", "bind", "prot"].edge_label_index,
        )
        return pred


model = Model(128, num_pep, num_prot)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 50):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["pep", "bind", "prot"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

edge_label_index = val_data["pep", "bind", "prot"].edge_label_index
edge_label = val_data["pep", "bind", "prot"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("pep", "bind", "prot"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)


preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["pep", "bind", "prot"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")


test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20, 10],
    edge_label_index=(("pep", "bind", "prot"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(test_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["pep", "bind", "prot"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Testing AUC: {auc:.4f}")
