import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HeteroConv,
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    to_hetero,
)
from .layer import GNNLayer
from torch_geometric.typing import Adj


# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels, gnn_method, dropping_method):
#         super().__init__()
#         # self.dropout_ratio = dropout_ratio
#         if gnn_method == "sage_conv":
#             self.conv1 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
#             self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
#         elif gnn_method == "gat_conv":
#             self.conv1 = GATConv(
#                 hidden_channels, hidden_channels, add_self_loops=False
#             )
#             self.conv2 = GATConv(
#                 hidden_channels, hidden_channels, add_self_loops=False
#             )

#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        backbone,
        dropping_method,
        heads,
        alpha,
        K,
    ):
        super(GNN, self).__init__()
        self.backbone = backbone
        self.gnn1 = GNNLayer(
            in_channels,
            out_channels,
            dropping_method,
            backbone,
            heads=heads,
            alpha=alpha,
            K=K,
        )
        self.gnn2 = GNNLayer(
            int(out_channels * heads),
            out_channels,
            dropping_method,
            backbone,
            alpha=alpha,
            K=K,
        )

    def forward(
        self, x: Tensor, edge_index: Adj, drop_rate: float = 0
    ) -> Tensor:
        x = self.gnn1(x, edge_index, drop_rate)
        if self.backbone == "GAT":
            x = F.elu(x)
        else:
            x = F.relu(x)
        x = self.gnn2(x, edge_index, drop_rate)
        print("test")
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


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


class PPIHetero(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_pep_nodes,
        num_prot_nodes,
        metadata,
        gnn_method,
        dropout_ratio,
        feat_src,
    ):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for peps and prots:
        self.prot_lin = torch.nn.Linear(1280, hidden_channels)
        self.pep_lin = torch.nn.Linear(1280, hidden_channels)
        self.pep_emb = torch.nn.Embedding(num_pep_nodes, hidden_channels)
        self.prot_emb = torch.nn.Embedding(num_prot_nodes, hidden_channels)
        self.feat_src = feat_src
        # Instantiate homogeneous GNN:
        if self.feat_src == "esm_emb":
            self.gnn = GNN(
                int(hidden_channels * 2),
                int(hidden_channels),
                gnn_method,
                "DropMessage",
                1,
                0.1,
                10,
            )
        else:
            self.gnn = GNN(
                int(hidden_channels),
                int(hidden_channels),
                gnn_method,
                "DropMessage",
                1,
                0.1,
                10,
            )
        self.dropout_ratio = {"pep": dropout_ratio, "prot": dropout_ratio}

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=metadata, debug=True)
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        if self.feat_src == "esm_emb":
            x_dict = {
                "pep": torch.cat(
                    [
                        self.pep_lin(data["pep"].x),
                        self.pep_emb(data["pep"].node_id),
                    ],
                    dim=1,
                ),
                "prot": torch.cat(
                    [
                        self.prot_lin(data["prot"].x),
                        self.prot_emb(data["prot"].node_id),
                    ],
                    dim=1,
                ),
            }
        elif self.feat_src == "emb":
            # only embedding
            x_dict = {
                "pep": self.pep_emb(data["pep"].node_id),
                "prot": self.prot_emb(data["prot"].node_id),
            }
        elif self.feat_src == "esm":
            # orig feature
            x_dict = {
                "pep": self.pep_lin(data["pep"].x),
                "prot": self.prot_lin(data["prot"].x),
            }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict, self.dropout_ratio)
        for k, v in x_dict.items():
            print(v.shape)
        print(data["pep", "bind", "prot"].edge_label_index.shape)
        pred = self.classifier(
            x_dict["pep"],
            x_dict["prot"],
            data["pep", "bind", "prot"].edge_label_index,
        )
        return pred
