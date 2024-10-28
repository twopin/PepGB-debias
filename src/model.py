import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    GINConv,
    SAGEConv,
    to_hetero,
)
from .dropout_gat import DropGATConv
from torch_sparse import SparseTensor
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq


class DropBlock:
    def __init__(self, dropping_method: str):
        super(DropBlock, self).__init__()
        self.dropping_method = dropping_method

    def drop(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.dropping_method == "DropNode":
            _, x_size = x.size(0)
            x = x * torch.bernoulli(torch.ones(x_size, 1) - drop_rate).to(
                x.device
            )
            x = x / (1 - drop_rate)
        elif self.dropping_method == "DropEdge":
            edge_reserved_size = int(edge_index.size(1) * (1 - drop_rate))
            if isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
                edge_index = torch.stack((row, col))
            perm = torch.randperm(edge_index.size(1))
            edge_index = edge_index.t()[perm][:edge_reserved_size].t()
        elif self.dropping_method == "Dropout":
            x = F.dropout(x, drop_rate)

        return x, edge_index


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        gnn_method,
        dropout_ratio,
        dropping_method,
        add_skip_connection,
        num_gnn_layers,
        add_self_loops=False,
    ):
        super().__init__()
        self.add_skip_connection = add_skip_connection
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.add_self_loops = add_self_loops
        if add_skip_connection:
            self.layer_1 = nn.Linear(in_channels, hidden_channels)
            self.in_channels = hidden_channels
        self.dropout_ratio = dropout_ratio
        self.dropping_method = dropping_method
        if dropping_method == "DropMessage":
            self.message_droprate = dropout_ratio
        else:
            self.message_droprate = 0
        self.drop_block = DropBlock(self.dropping_method)
        # self.convs = []
        self.convs = torch.nn.ModuleList()
        if gnn_method == "sage_conv":
            self.convs.append(
                SAGEConv(self.in_channels, self.hidden_channels, aggr="mean")
            )
            for _ in range(num_gnn_layers - 1):
                self.convs.append(
                    SAGEConv(
                        self.hidden_channels, self.hidden_channels, aggr="mean"
                    )
                )
        elif gnn_method == "gat_conv":
            # self.conv1 = GATConv(
            #     hidden_channels, hidden_channels, add_self_loops=False
            # )
            # self.conv2 = GATConv(
            #     hidden_channels, hidden_channels, add_self_loops=False
            # )

            self.convs.append(
                DropGATConv(
                    self.in_channels,
                    hidden_channels,
                    add_self_loops=self.add_self_loops,
                    message_droprate=self.message_droprate,
                )
            )
            for _ in range(num_gnn_layers - 1):
                self.convs.append(
                    DropGATConv(
                        hidden_channels,
                        hidden_channels,
                        add_self_loops=self.add_self_loops,
                        message_droprate=self.message_droprate,
                    )
                )
        elif gnn_method == "gin_conv":
            self.convs.append(
                GINConv(
                    Seq(
                        Lin(self.in_channels, self.hidden_channels),
                        ReLU(),
                        Lin(self.hidden_channels, self.hidden_channels),
                    )
                )
            )
            for _ in range(num_gnn_layers - 1):
                self.convs.append(
                    GINConv(
                        Seq(
                            Lin(self.hidden_channels, self.hidden_channels),
                            ReLU(),
                            Lin(self.hidden_channels, self.hidden_channels),
                        )
                    )
                )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.add_skip_connection:
            x = self.layer_1(x)
            identity = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # block 1
        if self.training:
            x, edge_index = self.drop_block.drop(
                x, edge_index, self.dropout_ratio
            )
        # if self.add_skip_connection:
        #     identity = x
        x = F.relu(self.convs[0](x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # block 2
        for _conv in self.convs[1:]:
            if self.training:
                x, edge_index = self.drop_block.drop(
                    x, edge_index, self.dropout_ratio
                )
            x = _conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        if self.add_skip_connection:
            x = F.relu(x + identity)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
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


class PPIHetero(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        metadata,
        gnn_method,
        dropout_ratio,
        feat_src,
        dropping_method,
        add_skip_connection,
        num_gnn_layers,
    ):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for peps and prots:
        # self.prot_lin = torch.nn.Linear(1280, hidden_channels)
        # self.pep_lin = torch.nn.Linear(1280, hidden_channels)
        # self.pep_emb = torch.nn.Embedding(num_pep_nodes, hidden_channels)
        # self.prot_emb = torch.nn.Embedding(num_prot_nodes, hidden_channels)
        self.hidden_channels = hidden_channels
        self.feat_src = feat_src
        # Instantiate homogeneous GNN:
        if self.feat_src == "esm_emb":
            self.gnn = GNN(
                in_channels * 2,
                hidden_channels,
                gnn_method,
                dropout_ratio,
                dropping_method,
                add_skip_connection,
                num_gnn_layers,
            )
        else:
            self.gnn = GNN(
                in_channels,
                hidden_channels,
                gnn_method,
                dropout_ratio,
                dropping_method,
                add_skip_connection,
                num_gnn_layers,
            )

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=metadata, debug=False)
        self.classifier = Classifier()
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_ratio),
        #     nn.Linear(self.hidden_channels * 2, hidden_channels),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_ratio),
        #     nn.Linear(self.hidden_channels, 1),
        # )

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
            # x_dict = {
            #     "pep": self.pep_lin(data["pep"].x),
            #     "prot": self.prot_lin(data["prot"].x),
            # }
            x_dict = {
                "pep": data["pep"].x,
                "prot": data["prot"].x,
            }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        # print(x_dict.dtype)
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["pep"],
            x_dict["prot"],
            data["pep", "bind", "prot"].edge_label_index,
        )

        # edge_label_index = data["pep", "bind", "prot"].edge_label_index
        # edge_feat_pep = x_dict["pep"][edge_label_index[0]]
        # edge_feat_prot = x_dict["prot"][edge_label_index[1]]
        # feat = torch.cat([edge_feat_pep, edge_feat_prot], dim=-1)
        # pred = self.classifier(feat).squeeze()
        return pred
