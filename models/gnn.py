"""HeteroGNN model for narrative graph embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import networkx as nx
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", message=".*node types.*do not get updated during message passing.*")


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for narrative story graphs.

    Architecture:
        Per-node-type LayerNorm -> Projection (in -> hidden) ->
        N x HeteroConv(SAGEConv) with residual connections ->
        Story node readout -> Linear projection (hidden -> out)
    """

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: int = 512,
        out_channels: int = 2048,
        num_layers: int = 3,
        metadata: Optional[Tuple] = None,
        aggr: str = "sum",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.metadata = metadata
        self.edge_types = metadata[1] if metadata else []

        # Per-node-type LayerNorm
        self.feature_norms = nn.ModuleDict()
        if metadata:
            for node_type in metadata[0]:
                self.feature_norms[node_type] = nn.LayerNorm(in_channels)

        # Projection layer: in_channels -> hidden_channels
        self.proj = nn.Linear(in_channels, hidden_channels)

        # Linear transforms for nodes without incoming edges
        self.node_lins = nn.ModuleDict()
        if metadata:
            for node_type in metadata[0]:
                self.node_lins[node_type] = nn.ModuleList(
                    [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
                )

        # Residual projection for first layer (in_channels projected -> hidden)
        self.residual_projs = nn.ModuleDict()
        if metadata:
            for node_type in metadata[0]:
                self.residual_projs[f"{node_type}_0"] = nn.Linear(hidden_channels, hidden_channels)

        # GNN layers
        self.convs = nn.ModuleList()
        if metadata:
            for _ in range(num_layers):
                conv_dict = {et: SAGEConv(hidden_channels, hidden_channels) for et in metadata[1]}
                self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        # Output projection
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        # LayerNorm per node type
        x_dict = {
            nt: self.feature_norms[nt](x) if nt in self.feature_norms else x
            for nt, x in x_dict.items()
        }

        # Project to hidden dimension
        x_dict = {nt: self.proj(x) for nt, x in x_dict.items()}

        # Store for residual
        x_initial = {k: v.clone() for k, v in x_dict.items()}

        # GNN message passing layers
        for i, conv in enumerate(self.convs):
            edge_filled = {}
            for et in self.edge_types:
                if et in edge_index_dict:
                    edge_filled[et] = edge_index_dict[et]
                else:
                    edge_filled[et] = torch.empty((2, 0), dtype=torch.long, device=next(self.parameters()).device)

            x_new = conv(x_dict, edge_filled)

            for nt in x_dict:
                if nt in x_new and x_new[nt] is not None:
                    h = F.relu(x_new[nt])
                else:
                    h = F.relu(self.node_lins[nt][i](x_dict[nt]))

                # Residual connection
                if i == 0:
                    key = f"{nt}_0"
                    if key in self.residual_projs:
                        h = h + self.residual_projs[key](x_initial[nt])
                else:
                    h = h + x_dict[nt]

                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)

                x_dict[nt] = h

        # Story node readout (CLS-token analogue)
        if "Story" in x_dict and x_dict["Story"].size(0) > 0:
            graph_emb = torch.mean(x_dict["Story"], dim=0)
        else:
            pooled = [torch.mean(x, dim=0) for x in x_dict.values() if x.size(0) > 0]
            graph_emb = torch.mean(torch.stack(pooled), dim=0) if pooled else torch.zeros(self.hidden_channels, device=next(self.parameters()).device)

        return self.lin(graph_emb)


def nx_to_pyg_hetero(G: nx.DiGraph) -> HeteroData:
    """Convert a NetworkX narrative graph to PyTorch Geometric HeteroData."""
    data = HeteroData()

    node_types = set(d.get("type", "Unknown") for _, d in G.nodes(data=True))
    node_mappings = {}

    for nt in node_types:
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == nt]
        if not nodes:
            continue
        node_mappings[nt] = {node: idx for idx, node in enumerate(nodes)}
        embeddings = []
        for n in nodes:
            emb = G.nodes[n].get("embedding")
            if emb is not None:
                embeddings.append(np.array(emb) if not isinstance(emb, np.ndarray) else emb)
            else:
                embeddings.append(np.zeros(384))
        data[nt].x = torch.FloatTensor(np.array(embeddings))

    edge_dict = {}
    for src, dst, ed in G.edges(data=True):
        src_type = G.nodes[src].get("type", "Unknown")
        dst_type = G.nodes[dst].get("type", "Unknown")
        rel = ed.get("rel", "connected_to")
        et = (src_type, rel, dst_type)
        if et not in edge_dict:
            edge_dict[et] = []
        if src_type in node_mappings and dst_type in node_mappings:
            si = node_mappings[src_type].get(src)
            di = node_mappings[dst_type].get(dst)
            if si is not None and di is not None:
                edge_dict[et].append([si, di])

    for et, edges in edge_dict.items():
        if edges:
            data[et].edge_index = torch.LongTensor(edges).t().contiguous()

    return data


def get_metadata_from_data(data: HeteroData) -> Tuple:
    """Extract (node_types, edge_types) from HeteroData."""
    return (list(data.node_types), list(data.edge_types))
