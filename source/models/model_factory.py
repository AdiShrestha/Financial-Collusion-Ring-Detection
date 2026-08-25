"""Unified Model Factory for Tabular, GNN, Edge-Aware, and Cellular Complex Architectures.

Contract C17-03 (T-COMP): Exposes standard instantiation factory for all benchmarked models:
Tabular (LR, HGB), GNN Baselines (GCN, GAT, GraphSAGE), Edge-Aware GNN (GINE),
and Topological Complexes (SCNN, CCNN).
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from source.models.cell_net import CellularComplexNet
from source.models.edge_aware_gnn import GINEBaseline
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.simplicial_net import SimplicialNet


def create_model(
    model_name: str,
    node_dim: int = 6,
    edge_dim: int = 9,
    cell_dim: int = 9,
    tabular_dim: int = 17,
    hidden_dim: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    random_seed: int = 42,
) -> Any:
    """Instantiate model by canonical identifier."""
    name = model_name.lower().strip().replace("-", "_")

    if name in ("logistic_regression", "lr"):
        return LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed)

    elif name in ("hist_gradient_boosting", "hgb", "gradient_boosting"):
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=random_seed,
        )

    elif name == "gcn":
        torch.manual_seed(random_seed)
        return GCNBaseline(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=2,
            dropout=dropout,
        )

    elif name == "gat":
        torch.manual_seed(random_seed)
        return GATBaseline(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=2,
            out_dim=2,
            dropout=dropout,
        )

    elif name in ("graphsage", "sage"):
        torch.manual_seed(random_seed)
        return GraphSAGEBaseline(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=2,
            dropout=dropout,
        )

    elif name == "gine":
        torch.manual_seed(random_seed)
        return GINEBaseline(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif name in ("simplicial_net", "scnn"):
        torch.manual_seed(random_seed)
        return SimplicialNet(
            in_dim_0=node_dim,
            in_dim_1=edge_dim,
            in_dim_2=cell_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
        )

    elif name in ("cell_complex_net", "ccnn", "cellular_net"):
        torch.manual_seed(random_seed)
        return CellularComplexNet(
            in_dim_0=node_dim,
            in_dim_1=edge_dim,
            in_dim_2=cell_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
