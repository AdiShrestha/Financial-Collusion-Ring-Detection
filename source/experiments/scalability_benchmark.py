"""Computational complexity and efficiency scalability benchmark profiler."""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from source.data.candidate_extractor import CandidateExample
from source.models.cell_net import CellularComplexNet
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.graph_view import GraphView


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_scale_synthetic_candidate(num_nodes: int, node_dim: int = 56) -> Dict[str, Any]:
    """Generate connected cycle/clique synthetic candidate complex at specified scale |V|."""
    nodes = [f"N_{i}" for i in range(num_nodes)]
    edges = []
    for i in range(num_nodes):
        next_i = (i + 1) % num_nodes
        edges.append((nodes[i], nodes[next_i], {"amount": 100.0, "timestamp": float(i + 1)}))

    # Add a cross chord for 3-cliques if |V| >= 3
    if num_nodes >= 3:
        edges.append((nodes[0], nodes[2], {"amount": 50.0, "timestamp": 2.5}))

    cand = CandidateExample(
        candidate_id=f"c_scale_{num_nodes}",
        dataset_track="amlworld",
        temporal_bounds=(1.0, float(num_nodes)),
        participant_ids=nodes,
        edges=edges,
        node_features={n: [float(i + idx) for idx in range(node_dim)] for i, n in enumerate(nodes)},
        target_y=1,
        typology_label="CYCLE",
        group_id=f"g_scale_{num_nodes}",
    )

    cell_view = CycleCellView.from_candidate_example(cand, max_k=6)
    simp_view = CliqueSimplicialView.from_candidate_example(cand)
    graph_view = GraphView.from_candidate_example(cand)

    return {
        "candidate_id": cand.candidate_id,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "cell_view": cell_view,
        "simplicial_view": simp_view,
        "graph_view": graph_view,
        "z_topo": np.ones(372, dtype=np.float32),
        "target_y": 1,
    }


class ScalabilityProfiler:
    """Measures inference latency, peak memory, and parameter counts across scales |V|."""

    MODEL_REGISTRY = {
        "GCNBaseline": GCNBaseline,
        "GATBaseline": GATBaseline,
        "GraphSAGEBaseline": GraphSAGEBaseline,
        "SimplicialComplexNet": SimplicialComplexNet,
        "CellularComplexNet": CellularComplexNet,
        "TopoRingNet": TopoRingNet,
    }

    def __init__(
        self,
        in_dim_node: int = 56,
        in_dim_edge: int = 2,
        in_dim_cell: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
        out_dim: int = 2,
    ):
        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.in_dim_cell = in_dim_cell
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

    def _init_model(self, model_name: str) -> nn.Module:
        if model_name == "GCNBaseline":
            return GCNBaseline(
                in_dim=self.in_dim_node,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                num_layers=self.num_layers,
            )
        elif model_name == "GATBaseline":
            return GATBaseline(
                in_dim=self.in_dim_node,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                num_layers=self.num_layers,
                num_heads=2,
            )
        elif model_name == "GraphSAGEBaseline":
            return GraphSAGEBaseline(
                in_dim=self.in_dim_node,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                num_layers=self.num_layers,
            )
        elif model_name == "SimplicialComplexNet":
            return SimplicialComplexNet(
                in_dim_0=self.in_dim_node,
                in_dim_1=self.in_dim_edge,
                in_dim_2=self.in_dim_cell,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                out_dim=self.out_dim,
            )
        elif model_name == "CellularComplexNet":
            return CellularComplexNet(
                in_dim_0=self.in_dim_node,
                in_dim_1=self.in_dim_edge,
                in_dim_2=self.in_dim_cell,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                out_dim=self.out_dim,
            )
        elif model_name == "TopoRingNet":
            return TopoRingNet(
                in_dim_node=self.in_dim_node,
                in_dim_edge=self.in_dim_edge,
                in_dim_cell=self.in_dim_cell,
                in_dim_topo=372,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                out_dim=self.out_dim,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _forward_model(self, model: nn.Module, item: Dict[str, Any]) -> torch.Tensor:
        if isinstance(model, TopoRingNet):
            view = item.get("cell_view", item.get("domain_view"))
            z_t = item.get("z_topo")
            z_topo_t = torch.tensor(z_t, dtype=torch.float32) if z_t is not None else torch.zeros(372, dtype=torch.float32)
            return model(view, z_topo=z_topo_t)
        elif isinstance(model, CellularComplexNet):
            return model(item.get("cell_view", item.get("domain_view")))
        elif isinstance(model, SimplicialComplexNet):
            return model(item.get("simplicial_view", item.get("domain_view")))
        else:
            return model(item.get("graph_view", item.get("domain_view")))

    def profile_model(
        self,
        model: nn.Module,
        item: Dict[str, Any],
        num_warmup: int = 10,
        num_repeats: int = 50,
    ) -> Dict[str, float]:
        """Profile single model on candidate item measuring latency, memory, and parameter counts."""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self._forward_model(model, item)

        # Timed execution
        timings = []
        with torch.no_grad():
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                _ = self._forward_model(model, item)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)  # ms

        mean_lat = float(np.mean(timings))
        std_lat = float(np.std(timings))
        p95_lat = float(np.percentile(timings, 95))
        throughput_qps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

        num_params = count_parameters(model)
        param_memory_mb = float(num_params * 4 / (1024 * 1024))  # float32 = 4 bytes

        return {
            "mean_latency_ms": mean_lat,
            "std_latency_ms": std_lat,
            "p95_latency_ms": p95_lat,
            "throughput_qps": throughput_qps,
            "trainable_parameters": num_params,
            "param_memory_mb": param_memory_mb,
            "num_repeats": num_repeats,
        }

    def run_scalability_suite(
        self,
        scales: Sequence[int] = (10, 25, 50, 100),
        model_names: Optional[Sequence[str]] = None,
        num_warmup: int = 5,
        num_repeats: int = 20,
    ) -> Dict[str, Any]:
        """Execute full scalability sweep across model architectures and graph scales |V|."""
        target_models = list(model_names) if model_names is not None else list(self.MODEL_REGISTRY.keys())
        results: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in target_models}

        for scale in scales:
            cand_item = generate_scale_synthetic_candidate(num_nodes=scale, node_dim=self.in_dim_node)

            for model_name in target_models:
                model = self._init_model(model_name)
                metrics = self.profile_model(
                    model=model,
                    item=cand_item,
                    num_warmup=num_warmup,
                    num_repeats=num_repeats,
                )
                results[model_name][scale] = metrics

        return {
            "results": results,
            "scales_evaluated": list(scales),
            "models_evaluated": target_models,
            "timestamp": time.time(),
        }
