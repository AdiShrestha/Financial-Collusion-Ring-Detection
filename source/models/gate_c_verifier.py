"""Gate C Architecture Realizability & Overfitting Smoke Test Verification Engine."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from source.data.candidate_extractor import CandidateExample
from source.models.cell_net import CellularComplexNet
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.ph.persistence_extractor import PersistenceExtractor
from source.ph.ph_graph_view import PHGraphView
from source.ph.vectorizer import PersistenceVectorizer
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.graph_view import GraphView


def generate_gate_c_synthetic_batch(size: int = 10) -> List[Dict[str, Any]]:
    """Generate a deterministic synthetic batch of candidate examples with domain views and PH vectors."""
    np.random.seed(42)
    torch.manual_seed(42)

    batch_items = []
    extractor = PersistenceExtractor()
    vectorizer = PersistenceVectorizer()

    for idx in range(size):
        if idx in [1, 3, 5, 7, 9]:
            is_cycle = True
            k = [3, 4, 3, 5, 6][((idx - 1) // 2) % 5]
            typology = "CYCLE"
            target_y = 1
        else:
            is_cycle = False
            k = [3, 4, 5, 4, 6][(idx // 2) % 5]
            typology = "NEGATIVE_CANDIDATE"
            target_y = 0

        nodes = [f"acc_{idx}_{i}" for i in range(k)]

        edges = []
        if is_cycle:
            # Closed directed cycle
            for i in range(k):
                next_i = (i + 1) % k
                edges.append(
                    (nodes[i], nodes[next_i], {
                        "amount": 100.0 + (i * 15.0) + (idx * 5.0),
                        "timestamp": 10.0 * (i + 1),
                    })
                )
        else:
            # Open path / chain
            for i in range(k - 1):
                edges.append(
                    (nodes[i], nodes[i + 1], {
                        "amount": 50.0 + (i * 10.0),
                        "timestamp": 10.0 * (i + 1),
                    })
                )

        node_feat_dict = {}
        for i, n in enumerate(nodes):
            feat = [0.0] * 56
            if is_cycle:
                feat[0] = 1.0
                feat[1] = 1.0
            else:
                feat[0] = 0.0
                feat[1] = 1.0 if (0 < i < k - 1) else 0.0
            feat[2] = float(k)
            feat[3] = float(i + 1)
            for feat_i in range(4, 56):
                feat[feat_i] = float((i + 1) * 0.1 + feat_i * 0.02 + idx * 0.05)
            node_feat_dict[n] = feat

        cand = CandidateExample(
            candidate_id=f"gate_c_cand_{idx:02d}",
            dataset_track="amlworld",
            temporal_bounds=(10.0, float(k * 10.0)),
            participant_ids=nodes,
            edges=edges,
            node_features=node_feat_dict,
            target_y=target_y,
            typology_label=typology,
            group_id=f"g_gate_c_{idx}",
        )

        g_view = GraphView.from_candidate_example(cand)
        s_view = CliqueSimplicialView.from_candidate_example(cand)
        c_view = CycleCellView.from_candidate_example(cand)

        ph_graph = PHGraphView.from_candidate_example(cand, filtration_type="temporal")
        diag = PersistenceExtractor.compute_diagram(ph_graph)
        z_topo = vectorizer.vectorize(diag)

        batch_items.append({
            "candidate": cand,
            "graph_view": g_view,
            "simplicial_view": s_view,
            "cell_view": c_view,
            "z_topo": z_topo,
            "target_y": target_y,
        })

    return batch_items


class GateCVerifier:
    """Automated engine certifying Gate C architecture realizability and memorization capability."""

    def __init__(self, output_report_path: str = "project/gate_c_report.json"):
        self.output_report_path = output_report_path

    def run_memorization_smoke_test(
        self,
        batch_items: Optional[List[Dict[str, Any]]] = None,
        max_epochs: int = 100,
        lr: float = 0.02,
    ) -> Dict[str, Any]:
        """Execute overfitting memorization test across all 6 model architectures."""
        if batch_items is None:
            batch_items = generate_gate_c_synthetic_batch(size=10)

        targets = torch.tensor([item["target_y"] for item in batch_items], dtype=torch.long)

        models_to_test = {
            "GCNBaseline": GCNBaseline(in_dim=56, hidden_dim=64, out_dim=2, num_layers=2, dropout=0.0),
            "GATBaseline": GATBaseline(in_dim=56, hidden_dim=64, out_dim=2, num_layers=2, dropout=0.0),
            "GraphSAGEBaseline": GraphSAGEBaseline(in_dim=56, hidden_dim=64, out_dim=2, num_layers=2, dropout=0.0),
            "SimplicialComplexNet": SimplicialComplexNet(in_dim_0=56, in_dim_1=2, in_dim_2=2, hidden_dim=64, num_layers=2, out_dim=2, dropout=0.0),
            "CellularComplexNet": CellularComplexNet(in_dim_0=56, in_dim_1=2, in_dim_2=2, hidden_dim=64, num_layers=2, out_dim=2, dropout=0.0),
            "TopoRingNet": TopoRingNet(in_dim_node=56, in_dim_edge=2, in_dim_cell=2, in_dim_topo=372, hidden_dim=64, num_layers=2, out_dim=2, dropout=0.0),
        }

        results: Dict[str, Any] = {}
        criterion = nn.CrossEntropyLoss()
        all_passed = True

        for model_name, model in models_to_test.items():
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            initial_loss = None
            final_loss = None
            final_acc = 0.0

            for epoch in range(max_epochs):
                optimizer.zero_grad()
                logits_list = []

                for item in batch_items:
                    if model_name in ["GCNBaseline", "GATBaseline", "GraphSAGEBaseline"]:
                        log_i = model(item["graph_view"])
                    elif model_name == "SimplicialComplexNet":
                        log_i = model(item["simplicial_view"])
                    elif model_name == "CellularComplexNet":
                        log_i = model(item["cell_view"])
                    elif model_name == "TopoRingNet":
                        log_i = model(item["cell_view"], z_topo=item["z_topo"])
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                    logits_list.append(log_i)

                batch_logits = torch.cat(logits_list, dim=0)
                loss = criterion(batch_logits, targets)

                if initial_loss is None:
                    initial_loss = float(loss.item())

                loss.backward()
                optimizer.step()

                final_loss = float(loss.item())
                preds = torch.argmax(batch_logits, dim=-1)
                final_acc = float((preds == targets).sum().item()) / float(len(targets))

                if final_loss < 0.05 and final_acc == 1.0:
                    break

            # Gradient audit
            param_grads = {}
            has_dead_params = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        has_dead_params = True
                        param_grads[name] = 0.0
                    else:
                        norm_val = float(torch.norm(param.grad).item())
                        param_grads[name] = norm_val
                        if norm_val == 0.0 or np.isnan(norm_val):
                            has_dead_params = True

            model_passed = (final_acc == 1.0) and (final_loss < 0.05) and (not has_dead_params)
            if not model_passed:
                all_passed = False

            results[model_name] = {
                "passed": model_passed,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "final_accuracy": final_acc,
                "epochs_trained": epoch + 1,
                "has_dead_parameters": has_dead_params,
                "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "gradient_norm_summary": {
                    "min_grad_norm": min(param_grads.values()) if param_grads else 0.0,
                    "max_grad_norm": max(param_grads.values()) if param_grads else 0.0,
                },
            }

        gate_c_status = "GATE_C_PASS" if all_passed else "GATE_C_FAIL"

        report = {
            "gate_c_status": gate_c_status,
            "num_models_tested": len(models_to_test),
            "batch_size": len(batch_items),
            "max_epochs_allowed": max_epochs,
            "models": results,
            "all_models_achieved_100_percent_accuracy": all(r["final_accuracy"] == 1.0 for r in results.values()),
            "all_models_converged": all(r["final_loss"] < 0.05 for r in results.values()),
            "all_gradients_healthy": all(not r["has_dead_parameters"] for r in results.values()),
        }

        # Write certification report
        os.makedirs(os.path.dirname(os.path.abspath(self.output_report_path)), exist_ok=True)
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
