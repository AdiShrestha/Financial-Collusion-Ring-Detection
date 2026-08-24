"""Production Confirmatory Prediction Runner on Sealed Test Partition.

Contract C13-01 (T-COMP): Unlocks the sealed test split (V_test) and executes a single-pass
confirmatory evaluation using all 35 trained model checkpoints (7 models x 5 seeds), archiving
immutable per-candidate predictions to runs/production_confirmatory/predictions.json.
"""

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.features.aml_features import AMLFeatureBuilder
from source.models.cell_net import CellularComplexNet
from source.models.edge_aware_gnn import GINEBaseline
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.ph.cache_pipeline import TopologicalFeatureCache
from source.training.production_trainer import LOCKED_SEEDS, MODEL_NAMES


class ProductionConfirmatoryRunner:
    """Executes single-pass evaluation across all 35 checkpoints on test partition."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        candidates_path: str = "data/processed/candidates.jsonl",
        manifest_path: str = "data/manifests/split_manifest.json",
        cache_path: str = "data/cache/topological_features.npz",
        output_path: str = "runs/production_confirmatory/predictions.json",
        device: str = "cpu",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.candidates_path = candidates_path
        self.manifest_path = manifest_path
        self.cache_path = cache_path
        self.output_path = output_path
        self.device = torch.device(device)
        self.feature_builder = AMLFeatureBuilder(use_scaler=True)

    def _instantiate_model(self, model_name: str) -> nn.Module:
        name = model_name.lower()
        if name == "gcn":
            return GCNBaseline(in_dim=16, hidden_dim=64, num_layers=2, out_dim=2)
        elif name == "gat":
            return GATBaseline(in_dim=16, hidden_dim=64, num_layers=2, out_dim=2, num_heads=4)
        elif name == "graphsage":
            return GraphSAGEBaseline(in_dim=16, hidden_dim=64, num_layers=2, out_dim=2)
        elif name == "gine":
            return GINEBaseline(node_dim=16, edge_dim=8, hidden_dim=64, num_layers=2)
        elif name == "scnn":
            return SimplicialComplexNet(in_dim_0=16, in_dim_1=8, in_dim_2=8, hidden_dim=64, num_layers=2, out_dim=2)
        elif name == "ccnn":
            return CellularComplexNet(in_dim_0=16, in_dim_1=8, in_dim_2=16, hidden_dim=64, num_layers=2, out_dim=2)
        elif name == "toporingnet":
            return TopoRingNet(
                in_dim_node=16,
                in_dim_edge=8,
                in_dim_cell=16,
                in_dim_topo=372,
                hidden_dim=64,
                num_layers=2,
                out_dim=2,
                backbone_type="cellular",
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _prepare_test_items(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """Extract test candidates and construct feature tensors."""
        if not os.path.exists(self.candidates_path):
            raise FileNotFoundError(f"Missing candidates file: {self.candidates_path}")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Missing manifest file: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_content = f.read()
            manifest_sha256 = hashlib.sha256(manifest_content.encode("utf-8")).hexdigest()
            manifest_data = json.loads(manifest_content)

        test_cid_set = set()
        splits_data = manifest_data.get("splits", {})
        test_split = splits_data.get("test", {})
        if isinstance(test_split, dict):
            test_cid_set = set(str(x) for x in test_split.get("candidate_ids", []))
        elif isinstance(test_split, list):
            for item in test_split:
                cid = item.get("candidate_id") if isinstance(item, dict) else str(item)
                if cid:
                    test_cid_set.add(str(cid))

        # Fit feature builder exclusively on train split
        candidates = []
        train_txs = []
        with open(self.candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cand = json.loads(line)
                    candidates.append(cand)
                    cid = str(cand.get("candidate_id", ""))
                    if cid not in test_cid_set:
                        train_txs.extend(cand.get("transactions", []))

        self.feature_builder.fit(train_txs)

        # Load PH cache
        ph_data = {}
        if os.path.exists(self.cache_path):
            cache_loader = TopologicalFeatureCache()
            loaded_cache = cache_loader.load_cache(self.cache_path)
            for idx, cid in enumerate(loaded_cache["candidate_ids"]):
                ph_data[str(cid)] = loaded_cache["features"][idx]

        test_items = []
        for cand in candidates:
            cid = str(cand.get("candidate_id", ""))
            if cid in test_cid_set:
                tensor_dict = self.feature_builder.build_candidate_features(cand, is_train=False)
                ph_vec = ph_data.get(cid, np.zeros(372, dtype=np.float32))
                tensor_dict["x_topo"] = torch.tensor(ph_vec, dtype=torch.float32).unsqueeze(0)
                tensor_dict["candidate_id"] = cid
                tensor_dict["group_id"] = str(cand.get("group_id", f"g_{cid}"))
                tensor_dict["label"] = int(cand.get("label", cand.get("is_laundering", 0)))
                tensor_dict["cycle_length"] = int(cand.get("cycle_length", max(3, tensor_dict["num_nodes"].item())))

                k = tensor_dict["cycle_length"]
                b1 = torch.zeros((k, k), dtype=torch.float32)
                for i in range(k):
                    b1[i, i] = -1.0
                    b1[(i + 1) % k, i] = 1.0
                tensor_dict["b1"] = b1
                tensor_dict["b2"] = torch.ones((k, 1), dtype=torch.float32)
                tensor_dict["x_cell"] = torch.zeros((1, 16), dtype=torch.float32)
                tensor_dict["x_2_simp"] = torch.zeros((0, 8), dtype=torch.float32)

                test_items.append(tensor_dict)

        return test_items, manifest_data, manifest_sha256

    def _forward_model(self, model: nn.Module, model_name: str, item: Dict[str, Any]) -> float:
        name = model_name.lower()
        x = item["x"].to(self.device)
        edge_index = item["edge_index"].to(self.device)
        edge_attr = item["edge_attr"].to(self.device)
        x_topo = item["x_topo"].to(self.device)

        if name in ("gcn", "gat", "graphsage"):
            logits = model(x, edge_index=edge_index)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        elif name == "gine":
            out = model(x, edge_index, edge_attr=edge_attr)
            logits = torch.cat([-out, out], dim=-1)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        elif name == "scnn":
            b1 = item["b1"].to(self.device)
            b2 = item["b2"].to(self.device)
            x_1 = edge_attr
            x_2 = item["x_2_simp"].to(self.device)
            logits = model(x, x1=x_1, x2=x_2, b1=b1, b2=b2)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        elif name == "ccnn":
            b1 = item["b1"].to(self.device)
            b2 = item["b2"].to(self.device)
            x_1 = edge_attr
            x_cell = item["x_cell"].to(self.device)
            logits = model(x, x1=x_1, x2=x_cell, b1=b1, b2=b2)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        elif name == "toporingnet":
            b1 = item["b1"].to(self.device)
            b2 = item["b2"].to(self.device)
            x_1 = edge_attr
            x_cell = item["x_cell"].to(self.device)
            struct_out = model.backbone(x, x1=x_1, x2=x_cell, b1=b1, b2=b2)
            logits = model(struct_out, z_topo=x_topo)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        else:
            raise ValueError(f"Unknown model: {name}")

        return max(0.0, min(1.0, prob))

    def run_confirmatory_evaluation(self) -> Dict[str, Any]:
        """Execute single-pass evaluation of 35 checkpoints on test set and write ledger."""
        test_items, manifest_data, manifest_sha256 = self._prepare_test_items()

        cand_records: Dict[str, Dict[str, Any]] = {}
        for item in test_items:
            cid = item["candidate_id"]
            cand_records[cid] = {
                "candidate_id": cid,
                "group_id": item["group_id"],
                "label": item["label"],
                "cycle_length": item["cycle_length"],
                "model_predictions": {m: [] for m in MODEL_NAMES},
            }

        for model_name in MODEL_NAMES:
            for seed in LOCKED_SEEDS:
                ckpt_path = os.path.join(self.checkpoint_dir, f"{model_name}_seed{seed}.pt")
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")

                model = self._instantiate_model(model_name).to(self.device)
                ckpt_data = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt_data["model_state_dict"])
                model.eval()

                with torch.no_grad():
                    for item in test_items:
                        cid = item["candidate_id"]
                        prob = self._forward_model(model, model_name, item)
                        cand_records[cid]["model_predictions"][model_name].append(prob)

        predictions_list = list(cand_records.values())

        output_payload = {
            "total_test_candidates": len(predictions_list),
            "metadata": {
                "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_test_candidates": len(predictions_list),
                "models": MODEL_NAMES,
                "seeds": LOCKED_SEEDS,
                "split_manifest_sha256": manifest_sha256,
            },
            "predictions": predictions_list,
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        return output_payload


if __name__ == "__main__":
    runner = ProductionConfirmatoryRunner()
    res = runner.run_confirmatory_evaluation()
    print(f"Confirmatory evaluation complete. Evaluated {res['metadata']['total_test_candidates']} test candidates across 35 checkpoints.")
