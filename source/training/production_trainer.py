"""Production multi-seed model training suite across 5 pre-registered random seeds.

Contract C12-04 (T-COMP): Trains all 7 architectures (GCN, GAT, GraphSAGE, GINE, SCNN, CCNN, TopoRingNet)
across 5 locked seeds (42, 43, 44, 45, 46) on train split, validates with early stopping on validation split,
and saves 35 genuine PyTorch checkpoint weights to checkpoints/<model>_seed<S>.pt.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.features.aml_features import AMLFeatureBuilder
from source.models.cell_net import CellularComplexNet
from source.models.edge_aware_gnn import GINEBaseline
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.ph.cache_pipeline import TopologicalFeatureCache

LOCKED_SEEDS = [42, 43, 44, 45, 46]
MODEL_NAMES = ["gcn", "gat", "graphsage", "gine", "scnn", "ccnn", "toporingnet"]


class ProductionTrainer:
    """Trains and serializes production checkpoints for all 7 models across 5 seeds."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        cache_path: str = "data/cache/topological_features.npz",
        manifest_path: str = "data/manifests/split_manifest.json",
        candidates_path: str = "data/processed/candidates.jsonl",
        device: str = "cpu",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.cache_path = cache_path
        self.manifest_path = manifest_path
        self.candidates_path = candidates_path
        self.device = torch.device(device)
        self.feature_builder = AMLFeatureBuilder(use_scaler=True)

    def _instantiate_model(self, model_name: str, seed: int) -> nn.Module:
        """Instantiate model with fixed seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)

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

    def _prepare_data_items(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load candidates, features, and split assignments."""
        if not os.path.exists(self.candidates_path):
            raise FileNotFoundError(f"Missing candidates file: {self.candidates_path}")

        # Load PH cache
        ph_data = {}
        if os.path.exists(self.cache_path):
            cache_loader = TopologicalFeatureCache()
            loaded_cache = cache_loader.load_cache(self.cache_path)
            for idx, cid in enumerate(loaded_cache["candidate_ids"]):
                ph_data[str(cid)] = loaded_cache["features"][idx]

        # Load split manifest
        split_map = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            for split_name, s_val in m_data.get("splits", {}).items():
                canon = "val" if split_name in ("val", "validation") else split_name
                if isinstance(s_val, dict):
                    for cid in s_val.get("candidate_ids", []):
                        split_map[str(cid)] = canon
                elif isinstance(s_val, list):
                    for item in s_val:
                        cid = item.get("candidate_id") if isinstance(item, dict) else str(item)
                        if cid:
                            split_map[cid] = canon

        # Load candidates
        candidates = []
        with open(self.candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        # Fit feature builder exclusively on train split
        train_txs = []
        for c in candidates:
            cid = str(c.get("candidate_id", ""))
            if split_map.get(cid, "train") == "train":
                train_txs.extend(c.get("transactions", []))

        self.feature_builder.fit(train_txs)

        train_items = []
        val_items = []
        test_items = []

        for c in candidates:
            cid = str(c.get("candidate_id", ""))
            split_tag = split_map.get(cid, "train")
            tensor_dict = self.feature_builder.build_candidate_features(c, is_train=(split_tag == "train"))

            ph_vec = ph_data.get(cid, np.zeros(372, dtype=np.float32))
            tensor_dict["x_topo"] = torch.tensor(ph_vec, dtype=torch.float32).unsqueeze(0)
            tensor_dict["candidate_id"] = cid

            k = int(c.get("cycle_length", max(3, tensor_dict["num_nodes"].item())))
            b1 = torch.zeros((k, k), dtype=torch.float32)
            for i in range(k):
                b1[i, i] = -1.0
                b1[(i + 1) % k, i] = 1.0
            tensor_dict["b1"] = b1
            tensor_dict["b2"] = torch.ones((k, 1), dtype=torch.float32)
            tensor_dict["x_cell"] = torch.zeros((1, 16), dtype=torch.float32)
            tensor_dict["x_2_simp"] = torch.zeros((0, 8), dtype=torch.float32)

            if split_tag == "train":
                train_items.append(tensor_dict)
            elif split_tag == "val":
                val_items.append(tensor_dict)
            elif split_tag == "test":
                test_items.append(tensor_dict)

        return train_items, val_items, test_items

    def _forward_model(self, model: nn.Module, model_name: str, item: Dict[str, Any]) -> torch.Tensor:
        """Call forward method according to model signature."""
        name = model_name.lower()
        x = item["x"].to(self.device)
        edge_index = item["edge_index"].to(self.device)
        edge_attr = item["edge_attr"].to(self.device)
        x_topo = item["x_topo"].to(self.device)

        if name in ("gcn", "gat", "graphsage"):
            out = model(x, edge_index=edge_index)
            return out
        elif name == "gine":
            out = model(x, edge_index, edge_attr=edge_attr)
            return torch.cat([-out, out], dim=-1)
        elif name == "scnn":
            b1 = item["b1"].to(self.device)
            b2 = item["b2"].to(self.device)
            x_1 = edge_attr
            x_2 = item["x_2_simp"].to(self.device)
            out = model(x, x1=x_1, x2=x_2, b1=b1, b2=b2)
            return out
        elif name == "ccnn":
            b1 = item["b1"].to(self.device)
            b2 = item["b2"].to(self.device)
            x_1 = edge_attr
            x_cell = item["x_cell"].to(self.device)
            out = model(x, x1=x_1, x2=x_cell, b1=b1, b2=b2)
            return out
        elif name == "toporingnet":
            b1 = item["b1"].to(self.device)
            b2 = item["b2"].to(self.device)
            x_1 = edge_attr
            x_cell = item["x_cell"].to(self.device)
            struct_out = model.backbone(x, x1=x_1, x2=x_cell, b1=b1, b2=b2)
            out = model(struct_out, z_topo=x_topo)
            return out
        else:
            raise ValueError(f"Unknown model: {name}")

    def train_single_model(
        self,
        model_name: str,
        seed: int,
        train_items: List[Dict[str, Any]],
        val_items: List[Dict[str, Any]],
        max_epochs: int = 20,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Train a single model architecture on a specific seed with validation early stopping."""
        model = self._instantiate_model(model_name, seed).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_auprc = -1.0
        best_epoch = 0
        best_state = None
        no_improve_count = 0

        train_losses = []
        val_losses = []
        val_auprcs = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0

            indices = np.random.permutation(len(train_items))
            for idx in indices:
                item = train_items[idx]
                optimizer.zero_grad()
                logits = self._forward_model(model, model_name, item)
                target = item["y"].to(self.device)
                loss = criterion(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item())

            train_loss = epoch_loss / max(1, len(train_items))
            train_losses.append(train_loss)

            model.eval()
            val_loss_tot = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for item in val_items:
                    logits = self._forward_model(model, model_name, item)
                    target = item["y"].to(self.device)
                    loss = criterion(logits, target)
                    val_loss_tot += float(loss.item())

                    prob_pos = torch.softmax(logits, dim=-1)[0, 1].item()
                    val_preds.append(prob_pos)
                    val_targets.append(int(target.item()))

            val_loss = val_loss_tot / max(1, len(val_items))
            val_losses.append(val_loss)

            if len(set(val_targets)) > 1:
                val_auprc = float(average_precision_score(val_targets, val_preds))
            else:
                val_auprc = 0.5
            val_auprcs.append(val_auprc)

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience and epoch >= 5:
                    break

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        checkpoint = {
            "model_name": model_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auprc": best_val_auprc,
            "model_state_dict": best_state,
            "training_history": {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_auprc": val_auprcs,
            },
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(self.checkpoint_dir, f"{model_name}_seed{seed}.pt")
        torch.save(checkpoint, ckpt_path)

        return {
            "model_name": model_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auprc": best_val_auprc,
            "checkpoint_path": ckpt_path,
        }

    def train_all_production_models(
        self,
        seeds: Optional[List[int]] = None,
        models: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Train all 7 models across 5 seeds and save 35 checkpoints."""
        target_seeds = seeds or LOCKED_SEEDS
        target_models = models or MODEL_NAMES

        train_items, val_items, _ = self._prepare_data_items()
        results = []

        for model_name in target_models:
            for seed in target_seeds:
                res = self.train_single_model(model_name, seed, train_items, val_items)
                results.append(res)

        return results


if __name__ == "__main__":
    trainer = ProductionTrainer()
    runs = trainer.train_all_production_models()
    print(f"Successfully trained and saved {len(runs)} production checkpoints.")
