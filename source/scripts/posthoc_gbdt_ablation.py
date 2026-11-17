#!/usr/bin/env python3
"""Post hoc GBDT feature ablations (exploratory; not part of the held-out protocol).

Refits the selected GBDT policy (scikit-learn HistGradientBoostingClassifier: learning rate
0.1, at most 7 leaves, at least 2 samples per leaf, no L2 penalty, early stopping off,
800 rounds, every training candidate weighted 1/312) on the four development training
worlds, once per feature subset. The staged round is chosen exactly as for the published
model: lowest equal-world validation BCE on the two development validation worlds among
every tenth round. Each refit is then scored once on the pilot and held-out worlds.
The all-20-feature refit must reproduce the published held-out world APs.

  python posthoc_gbdt_ablation.py --world-index data/world_index.json \
      --published-ap results/replication/ap_by_world.csv --output-csv gbdt_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_span_rule import grouped_ap, read_jsonl, tie_auroc  # noqa: E402

FEATURES = [
    "cycle_length",
    "selected_time_min",
    "selected_time_mean",
    "selected_time_max",
    "selected_time_population_sd",
    "selected_log_amount_min",
    "selected_log_amount_mean",
    "selected_log_amount_max",
    "selected_log_amount_population_sd",
    "selected_time_span",
    "selected_log_amount_span",
    "central_in_degree_min",
    "central_in_degree_mean",
    "central_in_degree_max",
    "central_out_degree_min",
    "central_out_degree_mean",
    "central_out_degree_max",
    "selected_directed_pair_multiplicity_min",
    "selected_directed_pair_multiplicity_mean",
    "selected_directed_pair_multiplicity_max",
]
TIME = [f for f in FEATURES if f.startswith("selected_time")]
SUBSETS = {
    "all_20": FEATURES,
    "without_time_span": [f for f in FEATURES if f != "selected_time_span"],
    "without_time_features": [f for f in FEATURES if f not in TIME],
    "time_span_only": ["selected_time_span"],
    "cycle_length_and_time_span": ["cycle_length", "selected_time_span"],
}


def load_benchmark():
    path = Path(__file__).resolve().with_name("check_s1_benchmark.py")
    spec = importlib.util.spec_from_file_location("s1_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def world_matrix(bench, world_dir):
    data = bench.load_inputs(world_dir)
    labels = {r["candidate_id"]: r["label"] for r in read_jsonl(world_dir / "candidate_labels.jsonl")}
    order = [r["candidate_id"] for r in data["ids"]]
    X = np.asarray([bench.gbdt_row(r) for r in data["rows"]], dtype=float)
    y = np.asarray([labels[c] for c in order], dtype=int)
    if X.shape != (78, len(FEATURES)) or y.sum() != 6:
        raise ValueError(f"{world_dir}: unexpected frame")
    return X, y


def bce(y, z):
    z = np.asarray(z, float)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def staged(model, X, rounds):
    wanted = set(rounds)
    return {
        k: np.asarray(s, float) for k, s in enumerate(model.staged_decision_function(X), 1) if k in wanted
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--world-index", type=Path, required=True)
    parser.add_argument(
        "--published-ap", type=Path, help="held-out ap_by_world.csv for the reproduction check"
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--training-matrix-csv",
        type=Path,
        help="also write the 312 x 20 GBDT training matrix with labels and weights",
    )
    args = parser.parse_args(argv)
    if args.output_csv.exists():
        raise FileExistsError(args.output_csv)
    bench = load_benchmark()
    base = (
        args.world_index.resolve().parent.parent
        if args.world_index.parent.name == "data"
        else args.world_index.resolve().parent
    )
    worlds = json.loads(args.world_index.read_text())["worlds"]
    data = {w["world_id"]: (w, world_matrix(bench, base / w["path"])) for w in worlds}
    by_split = lambda s: [v for v in data.values() if v[0]["split"] == s]
    train, val = by_split("development_train"), by_split("development_validation")
    if len(train) != 4 or len(val) != 2:
        raise ValueError("expected four training and two validation worlds")
    Xtr = np.vstack([m[0] for _, m in train])
    ytr = np.concatenate([m[1] for _, m in train])
    weights = np.full(len(ytr), 1 / len(ytr))
    if args.training_matrix_csv is not None:
        with args.training_matrix_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["world_id", "row", *FEATURES, "label", "sample_weight"])
            for w, (X, y) in train:
                for i in range(len(y)):
                    writer.writerow(
                        [w["world_id"], i, *map(repr, X[i].tolist()), int(y[i]), repr(1 / len(ytr))]
                    )
    rows = []
    for name, cols in SUBSETS.items():
        ix = [FEATURES.index(c) for c in cols]
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_leaf_nodes=7,
            min_samples_leaf=2,
            l2_regularization=0.0,
            early_stopping=False,
            validation_fraction=None,
            max_iter=800,
            random_state=11,
        )
        model.fit(Xtr[:, ix], ytr, sample_weight=weights)
        grid = range(10, 801, 10)
        val_scores = [staged(model, m[0][:, ix], grid) for _, m in val]
        losses = {r: float(np.mean([bce(m[1], s[r]) for (_, m), s in zip(val, val_scores)])) for r in grid}
        best = min(grid, key=lambda r: (losses[r], r))
        for w, (X, y) in data.values():
            if not w["split"].startswith(("pilot", "heldout")):
                continue
            z = staged(model, X[:, ix], [best])[best]
            rows.append(
                {
                    "subset": name,
                    "features": " ".join(cols),
                    "selected_round": best,
                    "validation_bce": f"{losses[best]:.10f}",
                    "split": w["split"],
                    "alias": w["alias"],
                    "world_id": w["world_id"],
                    "ap": f"{float(grouped_ap(list(y), list(z))):.12f}",
                    "auroc": f"{float(tie_auroc(list(y), list(z))):.12f}",
                }
            )
        held = [r for r in rows if r["subset"] == name and r["split"] == "heldout"]
        print(
            f"{name}: round {best}, held-out mean AP {np.mean([float(r['ap']) for r in held]):.4f}, "
            f"AUROC {np.mean([float(r['auroc']) for r in held]):.4f}"
        )
    if args.published_ap is not None:
        published = {
            r["world"]: float(r["mean_ap"])
            for r in csv.DictReader(args.published_ap.open())
            if r["arm"] == "gbdt_fixed20"
        }
        refit = {
            r["alias"]: float(r["ap"]) for r in rows if r["subset"] == "all_20" and r["split"] == "heldout"
        }
        worst = max(abs(published[k] - refit[k]) for k in published)
        print(f"all-20 refit vs published GBDT world AP: max abs difference {worst:.3g}")
        if worst > 1e-9:
            raise SystemExit("refit does not reproduce the published GBDT")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
