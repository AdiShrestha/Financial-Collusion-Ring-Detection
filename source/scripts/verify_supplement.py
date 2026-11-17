#!/usr/bin/env python3
"""Re-run inference and scoring from the released worlds and models.

Run from the supplement root:

  python code/scripts/verify_supplement.py            # all pilot and held-out worlds
  python code/scripts/verify_supplement.py --worlds W5 W6

Step 1 (no labels): rebuild every stored model, predict every candidate, and compare with the
released prediction files. Neural weights load with torch.load(weights_only=True). The GBDT is
refitted from models/gbdt_training_matrix.csv (the fit is deterministic); pass --use-gbdt-pickle
to load the stored scikit-learn objects instead (pickle: only for files you trust).
Step 2 (labels): score each released prediction file with grouped-tie AP and tie-aware AUROC
and compare with the released per-seed metric tables.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from posthoc_span_rule import grouped_ap, read_jsonl, tie_auroc  # noqa: E402

SUMMARY_ARM = "fixed20_tabular_neural_control"
GBDT_ARM = "gbdt_fixed20"
SEEDS = (11, 23, 37, 53, 71)


def load_benchmark():
    spec = importlib.util.spec_from_file_location("s1_benchmark", HERE / "check_s1_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def local(root, recorded):
    """Map a recorded model path onto the supplement layout."""
    tail = Path(recorded).as_posix().split("frozen_models/", 1)[1]
    return root / "models" / "frozen" / tail


def build_world(bench, world_dir):
    data = bench.load_inputs(world_dir)
    n_nodes, n_events = len(data["node_order"]), len(data["event_order"])
    views = []
    for row in data["rows"]:
        b1, b2 = bench.cellular(row, n_nodes, n_events)
        views.append({"r": row, "cell": (b1, b2), "simp": bench.simplicial(row, n_nodes, n_events)})
    return data, views


def neural_model(record, bench):
    arm, width = record["arm"], record.get("hidden_width")
    if arm == SUMMARY_ARM:
        return nn.Sequential(nn.Linear(20, 81), nn.ReLU(), nn.Linear(81, 81), nn.ReLU(), nn.Linear(81, 1))
    return {
        "cellular_cwn_style_local": lambda: bench.Cell(width),
        "cellular_hasse_mechanism_control": lambda: bench.Cell(width, hasse=True),
        "directed_local_edge_gnn": lambda: bench.Directed(width),
        "simplicial_mpsn_style_local": lambda: bench.Simp(width),
    }[arm]()


def gbdt_from_matrix(path):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows = list(csv.DictReader(path.open()))
    features = [k for k in rows[0] if k not in ("world_id", "row", "label", "sample_weight")]
    X = np.asarray([[float(r[f]) for f in features] for r in rows])
    y = np.asarray([int(r["label"]) for r in rows])
    w = np.asarray([float(r["sample_weight"]) for r in rows])
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
    return model.fit(X, y, sample_weight=w)


def staged70(model, X):
    for k, s in enumerate(model.staged_decision_function(X), 1):
        if k == 70:
            return np.asarray(s, float).reshape(-1)
    raise ValueError("model has fewer than 70 rounds")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=HERE.parents[1], help="supplement root")
    parser.add_argument(
        "--worlds", nargs="*", help="aliases to check (default: all pilot and held-out worlds)"
    )
    parser.add_argument("--use-gbdt-pickle", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max |logit difference| accepted")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    torch.set_num_threads(1)
    bench = load_benchmark()
    lock = json.loads((root / "models/model_lock.json").read_text())
    transform = json.loads((root / "models/frozen/fitted_transform.json").read_text())
    mean, scale = np.asarray(transform["mean_float64"]), np.asarray(transform["scale_float64"])
    worlds = [
        w
        for w in json.loads((root / "data/world_index.json").read_text())["worlds"]
        if w["split"] in ("pilot", "heldout") and (not args.worlds or w["alias"] in args.worlds)
    ]
    models = []
    gbdt_refit = None if args.use_gbdt_pickle else gbdt_from_matrix(root / "models/gbdt_training_matrix.csv")
    for record in lock["models"]:
        path = local(root, record["frozen"]["path"])
        if record["arm"] == GBDT_ARM:
            if args.use_gbdt_pickle:
                import pickle

                with path.open("rb") as stream:
                    model = pickle.load(stream)
            else:
                model = gbdt_refit
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = neural_model(record, bench)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
        models.append((record["arm"], record["seed"], model))
    metric_tables = {}
    for split, rel in (
        ("pilot", "results/metrics_by_world_initialization.csv"),
        ("heldout", "results/replication/metrics_by_world_initialization.csv"),
    ):
        for r in csv.DictReader((root / rel).open()):
            metric_tables[(split, r["arm"], r["world"], r["initialization"])] = r
    worst_logit = worst_ap = worst_auc = 0.0
    files = 0
    for w in worlds:
        world_dir = root / w["path"]
        data, views = build_world(bench, world_dir)
        ids = [r["candidate_id"] for r in data["ids"]]
        raw = np.asarray([bench.gbdt_row(r) for r in data["rows"]], float)
        normalized = ((raw - mean) / scale).astype(np.float32)
        labels = {r["candidate_id"]: r["label"] for r in read_jsonl(world_dir / "candidate_labels.jsonl")}
        folder = root / "predictions" / ("pilot_W1-W4" if w["split"] == "pilot" else "heldout_W5-W34")
        for arm, seed, model in models:
            if arm == GBDT_ARM:
                logits = staged70(model, raw)
            else:
                with torch.inference_mode():
                    logits = np.asarray(
                        [
                            float(
                                (
                                    model(torch.as_tensor(normalized[i]))
                                    if arm == SUMMARY_ARM
                                    else model(*bench.build_args(v, arm))
                                ).reshape(-1)[0]
                            )
                            for i, v in enumerate(views)
                        ]
                    )
            matches = list(folder.glob(f'*_{arm}_seed_{seed}_{w["world_id"]}.csv'))
            if len(matches) != 1:
                raise FileNotFoundError(f'prediction file for {arm} seed {seed} {w["world_id"]}')
            released = {r["candidate_id"]: float(r["raw_logit"]) for r in csv.DictReader(matches[0].open())}
            diff = max(abs(released[c] - z) for c, z in zip(ids, logits))
            worst_logit = max(worst_logit, diff)
            if diff > args.tolerance:
                raise SystemExit(f'{arm} seed {seed} {w["alias"]}: logits differ by {diff:.3g}')
            y = [labels[c] for c in released]
            s = list(released.values())
            ref = metric_tables[(w["split"], arm, w["alias"], f"I{SEEDS.index(seed) + 1}")]
            worst_ap = max(worst_ap, abs(float(grouped_ap(y, s)) - float(ref["ap"])))
            worst_auc = max(worst_auc, abs(float(tie_auroc(y, s)) - float(ref["auroc"])))
            files += 1
        print(f'{w["alias"]}: 30 models checked')
    print(
        f"{files} prediction files; max |logit diff| {worst_logit:.3g}; "
        f"max |AP diff| {worst_ap:.3g}; max |AUROC diff| {worst_auc:.3g}"
    )
    if worst_ap > 1e-9 or worst_auc > 1e-9:
        raise SystemExit("metrics differ from the released tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
