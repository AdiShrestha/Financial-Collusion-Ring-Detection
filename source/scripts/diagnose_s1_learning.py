#!/usr/bin/env python3
"""Run the bounded S1 learning-capability diagnostics on indexed development worlds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROCESS_START_MONO = time.monotonic()
PROCESS_START_UTC = datetime.now(timezone.utc).isoformat()

import numpy as np
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ARMS = (
    "cellular_cwn_style_local",
    "simplicial_mpsn_style_local",
    "directed_local_edge_gnn",
    "cellular_hasse_mechanism_control",
)
SEEDS = (11, 23, 37, 53, 71)
RATES = (0.001, 0.01, 0.03)
TRAIN_IDS = ("S1_dev_101", "S1_dev_103", "S1_dev_107", "S1_dev_109")
VAL_IDS = ("S1_dev_113", "S1_dev_127")
SUMMARY_LIMIT = 3000
MICRO_LIMIT = 1500
EVAL_EVERY = 25
BLOCK_EPOCHS = 100
WALL_LIMIT = 5400
OUTPUT_LIMIT = 536870912
WALL_RESERVE = 600
OUTPUT_RESERVE = 32 * 1024 * 1024
EXPECTED_WIDTHS = {
    "cellular_cwn_style_local": 34,
    "simplicial_mpsn_style_local": 36,
    "directed_local_edge_gnn": 36,
    "cellular_hasse_mechanism_control": 30,
}
STABILITY_RULE = {
    "window_epochs": 200,
    "minimum_epoch": 1600,
    "confirmation_tail_epochs": 400,
    "training_BCE_relative_range_max": 0.01,
    "per_world_validation_AP_range_max": 0.01,
    "adjacent_windows_required": 2,
}
INPUT_NAMES = (
    "candidate_identity.jsonl",
    "candidate_labels.jsonl",
    "candidate_observations.jsonl",
    "world_node_identity.json",
    "world_directed_multigraph_identity.jsonl",
    "cellular_polygonal_boundaries.jsonl",
    "simplicial_subdivision.jsonl",
    "hasse_incidence.jsonl",
    "sidecars/persisted_events.jsonl",
)


class BudgetStop(Exception):
    """A planned stop that preserves the current attempt and its partial state."""


class NonfiniteTrial(Exception):
    """A numeric failure belonging to one retained model trial."""


def sha_file(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def tensor_sha(tensors) -> dict:
    digest = hashlib.sha256()
    descriptions = []
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        array = value.numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        descriptions.append({"shape": list(array.shape), "dtype": str(array.dtype), "bytes": array.nbytes})
    return {"sha256": digest.hexdigest(), "tensors": descriptions}


def sha_array(array: np.ndarray) -> dict:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return {"sha256": digest.hexdigest(), "shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes}


def strict_json(path: Path):
    return TRAIN_HELPERS.read_json(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def output_bytes(root: Path, extras=()) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            total += path.stat().st_size
    for path in extras:
        if path is not None and Path(path).is_file():
            total += Path(path).stat().st_size
    return total


def measured_resources() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = float(usage.ru_maxrss)
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    return {
        "measurement": "resource.getrusage(RUSAGE_SELF)",
        "peak_resident_set_bytes": peak_bytes,
        "peak_resident_set_native_units": peak,
        "peak_resident_set_native_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }


def elapsed_seconds() -> float:
    return time.monotonic() - PROCESS_START_MONO


def current_environment() -> dict:
    # Record relevant runtime settings without dumping unrelated credentials.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "MPLCONFIGDIR", "XDG_CACHE_HOME", "PYTHONHASHSEED", "PYTHONNOUSERSITE")
    return {key: os.environ[key] for key in names if key in os.environ}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load project helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers():
    global TRAIN_HELPERS, BENCH
    TRAIN_HELPERS = import_module(ROOT / "source/scripts/train_s1_development.py", "s1_f_train_helpers")
    BENCH = TRAIN_HELPERS.load_bench()
    return TRAIN_HELPERS, BENCH


def strict_unique(rows, key: str, label: str) -> dict:
    result = {}
    for row in rows:
        if key not in row:
            raise ValueError(f"missing {key} in {label}")
        value = row[key]
        if value in result:
            raise ValueError(f"duplicate {key} in {label}: {value!r}")
        result[value] = row
    return result


def strict_join(identity_rows, label_rows, world_id: str) -> dict:
    identities = strict_unique(identity_rows, "candidate_id", f"{world_id} identity")
    labels = strict_unique(label_rows, "candidate_id", f"{world_id} labels")
    if set(identities) != set(labels):
        raise ValueError(f"strict identity/label join mismatch in {world_id}")
    for candidate_id, row in labels.items():
        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"invalid integer label for {world_id}/{candidate_id}")
    return labels


def finite_values(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite values in {name}")
    return array


def fixture_join_and_finite_checks() -> dict:
    duplicate_rejected = mismatch_rejected = nonfinite_rejected = False
    try:
        strict_unique([{"candidate_id": "x"}, {"candidate_id": "x"}], "candidate_id", "qualification fixture")
    except ValueError:
        duplicate_rejected = True
    try:
        strict_join([{"candidate_id": "x"}], [{"candidate_id": "y", "label": 0}], "qualification fixture")
    except ValueError:
        mismatch_rejected = True
    try:
        finite_values([1.0, float("nan")], "qualification fixture")
    except ValueError:
        nonfinite_rejected = True
    if not (duplicate_rejected and mismatch_rejected and nonfinite_rejected):
        raise ValueError("strict join/nonfinite focused qualification failed")
    return {
        "duplicate_identity_rejected": duplicate_rejected,
        "unmatched_label_join_rejected": mismatch_rejected,
        "nonfinite_numeric_rejected": nonfinite_rejected,
        "scope": "small API fixtures only; no fixture values entered an experiment",
    }


def resolve_worlds(index_path: Path, study: dict) -> tuple[dict, list]:
    index = strict_json(index_path)
    if type(index.get("worlds")) is not list:
        raise ValueError("world_index.json lacks its worlds list")
    train_ids = tuple(study["training_worlds"])
    val_ids = tuple(study["validation_worlds"])
    if train_ids != TRAIN_IDS or val_ids != VAL_IDS:
        raise ValueError("S1-F world allowlist differs from the declared contract")
    required = set(train_ids + val_ids)
    entries = {}
    excluded = []
    for entry in index["worlds"]:
        world_id = entry.get("world_id")
        if type(entry.get("replay")) is not bool:
            raise ValueError(f"world index replay flag must be boolean: {world_id}")
        is_replay = entry.get("replay") is True
        if is_replay or world_id not in required:
            excluded.append({"world_id": world_id, "attempt": entry.get("attempt"), "replay": is_replay, "status": entry.get("status")})
            continue
        if world_id in entries:
            raise ValueError(f"duplicate non-replay world index entry: {world_id}")
        expected_role = "provisional_development_train" if world_id in train_ids else "provisional_development_validation"
        if entry.get("role") != expected_role or entry.get("status") != "COMPLETED":
            raise ValueError(f"indexed world status/role mismatch: {world_id}")
        if type(entry.get("exit_code")) is not int or entry["exit_code"] != 0:
            raise ValueError(f"indexed world has no successful exit: {world_id}")
        attempt_value = entry.get("attempt")
        if type(attempt_value) is not str or not attempt_value:
            raise ValueError(f"indexed world attempt path missing: {world_id}")
        attempt = Path(attempt_value)
        if not attempt.is_absolute():
            attempt = ROOT / attempt
        attempt = Path(os.path.abspath(attempt))
        if not attempt.is_relative_to(ROOT):
            raise ValueError(f"indexed world attempt is outside project or is a symlink: {world_id}")
        cursor = ROOT
        for part in attempt.relative_to(ROOT).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"indexed world attempt contains a symlink component: {world_id}")
        attempt = attempt.resolve(strict=True)
        entries[world_id] = {"path": attempt, "role": expected_role, "seed": entry.get("seed"), "indexed_attempt": attempt_value}
    if set(entries) != required:
        raise ValueError(f"world index does not exactly cover the six authorized worlds: {sorted(set(entries) ^ required)}")
    return entries, excluded


def ensure_no_symlink_components(path: Path) -> None:
    raw = Path(os.path.abspath(path))
    if not raw.is_relative_to(ROOT):
        raise ValueError(f"input path escapes the project: {path}")
    current = ROOT
    relative = raw.relative_to(ROOT)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked input component: {current}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"input path escapes the project after resolution: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"required input is not a file: {resolved}")


def independent_fixed20(row: dict, native_times: list, native_amounts: list) -> list:
    selected = [int(x) for x in row["selected_idx"]]
    central = [i for i, value in enumerate(row["central"]) if float(value) > 0]
    times = [float(native_times[i]) for i in selected]
    amounts = [float(native_amounts[i]) for i in selected]
    src = [int(x) for x in row["src"]]
    dst = [int(x) for x in row["dst"]]
    indegree = Counter(dst)
    outdegree = Counter(src)
    pair_counts = Counter(zip(src, dst))
    in_values = [indegree[i] for i in central]
    out_values = [outdegree[i] for i in central]
    multi_values = [pair_counts[(src[i], dst[i])] for i in selected]

    def stats(values):
        mean = math.fsum(values) / len(values)
        variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
        return min(values), mean, max(values), math.sqrt(variance)

    def triplet(values):
        return min(values), math.fsum(values) / len(values), max(values)

    time_stats = stats(times)
    amount_stats = stats(amounts)
    return [
        float(row["cycle_length"]), *time_stats, *amount_stats,
        max(times) - min(times), max(amounts) - min(amounts),
        *triplet(in_values), *triplet(out_values), *triplet(multi_values),
    ]


def inspect_physical_world(world: dict, entry: dict, helper) -> dict:
    attempt = entry["path"]
    for name in INPUT_NAMES:
        ensure_no_symlink_components(attempt / name)
    identity_rows = helper.read_jsonl(attempt / "candidate_identity.jsonl")
    label_rows = helper.read_jsonl(attempt / "candidate_labels.jsonl")
    observation_rows = helper.read_jsonl(attempt / "candidate_observations.jsonl")
    labels_by_id = strict_join(identity_rows, label_rows, world["world_id"])
    if len(identity_rows) != 78 or len(observation_rows) != 78:
        raise ValueError(f"complete 78-candidate census required in {world['world_id']}")
    joined_ids = [x["candidate_id"] for x in identity_rows]
    if set(joined_ids) != set(labels_by_id):
        raise ValueError(f"strict helper identity/label join mismatch in {world['world_id']}")

    data = world["data"]
    graph_by_event = strict_unique(data["graph"], "physical_event_id", f"{world['world_id']} physical events")
    native_rows = helper.read_jsonl(attempt / "sidecars/persisted_events.jsonl")
    native_by_event = strict_unique(native_rows, "physical_event_id", f"{world['world_id']} native event sidecar")
    event_order = [int(x) for x in data["event_order"]]
    if set(event_order) != set(graph_by_event) or set(event_order) != set(native_by_event):
        raise ValueError(f"physical event coordinates do not join exactly in {world['world_id']}")
    native_times, native_amounts = [], []
    endpoint_pairs = []
    for event_id in event_order:
        native = native_by_event[event_id]
        graph = graph_by_event[event_id]
        fields = next(csv.reader([native["csv_line"]]))
        if len(fields) != 11:
            raise ValueError(f"native transaction row has unexpected width: {world['world_id']}/{event_id}")
        if fields[3] != graph["source_account"] or fields[6] != graph["target_account"]:
            raise ValueError(f"native CSV endpoints do not match physical event identity: {world['world_id']}/{event_id}")
        step = int(fields[0])
        amount = finite_values([float(fields[2])], f"native amount {world['world_id']}/{event_id}")[0]
        if amount <= 0:
            raise ValueError(f"nonpositive native amount: {world['world_id']}/{event_id}")
        native_times.append(step / 120.0)
        native_amounts.append(math.log1p(float(amount)))
        endpoint_pairs.append((graph["source_account"], graph["target_account"], event_id))
    finite_values(native_times, f"native time steps {world['world_id']}")
    finite_values(native_amounts, f"native log amounts {world['world_id']}")
    if not np.array_equal(np.asarray(data["rows"][0]["times"], dtype=np.float64), np.asarray(native_times, dtype=np.float64)):
        raise ValueError(f"full-world native time channel mismatch: {world['world_id']}")
    if not np.array_equal(np.asarray(data["rows"][0]["amounts"], dtype=np.float64), np.asarray(native_amounts, dtype=np.float64)):
        raise ValueError(f"full-world native amount channel mismatch: {world['world_id']}")

    feature_rows = []
    maximum_feature_gap = 0.0
    for index, (identity, observation, view) in enumerate(zip(identity_rows, observation_rows, world["views"])):
        candidate_id = identity["candidate_id"]
        if candidate_id != view["candidate_id"] or labels_by_id[candidate_id]["label"] != view["label"]:
            raise ValueError(f"candidate label join differs from actual helper path: {world['world_id']}/{candidate_id}")
        if observation["world_event_time_steps"] != native_times or observation["world_event_log1p_amounts"] != native_amounts:
            raise ValueError(f"candidate whole-world feature channels differ from native CSV: {world['world_id']}/{candidate_id}")
        selected_idx = [int(x) for x in view["r"]["selected_idx"]]
        # The selected-value sidecar follows the candidate's canonical event
        # order; the binary mask follows whole-world coordinate order.
        event_position = {event_id: j for j, event_id in enumerate(event_order)}
        canonical_idx = [event_position[event_id] for event_id in identity["event_ids"]]
        selected_times = [native_times[i] for i in canonical_idx]
        selected_amounts = [native_amounts[i] for i in canonical_idx]
        if observation["time_steps"] != selected_times or observation["log1p_amounts"] != selected_amounts:
            raise ValueError(f"selected native feature channel mismatch: {world['world_id']}/{candidate_id}")
        actual = [float(x) for x in BENCH.gbdt_row(view["r"])]
        reference = independent_fixed20(view["r"], native_times, native_amounts)
        if len(actual) != 20 or len(reference) != 20:
            raise ValueError("fixed20 extraction did not return exactly twenty values")
        gap = max(abs(a - b) for a, b in zip(actual, reference))
        maximum_feature_gap = max(maximum_feature_gap, gap)
        if not np.allclose(actual, reference, atol=1e-12, rtol=1e-12):
            raise ValueError(f"fixed20 native/degree/multiplicity arithmetic differs: {world['world_id']}/{candidate_id}")
        feature_rows.append({
            "world_id": world["world_id"], "candidate_id": candidate_id,
            "label": int(view["label"]), "features": actual,
        })

    ordered_pairs = defaultdict(list)
    loops = []
    for source, target, event_id in endpoint_pairs:
        ordered_pairs[(source, target)].append(int(event_id))
        if source == target:
            loops.append(int(event_id))
    parallel = [
        {"source": pair[0], "target": pair[1], "physical_event_ids": sorted(ids)}
        for pair, ids in sorted(ordered_pairs.items()) if len(ids) > 1
    ]
    reciprocal = []
    seen_unordered = set()
    for source, target in sorted(ordered_pairs):
        if source == target:
            continue
        unordered = tuple(sorted((source, target)))
        if unordered in seen_unordered:
            continue
        reverse = ordered_pairs.get((target, source), [])
        if reverse:
            reciprocal.append({
                "accounts": list(unordered),
                "forward_event_ids": sorted(ordered_pairs[(source, target)]),
                "reverse_event_ids": sorted(reverse),
            })
            seen_unordered.add(unordered)

    closure_checked = 0
    consumed_closure_failures = []
    for index, view in enumerate(world["views"]):
        b1, b2 = view["cell"]
        simp = view["simp"]
        cellular_closed = bool(BENCH.closed(b1, b2))
        simplicial_closed = bool(BENCH.closed(simp[3], simp[4]))
        if not cellular_closed or not simplicial_closed:
            consumed_closure_failures.append(view["candidate_id"])
        closure_checked += 1
    if consumed_closure_failures:
        raise ValueError(f"exact B1@B2 closure failed in {world['world_id']}: {consumed_closure_failures[:3]}")

    world["feature_records"] = feature_rows
    world["native"] = {"times": native_times, "log_amounts": native_amounts}
    world["identity_summary"] = {
        "world_id": world["world_id"],
        "attempt": str(attempt.relative_to(ROOT)),
        "physical_events": len(event_order),
        "physical_event_id_count": len(set(event_order)),
        "loops": {"count": len(loops), "physical_event_ids": sorted(loops)},
        "parallel_directed_pairs": parallel,
        "reciprocal_pairs": reciprocal,
        "candidate_rows": len(identity_rows),
        "candidate_sidecar_joined_rows": len(world["views"]),
        "exact_cellular_and_simplicial_closure_candidates": closure_checked,
        "closure_failures": consumed_closure_failures,
        "max_abs_fixed20_native_reference_gap": maximum_feature_gap,
    }
    return world["identity_summary"]


def all_input_paths(entries, index_path, config_path, failure_history_path=None, reference_json_path=None) -> list[Path]:
    paths = [index_path.resolve(strict=True), config_path.resolve(strict=True)]
    paths.extend([
        (ROOT / "source/scripts/check_s1_benchmark.py").resolve(strict=True),
        (ROOT / "source/scripts/train_s1_development.py").resolve(strict=True),
        (ROOT / "source/configs/s1_matched_smoke.json").resolve(strict=True),
        Path(__file__).resolve(strict=True),
    ])
    for optional_path in (failure_history_path, reference_json_path):
        if optional_path is not None:
            paths.append(Path(optional_path).resolve(strict=True))
    for entry in entries.values():
        paths.extend(entry["path"] / name for name in INPUT_NAMES)
    unique = {}
    for path in paths:
        ensure_no_symlink_components(path)
        unique[str(path)] = path
    return [unique[key] for key in sorted(unique)]


def feature_matrix(worlds, columns) -> tuple[np.ndarray, list[dict]]:
    records = [row for world in worlds for row in world["feature_records"]]
    if len(columns) != 20 or any(len(row["features"]) != 20 for row in records):
        raise ValueError("fixed20 feature order or width is invalid")
    matrix = finite_values([row["features"] for row in records], "fixed20 feature matrix")
    if matrix.shape != (len(records), 20):
        raise ValueError(f"unexpected fixed20 matrix shape: {matrix.shape}")
    return matrix, records


def fit_normalizer(training: np.ndarray) -> dict:
    values = finite_values(training, "training summary matrix")
    means = np.mean(values, axis=0, dtype=np.float64)
    scales = np.std(values, axis=0, ddof=0, dtype=np.float64)
    constant = np.all(values == values[0], axis=0)
    # Floating-point accumulation can give repeated non-binary decimals a tiny
    # nonzero SD. Exactly constant columns have zero population variance.
    scales[constant] = 0.0
    means[constant] = values[0, constant]
    effective = scales.copy()
    effective[constant] = 1.0
    if not np.isfinite(means).all() or not np.isfinite(effective).all():
        raise ValueError("nonfinite training-only normalization")
    return {"mean": means, "population_sd": scales, "scale": effective, "constant_columns": constant}


def apply_normalizer(values: np.ndarray, transform: dict) -> np.ndarray:
    x = finite_values(values, "summary features before scaling")
    normalized64 = (x - transform["mean"]) / transform["scale"]
    if not np.isfinite(normalized64).all():
        raise ValueError("nonfinite normalized summary features")
    result = normalized64.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite float32 summary tensor")
    return result


def exact_vector_collisions(records, columns) -> dict:
    groups = defaultdict(list)
    for row in records:
        vector = np.asarray(row["features"], dtype=np.float64)
        groups[vector.tobytes(order="C")].append(row)
    duplicate_groups = []
    for rows in groups.values():
        if len(rows) > 1:
            duplicate_groups.append({
                "count": len(rows),
                "identities": [{"world_id": x["world_id"], "candidate_id": x["candidate_id"], "label": x["label"]} for x in rows],
                "conflicting_labels": len({x["label"] for x in rows}) > 1,
            })
    return {
        "scope": "exact equality of the unchanged float64 gbdt_row vectors across the six authorized worlds",
        "columns": columns,
        "duplicate_vector_group_count": len(duplicate_groups),
        "duplicate_row_count": sum(x["count"] for x in duplicate_groups),
        "conflicting_label_group_count": sum(bool(x["conflicting_labels"]) for x in duplicate_groups),
        "groups": duplicate_groups,
        "interpretation": "No collision, if observed, rules out only an exact-vector obstruction; it does not establish representation adequacy.",
    }


def build_microset(train_worlds) -> list[dict]:
    selected = []
    for world in train_worlds:
        by_label = defaultdict(list)
        for view in world["views"]:
            by_label[int(view["label"])].append(view)
        for label in (1, 0):
            candidates = sorted(by_label[label], key=lambda x: x["candidate_id"])
            if len(candidates) < 2:
                raise ValueError(f"microset lacks two exact examples of class {label} in {world['world_id']}")
            for view in candidates[:2]:
                selected.append({
                    "world_id": world["world_id"],
                    "candidate_id": view["candidate_id"],
                    "label": int(view["label"]),
                    "view": view,
                })
    if len(selected) != 16 or Counter(x["label"] for x in selected) != Counter({0: 8, 1: 8}):
        raise ValueError("microset is not the declared 16-row balanced training capability subset")
    return selected


def make_summary_model() -> nn.Module:
    return nn.Sequential(nn.Linear(20, 81), nn.ReLU(), nn.Linear(81, 81), nn.ReLU(), nn.Linear(81, 1))


def copy_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def model_state_digest(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state().clone(),
    }


def activate_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def optimizer_for(model, rate: float):
    return torch.optim.Adam(model.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def scheduler_for(optimizer, spec: dict):
    settings = spec["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(settings["factor"]),
        patience=int(settings["patience_evaluations"]),
        threshold=float(settings["threshold"]),
        threshold_mode=settings["threshold_mode"],
        cooldown=0,
        min_lr=float(settings["min_lr"]),
        eps=float(settings["eps"]),
    )


def scheduler_state_for_json(scheduler) -> dict:
    result = scheduler.state_dict().copy()
    for key in ("best", "mode_worse"):
        value = result.get(key)
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            if not math.isinf(float(value)):
                raise ValueError(f"unexpected nonfinite scheduler field: {key}")
            result[key] = {
                "kind": "scheduler_comparison_sentinel",
                "value": "positive_infinity" if float(value) > 0 else "negative_infinity",
            }
    json.dumps(result, allow_nan=False)
    return result


def new_trial_record(kind: str, trial_id: str, seed: int, rate: float, arm=None) -> dict:
    return {
        "kind": kind,
        "trial_id": trial_id,
        "seed": int(seed),
        "learning_rate": float(rate),
        "arm": arm,
        "epoch": 0,
        "status": "not_started",
        "consecutive_endpoint_evaluations": 0,
        "history": [],
        "model": None,
        "optimizer": None,
        "scheduler": None,
        "rng": None,
        "best_state": None,
        "best_validation_bce": None,
        "best_epoch": None,
        "last_eval_logits": None,
        "last_gradient_norms": None,
        "endpoint_reason": None,
        "started_monotonic": None,
        "duration_seconds": 0.0,
        "trial_dir": None,
        "input_args": None,
        "world_order": None,
    }


def create_trial_state(kind, trial_id, seed, rate, arm, widths, spec, trial_dir, input_data) -> dict:
    state = new_trial_record(kind, trial_id, seed, rate, arm)
    state["trial_dir"] = trial_dir
    if kind == "summary":
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        random.seed(int(seed))
        model = make_summary_model()
        if sum(parameter.numel() for parameter in model.parameters()) != 8425:
            raise ValueError("summary-control model does not have exactly 8425 active parameters")
        model_state = copy_state(model.state_dict())
        input_data["initial_state_hashes"].setdefault(str(seed), {})[str(rate)] = model_state_digest(model_state)
        model.load_state_dict(model_state)
        model.float()
        state["model"] = model
        state["optimizer"] = optimizer_for(model, rate)
        state["scheduler"] = scheduler_for(state["optimizer"], spec)
        state["rng"] = capture_rng()
        state["model_state_sha256"] = model_state_digest(model_state)
        state["world_order"] = input_data["train_worlds"]
        state["x_train"] = input_data["x_train_tensor"]
        state["y_train"] = input_data["y_train_tensor"]
        state["x_validation"] = input_data["x_val_tensor"]
        state["y_validation"] = input_data["y_val_tensor"]
    else:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        random.seed(int(seed))
        model = TRAIN_HELPERS.make_models(BENCH, widths)[arm]
        base_state = copy_state(model.state_dict())
        input_data["initial_state_hashes"].setdefault(arm, {}).setdefault(str(seed), {})[str(rate)] = model_state_digest(base_state)
        model.load_state_dict(base_state)
        state["model"] = model
        state["optimizer"] = optimizer_for(model, rate)
        state["scheduler"] = scheduler_for(state["optimizer"], spec)
        state["rng"] = capture_rng()
        state["model_state_sha256"] = model_state_digest(base_state)
        state["input_args"] = input_data["micro_args"][arm]
        state["world_order"] = list(TRAIN_IDS)
    state["model"].train()
    return state


def check_same_initializations(hashes: dict, kind: str) -> dict:
    groups = []
    if kind == "summary":
        for seed, values in hashes.items():
            groups.append({"seed": int(seed), "rate_state_hashes": values, "identical_across_rates": len(set(values.values())) == 1})
    else:
        for arm, by_seed in hashes.items():
            for seed, values in by_seed.items():
                groups.append({"arm": arm, "seed": int(seed), "rate_state_hashes": values, "identical_across_rates": len(set(values.values())) == 1})
    if any(not x["identical_across_rates"] for x in groups):
        raise ValueError(f"initial model states differ across rates for {kind}")
    return {"groups_checked": len(groups), "identical_across_rates": True, "groups": groups}


def json_records_for_train(worlds, x_tensor, y_by_world):
    offset = 0
    records = {}
    for world in worlds:
        count = len(world["views"])
        records[world["world_id"]] = {
            "x": x_tensor[offset:offset + count],
            "y": y_by_world[world["world_id"]],
            "world": world,
        }
        offset += count
    if offset != len(x_tensor):
        raise ValueError("summary tensor/world identity ordering mismatch")
    return records


def summary_eval(model, world_data, bench) -> dict:
    model.eval()
    by_world = {}
    with torch.no_grad():
        for world_id, data in world_data.items():
            logits = model(data["x"]).reshape(-1)
            if not torch.isfinite(logits).all():
                raise NonfiniteTrial(f"nonfinite summary logits in {world_id}")
            y = data["y"]
            loss = float(nn.functional.binary_cross_entropy_with_logits(logits, y).item())
            values = logits.detach().cpu().numpy().astype(np.float64).tolist()
            labels = y.detach().cpu().numpy().astype(np.int64)
            by_world[world_id] = {
                "bce": loss,
                "ap": float(bench.grouped_ap(labels, values)),
                "auroc": float(bench.tie_auc(labels, values)),
                "logits": values,
            }
    model.train()
    return {
        "per_world": by_world,
        "equal_world_bce": float(np.mean([x["bce"] for x in by_world.values()])),
        "equal_world_ap": float(np.mean([x["ap"] for x in by_world.values()])),
        "equal_world_auroc": float(np.mean([x["auroc"] for x in by_world.values()])),
    }


def summary_train_epoch(state, train_data) -> float:
    model = state["model"]
    optimizer = state["optimizer"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for data in train_data.values():
        logits = model(data["x"]).reshape(-1)
        if not torch.isfinite(logits).all():
            raise NonfiniteTrial(f"nonfinite summary training logits in {data['world']['world_id']}")
        loss = nn.functional.binary_cross_entropy_with_logits(logits, data["y"])
        if not torch.isfinite(loss):
            raise NonfiniteTrial("nonfinite summary training loss")
        losses.append(loss)
    total = torch.stack(losses).mean()
    pre_update = float(total.detach().item())
    total.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise NonfiniteTrial("nonfinite summary gradient")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise NonfiniteTrial("nonfinite summary parameter after optimizer step")
    return pre_update


def micro_eval(model, selected, arm, capture=False) -> dict:
    model.eval()
    predictions = []
    world_losses = defaultdict(list)
    correct = 0
    observer = TensorObserver() if capture else None
    if observer is None:
        context = _NullContext()
    else:
        context = observer
    hooks = []
    if capture:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(_linear_output_hook(name, observer)))
    try:
        with torch.no_grad(), context:
            for row in selected:
                logit = model(*row["args"])
                if not torch.isfinite(logit).all():
                    raise NonfiniteTrial(f"nonfinite microfit logit {row['world_id']}/{row['candidate_id']}")
                z = float(logit.item())
                y = int(row["label"])
                value = float(np.logaddexp(0.0, z) - y * z)
                if not math.isfinite(value):
                    raise NonfiniteTrial("nonfinite microfit BCE")
                world_losses[row["world_id"]].append(value)
                correct += int((z >= 0.0) == bool(y))
                predictions.append({
                    "world_id": row["world_id"], "candidate_id": row["candidate_id"],
                    "label": y, "logit": z,
                })
    finally:
        for hook in hooks:
            hook.remove()
    equal_world_bce = float(np.mean([np.mean(values) for values in world_losses.values()]))
    model.train()
    result = {
        "equal_world_bce": equal_world_bce,
        "correct_at_logit_zero": int(correct),
        "total": len(selected),
        "predictions": predictions,
    }
    if observer is not None:
        result["activation_observations"] = observer.finish()
    return result


def micro_train_epoch(state, selected_by_world) -> float:
    model = state["model"]
    optimizer = state["optimizer"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    world_losses = []
    for world_id in TRAIN_IDS:
        losses = []
        for row in selected_by_world[world_id]:
            logit = model(*row["args"])
            if not torch.isfinite(logit).all():
                raise NonfiniteTrial(f"nonfinite microfit training logit {world_id}/{row['candidate_id']}")
            target = torch.tensor(float(row["label"]), dtype=torch.float32)
            value = nn.functional.binary_cross_entropy_with_logits(logit.reshape(()), target)
            if not torch.isfinite(value):
                raise NonfiniteTrial("nonfinite microfit training BCE")
            losses.append(value)
        world_losses.append(torch.stack(losses).mean())
    total = torch.stack(world_losses).mean()
    pre_update = float(total.detach().item())
    total.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise NonfiniteTrial("nonfinite microfit gradient")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise NonfiniteTrial("nonfinite microfit parameter after optimizer step")
    return pre_update


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Stats:
    def __init__(self):
        self.minimum = math.inf
        self.maximum = -math.inf
        self.count = 0
        self.saturated = 0

    def add(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        with torch._C._DisableTorchDispatch():
            value = tensor.detach().to(dtype=torch.float64)
            if not torch.isfinite(value).all():
                raise NonfiniteTrial("nonfinite intermediate activation observed")
            self.minimum = min(self.minimum, float(value.min().item()))
            self.maximum = max(self.maximum, float(value.max().item()))
            self.count += int(value.numel())
            self.saturated += int((value.abs() >= 0.99).sum().item())

    def value(self, saturation=False):
        if not self.count:
            return {"count": 0, "min": None, "max": None}
        result = {"count": self.count, "min": self.minimum, "max": self.maximum}
        if saturation:
            result["abs_ge_0_99_fraction"] = self.saturated / self.count
        return result


class TensorObserver(TorchDispatchMode):
    """Read functional activations and pooling tensors while returning original tensors unchanged."""
    def __init__(self):
        super().__init__()
        self.stats = defaultdict(_Stats)
        self.op_counts = Counter()

    def collect(self, key, value, saturation=False):
        self.stats[(key, saturation)].add(value)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        name = getattr(getattr(func, "_schema", None), "name", str(func))
        if "tanh" in name:
            if args:
                self.collect("functional_tanh_input", args[0])
            self.collect("functional_tanh_output", result, saturation=True)
        elif any(token in name for token in ("sum", "mean", "index_add", "mm", "matmul")):
            ordinal = self.op_counts[name]
            self.op_counts[name] += 1
            self.collect(f"functional_pool:{name}:{ordinal}", result)
        return result

    def finish(self):
        return {
            "linear_module_outputs": {
                key[0]: stats.value() for key, stats in self.stats.items()
                if key[0].startswith("linear_output:")
            },
            "functional_tanh": {
                "pre": self.stats[("functional_tanh_input", False)].value(),
                "post": self.stats[("functional_tanh_output", True)].value(saturation=True),
            },
            "functional_pooling_and_aggregation": {
                key[0]: stats.value() for key, stats in self.stats.items()
                if key[0].startswith("functional_pool:")
            },
            "forward_operation_counts": dict(self.op_counts),
        }


def _linear_output_hook(name, observer):
    key = f"linear_output:{name or '<root>'}"
    def hook(module, args, output):
        observer.collect(key, output)
    return hook


def gradient_norms_by_layer(model) -> dict:
    accumulators = defaultdict(float)
    missing = []
    for name, parameter in model.named_parameters():
        layer = name.split(".", 1)[0]
        if parameter.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise NonfiniteTrial(f"nonfinite gradient while recording layer norm: {name}")
        accumulators[layer] += float(parameter.grad.detach().double().pow(2).sum().item())
    return {
        "l2_by_layer": {key: math.sqrt(value) for key, value in sorted(accumulators.items())},
        "missing_gradient_parameters": missing,
    }


def micro_observer_qualification(model, args) -> dict:
    model.zero_grad(set_to_none=True)
    baseline = model(*args)
    baseline.square().backward()
    baseline_grads = {name: parameter.grad.detach().clone() if parameter.grad is not None else None for name, parameter in model.named_parameters()}
    baseline_value = baseline.detach().clone()
    model.zero_grad(set_to_none=True)
    observer = TensorObserver()
    hooks = [module.register_forward_hook(_linear_output_hook(name, observer)) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    try:
        with observer:
            observed = model(*args)
            observed.square().backward()
    finally:
        for hook in hooks:
            hook.remove()
    gradients_same = True
    for name, parameter in model.named_parameters():
        before = baseline_grads[name]
        after = parameter.grad
        if (before is None) != (after is None):
            gradients_same = False
        elif before is not None and not torch.equal(before, after):
            gradients_same = False
    logits_same = torch.equal(baseline_value, observed.detach())
    if not logits_same or not gradients_same:
        raise ValueError("observational activation diagnostics changed logits or gradients")
    return {
        "logits_bitwise_identical": logits_same,
        "parameter_gradients_bitwise_identical": gradients_same,
        "observed_functional_tanh_outputs": observer.stats[("functional_tanh_output", True)].count,
        "observed_functional_pool_operations": sum(observer.op_counts.values()),
        "observed_linear_module_outputs": sum(value.count for key, value in observer.stats.items() if key[0].startswith("linear_output:")),
    }


def micro_endpoint_update(counter: int, bce: float, correct: int, total: int) -> int:
    qualifies = math.isfinite(bce) and bce <= 0.02 and correct == total == 16
    return counter + 1 if qualifies else 0


def stability_window(history: list, end_epoch: int) -> dict:
    begin = end_epoch - STABILITY_RULE["window_epochs"]
    by_epoch = {int(row["epoch"]): row for row in history}
    expected = list(range(begin, end_epoch + 1, EVAL_EVERY))
    selected = [by_epoch[x] for x in expected if x in by_epoch]
    if len(selected) != len(expected) or not selected:
        return {"available": False, "stable": False, "end_epoch": end_epoch}
    losses = [float(row["training"]["equal_world_bce"]) for row in selected]
    relative_range = (max(losses) - min(losses)) / max(abs(float(np.mean(losses))), 1e-15)
    ap_ranges = {}
    for world_id in VAL_IDS:
        values = [float(row["validation"]["per_world"][world_id]["ap"]) for row in selected]
        ap_ranges[world_id] = max(values) - min(values)
    rates = [float(row[key]) for row in selected for key in ("learning_rate_before_scheduler", "learning_rate_after_scheduler")]
    same_rate = all(value == rates[0] for value in rates)
    stable = (
        relative_range <= STABILITY_RULE["training_BCE_relative_range_max"]
        and all(value <= STABILITY_RULE["per_world_validation_AP_range_max"] for value in ap_ranges.values())
        and same_rate
    )
    return {
        "available": True, "stable": bool(stable), "start_epoch": begin, "end_epoch": end_epoch,
        "training_BCE_relative_range": relative_range, "validation_AP_range_by_world": ap_ranges,
        "learning_rate": rates[-1], "learning_rate_unchanged": same_rate,
    }


def stability_status(history: list) -> dict:
    minimum = STABILITY_RULE["minimum_epoch"]
    ends = sorted({int(row["epoch"]) for row in history if int(row["epoch"]) >= minimum and int(row["epoch"]) % EVAL_EVERY == 0})
    trigger_end = None
    trigger_rate = None
    confirmation = []
    for end in ends:
        current = stability_window(history, end)
        if not current["available"]:
            trigger_end = trigger_rate = None
            confirmation = []
            continue
        if trigger_end is not None:
            if not current["stable"] or current["learning_rate"] != trigger_rate:
                trigger_end = trigger_rate = None
                confirmation = []
            else:
                confirmation.append(current)
                if end >= trigger_end + STABILITY_RULE["confirmation_tail_epochs"]:
                    return {
                        "status": "SUSTAINED_LOCAL_STABILITY_OBSERVED",
                        "first_qualifying_pair_end_epoch": trigger_end,
                        "confirmation_end_epoch": end,
                        "confirmed_rolling_windows": confirmation,
                        "interpretation": "finite observed local stability only; not proof of a global optimum",
                    }
        if trigger_end is None and end >= minimum and current["stable"]:
            previous = stability_window(history, end - STABILITY_RULE["window_epochs"])
            if previous["available"] and previous["stable"] and previous["learning_rate"] == current["learning_rate"]:
                trigger_end = end
                trigger_rate = current["learning_rate"]
                confirmation = []
    return {
        "status": "NOT_ESTABLISHED",
        "first_qualifying_pair_end_epoch": trigger_end,
        "confirmation_end_epoch": None,
        "confirmed_rolling_windows": confirmation,
        "interpretation": "no confirmed sustained local stability under the declared finite-window rule",
    }


def stability_fixture_check() -> bool:
    history = []
    for epoch in range(EVAL_EVERY, 2201, EVAL_EVERY):
        history.append({
            "epoch": epoch,
            "training": {"equal_world_bce": 0.5},
            "validation": {"per_world": {wid: {"ap": 0.5} for wid in VAL_IDS}},
            "learning_rate_before_scheduler": 0.001,
            "learning_rate_after_scheduler": 0.001,
        })
    result = stability_status(history)
    return result["status"] == "SUSTAINED_LOCAL_STABILITY_OBSERVED" and result["confirmation_end_epoch"] == 2000


def scheduler_fixture_check(spec) -> dict:
    model = nn.Linear(2, 1)
    optimizer = optimizer_for(model, 0.03)
    scheduler = scheduler_for(optimizer, spec)
    settings = spec["scheduler"]
    checks = {
        "class": type(scheduler).__name__,
        "patience": scheduler.patience == settings["patience_evaluations"],
        "cooldown": scheduler.cooldown == 0,
        "factor": scheduler.factor == settings["factor"],
        "threshold": scheduler.threshold == settings["threshold"],
        "threshold_mode": scheduler.threshold_mode == settings["threshold_mode"],
        "mode": scheduler.mode == "min",
        "min_lr": scheduler.min_lrs == [settings["min_lr"]],
        "eps": scheduler.eps == settings["eps"],
        "infinity_sentinel_encoded": "scheduler_comparison_sentinel" in json.dumps(scheduler_state_for_json(scheduler)),
    }
    if not all(value for key, value in checks.items() if key != "class"):
        raise ValueError("declared ReduceLROnPlateau focused qualification failed")
    return checks


def state_switch_fixture_check() -> dict:
    random.seed(919)
    np.random.seed(919)
    torch.manual_seed(919)
    state = capture_rng()
    expected = (random.random(), float(np.random.random()), torch.rand(3).tolist())
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    activate_rng(state)
    observed = (random.random(), float(np.random.random()), torch.rand(3).tolist())
    if expected != observed:
        raise ValueError("per-trial RNG state switch did not restore the same draws")
    return {"python_numpy_torch_rng_restore_exact": True}


def output_accounting_fixture_check() -> dict:
    with tempfile.TemporaryDirectory(prefix="s1f-accounting-") as temporary:
        root = Path(temporary) / "run"
        root.mkdir()
        (root / "a.bin").write_bytes(b"abc")
        sub = root / "nested"
        sub.mkdir()
        (sub / "b.bin").write_bytes(b"12345")
        report = Path(temporary) / "external-report.md"
        report.write_bytes(b"xy")
        observed = output_bytes(root, extras=(report,))
        if observed != 10:
            raise ValueError(f"output accounting omitted nested or external report bytes: {observed}")
    return {"nested_run_bytes_and_external_report_counted": True, "fixture_bytes": 10}


def build_trial_order() -> list[dict]:
    summaries = [{"kind": "summary", "seed": seed, "rate": rate} for seed in SEEDS for rate in RATES]
    micros = [{"kind": "microfit", "arm": arm, "seed": seed, "rate": rate} for arm in ARMS for seed in SEEDS for rate in RATES]
    order = []
    for i, summary in enumerate(summaries):
        order.append(summary)
        order.extend(micros[i * 4:(i + 1) * 4])
    if len(order) != 75 or sum(x["kind"] == "summary" for x in order) != 15 or sum(x["kind"] == "microfit" for x in order) != 60:
        raise ValueError("fixed S1-F trial order has an invalid trial count")
    return order


def endpoint_rule_fixture_check() -> bool:
    count = 0
    for _ in range(20):
        count = micro_endpoint_update(count, 0.02, 16, 16)
    established = count == 20
    reset = micro_endpoint_update(count, 0.03, 16, 16) == 0
    return established and reset


def input_tensor_hashes(microset, summary_data, columns) -> dict:
    result = {
        "summary_train": sha_array(summary_data["x_train_numpy"]),
        "summary_validation": sha_array(summary_data["x_val_numpy"]),
        "microfit_by_arm": {},
    }
    for arm, args_by_row in summary_data["micro_args"].items():
        flat = []
        for row in args_by_row:
            flat.extend(row)
        result["microfit_by_arm"][arm] = tensor_sha(flat)
    result["custody_note"] = "Candidate/world identities are retained as joins and subset selectors only; they are absent from model tensors."
    return result


def prepare_data(index_path: Path, config_path: Path, output_root: Path, write_artifacts: bool,
                 failure_history_path: Path | None = None, reference_json_path: Path | None = None) -> dict:
    load_helpers()
    spec = strict_json(config_path)
    if spec.get("experiment_id") != "S1-F" or spec.get("scope") != "exploratory existing-world learning diagnosis only":
        raise ValueError("unexpected S1-F study configuration")
    if int(spec.get("execution_wall_limit_seconds")) != WALL_LIMIT or int(spec.get("maximum_new_output_bytes")) != OUTPUT_LIMIT:
        raise ValueError("S1-F execution ceilings differ from the assigned contract")
    if tuple(spec["seeds"]) != SEEDS or tuple(float(x) for x in spec["rates"]) != RATES:
        raise ValueError("S1-F seed/rate grid differs from the assigned contract")
    if int(spec["summary_control"]["max_epochs"]) != SUMMARY_LIMIT or int(spec["microfit"]["max_epochs"]) != MICRO_LIMIT:
        raise ValueError("S1-F trial epoch ceilings differ from the assigned contract")
    if int(spec["summary_control"]["active_parameters"]) != 8425:
        raise ValueError("S1-F summary-control parameter count differs")
    if tuple(float(x) for x in spec["optimizer"]["betas"]) != (0.9, 0.999):
        raise ValueError("S1-F Adam beta values differ")

    entries, excluded = resolve_worlds(index_path, spec)
    raw_widths = strict_json(ROOT / "source/configs/s1_matched_smoke.json")["hidden_widths"]
    if raw_widths != EXPECTED_WIDTHS:
        raise ValueError("unchanged S1-E hidden widths differ from the assigned S1-F contract")
    worlds = {}
    structural = []
    for world_id in TRAIN_IDS + VAL_IDS:
        entry = entries[world_id]
        world = TRAIN_HELPERS.strict_inputs(BENCH, world_id, entry["path"], entry["role"])
        worlds[world_id] = world
        structural.append(inspect_physical_world(world, entry, TRAIN_HELPERS))

    train_worlds = [worlds[x] for x in TRAIN_IDS]
    val_worlds = [worlds[x] for x in VAL_IDS]
    if set(x["world_id"] for x in train_worlds) & set(x["world_id"] for x in val_worlds):
        raise ValueError("training and validation world IDs overlap")
    columns = list(TRAIN_HELPERS.FEATURE_COLUMNS)
    train_raw, train_records = feature_matrix(train_worlds, columns)
    val_raw, val_records = feature_matrix(val_worlds, columns)
    all_records = train_records + val_records
    transform = fit_normalizer(train_raw)
    x_train_numpy = apply_normalizer(train_raw, transform)
    x_val_numpy = apply_normalizer(val_raw, transform)
    y_train_numpy = np.asarray([row["label"] for row in train_records], dtype=np.float32)
    y_val_numpy = np.asarray([row["label"] for row in val_records], dtype=np.float32)
    if len(train_records) != 312 or len(val_records) != 156:
        raise ValueError("S1-F requires the complete 312/156 candidate populations")

    microset = build_microset(train_worlds)
    for row in microset:
        b1, b2 = row["view"]["cell"]
        simp = row["view"]["simp"]
        if not BENCH.closed(b1, b2) or not BENCH.closed(simp[3], simp[4]):
            raise ValueError(f"exact closure failed for consumed microfit candidate {row['world_id']}/{row['candidate_id']}")
    micro_args = {
        arm: [TRAIN_HELPERS.args_for(BENCH, row["view"], arm) for row in microset]
        for arm in ARMS
    }
    selected_for_fit = []
    for row in microset:
        selected_for_fit.append({
            "world_id": row["world_id"], "candidate_id": row["candidate_id"],
            "label": row["label"], "view": row["view"],
        })
    for arm, args_by_row in micro_args.items():
        for row, args in zip(selected_for_fit, args_by_row):
            row.setdefault("args_by_arm", {})[arm] = args
    micro_args_by_arm = {
        arm: [dict(row, args=args) for row, args in zip(selected_for_fit, args_by_row)]
        for arm, args_by_row in micro_args.items()
    }

    summary_train_data = json_records_for_train(
        train_worlds,
        torch.from_numpy(x_train_numpy),
        {wid: torch.from_numpy(np.asarray([view["label"] for view in world["views"]], dtype=np.float32)) for wid, world in zip(TRAIN_IDS, train_worlds)},
    )
    summary_val_data = json_records_for_train(
        val_worlds,
        torch.from_numpy(x_val_numpy),
        {wid: torch.from_numpy(np.asarray([view["label"] for view in world["views"]], dtype=np.float32)) for wid, world in zip(VAL_IDS, val_worlds)},
    )
    x_train_tensor = torch.from_numpy(x_train_numpy)
    y_train_tensor = torch.from_numpy(y_train_numpy)
    x_val_tensor = torch.from_numpy(x_val_numpy)
    y_val_tensor = torch.from_numpy(y_val_numpy)

    transform_json = {
        "columns": columns,
        "fit_worlds": list(TRAIN_IDS),
        "fit_candidate_count": len(train_records),
        "fit_candidate_ids": [{"world_id": row["world_id"], "candidate_id": row["candidate_id"]} for row in train_records],
        "mean_float64": transform["mean"].tolist(),
        "population_sd_float64_before_constant_replacement": transform["population_sd"].tolist(),
        "scale_float64": transform["scale"].tolist(),
        "constant_columns": [columns[i] for i, value in enumerate(transform["constant_columns"]) if value],
        "training_ranges": {
            columns[i]: {"min": float(np.min(train_raw[:, i])), "max": float(np.max(train_raw[:, i]))}
            for i in range(len(columns))
        },
        "normalization_dtype": "float64",
        "model_tensor_dtype": "float32",
        "normalization_rule": "training means and population SD; scale is 1 only for exactly constant training columns",
        "train_tensor_sha256": sha_array(x_train_numpy),
        "validation_tensor_sha256": sha_array(x_val_numpy),
    }
    tensor_hashes = input_tensor_hashes(microset, {"x_train_numpy": x_train_numpy, "x_val_numpy": x_val_numpy, "micro_args": micro_args}, columns)
    bindings = [sha_file(path) for path in all_input_paths(entries, index_path, config_path, failure_history_path, reference_json_path)]
    return {
        "spec": spec,
        "entries": entries,
        "excluded_worlds": excluded,
        "worlds": worlds,
        "train_worlds": train_worlds,
        "val_worlds": val_worlds,
        "columns": columns,
        "train_records": train_records,
        "val_records": val_records,
        "all_records": all_records,
        "normalizer": transform,
        "normalizer_json": transform_json,
        "x_train_numpy": x_train_numpy,
        "x_val_numpy": x_val_numpy,
        "y_train_numpy": y_train_numpy,
        "y_val_numpy": y_val_numpy,
        "x_train_tensor": x_train_tensor,
        "y_train_tensor": y_train_tensor,
        "x_val_tensor": x_val_tensor,
        "y_val_tensor": y_val_tensor,
        "summary_train_data": summary_train_data,
        "summary_val_data": summary_val_data,
        "microset": microset,
        "micro_args": micro_args_by_arm,
        "tensor_hashes": tensor_hashes,
        "world_structure": structural,
        "fixed20_collisions": exact_vector_collisions(all_records, columns),
        "input_hashes": bindings,
        "widths": raw_widths,
    }


def qualify_models(data: dict, spec: dict) -> dict:
    fixture = BENCH.boundary_fixture_checks()
    join_tests = fixture_join_and_finite_checks()
    if not fixture["valid_cycle_fixture_closes"] or not fixture["malformed_boundary_fixture_rejected"]:
        raise ValueError("actual representation fixture qualification failed")
    if not endpoint_rule_fixture_check():
        raise ValueError("microfit endpoint count/reset fixture failed")
    if not stability_fixture_check():
        raise ValueError("S1-E finite stability-rule fixture failed")
    scheduler = scheduler_fixture_check(spec)
    state_switch = state_switch_fixture_check()
    accounting = output_accounting_fixture_check()

    torch.set_num_threads(1)
    first_micro = data["microset"][0]
    model_observations = {}
    model_basis = {}
    widths = data["widths"]
    for arm in ARMS:
        torch.manual_seed(11)
        random.seed(11)
        model = TRAIN_HELPERS.make_models(BENCH, widths)[arm]
        model_observations[arm] = micro_observer_qualification(model, data["micro_args"][arm][0]["args"])
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        try:
            torch.manual_seed(11)
            random.seed(11)
            model_basis[arm] = BENCH.permute_check(model, TRAIN_HELPERS.args_for(BENCH, first_micro["view"], arm), TRAIN_HELPERS.kind_for(arm))
        finally:
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
        if not model_basis[arm]["pass_atol_1e-6_rtol_1e-5"]:
            raise ValueError(f"pre-fit microfit coordinate check failed: {arm}")

    summary_model = make_summary_model()
    active = sum(parameter.numel() for parameter in summary_model.parameters() if parameter.requires_grad)
    if active != 8425:
        raise ValueError("summary-control active parameter count qualification failed")
    normalizer_fixture = fit_normalizer(np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float64))
    transform_fixture_ok = (
        np.array_equal(normalizer_fixture["scale"], np.asarray([1.0, 1.0]))
        and np.array_equal(normalizer_fixture["mean"], np.asarray([1.0, 3.0]))
        and np.array_equal(apply_normalizer(np.asarray([[9.0, 5.0]]), normalizer_fixture), np.asarray([[8.0, 2.0]], dtype=np.float32))
    )
    if not transform_fixture_ok:
        raise ValueError("training-only normalization fixture failed")

    model_paths = {
        "summary_control": "20 float32 normalized columns -> Linear(20,81) -> ReLU -> Linear(81,81) -> ReLU -> Linear(81,1)",
        "cellular_cwn_style_local": "node central mask; event [time, log1p amount, selected mask]; absolute B1/B2 incidence; directed endpoints; functional tanh; edge/face means; scalar readout",
        "simplicial_mpsn_style_local": "role/central/event vertex features; relation edge features; event observations occur once on event vertices; absolute B1/B2 incidence; functional tanh; mean vertex/face readout",
        "directed_local_edge_gnn": "node central mask; event [time, log1p amount, selected mask]; directed source/destination aggregation; selection-masked event mean; cycle length in scalar readout",
        "cellular_hasse_mechanism_control": "Cell forward with hasse=True uses directed endpoint-indexed incidence transforms; loaded Hasse sidecar is checked as an input but is not supplied to this forward call",
    }
    return {
        "status": "PASS",
        "scope": "focused pre-fit qualification; no optimizer steps, no benchmark main routine, no fixture fitting",
        "strict_join_and_finite_rejection": join_tests,
        "actual_boundary_fixtures": fixture,
        "normalization_fixture_pass": bool(transform_fixture_ok),
        "stability_fixture_pass": True,
        "microfit_stopping_fixture_pass": True,
        "scheduler": scheduler,
        "state_switch": state_switch,
        "output_accounting": accounting,
        "summary_active_parameters": active,
        "microfit_observation_preserves_logits_and_gradients": model_observations,
        "pre_fit_coordinate_checks": model_basis,
        "worlds_and_exact_closure": data["world_structure"],
        "native_and_fixed20_max_gap": max(x["max_abs_fixed20_native_reference_gap"] for x in data["world_structure"]),
        "training_and_validation_rows": {"training": len(data["train_records"]), "validation": len(data["val_records"])},
        "microset": [{k: row[k] for k in ("world_id", "candidate_id", "label")} for row in data["microset"]],
        "model_paths_read_from_current_helpers": model_paths,
        "source_helpers": {
            "benchmark": str(BENCH.__file__),
            "training": str(TRAIN_HELPERS.__file__),
            "fixture_fit_count": 0,
            "optimizer_updates": 0,
        },
    }


def write_checkpoint(state: dict, *, archive=False, archive_name=None) -> None:
    if state["model"] is None:
        return
    state["rng"] = capture_rng()
    trial_dir = state["trial_dir"]
    checkpoint = {
        "kind": state["kind"],
        "trial_id": state["trial_id"],
        "arm": state["arm"],
        "seed": state["seed"],
        "learning_rate": state["learning_rate"],
        "epoch": state["epoch"],
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "scheduler": state["scheduler"].state_dict(),
        "rng": state["rng"],
        "best_state": state["best_state"],
        "best_validation_bce": state["best_validation_bce"],
        "best_epoch": state["best_epoch"],
        "consecutive_endpoint_evaluations": state["consecutive_endpoint_evaluations"],
        "status": state["status"],
    }
    atomic_torch(trial_dir / "latest_state.pt", checkpoint)
    if archive:
        name = archive_name or f"epoch_{state['epoch']:04d}.pt"
        atomic_torch(trial_dir / "checkpoints" / name, checkpoint)
    write_json(trial_dir / "latest_state.json", {
        "kind": state["kind"], "trial_id": state["trial_id"], "arm": state["arm"],
        "seed": state["seed"], "learning_rate": state["learning_rate"],
        "epoch": state["epoch"], "status": state["status"],
        "scheduler_state": scheduler_state_for_json(state["scheduler"]),
        "scheduler_state_source": "actual ReduceLROnPlateau state is retained in latest_state.pt",
    })


def write_trial_initial(state: dict) -> None:
    atomic_torch(state["trial_dir"] / "initial_model.pt", {
        "model": state["model"].state_dict(),
        "model_state_sha256": state["model_state_sha256"],
        "seed": state["seed"], "learning_rate": state["learning_rate"], "arm": state["arm"],
    })


def trial_path(root: Path, kind: str, seed: int, rate: float, arm=None) -> Path:
    if kind == "summary":
        return root / "summary_control" / f"seed_{seed}" / f"lr_{rate:g}"
    return root / "microfit" / arm / f"seed_{seed}" / f"lr_{rate:g}"


def record_summary_initial(state, train_data, val_data, bench) -> None:
    train_metric = summary_eval(state["model"], train_data, bench)
    val_metric = summary_eval(state["model"], val_data, bench)
    # Epoch zero was evaluated, so include it in the minimum-BCE/earliest-tie
    # checkpoint rule without treating it as a post-update evaluation.
    state["best_validation_bce"] = float(val_metric["equal_world_bce"])
    state["best_epoch"] = 0
    state["best_state"] = copy_state(state["model"].state_dict())
    state["best_checkpoint_path"] = str(state["trial_dir"] / "initial_model.pt")
    write_json(state["trial_dir"] / "selection.json", {
        "selected_checkpoint": "initial_model.pt", "epoch": 0,
        "equal_world_validation_bce": state["best_validation_bce"],
        "selection_rule": "lowest equal-world validation BCE; ties earliest evaluated epoch, including initialization",
    })
    write_json(state["trial_dir"] / "initialization.json", {
        "epoch": 0, "not_a_post_update_checkpoint": True,
        "training_equal_world_bce": train_metric["equal_world_bce"],
        "validation_equal_world_bce": val_metric["equal_world_bce"],
        "validation_per_world": {wid: {k: v[k] for k in ("bce", "ap", "auroc")} for wid, v in val_metric["per_world"].items()},
        "model_state_sha256": state["model_state_sha256"],
    })


def record_micro_initial(state, selected, arm) -> None:
    metrics = micro_eval(state["model"], selected, arm)
    write_json(state["trial_dir"] / "initialization.json", {
        "epoch": 0, "not_a_post_update_endpoint_evaluation": True,
        "training_equal_world_bce": metrics["equal_world_bce"],
        "correct_at_logit_zero": metrics["correct_at_logit_zero"],
        "total": metrics["total"], "predictions": metrics["predictions"],
        "model_state_sha256": state["model_state_sha256"],
    })


def summary_trace_record(state, pre_update_bce, train_metric, val_metric, lr_before, lr_after, gradient_norms) -> dict:
    return {
        "epoch": int(state["epoch"]),
        "training_loss_pre_update_equal_world_bce": float(pre_update_bce),
        "training": {"equal_world_bce": train_metric["equal_world_bce"], "per_world": {wid: {"bce": x["bce"]} for wid, x in train_metric["per_world"].items()}},
        "validation": {"equal_world_bce": val_metric["equal_world_bce"], "per_world": {wid: {"bce": x["bce"], "ap": x["ap"], "auroc": x["auroc"]} for wid, x in val_metric["per_world"].items()}},
        "learning_rate_before_scheduler": float(lr_before),
        "learning_rate_after_scheduler": float(lr_after),
        "gradient_norm_by_layer": gradient_norms,
    }


def append_summary_eval(state, pre_update_bce, train_data, val_data, bench) -> None:
    train_metric = summary_eval(state["model"], train_data, bench)
    val_metric = summary_eval(state["model"], val_data, bench)
    if not math.isfinite(train_metric["equal_world_bce"]) or not math.isfinite(val_metric["equal_world_bce"]):
        raise NonfiniteTrial("nonfinite summary evaluation BCE")
    lr_before = float(state["optimizer"].param_groups[0]["lr"])
    state["scheduler"].step(train_metric["equal_world_bce"])
    lr_after = float(state["optimizer"].param_groups[0]["lr"])
    gradients = gradient_norms_by_layer(state["model"])
    record = summary_trace_record(state, pre_update_bce, train_metric, val_metric, lr_before, lr_after, gradients)
    state["history"].append(record)
    append_jsonl(state["trial_dir"] / "trace.jsonl", record)
    val_bce = float(val_metric["equal_world_bce"])
    if state["best_validation_bce"] is None or val_bce < state["best_validation_bce"]:
        state["best_validation_bce"] = val_bce
        state["best_epoch"] = int(state["epoch"])
        state["best_state"] = copy_state(state["model"].state_dict())
        selected_path = state["trial_dir"] / "selected_checkpoints" / f"epoch_{state['best_epoch']:04d}.pt"
        state["best_checkpoint_path"] = str(selected_path)
        atomic_torch(selected_path, {
            "epoch": state["best_epoch"], "model": state["best_state"],
            "arm": "fixed20_tabular_neural_control", "seed": state["seed"],
            "learning_rate": state["learning_rate"],
            "equal_world_validation_bce": val_bce,
        })
        write_json(state["trial_dir"] / "selection.json", {
            "selected_checkpoint": str(selected_path.relative_to(state["trial_dir"])),
            "epoch": state["best_epoch"], "equal_world_validation_bce": val_bce,
            "selection_rule": "lowest equal-world validation BCE; ties earliest epoch",
        })
    write_checkpoint(state, archive=state["epoch"] % BLOCK_EPOCHS == 0)
    state["last_eval_logits"] = val_metric["per_world"]


def append_micro_eval(state, selected, arm, pre_update_bce) -> None:
    metrics = micro_eval(state["model"], selected, arm)
    if not math.isfinite(metrics["equal_world_bce"]):
        raise NonfiniteTrial("nonfinite microfit evaluation BCE")
    state["consecutive_endpoint_evaluations"] = micro_endpoint_update(
        state["consecutive_endpoint_evaluations"], metrics["equal_world_bce"],
        metrics["correct_at_logit_zero"], metrics["total"],
    )
    lr_before = float(state["optimizer"].param_groups[0]["lr"])
    state["scheduler"].step(metrics["equal_world_bce"])
    lr_after = float(state["optimizer"].param_groups[0]["lr"])
    gradients = gradient_norms_by_layer(state["model"])
    record = {
        "epoch": int(state["epoch"]),
        "training_loss_pre_update_equal_world_bce": float(pre_update_bce),
        "post_update_equal_world_bce": metrics["equal_world_bce"],
        "post_update_correct_at_logit_zero": metrics["correct_at_logit_zero"],
        "total": metrics["total"],
        "consecutive_qualifying_post_update_evaluations": state["consecutive_endpoint_evaluations"],
        "learning_rate_before_scheduler": lr_before,
        "learning_rate_after_scheduler": lr_after,
        "gradient_norm_by_layer": gradients,
    }
    state["history"].append(record)
    append_jsonl(state["trial_dir"] / "trace.jsonl", record)
    state["last_eval_logits"] = metrics["predictions"]
    state["last_gradient_norms"] = gradients
    if state["consecutive_endpoint_evaluations"] >= 20:
        state["status"] = "endpoint_established"
        state["endpoint_reason"] = "BCE<=0.02 and all 16 correct at logit zero for 20 consecutive post-update evaluations"
    write_checkpoint(
        state,
        archive=state["status"] == "endpoint_established" or state["epoch"] % BLOCK_EPOCHS == 0,
        archive_name=(f"endpoint_epoch_{state['epoch']:04d}.pt" if state["status"] == "endpoint_established" else None),
    )


def budget_guard(root: Path, report_path: Path | None, check_output=True) -> None:
    elapsed = elapsed_seconds()
    if elapsed >= WALL_LIMIT - WALL_RESERVE:
        raise BudgetStop(f"wall limit reserve reached at {elapsed:.1f}s; {WALL_RESERVE}s reserved for final evidence and reporting")
    if check_output:
        size = output_bytes(root, extras=(report_path,))
        if size >= OUTPUT_LIMIT - OUTPUT_RESERVE:
            raise BudgetStop(f"output limit reserve reached at {size} bytes; {OUTPUT_RESERVE} bytes reserved for final evidence and reporting")


def finalization_guard(root: Path) -> None:
    # Keep a small final-delivery margin inside the gross limits, after the
    # training reserve has already stopped updates.
    if elapsed_seconds() >= WALL_LIMIT - 30:
        raise BudgetStop("gross wall limit delivery margin reached during finalization")
    if output_bytes(root) >= OUTPUT_LIMIT - (4 * 1024 * 1024):
        raise BudgetStop("gross output limit delivery margin reached during finalization")


def save_status(state, status, reason=None) -> None:
    state["status"] = status
    if reason:
        state["endpoint_reason"] = reason
    state["rng"] = capture_rng()
    write_checkpoint(state, archive=True, archive_name=f"{status}_epoch_{state['epoch']:04d}.pt")


def init_all_trials(root: Path, spec: dict, data: dict, order: list[dict]) -> tuple[list[dict], dict]:
    states = []
    summary_hashes = {}
    micro_hashes = {}
    micro_by_arm = data["micro_args"]
    for item in order:
        kind, seed, rate = item["kind"], int(item["seed"]), float(item["rate"])
        arm = item.get("arm")
        trial_id = (f"summary/seed_{seed}/lr_{rate:g}" if kind == "summary" else f"{arm}/seed_{seed}/lr_{rate:g}")
        directory = trial_path(root, kind, seed, rate, arm)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite prior trial evidence: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        state = create_trial_state(kind, trial_id, seed, rate, arm, data["widths"], spec, directory, {
            "initial_state_hashes": summary_hashes if kind == "summary" else micro_hashes,
            "train_worlds": data["train_worlds"],
            "x_train_tensor": data["x_train_tensor"], "y_train_tensor": data["y_train_tensor"],
            "x_val_tensor": data["x_val_tensor"], "y_val_tensor": data["y_val_tensor"],
            "micro_args": micro_by_arm,
        })
        if kind == "summary":
            record_summary_initial(state, data["summary_train_data"], data["summary_val_data"], BENCH)
        else:
            selected = micro_by_arm[arm]
            record_micro_initial(state, selected, arm)
        state["started_monotonic"] = None
        write_trial_initial(state)
        write_checkpoint(state, archive=True, archive_name="initial_epoch_0000.pt")
        states.append(state)
    summary_init_check = check_same_initializations(summary_hashes, "summary")
    micro_init_check = check_same_initializations(micro_hashes, "microfit")
    return states, {"summary": summary_init_check, "microfit": micro_init_check}


def trial_qualifying_rows(state, data):
    if state["kind"] == "summary":
        return None
    return data["micro_args"][state["arm"]]


def mark_unstarted(states):
    for state in states:
        if state["epoch"] == 0 and state["status"] == "not_started":
            state["status"] = "not_started"


def run_round_robin(root: Path, states: list[dict], data: dict, spec: dict, report_path: Path | None) -> str | None:
    reason = None
    max_blocks = max(SUMMARY_LIMIT, MICRO_LIMIT) // BLOCK_EPOCHS
    for block_index in range(max_blocks):
        block_start = block_index * BLOCK_EPOCHS
        active_in_block = False
        for state in states:
            cap = SUMMARY_LIMIT if state["kind"] == "summary" else MICRO_LIMIT
            if state["status"] in ("failed", "endpoint_established", "cap_not_established") or state["epoch"] >= cap:
                continue
            if state["epoch"] >= min(cap, block_start + BLOCK_EPOCHS):
                continue
            active_in_block = True
            activate_rng(state["rng"])
            if state["started_monotonic"] is None:
                state["started_monotonic"] = time.monotonic()
                state["status"] = "running"
            target = min(cap, block_start + BLOCK_EPOCHS)
            while state["epoch"] < target:
                budget_guard(root, report_path, check_output=False)
                try:
                    if state["kind"] == "summary":
                        pre_update = summary_train_epoch(state, data["summary_train_data"])
                    else:
                        by_world = {
                            wid: [row for row in data["micro_args"][state["arm"]] if row["world_id"] == wid]
                            for wid in TRAIN_IDS
                        }
                        pre_update = micro_train_epoch(state, by_world)
                except NonfiniteTrial as error:
                    state["rng"] = capture_rng()
                    state["duration_seconds"] += time.monotonic() - state["started_monotonic"]
                    state["started_monotonic"] = None
                    save_status(state, "failed", str(error))
                    append_jsonl(root / "trial_failures.jsonl", {
                        "trial_id": state["trial_id"], "kind": state["kind"], "arm": state["arm"],
                        "seed": state["seed"], "learning_rate": state["learning_rate"],
                        "epoch": state["epoch"], "error_type": type(error).__name__, "error": str(error),
                    })
                    break
                state["epoch"] += 1
                append_jsonl(state["trial_dir"] / "epoch_losses.jsonl", {
                    "epoch": state["epoch"],
                    "training_loss_pre_update_equal_world_bce": float(pre_update),
                })
                if state["epoch"] % EVAL_EVERY == 0:
                    budget_guard(root, report_path, check_output=True)
                    if state["kind"] == "summary":
                        append_summary_eval(state, pre_update, data["summary_train_data"], data["summary_val_data"], BENCH)
                    else:
                        append_micro_eval(state, data["micro_args"][state["arm"]], state["arm"], pre_update)
                    if state["status"] == "endpoint_established":
                        state["duration_seconds"] += time.monotonic() - state["started_monotonic"]
                        state["started_monotonic"] = None
                        state["rng"] = capture_rng()
                        break
            else:
                # This path runs at the end of a 100-epoch block or a declared cap.
                pass
            state["rng"] = capture_rng()
            if state["started_monotonic"] is not None:
                state["duration_seconds"] += time.monotonic() - state["started_monotonic"]
                state["started_monotonic"] = None
            if state["kind"] == "summary" and state["epoch"] >= SUMMARY_LIMIT:
                state["status"] = "completed"
                state["endpoint_reason"] = "3000-epoch cap reached; no early stopping rule applied"
            elif state["kind"] == "microfit" and state["epoch"] >= MICRO_LIMIT and state["status"] != "endpoint_established":
                state["status"] = "cap_not_established"
                state["endpoint_reason"] = "1500-epoch cap reached without the declared 20-evaluation endpoint"
            if state["status"] in ("completed", "cap_not_established"):
                write_checkpoint(state, archive=True, archive_name=f"{state['status']}_epoch_{state['epoch']:04d}.pt")
            elif state["status"] in ("endpoint_established", "failed"):
                write_checkpoint(state)
            else:
                state["status"] = "in_progress"
                write_checkpoint(state)
        if not active_in_block:
            break
    return reason


def run_grid(root, states, data, spec, report_path) -> tuple[str | None, str | None]:
    unexpected = None
    planned_stop = None
    try:
        run_round_robin(root, states, data, spec, report_path)
    except BudgetStop as error:
        planned_stop = str(error)
        for state in states:
            if state["status"] == "running":
                state["rng"] = capture_rng()
                if state["started_monotonic"] is not None:
                    state["duration_seconds"] += time.monotonic() - state["started_monotonic"]
                    state["started_monotonic"] = None
                save_status(state, "interrupted", planned_stop)
        mark_unstarted(states)
    except Exception as error:
        unexpected = f"{type(error).__name__}: {error}"
        trace = traceback.format_exc()
        append_jsonl(root / "script_failures.jsonl", {
            "utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__, "error": str(error), "traceback": trace,
        })
        for state in states:
            if state["status"] == "running":
                state["rng"] = capture_rng()
                if state["started_monotonic"] is not None:
                    state["duration_seconds"] += time.monotonic() - state["started_monotonic"]
                    state["started_monotonic"] = None
                save_status(state, "interrupted", unexpected)
    return planned_stop, unexpected


def reference_metrics(labels, logits) -> dict:
    y = [int(x) for x in labels]
    scores = [float(x) for x in logits]
    if len(y) != len(scores) or not y or any(x not in (0, 1) for x in y) or any(not math.isfinite(x) for x in scores):
        raise ValueError("invalid joined rows for independent metric recomputation")
    positives = sum(y)
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("independent AUROC/AP require both observed classes")
    ordered = sorted(zip(scores, y), key=lambda pair: -pair[0])
    tp = fp = 0
    previous_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        group_positive = sum(label for _, label in ordered[i:j])
        group_size = j - i
        tp += group_positive
        fp += group_size - group_positive
        recall = tp / positives
        ap += (recall - previous_recall) * (tp / (tp + fp))
        previous_recall = recall
        i = j
    pair_wins = 0.0
    positive_scores = [score for score, label in zip(scores, y) if label == 1]
    negative_scores = [score for score, label in zip(scores, y) if label == 0]
    for pos in positive_scores:
        for neg in negative_scores:
            pair_wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    auc = pair_wins / (positives * negatives)
    losses = []
    for label, score in zip(y, scores):
        softplus = score + math.log1p(math.exp(-score)) if score > 0 else math.log1p(math.exp(score))
        losses.append(softplus - label * score)
    return {"bce": math.fsum(losses) / len(losses), "ap": ap, "auroc": auc, "n": len(y), "positives": positives}


def reload_summary_checkpoints(root: Path, states: list[dict], data: dict) -> dict:
    all_summary = [x for x in states if x["kind"] == "summary"]
    complete = [x for x in all_summary if x["status"] == "completed" and x["best_state"] is not None]
    grouped_by_rate = {rate: [x for x in complete if x["learning_rate"] == rate] for rate in RATES}
    full_grid = all(len(grouped_by_rate[rate]) == len(SEEDS) for rate in RATES)
    if full_grid:
        mean_bce = {
            str(rate): float(np.mean([x["best_validation_bce"] for x in grouped_by_rate[rate]]))
            for rate in RATES
        }
        chosen_rate = min(RATES, key=lambda rate: (mean_bce[str(rate)], rate))
    else:
        mean_bce = {str(rate): None for rate in RATES}
        chosen_rate = None

    checked_rows = []
    score_checks = []
    if chosen_rate is not None:
        state_by_seed = {x["seed"]: x for x in complete if x["learning_rate"] == chosen_rate}
        for seed in SEEDS:
            finalization_guard(root)
            state = state_by_seed[seed]
            model = make_summary_model()
            selected_checkpoint = Path(state["best_checkpoint_path"])
            selected = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(selected["model"])
            model.eval()
            by_world = {}
            with torch.no_grad():
                for world_id in VAL_IDS:
                    val = data["summary_val_data"][world_id]
                    logits = model(val["x"]).reshape(-1).cpu().numpy().astype(np.float64).tolist()
                    labels = val["y"].cpu().numpy().astype(np.int64).tolist()
                    metrics = reference_metrics(labels, logits)
                    helper_metrics = TRAIN_HELPERS.metrics(BENCH, labels, logits)
                    differences = {
                        "bce": abs(metrics["bce"] - float(helper_metrics["bce"])),
                        "ap": abs(metrics["ap"] - float(helper_metrics["average_precision_grouped_threshold"])),
                        "auroc": abs(metrics["auroc"] - float(helper_metrics["auroc_tie_aware"])),
                    }
                    if max(differences.values()) > 1e-12:
                        raise ValueError(f"independent joined metric recomputation mismatch: {seed}/{world_id}")
                    by_world[world_id] = {**metrics, "metric_difference_vs_training_helper": differences}
                    for record, logit in zip([x for x in data["val_records"] if x["world_id"] == world_id], logits):
                        checked_rows.append({
                            "seed": seed, "selected_rate": chosen_rate,
                            "world_id": world_id, "candidate_id": record["candidate_id"],
                            "label": record["label"], "logit": logit,
                        })
            score_checks.append({"seed": seed, "selected_rate": chosen_rate, "per_world": by_world})
        if len(checked_rows) != 5 * 156:
            raise ValueError("selected summary reload did not retain all 156 validation logits per seed")
        with (root / "summary_control" / "selected_validation_logits.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seed", "selected_rate", "world_id", "candidate_id", "label", "logit"])
            writer.writeheader()
            writer.writerows(checked_rows)

    return {
        "status": "COMPLETE_GRID" if full_grid else "INCOMPLETE_GRID_NO_SELECTED_RATE",
        "all_15_trials_complete": full_grid,
        "mean_best_equal_world_validation_bce_by_rate": mean_bce,
        "selected_rate": chosen_rate,
        "tie_rule": "lower learning rate",
        "checkpoint_tie_rule": "earliest epoch; exact equal BCE does not replace an earlier checkpoint",
        "selected_checkpoint_reload_and_independent_metrics": score_checks,
        "retained_validation_logit_rows": len(checked_rows),
        "prediction_join_scope": "156 indexed validation rows per seed; IDs rejoined from the same strict source label/identity join",
    }


def micro_endpoint_diagnostics(root: Path, states: list[dict], data: dict) -> dict:
    results = []
    for state in states:
        if state["kind"] != "microfit":
            continue
        finalization_guard(root)
        if state["epoch"] == 0 or state["model"] is None:
            results.append({"trial_id": state["trial_id"], "status": state["status"], "endpoint_activation_diagnostics": None, "coordinate_checks": []})
            continue
        arm = state["arm"]
        selected = data["micro_args"][arm]
        try:
            expected = micro_eval(state["model"], selected, arm)
        except NonfiniteTrial as error:
            result = {"trial_id": state["trial_id"], "status": state["status"], "epoch": state["epoch"],
                      "endpoint_activation_diagnostics": None, "coordinate_checks": [],
                      "endpoint_unavailable_reason": str(error)}
            write_json(state["trial_dir"] / "endpoint_diagnostics.json", result)
            results.append(result)
            continue
        observed = micro_eval(state["model"], selected, arm, capture=True)
        same_logits = all(a["logit"] == b["logit"] for a, b in zip(expected["predictions"], observed["predictions"]))
        if not same_logits:
            raise ValueError(f"endpoint observation changed logits: {state['trial_id']}")
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        checks = []
        try:
            for index, row in enumerate(data["microset"]):
                random.seed(state["seed"] + index)
                torch.manual_seed(state["seed"] + index)
                check = BENCH.permute_check(
                    state["model"],
                    TRAIN_HELPERS.args_for(BENCH, row["view"], arm),
                    TRAIN_HELPERS.kind_for(arm),
                )
                checks.append({"world_id": row["world_id"], "candidate_id": row["candidate_id"], **check})
        finally:
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
        result = {
            "trial_id": state["trial_id"], "status": state["status"],
            "epoch": state["epoch"], "post_update_equal_world_bce": expected["equal_world_bce"],
            "correct_at_logit_zero": expected["correct_at_logit_zero"],
            "all_16_observed_logits_bitwise_equal_to_uninstrumented": same_logits,
            "gradient_norm_by_layer_at_last_evaluation": state["last_gradient_norms"],
            "activation_ranges_and_saturation": observed["activation_observations"],
            "coordinate_basis_checks": checks,
            "coordinate_checks_passed": sum(bool(x["pass_atol_1e-6_rtol_1e-5"]) for x in checks),
            "coordinate_checks_total": len(checks),
            "coordinate_check_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        }
        write_json(state["trial_dir"] / "endpoint_diagnostics.json", result)
        results.append(result)
    return {
        "trials_with_model_updates": sum(x.get("epoch", 0) > 0 for x in results),
        "endpoint_or_terminal_diagnostics": results,
        "failed_coordinate_checks": [
            {"trial_id": trial["trial_id"], "world_id": check["world_id"], "candidate_id": check["candidate_id"], "difference": check["abs_difference"]}
            for trial in results for check in trial.get("coordinate_basis_checks", [])
            if not check["pass_atol_1e-6_rtol_1e-5"]
        ],
    }


def load_descriptive_references(reference_json_path: Path | None) -> dict:
    """Load an optional, explicitly supplied JSON reference without project-private defaults."""
    if reference_json_path is None:
        return {"status": "NOT_PROVIDED", "values": None}
    path = Path(reference_json_path).resolve(strict=True)
    return {
        "status": "PROVIDED",
        "source": sha_file(path),
        "values": strict_json(path),
    }


def summary_selection_metrics(states) -> dict:
    completed = [x for x in states if x["kind"] == "summary" and x["status"] == "completed"]
    complete_grid = len(completed) == 15
    means = {}
    for rate in RATES:
        rows = [x for x in completed if x["learning_rate"] == rate]
        means[str(rate)] = float(np.mean([x["best_validation_bce"] for x in rows])) if len(rows) == 5 else None
    chosen = min(RATES, key=lambda rate: (means[str(rate)], rate)) if complete_grid else None
    return {"grid_complete": complete_grid, "mean_best_validation_bce_by_rate": means, "selected_rate": chosen}


def summarize_trials(states) -> list[dict]:
    rows = []
    for state in states:
        rows.append({
            "kind": state["kind"], "trial_id": state["trial_id"], "arm": state["arm"],
            "seed": state["seed"], "learning_rate": state["learning_rate"],
            "status": state["status"], "epochs_completed": state["epoch"],
            "endpoint_reason": state["endpoint_reason"],
            "best_validation_bce": state["best_validation_bce"],
            "best_epoch": state["best_epoch"],
            "consecutive_endpoint_evaluations": state["consecutive_endpoint_evaluations"],
            "duration_seconds": state["duration_seconds"],
            "initial_state_sha256": state.get("model_state_sha256"),
        })
    return rows


def save_trial_statuses(root, states) -> None:
    rows = summarize_trials(states)
    write_json(root / "trial_statuses.json", {
        "trials": rows,
        "counts_by_kind_and_status": {
            kind: {status: sum(x["kind"] == kind and x["status"] == status for x in rows) for status in sorted({x["status"] for x in rows})}
            for kind in ("summary", "microfit")
        },
    })
    with (root / "trial_statuses.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(root, states, selection, diagnostics) -> list[str]:
    finalization_guard(root)
    paths = []
    summary_states = [x for x in states if x["kind"] == "summary"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0.001: "#3366aa", 0.01: "#dd8c2f", 0.03: "#3a8f64"}
    for state in summary_states:
        values = state["history"]
        if not values:
            continue
        epochs = [x["epoch"] for x in values]
        bce = [x["validation"]["equal_world_bce"] for x in values]
        ax.plot(epochs, bce, color=colors[state["learning_rate"]], alpha=0.30, linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Equal-world validation BCE")
    ax.set_title("Fixed20 tabular neural control: all retained rate/seed trajectories")
    ax.text(0.01, 0.01, "Two exposed validation worlds; curves are descriptive.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    path = root / "summary_learning_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = selection["mean_best_validation_bce_by_rate"]
    labels = [f"{rate:g}" for rate in RATES]
    values = [means[str(rate)] for rate in RATES]
    available = [(i, value, rate) for i, (value, rate) in enumerate(zip(values, RATES)) if value is not None]
    ax.bar([x[0] for x in available], [x[1] for x in available], color=[colors[x[2]] for x in available])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Mean best equal-world validation BCE")
    ax.set_title("S1-F summary-control rate selection")
    if not selection["grid_complete"]:
        ax.text(0.5, 0.92, "Incomplete grid: no rate selected", ha="center", transform=ax.transAxes)
        missing = [label for label, value in zip(labels, values) if value is None]
        ax.text(0.5, 0.82, "Unavailable rate means: " + ", ".join(missing), ha="center", transform=ax.transAxes)
    fig.tight_layout()
    path = root / "summary_rate_selection.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    rate_labels = {0.001: "0.001", 0.01: "0.01", 0.03: "0.03"}
    for axis, world_id in zip(axes, VAL_IDS):
        for position, rate in enumerate(RATES):
            offsets = np.linspace(-0.12, 0.12, len(SEEDS))
            for seed, offset in zip(SEEDS, offsets):
                state = next(x for x in summary_states if x["seed"] == seed and x["learning_rate"] == rate)
                chosen = next((x for x in state["history"] if x["epoch"] == state["best_epoch"]), None)
                if chosen is not None:
                    axis.scatter(position + offset, chosen["validation"]["per_world"][world_id]["ap"], color=colors[rate], s=20)
                elif state["best_epoch"] == 0:
                    initial = strict_json(state["trial_dir"] / "initialization.json")
                    axis.scatter(position + offset, initial["validation_per_world"][world_id]["ap"], color=colors[rate], s=20)
        axis.set_xticks(range(len(RATES)), [rate_labels[x] for x in RATES])
        axis.set_title(world_id)
        axis.set_xlabel("Learning rate")
    axes[0].set_ylabel("Validation average precision at the BCE-selected checkpoint")
    fig.suptitle("Summary-control development validation scores; seeds are not independent worlds")
    fig.tight_layout()
    path = root / "summary_validation_scores.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))

    micro = [x for x in states if x["kind"] == "microfit"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for state in micro:
        points = state["history"]
        if points:
            ax.plot([x["epoch"] for x in points], [x["post_update_equal_world_bce"] for x in points], alpha=0.16, linewidth=0.65)
    ax.axhline(0.02, color="#a33", linestyle="--", linewidth=1, label="declared BCE endpoint")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training microset equal-world BCE")
    ax.set_title("Unchanged-path microfit trajectories; no validation used")
    ax.legend()
    fig.tight_layout()
    path = root / "microfit_learning_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))

    statuses = Counter(x["status"] for x in micro)
    fig, ax = plt.subplots(figsize=(7, 4))
    status_labels = ["endpoint_established", "cap_not_established", "interrupted", "failed", "not_started"]
    ax.bar(status_labels, [statuses[x] for x in status_labels], color="#587b91")
    ax.set_ylabel("Trial count")
    ax.set_title("Microfit terminal states (all 60 arms/seeds/rates retained)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = root / "microfit_terminal_states.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))
    return paths


def finish_run(root, states, data, initial_checks, qualification, start_wall, run_manifest, planned_stop, unexpected, report_path=None) -> dict:
    finalization_guard(root)
    reload = reload_summary_checkpoints(root, states, data)
    summary = summary_selection_metrics(states)
    # The independent reload path is invoked only when the full 15-trial grid exists.
    if reload["status"] == "INCOMPLETE_GRID_NO_SELECTED_RATE" and summary["grid_complete"]:
        raise ValueError("summary grid selection and reload completeness disagree")
    micro_diagnostics = micro_endpoint_diagnostics(root, states, data)
    save_trial_statuses(root, states)
    stability = {
        state["trial_id"]: stability_status(state["history"])
        for state in states if state["kind"] == "summary"
    }
    write_json(root / "summary_stability_diagnostic.json", {
        "rule": STABILITY_RULE,
        "diagnostic_only": True,
        "extra_epochs_run": False,
        "trials": stability,
    })
    selected_rate_metrics = reload
    if selected_rate_metrics["selected_rate"] is not None:
        per_world = defaultdict(list)
        for seed in selected_rate_metrics["selected_checkpoint_reload_and_independent_metrics"]:
            for world_id, values in seed["per_world"].items():
                per_world[world_id].append({"seed": seed["seed"], **values})
        write_json(root / "summary_control/independently_recomputed_validation_metrics.json", dict(per_world))

    reference_path = run_manifest.get("reference_json_path")
    references = load_descriptive_references(Path(reference_path) if reference_path else None)
    figures = make_figures(root, states, summary, micro_diagnostics)
    trial_rows = summarize_trials(states)
    completed_summary = [x for x in trial_rows if x["kind"] == "summary" and x["status"] == "completed"]
    micro_endpoint_n = sum(x["kind"] == "microfit" and x["status"] == "endpoint_established" for x in trial_rows)
    updated_micro = sum(x["kind"] == "microfit" and x["epochs_completed"] > 0 for x in trial_rows)
    counts = {
        kind: {status: sum(x["kind"] == kind and x["status"] == status for x in trial_rows) for status in sorted({x["status"] for x in trial_rows})}
        for kind in ("summary", "microfit")
    }
    status = "FAILED" if unexpected else "COMPLETED" if len(completed_summary) == 15 and counts["microfit"].get("not_started", 0) == 0 and counts["microfit"].get("interrupted", 0) == 0 else "PARTIAL"
    result = {
        "experiment_id": "S1-F",
        "status": status,
        "scope": "bounded exploratory learning diagnostics on the existing four training/two validation worlds",
        "empirical_start_utc": start_wall,
        "script_process_start_utc": PROCESS_START_UTC,
        "elapsed_seconds": elapsed_seconds(),
        "limits": {"wall_seconds": WALL_LIMIT, "new_output_bytes": OUTPUT_LIMIT, "automatic_extension": False},
        "output_bytes_at_report_write": output_bytes(root, extras=(report_path,)),
        "resources": measured_resources(),
        "command": run_manifest["command"],
        "cwd": run_manifest["cwd"],
        "environment": run_manifest["environment"],
        "input_hashes": run_manifest["input_hashes"],
        "source_hash": run_manifest["source_hash"],
        "config_hash": run_manifest["config_hash"],
        "world_index_hash": run_manifest["world_index_hash"],
        "preserved_prior_failure_log": run_manifest["preserved_prior_failure_log"],
        "resolved_worlds": run_manifest["resolved_worlds"],
        "excluded_worlds": run_manifest["excluded_worlds"],
        "trial_order_path": "trial_order.json",
        "initialization_checks": initial_checks,
        "summary_control": {
            "trial_count": 15, "completed_trials": len(completed_summary),
            "all_trials": [x for x in trial_rows if x["kind"] == "summary"],
            "selection": reload,
            "stability": "see summary_stability_diagnostic.json; finite S1-E rule is diagnostic only",
            "selected_validation_logits_csv": "summary_control/selected_validation_logits.csv" if reload["retained_validation_logit_rows"] else None,
        },
        "microfit": {
            "trial_count": 60, "trials_with_updates": updated_micro,
            "endpoint_established_trials": micro_endpoint_n,
            "all_trials": [x for x in trial_rows if x["kind"] == "microfit"],
            "subset_path": "microfit/subset_identities.json",
            "endpoint_diagnostics_path": "microfit_endpoint_diagnostics.json",
            "no_validation_used_during_microfit": True,
        },
        "trial_counts_by_kind_and_status": counts,
        "fixed20": {
            "columns": data["columns"], "rows": {"training": len(data["train_records"]), "validation": len(data["val_records"])},
            "normalizer_path": "summary_control/fitted_transform.json",
            "feature_collision_path": "summary_feature_collisions.json",
            "input_tensor_hashes_path": "input_tensor_hashes.json",
            "max_native_reference_gap": max(x["max_abs_fixed20_native_reference_gap"] for x in data["world_structure"]),
        },
        "physical_identity_and_closure_path": "world_structure_checks.json",
        "qualification_path": "qualification.json",
        "qualification_status": qualification.get("status"),
        "descriptive_references": references,
        "figures": figures,
        "planned_stop_reason": planned_stop,
        "unexpected_script_error": unexpected,
        "claim_boundaries": [
            "A tabular neural control is not a topology model.",
            "The 16-row microset is deliberately label-conditioned training capability evidence, not an efficacy or validation cohort.",
            "Two exposed validation worlds do not support a population interval or superiority claim.",
            "Exact feature-vector collisions do not establish representation adequacy when absent.",
            "Microfit success does not establish generalization; microfit failure does not prove impossibility.",
            "This development diagnostic does not establish independent-world performance, a causal/topology mechanism, or real-world AML efficacy.",
            "The existing GUDHI limitation remains unresolved.",
        ],
    }
    write_json(root / "microfit_endpoint_diagnostics.json", micro_diagnostics)
    write_json(root / "summary_selection.json", reload)
    endpoint = [
        "# S1-F bounded learning diagnostic endpoint", "",
        f"Status: **{result['status']}**", "",
        f"Elapsed seconds: {result['elapsed_seconds']:.1f}; output bytes before the final hash/report manifests: {output_bytes(root, extras=(report_path,))}.", "",
        f"Summary trials completed: {len(completed_summary)}/15. Selected rate: {reload['selected_rate'] if reload['selected_rate'] is not None else 'not selected (grid incomplete)' }.", "",
        f"Microfit trials with updates: {updated_micro}/60; declared endpoint established: {micro_endpoint_n}.", "",
        "See `final_report.json`, `trial_statuses.csv`, all per-trial traces/checkpoints, and the figures listed there.", "",
        "This is exploratory evidence on two exposed validation worlds and a label-conditioned training subset. It is not a comparative qualification, generalization result, or causal/topology-mechanism finding.", "",
    ]
    (root / "endpoint_report.md").write_text("\n".join(endpoint), encoding="utf-8")
    write_json(root / "final_report.json", result)
    output_files = [
        path for path in root.rglob("*")
        if path.is_file() and path.name not in ("output_hashes.json", "final_report.json", "stdout.log", "stderr.log")
    ]
    write_json(root / "output_hashes.json", [sha_file(path) for path in sorted(output_files)])
    result["output_bytes_after_hash_manifest"] = output_bytes(root, extras=(report_path,))
    result["output_hash_scope"] = "closed artifacts; launch logs and actual exit are hashed by the caller after process exit"
    write_json(root / "final_report.json", result)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = [
            "# Contract Report — S1-F", "", "## Verification Summary", "",
            f"Status: {status}. This is bounded exploratory development evidence.",
            f"Summary trials completed: {len(completed_summary)}/15; selected rate: {reload['selected_rate']}.",
            f"Microfit trials with updates: {updated_micro}/60; endpoint established: {micro_endpoint_n}.",
            "", "## Definition of Done", "",
            "Trial records distinguish every registered setting, including unstarted, interrupted and failed trials, under the declared limits.",
            "All trial statuses, failures, source/input hashes, normalizer, subset identities, states, predictions, diagnostics and figures are linked in the retained endpoint artifacts.",
            "", "## Final Status", "",
            f"Planned stop: {planned_stop}; unexpected script error: {unexpected}.",
            "No population, superiority, causal/topology-mechanism, convergence-proof or real-world AML claim follows.",
            "", "## Evidence", "",
            f"Output directory: `{root}`.",
            f"Exact launch and input hashes: `{root / 'run_manifest.json'}`.",
            f"Actual findings, joined metrics, limitations and figure paths: `{root / 'final_report.json'}`.",
            f"Trial accounting: `{root / 'trial_statuses.csv'}`.",
            f"Finite stability: `{root / 'summary_stability_diagnostic.json'}`.",
            f"Microfit endpoint observations: `{root / 'microfit_endpoint_diagnostics.json'}`.",
            "Caller retains actual exit, stdout/stderr and final log hashes after process exit.", "",
        ]
        report_path.write_text("\n".join(report), encoding="utf-8")
    return result


def validate_specification(spec: dict) -> None:
    summary = spec["summary_control"]
    micro = spec["microfit"]
    scheduler = spec["scheduler"]
    if summary["architecture"] != "Linear20to81,ReLU,Linear81to81,ReLU,Linear81to1":
        raise ValueError("summary architecture differs from the assigned contract")
    if summary["features"] != "exact unchanged check_s1_benchmark.gbdt_row 20 columns; no IDs, equality-to29 or labels added":
        raise ValueError("summary feature contract differs")
    if micro["trials"] != 60 or micro["endpoint"] != "BCE<=0.02 and all16 correct at zero-logit threshold for20 consecutive evaluations; else cap/not-established":
        raise ValueError("microfit contract differs")
    if scheduler["name"] != "ReduceLROnPlateau" or scheduler["metric"] != "training BCE only":
        raise ValueError("scheduler contract differs")


def write_run_manifest(root: Path, args, data: dict) -> dict:
    command = [sys.executable, "-s", "-B", str(Path(__file__).resolve()), "--world-index", str(args.world_index), "--study-config", str(args.study_config), "--output-dir", str(args.output_dir)]
    for option, value in (
        ("--contract-report", args.contract_report),
        ("--failure-history", args.failure_history),
        ("--reference-json", args.reference_json),
    ):
        if value is not None:
            command.extend((option, str(value)))
    input_hashes = data["input_hashes"]
    source_hash = next(x for x in input_hashes if Path(x["path"]).name == "diagnose_s1_learning.py")
    config_hash = next(x for x in input_hashes if Path(x["path"]) == args.study_config.resolve())
    index_hash = next(x for x in input_hashes if Path(x["path"]) == args.world_index.resolve())
    resolved = [
        {"world_id": wid, "role": data["entries"][wid]["role"], "attempt": data["entries"][wid]["indexed_attempt"], "resolved_attempt": str(data["entries"][wid]["path"])}
        for wid in TRAIN_IDS + VAL_IDS
    ]
    run_manifest = {
        "experiment_id": "S1-F",
        "command": command,
        "cwd": str(Path.cwd()),
        "environment": current_environment(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "process_start_utc": PROCESS_START_UTC,
        "empirical_start_utc": datetime.now(timezone.utc).isoformat(),
        "limits": {"wall_seconds": WALL_LIMIT, "maximum_new_output_bytes": OUTPUT_LIMIT, "wall_reserve_seconds": WALL_RESERVE, "output_reserve_bytes": OUTPUT_RESERVE},
        "train_worlds": list(TRAIN_IDS), "validation_worlds": list(VAL_IDS),
        "resolved_worlds": resolved, "excluded_worlds": data["excluded_worlds"],
        "input_hashes": input_hashes, "source_hash": source_hash, "config_hash": config_hash, "world_index_hash": index_hash,
        "failure_history_path": str(args.failure_history.resolve()) if args.failure_history is not None else None,
        "preserved_prior_failure_log": next(
            (x for x in input_hashes if Path(x["path"]) == args.failure_history.resolve()), None
        ) if args.failure_history is not None else None,
        "reference_json_path": str(args.reference_json.resolve()) if args.reference_json is not None else None,
        "contract_report_path": str(args.contract_report.resolve()) if args.contract_report is not None else None,
        "resource_measurement": "resource.getrusage(RUSAGE_SELF), measured by the running process",
        "launch_log_files": ["stdout.log", "stderr.log", "process_exit.json"],
    }
    write_json(root / "run_manifest.json", run_manifest)
    return run_manifest


def qualify_only(args) -> int:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_manifest.json").exists() or any((root / name).exists() for name in ("trial_order.json", "summary_control", "microfit")):
        raise FileExistsError(f"S1-F empirical output already exists; qualification will not overwrite it: {root}")
    torch.set_num_threads(1)
    data = prepare_data(
        args.world_index.resolve(strict=True), args.study_config.resolve(strict=True), root,
        write_artifacts=False, failure_history_path=args.failure_history,
        reference_json_path=args.reference_json,
    )
    validate_specification(data["spec"])
    qualification = qualify_models(data, data["spec"])
    qualification["input_hashes"] = data["input_hashes"]
    qualification["fixed20_collisions"] = data["fixed20_collisions"]
    qualification["normalizer_preview"] = data["normalizer_json"]
    qualification["tensor_hashes"] = data["tensor_hashes"]
    qualification["elapsed_seconds"] = elapsed_seconds()
    qualification["resources"] = measured_resources()
    write_json(root / "qualification.json", qualification)
    print(json.dumps({"status": "PASS", "scope": qualification["scope"], "elapsed_seconds": qualification["elapsed_seconds"]}, indent=2))
    return 0


def run_empirical(args) -> int:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = args.contract_report.resolve() if args.contract_report is not None else None
    if report_path is not None and report_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing S1-F contract report: {report_path}")
    allowed_before = {"qualification.json", "qualification_stdout.log", "qualification_stderr.log", "stdout.log", "stderr.log"}
    unexpected_existing = [path.name for path in root.iterdir() if path.name not in allowed_before]
    if unexpected_existing:
        raise FileExistsError(f"refusing to overwrite existing S1-F output: {unexpected_existing[:5]}")
    load_helpers()
    qualification_path = root / "qualification.json"
    if not qualification_path.is_file():
        raise FileNotFoundError("focused qualification.json is required before empirical processing")
    qualification = strict_json(qualification_path)
    if qualification.get("status") != "PASS":
        raise ValueError("focused qualification did not pass")
    start_wall = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    data = prepare_data(
        args.world_index.resolve(strict=True), args.study_config.resolve(strict=True), root,
        write_artifacts=False, failure_history_path=args.failure_history,
        reference_json_path=args.reference_json,
    )
    validate_specification(data["spec"])
    qualified = {x["path"]: x["sha256"] for x in qualification["input_hashes"]}
    current = {x["path"]: x["sha256"] for x in data["input_hashes"]}
    if qualified != current:
        raise ValueError("focused qualification is stale for current source/config/world bytes")
    run_manifest = write_run_manifest(root, args, data)
    write_json(root / "world_structure_checks.json", data["world_structure"])
    write_json(root / "summary_control/fitted_transform.json", data["normalizer_json"])
    write_json(root / "summary_feature_collisions.json", data["fixed20_collisions"])
    write_json(root / "input_tensor_hashes.json", data["tensor_hashes"])
    write_json(root / "microfit/subset_identities.json", [{k: row[k] for k in ("world_id", "candidate_id", "label")} for row in data["microset"]])
    order = build_trial_order()
    write_json(root / "trial_order.json", {
        "round_robin_block_epochs": BLOCK_EPOCHS,
        "order_fixed_before_optimization": True,
        "order": order,
    })
    states, initialization_checks = init_all_trials(root, data["spec"], data, order)
    write_json(root / "initialization_checks.json", initialization_checks)
    planned_stop, unexpected = run_grid(root, states, data, data["spec"], report_path)
    try:
        result = finish_run(root, states, data, initialization_checks, qualification, start_wall, run_manifest, planned_stop, unexpected, report_path)
    except Exception as error:
        append_jsonl(root / "script_failures.jsonl", {
            "utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(), "phase": "finalization",
        })
        result = {
            "experiment_id": "S1-F", "status": "FAILED" if unexpected else "PARTIAL",
            "error": f"{type(error).__name__}: {error}", "elapsed_seconds": elapsed_seconds(),
            "trial_statuses": summarize_trials(states), "planned_stop_reason": planned_stop,
            "unexpected_script_error": unexpected,
        }
        write_json(root / "final_report.json", result)
        save_trial_statuses(root, states)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "# Contract Report — S1-F\n\n"
                f"Status: {result['status']}. Finalization limitation: {type(error).__name__}: {error}\n\n"
                f"Actual retained trial statuses: `{root / 'trial_statuses.csv'}`. "
                f"Actual launch, input hashes and states: `{root / 'run_manifest.json'}` and per-trial directories.\n\n"
                "The experiment is incomplete; missing diagnostics/figures/metrics are unavailable. "
                "No automatic extension, successful-run repetition or scientific qualification follows.\n",
                encoding="utf-8",
            )
    print(json.dumps({"status": result.get("status"), "elapsed_seconds": result.get("elapsed_seconds"), "output_bytes": output_bytes(root, extras=(report_path,)), "summary_trials_completed": sum(x["kind"] == "summary" and x["status"] == "completed" for x in states), "microfit_trials_with_updates": sum(x["kind"] == "microfit" and x["epoch"] > 0 for x in states)}, indent=2, allow_nan=False))
    return 0 if result.get("status") in ("COMPLETED", "PARTIAL") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-index", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path, help="optional external contract report; the run directory always retains its endpoint report")
    parser.add_argument("--failure-history", type=Path, help="optional prior-failure log to bind into the run manifest")
    parser.add_argument("--reference-json", type=Path, help="optional JSON file of descriptive reference values")
    parser.add_argument("--qualify-only", action="store_true", help="run focused API/data-path qualification without optimization")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.world_index.is_file() or not args.study_config.is_file():
        raise FileNotFoundError("world index and study config must be existing files")
    for name in ("failure_history", "reference_json"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"optional {name.replace('_', '-')} file does not exist: {path}")
    return qualify_only(args) if args.qualify_only else run_empirical(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"S1-F script error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
