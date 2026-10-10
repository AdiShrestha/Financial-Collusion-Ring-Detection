#!/usr/bin/env python3
"""Score a registered S1 holdout with frozen states; this process never reads labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SUMMARY_ARM = "fixed20_tabular_neural_control"
GBDT_ARM = "gbdt_fixed20"
NEURAL_ARMS = (
    SUMMARY_ARM,
    "cellular_cwn_style_local",
    "cellular_hasse_mechanism_control",
    "directed_local_edge_gnn",
    "simplicial_mpsn_style_local",
)
ARMS = (*NEURAL_ARMS, GBDT_ARM)
SEEDS = (11, 23, 37, 53, 71)
INPUT_ALLOWLIST = (
    "candidate_observations.jsonl",
    "candidate_identity.jsonl",
    "cellular_polygonal_boundaries.jsonl",
    "simplicial_subdivision.jsonl",
    "hasse_incidence.jsonl",
    "world_directed_multigraph_identity.jsonl",
    "world_node_identity.json",
)
PREDICTION_COLUMNS = (
    "experiment_id",
    "arm",
    "initialization_seed",
    "world_id",
    "candidate_id",
    "model_sha256",
    "world_input_digest",
    "raw_logit",
)
FORBIDDEN_OBSERVATION_FIELDS = {
    "label", "sar", "alert_id", "alertID", "seed", "generator", "role",
    "candidate_rank", "event_ordinal", "candidate_id", "instance_id",
    "world_id", "account_id", "event_ids",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
    return result


def strict_json_bytes(raw: bytes, where: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict JSON in {where}: {exc}") from exc


def strict_json_file(path: Path) -> Any:
    with path.open("rb") as handle:
        return strict_json_bytes(handle.read(), str(path))


def strict_jsonl_bytes(raw: bytes, where: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = strict_json_bytes(line.encode("utf-8"), f"{where}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {where}:{line_number}")
        rows.append(value)
    return rows


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": h.hexdigest()}


def _atomic_json(path: Path, value: Any) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_json_arg(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required input is missing or symlinked: {path}")
    return strict_json_file(path)


def _source_reference(lock: dict[str, Any], basename: str) -> tuple[Path, dict[str, Any]]:
    matches = [
        row for row in lock.get("source_reference", [])
        if Path(str(row.get("path", ""))).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"model lock must bind exactly one {basename} source reference")
    row = matches[0]
    copied = row.get("copy")
    selected = Path(copied["path"]) if isinstance(copied, dict) and copied.get("path") else Path(row["path"])
    expected = copied if isinstance(copied, dict) and copied.get("path") else row
    if not selected.is_file() or selected.is_symlink():
        raise FileNotFoundError(f"pinned source is missing or symlinked: {selected}")
    actual = file_binding(selected)
    if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
        raise ValueError(f"pinned source bytes differ from model lock: {selected}")
    return selected, actual


def load_benchmark(lock: dict[str, Any]):
    path, binding = _source_reference(lock, "check_s1_benchmark.py")
    module_name = f"s1g_pinned_benchmark_{binding['sha256'][:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned benchmark helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, binding


def validate_protocol(study: dict[str, Any], lock: dict[str, Any]) -> list[str]:
    experiment_id = study.get("experiment_id")
    if type(experiment_id) is not str or not experiment_id or study.get("phase") != 4:
        raise ValueError("study config lacks a valid experiment identity or evaluation phase")
    if study.get("status") != "REGISTERED_UNEXECUTED":
        raise ValueError("study config status differs from the registered unexecuted protocol")
    if study.get("model_states") != 30 or study.get("prediction_blocks") != 120:
        raise ValueError("registered model or prediction-block count differs")
    inference = study.get("inference", {})
    if any(inference.get(key) is not expected for key, expected in (
        ("training_authorized", False),
        ("optimizer_authorized", False),
        ("checkpoint_reselection_authorized", False),
        ("successful_block_rescoring_authorized", False),
    )):
        raise ValueError("study config authorizes an out-of-scope fitting or rescoring action")
    if inference.get("selected_gbdt_iteration") != 70 or inference.get("torch_threads") != 1:
        raise ValueError("selected GBDT iteration or thread count differs from the lock")
    if inference.get("device") != "cpu" or inference.get("mode") != "eval/no_grad":
        raise ValueError("inference device/mode differs from the lock")
    if lock.get("experiment_id") != experiment_id or lock.get("status") != "MODEL_AND_POLICY_BYTES_LOCKED":
        raise ValueError("model lock identity or status differs from the study")
    rows = lock.get("models")
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("model lock must contain exactly 30 frozen states")
    model_ids = [row.get("model_id") for row in rows]
    if len(set(model_ids)) != 30 or any(not isinstance(value, str) for value in model_ids):
        raise ValueError("model lock identifiers are missing or duplicated")
    arm_counts = {arm: 0 for arm in ARMS}
    seeds_by_arm: dict[str, set[int]] = {arm: set() for arm in ARMS}
    for row in rows:
        arm = row.get("arm")
        seed = row.get("seed")
        if arm not in arm_counts or type(seed) is not int or seed not in SEEDS:
            raise ValueError(f"unexpected locked model identity: {row.get('model_id')!r}")
        arm_counts[arm] += 1
        seeds_by_arm[arm].add(seed)
        if arm == GBDT_ARM:
            if row.get("selected_iteration") != 70 or row.get("refit_authorized") is not False:
                raise ValueError("GBDT state must remain fixed at staged iteration 70")
        elif type(row.get("selected_epoch")) is not int:
            raise ValueError(f"neural checkpoint epoch is absent: {row.get('model_id')}")
    if any(count != 5 for count in arm_counts.values()):
        raise ValueError(f"unexpected locked arm counts: {arm_counts}")
    if any(values != set(SEEDS) for values in seeds_by_arm.values()):
        raise ValueError("each registered arm must contain the exact five initialization settings")
    expected_order = study.get("test_worlds")
    if not isinstance(expected_order, list) or len(expected_order) != 4 or len(set(expected_order)) != 4:
        raise ValueError("study config must name four unique registered test worlds")
    if lock.get("number_of_model_states") != 30:
        raise ValueError("model-lock state count differs from the study config")
    return expected_order


def _safe_input_path(attempt_dir: Path, output_root: Path, name: str) -> Path:
    candidate = attempt_dir / name
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"allowlisted predictor input is missing or symlinked: {candidate}")
    attempt_resolved = attempt_dir.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(attempt_resolved):
        raise ValueError(f"predictor input escapes its attempt directory: {candidate}")
    if not resolved.is_relative_to(output_root):
        raise ValueError(f"predictor input escapes the registered output root: {candidate}")
    return resolved


def _safe_attempt_path(index_path: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    output_root = index_path.parent.resolve(strict=True)
    raw = entry.get("attempt_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("world index entry has no attempt_path")
    attempt = Path(raw)
    if not attempt.is_absolute():
        attempt = index_path.parent / attempt
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError(f"world attempt is missing or symlinked: {attempt}")
    resolved = attempt.resolve(strict=True)
    if not resolved.is_relative_to(output_root) or resolved == output_root:
        raise ValueError(f"world attempt escapes the world-index output root: {attempt}")
    current = output_root
    for component in resolved.relative_to(output_root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"symlink in world attempt path: {current}")
    return resolved, output_root


def _strict_world_inputs(attempt: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    parsed: dict[str, Any] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for name in INPUT_ALLOWLIST:
        path = attempt / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"allowlisted input is missing or symlinked: {path}")
        raw = path.read_bytes()
        bindings[name] = {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}
        if name.endswith(".jsonl"):
            parsed[name] = strict_jsonl_bytes(raw, str(path))
        else:
            parsed[name] = strict_json_bytes(raw, str(path))
    return parsed, bindings


def _candidate_ids(rows: list[dict[str, Any]], field: str, where: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(field)
        if type(value) is not str or not value:
            raise ValueError(f"invalid {field} in {where}")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field} values in {where}")
    return values


def _join_and_build_world(
    entry: dict[str, Any], index_path: Path, benchmark: Any,
) -> dict[str, Any]:
    world_id = entry.get("world_id")
    seed = entry.get("seed")
    if type(world_id) is not str or not world_id or type(seed) is not int:
        raise ValueError("world index entry has an invalid world identity")
    if entry.get("role") != "heldout_test" or entry.get("status") != "SOURCE_RECONCILED":
        raise ValueError(f"world is not marked as source-reconciled heldout: {world_id}")
    attempt, output_root = _safe_attempt_path(index_path, entry)
    parsed, input_hashes = _strict_world_inputs(attempt)
    ids = parsed["candidate_identity.jsonl"]
    observations = parsed["candidate_observations.jsonl"]
    row_ids = _candidate_ids(ids, "candidate_id", f"{world_id} identities")
    if not row_ids:
        raise ValueError(f"empty candidate frame in {world_id}")
    if len(observations) != len(ids):
        raise ValueError(f"observation/identity row count differs in {world_id}")
    for identity in ids:
        if identity.get("world_id") != world_id:
            raise ValueError(f"candidate world identity mismatch in {world_id}")
    sidecar_names = (
        "cellular_polygonal_boundaries.jsonl",
        "simplicial_subdivision.jsonl",
        "hasse_incidence.jsonl",
    )
    for name in sidecar_names:
        sidecar_ids = _candidate_ids(parsed[name], "candidate_id", f"{world_id} {name}")
        if set(sidecar_ids) != set(row_ids):
            raise ValueError(f"candidate identity join mismatch in {world_id} {name}")
        # The retained pinned constructor accepts row-aligned sidecars. Verify exact IDs
        # before passing data to it; observation rows intentionally contain no ID field.
        if sidecar_ids != row_ids:
            raise ValueError(f"row order differs from exact candidate identity order in {world_id} {name}")
    for observation in observations:
        forbidden = FORBIDDEN_OBSERVATION_FIELDS.intersection(observation)
        if forbidden:
            raise ValueError(f"forbidden label/identity values in predictor row: {sorted(forbidden)}")
    graph_ids = parsed["world_directed_multigraph_identity.jsonl"]
    event_ids = [row.get("physical_event_id") for row in graph_ids]
    if any(type(value) is not int for value in event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError(f"invalid or duplicate physical event IDs in {world_id}")
    node_doc = parsed["world_node_identity.json"]
    if not isinstance(node_doc, dict) or node_doc.get("world_id") != world_id:
        raise ValueError(f"world coordinate map does not identify {world_id}")
    node_order = node_doc.get("node_order")
    event_order = node_doc.get("event_order")
    if not isinstance(node_order, list) or not isinstance(event_order, list):
        raise ValueError(f"world coordinate maps are malformed in {world_id}")
    # Exact coordinate uniqueness and graph coverage are checked before the retained
    # builder creates any matrices.
    if len(node_order) != len(set(node_order)) or len(event_order) != len(set(event_order)):
        raise ValueError(f"duplicate world coordinate in {world_id}")
    if set(event_order) != set(event_ids):
        raise ValueError(f"physical event rows do not cover the event coordinate map in {world_id}")
    bench_input_paths = [attempt / name for name in INPUT_ALLOWLIST]
    data = benchmark.load_inputs(attempt)
    # load_inputs is label-free but its diagnostic inventory also includes a label path.
    # Remove that reference before returning any prediction input view to the scorer.
    data["inputs"] = [path for path in bench_input_paths if path.name in INPUT_ALLOWLIST]
    data.pop("labels_path", None)
    if len(data["rows"]) != len(row_ids) or [row["candidate_id"] for row in data["ids"]] != row_ids:
        raise ValueError(f"pinned builder row order differs from strict identity parse in {world_id}")
    n_nodes = len(data["node_order"])
    n_events = len(data["event_order"])
    views = []
    for row_number, row in enumerate(data["rows"]):
        b1, b2 = benchmark.cellular(row, n_nodes, n_events)
        simplicial = benchmark.simplicial(row, n_nodes, n_events)
        if not benchmark.closed(b1, b2):
            raise ValueError(f"cellular B1@B2 != 0 in {world_id} candidate row {row_number}")
        if not benchmark.closed(simplicial[3], simplicial[4]):
            raise ValueError(f"simplicial boundary closure failed in {world_id} candidate row {row_number}")
        views.append({"r": row, "cell": (b1, b2), "simp": simplicial})
    digest_input = json.dumps(
        [{"name": name, "sha256": input_hashes[name]["sha256"]} for name in INPUT_ALLOWLIST],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "world_id": world_id,
        "seed": seed,
        "role": "heldout_test",
        "attempt_path": str(attempt),
        "output_root": str(output_root),
        "data": data,
        "views": views,
        "candidate_ids": row_ids,
        "candidate_count": len(row_ids),
        "input_hashes": input_hashes,
        "world_input_digest": sha256_bytes(digest_input),
    }


def _read_locked_artifact(record: dict[str, Any], where: str) -> tuple[Path, dict[str, Any]]:
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"locked {where} artifact is missing or symlinked: {path}")
    actual = file_binding(path)
    if actual["bytes"] != record.get("bytes") or actual["sha256"] != record.get("sha256"):
        raise ValueError(f"locked {where} artifact hash differs: {path}")
    return path, actual


def load_summary_transform(lock: dict[str, Any], study: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = lock.get("summary_transform", {}).get("frozen")
    if not isinstance(ref, dict):
        raise ValueError("model lock lacks the frozen summary transform")
    path, binding = _read_locked_artifact(ref, "summary transform")
    transform = _load_json_arg(path)
    feature_order = study.get("features", {}).get("feature_order")
    if transform.get("columns") != feature_order or len(feature_order or []) != 20:
        raise ValueError("frozen transform feature order differs from the registered 20-column order")
    if transform.get("normalization_dtype") != "float64" or transform.get("model_tensor_dtype") != "float32":
        raise ValueError("frozen transform dtype differs from the registered contract")
    policy = lock.get("summary_transform", {})
    if policy.get("refit_authorized") is not False:
        raise ValueError("model lock must prohibit transform refitting")
    if transform.get("fit_candidate_count") != 312:
        raise ValueError("frozen transform was not fitted on the registered training rows")
    if transform.get("fit_worlds") != policy.get("fit_worlds") or transform.get("fit_worlds") != study.get("training_worlds_unchanged"):
        raise ValueError("frozen transform training worlds differ from the locked policy")
    mean = np.asarray(transform.get("mean_float64"), dtype=np.float64)
    scale = np.asarray(transform.get("scale_float64"), dtype=np.float64)
    if mean.shape != (20,) or scale.shape != (20,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("frozen normalization vectors are malformed")
    if np.any(scale <= 0):
        raise ValueError("frozen normalization contains a nonpositive scale")
    if transform.get("constant_columns"):
        columns = transform["columns"]
        for name in transform["constant_columns"]:
            index = columns.index(name)
            if scale[index] != 1.0:
                raise ValueError(f"constant feature {name} must retain scale one")
    return transform, binding


def summary_feature_matrices(world: dict[str, Any], benchmark: Any, transform: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray([benchmark.gbdt_row(row) for row in world["data"]["rows"]], dtype=np.float64)
    if raw.shape != (world["candidate_count"], 20) or not np.isfinite(raw).all():
        raise ValueError(f"fixed20 feature matrix is invalid in {world['world_id']}")
    mean = np.asarray(transform["mean_float64"], dtype=np.float64)
    scale = np.asarray(transform["scale_float64"], dtype=np.float64)
    normalized64 = (raw - mean[None, :]) / scale[None, :]
    if not np.isfinite(normalized64).all():
        raise ValueError(f"normalized summary matrix is non-finite in {world['world_id']}")
    return raw, normalized64.astype(np.float32)


def _new_neural_model(record: dict[str, Any], benchmark: Any) -> nn.Module:
    arm = record["arm"]
    if arm == SUMMARY_ARM:
        return nn.Sequential(nn.Linear(20, 81), nn.ReLU(), nn.Linear(81, 81), nn.ReLU(), nn.Linear(81, 1))
    width = record.get("hidden_width")
    if type(width) is not int or width <= 0:
        raise ValueError(f"locked hidden width is invalid for {record.get('model_id')}")
    constructors = {
        "cellular_cwn_style_local": lambda: benchmark.Cell(width),
        "cellular_hasse_mechanism_control": lambda: benchmark.Cell(width, hasse=True),
        "directed_local_edge_gnn": lambda: benchmark.Directed(width),
        "simplicial_mpsn_style_local": lambda: benchmark.Simp(width),
    }
    if arm not in constructors:
        raise ValueError(f"no inference constructor for locked arm {arm}")
    return constructors[arm]()


def load_neural_model_record(record: dict[str, Any], benchmark: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = record.get("frozen")
    if not isinstance(frozen, dict):
        raise ValueError(f"locked neural checkpoint is missing: {record.get('model_id')}")
    path, binding = _read_locked_artifact(frozen, record["model_id"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"locked neural checkpoint payload is malformed: {record.get('model_id')}")
    if payload.get("epoch") != record.get("selected_epoch"):
        raise ValueError(f"checkpoint epoch differs from model lock: {record.get('model_id')}")
    if payload.get("arm") != record.get("arm") or payload.get("seed") != record.get("seed"):
        raise ValueError(f"checkpoint identity differs from model lock: {record.get('model_id')}")
    model = _new_neural_model(record, benchmark)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device="cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {**record, "model": model, "model_sha256": binding["sha256"]}, binding


def load_gbdt_model_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = record.get("frozen")
    if not isinstance(frozen, dict):
        raise ValueError(f"locked GBDT file is missing: {record.get('model_id')}")
    path, binding = _read_locked_artifact(frozen, record["model_id"])
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if type(record.get("selected_iteration")) is not int or record["selected_iteration"] != 70:
        raise ValueError(f"GBDT selected iteration differs from 70: {record.get('model_id')}")
    if getattr(model, "n_iter_", None) != 800:
        raise ValueError(f"retained GBDT object does not contain its locked 800 fitted rounds: {record.get('model_id')}")
    if not callable(getattr(model, "staged_decision_function", None)):
        raise ValueError(f"retained GBDT object lacks staged inference: {record.get('model_id')}")
    return {**record, "model": model, "model_sha256": binding["sha256"]}, binding


def load_locked_models(lock: dict[str, Any], benchmark: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    artifact_bindings: list[dict[str, Any]] = []
    for record in lock["models"]:
        arm = record["arm"]
        if arm == GBDT_ARM:
            item, binding = load_gbdt_model_record(record)
            artifact_bindings.append(binding)
            loaded.append(item)
            continue
        item, binding = load_neural_model_record(record, benchmark)
        artifact_bindings.append(binding)
        loaded.append(item)
    if len(loaded) != 30:
        raise ValueError(f"expected 30 loaded frozen states; got {len(loaded)}")
    return loaded, artifact_bindings


def selected_gbdt_scores(model: Any, features: np.ndarray, iteration: int = 70) -> np.ndarray:
    if type(iteration) is not int or iteration != 70:
        raise ValueError("only the registered one-based staged GBDT iteration 70 is permitted")
    for step, scores in enumerate(model.staged_decision_function(features), 1):
        if step == iteration:
            values = np.asarray(scores, dtype=np.float64)
            if values.shape == (len(features), 1):
                values = values[:, 0]
            if values.shape != (len(features),) or not np.isfinite(values).all():
                raise ValueError("invalid staged GBDT scores at selected iteration 70")
            return values
    raise ValueError("retained GBDT object ended before staged iteration 70")


def predict_neural_logits(
    model: nn.Module,
    arm: str,
    world: dict[str, Any],
    benchmark: Any,
    normalized_summary: np.ndarray,
) -> np.ndarray:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for index, view in enumerate(world["views"]):
            if arm == SUMMARY_ARM:
                tensor = torch.as_tensor(normalized_summary[index], dtype=torch.float32, device="cpu")
                result = model(tensor)
            else:
                result = model(*benchmark.build_args(view, arm))
            value = float(result.detach().cpu().reshape(-1)[0].item())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite logit for {world['world_id']} row {index} arm {arm}")
            values.append(value)
    if len(values) != world["candidate_count"]:
        raise ValueError(f"neural prediction coverage differs in {world['world_id']} for {arm}")
    return np.asarray(values, dtype=np.float64)


def _write_block(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for row in rows:
            writer.writerow(row)
            handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)
    return file_binding(path)


def _write_combined_predictions(path: Path, block_paths: list[Path]) -> dict[str, Any]:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        output_handle.flush()
        for block_path in block_paths:
            with block_path.open("r", newline="", encoding="utf-8") as block_handle:
                reader = csv.DictReader(block_handle)
                if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                    raise ValueError(f"prediction block has an unexpected schema: {block_path}")
                for row in reader:
                    writer.writerow(row)
            output_handle.flush()
        os.fsync(output_handle.fileno())
    temporary.replace(path)
    return file_binding(path)


def score_registered_worlds(
    index_path: Path,
    study_path: Path,
    model_lock_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden_existing = [
        output_dir / "pre_label_prediction_manifest.json",
        output_dir / "predictions.csv",
        output_dir / "prediction_blocks",
    ]
    if any(path.exists() for path in forbidden_existing):
        raise FileExistsError("prediction output already exists; successful blocks are never overwritten or rescored")
    index = _load_json_arg(index_path)
    study = _load_json_arg(study_path)
    lock = _load_json_arg(model_lock_path)
    expected_worlds = validate_protocol(study, lock)
    experiment_id = study["experiment_id"]
    if file_binding(model_lock_path)["sha256"] != study.get("model_lock", {}).get("sha256"):
        raise ValueError("model-lock bytes differ from the study binding")
    if not isinstance(index, dict) or index.get("experiment_id") != experiment_id:
        raise ValueError("world index identity differs from the study")
    if index.get("schema") != "s1_phase4_heldout_world_index_v1":
        raise ValueError("world index schema is not the heldout wrapper schema")
    entries = index.get("worlds")
    if not isinstance(entries, list) or [entry.get("world_id") for entry in entries] != expected_worlds:
        raise ValueError("world index does not exactly cover the four registered worlds in order")
    seeds = [entry.get("seed") for entry in entries]
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(entries):
        raise ValueError("world index seeds must be unique integers; registry equality is checked by the caller")
    benchmark, benchmark_binding = load_benchmark(lock)
    transform, transform_binding = load_summary_transform(lock, study)
    worlds = [_join_and_build_world(entry, index_path, benchmark) for entry in entries]
    if len(worlds) != 4:
        raise ValueError("exactly four reconciled worlds are required before scoring")
    expected_blocks = 4 * 30
    torch.set_num_threads(1)
    models, model_bindings = load_locked_models(lock, benchmark)
    output_dir = output_dir.resolve()
    blocks_dir = output_dir / "prediction_blocks"
    blocks_dir.mkdir()
    block_records: list[dict[str, Any]] = []
    block_paths: list[Path] = []
    completed = 0
    try:
        for model_record in models:
            arm = model_record["arm"]
            seed = model_record["seed"]
            for world in worlds:
                raw_summary, normalized_summary = summary_feature_matrices(world, benchmark, transform)
                if arm == GBDT_ARM:
                    logits = selected_gbdt_scores(model_record["model"], raw_summary, 70)
                else:
                    logits = predict_neural_logits(
                        model_record["model"], arm, world, benchmark, normalized_summary,
                    )
                if logits.shape != (world["candidate_count"],) or not np.isfinite(logits).all():
                    raise ValueError(f"score block has invalid shape or values: {model_record['model_id']} / {world['world_id']}")
                block_name = f"{completed + 1:03d}_{arm}_seed_{seed}_{world['world_id']}.csv"
                block_path = blocks_dir / block_name
                rows = [
                    {
                        "experiment_id": experiment_id,
                        "arm": arm,
                        "initialization_seed": seed,
                        "world_id": world["world_id"],
                        "candidate_id": candidate_id,
                        "model_sha256": model_record["model_sha256"],
                        "world_input_digest": world["world_input_digest"],
                        "raw_logit": format(float(logit), ".17g"),
                    }
                    for candidate_id, logit in zip(world["candidate_ids"], logits, strict=True)
                ]
                binding = _write_block(block_path, rows)
                block_paths.append(block_path)
                block_records.append({
                    "arm": arm,
                    "initialization_seed": seed,
                    "model_id": model_record["model_id"],
                    "world_id": world["world_id"],
                    "candidate_count": world["candidate_count"],
                    "candidate_ids_sha256": sha256_bytes("\n".join(world["candidate_ids"]).encode("utf-8")),
                    "world_input_digest": world["world_input_digest"],
                    "file": str(block_path.relative_to(output_dir)),
                    "bytes": binding["bytes"],
                    "sha256": binding["sha256"],
                })
                completed += 1
        if completed != expected_blocks:
            raise ValueError(f"expected {expected_blocks} prediction blocks; completed {completed}")
        predictions_binding = _write_combined_predictions(output_dir / "predictions.csv", block_paths)
        manifest = {
            "schema": "s1_phase4_prelabel_prediction_manifest_v1",
            "experiment_id": experiment_id,
            "labels_opened_by_this_process": False,
            "model_states": len(models),
            "registered_worlds": len(worlds),
            "planned_prediction_blocks": expected_blocks,
            "completed_prediction_blocks": completed,
            "score_columns": list(PREDICTION_COLUMNS),
            "input_allowlist": list(INPUT_ALLOWLIST),
            "pinned_benchmark": benchmark_binding,
            "frozen_transform": transform_binding,
            "frozen_models": model_bindings,
            "worlds": [
                {
                    "world_id": world["world_id"],
                    "seed": world["seed"],
                    "role": "heldout_test",
                    "candidate_count": world["candidate_count"],
                    "candidate_ids_sha256": sha256_bytes("\n".join(world["candidate_ids"]).encode("utf-8")),
                    "world_input_digest": world["world_input_digest"],
                    "input_hashes": [world["input_hashes"][name] for name in INPUT_ALLOWLIST],
                }
                for world in worlds
            ],
            "blocks": block_records,
            "predictions_csv": {
                "file": "predictions.csv",
                "bytes": predictions_binding["bytes"],
                "sha256": predictions_binding["sha256"],
            },
            "prediction_blocks_directory": "prediction_blocks",
        }
        _atomic_json(output_dir / "pre_label_prediction_manifest.json", manifest)
        return {
            "status": "PREDICTIONS_COMPLETE_LABELS_NOT_OPENED",
            "completed_prediction_blocks": completed,
            "candidate_counts_by_world": {world["world_id"]: world["candidate_count"] for world in worlds},
            "predictions_sha256": predictions_binding["sha256"],
            "manifest_sha256": file_binding(output_dir / "pre_label_prediction_manifest.json")["sha256"],
        }
    except Exception as exc:
        partial = {
            "schema": "s1_phase4_prediction_partial_v1",
            "experiment_id": experiment_id,
            "status": "FAILED_PARTIAL_NO_RESUME",
            "completed_prediction_blocks": completed,
            "planned_prediction_blocks": expected_blocks,
            "completed_block_records": block_records,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not (output_dir / "partial_result.json").exists():
            _atomic_json(output_dir / "partial_result.json", partial)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-index", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = score_registered_worlds(args.world_index, args.study_config, args.model_lock, args.output_dir)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
