#!/usr/bin/env python3
"""Continue learning-rate and stability analysis on the existing S1 development data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from types import SimpleNamespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import scipy
import sklearn

ARMS = (
    "cellular_cwn_style_local",
    "simplicial_mpsn_style_local",
    "directed_local_edge_gnn",
    "cellular_hasse_mechanism_control",
)
SEEDS = (11, 23, 37, 53, 71)
RATES = (0.01, 0.03)
TRAIN_IDS = ("S1_dev_101", "S1_dev_103", "S1_dev_107", "S1_dev_109")
VAL_IDS = ("S1_dev_113", "S1_dev_127")
ALL_IDS = TRAIN_IDS + VAL_IDS
MAX_SECONDS = 14_400
MAX_BYTES = 2_147_483_648
OUTPUT_RESERVE = 128 * 1024 * 1024
POSTPROCESS_RESERVE_SECONDS = 600
ACTIVE_OUTPUT_ROOT = None


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON number: {value}")
    return number


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=strict_pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


class BoundedStop(Exception):
    pass


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def bytes_dump(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(value)
    os.replace(tmp, path)


def torch_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def sha256_file(path: Path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return {"path": str(path), "bytes": size, "sha256": h.hexdigest()}


def output_size(root: Path):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))


def enforce_elapsed(started_epoch: float, processing: bool = True):
    elapsed = time.time() - started_epoch
    time_limit = MAX_SECONDS - POSTPROCESS_RESERVE_SECONDS if processing else MAX_SECONDS
    if elapsed >= time_limit:
        raise BoundedStop(
            "four-hour wall-time ceiling reached (post-processing reserve applied)"
            if processing
            else "four-hour gross wall-time ceiling reached"
        )


def enforce_limits(root: Path, started_epoch: float, reserve_bytes: int = 0, processing: bool = True):
    enforce_elapsed(started_epoch, processing)
    used = output_size(root)
    cap = MAX_BYTES - OUTPUT_RESERVE if processing else MAX_BYTES
    if used + reserve_bytes >= cap:
        raise BoundedStop(
            "2-GiB new-output ceiling reached (output reserve applied)"
            if processing
            else "2-GiB gross new-output ceiling reached"
        )


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("s1_learning_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load project learning helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    bench = module.load_bench()
    # Training helpers take an explicit representation/model interface.
    # Compose that interface without invoking either module's command entry point.
    values = dict(vars(bench))
    values.update(vars(module))
    values["benchmark_source"] = Path(bench.__file__)
    return SimpleNamespace(**values)


def record_attempt_failure(root: Path, exc: BaseException):
    record = {
        "timestamp_epoch": time.time(),
        "error_type": type(exc).__name__,
        "literal_error": str(exc),
        "traceback": traceback.format_exc(),
        "scientific_processing_started": bool((root / "run_manifest.json").exists()),
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as log_error:
        print(f"Could not retain failure log: {log_error}", file=sys.stderr)


def load_index_entries(index_path: Path):
    index = read_json(index_path)
    entries = {}
    for row in index.get("worlds", []):
        if row.get("replay") or row.get("world_id") == "S1_dev_1729":
            continue
        wid = row.get("world_id")
        if wid in ALL_IDS:
            if wid in entries:
                raise ValueError(f"duplicate indexed world: {wid}")
            role = (
                "provisional_development_train" if wid in TRAIN_IDS else "provisional_development_validation"
            )
            if row.get("status") != "COMPLETED" or row.get("role") != role:
                raise ValueError(f"indexed world status or role mismatch for {wid}")
            attempt = Path(row["attempt"])
            if not attempt.is_dir() or attempt.is_symlink():
                raise FileNotFoundError(f"indexed attempt path is missing or symlinked for {wid}: {attempt}")
            entries[wid] = attempt
    if set(entries) != set(ALL_IDS):
        raise ValueError("world index does not exactly cover the six authorized development worlds")
    return entries


def native_span_rows(entries, helper):
    """Build all scores from persisted physical-event lines before opening labels."""
    unlabeled = {}
    for wid in ALL_IDS:
        root = entries[wid]
        identity_rows = helper.read_jsonl(root / "candidate_identity.jsonl")
        persisted_rows = helper.read_jsonl(root / "sidecars" / "persisted_events.jsonl")
        event_by_id = {}
        for event in persisted_rows:
            event_id = event.get("physical_event_id")
            if type(event_id) is not int or event_id <= 0 or event_id in event_by_id:
                raise ValueError(f"invalid or duplicate persisted physical-event ID in {wid}")
            raw_line = event.get("csv_line")
            if type(raw_line) is not str:
                raise ValueError(f"persisted event lacks its raw transaction row in {wid}")
            columns = next(csv.reader([raw_line]))
            if len(columns) < 1 or not columns[0].isdigit():
                raise ValueError(f"native transaction step is not an integer in {wid}: {event_id}")
            event_by_id[event_id] = {"step": int(columns[0]), "csv_line": raw_line}
        candidates = []
        seen = set()
        for identity in identity_rows:
            candidate_id = identity.get("candidate_id")
            event_ids = identity.get("event_ids")
            if type(candidate_id) is not str or not candidate_id or candidate_id in seen:
                raise ValueError(f"invalid or duplicate candidate identity in {wid}")
            if not isinstance(event_ids, list) or not event_ids or any(type(x) is not int for x in event_ids):
                raise ValueError(f"invalid selected physical-event identity list in {wid}:{candidate_id}")
            if len(set(event_ids)) != len(event_ids) or any(eid not in event_by_id for eid in event_ids):
                raise ValueError(
                    f"candidate event IDs do not resolve uniquely to persisted events in {wid}:{candidate_id}"
                )
            steps = [event_by_id[eid]["step"] for eid in event_ids]
            span = max(steps) - min(steps)
            candidates.append(
                {
                    "world_id": wid,
                    "candidate_id": candidate_id,
                    "event_ids": event_ids,
                    "native_steps": steps,
                    "native_step_span": span,
                    "score": int(span == 29),
                }
            )
            seen.add(candidate_id)
        if len(candidates) != 78:
            raise ValueError(f"native span calculation did not cover the full 78-candidate world {wid}")
        unlabeled[wid] = candidates
    return unlabeled


def join_native_labels(unlabeled, entries, helper):
    """Labels are read only after all six candidate scores have been fixed."""
    joined = {}
    for wid in ALL_IDS:
        labels = helper.read_jsonl(entries[wid] / "candidate_labels.jsonl")
        by_id = {}
        for row in labels:
            cid = helper.exact_id(row, "native span label join")
            if cid in by_id:
                raise ValueError(f"duplicate annotation candidate key in {wid}")
            by_id[cid] = row
        scores = unlabeled[wid]
        score_ids = {x["candidate_id"] for x in scores}
        if score_ids != set(by_id) or len(by_id) != 78:
            raise ValueError(f"native score/annotation coverage mismatch in {wid}")
        rows = []
        for score in scores:
            label_row = by_id[score["candidate_id"]]
            rows.append(
                {**score, "label": helper.label_value(label_row), "instance_id": label_row.get("instance_id")}
            )
        joined[wid] = rows
    return joined


def ref_metrics(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores)
    if (
        y.dtype.kind not in "iu"
        or s.dtype.kind not in "iuf"
        or y.ndim != 1
        or not len(y)
        or s.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError("reference metric inputs are malformed")
    y = y.astype(np.int64)
    s = s.astype(np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        ap = None
    else:
        order = np.argsort(-s, kind="mergesort")
        tp = fp = 0
        prev_recall = 0.0
        ap = 0.0
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and s[order[j]] == s[order[i]]:
                j += 1
            group = y[order[i:j]]
            tp += int(group.sum())
            fp += len(group) - int(group.sum())
            recall = tp / positives
            ap += (recall - prev_recall) * (tp / (tp + fp))
            prev_recall = recall
            i = j
        ap = float(ap)
    if positives == 0 or negatives == 0:
        auc = None
    else:
        order = np.argsort(-s, kind="mergesort")
        negative_above = 0
        wins = 0
        ties = 0
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and s[order[j]] == s[order[i]]:
                j += 1
            group = y[order[i:j]]
            gp = int(group.sum())
            gn = len(group) - gp
            wins += gp * (negatives - negative_above - gn)
            ties += gp * gn
            negative_above += gn
            i = j
        auc = float((wins + 0.5 * ties) / (positives * negatives))
    bce = float(np.mean(np.logaddexp(0.0, s) - y * s))
    return {
        "average_precision_grouped_threshold": ap,
        "auroc_half_tie": auc,
        "binary_cross_entropy_logits": bce,
        "positives": positives,
        "negatives": negatives,
        "n": int(len(y)),
    }


def summarize_eval(helper, worlds, evaluated):
    per_world = {}
    for world in worlds:
        wid = world["world_id"]
        per_world[wid] = {
            "bce": float(evaluated[wid]["bce"]),
            "ap": float(evaluated[wid]["average_precision_grouped_threshold"]),
            "auc": float(evaluated[wid]["auroc_tie_aware"]),
        }
    return {
        "per_world": per_world,
        "equal_world_bce": float(np.mean([x["bce"] for x in per_world.values()])),
        "equal_world_ap": float(np.mean([x["ap"] for x in per_world.values()])),
        "equal_world_auc": float(np.mean([x["auc"] for x in per_world.values()])),
    }


def metric_gap(observed, reference):
    keys = (
        ("bce", "binary_cross_entropy_logits"),
        ("ap", "average_precision_grouped_threshold"),
        ("auc", "auroc_half_tie"),
    )
    return max(abs(float(observed[a]) - float(reference[b])) for a, b in keys)


def write_eval_artifact(
    path,
    trial_id,
    epoch,
    phase,
    arm,
    seed,
    rate,
    train,
    val,
    train_eval,
    val_eval,
    lr_before,
    lr_after,
    gradient_norms,
):
    rows = []
    inference = {"training": {}, "validation": {}}
    for split, worlds, eval_result in (("training", train, train_eval), ("validation", val, val_eval)):
        for world in worlds:
            wid = world["world_id"]
            inference[split][wid] = float(eval_result["inference_seconds_by_world"][wid])
            logits = eval_result[wid]["logits"]
            if len(logits) != 78:
                raise ValueError(f"candidate score count mismatch at {trial_id}:{epoch}:{wid}")
            for view, logit in zip(world["views"], logits):
                clipped = max(-80.0, min(80.0, float(logit)))
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "initial_rate": rate,
                        "trial_id": trial_id,
                        "phase": phase,
                        "epoch": epoch,
                        "split": split,
                        "world_id": wid,
                        "candidate_id": view["candidate_id"],
                        "instance_id": view["instance_id"],
                        "label": int(view["label"]),
                        "logit": float(logit),
                        "probability": float(1.0 / (1.0 + math.exp(-clipped))),
                    }
                )
    artifact = {
        "trial_id": trial_id,
        "arm": arm,
        "seed": seed,
        "initial_rate": rate,
        "phase": phase,
        "epoch": epoch,
        "learning_rate_before_scheduler": float(lr_before),
        "learning_rate_after_scheduler": float(lr_after),
        "training_metrics": summarize_eval(None, train, train_eval),
        "validation_metrics": summarize_eval(None, val, val_eval),
        "inference_seconds": inference,
        "gradient_norms_since_previous_evaluation": [float(x) for x in gradient_norms],
        "predictions": rows,
    }
    encoded = (json.dumps(artifact, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    bytes_dump(path, encoded)
    return artifact


def make_scheduler(optimizer, spec):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=spec["mode"],
        factor=float(spec["factor"]),
        patience=int(spec["patience_evaluations"]),
        threshold=float(spec["threshold"]),
        threshold_mode=spec["threshold_mode"],
        cooldown=int(spec["cooldown_evaluations"]),
        min_lr=float(spec["min_lr"]),
        eps=float(spec["eps"]),
    )


def scheduler_json_state(scheduler):
    """Describe the comparison sentinel without changing the resumable state."""
    state = copy.deepcopy(scheduler.state_dict())
    sentinel = state.get("mode_worse")
    if isinstance(sentinel, float) and math.isinf(sentinel):
        state["mode_worse"] = {
            "kind": "scheduler_comparison_sentinel",
            "value": "positive_infinity" if sentinel > 0 else "negative_infinity",
        }
    # Other nonfinite values remain errors; only the known sentinel is represented.
    json.dumps(state, allow_nan=False)
    return state


def scheduler_fixture(config):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.03,
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    loss = model(torch.tensor([[1.0, -1.0]])).square().mean()
    loss.backward()
    optimizer.step()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    second = nn.Linear(2, 1)
    second.load_state_dict(initial)
    second_optimizer = torch.optim.Adam(
        second.parameters(),
        lr=0.03,
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    second_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    assert all(torch.equal(initial[k], second.state_dict()[k]) for k in initial)
    assert (
        optimizer_state["param_groups"][0]["betas"]
        == second_optimizer.state_dict()["param_groups"][0]["betas"]
    )
    sched_spec = config["scheduler"]
    sched = make_scheduler(second_optimizer, sched_spec)
    sched.step(1.0)  # The initial observed training loss seeds the fresh scheduler state.
    initial_scheduler_state = copy.deepcopy(sched.state_dict())
    for _ in range(int(sched_spec["patience_evaluations"]) + 1):
        sched.step(1.1)
    reduced = float(second_optimizer.param_groups[0]["lr"])
    assert reduced < 0.03
    restored = make_scheduler(second_optimizer, sched_spec)
    restored.load_state_dict(copy.deepcopy(sched.state_dict()))
    assert restored.state_dict() == sched.state_dict()
    assert initial_scheduler_state["best"] == 1.0
    return {
        "status": "PASS",
        "fixture_scope": "tiny nonfinancial optimizer/scheduler API control",
        "adam_moments_restore": True,
        "scheduler_metric_is_explicit_scalar": True,
        "training_metric_seeds_scheduler": True,
        "plateau_reduces_rate": True,
        "scheduler_state_round_trip": True,
        "example_rate_after_declared_patience": reduced,
    }


def load_rng_state(helper, value):
    helper.restore_rng(value)


def model_for(helper, arm, widths):
    return helper.make_models(helper, widths)[arm]


def terminal_reconstruction(helper, arm, seed, prior, train, val, widths):
    rate = prior["selected_learning_rates"][arm]
    tag = f"lr_{float(rate):g}"
    directory = prior["root"] / "neural" / arm / f"seed_{seed}" / tag / "extension"
    terminal_path = directory / "latest.pt"
    selected_path = directory / "selected_extended.pt"
    trace_path = directory / "trace.jsonl"
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    trace = helper.read_trace(trace_path)
    if terminal.get("epoch") != 800 or not trace or trace[-1].get("epoch") != 800:
        raise ValueError(f"S1-D terminal state or trace is not at epoch800: {arm}/{seed}")
    best = helper.choose_best(trace)
    if selected.get("epoch") != best["epoch"]:
        raise ValueError(f"S1-D selected checkpoint epoch does not match trace minimum: {arm}/{seed}")
    state_best = terminal.get("best_state")
    if set(state_best or {}) != set(selected.get("model", {})):
        raise ValueError(
            f"S1-D selected state tensor keys differ from terminal best-state record: {arm}/{seed}"
        )
    max_state_difference = 0.0
    for key, value in selected["model"].items():
        max_state_difference = max(
            max_state_difference, float(torch.max(torch.abs(value.cpu() - state_best[key].cpu())).item())
        )
    if max_state_difference != 0.0:
        raise ValueError(f"S1-D best-state copy mismatch for {arm}/{seed}")
    model = model_for(helper, arm, widths)
    model.load_state_dict(terminal["model"])
    model.eval()
    train_eval = helper.eval_worlds(helper, model, train, arm)
    val_eval = helper.eval_worlds(helper, model, val, arm)
    recorded = trace[-1]
    max_metric_error = 0.0
    max_logit_error = 0.0
    for split, worlds, evaluated in (("train", train, train_eval), ("validation", val, val_eval)):
        for world in worlds:
            wid = world["world_id"]
            max_logit_error = max(
                max_logit_error,
                max(abs(a - b) for a, b in zip(evaluated[wid]["logits"], recorded[split][wid]["logits"])),
            )
            for key in ("bce", "average_precision_grouped_threshold", "auroc_tie_aware"):
                max_metric_error = max(
                    max_metric_error, abs(float(evaluated[wid][key]) - float(recorded[split][wid][key]))
                )
    if max_metric_error > 1e-12 or max_logit_error > 1e-7:
        raise ValueError(
            f"S1-D terminal predictions do not reproduce epoch800 trace metrics: {arm}/{seed}; gap={max_metric_error}"
        )
    return {
        "rate": float(rate),
        "directory": directory,
        "terminal_path": terminal_path,
        "selected_path": selected_path,
        "trace_path": trace_path,
        "terminal": terminal,
        "selected": selected,
        "trace": trace,
        "max_terminal_metric_error": max_metric_error,
        "max_terminal_logit_error": max_logit_error,
        "max_selected_state_copy_error": max_state_difference,
        "terminal_eval": {"train": train_eval, "validation": val_eval},
    }


def max_state_diff(a, b):
    if set(a) != set(b):
        return math.inf
    result = 0.0
    for key in a:
        result = max(result, float(torch.max(torch.abs(a[key].cpu() - b[key].cpu())).item()))
    return result


def branch_directory(root, arm, seed, rate):
    return root / "branches" / arm / f"seed_{seed}" / f"rate_{rate:.2f}"


def jsonl_write(path, rows):
    payload = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n" for row in rows
    )
    bytes_dump(path, payload.encode())


def write_trace(path, history):
    jsonl_write(path, history)


def copy_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_selected_continuation(directory):
    selected = torch.load(directory / "continuation" / "selected.pt", map_location="cpu", weights_only=False)
    terminal = torch.load(directory / "continuation" / "latest.pt", map_location="cpu", weights_only=False)
    if (
        selected["epoch"] != terminal["best_epoch"]
        or max_state_diff(selected["model"], terminal["best_state"]) != 0.0
    ):
        raise ValueError(f"continuation best checkpoint mismatch: {directory}")
    return selected


def branch_initialization(helper, root, arm, seed, rate, source, train, val, widths, config, started_epoch):
    directory = branch_directory(root, arm, seed, rate)
    latest_path = directory / "latest.pt"
    if latest_path.exists():
        return torch.load(latest_path, map_location="cpu", weights_only=False)
    terminal = source["terminal"]
    model = model_for(helper, arm, widths)
    model.load_state_dict(terminal["model"])
    optimizer_spec = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(rate),
        betas=tuple(optimizer_spec["betas"]),
        eps=float(optimizer_spec["eps"]),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    optimizer.load_state_dict(copy.deepcopy(terminal["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = float(rate)
    evals = source["terminal_eval"]
    initial_train_bce = float(summarize_eval(helper, train, evals["train"])["equal_world_bce"])
    scheduler = make_scheduler(optimizer, config["scheduler"])
    scheduler.step(initial_train_bce)
    best_trace = helper.choose_best(source["trace"])
    best_state = {k: v.detach().cpu().clone() for k, v in source["selected"]["model"].items()}
    initial_record = {
        "epoch": 800,
        "phase": "rate_bracket",
        "training_metrics": summarize_eval(helper, train, evals["train"]),
        "validation_metrics": summarize_eval(helper, val, evals["validation"]),
        "learning_rate_before_scheduler": float(rate),
        "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
        "scheduler_training_metric": initial_train_bce,
        "gradient_norms_since_previous_evaluation": [],
        "elapsed_seconds": time.time() - started_epoch,
    }
    score_path = directory / "scores" / "epoch_0800.json"
    trial_id = f"{arm}/seed_{seed}/rate_{rate:.2f}"
    write_eval_artifact(
        score_path,
        trial_id,
        800,
        "rate_bracket",
        arm,
        seed,
        rate,
        train,
        val,
        evals["train"],
        evals["validation"],
        rate,
        rate,
        [],
    )
    state = {
        "trial_id": trial_id,
        "arm": arm,
        "seed": seed,
        "initial_rate": float(rate),
        "phase": "rate_bracket",
        "epoch": 800,
        "model": copy_model_state(model),
        "optimizer": optimizer.state_dict(),
        "rng": copy.deepcopy(terminal["rng"]),
        "scheduler": scheduler.state_dict(),
        "best_state": best_state,
        "best_epoch": int(best_trace["epoch"]),
        "best_validation_bce": float(best_trace["validation"]["equal_world_bce"]),
        "history": [initial_record],
        "elapsed_seconds": time.time() - started_epoch,
        "source_bindings": {
            "terminal_checkpoint": sha256_file(source["terminal_path"]),
            "historical_selected_checkpoint": sha256_file(source["selected_path"]),
            "historical_trace": sha256_file(source["trace_path"]),
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    torch_dump(
        directory / "selected.pt",
        {"epoch": state["best_epoch"], "arm": arm, "seed": seed, "initial_rate": rate, "model": best_state},
    )
    torch_dump(latest_path, state)
    write_trace(directory / "trace.jsonl", state["history"])
    json_dump(
        directory / "initialization_check.json",
        {
            "arm": arm,
            "seed": seed,
            "initial_rate": rate,
            "terminal_epoch": 800,
            "terminal_metric_error_max": source["max_terminal_metric_error"],
            "historical_selected_state_error_max": source["max_selected_state_copy_error"],
            "historical_selected_epoch": state["best_epoch"],
            "training_bce_used_to_seed_scheduler": initial_train_bce,
            "scheduler_initial_state": scheduler_json_state(scheduler),
            "status": "PASS",
        },
    )
    return state


def train_one_epoch(helper, model, optimizer, worlds, arm):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for world in worlds:
        losses = []
        for view in world["views"]:
            target = torch.tensor(float(view["label"]), dtype=torch.float32)
            logit = model(*helper.args_for(helper, view, arm))
            if not torch.isfinite(logit).all():
                raise FloatingPointError(
                    f"nonfinite logit at {arm}/{world['world_id']}/{view['candidate_id']}"
                )
            losses.append(nn.functional.binary_cross_entropy_with_logits(logit, target))
        torch.stack(losses).mean().div(len(worlds)).backward()
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"nonfinite gradient at {arm}")
            squared_norm += float(torch.sum(parameter.grad.detach() ** 2).item())
    gradient_norm = math.sqrt(squared_norm)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError(f"nonfinite updated parameter at {arm}")
    return gradient_norm


def load_runtime(helper, state, widths, scheduler_spec):
    model = model_for(helper, state["arm"], widths)
    model.load_state_dict(state["model"])
    optimizer_spec = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(state["initial_rate"]),
        betas=optimizer_spec["betas"],
        eps=optimizer_spec["eps"],
        weight_decay=optimizer_spec["weight_decay"],
    )
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    scheduler = make_scheduler(optimizer, scheduler_spec)
    scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    return model, optimizer, scheduler


def evaluate_and_record(
    helper,
    model,
    optimizer,
    scheduler,
    state,
    worlds_train,
    worlds_val,
    phase,
    score_path,
    gradient_norms,
    started_epoch,
):
    arm = state["arm"]
    seed = int(state["seed"])
    rate = float(state["initial_rate"])
    before = float(optimizer.param_groups[0]["lr"])
    train_eval = helper.eval_worlds(helper, model, worlds_train, arm)
    val_eval = helper.eval_worlds(helper, model, worlds_val, arm)
    train_metrics = summarize_eval(helper, worlds_train, train_eval)
    val_metrics = summarize_eval(helper, worlds_val, val_eval)
    scheduler.step(train_metrics["equal_world_bce"])
    after = float(optimizer.param_groups[0]["lr"])
    record = {
        "epoch": int(state["epoch"]),
        "phase": phase,
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "learning_rate_before_scheduler": before,
        "learning_rate_after_scheduler": after,
        "scheduler_training_metric": train_metrics["equal_world_bce"],
        "gradient_norms_since_previous_evaluation": [float(x) for x in gradient_norms],
        "elapsed_seconds": time.time() - started_epoch,
    }
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if not score_path.exists():
        write_eval_artifact(
            score_path,
            state["trial_id"],
            state["epoch"],
            phase,
            arm,
            seed,
            rate,
            worlds_train,
            worlds_val,
            train_eval,
            val_eval,
            before,
            after,
            gradient_norms,
        )
    record["inference_seconds"] = {
        "training": {k: float(v) for k, v in train_eval["inference_seconds_by_world"].items()},
        "validation": {k: float(v) for k, v in val_eval["inference_seconds_by_world"].items()},
    }
    state["history"].append(record)
    if val_metrics["equal_world_bce"] < float(state["best_validation_bce"]):
        state["best_validation_bce"] = float(val_metrics["equal_world_bce"])
        state["best_epoch"] = int(state["epoch"])
        state["best_state"] = copy_model_state(model)
        if phase in ("rate_bracket", "stability_continuation"):
            torch_dump(
                score_path.parent.parent / "selected.pt",
                {
                    "epoch": state["best_epoch"],
                    "arm": arm,
                    "seed": seed,
                    "initial_rate": rate,
                    "model": state["best_state"],
                },
            )
    state["model"] = copy_model_state(model)
    state["optimizer"] = optimizer.state_dict()
    state["scheduler"] = scheduler.state_dict()
    state["rng"] = helper.rng_state()
    state["elapsed_seconds"] = time.time() - started_epoch
    write_trace(score_path.parent.parent / "trace.jsonl", state["history"])
    return record


def save_trial_state(path, state):
    torch_dump(path, state)


def run_trial_block(
    helper, root, state, directory, phase, block_target, train, val, widths, config, started_epoch
):
    latest_path = (
        directory / "latest.pt" if phase == "rate_bracket" else directory / "continuation" / "latest.pt"
    )
    if state["epoch"] >= block_target:
        return state
    model, optimizer, scheduler = load_runtime(helper, state, widths, config["scheduler"])
    load_rng_state(helper, state["rng"])
    pending_gradients = []
    pending_training_seconds = 0.0
    while int(state["epoch"]) < int(block_target):
        # Output bytes do not change between evaluations/checkpoint writes.
        enforce_elapsed(started_epoch)
        epoch_started = time.monotonic()
        gradient_norm = train_one_epoch(helper, model, optimizer, train, state["arm"])
        pending_training_seconds += time.monotonic() - epoch_started
        pending_gradients.append(gradient_norm)
        state["epoch"] += 1
        if state["epoch"] % 25 == 0:
            enforce_limits(root, started_epoch, reserve_bytes=256_000)
            if phase == "rate_bracket":
                score_path = directory / "scores" / f"epoch_{state['epoch']:04d}.json"
            else:
                score_path = directory / "continuation" / "scores" / f"epoch_{state['epoch']:04d}.json"
            evaluate_and_record(
                helper,
                model,
                optimizer,
                scheduler,
                state,
                train,
                val,
                phase,
                score_path,
                pending_gradients,
                started_epoch,
            )
            state["history"][-1]["training_seconds_since_previous_evaluation"] = pending_training_seconds
            pending_gradients = []
            pending_training_seconds = 0.0
            if phase == "stability_continuation":
                state["stability_status"] = stability_status(state["history"], state["stability_rule"])
            save_trial_state(latest_path, state)
            write_trace(
                (
                    (directory / "trace.jsonl")
                    if phase == "rate_bracket"
                    else (directory / "continuation" / "trace.jsonl")
                ),
                state["history"],
            )
            enforce_limits(root, started_epoch)
            if (
                phase == "stability_continuation"
                and state["stability_status"]["status"] == "SUSTAINED_LOCAL_STABILITY_OBSERVED"
            ):
                break
    if int(state["epoch"]) == block_target:
        save_trial_state(latest_path, state)
    return state


def stability_window(history, end_epoch, spec):
    begin = end_epoch - int(spec["window_epochs"])
    rows = {int(x["epoch"]): x for x in history}
    expected = list(range(begin, end_epoch + 1, 25))
    selected = [rows[x] for x in expected if x in rows]
    if len(selected) != len(expected) or not selected:
        return {"available": False, "stable": False, "end_epoch": end_epoch}
    losses = [float(x["training_metrics"]["equal_world_bce"]) for x in selected]
    relative_range = (max(losses) - min(losses)) / max(abs(float(np.mean(losses))), 1e-15)
    ap_ranges = {}
    for wid in VAL_IDS:
        values = [float(x["validation_metrics"]["per_world"][wid]["ap"]) for x in selected]
        ap_ranges[wid] = max(values) - min(values)
    rates = [
        float(x[key])
        for x in selected
        for key in ("learning_rate_before_scheduler", "learning_rate_after_scheduler")
    ]
    same_rate = all(x == rates[0] for x in rates)
    stable = (
        relative_range <= float(spec["training_BCE_relative_range_max"])
        and all(x <= float(spec["per_world_validation_AP_range_max"]) for x in ap_ranges.values())
        and same_rate
    )
    return {
        "available": True,
        "stable": bool(stable),
        "start_epoch": begin,
        "end_epoch": end_epoch,
        "training_BCE_relative_range": relative_range,
        "validation_AP_range_by_world": ap_ranges,
        "learning_rate": rates[-1],
        "learning_rate_unchanged": same_rate,
    }


def stability_status(history, spec):
    minimum = int(spec["minimum_epoch"])
    ends = sorted(
        {int(x["epoch"]) for x in history if int(x["epoch"]) >= minimum and int(x["epoch"]) % 25 == 0}
    )
    trigger_end = None
    trigger_rate = None
    confirmation = []
    for end in ends:
        current = stability_window(history, end, spec)
        if not current["available"]:
            trigger_end = None
            trigger_rate = None
            confirmation = []
            continue
        if trigger_end is not None:
            if not current["stable"] or current["learning_rate"] != trigger_rate:
                trigger_end = None
                trigger_rate = None
                confirmation = []
            else:
                confirmation.append(current)
                if end >= trigger_end + int(spec["confirmation_tail_epochs"]):
                    return {
                        "status": "SUSTAINED_LOCAL_STABILITY_OBSERVED",
                        "first_qualifying_pair_end_epoch": trigger_end,
                        "confirmation_end_epoch": end,
                        "confirmed_rolling_windows": confirmation,
                        "interpretation": "finite observed local stability only; not proof of a global optimum",
                    }
        if trigger_end is None and end >= minimum and current["stable"]:
            previous = stability_window(history, end - int(spec["window_epochs"]), spec)
            if (
                previous["available"]
                and previous["stable"]
                and previous["learning_rate"] == current["learning_rate"]
            ):
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


def compare_reference_to_record(record, split, world_id, reference):
    observed = record[f"{split}_metrics"]["per_world"][world_id]
    gaps = {
        "bce": abs(float(observed["bce"]) - float(reference["binary_cross_entropy_logits"])),
        "ap": abs(float(observed["ap"]) - float(reference["average_precision_grouped_threshold"])),
        "auc": abs(float(observed["auc"]) - float(reference["auroc_half_tie"])),
    }
    return gaps


def verify_score_artifacts(root, populations):
    files = sorted(root.glob("branches/**/scores/epoch_*.json")) + sorted(
        root.glob("branches/**/selected_checkpoint_predictions.json")
    )
    checks = []
    seen = set()
    for path in files:
        doc = read_json(path)
        rows = doc.get("predictions", [])
        if len(rows) != 468:
            raise ValueError(f"retained evaluation does not contain the full 468 candidate census: {path}")
        groups = defaultdict(list)
        for row in rows:
            key = (
                doc["trial_id"],
                doc["phase"],
                int(doc["epoch"]),
                row["split"],
                row["world_id"],
                row["candidate_id"],
            )
            if key in seen:
                raise ValueError(f"duplicate retained prediction key: {key}")
            seen.add(key)
            expected = populations[(row["split"], row["world_id"])].get(row["candidate_id"])
            if (
                expected is None
                or expected["label"] != int(row["label"])
                or expected["instance_id"] != row["instance_id"]
            ):
                raise ValueError(f"prediction custody join mismatch: {key}")
            if row["split"] != ("training" if row["world_id"] in TRAIN_IDS else "validation"):
                raise ValueError(f"world role mismatch in retained score: {key}")
            groups[(row["split"], row["world_id"])].append(row)
        for split, worlds in (("training", TRAIN_IDS), ("validation", VAL_IDS)):
            if any((split, wid) not in groups or len(groups[(split, wid)]) != 78 for wid in worlds):
                raise ValueError(f"evaluation file lacks one of the six complete worlds: {path}")
        max_gap = 0.0
        for split, world_ids in (("training", TRAIN_IDS), ("validation", VAL_IDS)):
            refs = []
            for wid in world_ids:
                rows_world = groups[(split, wid)]
                reference = ref_metrics([x["label"] for x in rows_world], [x["logit"] for x in rows_world])
                gaps = compare_reference_to_record(doc, split, wid, reference)
                max_gap = max(max_gap, *gaps.values())
                refs.append(reference)
            for key, refkey in (
                ("equal_world_bce", "binary_cross_entropy_logits"),
                ("equal_world_ap", "average_precision_grouped_threshold"),
                ("equal_world_auc", "auroc_half_tie"),
            ):
                refmean = float(np.mean([x[refkey] for x in refs]))
                gap = abs(float(doc[f"{split}_metrics"][key]) - refmean)
                max_gap = max(max_gap, gap)
        if max_gap > 1e-10:
            raise ValueError(f"independent metric recomputation mismatch in {path}: {max_gap}")
        checks.append(
            {"path": str(path), "rows": len(rows), "max_metric_difference": max_gap, "status": "PASS"}
        )
    return {
        "status": "PASS",
        "evaluation_files": len(files),
        "prediction_rows": sum(x["rows"] for x in checks),
        "checks": checks,
    }


def native_diagnostic(unlabeled, joined, helper, out):
    result = {
        "calculation": "score=1 iff max(native persisted selected-event integer step)-min(...) equals 29; labels not read until scores for all six worlds were fixed",
        "score_rows": [],
        "world_summary": {},
        "false_matches": [],
    }
    for wid in ALL_IDS:
        rows = joined[wid]
        labels = [x["label"] for x in rows]
        scores = [x["score"] for x in rows]
        metric = ref_metrics(labels, scores)
        positives = sum(labels)
        members = [x for x in rows if x["score"] == 1]
        false_matches = [x for x in members if x["label"] == 0]
        instance_ids = sorted(
            {x["instance_id"] for x in rows if x["label"] == 1 and x["instance_id"] is not None}
        )
        result["world_summary"][wid] = {
            "candidate_count": len(rows),
            "positive_count": positives,
            "negative_count": len(rows) - positives,
            "span_29_members": len(members),
            "span_29_positive_members": sum(x["label"] for x in members),
            "span_29_false_matches": len(false_matches),
            "positive_instance_count": len(instance_ids),
            "positive_instance_ids": instance_ids,
            "AP": metric["average_precision_grouped_threshold"],
            "AUROC_half_tie": metric["auroc_half_tie"],
        }
        result["score_rows"].extend(rows)
        result["false_matches"].extend(false_matches)
    if sum(len(x) for x in joined.values()) != 468 or any(
        x["candidate_count"] != 78 for x in result["world_summary"].values()
    ):
        raise ValueError("native span diagnostic did not cover the complete six-world frame")
    json_dump(out, result)
    return result


def load_baseline(prior_root, populations, out):
    source = prior_root / "joined_predictions.csv"
    rows = []
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["arm"] == "gbdt_fixed20"
                and row["checkpoint_stage"] == "400_selected"
                and row["selected_setting"] == "7"
            ):
                rows.append(row)
    if len(rows) != 5 * 6 * 78:
        raise ValueError(f"retained fixed20 scores do not have the expected 2340 rows: {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        split = row["split"]
        wid = row["world_id"]
        expected = populations[(split, wid)].get(row["candidate_id"])
        if (
            expected is None
            or int(row["label"]) != expected["label"]
            or row["instance_id"] != ("" if expected["instance_id"] is None else str(expected["instance_id"]))
        ):
            raise ValueError(f"retained fixed20 score join mismatch for {wid}/{row['candidate_id']}")
        grouped[(int(row["seed"]), split, wid)].append(row)
    for key, values in grouped.items():
        if len(values) != 78:
            raise ValueError(f"retained fixed20 group incomplete: {key}")
    jsonl_write(out, rows)
    states = []
    for seed in SEEDS:
        d = prior_root / "gbdt" / f"seed_{seed}" / "leaves_7"
        complete = read_json(d / "complete.json")
        if complete.get("max_leaf_nodes") != 7 or complete.get("best_iteration") != 70:
            raise ValueError(f"retained fixed20 fit metadata changed for seed {seed}")
        states.append(
            {
                "seed": seed,
                "selected_iteration": complete["best_iteration"],
                "model_state": sha256_file(d / "model400.pkl"),
                "score_trace": sha256_file(d / "trace.jsonl"),
            }
        )
    summary = {
        "status": "REUSED",
        "source_score_file": sha256_file(source),
        "score_rows": len(rows),
        "seeds": SEEDS,
        "states": states,
        "no_fit_or_refit": True,
    }
    json_dump(out.parent / "fixed20_baseline_reference.json", summary)
    return rows, summary


def exact_boundary_checks(helper, worlds):
    checked = 0
    for world in worlds:
        data = world["data"]
        n = len(data["node_order"])
        e = len(data["event_order"])
        for view in world["views"]:
            b1, b2 = helper.cellular(view["r"], n, e)
            sb = helper.simplicial(view["r"], n, e)
            if np.any(b1.astype(np.int64) @ b2.astype(np.int64) != 0):
                raise ValueError(
                    f"exact cellular B1@B2 failure at {world['world_id']}/{view['candidate_id']}"
                )
            if np.any(sb[3].astype(np.int64) @ sb[4].astype(np.int64) != 0):
                raise ValueError(
                    f"exact simplicial boundary failure at {world['world_id']}/{view['candidate_id']}"
                )
            checked += 1
    return {
        "status": "PASS",
        "candidate_rows_checked": checked,
        "identity_and_mask_path": "unchanged S1 representation builder",
        "cellular_B1_B2": "exact integer zero",
        "simplicial_boundary": "exact integer zero",
    }


def populations_from_worlds(worlds):
    populations = defaultdict(dict)
    for world in worlds:
        split = "training" if world["world_id"] in TRAIN_IDS else "validation"
        for view in world["views"]:
            populations[(split, world["world_id"])][view["candidate_id"]] = {
                "label": int(view["label"]),
                "instance_id": view["instance_id"],
            }
    return populations


def test_world_populations(worlds):
    rows = []
    for world in worlds:
        ids = {x["instance_id"] for x in world["views"] if x["label"] == 1 and x["instance_id"] is not None}
        rows.append(
            {
                "world_id": world["world_id"],
                "role": world["role"],
                "candidate_count": len(world["views"]),
                "positive_count": sum(x["label"] for x in world["views"]),
                "negative_count": len(world["views"]) - sum(x["label"] for x in world["views"]),
                "positive_instance_count": len(ids),
                "positive_instance_ids": sorted(ids),
            }
        )
    return rows


def persist_worlds(helper, entries):
    train = [helper.strict_inputs(helper, wid, entries[wid], "training") for wid in TRAIN_IDS]
    val = [helper.strict_inputs(helper, wid, entries[wid], "validation") for wid in VAL_IDS]
    worlds = train + val
    if set(x["world_id"] for x in train) & set(x["world_id"] for x in val):
        raise ValueError("training and validation worlds are not disjoint")
    if any(len(x["views"]) != 78 for x in worlds):
        raise ValueError("an authorized world does not retain all 78 candidates")
    return train, val, worlds


def build_manifest(helper, root, args, config, entries, worlds, prior_root):
    files = [
        args.world_index.resolve(),
        args.study_config.resolve(),
        Path(config["model_config"]).resolve(),
        Path(__file__).resolve(),
        Path(helper.__file__).resolve(),
        helper.benchmark_source,
        prior_root / "model_config.json",
        prior_root / "final_report.json",
        prior_root / "joined_predictions.csv",
    ]
    for world in worlds:
        files.extend(world["data"]["inputs"])
        attempt = entries[world["world_id"]]
        files.extend([attempt / "candidate_labels.jsonl", attempt / "sidecars" / "persisted_events.jsonl"])
    for arm in ARMS:
        for seed in SEEDS:
            selected_rate = float(read_json(prior_root / "final_report.json")["selected_learning_rates"][arm])
            prior = prior_root / "neural" / arm / f"seed_{seed}" / f"lr_{selected_rate:g}" / "extension"
            files.extend([prior / "latest.pt", prior / "selected_extended.pt", prior / "trace.jsonl"])
    unique = sorted({Path(p).resolve() for p in files}, key=str)
    for path in unique:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"manifest input is missing or symlinked: {path}")
    return {
        "experiment": "S1-E",
        "command": [
            sys.executable,
            "-s",
            "-B",
            str(Path(__file__).resolve()),
            "--world-index",
            str(args.world_index),
            "--prior-run",
            str(args.prior_run),
            "--study-config",
            str(args.study_config),
            "--output-dir",
            str(root),
        ],
        "cwd": str(Path.cwd()),
        "started_utc_epoch": time.time(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MPLCONFIGDIR",
                "XDG_CACHE_HOME",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
        },
        "training_worlds": TRAIN_IDS,
        "validation_worlds": VAL_IDS,
        "excluded_worlds": config["excluded_worlds"],
        "input_hashes": [sha256_file(p) for p in unique],
        "limits": {"wall_seconds": MAX_SECONDS, "output_bytes": MAX_BYTES},
        "run_type": "existing-world exploratory learning continuation; no population inference",
    }


def compare_json_values(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(compare_json_values(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(compare_json_values(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return a == b


def generate_figures(root, selection, baseline_rows, native_result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = {}
    for path in root.glob("branches/*/seed_*/rate_*/trace.jsonl"):
        parts = path.parts
        arm = parts[-4]
        seed = int(parts[-3].split("_")[1])
        rate = float(parts[-2].split("_")[1])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        cont_path = path.parent / "continuation" / "trace.jsonl"
        if cont_path.is_file():
            records = [
                json.loads(line) for line in cont_path.read_text(encoding="utf-8").splitlines() if line
            ]
        traces[(arm, seed, rate)] = records
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    for row, arm in enumerate(ARMS):
        for col, rate in enumerate(RATES):
            ax = axes[row, col]
            for seed in SEEDS:
                series = traces.get((arm, seed, rate), [])
                x = [int(r["epoch"]) for r in series]
                y = [float(r["training_metrics"]["equal_world_bce"]) for r in series]
                if x:
                    ax.plot(x, y, linewidth=0.8, alpha=0.65)
            ax.set_title(f"{arm} — initial rate {rate:g}", fontsize=8)
            ax.set_ylabel("Training BCE")
            ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    fig.suptitle("Training loss trajectories on existing development worlds")
    fig.tight_layout()
    fig.savefig(root / "training_loss_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    for row, arm in enumerate(ARMS):
        for col, rate in enumerate(RATES):
            ax = axes[row, col]
            for seed in SEEDS:
                series = traces.get((arm, seed, rate), [])
                x = [int(r["epoch"]) for r in series]
                y = [float(r["validation_metrics"]["equal_world_bce"]) for r in series]
                if x:
                    ax.plot(x, y, linewidth=0.8, alpha=0.65)
            ax.set_title(f"{arm} — initial rate {rate:g}", fontsize=8)
            ax.set_ylabel("Validation BCE")
            ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    fig.suptitle("Validation loss trajectories; checkpoint selection uses equal-world BCE")
    fig.tight_layout()
    fig.savefig(root / "validation_loss_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for arm in ARMS:
        for rate in RATES:
            series = []
            for seed in SEEDS:
                rows = traces.get((arm, seed, rate), [])
                series.extend([(int(x["epoch"]), float(x["learning_rate_after_scheduler"])) for x in rows])
            if series:
                by_epoch = defaultdict(list)
                for epoch, lr in series:
                    by_epoch[epoch].append(lr)
                xs = sorted(by_epoch)
                ax.plot(xs, [float(np.mean(by_epoch[x])) for x in xs], label=f"{arm}, initial {rate:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate after scheduler")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Training-loss-driven learning-rate schedules")
    fig.tight_layout()
    fig.savefig(root / "learning_rate_curves.png", dpi=140)
    plt.close(fig)

    selected_path = root / "selected_checkpoint_metrics.csv"
    if selected_path.is_file():
        buckets = defaultdict(list)
        with selected_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] == "validation":
                    buckets[row["arm"], row["world_id"]].append(
                        float(row["average_precision_grouped_threshold"])
                    )
        for row in baseline_rows:
            if row["split"] == "validation":
                buckets["gbdt_fixed20", row["world_id"]].append(row)
        for wid in VAL_IDS:
            native = native_result["world_summary"][wid]
            buckets["native_span_equals29", wid] = [float(native["AP"])]
            tree = buckets["gbdt_fixed20", wid]
            by_seed = defaultdict(list)
            for row in tree:
                by_seed[int(row["seed"])].append(row)
            buckets["gbdt_fixed20", wid] = [
                ref_metrics([int(x["label"]) for x in rows], [float(x["logit"]) for x in rows])[
                    "average_precision_grouped_threshold"
                ]
                for rows in by_seed.values()
            ]
        arms = ARMS + ("gbdt_fixed20", "native_span_equals29")
        fig, ax = plt.subplots(figsize=(11, 5))
        for wid, offset in zip(VAL_IDS, (-0.10, 0.10)):
            for index, arm in enumerate(arms):
                values = buckets[arm, wid]
                ax.scatter(
                    [index + offset] * len(values),
                    values,
                    s=20,
                    alpha=0.75,
                    label=wid if index == 0 else None,
                )
        ax.set_xticks(
            range(len(arms)), [x.replace("_", " ") for x in arms], rotation=25, ha="right", fontsize=7
        )
        ax.set_ylabel("Development validation AP")
        ax.set_ylim(0, 1.03)
        ax.set_title("Selected development checkpoints, retained GBDT and post-hoc native-span ranking")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "validation_comparison.png", dpi=140)
        plt.close(fig)

    metrics = []
    for p in sorted(root.glob("branches/**/scores/epoch_1600.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        for wid, v in doc["validation_metrics"]["per_world"].items():
            metrics.append((doc["arm"], float(doc["initial_rate"]), wid, float(v["ap"])))
    if metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = {(arm, rate): i for i, (arm, rate) in enumerate((a, r) for a in ARMS for r in RATES)}
        for arm, rate, wid, ap in metrics:
            offset = -0.08 if wid == VAL_IDS[0] else 0.08
            ax.scatter(
                positions[(arm, rate)] + offset,
                ap,
                s=18,
                alpha=0.8,
                label=wid if (arm, rate) == (ARMS[0], RATES[0]) else None,
            )
        ax.set_xticks(
            range(len(positions)),
            [f"{a.replace('_', ' ')}\n{r:g}" for a, r in positions],
            rotation=25,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Development validation AP")
        ax.set_ylim(0, 1.03)
        ax.set_title("Per-world scores at epoch 1600 (training initialization repeats)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "epoch1600_validation_scores.png", dpi=140)
        plt.close(fig)


def summarize_endpoint(root, selection, branch_states, continuation_states, native, baseline, started_epoch):
    endpoint_rows = []
    stability = {}
    for arm in ARMS:
        selected_rate = selection.get(arm, {}).get("selected_initial_rate")
        stability[arm] = {}
        for seed in SEEDS:
            key = f"{arm}/seed_{seed}/rate_{float(selected_rate):.2f}" if selected_rate is not None else None
            state = continuation_states.get(key) if key else None
            if state is None:
                stability[arm][str(seed)] = {"status": "NOT_RUN"}
                continue
            status = stability_status(state["history"], state["stability_rule"])
            stability[arm][str(seed)] = {
                **status,
                "last_epoch": int(state["epoch"]),
                "last_learning_rate": float(state["optimizer"]["param_groups"][0]["lr"]),
            }
            last = state["history"][-1]
            for split, values in (
                ("training", last["training_metrics"]),
                ("validation", last["validation_metrics"]),
            ):
                for wid, m in values["per_world"].items():
                    endpoint_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "selected_initial_rate": selected_rate,
                            "last_epoch": state["epoch"],
                            "stability_status": status["status"],
                            "split": split,
                            "world_id": wid,
                            "bce": m["bce"],
                            "ap": m["ap"],
                            "auc": m["auc"],
                        }
                    )
    return {
        "stability": stability,
        "endpoint_metrics": endpoint_rows,
        "elapsed_seconds": time.time() - started_epoch,
        "output_bytes": output_size(root),
        "native_span": native["world_summary"],
        "fixed20_baseline_reused": baseline,
    }


def write_endpoint_report(root, report):
    lines = [
        "# S1-E endpoint report",
        "",
        f"Status: {report['status']}",
        "",
        "This is an exploratory analysis of the existing development worlds and complete candidate rows. It is not a population, financial-detection, or topology-mechanism result.",
        "",
        "## Optimization policy",
        "",
        "| Architecture | Selected initial rate | Mean best validation BCE at epoch 1600 | Seeds reaching local-stability endpoint |",
        "|---|---:|---:|---:|",
    ]
    for arm, info in report.get("policy_selection", {}).items():
        st = report.get("stability", {}).get(arm, {})
        stable_n = sum(x.get("status") == "SUSTAINED_LOCAL_STABILITY_OBSERVED" for x in st.values())
        lines.append(
            f"| {arm} | {info.get('selected_initial_rate', 'not selected')} | {info.get('mean_best_validation_bce_by_rate', {}).get(str(info.get('selected_initial_rate')), 'incomplete')} | {stable_n}/5 |"
        )
    lines.extend(
        [
            "",
            "## Native integer-span ranking",
            "",
            "| World | Candidates | Positive / negative | Score-1 members | False matches | AP | Half-tie AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for wid, row in report.get("native_span", {}).items():
        lines.append(
            f"| {wid} | {row['candidate_count']} | {row['positive_count']} / {row['negative_count']} | {row['span_29_members']} | {row['span_29_false_matches']} | {row['AP']:.4f} | {row['AUROC_half_tie']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The ranking was calculated from selected physical-event native integer steps before annotation labels were opened. Every false match and its physical event IDs and steps are retained in `native_span_diagnostic.json`. Scores are binary rankings, not probabilities. The inspected native cycle schedule fixes both period endpoints; this observed construction cue does not establish leakage or a causal mechanism.",
            "",
            "## Limits and disposition",
            "",
            "S1-D fixed20 GBDT scores and fitted-state hashes were reused without refitting; identical deterministic seed outcomes remain disclosed in `fixed20_baseline_reference.json`. All score tables are descriptive on the same six fixed worlds. No population intervals or independent-world claims are supported. The stability result is a finite local diagnostic, not proof of a global optimum. GUDHI external parity is unavailable. No new worlds, test access, main benchmark, final split, freeze, or certification were used.",
            "",
            f"Elapsed wall time: {report.get('elapsed_seconds', 0):.1f} seconds. New output bytes: {report.get('output_bytes', 0)}.",
            "",
        ]
    )
    (root / "endpoint_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global ACTIVE_OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-index", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    ACTIVE_OUTPUT_ROOT = root
    root.mkdir(parents=True, exist_ok=True)
    start_path = root / "execution_start.json"
    if start_path.exists():
        started_epoch = float(read_json(start_path)["started_utc_epoch"])
    else:
        started_epoch = time.time()
        json_dump(
            start_path,
            {
                "started_utc_epoch": started_epoch,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch)),
                "wall_limit_seconds": MAX_SECONDS,
                "cumulative_across_resume": True,
            },
        )
    if (root / "final_report.json").exists():
        print(json.dumps(read_json(root / "final_report.json"), indent=2))
        return 0
    helper = load_helper(Path(__file__).with_name("train_s1_development.py"))
    config = helper.read_json(args.study_config)
    required = {
        "experiment_id": "S1-E",
        "branch_start_epoch": 800,
        "branch_end_epoch": 1600,
        "evaluate_every_epochs": 25,
        "execution_wall_time_ceiling_seconds": MAX_SECONDS,
        "maximum_new_output_bytes": MAX_BYTES,
        "maximum_total_epochs_per_selected_trial": 4000,
        "automatic_extension_authorized": False,
        "new_world_generation_authorized": False,
        "test_seed_draw_authorized": False,
        "main_benchmark_authorized": False,
        "final_split_authorized": False,
        "freeze_authorized": False,
        "certification_authorized": False,
    }
    for key, value in required.items():
        if type(config.get(key)) is not type(value) or config.get(key) != value:
            raise ValueError(f"study configuration mismatch for {key}")
    if (
        tuple(config["arms"]) != ARMS
        or tuple(config["initialization_seeds"]) != SEEDS
        or tuple(config["branch_initial_learning_rates"]) != RATES
    ):
        raise ValueError("S1-E arm, seed or rate bracket differs from the approved configuration")
    if tuple(config["training_worlds"]) != TRAIN_IDS or tuple(config["validation_worlds"]) != VAL_IDS:
        raise ValueError("S1-E world allowlist differs from the approved configuration")
    if config.get("candidate_sampling") != "none; complete existing 78-row census per world":
        raise ValueError("candidate sampling is not the complete existing census")
    prior_root = args.prior_run.resolve()
    prior_report = helper.read_json(prior_root / "final_report.json")
    prior_report["root"] = prior_root
    if prior_report.get("status") != "COMPLETED" or prior_report.get("neural_trials") != 60:
        raise ValueError("S1-D reference run is not the accepted complete development result")
    if set(prior_report.get("selected_learning_rates", {})) != set(ARMS):
        raise ValueError("S1-D selected-rate map does not cover the four architectures")
    entries = load_index_entries(args.world_index.resolve())

    # Calculate the native physical-event score for all worlds before reading any labels.
    native_unlabeled = native_span_rows(entries, helper)
    native_joined = join_native_labels(native_unlabeled, entries, helper)
    native_file = root / "native_span_diagnostic.json"
    native_result = native_diagnostic(native_unlabeled, native_joined, helper, native_file)

    train, val, worlds = persist_worlds(helper, entries)
    torch.set_num_threads(1)
    boundary_check = exact_boundary_checks(helper, worlds)
    population_rows = test_world_populations(worlds)
    if any(x["candidate_count"] != 78 or x["positive_count"] != 6 for x in population_rows):
        raise ValueError("registered development population counts differ from the accepted six-world census")
    populations = populations_from_worlds(worlds)
    model_config = helper.read_json(Path(config["model_config"]))
    prior_model_config = helper.read_json(prior_root / "model_config.json")
    if not compare_json_values(model_config, prior_model_config):
        raise ValueError("S1-E model configuration differs from the accepted S1-D run")
    widths = model_config["hidden_widths"]
    fixture = scheduler_fixture(config)
    json_dump(
        root / "engineering_qualification.json",
        {
            "scheduler_fixture": fixture,
            "exact_boundary_checks": boundary_check,
            "world_populations": population_rows,
        },
    )

    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = helper.read_json(run_manifest_path)
        if manifest.get("experiment") != "S1-E":
            raise ValueError("existing output directory belongs to another experiment")
        for binding in manifest["input_hashes"]:
            if sha256_file(Path(binding["path"])) != binding:
                raise ValueError(f"input changed since this continuation began: {binding['path']}")
    else:
        manifest = build_manifest(helper, root, args, config, entries, worlds, prior_root)
        json_dump(run_manifest_path, manifest)
    enforce_limits(root, started_epoch)

    # Reuse the existing fitted fixed20 states and score rows; no model is fitted here.
    baseline_rows, baseline_summary = load_baseline(
        prior_root, populations, root / "baseline" / "fixed20_selected_predictions.jsonl"
    )
    json_dump(root / "fixed20_baseline_reference.json", baseline_summary)

    widths_config = model_config["hidden_widths"]
    starting_checks = {}
    sources = {}
    for arm in ARMS:
        for seed in SEEDS:
            source = terminal_reconstruction(helper, arm, seed, prior_report, train, val, widths_config)
            sources[(arm, seed)] = source
            starting_checks[f"{arm}/seed_{seed}"] = {
                "terminal_checkpoint": sha256_file(source["terminal_path"]),
                "selected_checkpoint": sha256_file(source["selected_path"]),
                "trace": sha256_file(source["trace_path"]),
                "epoch": 800,
                "max_epoch800_metric_error": source["max_terminal_metric_error"],
                "max_epoch800_logit_error": source["max_terminal_logit_error"],
                "max_historical_best_state_difference": source["max_selected_state_copy_error"],
                "status": "PASS",
            }
    json_dump(root / "starting_state_checks.json", starting_checks)

    trial_order = [(arm, seed, rate) for arm in ARMS for seed in SEEDS for rate in RATES]
    progress_path = root / "branch_round_robin.json"
    progress = (
        helper.read_json(progress_path)
        if progress_path.exists()
        else {"next_round": 0, "block_epochs": 100, "stage": "rate_bracket"}
    )
    try:
        for round_number in range(int(progress.get("next_round", 0)), 8):
            block_target = 800 + (round_number + 1) * 100
            for arm, seed, rate in trial_order:
                directory = branch_directory(root, arm, seed, rate)
                state = branch_initialization(
                    helper,
                    root,
                    arm,
                    seed,
                    rate,
                    sources[(arm, seed)],
                    train,
                    val,
                    widths_config,
                    config,
                    started_epoch,
                )
                state["trial_id"] = f"{arm}/seed_{seed}/rate_{rate:.2f}"
                if state["epoch"] < block_target:
                    state = run_trial_block(
                        helper,
                        root,
                        state,
                        directory,
                        "rate_bracket",
                        block_target,
                        train,
                        val,
                        widths_config,
                        config,
                        started_epoch,
                    )
                if state["epoch"] >= 1600:
                    json_dump(
                        directory / "complete.json",
                        {
                            "arm": arm,
                            "seed": seed,
                            "initial_rate": rate,
                            "terminal_epoch": state["epoch"],
                            "best_epoch_0_to_1600": state["best_epoch"],
                            "best_validation_bce": state["best_validation_bce"],
                            "status": "COMPLETE",
                        },
                    )
            progress["next_round"] = round_number + 1
            json_dump(progress_path, progress)
            enforce_limits(root, started_epoch)
    except BoundedStop as stop:
        return bounded_report(
            root,
            stop,
            started_epoch,
            native_result,
            baseline_summary,
            boundary_check,
            population_rows,
            progress,
        )

    branch_states = {}
    for arm, seed, rate in trial_order:
        directory = branch_directory(root, arm, seed, rate)
        state = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
        if int(state["epoch"]) != 1600:
            raise ValueError(f"rate branch did not reach epoch1600: {arm}/{seed}/{rate}")
        branch_states[f"{arm}/seed_{seed}/rate_{rate:.2f}"] = state
    policy_selection = {}
    for arm in ARMS:
        means = {}
        per_seed = {}
        for rate in RATES:
            values = [
                float(branch_states[f"{arm}/seed_{seed}/rate_{rate:.2f}"]["best_validation_bce"])
                for seed in SEEDS
            ]
            means[str(rate)] = float(np.mean(values))
            per_seed[str(rate)] = {
                str(seed): float(branch_states[f"{arm}/seed_{seed}/rate_{rate:.2f}"]["best_validation_bce"])
                for seed in SEEDS
            }
        selected_rate = min(RATES, key=lambda rate: (means[str(rate)], rate))
        policy_selection[arm] = {
            "selected_initial_rate": selected_rate,
            "mean_best_validation_bce_by_rate": means,
            "per_seed_best_validation_bce_by_rate": per_seed,
            "selection_rule": "lowest mean best equal-world validation BCE across five seeds; ties lower rate",
        }
    json_dump(root / "policy_selection.json", policy_selection)

    continuation_dir = root / "continuations"
    cont_progress_path = root / "continuation_round_robin.json"
    cont_progress = (
        helper.read_json(cont_progress_path)
        if cont_progress_path.exists()
        else {"next_round": 0, "block_epochs": 100, "stage": "selected_policy_continuation"}
    )
    continuation_states = {}
    selected_trials = [
        (arm, seed, float(policy_selection[arm]["selected_initial_rate"])) for arm in ARMS for seed in SEEDS
    ]
    try:
        for round_number in range(int(cont_progress.get("next_round", 0)), 24):
            block_target = 1600 + (round_number + 1) * 100
            for arm, seed, rate in selected_trials:
                branch_key = f"{arm}/seed_{seed}/rate_{rate:.2f}"
                directory = branch_directory(root, arm, seed, rate)
                cont_root = directory / "continuation"
                cont_latest = cont_root / "latest.pt"
                if cont_latest.exists():
                    state = torch.load(cont_latest, map_location="cpu", weights_only=False)
                else:
                    state = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
                    state = copy.deepcopy(state)
                    state["phase"] = "stability_continuation"
                    state["trial_id"] = branch_key
                    state["stability_rule"] = config["stability_rule"]
                    state["stability_status"] = stability_status(state["history"], config["stability_rule"])
                    cont_root.mkdir(parents=True, exist_ok=True)
                    torch_dump(cont_latest, state)
                    torch_dump(
                        cont_root / "selected.pt",
                        {
                            "epoch": state["best_epoch"],
                            "arm": arm,
                            "seed": seed,
                            "initial_rate": rate,
                            "model": state["best_state"],
                        },
                    )
                    write_trace(cont_root / "trace.jsonl", state["history"])
                if (
                    state["stability_status"]["status"] == "SUSTAINED_LOCAL_STABILITY_OBSERVED"
                    or int(state["epoch"]) >= 4000
                ):
                    state["complete_reason"] = (
                        state["stability_status"]["status"] if state["epoch"] < 4000 else "4000_EPOCH_CEILING"
                    )
                    json_dump(
                        cont_root / "complete.json",
                        {
                            "arm": arm,
                            "seed": seed,
                            "initial_rate": rate,
                            "last_epoch": state["epoch"],
                            "status": state["stability_status"]["status"],
                            "reason": state["complete_reason"],
                        },
                    )
                    continuation_states[branch_key] = state
                    continue
                target = min(block_target, 4000)
                if int(state["epoch"]) < target:
                    state["stability_rule"] = config["stability_rule"]
                    state = run_trial_block(
                        helper,
                        root,
                        state,
                        directory,
                        "stability_continuation",
                        target,
                        train,
                        val,
                        widths_config,
                        config,
                        started_epoch,
                    )
                    state["stability_status"] = stability_status(state["history"], config["stability_rule"])
                    torch_dump(cont_latest, state)
                    write_trace(cont_root / "trace.jsonl", state["history"])
                if (
                    state["stability_status"]["status"] == "SUSTAINED_LOCAL_STABILITY_OBSERVED"
                    or int(state["epoch"]) >= 4000
                ):
                    state["complete_reason"] = (
                        state["stability_status"]["status"] if state["epoch"] < 4000 else "4000_EPOCH_CEILING"
                    )
                    json_dump(
                        cont_root / "complete.json",
                        {
                            "arm": arm,
                            "seed": seed,
                            "initial_rate": rate,
                            "last_epoch": state["epoch"],
                            "status": state["stability_status"]["status"],
                            "reason": state["complete_reason"],
                        },
                    )
                    continuation_states[branch_key] = state
            cont_progress["next_round"] = round_number + 1
            json_dump(cont_progress_path, cont_progress)
            enforce_limits(root, started_epoch)
    except BoundedStop as stop:
        return bounded_report(
            root,
            stop,
            started_epoch,
            native_result,
            baseline_summary,
            boundary_check,
            population_rows,
            cont_progress,
            policy_selection,
        )

    # Reload every selected checkpoint and retain its actual scores and coordinate checks.
    selected_checks = {}
    selected_prediction_rows = []
    for arm, seed, rate in selected_trials:
        branch_key = f"{arm}/seed_{seed}/rate_{rate:.2f}"
        directory = branch_directory(root, arm, seed, rate)
        enforce_limits(root, started_epoch, processing=False)
        selected = load_selected_continuation(directory)
        model = model_for(helper, arm, widths_config)
        model.load_state_dict(selected["model"])
        model.eval()
        train_eval = helper.eval_worlds(helper, model, train, arm)
        val_eval = helper.eval_worlds(helper, model, val, arm)
        selected_epoch = int(selected["epoch"])
        if selected_epoch <= 800:
            historical = next(x for x in sources[(arm, seed)]["trace"] if x["epoch"] == selected_epoch)
            expected = {
                split: {wid: historical[key][wid]["logits"] for wid in world_ids}
                for split, key, world_ids in (
                    ("training", "train", TRAIN_IDS),
                    ("validation", "validation", VAL_IDS),
                )
            }
        else:
            score_dir = directory / ("continuation/scores" if selected_epoch > 1600 else "scores")
            saved = read_json(score_dir / f"epoch_{selected_epoch:04d}.json")
            by_key = {
                (x["split"], x["world_id"], x["candidate_id"]): x["logit"] for x in saved["predictions"]
            }
            expected = {
                split: {
                    w["world_id"]: [by_key[split, w["world_id"], v["candidate_id"]] for v in w["views"]]
                    for w in ws
                }
                for split, ws in (("training", train), ("validation", val))
            }
        reload_error = max(
            abs(a - b)
            for split, ws, evaluated in (("training", train, train_eval), ("validation", val, val_eval))
            for w in ws
            for a, b in zip(evaluated[w["world_id"]]["logits"], expected[split][w["world_id"]])
        )
        if reload_error > 1e-7:
            raise ValueError(f"selected checkpoint logits differ from retained score trace: {branch_key}")
        checks = helper.score_qualification(helper, model, worlds, arm)
        if not all(x["pass"] for x in checks.values()):
            raise ValueError(f"selected checkpoint coordinate/basis check failed: {branch_key}")
        selected_checks[branch_key] = {
            "selected_epoch": selected["epoch"],
            "maximum_reload_logit_error": reload_error,
            "coordinate_basis_checks": checks,
            "all_pass": True,
        }
        score_doc = write_eval_artifact(
            directory / "selected_checkpoint_predictions.json",
            branch_key,
            int(selected["epoch"]),
            "validation_selected_checkpoint",
            arm,
            seed,
            rate,
            train,
            val,
            train_eval,
            val_eval,
            rate,
            rate,
            [],
        )
        selected_prediction_rows.extend(score_doc["predictions"])
    json_dump(root / "selected_checkpoint_checks.json", selected_checks)
    with (root / "selected_checkpoint_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "arm",
            "seed",
            "initial_rate",
            "phase",
            "epoch",
            "split",
            "world_id",
            "candidate_id",
            "instance_id",
            "label",
            "logit",
            "probability",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected_prediction_rows:
            writer.writerow({k: row[k] for k in fields})

    populations_by_split = {}
    for world in worlds:
        split = "training" if world["world_id"] in TRAIN_IDS else "validation"
        populations_by_split[(split, world["world_id"])] = {
            v["candidate_id"]: {"label": v["label"], "instance_id": v["instance_id"]} for v in world["views"]
        }
    score_verification = verify_score_artifacts(root, populations_by_split)
    selected_references = []
    for row in selected_prediction_rows:
        selected_references.append(row)
    selected_metrics = defaultdict(list)
    for row in selected_prediction_rows:
        selected_metrics[(row["arm"], row["seed"], row["phase"], row["split"], row["world_id"])].append(row)
    selected_reference_rows = []
    for key, values in sorted(selected_metrics.items()):
        metric = ref_metrics([x["label"] for x in values], [x["logit"] for x in values])
        selected_reference_rows.append(
            {"arm": key[0], "seed": key[1], "phase": key[2], "split": key[3], "world_id": key[4], **metric}
        )
    with (root / "selected_checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            list(selected_reference_rows[0].keys())
            if selected_reference_rows
            else ["arm", "seed", "phase", "split", "world_id"]
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_reference_rows)

    continuation_summary = summarize_endpoint(
        root,
        policy_selection,
        branch_states,
        continuation_states,
        native_result,
        baseline_summary,
        started_epoch,
    )
    report = {
        "status": "COMPLETED",
        "scope": "existing six-world development data only",
        "policy_selection": policy_selection,
        "stability": continuation_summary["stability"],
        "endpoint_metrics": continuation_summary["endpoint_metrics"],
        "native_span": native_result["world_summary"],
        "fixed20_baseline": baseline_summary,
        "starting_state_checks": starting_checks,
        "boundary_checks": boundary_check,
        "world_populations": population_rows,
        "selected_checkpoint_checks": selected_checks,
        "score_reference_verification": score_verification,
        "selected_checkpoint_reference_metrics": selected_reference_rows,
        "known_limits": [
            "six fixed worlds from one generator family",
            "validation worlds were used for policy/checkpoint selection",
            "local stability does not establish global convergence",
            "no independent-world inference",
            "no external PH backend parity",
        ],
        "elapsed_seconds": time.time() - started_epoch,
        "output_bytes": output_size(root),
    }
    write_endpoint_report(root, report)
    generate_figures(root, policy_selection, baseline_rows, native_result)
    enforce_limits(root, started_epoch, processing=False)
    json_dump(
        root / "cost_accounting.json",
        {
            "elapsed_gross_seconds": time.time() - started_epoch,
            "output_bytes_before_hash_manifest": output_size(root),
            "branches": 40,
            "selected_continuations": 20,
            "evaluation_files_verified": score_verification["evaluation_files"],
            "prediction_rows_reference_checked": score_verification["prediction_rows"],
            "generation_preprocessing_custody": "not measured by this continuation study",
        },
    )
    report["output_bytes"] = output_size(root)
    report["output_size_scope"] = "measured before writing final report and output hash manifest"
    report["elapsed_seconds"] = time.time() - started_epoch
    json_dump(root / "final_report.json", report)
    artifacts = [p for p in root.rglob("*") if p.is_file() and p.name != "output_hashes.json"]
    json_dump(root / "output_hashes.json", [sha256_file(p) for p in sorted(artifacts)])
    enforce_limits(root, started_epoch, processing=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(root),
                "elapsed_seconds": report["elapsed_seconds"],
                "output_bytes": report["output_bytes"],
                "policy_selection": policy_selection,
                "stability": report["stability"],
                "native_span": report["native_span"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def bounded_report(
    root,
    stop,
    started_epoch,
    native_result,
    baseline_summary,
    boundary_check,
    population_rows,
    progress,
    policy_selection=None,
):
    branches = []
    for path in sorted(root.glob("branches/**/latest.pt")):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            branches.append(
                {
                    "trial_id": state.get("trial_id"),
                    "arm": state.get("arm"),
                    "seed": state.get("seed"),
                    "initial_rate": state.get("initial_rate"),
                    "epoch": state.get("epoch"),
                    "best_epoch": state.get("best_epoch"),
                    "best_validation_bce": state.get("best_validation_bce"),
                }
            )
        except Exception as exc:
            branches.append({"path": str(path), "state_read_error": f"{type(exc).__name__}: {exc}"})
    report = {
        "status": "BOUNDED_PARTIAL",
        "stop_reason": str(stop),
        "elapsed_seconds": time.time() - started_epoch,
        "output_bytes": output_size(root),
        "resumable_states_retained": len(branches),
        "branch_states": branches,
        "branch_round_robin": progress,
        "policy_selection": policy_selection,
        "native_span": native_result.get("world_summary", {}),
        "fixed20_baseline": baseline_summary,
        "boundary_checks": boundary_check,
        "world_populations": population_rows,
        "scope_note": "No automatic extension; future execution requires a new explicit decision.",
    }
    json_dump(root / "partial_report.json", report)
    (root / "endpoint_report.md").write_text(
        "# S1-E bounded partial endpoint\n\n" + json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "stop_reason": str(stop),
                "elapsed_seconds": report["elapsed_seconds"],
                "output_bytes": report["output_bytes"],
                "resumable_states_retained": len(branches),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BoundedStop as exc:
        if ACTIVE_OUTPUT_ROOT is not None:
            json_dump(
                ACTIVE_OUTPUT_ROOT / "partial_report.json",
                {
                    "status": "BOUNDED_PARTIAL",
                    "stop_reason": str(exc),
                    "phase": "initialization_or_postprocessing",
                    "resumable_states_retained": True,
                },
            )
        raise SystemExit(str(exc))
    except Exception as exc:
        if ACTIVE_OUTPUT_ROOT is not None:
            record_attempt_failure(ACTIVE_OUTPUT_ROOT, exc)
        raise
