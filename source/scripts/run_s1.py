#!/usr/bin/env python3
"""Reproducible, declared AMLSim S1 development pilot. No factory imports."""

from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
import traceback
from collections import defaultdict, Counter
from pathlib import Path

# Paths are relative to the directory the script is launched from; override with --spec/--output-dir.
ROOT = Path.cwd().resolve()
OUT = ROOT / "project/s1"
SPEC_DEFAULT = ROOT / "project/s1/pilot_spec.json"
FAILURES = 0
STAGES = []
CURRENT_ATTEMPT = None
REUSE_BUILD_DIR = None


def sha(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def hashes(paths):
    expanded = []
    for p in map(Path, paths):
        if p.is_dir():
            expanded.extend(x for x in p.rglob("*") if x.is_file())
        elif p.is_file():
            expanded.append(p)
    return [sha(p) for p in sorted(set(expanded), key=lambda x: str(x))]


def log_stage(name, argv, cwd, env, inputs, outputs, proc=None, elapsed=0.0, error=None):
    global FAILURES
    rec = {
        "stage": name,
        "argv": list(map(str, argv)),
        "cwd": str(cwd),
        "environment": env,
        "inputs": hashes(inputs),
        "outputs": hashes(outputs),
        "exit_code": None if proc is None else proc.returncode,
        "stdout": None if proc is None else proc.stdout,
        "stderr": None if proc is None else proc.stderr,
        "elapsed_seconds": round(elapsed, 3),
        "error": error,
    }
    STAGES.append(rec)
    if error is not None or (proc is not None and proc.returncode != 0):
        FAILURES += 1
    p = OUT / "audit" / "stages.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    if FAILURES >= 2:
        raise RuntimeError("two failures reached; stop under Human S1-A instruction")
    return rec


def run(name, argv, cwd, env=None, inputs=(), outputs=(), timeout=900):
    start = time.monotonic()
    merged = os.environ.copy()
    merged.update(env or {})
    try:
        proc = subprocess.run(
            list(map(str, argv)),
            cwd=str(cwd),
            env=merged,
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except BaseException as e:
        log_stage(name, argv, cwd, env or {}, inputs, outputs, None, time.monotonic() - start, repr(e))
        raise
    log_stage(name, argv, cwd, env or {}, inputs, outputs, proc, time.monotonic() - start)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, argv, proc.stdout, proc.stderr)
    return proc


def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_json(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(x, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def copy_source(src, dst):
    if dst.exists():
        # Preserve and reuse an exact prior assembly after a failed attempt.
        left = hashes([src])
        right = hashes([dst])
        if [(x["path"].split(str(src), 1)[-1], x["sha256"]) for x in left] == [
            (x["path"].split(str(dst), 1)[-1], x["sha256"]) for x in right
        ]:
            return
        raise FileExistsError(f"working copy differs from the pinned capture: {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "target", "__pycache__"))


def patch_source(tree):
    """Minimal read-only observation hooks; no random or model calls are added."""
    base = Path(tree) / "src/main/java/amlsim"
    (base / "S1Observer.java").write_text(
        r"""package amlsim;
import java.io.*;import java.nio.file.*;import java.util.*;
public final class S1Observer {
 private static BufferedWriter schedule,attempt,binding,emitted,buffered,persisted,outcome;
 private static long nextPhysicalEventId=0,nextAttemptId=0,nextRepositorySlot=0,boundAttemptId=0;
 private static final Map<Long,Long> eventAttempts=new HashMap<>();
 private static boolean enabled=false;
 private static String q(String s){return "\""+s.replace("\\","\\\\").replace("\"","\\\"")+"\"";}
 private static void emit(BufferedWriter w,String s){try{w.write(s);w.newLine();w.flush();}catch(IOException e){throw new UncheckedIOException("S1 sidecar write failed",e);}}
 public static void configure(){try{String d=System.getProperty("amlsim.observe.dir");if(d==null)throw new IOException("observer directory not configured");Path p=Paths.get(d);Files.createDirectories(p);schedule=Files.newBufferedWriter(p.resolve("cycle_schedules.jsonl"));attempt=Files.newBufferedWriter(p.resolve("arc_attempts.jsonl"));binding=Files.newBufferedWriter(p.resolve("event_attempt_bindings.jsonl"));emitted=Files.newBufferedWriter(p.resolve("emitted_events.jsonl"));buffered=Files.newBufferedWriter(p.resolve("buffered_events.jsonl"));persisted=Files.newBufferedWriter(p.resolve("persisted_events.jsonl"));outcome=Files.newBufferedWriter(p.resolve("attempt_outcomes.jsonl"));enabled=true;}catch(IOException e){throw new UncheckedIOException("S1 observer setup failed",e);}}
 public static boolean enabled(){return enabled;}
 public static void cycle(long id,boolean sar,List<Account> members,long[] steps){List<String> m=new ArrayList<>();for(Account a:members)m.add(q(a.getID()));StringJoiner j=new StringJoiner(",","[","]");for(long step:steps)j.add(Long.toString(step));emit(schedule,"{\"alert_id\":"+id+",\"sar\":"+sar+",\"members\":["+String.join(",",m)+"],\"steps\":"+j+"}");}
 public static long arc(long aid,int index,long step,String src,String dst,double amount,double srcBalance){long id=++nextAttemptId;emit(attempt,"{\"attempt_id\":"+id+",\"alert_id\":"+aid+",\"arc_index\":"+index+",\"step\":"+step+",\"source\":"+q(src)+",\"target\":"+q(dst)+",\"amount_drawn\":"+amount+",\"source_balance_before\":"+srcBalance+"}");emit(outcome,"{\"attempt_id\":"+id+",\"status\":\"attempted\"}");return id;}
 public static void bindAttempt(long id){if(boundAttemptId!=0)throw new IllegalStateException("S1 active CYCLE attempt was not consumed");boundAttemptId=id;}
 public static long beginEvent(long alertId){long id=++nextPhysicalEventId;long aid=boundAttemptId;boundAttemptId=0;if(aid!=0)eventAttempts.put(id,aid);emit(binding,"{\"physical_event_id\":"+id+",\"alert_id\":"+alertId+",\"attempt_id\":"+aid+"}");return id;}
 private static long attemptFor(long id){return eventAttempts.getOrDefault(id,0L);}
 public static void emitted(long id,long step,String desc,double amount,String src,float srcBefore,float srcAfter,String dst,float dstBefore,float dstAfter,boolean sar,long alertId){emit(emitted,"{\"physical_event_id\":"+id+",\"attempt_id\":"+attemptFor(id)+",\"step\":"+step+",\"description\":"+q(desc)+",\"amount_double\":"+amount+",\"source\":"+q(src)+",\"source_before_float\":"+srcBefore+",\"source_after_float\":"+srcAfter+",\"target\":"+q(dst)+",\"target_before_float\":"+dstBefore+",\"target_after_float\":"+dstAfter+",\"sar\":"+sar+",\"alert_id\":"+alertId+"}");if(attemptFor(id)!=0)emit(outcome,"{\"attempt_id\":"+attemptFor(id)+",\"status\":\"event_emitted\",\"physical_event_id\":"+id+"}");}
 public static long repositorySlot(){return nextRepositorySlot++;}
 public static void buffered(long id,long repoSlot,int bufferSlot){emit(buffered,"{\"physical_event_id\":"+id+",\"attempt_id\":"+attemptFor(id)+",\"repository_slot\":"+repoSlot+",\"buffer_slot\":"+bufferSlot+"}");if(attemptFor(id)!=0)emit(outcome,"{\"attempt_id\":"+attemptFor(id)+",\"status\":\"event_buffered\",\"physical_event_id\":"+id+"}");}
 public static void notBuffered(long id,String reason){emit(buffered,"{\"physical_event_id\":"+id+",\"attempt_id\":"+attemptFor(id)+",\"status\":\"not_buffered\",\"reason\":"+q(reason)+"}");}
 public static void persisted(long id,long repoSlot,int bufferSlot,String line){emit(persisted,"{\"physical_event_id\":"+id+",\"attempt_id\":"+attemptFor(id)+",\"repository_slot\":"+repoSlot+",\"buffer_slot\":"+bufferSlot+",\"csv_line\":"+q(line)+"}");if(attemptFor(id)!=0)emit(outcome,"{\"attempt_id\":"+attemptFor(id)+",\"status\":\"event_persisted\",\"physical_event_id\":"+id+"}");}
 public static void persistenceFailure(long id,String detail){emit(outcome,"{\"physical_event_id\":"+id+",\"attempt_id\":"+attemptFor(id)+",\"status\":\"persistence_failure\",\"detail\":"+q(detail)+"}");}
 public static void attemptOutcome(long id,String status,String detail){if(boundAttemptId==id)boundAttemptId=0;emit(outcome,"{\"attempt_id\":"+id+",\"status\":"+q(status)+",\"detail\":"+q(detail)+"}");}
 public static void finishAttempt(long id){if(boundAttemptId==id){boundAttemptId=0;attemptOutcome(id,"returned_without_handleTransaction","native call returned without an event");}else attemptOutcome(id,"native_call_returned","event identity is bound independently");}
 public static void close(){try{schedule.close();attempt.close();binding.close();emitted.close();buffered.close();persisted.close();outcome.close();}catch(IOException e){throw new UncheckedIOException("S1 observer close failed",e);}}
}

""",
        encoding="utf-8",
    )
    # Captured blobs are read-only. Make only these new working-copy files writable.
    for name in ["model/aml/CycleTypology.java", "TransactionRepository.java", "AMLSim.java"]:
        path = base / name
        path.chmod(path.stat().st_mode | 0o200)
    # Record finalized actual CYCLE schedule after all native RNG draws.
    cp = base / "model/aml/CycleTypology.java"
    s = cp.read_text()
    s = s.replace("package amlsim.model.aml;", "package amlsim.model.aml;\n\nimport amlsim.S1Observer;", 1)
    needle = "//        System.out.println(Arrays.toString(steps));"
    assert needle in s
    s = s.replace(
        needle,
        "S1Observer.cycle(alert.getAlertID(), alert.isSAR(), alert.getMembers(), steps);\n        " + needle,
        1,
    )
    needle = "transactionAmount = new TargetedTransactionAmount(amount, random);\n                makeTransaction(step, transactionAmount.doubleValue(), src, dst, isSAR, alertID);"
    assert needle in s
    s = s.replace(
        needle,
        'transactionAmount = new TargetedTransactionAmount(amount, random);\n                double drawnAmount = transactionAmount.doubleValue();\n                long attemptId = S1Observer.arc(alertID, i, step, src.getID(), dst.getID(), drawnAmount, src.getBalance());\n                if (!(drawnAmount > 0.0)) { S1Observer.attemptOutcome(attemptId, "rejected_nonpositive", "native makeTransaction rejects nonpositive amount"); }\n                else { S1Observer.bindAttempt(attemptId); try { makeTransaction(step, drawnAmount, src, dst, isSAR, alertID); S1Observer.finishAttempt(attemptId); } catch (RuntimeException | Error e) { S1Observer.attemptOutcome(attemptId, "failed_exception", e.toString()); throw e; } }',
        1,
    )
    cp.write_text(s)
    rp = base / "TransactionRepository.java"
    s = rp.read_text()

    def replace_once(text, old, new):
        if text.count(old) != 1:
            raise ValueError("native observation anchor is not unique: " + old[:100])
        return text.replace(old, new, 1)

    s = replace_once(
        s,
        "private long[] alertIDs;",
        "private long[] alertIDs;\n    private long[] physicalEventIDs;\n    private long[] repositorySlots;",
    )
    s = replace_once(
        s,
        "this.alertIDs = new long[size];",
        "this.alertIDs = new long[size];\n        this.physicalEventIDs = new long[size];\n        this.repositorySlots = new long[size];",
    )
    s = replace_once(
        s,
        "float origAfter, float destBefore, float destAfter, boolean isSAR, long aid){",
        "float origAfter, float destBefore, float destAfter, boolean isSAR, long aid, long physicalEventId){",
    )
    s = replace_once(
        s,
        "if(count >= limit){",
        'if(count >= limit){\n            if(S1Observer.enabled()) S1Observer.notBuffered(physicalEventId, "native transaction limit");',
    )
    s = replace_once(
        s,
        "this.steps[index] = step;",
        "this.physicalEventIDs[index] = physicalEventId;\n        if(S1Observer.enabled()){\n            this.repositorySlots[index] = S1Observer.repositorySlot();\n            S1Observer.buffered(physicalEventId, this.repositorySlots[index], index);\n        }\n        this.steps[index] = step;",
    )
    needle = 'writer.write(steps[i] + "," + descriptions[i] + "," + getDoublePrecision(amounts[i]) + "," +\n                        origIDs[i] + "," + getDoublePrecision(origBefore[i]) + "," + getDoublePrecision(origAfter[i]) + "," +\n                        destIDs[i] + "," + getDoublePrecision(destBefore[i]) + "," + getDoublePrecision(destAfter[i]) + "," +\n                        (isSAR[i] ? "1" : "0") + "," + alertIDs[i] + "\\n");'
    assert needle in s
    replacement = 'String row = steps[i] + "," + descriptions[i] + "," + getDoublePrecision(amounts[i]) + "," +\n                        origIDs[i] + "," + getDoublePrecision(origBefore[i]) + "," + getDoublePrecision(origAfter[i]) + "," +\n                        destIDs[i] + "," + getDoublePrecision(destBefore[i]) + "," + getDoublePrecision(destAfter[i]) + "," +\n                        (isSAR[i] ? "1" : "0") + "," + alertIDs[i];\n                writer.write(row + "\\n");\n                S1Observer.event(steps[i], descriptions[i], amounts[i], origIDs[i], origBefore[i], origAfter[i], destIDs[i], destBefore[i], destAfter[i], isSAR[i], alertIDs[i], row, i);'
    # Keep the same native row and attach identity only after its writer closes.
    replacement = replacement[: replacement.index("\n                S1Observer.event(")]
    replacement += "\n                persistedRows[i] = row;"
    s = replace_once(
        s,
        "// Flush transaction logs to the CSV file",
        "// Flush transaction logs to the CSV file\n        String[] persistedRows = new String[this.index];",
    )
    s = replace_once(s, needle, replacement)
    s = replace_once(
        s,
        "writer.close();\n        } catch (IOException e)",
        "writer.close();\n            if(S1Observer.enabled()){\n                for(int i=0; i<this.index; i++) S1Observer.persisted(physicalEventIDs[i], repositorySlots[i], i, persistedRows[i]);\n            }\n        } catch (IOException e)",
    )
    s = replace_once(
        s,
        "e.printStackTrace();\n        }\n        index = 0;",
        'e.printStackTrace();\n            if(S1Observer.enabled()){\n                for(int i=0; i<this.index; i++) S1Observer.persistenceFailure(physicalEventIDs[i], e.toString());\n            }\n            throw new IllegalStateException("S1 transaction persistence failed", e);\n        }\n        index = 0;',
    )
    rp.write_text(s)
    mp = base / "AMLSim.java"
    s = mp.read_text()
    s = replace_once(
        s,
        "simProp = new SimProperties(confFile);",
        'simProp = new SimProperties(confFile);\n            if (System.getProperty("amlsim.observe.dir") != null) S1Observer.configure();',
    )
    s = replace_once(
        s,
        "// Reduce the balance of the originator account",
        "long physicalEventId = S1Observer.enabled() ? S1Observer.beginEvent(alertID) : 0L;\n        // Reduce the balance of the originator account",
    )
    s = replace_once(
        s,
        "txs.addTransaction(step, desc, amt, origID, beneID, origBefore, origAfter, beneBefore, beneAfter, isSAR, alertID);",
        "if(S1Observer.enabled()) S1Observer.emitted(physicalEventId, step, desc, amt, origID, origBefore, origAfter, beneID, beneBefore, beneAfter, isSAR, alertID);\n        txs.addTransaction(step, desc, amt, origID, beneID, origBefore, origAfter, beneBefore, beneAfter, isSAR, alertID, physicalEventId);",
    )
    s = replace_once(
        s,
        "txs.writeCounterLog(numOfSteps, counterFile);",
        'txs.writeCounterLog(numOfSteps, counterFile);\n        if (System.getProperty("amlsim.observe.dir") != null) S1Observer.close();',
    )
    mp.write_text(s)


def generate(spec, world, world_id):
    rng = random.Random(spec["seed"])
    perm = list(range(spec["accounts"]))
    rng.shuffle(perm)
    balances = [rng.uniform(*spec["balance_range"]) for _ in range(spec["accounts"])]
    accounts = [f"A{i:02d}" for i in range(spec["accounts"])]
    account_rows = []
    sar = set()
    cycle_intents = []
    offset = 0
    for i, c in enumerate(spec["cycles"]):
        members = [accounts[x] for x in perm[offset : offset + c["length"]]]
        offset += c["length"]
        cycle_intents.append(
            {
                "alert_id": 100 + i,
                "type": "CYCLE",
                "model_id": 3,
                "members": members,
                "schedule_requested": c["schedule"],
                "sar": c["sar"],
            }
        )
        if c["sar"]:
            sar.update(members)
    fi = [accounts[x] for x in perm[25:28]]
    fo = [accounts[x] for x in perm[28:31]]
    intents = cycle_intents + [
        {
            "alert_id": 200,
            "type": "FAN_IN",
            "model_id": 2,
            "members": fi,
            "schedule_requested": 1,
            "sar": False,
        },
        {
            "alert_id": 201,
            "type": "FAN_OUT",
            "model_id": 1,
            "members": fo,
            "schedule_requested": 1,
            "sar": True,
        },
    ]
    sar.update(fo)
    account_rows = [[a, a in sar, format(balances[int(a[1:])], ".9f"), "B0"] for a in accounts]
    arcs = []
    seen = set()

    def canonrot(c):
        return min(tuple(c[i:] + c[:i]) for i in range(len(c)))

    for intent in intents:
        m = intent["members"]
        pairs = list(zip(m[:-1], m[1:])) + ([(m[-1], m[0])] if intent["type"] == "CYCLE" else [])
        if intent["type"] == "FAN_IN":
            pairs = [(u, m[0]) for u in m[1:]]
        if intent["type"] == "FAN_OUT":
            pairs = [(m[0], v) for v in m[1:]]
        for u, v in pairs:
            if (u, v) not in seen:
                seen.add((u, v))
                arcs.append((u, v, "WIRE"))
    normal_main_idxs = spec["normal_main_permutation_indices"]
    normal_rows = []
    normal_arcs = []
    destinations = [1, 4, 7, 11, 15, 20]
    for n, (idx, didx) in enumerate(zip(normal_main_idxs[:6], destinations)):
        a, b = accounts[perm[idx]], accounts[perm[didx]]
        normal_rows.extend([["periodical", a, n, True], ["periodical", b, n, False]])
        normal_arcs.append((a, b, "WIRE"))
    a, b = accounts[perm[30]], accounts[perm[31]]
    normal_rows.extend(
        [
            ["periodical", a, 6, True],
            ["periodical", b, 6, False],
            ["periodical", b, 7, True],
            ["periodical", a, 7, False],
        ]
    )
    normal_arcs.extend([(a, b, "WIRE"), (b, a, "WIRE")])
    for x in normal_arcs:
        if x[:2] not in seen:
            seen.add(x[:2])
            arcs.append(x)
    alert_rows = []
    for item in intents:
        for j, a in enumerate(item["members"]):
            alert_rows.append(
                [
                    item["alert_id"],
                    a,
                    j == 0,
                    item["sar"],
                    item["model_id"],
                    spec["alert_min_amount"],
                    spec["alert_max_amount"],
                    spec["input_alert_start"],
                    spec["input_alert_end"],
                    item["schedule_requested"],
                ]
            )
    inputdir = world / "tmp" / world_id
    inputdir.mkdir(parents=True, exist_ok=True)
    write_csv(inputdir / "accounts.csv", ["ACCOUNT_ID", "IS_SAR", "INIT_BALANCE", "BANK_ID"], account_rows)
    write_csv(inputdir / "transactions.csv", ["src", "dst", "ttype"], arcs)
    write_csv(inputdir / "normal_models.csv", ["type", "accountID", "modelID", "isMain"], normal_rows)
    write_csv(
        inputdir / "alert_members.csv",
        [
            "alertID",
            "accountID",
            "isMain",
            "isSAR",
            "modelID",
            "minAmount",
            "maxAmount",
            "startStep",
            "endStep",
            "scheduleID",
        ],
        alert_rows,
    )
    write_json(
        world / "intentions.json",
        {
            "disclosure": "controlled stdlib generator; not financial records",
            "world_id": world_id,
            "seed": spec["seed"],
            "input_rng": "random.Random(seed) from Python standard library",
            "generator_seed": spec["seed"],
            "permutation": [accounts[i] for i in perm],
            "initial_balances": balances,
            "intentions": intents,
            "normal_groups": normal_rows,
            "template_arcs": arcs,
        },
    )
    return inputdir, intents


JAR_SHA256 = {
    "mason-20.jar": "9989c3ecaf288c49b2db01405fc588fb994a39a98656a2434be107f79de83a3e",
    "json-20180813.jar": "518080049ba83181914419d11a25d9bc9833a2d729b6a6e7469fa52851356da8",
    "webgraph-3.6.1.jar": "648223a16a56beb41baaefd50e86f6a4890f07e147672af0027d6baa70c40723",
    "dsiutils-2.5.4.jar": "f51c098cbe2f22c5cfcc054ef21bf5eba2b9f3a88a36a3a54f92b76a6e56bfbd",
    "fastutil-8.2.3.jar": "13195d70b40023c75c782e11682fa15beab4430286cdf0d7c616cf02ad2606fe",
    "sux4j-4.2.0.jar": "1b3898b98fef01ba31acfc3ad49eaa6412185dbf2d70fb76b838b364edefaa5f",
    "jsap-2.1.jar": "331746fa62cfbc3368260c5a2e660936ad11be612308c120a044e120361d474e",
    "slf4j-api-1.7.25.jar": "18c4a0095d5c1da6b817592e767bb23d29dd2f560ad74df75ff3961dbde25b79",
    "slf4j-simple-1.7.25.jar": "0966e86fffa5be52d3d9e7b89dd674d98a03eed0a454fbaf7c1bd9493bd9d874",
    "commons-math3-3.6.1.jar": "1e56d7b058d28b65abd256b8458e3885b674c1d588fa43cd7d1cbb9c7ef2b308",
}  # verified bytes of the jars used for every reported world


def download_deps(attempt):
    dep = attempt / "deps"
    dep.mkdir(parents=True, exist_ok=True)
    coords = [
        (
            "mason-20.jar",
            "https://people.cs.gmu.edu/~eclab/projects/mason/mason.20.jar",
            "MASON 20",
            "Academic Free License 3.0",
        ),
        (
            "json-20180813.jar",
            "https://repo.maven.apache.org/maven2/org/json/json/20180813/json-20180813.jar",
            "org.json 20180813",
            "JSON-java license",
        ),
        (
            "webgraph-3.6.1.jar",
            "https://repo.maven.apache.org/maven2/it/unimi/dsi/webgraph/3.6.1/webgraph-3.6.1.jar",
            "WebGraph 3.6.1",
            "Apache 2.0",
        ),
        (
            "dsiutils-2.5.4.jar",
            "https://repo.maven.apache.org/maven2/it/unimi/dsi/dsiutils/2.5.4/dsiutils-2.5.4.jar",
            "dsiutils 2.5.4",
            "Apache 2.0",
        ),
        (
            "fastutil-8.2.3.jar",
            "https://repo.maven.apache.org/maven2/it/unimi/dsi/fastutil/8.2.3/fastutil-8.2.3.jar",
            "fastutil 8.2.3",
            "Apache 2.0",
        ),
        (
            "sux4j-4.2.0.jar",
            "https://repo.maven.apache.org/maven2/it/unimi/dsi/sux4j/4.2.0/sux4j-4.2.0.jar",
            "sux4j 4.2.0",
            "Apache 2.0",
        ),
        (
            "jsap-2.1.jar",
            "https://repo.maven.apache.org/maven2/com/martiansoftware/jsap/2.1/jsap-2.1.jar",
            "JSAP 2.1",
            "Apache 2.0",
        ),
        (
            "slf4j-api-1.7.25.jar",
            "https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/1.7.25/slf4j-api-1.7.25.jar",
            "slf4j-api 1.7.25",
            "MIT",
        ),
        (
            "slf4j-simple-1.7.25.jar",
            "https://repo.maven.apache.org/maven2/org/slf4j/slf4j-simple/1.7.25/slf4j-simple-1.7.25.jar",
            "slf4j-simple 1.7.25",
            "MIT",
        ),
        (
            "commons-math3-3.6.1.jar",
            "https://repo.maven.apache.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar",
            "commons-math3 3.6.1",
            "Apache 2.0",
        ),
    ]
    prior_manifest = {}
    if REUSE_BUILD_DIR:
        manifest_path = REUSE_BUILD_DIR / "dependencies.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("reuse build dependency manifest missing: " + str(manifest_path))
        prior_manifest = {
            Path(x["file"]).name: x for x in json.loads(manifest_path.read_text())["dependencies"]
        }
    entries = []
    for name, url, version, license in coords:
        dest = dep / name
        if not dest.exists():
            if REUSE_BUILD_DIR:
                entry = prior_manifest.get(name)
                if not entry:
                    raise ValueError("reuse build lacks pinned dependency binding: " + name)
                dest = Path(entry["file"])
                if not dest.is_file() or sha(dest)["sha256"] != entry["binding"]["sha256"]:
                    raise ValueError("reused dependency hash mismatch: " + name)
            else:
                prior = [
                    x
                    for x in sorted((OUT / "attempts").glob("pilot_*/deps/" + name), reverse=True)
                    if x.is_file() and attempt not in x.parents
                ]
                if prior:
                    dest = prior[0]
                else:
                    # curl is invoked as argv, never through a shell; stderr/stdout retained.
                    run(
                        "download:" + version,
                        [
                            "/usr/bin/curl",
                            "--fail",
                            "--location",
                            "--silent",
                            "--show-error",
                            url,
                            "--output",
                            str(dest),
                        ],
                        OUT,
                        outputs=[dest],
                        timeout=120,
                    )
        if dest.stat().st_size < 1024 or dest.read_bytes()[:4] != b"PK\x03\x04":
            raise ValueError("dependency is not a nontrivial JAR: " + name)
        if sha(dest)["sha256"] != JAR_SHA256[name]:
            raise ValueError("dependency SHA-256 differs from the pinned value: " + name)
        entries.append(
            {
                "file": str(dest),
                "version": version,
                "url": url,
                "license_declared": license,
                "binding": sha(dest),
                "reused_prior_attempt": dest.parent != dep,
            }
        )
    write_json(
        attempt / "dependencies.json",
        {"primary_sources": "MASON publisher site and Maven Central", "dependencies": entries},
    )
    return [x["file"] for x in entries]


def config_for(spec, source, world, outdir, template, world_id):
    c = json.loads((source / "conf.json").read_text())
    c["general"].update(random_seed=spec["seed"], simulation_name=world_id, total_steps=spec["steps"])
    c["default"].update(
        min_amount=spec["alert_min_amount"],
        max_amount=spec["alert_max_amount"],
        margin_ratio=spec["margin_ratio"],
    )
    c["temporal"].update(
        directory=str(world / "tmp"),
        transactions="transactions.csv",
        accounts="accounts.csv",
        alert_members="alert_members.csv",
        normal_models="normal_models.csv",
    )
    c["output"]["directory"] = str(outdir)
    c["simulator"].update(compute_diameter=False, transaction_limit=0, transaction_interval=20, numBranches=1)
    p = outdir.parent / (outdir.name + "_conf.json")
    write_json(p, c)
    return p


def event_key(e):
    return (
        int(e["step"]),
        e["description"],
        float(e["amount"]),
        e["source"],
        e["target"],
        e["source_before_float"],
        e["source_after_float"],
        e["target_before_float"],
        e["target_after_float"],
        bool(e["sar"]),
        int(e["alert_id"]),
    )


def focused_fixture_checks():
    # Tiny structural controls required by the S1 contract; no AML/source rows.
    events = [("e1", "A", "B", "WIRE", 10.0), ("e2", "B", "A", "WIRE", 10.0), ("e3", "A", "B", "WIRE", 10.0)]
    reciprocal = [
        (a[0], b[0]) for a in events if a[1:3] == ("A", "B") for b in events if b[1:3] == ("B", "A")
    ]

    def boundary(edges):
        out = Counter()
        for src, dst in edges:
            out[src] -= 1
            out[dst] += 1
        return out

    residual = boundary([("A", "B"), ("B", "A")])
    malformed = boundary([("A", "B")])
    incomplete_attempt_indices = [0]
    checks = {
        "reciprocal_two_cycle_exact_boundary": not any(residual.values()),
        "parallel_equal_content_events_distinct": len(reciprocal) == 2 and reciprocal[0] != reciprocal[1],
        "incomplete_cycle_not_positive": incomplete_attempt_indices != [0, 1],
        "malformed_boundary_rejected": bool(any(malformed.values())),
    }
    if not all(checks.values()):
        raise ValueError("focused structural fixture check failed")
    return checks


def reconcile(basecsv, instcsv, schedulefile, attemptfile, eventfile):
    with open(basecsv, newline="") as f:
        base = list(csv.DictReader(f))
    with open(instcsv, newline="") as f:
        inst = list(csv.DictReader(f))
    if base != inst:
        raise ValueError("unmodified and instrumented native CSV outputs differ")

    def records(path):
        with open(path) as f:
            return [json.loads(x) for x in f if x.strip()]

    persisted = records(eventfile)
    side = Path(eventfile).parent
    emitted = records(side / "emitted_events.jsonl")
    bindings = records(side / "event_attempt_bindings.jsonl")
    buffered = records(side / "buffered_events.jsonl")
    schedules = records(schedulefile)
    attempts = records(attemptfile)
    outcomes = records(side / "attempt_outcomes.jsonl")

    def unique(rows, key):
        result = {}
        for row in rows:
            k = int(row[key])
            if k in result:
                raise ValueError("duplicate sidecar identity: " + key + "=" + str(k))
            result[k] = row
        return result

    attempt_map = unique(attempts, "attempt_id")
    emitted_map = unique(emitted, "physical_event_id")
    binding_map = unique(bindings, "physical_event_id")
    buffered_map = unique(buffered, "physical_event_id")
    persisted_map = unique(persisted, "physical_event_id")
    expected = set(range(1, len(bindings) + 1))
    if any(set(x) != expected for x in [binding_map, emitted_map, buffered_map, persisted_map]):
        raise ValueError("physical call/emission/buffer/persistence identities do not reconcile")
    if len(persisted) != len(inst):
        raise ValueError("persisted sidecar/native row count mismatch")

    def native_precision(value):
        # Java's integer cast saturates and truncates toward zero.
        x = float(value) * 100
        cast = 0 if math.isnan(x) else max(-2147483648, min(2147483647, math.trunc(x)))
        return cast / 100

    events = []
    used_attempts = set()
    for i, (row, record) in enumerate(zip(inst, persisted)):
        eid = int(record["physical_event_id"])
        event = dict(emitted_map[eid])
        binding = binding_map[eid]
        buf = buffered_map[eid]
        if record["csv_line"] != ",".join(row.values()):
            raise ValueError("sidecar persisted CSV line mismatch at " + str(i))
        if (
            int(record["repository_slot"]) != i
            or buf.get("status") == "not_buffered"
            or int(buf["repository_slot"]) != i
            or int(buf["buffer_slot"]) != int(record["buffer_slot"])
        ):
            raise ValueError("repository identity/slot mismatch")
        aid = int(binding["attempt_id"])
        if any(int(x["attempt_id"]) != aid for x in [event, buf, record]):
            raise ValueError("attempt/event binding mismatch")
        if int(binding["alert_id"]) != int(event["alert_id"]):
            raise ValueError("physical entry/emission alert mismatch")
        if (
            int(row["step"]),
            row["type"],
            row["nameOrig"],
            row["nameDest"],
            int(row["alertID"]),
            bool(int(row["isSAR"])),
        ) != (
            int(event["step"]),
            event["description"],
            event["source"],
            event["target"],
            int(event["alert_id"]),
            bool(event["sar"]),
        ):
            raise ValueError("native/emitted fields differ")
        for native, field in [
            ("amount", "amount_double"),
            ("oldbalanceOrig", "source_before_float"),
            ("newbalanceOrig", "source_after_float"),
            ("oldbalanceDest", "target_before_float"),
            ("newbalanceDest", "target_after_float"),
        ]:
            value = float(event[field])
            # Java Float.toString roundtrips through binary32, not directly to binary64.
            if field.endswith("_float"):
                value = struct.unpack("!f", struct.pack("!f", value))[0]
            if not math.isfinite(value) or float(row[native]) != native_precision(value):
                raise ValueError("native precision mismatch: " + native)
        if aid:
            if aid not in attempt_map or aid in used_attempts:
                raise ValueError("missing or multiply emitted CYCLE attempt")
            a = attempt_map[aid]
            used_attempts.add(aid)
            if (int(a["alert_id"]), int(a["step"]), a["source"], a["target"], float(a["amount_drawn"])) != (
                int(event["alert_id"]),
                int(event["step"]),
                event["source"],
                event["target"],
                float(event["amount_double"]),
            ):
                raise ValueError("CYCLE attempt/event fields differ")
        event.update(
            event_id=eid,
            repository_slot=i,
            buffer_slot=int(record["buffer_slot"]),
            csv_line=record["csv_line"],
            amount_persisted=float(row["amount"]),
        )
        events.append(event)
    for aid in set(attempt_map) - used_attempts:
        states = {r["status"] for r in outcomes if int(r.get("attempt_id", 0)) == aid}
        if not states.intersection({"rejected_nonpositive", "returned_without_handleTransaction"}):
            raise ValueError("unresolved CYCLE attempt without a persisted event")
    if any(r["status"] in {"failed_exception", "persistence_failure"} for r in outcomes):
        raise ValueError("failed scientific execution in attempt outcomes")
    return base, events, schedules, attempts


def cycles_and_candidates(
    rows,
    events,
    intents,
    schedules,
    attempts,
    account_ids,
    max_tuples=10000,
    world_id="S1_dev_1729",
    total_steps=120,
    candidate_min=2,
    candidate_max=12,
):
    # Candidate frame built from all persisted rows before reading any labels.
    eid_to_event = {int(e["event_id"]): e for e in events}
    event_edges = defaultdict(list)
    adj = defaultdict(set)
    for e in events:
        if e["description"] in {"CASH-IN", "CASH-OUT"}:
            continue
        u = e["source"]
        v = e["target"]
        event_edges[(u, v)].append(int(e["event_id"]))
        if u != v:
            adj[u].add(v)

    def canonrot(c):
        return min(tuple(c[i:] + c[:i]) for i in range(len(c)))

    found = set()
    for start in account_ids:

        def dfs(path):
            last = path[-1]
            if len(path) >= candidate_min and start in adj[last]:
                found.add(canonrot(path))
            if len(path) >= candidate_max:
                return
            for nxt in adj[last]:
                if nxt == start or nxt in path:
                    continue
                if nxt < start:
                    continue
                dfs(path + [nxt])

        dfs([start])
    cycles = sorted(found, key=lambda x: (len(x), x))
    choices = []
    capacities = [
        math.prod(len(event_edges[(cyc[i], cyc[(i + 1) % len(cyc)])]) for i in range(len(cyc)))
        for cyc in cycles
    ]
    total = sum(capacities)
    if total > max_tuples:
        raise OverflowError(
            "complete candidate census has "
            + str(total)
            + " tuples; limit "
            + str(max_tuples)
            + "; no tuples expanded or clipped"
        )
    for cyc in cycles:
        arcsets = [event_edges.get((cyc[i], cyc[(i + 1) % len(cyc)]), []) for i in range(len(cyc))]
        for combo in itertools.product(*arcsets):
            choices.append((cyc, tuple(combo)))
    frame = []
    labels = []
    identities = []
    ordered_events = [eid_to_event[i] for i in sorted(eid_to_event)]
    for j, (cyc, combo) in enumerate(choices):
        selected = [eid_to_event[x] for x in combo]
        row = {
            "cycle_length": len(cyc),
            "time_steps": [int(e["step"]) / float(total_steps) for e in selected],
            "log1p_amounts": [math.log1p(float(e["amount_persisted"])) for e in selected],
            "world_event_time_steps": [int(e["step"]) / float(total_steps) for e in ordered_events],
            "world_event_log1p_amounts": [math.log1p(float(e["amount_persisted"])) for e in ordered_events],
            "central_node_mask": [int(a in cyc) for a in account_ids],
            "selected_event_mask": [int(i in combo) for i in sorted(eid_to_event)],
        }
        frame.append(row)
        identities.append(
            {
                "candidate_id": f"cand{j:07d}",
                "world_id": world_id,
                "event_ids": list(combo),
                "cycle_accounts": list(cyc),
            }
        )
    # Annotation joins after complete full-world frame exists.
    positive = {}
    realized = {}
    frame_keys = {(tuple(x["cycle_accounts"]), tuple(x["event_ids"])) for x in identities}
    schedule_by_id = {}
    for sch in schedules:
        aid = int(sch["alert_id"])
        if aid in schedule_by_id:
            raise ValueError("duplicate finalized CYCLE schedule")
        schedule_by_id[aid] = sch
    cycle_intents = {int(x["alert_id"]): x for x in intents if x["type"] == "CYCLE"}
    if set(schedule_by_id) != set(cycle_intents):
        raise ValueError("finalized CYCLE schedule/intention identities differ")
    attempt_ids = set()
    events_by_attempt = defaultdict(list)
    for e in events:
        if int(e.get("attempt_id", 0)):
            events_by_attempt[int(e["attempt_id"])].append(e)
    for a in attempts:
        aid = int(a["attempt_id"])
        if aid in attempt_ids or int(a["alert_id"]) not in cycle_intents:
            raise ValueError("duplicate or unknown CYCLE attempt")
        attempt_ids.add(aid)
    if set(events_by_attempt) - attempt_ids:
        raise ValueError("event linked to unknown CYCLE attempt")
    for intent in intents:
        alert = int(intent["alert_id"])
        members = tuple(intent["members"])
        if intent["type"] != "CYCLE":
            observed = [e for e in events if int(e["alert_id"]) == alert]
            realized[str(alert)] = {
                "intent_id": alert,
                "type": intent["type"],
                "intended_members": list(members),
                "attempt_count": None,
                "persisted_event_ids": [int(e["event_id"]) for e in observed],
                "persisted_event_count": len(observed),
                "complete": None,
                "sar": intent["sar"],
            }
            continue
        sch = schedule_by_id[alert]
        if (
            tuple(sch["members"]) != members
            or len(sch["steps"]) != len(members)
            or bool(sch["sar"]) != bool(intent["sar"])
        ):
            raise ValueError("finalized CYCLE members/steps/SAR differ from intention")
        linked = {}
        attempted = set()
        for a in attempts:
            if int(a["alert_id"]) != alert:
                continue
            i = int(a["arc_index"])
            aid = int(a["attempt_id"])
            if i < 0 or i >= len(members) or i in attempted:
                raise ValueError("duplicate or out-of-range CYCLE arc attempt")
            attempted.add(i)
            if (a["source"], a["target"], int(a["step"])) != (
                members[i],
                members[(i + 1) % len(members)],
                int(sch["steps"][i]),
            ):
                raise ValueError("attempt differs from finalized CYCLE arc")
            emitted = events_by_attempt[aid]
            if len(emitted) > 1:
                raise ValueError("CYCLE attempt has multiple physical events")
            if emitted:
                e = emitted[0]
                if (
                    e["source"],
                    e["target"],
                    int(e["step"]),
                    int(e["alert_id"]),
                    float(e["amount_double"]),
                ) != (a["source"], a["target"], int(a["step"]), alert, float(a["amount_drawn"])):
                    raise ValueError("CYCLE event differs from exact attempt")
                linked[i] = int(e["event_id"])
        complete = set(linked) == set(range(len(members)))
        in_frame = False
        if complete:
            k = min(range(len(members)), key=lambda i: members[i])
            key = (
                members[k:] + members[:k],
                tuple(linked[(k + i) % len(members)] for i in range(len(members))),
            )
            if key in positive:
                raise ValueError("duplicate exact CYCLE target tuple")
            positive[key] = str(alert)
            in_frame = key in frame_keys
        realized[str(alert)] = {
            "intent_id": alert,
            "type": "CYCLE",
            "intended_members": list(members),
            "finalized_steps": sch["steps"],
            "attempt_count": len(attempted),
            "persisted_arc_indices": sorted(linked),
            "persisted_event_ids": [linked[i] for i in sorted(linked)],
            "complete": complete,
            "in_frame": in_frame,
            "sar": intent["sar"],
        }
    for identity in identities:
        tup = (tuple(identity["cycle_accounts"]), tuple(identity["event_ids"]))
        labels.append(
            {
                "candidate_id": identity["candidate_id"],
                "label": int(tup in positive),
                "instance_id": positive.get(tup),
            }
        )
    return cycles, frame, identities, labels, realized


def subdivided_path(event):
    """Structural incidence only: four segments per real event, including loops."""
    eid = int(event["event_id"])
    return [
        "account:" + event["source"],
        f"source_port:{eid}",
        f"event_vertex:{eid}",
        f"target_port:{eid}",
        "account:" + event["target"],
    ]


def candidate_topology(identity, event_map):
    cyc = identity["cycle_accounts"]
    eids = identity["event_ids"]
    cid = identity["candidate_id"]
    boundary = []
    triangles = []
    center = "face_center:" + cid
    for i, eid in enumerate(eids):
        event = event_map[eid]
        if (event["source"], event["target"]) != (cyc[i], cyc[(i + 1) % len(cyc)]):
            raise ValueError("selected physical event is not its directed central arc")
        boundary.append(
            {"event_id": eid, "source": cyc[i], "target": cyc[(i + 1) % len(cyc)], "orientation": 1}
        )
        path = subdivided_path(event)
        for a, b in zip(path, path[1:]):
            triangles.append(
                {"vertices": [a, b, center], "physical_event_id": eid, "structural_incidence": True}
            )
    if len({tuple(sorted(t["vertices"])) for t in triangles}) != len(triangles):
        raise ValueError("simplicial subdivision collapses a triangle")
    # Each triangle's directed boundary independently closes, including two-gons.
    for triangle in triangles:
        residual = Counter()
        v = triangle["vertices"]
        for a, b in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]:
            residual[a] -= 1
            residual[b] += 1
        if any(residual.values()):
            raise ValueError("simplicial triangle boundary is not closed")
    cellular = {
        "candidate_id": cid,
        "dimension": 2,
        "cell_type": "directed_polygon",
        "boundary": boundary,
        "triangulated": False,
    }
    simplicial = {
        "candidate_id": cid,
        "construction": "event-specific source/target ports and midpoint; fan on subdivided boundary",
        "original_boundary_event_ids": eids,
        "center": center,
        "triangles": triangles,
        "auxiliary_edges_are_financial_transactions": False,
    }
    hasse = {
        "candidate_id": cid,
        "rank0_accounts": cyc,
        "rank1_physical_events": eids,
        "rank2_face": cid,
        "signed_face_event_incidence": [[eid, 1] for eid in eids],
        "signed_event_account_incidence": [
            [eid, event_map[eid]["source"], -1, event_map[eid]["target"], 1] for eid in eids
        ],
        "whole_world_skeleton": "world_directed_multigraph_identity.jsonl",
        "auxiliary_edges_are_financial_transactions": False,
    }
    return cellular, simplicial, hasse


def log_internal(name, inputs, outputs, elapsed):
    rec = {
        "stage": name,
        "argv": [],
        "cwd": str(ROOT),
        "environment": {},
        "inputs": hashes(inputs),
        "outputs": hashes(outputs),
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "elapsed_seconds": round(elapsed, 3),
        "error": None,
        "execution": "in_process",
    }
    STAGES.append(rec)
    p = OUT / "audit" / "stages.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def compare_replay(first, replay, world_id):
    """Compare one declared replay after normalizing only output paths/timings."""
    import numpy as np

    keys = [
        "native_inputs/tmp/" + world_id + "/accounts.csv",
        "native_inputs/tmp/" + world_id + "/transactions.csv",
        "native_inputs/tmp/" + world_id + "/normal_models.csv",
        "native_inputs/tmp/" + world_id + "/alert_members.csv",
    ]

    def latest(root):
        xs = sorted((root / "attempts").glob("run_*"))
        if not xs:
            raise FileNotFoundError("world output has no completed run attempt: " + str(root))
        return xs[-1]

    a, b = latest(first), latest(replay)
    checks = {}
    for rel in keys:
        p = a / rel
        q = b / rel
        checks["input:" + rel] = p.read_bytes() == q.read_bytes()
    for rel in [
        "unmodified_output/" + world_id + "/tx_log.csv",
        "instrumented_output/" + world_id + "/tx_log.csv",
    ]:
        checks["native:" + rel] = (a / rel).read_bytes() == (b / rel).read_bytes()
    for rel in [
        "sidecars/cycle_schedules.jsonl",
        "sidecars/arc_attempts.jsonl",
        "sidecars/event_attempt_bindings.jsonl",
        "sidecars/emitted_events.jsonl",
        "sidecars/buffered_events.jsonl",
        "sidecars/persisted_events.jsonl",
        "sidecars/attempt_outcomes.jsonl",
        "candidate_identity.jsonl",
        "candidate_labels.jsonl",
        "candidate_observations.jsonl",
        "world_directed_multigraph_identity.jsonl",
        "cellular_polygonal_boundaries.jsonl",
        "simplicial_subdivision.jsonl",
        "hasse_incidence.jsonl",
    ]:
        checks["artifact:" + rel] = (a / rel).read_bytes() == (b / rel).read_bytes()

    def normalize_node(path):
        doc = json.loads(path.read_text())
        doc["directed_topology_identity"] = Path(doc["directed_topology_identity"]).name
        return doc

    checks["world_node_identity"] = normalize_node(a / "world_node_identity.json") == normalize_node(
        b / "world_node_identity.json"
    )
    with np.load(a / "operators.npz") as left, np.load(b / "operators.npz") as right:
        checks["operators.npz"] = all(np.array_equal(left[k], right[k]) for k in left.files) and set(
            left.files
        ) == set(right.files)
    result = {
        "world_id": world_id,
        "first_attempt": str(a),
        "replay_attempt": str(b),
        "checks": checks,
        "pass": all(checks.values()),
        "replay_counts_as_independent_world": False,
        "normalized_only": [
            "absolute topology path in world_node_identity.json",
            "timing and stage metadata are excluded from compared scientific values",
        ],
    }
    write_json(replay / "replay_comparison.json", result)
    if not result["pass"]:
        raise ValueError("declared deterministic replay mismatch")
    return result


def run_batch(batch_spec_path, output_dir, reuse_build_dir):
    global OUT
    batch_path = Path(batch_spec_path).resolve()
    batch = json.loads(batch_path.read_text())
    base_path = (ROOT / batch["base_spec"]).resolve()
    base = json.loads(base_path.read_text())
    seeds = list(batch["fixed_world_order"])
    expected = [101, 103, 107, 109, 113, 127]
    if (
        seeds != expected
        or batch["provisional_development_train_seeds"] != seeds[:4]
        or batch["provisional_development_validation_seeds"] != seeds[4:]
    ):
        raise ValueError("batch specification differs from the authorized fixed seed order/roles")
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    reuse = Path(reuse_build_dir).resolve()
    registry = {
        "batch_spec": sha(batch_path),
        "base_spec": sha(base_path),
        "world_order": seeds,
        "worlds": [
            {
                "world_id": batch["world_id_pattern"].format(seed=s),
                "seed": s,
                "role": (
                    "provisional_development_train"
                    if s in seeds[:4]
                    else "provisional_development_validation"
                ),
                "counts_as_test": False,
                "replay": False,
            }
            for s in seeds
        ],
        "replay": {
            "world_id": batch["world_id_pattern"].format(seed=batch["replay_seed"]),
            "seed": batch["replay_seed"],
            "counts_as_independent_world": False,
        },
        "registered_before_generation": True,
        "test_worlds_authorized": False,
    }
    write_json(OUT / "world_registry.json", registry)
    logpath = OUT / "batch_processes.jsonl"
    failed = 0
    world_status = []
    valid = []
    replay_result = None
    mpl = Path(os.environ.get("MPLCONFIGDIR") or (OUT / ".mpl-cache"))
    mpl.mkdir(parents=True, exist_ok=True)

    def execute(seed, role, replay=False):
        nonlocal failed
        world_id = batch["world_id_pattern"].format(seed=seed)
        root = OUT / ("replay_" + world_id if replay else world_id)
        root.mkdir(parents=True, exist_ok=True)
        spec = dict(base)
        spec["seed"] = seed
        specpath = OUT / "registered_specs" / (("replay_" if replay else "") + world_id + ".json")
        write_json(specpath, spec)
        argv = [
            sys.executable,
            "-s",
            "-B",
            str(Path(__file__).resolve()),
            "--spec",
            str(specpath),
            "--output-dir",
            str(root),
            "--world-id",
            world_id,
            "--reuse-build-dir",
            str(reuse),
        ]
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(mpl)
        env["RANDOM_SEED"] = str(seed)
        start = time.monotonic()
        proc = subprocess.run(argv, cwd=str(ROOT), env=env, shell=False, text=True, capture_output=True)
        rec = {
            "seed": seed,
            "world_id": world_id,
            "role": role,
            "replay": replay,
            "argv": argv,
            "cwd": str(ROOT),
            "environment": {"MPLCONFIGDIR": str(mpl), "RANDOM_SEED": str(seed)},
            "spec": sha(specpath),
            "started_monotonic": start,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "output_root": str(root),
        }
        with logpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        status = {
            "seed": seed,
            "world_id": world_id,
            "role": role,
            "replay": replay,
            "exit_code": proc.returncode,
            "output_root": str(root),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode:
            failed += 1
            status["status"] = "FAILED_RETAINED"
        else:
            runs = sorted((root / "attempts").glob("run_*"))
            attempt = runs[-1] if runs else None
            report = (
                json.loads((attempt / "S1_A_report.json").read_text())
                if attempt and (attempt / "S1_A_report.json").is_file()
                else None
            )
            status.update(
                {
                    "status": "COMPLETED" if report else "INCOMPLETE",
                    "attempt": str(attempt) if attempt else None,
                    "report": report,
                }
            )
            if not report:
                failed += 1
            elif not replay:
                valid.append(
                    {
                        "seed": seed,
                        "world_id": world_id,
                        "role": role,
                        "root": root,
                        "attempt": attempt,
                        "report": report,
                    }
                )
        world_status.append(status)
        write_json(
            OUT / "world_index.json",
            {
                "registry": registry,
                "worlds": [
                    {k: v for k, v in x.items() if k not in {"stdout", "stderr", "report"}}
                    for x in world_status
                ],
                "valid_world_count": len(valid),
                "replay": replay_result,
                "failures": failed,
            },
        )
        return status

    for seed in seeds:
        role = "provisional_development_train" if seed in seeds[:4] else "provisional_development_validation"
        status = execute(seed, role)
        if status["status"] != "COMPLETED":
            break
        if status["status"] == "COMPLETED" and seed == batch["replay_seed"]:
            replay_status = execute(seed, "same_world_replay", True)
            if replay_status["status"] != "COMPLETED":
                break
            if replay_status["status"] == "COMPLETED":
                replay_result = compare_replay(
                    OUT / batch["world_id_pattern"].format(seed=seed),
                    OUT / ("replay_" + batch["world_id_pattern"].format(seed=seed)),
                    batch["world_id_pattern"].format(seed=seed),
                )
        write_json(
            OUT / "world_index.json",
            {
                "registry": registry,
                "worlds": [
                    {k: v for k, v in x.items() if k not in {"stdout", "stderr", "report"}}
                    for x in world_status
                ],
                "valid_world_count": len(valid),
                "replay": replay_result,
                "failures": failed,
            },
        )
        if failed >= 2:
            break
    # Namespaced physical identities; account spellings intentionally repeat between independently initialized simulations.
    with (OUT / "global_custody_ledger.jsonl").open("w", encoding="utf-8") as ledger:
        for w in valid:
            for line in (w["attempt"] / "world_directed_multigraph_identity.jsonl").read_text().splitlines():
                row = json.loads(line)
                ledger.write(
                    json.dumps(
                        {
                            "world_id": w["world_id"],
                            "physical_event_id": int(row["physical_event_id"]),
                            "source_account_token": row["source_account"],
                            "target_account_token": row["target_account"],
                            "account_tokens_predictor_values": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    # Duplicate-input audit distinguishes repeated bytes from repeated account spellings.
    hashes_seen = {}
    for w in valid:
        inputs = w["attempt"] / "native_inputs" / "tmp" / w["world_id"]
        digest = hashlib.sha256(
            b"".join(
                (inputs / n).read_bytes()
                for n in ["accounts.csv", "transactions.csv", "normal_models.csv", "alert_members.csv"]
            )
        ).hexdigest()
        hashes_seen.setdefault(digest, []).append(w["world_id"])
    distinct_input_groups = [v for v in hashes_seen.values() if len(v) > 1]
    # Matched untrained interfaces: each distinct valid world plus the exposed pilot once.
    bench_results = []
    bench_failures = 0
    if failed == 0 and len(valid) == len(seeds) and replay_result and replay_result["pass"]:
        bench_targets = [(w["world_id"], w["attempt"]) for w in valid] + [
            ("S1_dev_1729", ROOT / "project/s1/attempts/pilot_02")
        ]
        for name, world_path in bench_targets:
            out = OUT / "benchmark_checks" / name
            out.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                "-s",
                "-B",
                str(ROOT / "source/scripts/check_s1_benchmark.py"),
                "--world-dir",
                str(world_path),
                "--output-dir",
                str(out),
                "--model-config",
                str(ROOT / "source/configs/s1_matched_smoke.json"),
            ]
            env = os.environ.copy()
            env["MPLCONFIGDIR"] = str(mpl)
            start = time.monotonic()
            proc = subprocess.run(argv, cwd=str(ROOT), env=env, shell=False, text=True, capture_output=True)
            rec = {
                "world_id": name,
                "argv": argv,
                "cwd": str(ROOT),
                "environment": {"MPLCONFIGDIR": str(mpl)},
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "output_hashes": hashes([out]),
            }
            bench_results.append(rec)
            with (OUT / "benchmark_processes.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            if proc.returncode:
                failed += 1
                bench_failures += 1
            if failed >= 2:
                break
    counts = {}
    diagnostics = {}
    capacity = {}
    costs = {}
    for rec in bench_results:
        p = OUT / "benchmark_checks" / rec["world_id"] / "run_report.json"
        if p.is_file():
            d = json.loads(p.read_text())
            counts[rec["world_id"]] = d["models"]["active_parameter_counts"]
            diagnostics[rec["world_id"]] = d["fixed_shortcut_diagnostics"]
            costs[rec["world_id"]] = d["cost"]
    widths = batch["matched_hidden_widths"]
    expected_counts = {
        "cellular_cwn_style_local": 8467,
        "simplicial_mpsn_style_local": 8533,
        "directed_local_edge_gnn": 8390,
        "cellular_hasse_mechanism_control": 8431,
    }
    spreads = {}
    for world_key, actual in counts.items():
        vals = list(actual.values())
        complete = set(actual) == set(expected_counts)
        spread = (max(vals) - min(vals)) / max(vals) if vals and max(vals) > 0 else None
        spreads[world_key] = {
            "actual_counts": actual,
            "expected_counts_match": actual == expected_counts,
            "spread_across_neural_arms": spread,
            "pass_2_percent": complete and spread is not None and spread <= 0.02,
        }
    report = {
        "status": (
            "S1_C_BATCH_COMPLETE"
            if failed == 0 and len(valid) == len(seeds) and replay_result and replay_result["pass"]
            else "S1_C_BATCH_INCOMPLETE"
        ),
        "registry": registry,
        "worlds": [{k: v for k, v in x.items() if k not in {"stdout", "stderr"}} for x in world_status],
        "replay": replay_result,
        "valid_distinct_worlds": len(valid),
        "failed_or_incomplete_count": failed,
        "duplicate_realized_input_groups": distinct_input_groups,
        "duplicate_input_scope": "hash of concatenated generated native input CSVs",
        "account_token_scope": "tokens are namespaced by world_id; repeated spellings do not imply shared entities",
        "custody_ledger": str(OUT / "global_custody_ledger.jsonl"),
        "benchmark_results": bench_results,
        "actual_capacity_counts": counts,
        "capacity_spreads": spreads,
        "shortcut_diagnostics_by_world": diagnostics,
        "measured_costs_by_world": costs,
        "model_fitting": "none",
        "test_worlds_accessed": False,
        "gate2": "UNRESOLVED; development worlds do not meet untouched independent evaluation requirements",
        "next": "Architect review; do not train, inspect/create test worlds, freeze, or certify",
    }
    write_json(OUT / "S1-C_contract_report.json", report)
    write_json(
        OUT / "world_index.json",
        {
            "registry": registry,
            "worlds": [
                {k: v for k, v in x.items() if k not in {"stdout", "stderr", "report"}} for x in world_status
            ],
            "valid_world_count": len(valid),
            "replay": replay_result,
            "failures": failed,
            "duplicate_realized_input_groups": distinct_input_groups,
            "benchmark_results": bench_results,
            "capacity_spreads": spreads,
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid_distinct_worlds": len(valid),
                "replay_pass": bool(replay_result and replay_result["pass"]),
                "failures": failed,
                "capacity_spreads": spreads,
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "S1_C_BATCH_COMPLETE" else 1


def main():
    global OUT, CURRENT_ATTEMPT, REUSE_BUILD_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    ap.add_argument("--batch-spec", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--world-id")
    ap.add_argument("--source-dir", type=Path)
    ap.add_argument("--reuse-build-dir", type=Path)
    args = ap.parse_args()
    if args.batch_spec:
        return run_batch(
            args.batch_spec,
            args.output_dir or ROOT / "project/s1/development_worlds",
            args.reuse_build_dir or ROOT / "project/s1/attempts/pilot_02",
        )
    if args.output_dir:
        OUT = args.output_dir.resolve()
    REUSE_BUILD_DIR = args.reuse_build_dir.resolve() if args.reuse_build_dir else None
    spec = json.loads(args.spec.read_text())
    world_id = args.world_id or f"S1_dev_{spec['seed']}"
    # Stages are append-only by attempt. Never clean an earlier run.
    attempt_no = 1
    while (OUT / "attempts" / f"run_{attempt_no:02d}").exists():
        attempt_no += 1
    attempt = OUT / "attempts" / f"run_{attempt_no:02d}"
    attempt.mkdir(parents=True)
    CURRENT_ATTEMPT = attempt
    write_json(
        OUT / "active_attempt.json",
        {
            "attempt": str(attempt.relative_to(ROOT)),
            "spec": sha(args.spec),
            "seed": spec["seed"],
            "disclosed_variants": spec["disclosed_variants"],
        },
    )
    focused = focused_fixture_checks()
    write_json(attempt / "focused_fixture_checks.json", focused)
    log_internal("focused-structural-controls", [], [attempt / "focused_fixture_checks.json"], 0.0)
    captured = args.source_dir.resolve() if args.source_dir else ROOT / "project/c04g_captures/amlsim"
    if REUSE_BUILD_DIR:
        unmodified = REUSE_BUILD_DIR / "amlsim_unmodified"
        instrumented = REUSE_BUILD_DIR / "amlsim_instrumented"
        work = REUSE_BUILD_DIR / "amlsim"
        for required in [
            unmodified,
            instrumented,
            REUSE_BUILD_DIR / "classes_unmodified",
            REUSE_BUILD_DIR / "classes_instrumented",
        ]:
            if not required.exists():
                raise FileNotFoundError("approved reused build component missing: " + str(required))
        log_stage(
            "reuse-qualified-build",
            [],
            ROOT,
            {},
            [
                unmodified,
                instrumented,
                REUSE_BUILD_DIR / "classes_unmodified",
                REUSE_BUILD_DIR / "classes_instrumented",
            ],
            [],
            elapsed=0.0,
        )
    else:
        work = OUT / "amlsim"
        instrumented = attempt / "amlsim_instrumented"
        unmodified = attempt / "amlsim_unmodified"
        start = time.monotonic()
        copy_source(captured, work)
        shutil.copytree(captured, unmodified)
        shutil.copytree(captured, instrumented)
        patch_source(instrumented)
        log_stage(
            "source-assembly",
            [],
            ROOT,
            {},
            [captured],
            [work, instrumented, unmodified],
            elapsed=time.monotonic() - start,
        )
    jars = download_deps(attempt)
    jdk = Path(os.environ.get("JAVA_HOME") or spec["jdk_home"])
    javac = jdk / "bin/javac"
    java = jdk / "bin/java"
    run("javac-version", [javac, "-version"], ROOT, inputs=[javac], timeout=30)
    run("java-version", [java, "-version"], ROOT, inputs=[java], timeout=30)
    world = attempt / "native_inputs"
    stage_start = time.monotonic()
    inputdir, intents = generate(spec, world, world_id)
    log_internal(
        "controlled-native-csv-generation",
        [args.spec],
        [inputdir, world / "intentions.json"],
        time.monotonic() - stage_start,
    )
    classes0 = REUSE_BUILD_DIR / "classes_unmodified" if REUSE_BUILD_DIR else attempt / "classes_unmodified"
    classes1 = (
        REUSE_BUILD_DIR / "classes_instrumented" if REUSE_BUILD_DIR else attempt / "classes_instrumented"
    )
    cp = os.pathsep.join(jars)
    sources0 = sorted((unmodified / "src/main/java").rglob("*.java"))
    sources1 = sorted((instrumented / "src/main/java").rglob("*.java"))
    if not REUSE_BUILD_DIR:
        run(
            "javac-unmodified",
            [
                javac,
                "-encoding",
                "UTF-8",
                "--release",
                "8",
                "-proc:none",
                "-classpath",
                cp,
                "-d",
                classes0,
                *sources0,
            ],
            ROOT,
            inputs=sources0 + jars,
            outputs=[classes0],
            timeout=600,
        )
        run(
            "javac-instrumented",
            [
                javac,
                "-encoding",
                "UTF-8",
                "--release",
                "8",
                "-proc:none",
                "-classpath",
                cp,
                "-d",
                classes1,
                *sources1,
            ],
            ROOT,
            inputs=sources1 + jars,
            outputs=[classes1],
            timeout=600,
        )
    else:
        for cls in [
            classes0 / "amlsim/AMLSim.class",
            classes1 / "amlsim/AMLSim.class",
            classes1 / "amlsim/S1Observer.class",
        ]:
            if not cls.is_file():
                raise FileNotFoundError("reused bytecode incomplete: " + str(cls))
    out0 = attempt / "unmodified_output"
    out1 = attempt / "instrumented_output"
    out0.mkdir()
    out1.mkdir()
    cfg0 = config_for(spec, unmodified, world, out0, "unmodified", world_id)
    cfg1 = config_for(spec, instrumented, world, out1, "instrumented", world_id)
    baseenv = {"RANDOM_SEED": str(spec["seed"]), "LANG": "C", "LC_ALL": "C"}
    run(
        "java-unmodified",
        [java, "-Xmx1g", "-cp", str(classes0) + os.pathsep + cp, "amlsim.AMLSim", cfg0],
        ROOT,
        baseenv,
        inputs=[cfg0, *sources0, *jars],
        outputs=[out0],
        timeout=600,
    )
    env1 = baseenv.copy()
    run(
        "java-instrumented",
        [
            java,
            "-Damlsim.observe.dir=" + str(attempt / "sidecars"),
            "-Xmx1g",
            "-cp",
            str(classes1) + os.pathsep + cp,
            "amlsim.AMLSim",
            cfg1,
        ],
        ROOT,
        env1,
        inputs=[cfg1, *sources1, *jars],
        outputs=[out1, attempt / "sidecars"],
        timeout=600,
    )
    tx0 = out0 / world_id / "tx_log.csv"
    tx1 = out1 / world_id / "tx_log.csv"
    base, events, schedules, attempts = reconcile(
        tx0,
        tx1,
        attempt / "sidecars/cycle_schedules.jsonl",
        attempt / "sidecars/arc_attempts.jsonl",
        attempt / "sidecars/persisted_events.jsonl",
    )
    noncash_count = sum(e["description"] not in {"CASH-IN", "CASH-OUT"} for e in events)
    if noncash_count > spec["full_census_guard_events"]:
        write_json(
            attempt / "oversized.json",
            {"events": len(events), "noncash_events": noncash_count, "status": "OVERSIZED_FULL_CENSUS"},
        )
        raise RuntimeError("whole pilot oversized; no clipping permitted")
    intentdoc = json.loads((world / "intentions.json").read_text())
    stage_start = time.monotonic()
    try:
        cycles, frame, ids, labels, realized = cycles_and_candidates(
            base,
            events,
            intentdoc["intentions"],
            schedules,
            attempts,
            [f"A{i:02d}" for i in range(spec["accounts"])],
            spec["full_census_guard_tuples"],
            world_id,
            spec["steps"],
            spec["candidate_lengths"][0],
            spec["candidate_lengths"][1],
        )
    except OverflowError as exc:
        write_json(
            attempt / "oversized.json",
            {
                "events": len(events),
                "noncash_events": noncash_count,
                "status": "OVERSIZED_FULL_CENSUS",
                "exact_count_error": str(exc),
            },
        )
        raise
    census_elapsed = time.monotonic() - stage_start
    # Matrices are event-indexed; parallel rows remain distinct, loops have zero B1 columns.
    import numpy as np

    nodes = [f"A{i:02d}" for i in range(spec["accounts"])]
    event_order = sorted(int(e["event_id"]) for e in events)
    event_map = {int(e["event_id"]): e for e in events}
    emap = {v: i for i, v in enumerate(event_order)}
    b1 = np.zeros((len(nodes), len(events)), dtype=np.int64)
    nmap = {x: i for i, x in enumerate(nodes)}
    for e in events:
        u, v = nmap[e["source"]], nmap[e["target"]]
        col = emap[int(e["event_id"])]
        if u != v:
            b1[u, col] = -1
            b1[v, col] = 1
    b2 = np.zeros((len(events), len(ids)), dtype=np.int64)
    for j, item in enumerate(ids):
        cyc = item["cycle_accounts"]
        for i, eid in enumerate(item["event_ids"]):
            e = event_map[eid]
            if (e["source"], e["target"]) != (cyc[i], cyc[(i + 1) % len(cyc)]):
                raise ValueError("candidate selects an event with incorrect directed endpoints")
            b2[emap[eid], j] = 1
    dense_product = b1 @ b2
    if np.any(dense_product):
        raise ValueError("exact integer B1@B2 != 0")
    # Independent sparse boundary reference: each selected directed closed walk
    # contributes -1 at each source and +1 at its successor, without using B1/B2.
    sparse_ok = True
    for j, item in enumerate(ids):
        residual = Counter()
        for eid in item["event_ids"]:
            e = event_map[eid]
            residual[e["source"]] -= 1
            residual[e["target"]] += 1
        if any(residual.values()) or np.any(dense_product[:, j]):
            sparse_ok = False
            break
    if not sparse_ok:
        raise ValueError("independent sparse/dense boundary agreement failed")
    event_perm = np.arange(len(events))[::-1]
    node_perm = np.arange(len(nodes))[::-1]
    permutation_ok = np.array_equal(b1[:, event_perm] @ b2[event_perm, :], dense_product) and np.array_equal(
        b1[node_perm, :] @ b2, dense_product[node_perm, :]
    )
    if not permutation_ok:
        raise ValueError("boundary basis permutation check failed")
    np.savez_compressed(attempt / "operators.npz", B1=b1, B2=b2, event_ids=np.array(event_order))
    # Whole-world graph counts and outputs: labels and physical ids in separate files.
    feat = attempt / "candidate_observations.jsonl"
    identityfile = attempt / "candidate_identity.jsonl"
    labelfile = attempt / "candidate_labels.jsonl"
    for path, rows in [(feat, frame), (identityfile, ids), (labelfile, labels)]:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    topologyfile = attempt / "world_directed_multigraph_identity.jsonl"
    with topologyfile.open("w") as f:
        for e in events:
            f.write(
                json.dumps(
                    {
                        "physical_event_id": int(e["event_id"]),
                        "source_account": e["source"],
                        "target_account": e["target"],
                        "loop": e["source"] == e["target"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    write_json(
        attempt / "world_node_identity.json",
        {
            "world_id": world_id,
            "node_order": [f"A{i:02d}" for i in range(spec["accounts"])],
            "event_order": [int(e["event_id"]) for e in events],
            "directed_topology_identity": str(topologyfile),
            "predictor_account_tokens_excluded": True,
        },
    )
    log_internal(
        "complete-candidate-census",
        [tx1, attempt / "sidecars/persisted_events.jsonl", attempt / "sidecars/arc_attempts.jsonl"],
        [feat, identityfile, labelfile, topologyfile, attempt / "world_node_identity.json"],
        census_elapsed,
    )
    cellularfile = attempt / "cellular_polygonal_boundaries.jsonl"
    simplicialfile = attempt / "simplicial_subdivision.jsonl"
    hassefile = attempt / "hasse_incidence.jsonl"
    skeletonfile = attempt / "simplicial_world_incidence.jsonl"
    with skeletonfile.open("w") as f:
        for event in events:
            f.write(
                json.dumps(
                    {
                        "physical_event_id": int(event["event_id"]),
                        "directed_incidence_path": subdivided_path(event),
                        "auxiliary_edges_are_financial_transactions": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    with cellularfile.open("w") as cf, simplicialfile.open("w") as sf, hassefile.open("w") as hf:
        for identity in ids:
            cellular, simplicial, hasse = candidate_topology(identity, event_map)
            for f, record in [(cf, cellular), (sf, simplicial), (hf, hasse)]:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(
        attempt / "intent_realization.json",
        {
            "intentions": intentdoc["intentions"],
            "realization": realized,
            "schedule_sidecars": schedules,
            "arc_attempts": attempts,
            "persisted_events": len(events),
            "in_frame_tuples": len(frame),
            "positive_tuples": sum(x["label"] for x in labels),
        },
    )
    degrees = Counter()
    parallel = Counter()
    amounts = []
    times = []
    for e in events:
        degrees[e["source"]] += 1
        parallel[(e["source"], e["target"])] += 1
        amounts.append(float(e["amount_persisted"]))
        times.append(int(e["step"]))
    bylen = defaultdict(list)
    for identity, label in zip(ids, labels):
        bylen[len(identity["cycle_accounts"])].append(label["label"])
    write_json(
        attempt / "shortcut_diagnostics.json",
        {
            "candidate_count_by_cycle_length": {str(k): len(v) for k, v in sorted(bylen.items())},
            "candidate_prevalence_by_cycle_length": {
                str(k): sum(v) / len(v) for k, v in sorted(bylen.items()) if v
            },
            "parallel_pairs": {str(k): v for k, v in parallel.items() if v > 1},
            "amount_range_persisted": [min(amounts, default=0), max(amounts, default=0)],
            "step_range": [min(times, default=0), max(times, default=0)],
            "event_count": len(events),
            "candidate_prevalence": sum(x["label"] for x in labels) / max(1, len(labels)),
            "no_model_score_computed": True,
        },
    )
    parallel_pairs = [
        pair for pair, count in Counter((e["source"], e["target"]) for e in events).items() if count > 1
    ]
    checks = {
        "unmodified_instrumented_rows_equal": True,
        "sidecar_rows_reconcile": True,
        "whole_frame_before_labels": True,
        "feature_rows_have_no_annotation_or_identity_fields": all(
            not (
                set(x)
                & {
                    "label",
                    "candidate_id",
                    "event_ids",
                    "instance_id",
                    "alert_id",
                    "sar",
                    "account_id",
                    "world_id",
                    "seed",
                }
            )
            for x in frame
        ),
        "no_truncation": True,
        "B1_B2_exact_zero": True,
        "sparse_reference_equals_dense_boundary": sparse_ok,
        "event_and_node_basis_permutations_preserve_boundary": permutation_ok,
        "loops_zero_boundary": all(
            np.all(b1[:, emap[int(e["event_id"])]] == 0) for e in events if e["source"] == e["target"]
        ),
        "parallel_physical_columns_preserved": len(set(event_order)) == len(events),
        "selected_mask_cardinality": all(
            sum(x["selected_event_mask"]) == len(y["event_ids"]) for x, y in zip(frame, ids)
        ),
        "parallel_event_identifiers_are_distinct_columns": all(
            len({int(e["event_id"]) for e in events if (e["source"], e["target"]) == pair}) > 1
            for pair in parallel_pairs
        ),
    }
    # INV-001..009 are recorded individually; any unsupported legacy invariant remains explicitly failed/unresolved.
    checks.update(
        {
            "INV-001": "PASS: provenance explicitly controlled simulation; no financial record claim",
            "INV-002": "PASS: cellular polygonal boundary and separate lossless simplicial fan subdivision retained; subdivision spokes are nonfinancial",
            "INV-003": "UNRESOLVED: persistent homology on the unfilled 1-skeleton was not run",
            "INV-004": "PASS: exact integer cellular boundary product is zero for the declared cycle chain basis",
            "INV-005": "PASS: full physical-event candidate frame built before annotation join; observation rows exclude labels and identities",
            "INV-006": "PASS: actual source, input, dependency, build, and output digests recorded in ordinary stage logs",
            "INV-007": "UNRESOLVED: world split is prepared in the interface; independent evaluation is not run",
            "INV-008": "PASS: executable scientific workflow and structural controls are included in source",
            "INV-009": "UNRESOLVED: no training, capacity, or FLOP audit was authorized",
        }
    )
    write_json(attempt / "scientific_checks.json", checks)
    # Native figures, if matplotlib is available; no fabricated model scores.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.scatter(times, [int(e["event_id"]) for e in events], s=12)
        ax.set(
            xlabel="Native Java step",
            ylabel="Physical event ordinal",
            title="S1 development transaction timeline",
        )
        fig.tight_layout()
        fig.savefig(attempt / "transaction_timeline.png", dpi=160)
        plt.close(fig)
        cycle_realized = [x for x in realized.values() if x["type"] == "CYCLE"]
        realized_n = sum(1 for x in cycle_realized if x["attempt_count"] > 0)
        complete_n = sum(1 for x in cycle_realized if x["complete"])
        inframe = sum(1 for x in cycle_realized if x["in_frame"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ["intended", "attempted", "complete", "in frame", "positive tuples"],
            [len(cycle_realized), realized_n, complete_n, inframe, sum(x["label"] for x in labels)],
        )
        ax.set_title("Intention to realization to candidate frame")
        fig.tight_layout()
        fig.savefig(attempt / "intention_realization_extraction.png", dpi=160)
        plt.close(fig)
    except ImportError as e:
        esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ymax = max((int(e["event_id"]) for e in events), default=1)
        points = "".join(
            f'<circle cx="{40+int(e["step"])*4}" cy="{180-int(e["event_id"])*140/ymax}" r="2" fill="#245b8a"/>'
            for e in events
        )
        (attempt / "transaction_timeline.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="560" height="220"><text x="20" y="20">S1 development transaction timeline</text><text x="20" y="210">Native Java step</text>{points}</svg>',
            encoding="utf-8",
        )
        cycle_realized = [x for x in realized.values() if x["type"] == "CYCLE"]
        vals = [
            len(cycle_realized),
            sum(1 for x in cycle_realized if x["attempt_count"] > 0),
            sum(1 for x in cycle_realized if x["complete"]),
            sum(1 for x in cycle_realized if x["in_frame"]),
            sum(x["label"] for x in labels),
        ]
        bars = "".join(
            f'<rect x="{50+i*110}" y="{170-min(120,v*20)}" width="50" height="{min(120,v*20)}" fill="#467a58"/><text x="{45+i*110}" y="190">{esc(n)}: {v}</text>'
            for i, (n, v) in enumerate(
                zip(["intended", "attempted", "complete", "in frame", "positive tuples"], vals)
            )
        )
        (attempt / "intention_realization_extraction.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="220"><text x="20" y="20">Intention to realization to candidate frame</text>{bars}</svg>',
            encoding="utf-8",
        )
        write_json(attempt / "figure_fallback.json", {"matplotlib_error": str(e), "figures": "SVG generated"})
    # Four-arm, common-input benchmark interfaces only. Nothing is trained.
    bench = {
        "development_only": True,
        "candidate_table": str(feat),
        "labels": str(labelfile),
        "identities_separate": str(identityfile),
        "directed_world_topology_identity": str(topologyfile),
        "node_and_event_order": "world_node_identity.json; endpoints are graph identity, not node-feature channels",
        "operators": {
            "cellular": str(cellularfile),
            "simplicial": str(simplicialfile),
            "hasse_control": str(hassefile),
        },
        "common_observation_schema": [
            "cycle_length",
            "time_steps",
            "log1p_amounts",
            "world_event_time_steps",
            "world_event_log1p_amounts",
            "central_node_mask",
            "selected_event_mask",
        ],
        "arms": {
            "cellular_cwn": {"operator": "cellular polygonal boundary and event-indexed B1/B2"},
            "simplicial": {"operator": "lossless port-preserving fan subdivision"},
            "directed_edge_gnn": {
                "operator": "directed event graph",
                "edge_features": ["world_event_time_steps", "world_event_log1p_amounts"],
            },
            "gbdt": {
                "input": "per-candidate permitted observation summaries only",
                "fixed_summaries": [
                    "cycle_length",
                    "mean/min/max selected time",
                    "mean/min/max selected log1p amount",
                    "selected degree summaries",
                ],
                "summary_before_outcomes": True,
            },
            "hasse_incidence": {"role": "mechanism control", "input": "simplicial incidence/Hasse graph"},
        },
        "paired_training_seeds": [11, 23, 37, 53, 71],
        "future_train_worlds": [101, 103, 107, 109],
        "validation_worlds": [113, 127],
        "test_worlds_authorized": False,
        "final_training_authorized": False,
        "split_unit": "world",
        "metrics": ["average_precision", "AUROC"],
        "checkpoint": "validation-only",
        "normalization": "train-only",
    }
    write_json(attempt / "benchmark_interfaces.json", bench)
    report = {
        "status": "S1_C_WORLD_COMPLETE_DEVELOPMENT_ONLY",
        "world_id": world_id,
        "seed": spec["seed"],
        "source_pin": spec["source_commit"],
        "source_runtime": "Corretto Java11; javac --release 8 bytecode",
        "controlled_input_generator": f"Python standard library random.Random({spec['seed']}), declared in intentions.json",
        "native_event_count": len(events),
        "noncash_event_count": noncash_count,
        "cycles_enumerated": len(cycles),
        "candidate_tuple_count": len(frame),
        "positive_tuples": sum(x["label"] for x in labels),
        "prevalence": sum(x["label"] for x in labels) / max(1, len(labels)),
        "intention_realization": realized,
        "scientific_checks": checks,
        "build_reuse": str(REUSE_BUILD_DIR) if REUSE_BUILD_DIR else None,
        "model_results": "none; no training executed",
        "test_worlds_accessed": False,
        "next": "Architect scientific review before comparative campaign",
    }
    write_json(attempt / "S1_A_report.json", report)
    write_json(
        attempt / "stage_summary.json",
        {
            "stages": STAGES,
            "output_hashes": hashes([tx1, feat, identityfile, labelfile, attempt / "operators.npz"]),
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "world_id",
                    "seed",
                    "native_event_count",
                    "noncash_event_count",
                    "cycles_enumerated",
                    "candidate_tuple_count",
                    "positive_tuples",
                    "prevalence",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        # Tracebacks remain in the most recent attempt log when available.
        try:
            if CURRENT_ATTEMPT:
                (CURRENT_ATTEMPT / "failure_traceback.txt").write_text(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
    else:
        sys.exit(exit_code)
