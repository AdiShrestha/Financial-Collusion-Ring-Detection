#!/usr/bin/env python3
"""Bounded exploratory learning on the registered S1 development worlds."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn, scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS=("cellular_cwn_style_local","simplicial_mpsn_style_local","directed_local_edge_gnn","cellular_hasse_mechanism_control")
SEEDS=(11,23,37,53,71)
RATES=(1e-4,1e-3,1e-2)
LEAVES=(3,7,15)
TRAIN_IDS=("S1_dev_101","S1_dev_103","S1_dev_107","S1_dev_109")
VAL_IDS=("S1_dev_113","S1_dev_127")
MAX_SECONDS=14400
MAX_BYTES=2147483648
START=time.monotonic()
CAMPAIGN_START_EPOCH=None
STOP_REASON=None

class BoundReached(Exception): pass

def strict_pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError(f"duplicate JSON object key: {k}")
        out[k]=v
    return out

def reject_constant(s): raise ValueError(f"nonfinite JSON constant: {s}")
def finite_json_float(text):
    value=float(text)
    if not math.isfinite(value): raise ValueError(f"nonfinite JSON number: {text}")
    return value
def strict_json(text): return json.loads(text,object_pairs_hook=strict_pairs,parse_constant=reject_constant,parse_float=finite_json_float)
def read_json(path): return strict_json(path.read_text(encoding="utf-8"))
def read_jsonl(path):
    result=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: result.append(strict_json(line))
            except Exception as e: raise ValueError(f"{path}:{n}: {type(e).__name__}: {e}") from e
    return result

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def atomic_torch(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256(); n=0
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b); n+=len(b)
    return {"path":str(path),"bytes":n,"sha256":h.hexdigest()}

def output_bytes(root): return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
def guard(root):
    elapsed=time.time()-CAMPAIGN_START_EPOCH if CAMPAIGN_START_EPOCH is not None else time.monotonic()-START
    if elapsed>=MAX_SECONDS: raise BoundReached("four-hour wall-time ceiling")
    if output_bytes(root)>=MAX_BYTES: raise BoundReached("2-GiB new-output ceiling")

def load_bench():
    p=Path(__file__).with_name("check_s1_benchmark.py")
    spec=importlib.util.spec_from_file_location("s1_benchmark_helpers",p)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod

def label_value(x):
    y=x.get("label")
    if type(y) is not int or y not in (0,1): raise ValueError(f"label must be JSON integer 0 or 1, got {y!r}")
    return y

def exact_id(x,where):
    z=x.get("candidate_id")
    if type(z) is not str or not z: raise ValueError(f"invalid candidate_id in {where}")
    return z

def strict_inputs(bench,world_id,attempt,role):
    names=("candidate_observations.jsonl","candidate_identity.jsonl","candidate_labels.jsonl")
    paths=[attempt/n for n in names]
    for p in paths:
        if not p.is_file() or p.is_symlink(): raise ValueError(f"missing or symlinked input: {p}")
    obs,ids,labs=[read_jsonl(p) for p in paths]
    if not(len(obs)==len(ids)==len(labs)==78): raise ValueError(f"{world_id} must contain the full 78-row census")
    # Observations intentionally contain no IDs. Their row alignment is checked
    # against the separate identity/representation maps by the input builder.
    iid=[exact_id(x,"identity") for x in ids]; lid=[exact_id(x,"labels") for x in labs]
    for label,keys in [("identity",iid),("label",lid)]:
        if len(set(keys))!=len(keys): raise ValueError(f"duplicate candidate IDs in {world_id} {label} rows")
    if set(iid)!=set(lid): raise ValueError(f"candidate ID join mismatch in {world_id}")
    for identity in ids:
        event_ids=identity.get("event_ids")
        if not isinstance(event_ids,list) or not event_ids or any(type(x) is not int or x<=0 for x in event_ids) or len(set(event_ids))!=len(event_ids):
            raise ValueError(f"invalid or duplicate physical event IDs in {world_id}")
    if any(x.get("world_id")!=world_id for x in ids): raise ValueError(f"world identity mismatch in {world_id}")
    forbidden=set(bench.FORBIDDEN)
    if any(forbidden.intersection(x) for x in obs): raise ValueError(f"forbidden identity/label fields in {world_id} observations")
    labels_by={exact_id(x,"labels"):label_value(x) for x in labs}
    for name in ("cellular_polygonal_boundaries.jsonl","simplicial_subdivision.jsonl","hasse_incidence.jsonl","world_directed_multigraph_identity.jsonl"):
        read_jsonl(attempt/name)
    read_json(attempt/"world_node_identity.json")
    # The model input builder independently validates event identities, masks, coordinate maps and boundaries.
    data=bench.load_inputs(attempt)
    order=[exact_id(x,"identity") for x in data["ids"]]
    if set(order)!=set(labels_by): raise ValueError(f"builder-to-label join mismatch in {world_id}")
    rows=data["rows"]
    joined=[labels_by[k] for k in order]
    if len(set(joined))<2: raise ValueError(f"single-class {role} world {world_id}; retain as invalid, do not replace")
    n=len(data["node_order"]); e=len(data["event_order"]); views=[]
    for i,row in enumerate(rows):
        b1,b2=bench.cellular(row,n,e); si=bench.simplicial(row,n,e)
        if not bench.closed(b1,b2): raise ValueError(f"cellular B1@B2 failed: {world_id} row {i}")
        if not bench.closed(si[3],si[4]): raise ValueError(f"simplicial closure failed: {world_id} row {i}")
        views.append({"r":row,"cell":(b1,b2),"simp":si,"label":joined[i],"candidate_id":order[i],"instance_id":next((x.get("instance_id") for x in labs if x["candidate_id"]==order[i]),None)})
    return {"world_id":world_id,"role":role,"attempt":str(attempt),"data":data,"views":views,"labels":joined,"candidates":order}

def fixture_qualification(bench):
    # Small explicit mathematical structures exercise actual operator constructors.
    def row(src,dst,selected,cycle_idx,cycle_length):
        e=len(src); n=5
        return {"src":np.asarray(src,dtype=np.int64),"dst":np.asarray(dst,dtype=np.int64),"selected":np.asarray(selected,dtype=np.float32),"central":np.asarray([1,1,1,1,1],dtype=np.float32),"times":np.linspace(0,1,e,dtype=np.float32),"amounts":np.linspace(1,2,e,dtype=np.float32),"selected_idx":np.flatnonzero(selected),"cycle_idx":np.asarray(cycle_idx,dtype=np.int64),"cycle_length":float(cycle_length)}
    fixture=row([0,1,1,0,0,1,2,3,4,4],[1,0,1,2,1,2,0,4,3,4],[1,0,0,0,0,1,1,0,0,0],[0,5,6],3)
    b1,b2=bench.cellular(fixture,5,len(fixture["src"]))
    sv=bench.simplicial(fixture,5,len(fixture["src"]))
    assert bench.closed(b1,b2) and bench.closed(sv[3],sv[4])
    # Filled triangle and square (two triangles), plus disconnected edge and empty rank.
    tri_b1=np.asarray([[-1,0,1],[1,-1,0],[0,1,-1]],dtype=np.int64); tri_b2=np.ones((3,1),dtype=np.int64)
    sq_b1=np.zeros((4,5),dtype=np.int64)
    for j,(u,v) in enumerate([(0,1),(1,2),(2,3),(3,0),(0,2)]): sq_b1[u,j]=-1;sq_b1[v,j]=1
    sq_b2=np.asarray([[1,0],[1,0],[0,1],[0,1],[-1,1]],dtype=np.int64)
    malformed=np.asarray([[1],[1],[0]],dtype=np.int64)
    assert bench.closed(tri_b1,tri_b2) and bench.closed(sq_b1,sq_b2)
    assert bench.closed(np.zeros((0,0),dtype=np.int64),np.zeros((0,0),dtype=np.int64))
    assert not bench.closed(tri_b1,malformed)
    assert np.array_equal(b1[:,0],b1[:,4])  # distinct parallel physical columns
    assert np.array_equal(b1[:,0],-b1[:,1])  # reciprocal physical directions
    assert not b1[:,2].any() and not b1[:,9].any()  # retained loops
    for lower,upper in ((b1,b2),(sv[3],sv[4]),(sq_b1,sq_b2)):
        lap=lower.T@lower+upper@upper.T
        assert np.allclose(lap,lap.T) and np.linalg.eigvalsh(lap).min()>=-1e-8
    # Parallel and reciprocal physical events remain distinct columns; loop has zero cellular boundary.
    event_src=[0,1,1,0,0]; event_dst=[1,0,1,2,1]
    loop_src=[2]; loop_dst=[2]
    loop=row(loop_src,loop_dst,[1],[0],1); lb1,lb2=bench.cellular(loop,5,1)
    assert lb1.shape==(5,1) and not lb1.any() and bench.closed(lb1,lb2)
    return {"status":"PASS","checks":{"actual_directed_reciprocal_parallel_loop_maps":"PASS","actual_cellular_and_simplicial_closure":"PASS","filled_triangle":"PASS","two_triangle_square":"PASS","empty_rank":"PASS","disconnected_components_in_directed_fixture":"PASS","malformed_boundary_rejected":"PASS"},"fixture_is_not_performance_evidence":True,"events_are_physical_coordinates_not_auxiliary_segments":True}

def make_models(bench,widths):
    return {"cellular_cwn_style_local":bench.Cell(widths["cellular_cwn_style_local"]),"simplicial_mpsn_style_local":bench.Simp(widths["simplicial_mpsn_style_local"]),"directed_local_edge_gnn":bench.Directed(widths["directed_local_edge_gnn"]),"cellular_hasse_mechanism_control":bench.Cell(widths["cellular_hasse_mechanism_control"],hasse=True)}
def kind_for(name): return "simplicial" if name.startswith("simplicial") else "directed" if name.startswith("directed") else "hasse" if "hasse" in name else "cellular"
def args_for(bench,v,name):
    x=bench.build_args(v,name)
    return x

def metrics(bench,labels,logits):
    z=np.asarray(logits,dtype=float); y=np.asarray(labels,dtype=int)
    if z.shape!=y.shape or not np.isfinite(z).all() or not np.isin(y,[0,1]).all():
        raise ValueError("invalid labels or nonfinite model logits")
    # Stable BCE agrees with the optimizer objective. Logits preserve ranking
    # when floating-point sigmoid probabilities saturate at zero/one.
    return {"bce":float(np.mean(np.logaddexp(0,z)-y*z)),"average_precision_grouped_threshold":bench.grouped_ap(y,z),"auroc_tie_aware":bench.tie_auc(y,z)}

def eval_worlds(bench,model,worlds,name):
    out={"inference_seconds_by_world":{}}
    model.eval()
    with torch.no_grad():
        for w in worlds:
            prediction_started=time.monotonic()
            logits=[float(model(*args_for(bench,v,name)).item()) for v in w["views"]]
            out["inference_seconds_by_world"][w["world_id"]]=time.monotonic()-prediction_started
            out[w["world_id"]]={**metrics(bench,w["labels"],logits),"logits":logits}
    out["equal_world_bce"]=float(np.mean([out[w["world_id"]]["bce"] for w in worlds]))
    out["equal_world_ap"]=float(np.mean([out[w["world_id"]]["average_precision_grouped_threshold"] for w in worlds]))
    out["equal_world_auc"]=float(np.mean([out[w["world_id"]]["auroc_tie_aware"] for w in worlds]))
    return out

def save_trial_trace(path,history):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for x in history: f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)

def rng_state(): return {"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate()}
def restore_rng(s): torch.set_rng_state(s["torch"]);np.random.set_state(s["numpy"]);random.setstate(s["python"])

def train_epoch(bench,model,opt,worlds,name):
    model.train(); opt.zero_grad(set_to_none=True)
    for w in worlds:
        losses=[]
        for v in w["views"]:
            y=torch.tensor(float(v["label"]),dtype=torch.float32)
            logit=model(*args_for(bench,v,name))
            if not torch.isfinite(logit).all(): raise FloatingPointError(f"nonfinite logit: {name} {w["world_id"]} {v["candidate_id"]}")
            losses.append(nn.functional.binary_cross_entropy_with_logits(logit,y))
        torch.stack(losses).mean().div(len(worlds)).backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError(f"nonfinite optimization gradient: {name}")
    opt.step()

def choose_best(history):
    valid=[x for x in history if x.get("validation")]
    return min(valid,key=lambda x:(x["validation"]["equal_world_bce"],x["epoch"]))

def flatten_pred_rows(bench,model,worlds,name,arm,seed,setting,stage,split):
    result=[]; model.eval()
    with torch.no_grad():
        for w in worlds:
            for v in w["views"]:
                logit=float(model(*args_for(bench,v,name)).item()); prob=1/(1+math.exp(-max(-80,min(80,logit))))
                result.append([arm,seed,setting,stage,split,w["world_id"],v["candidate_id"],v["instance_id"],v["label"],logit,prob])
    return result

def atomic_csv(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    os.replace(tmp,path)

def channel_qualification(bench,world,models):
    v=world["views"][0]; out={}
    for name,m in models.items():
        base=args_for(bench,v,name); base_score=float(m(*base).detach().item()); row={}
        for channel in ("time","amount","selection"):
            x=[t.clone() if torch.is_tensor(t) else t for t in base]
            if name.startswith("simplicial"):
                k=int(v["r"]["selected_idx"][0]); loc=len(v["r"]["central"])+len(v["r"]["src"])+k
                col={"time":5,"amount":6,"selection":7}[channel]
                if channel=="selection": x[0][loc,col]=1-x[0][loc,col]
                else: x[0][loc,col]+=0.17
            else:
                col={"time":0,"amount":1,"selection":2}[channel]; k=int(v["r"]["selected_idx"][0])
                if channel=="selection": x[1][k,col]=1-x[1][k,col]
                else:x[1][k,col]+=0.17
            after=float(m(*x).detach().item())
            row[channel]={"score_delta":after-base_score,"score_changed":abs(after-base_score)>1e-10,"path_mutation_executed":True}
        out[name]=row
    return out

def neural_qualification(bench,config,worlds,out):
    torch.set_num_threads(1); widths=config["hidden_widths"]
    models={}
    seed=11; torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    for n,m in make_models(bench,widths).items():models[n]=m
    grads={}
    for name,m in models.items():
        m.zero_grad(set_to_none=True)
        for v in worlds[0]["views"]:
            (m(*args_for(bench,v,name))**2).backward()
        grads[name]=bench.count_grads(m);m.zero_grad(set_to_none=True)
    channels=channel_qualification(bench,worlds[0],models)
    basis={}
    for name,m in models.items():
        torch.manual_seed(11);random.seed(11)
        basis[name]=bench.permute_check(m,args_for(bench,worlds[0]["views"][0],name),kind_for(name))
    # Strict parser regression cases are evaluated without disk side effects.
    strict=[]
    for raw in ['{"a":1,"a":2}','{"x":NaN}','{"x":1e999}']:
        try: strict_json(raw); strict.append(False)
        except ValueError: strict.append(True)
    invalid_labels=[]
    for x in [True,1.0,0.5,"1",2,None]:
        try: label_value({"label":x});invalid_labels.append(False)
        except ValueError: invalid_labels.append(True)
    fixture=fixture_qualification(bench)
    if not all(strict) or not all(invalid_labels) or not all(x["pass"] for x in grads.values()) or not all(x["pass_atol_1e-6_rtol_1e-5"] for x in basis.values()):
        raise ValueError("pre-fit parsing, gradient or basis qualification failed")
    if not all(x["path_mutation_executed"] and x["score_changed"] for channels_by_arm in channels.values() for x in channels_by_arm.values()):
        raise ValueError("pre-fit isolated channel reachability failed")
    return {"strict_json_duplicate_and_nonfinite_rejected":all(strict),"boolean_fractional_string_out_of_range_and_null_labels_rejected":all(invalid_labels),"train_validation_world_ids_disjoint":not(set(TRAIN_IDS)&set(VAL_IDS)),"actual_operator_fixtures":fixture,"gradient_reachability":grads,"pre_fit_feature_channel_paths":channels,"pre_fit_basis_coordinate_checks":basis,"reachability_interpretation":"Finite gradient/score changes show pathway access only; efficacy is assessed from held-out development predictions separately."}

def run_neural_trial(bench,arm,seed,lr,train,val,root,config):
    key=f"{arm}/seed_{seed}/lr_{lr:g}"; d=root/"neural"/key;d.mkdir(parents=True,exist_ok=True)
    done=d/"complete.json"; latest=d/"latest.pt"
    if done.is_file(): return read_json(done)
    torch.set_num_threads(1)
    if latest.is_file():
        state=torch.load(latest,map_location="cpu",weights_only=False)
        model=make_models(bench,config["hidden_widths"])[arm];model.load_state_dict(state["model"])
        opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0);opt.load_state_dict(state["optimizer"])
        history=state["history"];start=int(state["epoch"])+1;best_val=state["best_val"];best_state=state["best_state"];restore_rng(state["rng"])
    else:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        model=make_models(bench,config["hidden_widths"])[arm]
        opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
        history=[];start=1;best_val=float("inf");best_state=None
        val0=eval_worlds(bench,model,val,arm); tr0=eval_worlds(bench,model,train,arm)
        history.append({"epoch":0,"train":{k:v for k,v in tr0.items() if k!="logits"},"validation":{k:v for k,v in val0.items() if k!="logits"},"elapsed_seconds":0.0})
        best_val=val0["equal_world_bce"];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    t0=time.monotonic();training_block_seconds=0.0
    for epoch in range(start,401):
        guard(root);epoch_started=time.monotonic();train_epoch(bench,model,opt,train,arm)
        training_block_seconds+=time.monotonic()-epoch_started
        if epoch%10==0 or epoch==400:
            t=time.monotonic();tr=eval_worlds(bench,model,train,arm);va=eval_worlds(bench,model,val,arm)
            hist={"epoch":epoch,"train":{k:v for k,v in tr.items() if k!="logits"},"validation":{k:v for k,v in va.items() if k!="logits"},"evaluation_seconds":time.monotonic()-t,"training_seconds_since_last_evaluation":training_block_seconds}
            history.append(hist);training_block_seconds=0.0
            if va["equal_world_bce"]<best_val:
                best_val=va["equal_world_bce"];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_trial_trace(d/"trace.jsonl",history)
            state={"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"rng":rng_state(),"history":history,"best_val":best_val,"best_state":best_state}
            atomic_torch(latest,state)
            if epoch%50==0: print(f"NEURAL {key} epoch={epoch} val_bce={va['equal_world_bce']:.6f}",flush=True)
            guard(root)
    model.load_state_dict(best_state)
    summary={"arm":arm,"seed":seed,"learning_rate":lr,"best_epoch":choose_best(history)["epoch"],"best_equal_world_validation_bce":best_val,"history_points":len(history),"duration_seconds":time.monotonic()-t0,"checkpoint":str(d/"selected.pt"),"state_at_epoch400":str(latest)}
    atomic_torch(d/"selected.pt",{"epoch":summary["best_epoch"],"model":best_state,"arm":arm,"seed":seed,"learning_rate":lr})
    atomic_torch(d/"epoch400_state.pt",torch.load(latest,map_location="cpu",weights_only=False))
    write_json(done,summary)
    return summary

def extend_neural(bench,arm,seed,lr,train,val,root,config):
    d=root/"neural"/f"{arm}/seed_{seed}/lr_{lr:g}"; ed=d/"extension";ed.mkdir(parents=True,exist_ok=True);done=ed/"complete.json"
    if done.is_file():return read_json(done)
    latest=ed/"latest.pt"
    if latest.is_file():
        s=torch.load(latest,map_location="cpu",weights_only=False);model=make_models(bench,config["hidden_widths"])[arm];model.load_state_dict(s["model"]);opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.);opt.load_state_dict(s["optimizer"]);hist=s["history"];start=s["epoch"]+1;best=s["best_val"];beststate=s["best_state"];restore_rng(s["rng"])
    else:
        s=torch.load(d/"epoch400_state.pt",map_location="cpu",weights_only=False);model=make_models(bench,config["hidden_widths"])[arm];model.load_state_dict(s["model"]);opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.);opt.load_state_dict(s["optimizer"]);hist=list(s["history"]);start=401;best=s["best_val"];beststate=s["best_state"];restore_rng(s["rng"])
    t0=time.monotonic();training_block_seconds=0.0
    for epoch in range(start,801):
        guard(root);epoch_started=time.monotonic();train_epoch(bench,model,opt,train,arm)
        training_block_seconds+=time.monotonic()-epoch_started
        if epoch%10==0 or epoch==800:
            t=time.monotonic();tr=eval_worlds(bench,model,train,arm);va=eval_worlds(bench,model,val,arm)
            hist.append({"epoch":epoch,"train":{k:v for k,v in tr.items() if k!="logits"},"validation":{k:v for k,v in va.items() if k!="logits"},"evaluation_seconds":time.monotonic()-t,"training_seconds_since_last_evaluation":training_block_seconds})
            training_block_seconds=0.0
            if va["equal_world_bce"]<best:best=va["equal_world_bce"];beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_trial_trace(ed/"trace.jsonl",hist);atomic_torch(latest,{"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"rng":rng_state(),"history":hist,"best_val":best,"best_state":beststate});guard(root)
    model.load_state_dict(beststate); best_epoch=min((x for x in hist if x.get("validation")),key=lambda x:(x["validation"]["equal_world_bce"],x["epoch"]))["epoch"]
    atomic_torch(ed/"selected_extended.pt",{"epoch":best_epoch,"model":beststate,"arm":arm,"seed":seed,"learning_rate":lr})
    summary={"arm":arm,"seed":seed,"learning_rate":lr,"best_epoch_from_0_to_800":best_epoch,"best_equal_world_validation_bce":best,"duration_seconds":time.monotonic()-t0,"checkpoint":str(ed/"selected_extended.pt")}
    write_json(done,summary);return summary

FEATURE_COLUMNS=["cycle_length","selected_time_min","selected_time_mean","selected_time_max","selected_time_population_sd","selected_log_amount_min","selected_log_amount_mean","selected_log_amount_max","selected_log_amount_population_sd","selected_time_span","selected_log_amount_span","central_in_degree_min","central_in_degree_mean","central_in_degree_max","central_out_degree_min","central_out_degree_mean","central_out_degree_max","selected_directed_pair_multiplicity_min","selected_directed_pair_multiplicity_mean","selected_directed_pair_multiplicity_max"]
def gbdt_matrix(bench,worlds):
    X=[];y=[];weights=[]
    for w in worlds:
        for v in w["views"]:X.append(bench.gbdt_row(v["r"]));y.append(v["label"]);weights.append(1/(len(worlds)*len(w["views"])))
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int);weights=np.asarray(weights,dtype=float)
    if X.shape!=(len(worlds)*78,20) or not np.isfinite(X).all():raise ValueError("GBDT fixed20 matrix invalid")
    return X,y,weights

def gbdt_eval(bench,worlds,raw_scores):
    ix=0;out={}
    for w in worlds:
        y=w["labels"];logits=np.asarray(raw_scores[ix:ix+len(y)],float);ix+=len(y)
        out[w["world_id"]]={**metrics(bench,y,logits),"logits":logits.tolist()}
    if ix!=len(raw_scores): raise ValueError("GBDT score rows do not cover worlds exactly")
    for name,key in (("bce","bce"),("ap","average_precision_grouped_threshold"),("auc","auroc_tie_aware")):
        out["equal_world_"+name]=float(np.mean([out[w["world_id"]][key] for w in worlds]))
    return out

def selected_gbdt_scores(model,X,iteration):
    if type(iteration) is not int or iteration<1: raise ValueError("invalid selected GBDT iteration")
    for step,scores in enumerate(model.staged_decision_function(X),1):
        if step==iteration:
            scores=np.asarray(scores,float)
            if scores.shape!=(len(X),) or not np.isfinite(scores).all(): raise ValueError("invalid GBDT selected scores")
            return scores
    raise ValueError("selected GBDT iteration exceeds fitted rounds")

def run_gbdt_trial(bench,seed,leaves,train,val,root):
    key=f"seed_{seed}/leaves_{leaves}";d=root/"gbdt"/key;d.mkdir(parents=True,exist_ok=True);done=d/"complete.json"
    if done.is_file():return read_json(done)
    X,y,sw=gbdt_matrix(bench,train);XV,yv,swv=gbdt_matrix(bench,val)
    t0=time.monotonic();model=HistGradientBoostingClassifier(learning_rate=.1,max_leaf_nodes=leaves,min_samples_leaf=2,l2_regularization=0.,early_stopping=False,validation_fraction=None,max_iter=400,random_state=seed,warm_start=False)
    model.fit(X,y,sample_weight=sw);fit_seconds=time.monotonic()-t0
    stages=model.staged_decision_function(X);stagesv=model.staged_decision_function(XV);history=[]
    for ep,(pt,pv) in enumerate(zip(stages,stagesv),1):
        if ep%10==0 or ep==400:
            tr=gbdt_eval(bench,train,pt);va=gbdt_eval(bench,val,pv);history.append({"epoch":ep,"train":{k:v for k,v in tr.items() if k!="logits"},"validation":{k:v for k,v in va.items() if k!="logits"}})
    best=choose_best(history);best_ep=best["epoch"]
    # Persist exact fitted 400-round model and complete trace; no hidden stopping.
    import pickle
    with (d/"model400.pkl.tmp").open("wb") as f:pickle.dump(model,f,protocol=4)
    os.replace(d/"model400.pkl.tmp",d/"model400.pkl")
    save_trial_trace(d/"trace.jsonl",history)
    summary={"seed":seed,"max_leaf_nodes":leaves,"learning_rate":.1,"min_samples_leaf":2,"l2_regularization":0.,"early_stopping":False,"best_iteration":best_ep,"best_equal_world_validation_bce":best["validation"]["equal_world_bce"],"duration_seconds":time.monotonic()-t0,"fit_seconds":fit_seconds,"deterministic_seed_retained":seed}
    write_json(done,summary);return summary

def extend_gbdt(bench,seed,leaves,train,val,root):
    d=root/"gbdt"/f"seed_{seed}/leaves_{leaves}";ed=d/"extension";ed.mkdir(parents=True,exist_ok=True);done=ed/"complete.json"
    if done.is_file():return read_json(done)
    import pickle
    with (d/"model400.pkl").open("rb") as f:model=pickle.load(f)
    X,y,sw=gbdt_matrix(bench,train);XV,yv,swv=gbdt_matrix(bench,val)
    t0=time.monotonic();model.set_params(max_iter=800,warm_start=True);model.fit(X,y,sample_weight=sw);fit_seconds=time.monotonic()-t0
    history=[]
    for ep,(pt,pv) in enumerate(zip(model.staged_decision_function(X),model.staged_decision_function(XV)),1):
        if ep%10==0 or ep==800:
            tr=gbdt_eval(bench,train,pt);va=gbdt_eval(bench,val,pv);history.append({"epoch":ep,"train":{k:v for k,v in tr.items() if k!="logits"},"validation":{k:v for k,v in va.items() if k!="logits"}})
    best=choose_best(history);best_ep=best["epoch"]
    with (ed/"model800.pkl.tmp").open("wb") as f:pickle.dump(model,f,protocol=4)
    os.replace(ed/"model800.pkl.tmp",ed/"model800.pkl");save_trial_trace(ed/"trace.jsonl",history)
    s={"seed":seed,"max_leaf_nodes":leaves,"max_iter":800,"best_evaluated_iteration_up_to_800":best_ep,"best_equal_world_validation_bce":best["validation"]["equal_world_bce"],"duration_seconds":time.monotonic()-t0,"fit_seconds":fit_seconds};write_json(done,s);return s

def score_qualification(bench,model,worlds,arm):
    result={};model.eval()
    with torch.no_grad():
        for w in worlds:
            checks=[bench.permute_check(model,args_for(bench,v,arm),kind_for(arm)) for v in w["views"]]
            result[w["world_id"]]={"candidates_checked":len(checks),"max_abs_logit_difference":max(x["abs_difference"] for x in checks),"pass":all(x["pass_atol_1e-6_rtol_1e-5"] for x in checks)}
    return result

def prediction_join_check(rows,worlds):
    expected={(w["world_id"],v["candidate_id"]):v["label"] for w in worlds for v in w["views"]}
    keys=set();groups={}
    for r in rows:
        arm,seed,setting,stage,split,wid,cid,instance,label,logit,probability=r
        key=(arm,seed,stage,split,wid,cid)
        if key in keys: raise ValueError("duplicate prediction custody key")
        keys.add(key)
        if (wid,cid) not in expected or label!=expected[wid,cid]: raise ValueError("prediction-to-source label join mismatch")
        if not math.isfinite(logit) or not math.isfinite(probability) or not 0<=probability<=1: raise ValueError("invalid prediction")
        required_split="training" if wid in TRAIN_IDS else "validation"
        if split!=required_split: raise ValueError("prediction split mismatch")
        groups.setdefault((arm,seed,stage),set()).add((wid,cid))
    if len(groups)!=5*5*2 or any(x!=set(expected) for x in groups.values()): raise ValueError("selected predictions do not cover the full registered population")
    return {"row_count":len(rows),"unique_prediction_keys":len(keys),"groups_checked":len(groups),"pass":True}

def main():
    global CAMPAIGN_START_EPOCH
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--world-index",type=Path,required=True);ap.add_argument("--pilot-config",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);args=ap.parse_args()
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    start_file=root/"execution_start.json"
    if start_file.exists(): CAMPAIGN_START_EPOCH=read_json(start_file)["started_utc_epoch"]
    else:
        CAMPAIGN_START_EPOCH=time.time();write_json(start_file,{"started_utc_epoch":CAMPAIGN_START_EPOCH,"scope":"four-hour S1-D wall window; unchanged across resumes"})
    guard(root)
    if (root/"final_report.json").exists():
        print(json.dumps(read_json(root/"final_report.json"),indent=2));return 0
    conf=read_json(args.pilot_config);idx=read_json(args.world_index)
    if tuple(conf["initialization_seeds"])!=SEEDS or tuple(conf["neural_optimizer"]["learning_rates"])!=RATES or tuple(conf["gbdt"]["max_leaf_nodes_trials"])!=LEAVES or conf["initial_epoch_ceiling"]!=400 or conf["selected_setting_extended_epoch_ceiling"]!=800 or conf["evaluate_every_epochs"]!=10:
        raise ValueError("pilot tuning grid or epoch/checkpoint rules changed")
    if conf.get("experiment_id")!="S1-D" or conf.get("status")!="authorized_exploratory_development_learning_only":raise ValueError("unexpected pilot contract")
    if tuple(conf["training_worlds"])!=TRAIN_IDS or tuple(conf["validation_worlds"])!=VAL_IDS:raise ValueError("pilot world allowlist mismatch")
    if conf.get("candidate_sampling")!="none; full guarded census in every existing world" or conf.get("maximum_new_output_bytes")!=MAX_BYTES or conf.get("execution_wall_time_ceiling_seconds")!=MAX_SECONDS:raise ValueError("scope ceilings or census rule changed")
    entries={}
    for w in idx["worlds"]:
        if w.get("replay") or w["world_id"]=="S1_dev_1729":continue
        if w["world_id"] in TRAIN_IDS+VAL_IDS:
            if w.get("status")!="COMPLETED":raise ValueError(f"registered world incomplete: {w['world_id']}")
            expected_role="provisional_development_train" if w["world_id"] in TRAIN_IDS else "provisional_development_validation"
            if w.get("role")!=expected_role:raise ValueError(f"registered role mismatch for {w['world_id']}: {w.get('role')!r}")
            entries[w["world_id"]]=Path(w["attempt"])
    if set(entries)!=set(TRAIN_IDS+VAL_IDS):raise ValueError("registered world index does not exactly cover S1-D allowlist")
    bench=load_bench();start_wall=time.time();read_start=time.monotonic()
    train=[strict_inputs(bench,x,entries[x],"training") for x in TRAIN_IDS]
    val=[strict_inputs(bench,x,entries[x],"validation") for x in VAL_IDS]
    input_time=time.monotonic()-read_start
    assert set(x["world_id"] for x in train).isdisjoint(x["world_id"] for x in val)
    model_cfg=read_json(Path(conf["model_config"]));write_json(root/"model_config.json",model_cfg)
    pre=neural_qualification(bench,model_cfg,train,root);write_json(root/"pre_fit_qualification.json",pre)
    if not (root/"run_manifest.json").exists():
        files=[args.world_index.resolve(),args.pilot_config.resolve(),Path("source/configs/s1_matched_smoke.json").resolve(),Path(__file__).resolve(),Path(bench.__file__).resolve()]
        for w in train+val:files.extend(w["data"]["inputs"])
        write_json(root/"run_manifest.json",{"experiment":"S1-D","command":[sys.executable,"-s","-B",str(Path(__file__).resolve()),"--world-index",str(args.world_index),"--pilot-config",str(args.pilot_config),"--output-dir",str(root)],"cwd":str(Path.cwd()),"started_utc_epoch":start_wall,"input_hashes":[sha(p) for p in files],"python":sys.version,"platform":platform.platform(),"versions":{"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"sklearn":sklearn.__version__},"train_worlds":TRAIN_IDS,"validation_worlds":VAL_IDS,"excluded_worlds":["S1_dev_1729","replay_S1_dev_101"],"input_preparation_seconds":input_time,"limits":{"wall_seconds":MAX_SECONDS,"output_bytes":MAX_BYTES}})
    neural=[];gbdt=[];stop=None
    try:
        for arm in ARMS:
            for seed in SEEDS:
                for lr in RATES:
                    guard(root);neural.append(run_neural_trial(bench,arm,seed,lr,train,val,root,model_cfg))
        for seed in SEEDS:
            for leaf in LEAVES:
                guard(root);gbdt.append(run_gbdt_trial(bench,seed,leaf,train,val,root))
    except BoundReached as e:stop=str(e)
    completed_neural=[x for x in neural]
    # When resumed, aggregate retained complete trial summaries from disk.
    neural=[]
    for arm in ARMS:
        for seed in SEEDS:
            for lr in RATES:
                p=root/"neural"/f"{arm}/seed_{seed}/lr_{lr:g}/complete.json"
                if p.exists():neural.append(read_json(p))
    gbdt=[]
    for seed in SEEDS:
        for leaf in LEAVES:
            p=root/"gbdt"/f"seed_{seed}/leaves_{leaf}/complete.json"
            if p.exists():gbdt.append(read_json(p))
    if stop or len(neural)!=len(ARMS)*len(SEEDS)*len(RATES) or len(gbdt)!=len(SEEDS)*len(LEAVES):
        write_json(root/"partial_report.json",{"status":"PARTIAL","stop_reason":stop or "time/output ceiling reached before full grid completion","completed_neural_trials":len(neural),"expected_neural_trials":len(ARMS)*len(SEEDS)*len(RATES),"completed_gbdt_trials":len(gbdt),"expected_gbdt_trials":len(SEEDS)*len(LEAVES),"elapsed_seconds":time.monotonic()-START,"output_bytes":output_bytes(root),"resume_command":[sys.executable,"-s","-B",str(Path(__file__).resolve()),"--world-index",str(args.world_index),"--pilot-config",str(args.pilot_config),"--output-dir",str(root)]})
        return 0
    select_lr={}
    for arm in ARMS:
        means={lr:float(np.mean([x["best_equal_world_validation_bce"] for x in neural if x["arm"]==arm and x["learning_rate"]==lr])) for lr in RATES}
        select_lr[arm]=min(RATES,key=lambda lr:(means[lr],lr))
    ext=[]
    try:
        for arm,lr in select_lr.items():
            for seed in SEEDS:guard(root);ext.append(extend_neural(bench,arm,seed,lr,train,val,root,model_cfg))
        leafmeans={leaf:float(np.mean([x["best_equal_world_validation_bce"] for x in gbdt if x["max_leaf_nodes"]==leaf])) for leaf in LEAVES}
        selected_leaf=min(LEAVES,key=lambda leaf:(leafmeans[leaf],leaf))
        for seed in SEEDS:guard(root);ext.append(extend_gbdt(bench,seed,selected_leaf,train,val,root))
    except BoundReached as e:stop=str(e)
    if stop or len([p for p in root.rglob("complete.json") if p.is_file()])<len(ARMS)*len(SEEDS)*(len(RATES)+1)+len(SEEDS)*(len(LEAVES)+1):
        write_json(root/"partial_report.json",{"status":"PARTIAL","stop_reason":stop or "extension grid incomplete","elapsed_seconds":time.monotonic()-START,"output_bytes":output_bytes(root)})
        return 0
    # Validate both selected-budget states against their retained trace predictions.
    post={};reload_checks={};late_flags={};pred_rows=[];gbdt_inference_seconds={}
    for arm,lr in select_lr.items():
        post[arm]={};late_flags[arm]={}
        for seed in SEEDS:
            d=root/"neural"/f"{arm}/seed_{seed}/lr_{lr:g}"
            hist400=read_trace(d/"trace.jsonl");hist800=read_trace(d/"extension/trace.jsonl")
            post[arm][str(seed)]={}
            for stage,checkpoint,history in (("400_selected",d/"selected.pt",hist400),("800_extended_selected",d/"extension/selected_extended.pt",hist800)):
                state=torch.load(checkpoint,map_location="cpu",weights_only=False)
                model=make_models(bench,model_cfg["hidden_widths"])[arm];model.load_state_dict(state["model"])
                tr=eval_worlds(bench,model,train,arm);va=eval_worlds(bench,model,val,arm)
                chosen=next(x for x in history if x["epoch"]==state["epoch"])
                delta=max(abs(va[w["world_id"]]["logits"][i]-chosen["validation"][w["world_id"]]["logits"][i]) for w in val for i in range(78))
                invariance=score_qualification(bench,model,train+val,arm)
                reload_checks[f"{arm}:{seed}:{stage}"]={"max_abs_logit_difference_from_trace":delta,"tolerance":1e-7,"pass":delta<=1e-7}
                post[arm][str(seed)][stage]={"selected_epoch":state["epoch"],"validation":va,"channels":channel_qualification(bench,val[0],{arm:model})[arm],"all_world_candidate_basis_checks":invariance}
                if delta>1e-7 or not all(x["pass"] for x in invariance.values()): raise ValueError(f"selected-checkpoint qualification failed: {arm} {seed} {stage}")
                for split,ws in (("training",train),("validation",val)):
                    pred_rows.extend(flatten_pred_rows(bench,model,ws,arm,arm,seed,lr,stage,split))
            end400=next(x for x in hist400 if x["epoch"]==400);end800=next(x for x in hist800 if x["epoch"]==800)
            selected400=choose_best(hist400);selected800=choose_best(hist800)
            late={}
            for budget,history in ((400,hist400),(800,hist800)):
                startpoint=next(x for x in history if x["epoch"]==budget-100);endpoint=next(x for x in history if x["epoch"]==budget)
                decrease=(startpoint["train"]["equal_world_bce"]-endpoint["train"]["equal_world_bce"])/max(abs(startpoint["train"]["equal_world_bce"]),1e-15)
                late[str(budget)]={"last100_train_bce_relative_decrease":decrease,"flag_gt_0_01":decrease>.01}
            ap_delta=abs(end800["validation"]["equal_world_ap"]-end400["validation"]["equal_world_ap"])
            late_flags[arm][str(seed)]={"budget_end_AP_change_400_to_800":ap_delta,"flag_gt_0_01_AP":ap_delta>.01,"selected_checkpoint_AP_change_400_to_800":selected800["validation"]["equal_world_ap"]-selected400["validation"]["equal_world_ap"],"late_loss_by_budget":late}
    # Prediction scores come from the validation-selected tree stage, not the terminal fit.
    import pickle
    for seed in SEEDS:
        for stage,directory,modelname,bestkey in (("400_selected",root/"gbdt"/f"seed_{seed}/leaves_{selected_leaf}","model400.pkl","best_iteration"),("800_extended_selected",root/"gbdt"/f"seed_{seed}/leaves_{selected_leaf}/extension","model800.pkl","best_evaluated_iteration_up_to_800")):
            info=read_json(directory/"complete.json");iteration=info[bestkey]
            with (directory/modelname).open("rb") as f:model=pickle.load(f)
            expected=next(x for x in read_trace(directory/"trace.jsonl") if x["epoch"]==iteration)
            for split,ws in (("training",train),("validation",val)):
                X,y,sw=gbdt_matrix(bench,ws);prediction_started=time.monotonic();logits=selected_gbdt_scores(model,X,iteration)
                gbdt_inference_seconds[f"{seed}:{stage}:{split}"]={"seconds":time.monotonic()-prediction_started,"worlds":len(ws),"candidate_rows":len(X),"selected_iteration":iteration}
                probability=1/(1+np.exp(-np.clip(logits,-80,80)));offset=0
                expected_scores=[z for w in ws for z in expected["train" if split=="training" else "validation"][w["world_id"]]["logits"]]
                delta=float(np.max(np.abs(logits-np.asarray(expected_scores))))
                reload_checks[f"gbdt:{seed}:{stage}:{split}"]={"selected_iteration":iteration,"max_abs_logit_difference_from_trace":delta,"tolerance":1e-7,"pass":delta<=1e-7}
                if delta>1e-7: raise ValueError("GBDT selected stage differs from retained trace")
                for w in ws:
                    for i,v in enumerate(w["views"]): pred_rows.append(["gbdt_fixed20",seed,selected_leaf,stage,split,w["world_id"],v["candidate_id"],v["instance_id"],v["label"],float(logits[offset+i]),float(probability[offset+i])])
                    offset+=len(w["views"])
    predcheck=prediction_join_check(pred_rows,train+val);predcheck["selected_checkpoint_reload_checks"]=reload_checks
    atomic_csv(root/"joined_predictions.csv",["arm","seed","selected_setting","checkpoint_stage","split","world_id","candidate_id","instance_id","label","logit","probability"],pred_rows)
    write_json(root/"post_fit_qualification.json",{"selected_model_checks":post,"reload_checks":predcheck,"stability_flags":late_flags})
    # Full per-trial retained summaries and setting selection.
    lrmeans={arm:{str(lr):float(np.mean([x["best_equal_world_validation_bce"] for x in neural if x["arm"]==arm and x["learning_rate"]==lr])) for lr in RATES} for arm in ARMS}
    leafmeans={str(leaf):float(np.mean([x["best_equal_world_validation_bce"] for x in gbdt if x["max_leaf_nodes"]==leaf])) for leaf in LEAVES}
    traces=[]
    for arm in ARMS:
        for seed in SEEDS:
            for lr in RATES:
                traces.append({"arm":arm,"seed":seed,"learning_rate":lr,"trace":read_trace(root/"neural"/f"{arm}/seed_{seed}/lr_{lr:g}/trace.jsonl")})
    write_json(root/"trial_summaries.json",{"neural_trials":neural,"gbdt_trials":gbdt,"selected_learning_rates":select_lr,"mean_selected_validation_bce_by_rate":lrmeans,"selected_gbdt_leaf_nodes":selected_leaf,"mean_selected_validation_bce_by_leaf":leafmeans,"extended_neural_trials":[read_json(p) for p in root.glob("neural/*/seed_*/lr_*/extension/complete.json")],"extended_gbdt_trials":[read_json(p) for p in root.glob("gbdt/seed_*/leaves_*/extension/complete.json")]})
    # Figures: means of per-world validation BCE by arm and registered setting, and actual fit wall time.
    fig,ax=plt.subplots(figsize=(9,5))
    for arm in ARMS:
        for lr in RATES:
            curves=[]
            for seed in SEEDS:
                tr=read_trace(root/"neural"/f"{arm}/seed_{seed}/lr_{lr:g}/trace.jsonl");curves.append([(x["epoch"],x["validation"]["equal_world_bce"]) for x in tr])
            if curves:
                epochs=[x[0] for x in curves[0]]; vals=np.mean([[y for _,y in c] for c in curves],axis=0)
                ax.plot(epochs,vals,label=f"{arm.replace('_',' ')} lr={lr:g}")
    ax.set_xlabel("Epoch");ax.set_ylabel("Equal-world validation BCE");ax.set_title("S1-D development learning curves");ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(root/"learning_curves.png",dpi=150);plt.close(fig)
    # Selected score figure is computed from joined raw predictions, per world.
    groups={}
    for prediction in pred_rows:
        if prediction[4]=="validation": groups.setdefault((prediction[0],prediction[1],prediction[3],prediction[5]),[]).append(prediction)
    selected_metrics=[]
    for (arm,seed,stage,wid),records in sorted(groups.items()):
        labels=[x[8] for x in records];logits=[x[9] for x in records]
        selected_metrics.append({"arm":arm,"seed":seed,"stage":stage,"world_id":wid,"candidate_count":len(records),"positives":sum(labels),**metrics(bench,labels,logits)})
    write_json(root/"selected_validation_metrics.json",selected_metrics)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    plot_arms=list(ARMS)+["gbdt_fixed20"]
    for ax,stage in zip(axes,("400_selected","800_extended_selected")):
        for wid,offset in zip(VAL_IDS,(-.1,.1)):
            for i,arm in enumerate(plot_arms):
                values=[x["average_precision_grouped_threshold"] for x in selected_metrics if x["arm"]==arm and x["stage"]==stage and x["world_id"]==wid]
                ax.scatter([i+offset]*len(values),values,s=12,label=wid if i==0 else None)
        ax.set_xticks(range(len(plot_arms)),["Cellular","Simplicial","Directed","Hasse","GBDT"],rotation=25)
        ax.set_title(stage.replace("_"," "));ax.set_ylim(0,1.03);ax.legend(fontsize=7)
    axes[0].set_ylabel("Per-world development validation AP")
    fig.suptitle("Initialization repeats; no population confidence intervals")
    fig.tight_layout();fig.savefig(root/"validation_scores.png",dpi=150);plt.close(fig)
    # Trial elapsed cost and measured optimization time have separate scopes.
    cost={"neural_grid_seconds":float(sum(x["duration_seconds"] for x in neural)),"gbdt_grid_seconds":float(sum(x["duration_seconds"] for x in gbdt)),"neural_extension_seconds":float(sum(x["duration_seconds"] for x in read_all_complete(root,"neural"))),"gbdt_extension_seconds":float(sum(x["duration_seconds"] for x in read_all_complete(root,"gbdt"))),"preprocessing_seconds":input_time,"total_elapsed_seconds":time.monotonic()-START,"output_bytes":output_bytes(root)}
    cost["full_pilot_measured_seconds"]=cost["neural_grid_seconds"]+cost["gbdt_grid_seconds"]+cost["neural_extension_seconds"]+cost["gbdt_extension_seconds"]+cost["preprocessing_seconds"]
    cost["proposed_future_main_campaign_projection"]={"route":"cellular_cwn_style_local vs fixed20 GBDT, five seeds, selected settings, 800 training epochs/iterations","measured_neural_epoch_seconds_by_seed":{},"measured_gbdt_iteration_seconds_by_seed":{},"scope":"conditional engineering projection on the current four training worlds/two validation worlds; not population or holdout access cost"}
    for arm,lr in select_lr.items():
        ts=[read_trace(root/"neural"/f"{arm}/seed_{s}/lr_{lr:g}/trace.jsonl") for s in SEEDS]
        cost["proposed_future_main_campaign_projection"]["measured_neural_epoch_seconds_by_seed"][arm]={str(s):sum(x["training_seconds_since_last_evaluation"] for x in t if x["epoch"]>0)/400 for s,t in zip(SEEDS,ts)}
    cost["proposed_future_main_campaign_projection"]["measured_gbdt_iteration_seconds_by_seed"]={str(s):read_json(root/"gbdt"/f"seed_{s}/leaves_{selected_leaf}/complete.json")["fit_seconds"]/400 for s in SEEDS}
    projected_neural=sum(sum(cost["proposed_future_main_campaign_projection"]["measured_neural_epoch_seconds_by_seed"][arm].values())*800 for arm in ["cellular_cwn_style_local"])
    projected_gbdt=sum(cost["proposed_future_main_campaign_projection"]["measured_gbdt_iteration_seconds_by_seed"].values())*800
    cost["proposed_future_main_campaign_projection"]["current_corpus_fit_seconds_for_two_primary_arms"] = projected_neural+projected_gbdt
    cost["neural_selected_validation_inference_seconds"]={arm:{seed:{stage:values["validation"]["inference_seconds_by_world"] for stage,values in stages.items()} for seed,stages in byseed.items()} for arm,byseed in post.items()}
    cost["gbdt_selected_inference_seconds"]=gbdt_inference_seconds
    primary_neural_per_world=sum(max(post["cellular_cwn_style_local"][str(seed)]["800_extended_selected"]["validation"]["inference_seconds_by_world"].values()) for seed in SEEDS)
    primary_tree_per_world=sum(gbdt_inference_seconds[f"{seed}:800_extended_selected:validation"]["seconds"]/len(val) for seed in SEEDS)
    cost["conditional_2952_world_primary_scoring_seconds"]=2952*(primary_neural_per_world+primary_tree_per_world)
    cost["conditional_scoring_projection_scope"]="Measured fitted five-seed cellular/GBDT selected-stage inference on existing 78-candidate worlds; assumes this yield and runtime generalize. Excludes future generation, preprocessing, failed worlds and custody overhead. Not campaign qualification."
    write_json(root/"cost_accounting.json",cost)
    # Stability/score summaries and conservative precision arithmetic.
    n_req=math.ceil(2*math.log(2/.05)/(.05**2))
    go=[{"question":"registered grid completed","status":"PASS" if len(neural)==60 and len(gbdt)==15 else "FAIL","evidence":f"{len(neural)}/60 neural; {len(gbdt)}/15 GBDT"},{"question":"optimization and learning","status":"REVIEW","evidence":"see per-trial traces and validation-only checkpoint selection"},{"question":"simple-baseline ceiling","status":"REVIEW","evidence":"see fixed20 GBDT and selected validation metrics; no claim from two worlds"},{"question":"input, identity, mathematical and feature-path invariants","status":"PASS" if predcheck["pass"] else "FAIL","evidence":"see pre/post qualification records"},{"question":"scope and generalization","status":"BLOCKED","evidence":"two fixed validation worlds; same controlled generator family; no population inference"},{"question":"convergence","status":"NOT_ESTABLISHED","evidence":"400-to-800 flags are investigations only"},{"question":"untouched test sampling and custody","status":"BLOCKED","evidence":f"prospective bounded-difference design requires {n_req} independent eligible worlds under stated assumptions; no seeds drawn or custody asserted"},{"question":"full campaign budget","status":"CONDITIONAL","evidence":"measured current-corpus CWN/GBDT projection in cost_accounting.json; future world counts/yields unknown"}]
    write_json(root/"go_no_go.json",{"status":"COMPLETED","table":go,"precision_bound":{"alpha":.05,"half_width":.05,"range":[-1,1],"formula":"ceil(2*ln(2/alpha)/half_width^2)","n":n_req,"derivation_independently_reproduced":n_req==2952,"design_only":True},"mean_unit":"world-level paired AP under a fixed five-seed scoring procedure; seeds and candidate rows are not independent worlds","invalid_worlds":"retain as invalid; no replacement, clipping, or invented AP; any invalid draw reduces eligible n and requires a revised design","protocol_proposal":"predefine and publish a finite eligible seed frame or use a documented uniform draw from a declared finite population; keep the draw and world IDs inaccessible to implementors until predictions are sealed; preserve invalid draws and report the realized denominator; use an independent custodian to generate/release seeds and audit access; do not claim financial-source independence from simulator seed sampling","future_main":"CWN-style cellular model versus fixed20 GBDT; other arms remain exploratory diagnostics","gate2":"UNRESOLVED"})
    # Cost plot.
    fig,ax=plt.subplots(figsize=(8,4));names=["Neural grid","GBDT grid","Neural extend","GBDT extend"];vals=[cost["neural_grid_seconds"],cost["gbdt_grid_seconds"],cost["neural_extension_seconds"],cost["gbdt_extension_seconds"]];ax.bar(names,vals,color="#416b8e");ax.set_ylabel("Measured seconds");ax.set_title("S1-D measured fitting cost");fig.tight_layout();fig.savefig(root/"measured_costs.png",dpi=150);plt.close(fig)
    files=[p for p in root.rglob("*") if p.is_file() and p.name not in ("output_hashes.json","final_report.json")]
    write_json(root/"output_hashes.json",[sha(p) for p in sorted(files)])
    report={"status":"COMPLETED","scope":"authorized exploratory S1-D only","train_worlds":TRAIN_IDS,"validation_worlds":VAL_IDS,"excluded_worlds":["S1_dev_1729","replay_S1_dev_101"],"candidate_count_per_world":78,"neural_trials":len(neural),"gbdt_trials":len(gbdt),"selected_learning_rates":select_lr,"selected_gbdt_leaves":selected_leaf,"extension_trials":len(ext),"post_fit_prediction_rows":len(pred_rows),"joined_predictions_sha256":sha(root/"joined_predictions.csv"),"go_no_go_table":go,"cost":cost,"known_limits":["two validation worlds do not support population confidence claims","all worlds share a fixed generator template","GUDHI remains unavailable; no external PH parity claim","no convergence or Gate 2 passage"],"elapsed_seconds":time.monotonic()-START,"output_bytes":output_bytes(root),"next_command":"No main benchmark command is authorized; Architect review is the next action."}
    write_json(root/"final_report.json",report);print(json.dumps(report,indent=2,allow_nan=False))
    return 0

def read_trace(path):
    if not path.exists():return []
    return [strict_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def read_all_complete(root,kind):
    return [read_json(p) for p in (root/kind).rglob("complete.json") if p.is_file() and p.parent.name=="extension"]

if __name__=="__main__":
    try: raise SystemExit(main())
    except BoundReached as e:
        print(f"BOUNDED_STOP: {e}",file=sys.stderr);raise SystemExit(0)
