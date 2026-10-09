#!/usr/bin/env python3
"""Untrained S1 benchmark-input and scientific qualification on an existing world."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.metadata as metadata, importlib.util, json, math, platform, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata
import torch
from torch import nn
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORBIDDEN = {"label","sar","alert_id","alertID","seed","generator","role","candidate_rank","event_ordinal","candidate_id","instance_id","world_id","account_id","event_ids"}
DEFAULT_HIDDEN=24
ARM_NAMES=("cellular_cwn_style_local","simplicial_mpsn_style_local","directed_local_edge_gnn","cellular_hasse_mechanism_control")

def sha(path):
    h=hashlib.sha256(); n=0
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b); n+=len(b)
    return {"path":str(path),"bytes":n,"sha256":h.hexdigest()}

def read_jsonl(path):
    with path.open(encoding="utf-8") as f: return [json.loads(s) for s in f if s.strip()]

def dump(path,obj): path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def finite(a,name):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError(f"nonfinite values in {name}")
    return a

def load_inputs(world):
    names=["candidate_observations.jsonl","candidate_identity.jsonl","cellular_polygonal_boundaries.jsonl","simplicial_subdivision.jsonl","hasse_incidence.jsonl","world_directed_multigraph_identity.jsonl","world_node_identity.json"]
    paths={n:world/n for n in names}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError("missing pilot inputs: "+", ".join(missing))
    obs,ids,cell,simp,hasse,graph=[read_jsonl(paths[n]) for n in names[:-1]]
    node_doc=json.loads(paths[names[-1]].read_text(encoding="utf-8"))
    if not (len(obs)==len(ids)==len(cell)==len(simp)==len(hasse)): raise ValueError("candidate sidecars lack complete aligned row coverage")
    if len({x["candidate_id"] for x in ids})!=len(ids): raise ValueError("candidate identity IDs are not unique")
    node_order=node_doc["node_order"]; event_order=[int(x) for x in node_doc["event_order"]]
    if len(set(node_order))!=len(node_order) or len(set(event_order))!=len(event_order): raise ValueError("coordinate maps contain duplicates")
    nix={x:i for i,x in enumerate(node_order)}; eix={x:i for i,x in enumerate(event_order)}
    emap={int(x["physical_event_id"]):x for x in graph}
    if set(emap)!=set(event_order): raise ValueError("world topology rows do not cover event coordinates")
    if any(x["source_account"] not in nix or x["target_account"] not in nix for x in graph): raise ValueError("event endpoint missing from node coordinates")
    src=np.asarray([nix[emap[e]["source_account"]] for e in event_order],dtype=np.int64)
    dst=np.asarray([nix[emap[e]["target_account"]] for e in event_order],dtype=np.int64)
    times=finite(obs[0]["world_event_time_steps"],"world times"); amounts=finite(obs[0]["world_event_log1p_amounts"],"world log amounts")
    if len(times)!=len(event_order) or len(amounts)!=len(event_order): raise ValueError("world vectors do not align with event coordinates")
    if np.max(times,initial=0)>1.0000001 or np.min(times,initial=0)<0: raise ValueError("time must already use the declared step/120 scale in [0,1]")
    rows=[]; mappings=[]
    for i,(o,ident,c,s,h) in enumerate(zip(obs,ids,cell,simp,hasse)):
        if FORBIDDEN.intersection(o): raise ValueError(f"forbidden observation fields: {sorted(FORBIDDEN.intersection(o))}")
        if not (ident["candidate_id"]==c["candidate_id"]==s["candidate_id"]==h["candidate_id"]): raise ValueError(f"sidecar mismatch at row {i}")
        selected_raw=finite(o["selected_event_mask"],"selected-event mask"); central_raw=finite(o["central_node_mask"],"central-node mask")
        if not np.isin(selected_raw,[0,1]).all() or not np.isin(central_raw,[0,1]).all(): raise ValueError(f"candidate {i} masks must be finite binary values")
        selected=selected_raw.astype(np.float32); central=central_raw.astype(np.float32)
        t=finite(o["world_event_time_steps"],"world times"); a=finite(o["world_event_log1p_amounts"],"world log amounts")
        if selected.shape!=(len(event_order),) or central.shape!=(len(node_order),) or len(t)!=len(event_order) or len(a)!=len(event_order): raise ValueError("feature dimensions misalign with coordinates")
        if not np.array_equal(t,times) or not np.array_equal(a,amounts): raise ValueError("candidate rows do not share whole-world observations")
        selected_ids={event_order[j] for j in np.flatnonzero(selected)}
        identity_ids=set(map(int,ident["event_ids"]))
        if selected_ids!=identity_ids: raise ValueError(f"candidate {i} selected mask and identity map differ")
        declared_length=float(o["cycle_length"])
        if not np.isfinite(declared_length) or declared_length<2 or declared_length!=int(declared_length): raise ValueError(f"candidate {i} cycle length is invalid")
        account_ids=list(ident["cycle_accounts"]); account_set=set(account_ids)
        central_ids={node_order[j] for j in np.flatnonzero(central)}
        if central_ids!=account_set: raise ValueError(f"candidate {i} central-node mask disagrees with its coordinate map")
        if len(selected_ids)!=int(declared_length) or len(account_set)!=int(declared_length): raise ValueError(f"candidate {i} masks and identity maps disagree with declared cycle length")
        if set(int(x["event_id"]) for x in c["boundary"])!=identity_ids: raise ValueError("cellular boundary does not map to chosen events")
        cycle_ids=[int(x["event_id"]) for x in c["boundary"]]
        if len(cycle_ids)!=int(declared_length) or len(set(cycle_ids))!=len(cycle_ids): raise ValueError(f"candidate {i} cellular boundary has an invalid length or repeated event")
        cycle_idx=np.asarray([eix[x] for x in cycle_ids],dtype=np.int64)
        for j,k in zip(cycle_idx,np.roll(cycle_idx,-1)):
            if dst[j]!=src[k]: raise ValueError(f"candidate {i} directed boundary is not a closed transfer cycle")
        rows.append({"selected":selected,"central":central,"cycle_length":declared_length,"times":times,"amounts":amounts,"selected_idx":np.flatnonzero(selected),"cycle_idx":cycle_idx,"src":src,"dst":dst})
        mappings.append({"row":i,"candidate_key":ident["candidate_id"],"event_coordinate_positions":sorted(eix[x] for x in identity_ids),"account_coordinate_positions":sorted(nix[x] for x in ident["cycle_accounts"]),"physical_events":len(event_order),"identities_are_predictor_values":False})
    return {"rows":rows,"ids":ids,"labels_path":world/"candidate_labels.jsonl","graph":graph,"node_order":node_order,"event_order":event_order,"src":src,"dst":dst,"map_records":mappings,"inputs":[*paths.values(),world/"candidate_labels.jsonl"]}

def cellular(row,n,e):
    b1=np.zeros((n,e),np.float32)
    for j,(u,v) in enumerate(zip(row["src"],row["dst"])):
        if u!=v: b1[u,j]=-1.; b1[v,j]=1.
    return b1,row["selected"][:,None].astype(np.float32)

def simplicial(row,n,e):
    # Four structural port segments per event; event observations occur once on the event vertex.
    vertices=[("account",i) for i in range(n)]+[("source_port",j) for j in range(e)]+[("event",j) for j in range(e)]+[("target_port",j) for j in range(e)]
    edges=[]; rel=[]
    for j,(u,v) in enumerate(zip(row["src"],row["dst"])):
        for a,b,k in [(int(u),n+j,0),(n+j,n+e+j,1),(n+e+j,n+2*e+j,2),(n+2*e+j,int(v),3)]: edges.append((a,b)); rel.append(k)
    perimeter=[]
    for j in row["cycle_idx"]:
        seq=[int(row["src"][j]),n+int(j),n+e+int(j),n+2*e+int(j),int(row["dst"][j])]
        perimeter.extend(seq if not perimeter else seq[1:])
    if len(perimeter)>1 and perimeter[-1]==perimeter[0]: perimeter.pop()
    center=len(vertices); vertices.append(("face_center",0))
    edge_ix={tuple(sorted(x)):i for i,x in enumerate(edges)}
    tris=[]
    for a,b in zip(perimeter,perimeter[1:]+perimeter[:1]):
        if a==b: continue
        for z in (a,b):
            key=tuple(sorted((center,z)))
            if key not in edge_ix: edge_ix[key]=len(edges); edges.append((center,z)); rel.append(4)
        tris.append((a,b,center))
    b1=np.zeros((len(vertices),len(edges)),np.float32)
    for j,(a,b) in enumerate(edges): b1[a,j]=-1.; b1[b,j]=1.
    b2=np.zeros((len(edges),len(tris)),np.float32)
    for k,(a,b,c) in enumerate(tris):
        for x,y,sgn in [(a,b,1),(b,c,1),(a,c,-1)]:
            q=edge_ix[tuple(sorted((x,y)))]
            b2[q,k]+=sgn*(1 if edges[q]==(x,y) else -1)
    vf=np.zeros((len(vertices),8),np.float32)
    for i,(typ,j) in enumerate(vertices):
        vf[i,1+{"account":0,"source_port":1,"event":2,"target_port":3,"face_center":0}[typ]]=1.
        if typ=="account": vf[i,0]=row["central"][j]
        if typ=="event": vf[i,5:8]=[row["times"][j],row["amounts"][j],row["selected"][j]]
    ef=np.zeros((len(edges),5),np.float32)
    for i,k in enumerate(rel):ef[i,k]=1.
    ff=np.full((len(tris),1),row["cycle_length"],np.float32)
    return vf,ef,ff,b1,b2,{"vertices":vertices,"edges":edges,"relations":rel,"triangles":tris,"physical_event_map":list(range(e)),"auxiliary_segments_are_transactions":False}

def closed(b1,b2):
    for matrix in (b1,b2):
        if not np.isfinite(matrix).all() or not np.array_equal(matrix,np.rint(matrix)):return False
    return not np.any(b1.astype(np.int64)@b2.astype(np.int64))

def boundary_fixture_checks():
    empty=closed(np.zeros((0,0),dtype=np.int64),np.zeros((0,0),dtype=np.int64))
    b1=np.asarray([[-1,0,1],[1,-1,0],[0,1,-1]],dtype=np.int64)
    good=np.ones((3,1),dtype=np.int64); bad=np.asarray([[1],[1],[0]],dtype=np.int64)
    malformed_rejected=not closed(b1,bad)
    # A self-loop's port expansion is a four-edge cycle with no transaction self-edge in the algebraic boundary.
    loop_path=[0,1,2,3,0]
    loop_edges=list(zip(loop_path,loop_path[1:]))
    loop_cycle_rank=len(loop_edges)-4+1
    # A reciprocal two-gon has two distinct event paths and preserves both physical event coordinates.
    two_gon_paths=[(0,1,4,5,1),(1,2,6,7,0)]
    two_gon_event_count=2
    return {"empty_rank_boundary_closes":bool(empty),"valid_cycle_fixture_closes":bool(closed(b1,good)),"malformed_boundary_fixture_rejected":bool(malformed_rejected),"loop_port_skeleton_cycle_rank":loop_cycle_rank,"reciprocal_two_gon_distinct_physical_event_count":two_gon_event_count,"fixture_scope":"small mathematical representation controls only; no world or model data generated"}

def gf2_graph_rank(edges):
    basis={}
    for a,b in edges:
        bits=(1<<int(a)) ^ (1<<int(b))
        while bits:
            pivot=bits.bit_length()-1
            if pivot in basis: bits ^= basis[pivot]
            else: basis[pivot]=bits; break
    return len(basis)

class Cell(nn.Module):
    def __init__(self,hidden=DEFAULT_HIDDEN,hasse=False):
        super().__init__(); self.hasse=hasse
        self.node=nn.Linear(1,hidden); self.edge=nn.Linear(3,hidden); self.src=nn.Linear(hidden,hidden,bias=False); self.dst=nn.Linear(hidden,hidden,bias=False)
        self.eu=nn.Linear(3*hidden,hidden); self.face=nn.Linear(1,hidden); self.fu=nn.Linear(2*hidden,hidden); self.out=nn.Linear(hidden,1)
        self.isrc=nn.Linear(hidden,hidden,bias=False) if hasse else None; self.idst=nn.Linear(hidden,hidden,bias=False) if hasse else None
    def forward(self,v,e,b1,b2,src,dst,cycle):
        hv=torch.tanh(self.node(v)); he0=torch.tanh(self.edge(e)); directional=self.src(hv[src])+self.dst(hv[dst])
        if self.hasse:
            low=self.isrc(hv[src])+self.idst(hv[dst])
        else: low=torch.abs(b1).T@hv
        f0=torch.tanh(self.face(cycle.reshape(1,1))); upper=torch.abs(b2)@f0
        he=torch.tanh(self.eu(torch.cat([he0,low+directional,upper],-1)))
        hf=torch.tanh(self.fu(torch.cat([f0,torch.abs(b2).T@he],-1)))
        return self.out(hf+he.mean(0,keepdim=True)).reshape(())

class Simp(nn.Module):
    def __init__(self,hidden=DEFAULT_HIDDEN):
        super().__init__(); self.v=nn.Linear(8,hidden); self.rel=nn.Linear(5,hidden); self.e=nn.Linear(2*hidden,hidden); self.vu=nn.Linear(2*hidden,hidden); self.f=nn.Linear(1,hidden); self.fu=nn.Linear(2*hidden,hidden); self.out=nn.Linear(hidden,1)
    def forward(self,v,e,f,b1,b2):
        hv=torch.tanh(self.v(v)); he=torch.tanh(self.e(torch.cat([torch.tanh(self.rel(e)),torch.abs(b1).T@hv],-1)))
        hv2=torch.tanh(self.vu(torch.cat([hv,torch.abs(b1)@he],-1))); hf=torch.tanh(self.fu(torch.cat([torch.tanh(self.f(f)),torch.abs(b2).T@he],-1)))
        return self.out(hf.mean(0)+hv2.mean(0)).reshape(())

class Directed(nn.Module):
    def __init__(self,hidden=DEFAULT_HIDDEN):
        super().__init__(); self.n=nn.Linear(1,hidden); self.om=nn.Linear(hidden+3,hidden); self.im=nn.Linear(hidden+3,hidden); self.up=nn.Linear(3*hidden,hidden); self.er=nn.Linear(hidden+3,hidden); self.read=nn.Linear(2*hidden+1,1)
    def forward(self,node,edge,src,dst,sel,cycle):
        h=torch.tanh(self.n(node)); outm=torch.tanh(self.om(torch.cat([h[dst],edge],-1))); inm=torch.tanh(self.im(torch.cat([h[src],edge],-1)))
        outgoing=torch.zeros_like(h); incoming=torch.zeros_like(h); outgoing.index_add_(0,src,outm); incoming.index_add_(0,dst,inm)
        h2=torch.tanh(self.up(torch.cat([h,incoming,outgoing],-1))); eh=torch.tanh(self.er(torch.cat([h2[src],edge],-1)))
        cm=node[:,0:1]; nr=(h2*cm).sum(0)/(cm.sum()+1e-6); er=(eh*sel[:,None]).sum(0)/(sel.sum()+1e-6)
        return self.read(torch.cat([nr,er,cycle.reshape(1)],-1)).reshape(())

def gbdt_row(r):
    ix=r["selected_idx"]; ts=r["times"][ix]; am=r["amounts"][ix]
    if not len(ix): raise ValueError("empty selected-event mask")
    indeg=np.bincount(r["dst"],minlength=len(r["central"])); outdeg=np.bincount(r["src"],minlength=len(r["central"]))
    ci=indeg[r["central"]>0]; co=outdeg[r["central"]>0]
    pair=Counter(zip(map(int,r["src"]),map(int,r["dst"]))); mult=[pair[(int(r["src"][j]),int(r["dst"][j]))] for j in ix]
    def stat(x): return [float(np.min(x)),float(np.mean(x)),float(np.max(x)),float(np.std(x,ddof=0))]
    return [r["cycle_length"],*stat(ts),*stat(am),float(np.ptp(ts)),float(np.ptp(am)),float(ci.min()),float(ci.mean()),float(ci.max()),float(co.min()),float(co.mean()),float(co.max()),float(min(mult)),float(np.mean(mult)),float(max(mult))]

def grouped_ap(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); p=int(y.sum())
    if not p:return None
    order=np.argsort(-s,kind="mergesort"); tp=fp=0; ap=prev=0.; i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]:j+=1
        z=y[order[i:j]]; tp+=int(z.sum()); fp+=len(z)-int(z.sum()); recall=tp/p; ap+=(recall-prev)*(tp/(tp+fp)); prev=recall; i=j
    return float(ap)

def tie_auc(y,s):
    y=np.asarray(y,int); p=int(y.sum()); n=len(y)-p
    return None if not p or not n else float((rankdata(s,method="average")[y==1].sum()-p*(p+1)/2)/(p*n))

def shortcut(rows,labels,ids):
    scores={"cycle_length":[],"minimum_selected_log_amount":[],"negative_log_amount_span":[],"negative_time_span":[]}
    for r in rows:
        a=r["amounts"][r["selected_idx"]]; t=r["times"][r["selected_idx"]]
        scores["cycle_length"].append(r["cycle_length"]); scores["minimum_selected_log_amount"].append(float(a.min()))
        scores["negative_log_amount_span"].append(float(-np.ptp(a))); scores["negative_time_span"].append(float(-np.ptp(t)))
    y=np.asarray(labels,int); cls={}
    for k in sorted({len(x["cycle_accounts"]) for x in ids}):
        ix=[i for i,x in enumerate(ids) if len(x["cycle_accounts"])==k]
        cls[str(k)]={"n":len(ix),"positives":int(y[ix].sum()),"negatives":int(len(ix)-y[ix].sum())}
    return {"interpretation":"fixed untrained construction diagnostics only","prevalence":float(y.mean()),"classes":{"positive":int(y.sum()),"negative":int(len(y)-y.sum()),"total":len(y)},"class_counts_by_cycle_length":cls,"rankings":{k:{"direction":"high score ranks positive","average_precision_grouped_threshold":grouped_ap(y,v),"auroc_tie_aware":tie_auc(y,v)} for k,v in scores.items()}}

def count_grads(model):
    missing=[k for k,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    zero=[k for k,p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum())==0]
    return {"active_trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"all_trainable_have_finite_gradient":not missing,"zero_gradient_parameters":zero,"missing_or_nonfinite":missing,"pass":not missing and not zero}

def build_args(view,name):
    r=view["r"]; cyc=torch.tensor([r["cycle_length"]],dtype=torch.float32)
    if name.startswith("cellular"): return [torch.tensor(r["central"][:,None]),torch.tensor(np.column_stack([r["times"],r["amounts"],r["selected"]]),dtype=torch.float32),torch.tensor(view["cell"][0]),torch.tensor(view["cell"][1]),torch.tensor(r["src"],dtype=torch.long),torch.tensor(r["dst"],dtype=torch.long),cyc]
    if name.startswith("simplicial"): return [torch.as_tensor(x,dtype=torch.float32) for x in view["simp"][:5]]
    return [torch.tensor(r["central"][:,None]),torch.tensor(np.column_stack([r["times"],r["amounts"],r["selected"]]),dtype=torch.float32),torch.tensor(r["src"],dtype=torch.long),torch.tensor(r["dst"],dtype=torch.long),torch.tensor(r["selected"],dtype=torch.float32),cyc]

def permute_check(model,args,name):
    with torch.no_grad(): base=model(*args).item()
    if name in ("cellular","hasse"):
        v,e,b1,b2,src,dst,cy=args; p=torch.randperm(len(v)); q=torch.randperm(len(e)); inv=torch.empty_like(p); inv[p]=torch.arange(len(p))
        d0=torch.where(torch.rand(len(v))>.5,1.,-1.); d1=torch.where(torch.rand(len(e))>.5,1.,-1.); d2=torch.tensor([-1. if random.random()<.5 else 1.])
        result=model(v[p],e[q],b1[p][:,q]*d0[:,None]*d1[None,:],b2[q]*d1[:,None]*d2[None,:],inv[src[q]],inv[dst[q]],cy)
    elif name=="simplicial":
        v,e,f,b1,b2=args; p=torch.randperm(len(v)); q=torch.randperm(len(e)); z=torch.randperm(len(f)); d0=torch.where(torch.rand(len(v))>.5,1.,-1.); d1=torch.where(torch.rand(len(e))>.5,1.,-1.); d2=torch.where(torch.rand(len(f))>.5,1.,-1.)
        result=model(v[p],e[q],f[z],b1[p][:,q]*d0[:,None]*d1[None,:],b2[q][:,z]*d1[:,None]*d2[None,:])
    else:
        node,edge,src,dst,sel,cy=args; p=torch.randperm(len(node)); q=torch.randperm(len(edge)); inv=torch.empty_like(p); inv[p]=torch.arange(len(p))
        result=model(node[p],edge[q],inv[src[q]],inv[dst[q]],sel[q],cy)
    delta=abs(base-result.item())
    return {"base":float(base),"permuted_basis_flipped":float(result.item()),"abs_difference":delta,"pass_atol_1e-6_rtol_1e-5":bool(torch.allclose(torch.tensor(base),result.detach(),atol=1e-6,rtol=1e-5))}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--world-dir",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--model-config",type=Path)
    a=ap.parse_args(); world=a.world_dir.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    model_config={"hidden_widths":{name:DEFAULT_HIDDEN for name in ARM_NAMES},"initialization_seed":11}
    model_config_path=None
    if a.model_config:
        model_config_path=a.model_config.resolve(); model_config=json.loads(model_config_path.read_text(encoding="utf-8"))
        if set(model_config.get("hidden_widths",{}))!=set(ARM_NAMES): raise ValueError("model config must define exactly the four registered neural arm widths")
        widths=model_config["hidden_widths"]
        if any(not isinstance(widths[k],int) or not 8<=widths[k]<=64 for k in ARM_NAMES): raise ValueError("hidden widths must be integers in the declared 8..64 calibration range")
        if not isinstance(model_config.get("initialization_seed"),int): raise ValueError("model config initialization_seed must be an integer")
    torch.set_num_threads(1); torch.manual_seed(model_config["initialization_seed"]); np.random.seed(model_config["initialization_seed"]); random.seed(model_config["initialization_seed"]); started=time.time()
    d=load_inputs(world); rows=d["rows"]; n=len(d["node_order"]); e=len(d["event_order"])
    views=[]
    for r in rows:
        b1,b2=cellular(r,n,e); sv=simplicial(r,n,e)
        if not closed(b1,b2): raise ValueError("cellular B1@B2 != 0")
        if not closed(sv[3],sv[4]): raise ValueError("simplicial boundary closure failed")
        views.append({"r":r,"cell":(b1,b2),"simp":sv})
    pairs=Counter(zip(d["src"].tolist(),d["dst"].tolist()))
    w=model_config["hidden_widths"]
    models={"cellular_cwn_style_local":Cell(w["cellular_cwn_style_local"]),"simplicial_mpsn_style_local":Simp(w["simplicial_mpsn_style_local"]),"directed_local_edge_gnn":Directed(w["directed_local_edge_gnn"]),"cellular_hasse_mechanism_control":Cell(w["cellular_hasse_mechanism_control"],hasse=True)}
    grad={}
    for name,model in models.items():
        model.eval()
        for v in views: (model(*build_args(v,name))**2).backward()
        grad[name]=count_grads(model); model.zero_grad(set_to_none=True)
    # Actual local feature access by time/amount/selection mutation on the first view.
    access={}
    for name in ("cellular_cwn_style_local","simplicial_mpsn_style_local","directed_local_edge_gnn","cellular_hasse_mechanism_control"):
        v=views[0]; x=build_args(v,name)
        with torch.no_grad():
            before=models[name](*x)
            if name.startswith("simplicial"):
                x[0]=x[0].clone(); event_i=n+e+int(v["r"]["selected_idx"][0]); x[0][event_i,5]+=0.2; x[0][event_i,6]+=0.3; x[0][event_i,7]=1-x[0][event_i,7]
            else:
                x[1]=x[1].clone(); j=int(v["r"]["selected_idx"][0]); x[1][j,0]+=0.2; x[1][j,1]+=0.3; x[1][j,2]=1-x[1][j,2]
            after=models[name](*x)
        access[name]={"changed":bool(abs(float(before-after))>1e-9),"difference":float(abs(float(before-after))),"check":"untrained local feature mutation smoke; no efficacy"}
    perms={}
    for name,kind in [(k,"simplicial" if k.startswith("simplicial") else "directed" if k.startswith("directed") else "hasse" if "hasse" in k else "cellular") for k in models]:
        random.seed(11); torch.manual_seed(11); perms[name]=permute_check(models[name],build_args(views[0],name),kind)
    # Only the isolated fixed-ranking evaluator reads annotation values.
    lab=read_jsonl(d["labels_path"]); byid={x["candidate_id"]:int(x["label"]) for x in lab}
    if len(byid)!=len(lab) or set(byid)!={x["candidate_id"] for x in d["ids"]}: raise ValueError("label sidecar does not exactly join candidate identities")
    labels=[byid[x["candidate_id"]] for x in d["ids"]]
    shortcuts=shortcut(rows,labels,d["ids"])
    colnames=["cycle_length","selected_time_min","selected_time_mean","selected_time_max","selected_time_population_sd","selected_log_amount_min","selected_log_amount_mean","selected_log_amount_max","selected_log_amount_population_sd","selected_time_span","selected_log_amount_span","central_in_degree_min","central_in_degree_mean","central_in_degree_max","central_out_degree_min","central_out_degree_mean","central_out_degree_max","selected_directed_pair_multiplicity_min","selected_directed_pair_multiplicity_mean","selected_directed_pair_multiplicity_max"]
    X=np.asarray([gbdt_row(r) for r in rows],float)
    if X.shape!=(len(rows),20) or not np.isfinite(X).all(): raise ValueError("fixed GBDT feature matrix is invalid")
    fx=np.array([[i/16,(i%4)/4,((i//4)%4)/4] for i in range(16)],dtype=np.float32); fy=np.asarray([(i//4)%2 for i in range(16)])
    clf=HistGradientBoostingClassifier(max_iter=3,max_leaf_nodes=4,random_state=11).fit(fx,fy); pred=clf.predict(fx)
    gbdt={"implementation":"sklearn HistGradientBoostingClassifier","fixture":"separate deterministic arithmetic toy, 16 rows; fit/predict API smoke only","fixture_predictions":len(pred),"pilot_feature_matrix_rows":len(rows),"pilot_feature_matrix_columns":colnames,"pilot_matrix_sha256":hashlib.sha256(X.tobytes()).hexdigest(),"lossy_map":"20 scalar summaries; topology and event identities collapse to listed summaries","pilot_labels_used_for_fit":False}
    # Essential H1 of the unfilled port-expanded graph. Batch equal-time segments.
    V=n+3*e; parent=list(range(V)); rank=[0]*V; comp=V
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    batches=defaultdict(list)
    for j,(u,v) in enumerate(zip(d["src"],d["dst"])):
        path=[int(u),n+j,n+e+j,n+2*e+j,int(v)]
        for x,y in zip(path,path[1:]): batches[float(rows[0]["times"][j])].append((x,y))
    births=[]; edges=0
    for t in sorted(batches):
        cyc=0
        for x,y in batches[t]:
            edges+=1; rx,ry=find(x),find(y)
            if rx==ry:cyc+=1
            else:
                if rank[rx]<rank[ry]:rx,ry=ry,rx
                parent[ry]=rx
                if rank[rx]==rank[ry]:rank[rx]+=1
                comp-=1
        if cyc:births.append({"filtration_time":t,"essential_H1_births":cyc,"death":"infinity"})
    b1=edges-V+comp; dsu=sum(x["essential_H1_births"] for x in births)
    algebraic_rank=gf2_graph_rank([(x,y) for t,items in batches.items() for x,y in items])
    alg=edges-algebraic_rank
    if b1!=dsu or b1!=alg: raise ValueError("PH DSU and algebraic graph references differ")
    gudhi_spec=importlib.util.find_spec("gudhi")
    ph={"object":"unfilled direction-forgetting port-expanded skeleton, retaining parallel events and loops","vertices":V,"structural_segments":edges,"components":comp,"essential_H1_count":b1,"births_by_equal_time_batch":births,"all_deaths":"infinity","dsu_reference_count":dsu,"algebraic_GF2_incidence_rank":algebraic_rank,"algebraic_GF2_graph_reference_count":alg,"common_object_parity":"PASS","gudhi_available":gudhi_spec is not None}
    if gudhi_spec is None:
        ph["external_backend_parity"]="UNRESOLVED; GUDHI is unavailable in the installed environment"
    else:
        try:
            import gudhi
            tree=gudhi.SimplexTree()
            for v in range(V): tree.insert([v],filtration=0.0)
            for t,items in batches.items():
                for x,y in items: tree.insert([int(x),int(y)],filtration=float(t))
            tree.make_filtration_non_decreasing()
            diagram=tree.persistence(homology_coeff_field=2,persistence_dim_max=True)
            essential_h1=[float(birth) for dim,(birth,death) in diagram if dim==1 and math.isinf(float(death))]
            ph["gudhi_essential_H1_count"]=len(essential_h1)
            ph["gudhi_essential_H1_births"]=sorted(essential_h1)
            ph["external_backend_parity"]="PASS" if len(essential_h1)==b1 else "FAIL"
            ph["external_backend_supported_object"]="unfilled graph skeleton only; no direction, amount flow, or 2-cell semantics"
        except Exception as exc:
            ph["external_backend_parity"]="UNRESOLVED; available backend failed to execute"
            ph["external_backend_error"]=f"{type(exc).__name__}: {exc}"
    # One measured census pass and two timing repeats; no optimizer or pilot fitting.
    costs={}; params={}
    for name,m in models.items():
        params[name]=sum(p.numel() for p in m.parameters() if p.requires_grad)
        def runpass():
            m.zero_grad(set_to_none=True)
            for v in views:(m(*build_args(v,name))**2).backward()
            m.zero_grad(set_to_none=True)
        layer_counts={}
        handles=[]
        for layer_name,layer in m.named_modules():
            if isinstance(layer,nn.Linear):
                def counter(mod,inputs,output,key=layer_name):
                    x=inputs[0]; rows_seen=int(x.numel()//mod.in_features)
                    record=layer_counts.setdefault(key,{"calls":0,"input_rows":0,"input_features":mod.in_features,"output_features":mod.out_features,"input_shapes":[],"multiply_accumulates":0})
                    record["calls"]+=1; record["input_rows"]+=rows_seen
                    shape=list(x.shape)
                    if shape not in record["input_shapes"]: record["input_shapes"].append(shape)
                    record["multiply_accumulates"]+=rows_seen*mod.in_features*mod.out_features
                handles.append(layer.register_forward_hook(counter))
        t=time.perf_counter();runpass(); measured=time.perf_counter()-t
        for handle in handles: handle.remove()
        repeats=[]
        for _ in range(2):
            t=time.perf_counter();runpass();repeats.append(time.perf_counter()-t)
        macs=sum(v["multiply_accumulates"] for v in layer_counts.values())
        costs[name]={"active_trainable_parameters":params[name],"forward_backward_seconds_measured_pass":measured,"forward_backward_seconds_timing_repeats":repeats,"dense_linear_forward":{"layer_calls":sum(v["calls"] for v in layer_counts.values()),"input_rows_processed":sum(v["input_rows"] for v in layer_counts.values()),"multiply_accumulates":macs,"estimated_flops":2*macs,"per_layer":layer_counts,"method":"forward hooks count actual nn.Linear call input shapes; 2 FLOPs per multiply-accumulate"},"flop_scope":"dense linear forward subtotal only; excludes incidence matrix multiplication, scatter/index aggregation, nonlinearities, loss and backward work"}
    matching={}
    names=list(params)
    for i,a0 in enumerate(names):
        for b0 in names[i+1:]:
            delta=abs(params[a0]-params[b0])/max(params[a0],params[b0])
            matching[a0+"__"+b0]={"relative_difference":delta,"within_2_percent":delta<=0.02}
    pair_groups=defaultdict(list)
    for j,key in enumerate(zip(d["src"].tolist(),d["dst"].tolist())): pair_groups[key].append(j)
    selected_parallel_checks=[]
    for key,ix in pair_groups.items():
        if len(ix)>1:
            selected_values=[int(r["selected"][j]) for r in rows for j in ix]
            selected_parallel_checks.append({"directed_pair":list(key),"event_coordinate_positions":ix,"all_event_coordinates_distinct":len(set(ix))==len(ix),"selection_patterns_by_candidate":len({tuple(int(r["selected"][j]) for j in ix) for r in rows}),"selected_occurrence_seen":1 in selected_values,"unselected_occurrence_seen":0 in selected_values,"both_selected_and_unselected_observed":(1 in selected_values and 0 in selected_values)})
    view_summary={"candidates":len(rows),"nodes":n,"physical_events":e,"parallel_directed_pairs":{f"{x}->{y}":c for (x,y),c in pairs.items() if c>1},"selected_unselected_parallel_event_checks":selected_parallel_checks,"self_loop_events":int(np.sum(d["src"]==d["dst"])),"reciprocal_two_gons":sum(1 for r in rows if r["cycle_length"]==2),"cellular_B1B2_all_candidates_zero":True,"simplicial_boundary_closure_all_candidates":True,"mask_validation":"finite binary values; lengths and coordinate maps exactly reconciled","time_validation":"pre-normalized step/120 values in [0,1], no silent rescaling","event_coordinate_coverage":"all physical event columns retained","forbidden_predictor_fields_absent":True,"mathematical_boundary_fixtures":boundary_fixture_checks()}
    model_detail={"arm_names":{"cellular_cwn_style_local":"CWN-style local cellular adaptation; typed endpoints preserve direction; absolute incidence gives scalar basis invariance, not signed-cochain reasoning","simplicial_mpsn_style_local":"MPSN-style local simplicial adaptation; full event-port subdivision and candidate fan; event features once per event vertex, structural segments/spokes are not transfers","directed_local_edge_gnn":"directed local edge-feature messages with distinct incoming/outgoing aggregation","cellular_hasse_mechanism_control":"cellular Hasse ranks 0/1/2 with typed source/target incidences"},"active_parameter_counts":params,"gradient_checks":grad,"permutation_basis_checks":perms,"actual_layer_feature_mutations":access,"inv009_pairwise_2_percent_matching":matching,"training":"none; no pilot-label fitting, optimizer, split, or efficacy metrics"}
    mapping={"rows":d["map_records"],"physical_event_mapping":"Every arm retains one coordinate for each physical event. Simplicial paths map four segments to one event vertex; fan spokes have no event identity.","basis_direction_rule":"Basis flips do not reverse physical source/target relations."}
    campaign={"basis":"78-candidate one-world smoke; not convergence or campaign evidence","conditional_projection":{"candidate_count_per_world":"C (pilot smoke 78; future yield unknown)","train_worlds":"W_train, not observed or authorized","validation_worlds":"W_val, not observed or authorized","paired_seeds":"S (5 is historical interface proposal only, not frozen)","epochs":"E remains symbolic; no convergence budget invented","formula":"forward/backward seconds = C/78 * W_train * S * E * measured_pass_seconds; validation forward-only unmeasured and excluded","assumption":"linear scaling and pilot-like candidate density; not a campaign authorization"}}
    runtime={"python":sys.version,"platform":platform.platform(),"packages":{"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"scikit_learn":sklearn.__version__,"matplotlib":matplotlib.__version__},"gudhi_available":gudhi_spec is not None,"gudhi_version":metadata.version("gudhi") if gudhi_spec is not None else None}
    neural_checks={}
    failed_conditions=[]
    for name in models:
        arm_ok=bool(grad[name]["pass"] and access[name]["changed"] and perms[name]["pass_atol_1e-6_rtol_1e-5"])
        neural_checks[name]={"gradient_reachability_pass":bool(grad[name]["pass"]),"local_feature_access_pass":bool(access[name]["changed"]),"permutation_and_basis_pass":bool(perms[name]["pass_atol_1e-6_rtol_1e-5"]),"interface_checks_pass":arm_ok,"model_qualified":False,"qualification_note":"interface checks do not establish predictive/model qualification"}
        for check_key,passed in [("gradient",grad[name]["pass"]),("feature_access",access[name]["changed"]),("permutation_basis",perms[name]["pass_atol_1e-6_rtol_1e-5"])]:
            if not passed: failed_conditions.append(f"{name}:{check_key}_failed")
    unmatched=[k for k,v in matching.items() if not v["within_2_percent"]]
    if unmatched: failed_conditions.append("INV-009 unmatched capacity pairs: "+", ".join(unmatched))
    if ph.get("external_backend_parity")=="FAIL": failed_conditions.append("external persistent-homology backend parity failed")
    if ph.get("external_backend_parity","").startswith("UNRESOLVED"): failed_conditions.append("external persistent-homology backend parity unresolved")
    all_interfaces=all(v["interface_checks_pass"] for v in neural_checks.values())
    qualification={"execution_completed":True,"all_neural_interface_checks_pass":all_interfaces,"all_arms_qualified":False,"inv009_all_pairwise_parameter_counts_within_2_percent":not unmatched,"unmatched_parameter_pairs":unmatched,"failed_or_unresolved_checks":failed_conditions,"note":"No arm is model-qualified: this is untrained one-world engineering evidence; capacity matching is assessed without padding."}
    model_detail={"arm_names":{"cellular_cwn_style_local":"CWN-style local cellular adaptation. One shared whole-world node/event representation per candidate; typed source/target endpoint projections retain physical direction. Absolute B1/B2 incidence provides scalar algebraic-basis invariance, not signed-cochain reasoning. Event states update from node incidences, endpoint messages, and the selected rank-2 face; the scalar readout combines that face state with a mean over all whole-world rank-1 event states.","simplicial_mpsn_style_local":"MPSN-style local simplicial adaptation on the full whole-world event-port subdivision plus the selected face fan. Each physical transfer has one event vertex carrying time/log amount/selection once; four path segments are structural incidences and fan spokes are not transactions. Vertex and face states exchange through absolute boundary incidence; scalar readout combines mean face and whole-world vertex states.","directed_local_edge_gnn":"Directed local edge-feature adaptation. Separate source/outgoing and target/incoming messages consume the neighboring endpoint state plus each physical event's time/log amount/selection; node states update from both aggregates. Readout averages central-node states and selected-event embeddings. Physical event direction remains explicit.","cellular_hasse_mechanism_control":"Cellular Hasse incidence control with rank-0 accounts, rank-1 physical events and one rank-2 face. Typed source/target incidence projections preserve direction independently of basis signs. Readout combines the face state with the mean over all whole-world event states.","gbdt":"Fixed 20-column per-candidate summaries. Central in/out degree uses the full world graph; selected directed-pair multiplicity counts all physical events in the world. This is a lossy map; only a separate mathematical fixture is fit."},"active_parameter_counts":params,"gradient_checks":grad,"permutation_basis_checks":perms,"actual_layer_feature_mutations":access,"inv009_pairwise_2_percent_matching":matching,"per_arm_interface_findings":neural_checks,"training":"none; no pilot-label fitting, optimizer, split, or efficacy metrics"}
    blockers=["single exposed development world; no independent-world precision/generalization","construction shortcuts remain available to all comparators","no learned efficacy metrics or predictive sampling authorized","no convergence budget or campaign cost established","protocols and splits remain unfrozen",*failed_conditions]
    command=[sys.executable,"-s","-B",str(Path(__file__).resolve()),"--world-dir",str(world),"--output-dir",str(out)]
    if model_config_path: command.extend(["--model-config",str(model_config_path)])
    report={"execution_status":"COMPLETED","scientific_qualification_status":"UNTRAINED_INTERFACE_CHECKS_REPORTED_NOT_MODEL_QUALIFIED","qualification_summary":qualification,"command":command,"cwd":str(Path.cwd()),"code_hash":sha(Path(__file__).resolve()),"model_config":model_config,"model_config_hash":sha(model_config_path) if model_config_path else None,"input_hashes":[sha(p) for p in d["inputs"]],"runtime":runtime,"view_summary":view_summary,"models":model_detail,"gbdt_fixture_smoke":gbdt,"fixed_shortcut_diagnostics":shortcuts,"persistent_homology":ph,"cost":costs,"conditional_campaign_cost":campaign,"scientific_blockers":blockers,"next_command":"No additional empirical run is part of this execution; subsequent scientific work requires separate review."}
    dump(out/"representation_map.json",mapping);dump(out/"views_and_math.json",view_summary);dump(out/"shortcut_diagnostics.json",shortcuts);dump(out/"persistent_homology.json",ph);dump(out/"model_checks.json",model_detail);dump(out/"gbdt_fixture_smoke.json",gbdt);dump(out/"cost_accounting.json",{"cost":costs,"conditional_campaign_cost":campaign})
    with (out/"cost_table.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["arm","active_trainable_parameters","forward_backward_measured_pass_seconds","timing_repeat_1_seconds","timing_repeat_2_seconds","dense_linear_forward_flop_subtotal"])
        for arm,record in costs.items():
            writer.writerow([arm,record["active_trainable_parameters"],record["forward_backward_seconds_measured_pass"],*record["forward_backward_seconds_timing_repeats"],record["dense_linear_forward"]["estimated_flops"]])
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(range(len(costs)),[costs[k]["forward_backward_seconds_measured_pass"] for k in costs],color="#416b8e"); ax.set_xticks(range(len(costs)),[x.replace("_","\n") for x in costs],fontsize=8); ax.set_ylabel("Seconds per 78-candidate forward/backward pass"); ax.set_title("Untrained S1 arm cost smoke"); fig.tight_layout();fig.savefig(out/"cost_smoke.png",dpi=160);plt.close(fig)
    report["output_hashes"]=[sha(p) for p in sorted(out.iterdir()) if p.is_file()]
    report["elapsed_seconds"]=time.time()-started;dump(out/"run_report.json",report)
    print(json.dumps({"execution_status":report["execution_status"],"qualification_status":report["scientific_qualification_status"],"candidates":len(rows),"arms":list(models),"output_dir":str(out)},indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
