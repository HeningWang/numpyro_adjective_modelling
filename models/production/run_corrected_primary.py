"""Run one source-bound primary production fit with reusable semantic features."""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import log_likelihood
import arviz as az
import modelSpecification as ms
from helper import import_dataset_hier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data/production_model_input.csv"
DATA_SHA256 = "44a41b44b83dfef7a2d1788c6ac399a637cb9cb48b63eb3751339ed849bf4ac1"
MODELS = {**ms.V10_ORDER_MODELS, **ms.V10_ORDER_UPD_MODELS,
          **ms.V11_JOINT_PARTICIPANT_MODELS, **ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def run(args):
    if args.recovery and (args.model not in {"G-HO", "G-UPD-HO"}
            or (args.warmup, args.samples, args.max_tree_depth) != (1000, 4000, 8)):
        raise ValueError("Recovery must match the approved two-model sampler contract")
    if jax.default_backend()!="gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Authoritative fits require CUDA and JAX x64")
    if args.input is None:
        if sha256(DATA)!=DATA_SHA256:raise ValueError("Dataset hash mismatch")
    else:
        if not args.input_sha256 or sha256(args.input)!=args.input_sha256:
            raise ValueError("Encoded input hash mismatch")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    artifact=out/f"{args.model}.nc"
    manifest_path=out/f"{args.model}_manifest.json"
    if artifact.exists() or manifest_path.exists():
        raise FileExistsError("Use a new output directory; existing artifacts are immutable")
    start=time.perf_counter()
    stages={}
    manifest=dict(model=args.model,dataset_sha256=DATA_SHA256,
        encoded_input_sha256=args.input_sha256,
        warmup=args.warmup,samples=args.samples,chains=4,seed=4711,
        target_accept=.9,max_tree_depth=args.max_tree_depth,chain_method="vectorized",
        artifact_role="recovery" if args.recovery else ("timing_pilot" if args.warmup!=1000 or args.samples!=1000 else "primary"),
        backend=jax.default_backend(),x64=jax.config.x64_enabled,
        devices=[str(x) for x in jax.devices()],jax=jax.__version__,numpyro=numpyro.__version__,
        source_hashes={p.name:sha256(p) for p in (Path(__file__),Path(ms.__file__),
            Path(__file__).with_name("helper.py"),Path(__file__).with_name("discovery.py"),
            Path(__file__).with_name("principled_features.py"))},
        stage="initializing",complete=False,stage_seconds=stages)
    def mark(stage,t0=None):
        if t0 is not None:stages[stage]=time.perf_counter()-t0
        manifest.update(stage=stage,total_seconds=time.perf_counter()-start)
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"model":args.model,"stage":stage,"seconds":stages.get(stage),
                          "total_seconds":manifest["total_seconds"]}),flush=True)
    mark("initializing")
    t0=time.perf_counter()
    if args.input is None:
        d=import_dataset_hier(file_path=DATA,state_encoding="target_match")
    else:
        with np.load(args.input,allow_pickle=False) as encoded:
            d={k:jnp.asarray(v) for k,v in encoded.items()}
        d["n_participants"]=int(d["n_participants"])
        if not np.array_equal(np.asarray(d["canonical_row_position"]),np.arange(9100)):
            raise ValueError("Encoded observation order mismatch")
    if len(d["states_train"])!=9100:raise ValueError("Expected 9,100 rows")
    kwargs=dict(states=d["states_train"],empirical=d["empirical_seq_flat"],
        participant_idx=d["participant_idx"],n_participants=d["n_participants"],
        sufficient_dim=d["sufficient_dim"],has_one_word_solution=d["has_one_word_solution"],
        is_sharp=d["sharpness_idx"],is_colour_sufficient=d["is_colour_sufficient"])
    kwargs["precomputed_features"]=ms.precompute_principled_discovery_features(
        kwargs["states"],kwargs["is_sharp"],recursive="UPD" in args.model)
    jax.block_until_ready(kwargs["precomputed_features"])
    mark("precompute",t0)
    model=MODELS[args.model]
    kernel=NUTS(model,target_accept_prob=.9,max_tree_depth=args.max_tree_depth)
    mcmc=MCMC(kernel,num_warmup=args.warmup,num_samples=args.samples,num_chains=4,
              chain_method="vectorized",progress_bar=False)
    _,key=random.split(random.PRNGKey(4711))
    extra=("energy","potential_energy","num_steps","accept_prob","mean_accept_prob")
    t0=time.perf_counter();mark("warmup_running")
    mcmc.warmup(key,extra_fields=extra,**kwargs)
    jax.block_until_ready(mcmc.last_state)
    mark("warmup",t0)
    t0=time.perf_counter();mark("sampling_running")
    mcmc.run(mcmc.post_warmup_state.rng_key,extra_fields=extra,**kwargs)
    jax.block_until_ready(mcmc.last_state)
    mark("sampling",t0)
    posterior={k:np.asarray(v) for k,v in mcmc.get_samples(group_by_chain=True).items()}
    stats={k:np.asarray(v) for k,v in mcmc.get_extra_fields(group_by_chain=True).items()}
    stats["n_steps"]=stats.pop("num_steps")
    # Preserve completed sampling if prediction or serialization is interrupted.
    np.savez_compressed(out/f"{args.model}_sampling_checkpoint.npz",
        **{f"posterior__{k}":v for k,v in posterior.items()},
        **{f"stats__{k}":v for k,v in stats.items()})
    total=4*args.samples
    flat={k:jnp.asarray(v.reshape(total,*v.shape[2:])) for k,v in posterior.items()}
    # Reproduce inference-time likelihoods using stochastic sites only.
    trace=handlers.trace(handlers.seed(model,0)).get_trace(**kwargs)
    stochastic={k for k,v in trace.items() if v["type"]=="sample" and not v["is_observed"]}
    # Both stochastic replay and stored-deterministic substitution must agree.
    all_first={k:v[0] for k,v in flat.items()}
    stochastic_first={k:v for k,v in all_first.items() if k in stochastic}
    replay_a=handlers.trace(handlers.substitute(model,data=all_first)).get_trace(**kwargs)
    replay_b=handlers.trace(handlers.condition(model,data=stochastic_first)).get_trace(**kwargs)
    replay_error=float(jnp.max(jnp.abs(replay_a["obs"]["fn"].log_prob(kwargs["empirical"])
                                     -replay_b["obs"]["fn"].log_prob(kwargs["empirical"]))))
    manifest["replay_max_abs_loglik_difference"]=replay_error
    if replay_error>=1e-8:raise AssertionError(f"Posterior replay mismatch {replay_error}")
    ll=np.empty((total,9100),dtype=np.float64)
    pp=np.empty((total,9100),dtype=np.int8)
    pp_kwargs={**kwargs,"empirical":None}
    t0=time.perf_counter();mark("prediction_running")
    for begin in range(0,total,args.batch):
        end=min(begin+args.batch,total)
        batch={k:v[begin:end] for k,v in flat.items() if k in stochastic}
        ll[begin:end]=np.asarray(log_likelihood(model,batch,parallel=True,**kwargs)["obs"])
        pp[begin:end]=np.asarray(Predictive(model,batch,return_sites=["obs"],parallel=True)(
            random.fold_in(random.PRNGKey(1),begin),**pp_kwargs)["obs"])
    mark("prediction",t0)
    # Ensure the batched serialization path matches single-draw inference replay.
    batched_error=float(np.max(np.abs(ll[0]-np.asarray(replay_b["obs"]["fn"].log_prob(kwargs["empirical"])))))
    manifest["batched_replay_max_abs_loglik_difference"]=batched_error
    if batched_error>=1e-8:raise AssertionError(f"Batched replay mismatch {batched_error}")
    dims={k:["participants"] for k,v in posterior.items() if v.ndim==3 and v.shape[-1]==d["n_participants"]}
    dims["obs"]=["item"]
    idata=az.from_dict(posterior=posterior,sample_stats=stats,
        log_likelihood={"obs":ll.reshape(4,args.samples,9100)},
        posterior_predictive={"obs":pp.reshape(4,args.samples,9100)},
        observed_data={"obs":np.asarray(kwargs["empirical"])},dims=dims,
        coords={"participants":np.arange(d["n_participants"]),"item":np.arange(9100)})
    idata.attrs.update(source_sha256=manifest["source_hashes"]["modelSpecification.py"],
        dataset_sha256=DATA_SHA256,model=args.model,semantic_contract="prefix_once_right_to_left_float64",
        seed=4711,artifact_role=manifest["artifact_role"])
    t0=time.perf_counter();idata.to_netcdf(artifact);mark("serialization",t0)
    manifest.update(artifact_sha256=sha256(artifact),complete=True)
    mark("complete")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--warmup",type=int,default=1000);p.add_argument("--samples",type=int,default=1000)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--max-tree-depth",type=int,default=6)
    p.add_argument("--recovery",action="store_true")
    p.add_argument("--input",type=Path)
    p.add_argument("--input-sha256")
    run(p.parse_args())
