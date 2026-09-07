"""Independent semantic-contract and exact-precomputation regression tests."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import erf, logsumexp
import jax
import jax.numpy as jnp
import numpyro.handlers as handlers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelSpecification as ms
from helper import import_dataset_hier

ROOT = Path(__file__).resolve().parents[3]
SCENE = np.array([[30,1,1],[25,1,0],[20,0,1],[15,0,0],[10,0,1],[1,0,0]],dtype=float)
UTTS = [tuple(int(t) for t in u if t>=0) for u in np.asarray(ms.utterance_list)]


def reference_listener(scene, prefix, updating, colour=.59, form=.5, k=.5, wf=.6856):
    prior = np.ones(len(scene))/len(scene)
    post = prior.copy()
    sizes = scene[:,0]
    order = np.argsort(sizes)
    for token in reversed(prefix):
        if token == 0:
            p = (post if updating else prior)[order]
            cdf = np.cumsum(p/(p.sum()+1e-8))
            lo, hi = [sizes[order[min(np.searchsorted(cdf,q),len(scene)-1)]] for q in (.2,.8)]
            theta = hi-k*(hi-lo)
            z = (sizes-theta)/(wf*np.sqrt(sizes**2+theta**2+1e-8))
            meaning = np.clip(.5*(1+erf(z/np.sqrt(2))),1e-8,None)
        else:
            reliability = colour if token==1 else form
            meaning = np.where(scene[:,token]==1,reliability,1-reliability)+1e-8
        post *= meaning
        post /= post.sum()
    return post


def test_prefix_reference_and_semantic_intervention():
    # Includes a five-object uniform-CDF boundary and tied-size cases.
    scenes = [SCENE, SCENE[:5], SCENE.copy()]
    scenes[-1][:,0] = [30,30,20,20,10,10]
    for scene in scenes:
        for updating in (False,True):
            expected=np.array([np.log(max(reference_listener(scene,u,updating)[0],1e-8)) for u in UTTS])
            actual=np.asarray(ms.principled_prefix_log_listeners(scene,recursive=updating))
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
            # Fixed listener is commutative, with every modifier applied once.
            if not updating:
                np.testing.assert_allclose(actual[1],actual[6],atol=1e-10,rtol=0)
    fixed=ms.principled_prefix_log_listeners(SCENE,recursive=False)
    updating=ms.principled_prefix_log_listeners(SCENE,recursive=True)
    assert float(jnp.max(jnp.abs(fixed-updating)))>1e-3
    for recursive in (False,True):
        neutral=ms.principled_prefix_log_listeners(SCENE,color_semval=.5,recursive=recursive)
        neutral_fixed=ms.principled_prefix_log_listeners(SCENE,color_semval=.5,recursive=False)
        np.testing.assert_allclose(neutral,neutral_fixed,atol=1e-10,rtol=0)


def test_path_reference_and_gradients():
    for updating in (False,True):
        raw=[]; norms=[]
        for u in UTTS:
            s=c=0.
            for t,token in enumerate(u):
                available=[a for a in range(3) if a not in u[:t]]
                scores=[1.441*np.log(reference_listener(SCENE,u[:t]+(a,),updating)[0]) for a in available]
                s+=scores[available.index(token)];c+=logsumexp(scores)
            raw.append(s);norms.append(c)
        actual=ms.principled_incremental_path_components(SCENE,1.,alpha=1.441,
                                                        recursive=updating,prefix_mode="B")
        np.testing.assert_allclose(actual[0],raw,atol=1e-10,rtol=0)
        np.testing.assert_allclose(actual[1],norms,atol=1e-10,rtol=0)
        def score(p):
            a,b,f=ms.principled_incremental_path_components(SCENE,1.,alpha=p[0],
                lambda_salience=p[1],recursive=updating,prefix_mode="B")
            return jnp.sum(a-.45*(b-f))
        p=jnp.array([1.441,.3]); grad=np.asarray(jax.grad(score)(p))
        numerical=np.array([(float(score(p.at[i].add(1e-5)))-float(score(p.at[i].add(-1e-5))))/2e-5 for i in (0,1)])
        np.testing.assert_allclose(grad,numerical,atol=1e-7,rtol=0)


def all_models():
    return {**ms.V10_ORDER_MODELS,**ms.V10_ORDER_UPD_MODELS,
            **ms.V11_JOINT_PARTICIPANT_MODELS,**ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS}


def test_registered_intervention_and_interpolation():
    args=dict(states=jnp.asarray(SCENE[None]),empirical=None,participant_idx=jnp.array([0]),
        n_participants=1,sufficient_dim=jnp.array([-1]),has_one_word_solution=jnp.array([0.]),
        is_sharp=jnp.array([1.]),is_colour_sufficient=jnp.array([0.]))
    def evaluate(model,params=None):
        fn=handlers.seed(model,17)
        if params is not None:fn=handlers.condition(fn,data=params)
        return handlers.trace(fn).get_trace(**args)
    models=all_models()
    for a,b in [("G-HO","G-UPD-HO"),("I-HO","I-UPD-HO"),
                ("K-HO","K-UPD-HO"),("K-HKO","K-UPD-HKO")]:
        first=evaluate(models[a])
        params={k:v["value"] for k,v in first.items() if v["type"]=="sample" and k!="obs"}
        second=evaluate(models[b],params)
        assert float(jnp.max(jnp.abs(first["obs"]["fn"].probs-second["obs"]["fn"].probs)))>1e-8
    for suffix in ("-HO","-UPD-HO"):
        model=models["K"+suffix];trace=evaluate(model)
        params={k:v["value"] for k,v in trace.items() if v["type"]=="sample" and k!="obs"}
        distributions=[]
        for kappa in (0.,.4,1.):
            params["kappa"]=jnp.asarray(kappa)
            distributions.append(np.asarray(evaluate(model,params)["obs"]["fn"].probs))
        for index,endpoint in ((0,"G"),(2,"I")):
            np.testing.assert_allclose(distributions[index],evaluate(models[endpoint+suffix],params)["obs"]["fn"].probs,
                                       atol=1e-10,rtol=0)
        g,k,i=[(p-.003/15)/.997 for p in distributions]
        expected=np.exp(.6*np.log(g)+.4*np.log(i));expected/=expected.sum(axis=-1,keepdims=True)
        np.testing.assert_allclose(k,expected,atol=1e-10,rtol=0)


def test_full_data_precomputation_and_replay(encoded_data=None):
    if encoded_data is None:
        data=import_dataset_hier(file_path=ROOT/"analysis_revision/12_deterministic_encoding/model_input_raw_observed_9100.csv",
                                 state_encoding="target_match")
    else:
        with np.load(encoded_data,allow_pickle=False) as f:
            data={k:jnp.asarray(v) for k,v in f.items()}
        data["n_participants"]=int(data["n_participants"])
    args=dict(states=data["states_train"],empirical=data["empirical_seq_flat"],
        participant_idx=data["participant_idx"],n_participants=data["n_participants"],
        sufficient_dim=data["sufficient_dim"],has_one_word_solution=data["has_one_word_solution"],
        is_sharp=data["sharpness_idx"],is_colour_sufficient=data["is_colour_sufficient"])
    features={r:ms.precompute_principled_discovery_features(args["states"],args["is_sharp"],recursive=r) for r in (False,True)}
    receipts=[]
    for name,model in all_models().items():
        trace=handlers.trace(handlers.seed(model,17)).get_trace(**args)
        params={k:v["value"] for k,v in trace.items() if v["type"]=="sample" and not v["is_observed"]}
        fast_args={**args,"precomputed_features":features["UPD" in name]}
        fast=handlers.trace(handlers.condition(model,data=params)).get_trace(**fast_args)
        p=np.asarray(trace["obs"]["fn"].probs);q=np.asarray(fast["obs"]["fn"].probs)
        error=float(np.max(np.abs(p-q)))
        np.testing.assert_allclose(p,q,atol=1e-10,rtol=0)
        assert np.isfinite(q).all() and np.min(q)>=.003/15-1e-12
        np.testing.assert_allclose(q.sum(-1),1,atol=1e-12,rtol=0)
        # Copying posterior sites through NumPy removes weak typing as NetCDF does.
        replay_params={k:jnp.asarray(np.array(v["value"])) for k,v in fast.items()
                       if v["type"] in ("sample","deterministic") and k!="obs"}
        replay=handlers.trace(handlers.substitute(model,data=replay_params)).get_trace(**fast_args)
        ll=np.asarray(fast["obs"]["fn"].log_prob(args["empirical"]))
        ll2=np.asarray(replay["obs"]["fn"].log_prob(args["empirical"]))
        replay_error=float(np.max(np.abs(ll-ll2)))
        np.testing.assert_allclose(ll,ll2,atol=1e-8,rtol=0)
        receipts.append(dict(model=name,rows=len(p),probability_error=error,replay_loglik_error=replay_error))
    return receipts


if __name__=="__main__":
    import argparse,csv,json
    parser=argparse.ArgumentParser();parser.add_argument("--output");parser.add_argument("--encoded-data");args=parser.parse_args()
    assert jax.config.x64_enabled
    test_prefix_reference_and_semantic_intervention()
    test_path_reference_and_gradients()
    test_registered_intervention_and_interpolation()
    rows=test_full_data_precomputation_and_replay(args.encoded_data)
    if args.output:
        with open(args.output,"w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
