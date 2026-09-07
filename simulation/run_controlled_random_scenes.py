"""Controlled model interventions on the preserved two-order random scenes.

No fitting or scene generation. Reads the existing sweep, checks identities,
and saves scene-paired outcomes and summaries under a separate output path.
"""
from __future__ import annotations
import os
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'models/production'))
import modelSpecification as ms
import core_rsa as old

LABELS = ms.UTTERANCE_LABELS
SEQS = np.asarray(ms.utterance_list)
SUPPORTS = {'two': [1, 6], 'four': [0, 1, 5, 6], 'fifteen': list(range(15))}
RULES = [('terminal', None), ('prefix_k0', 0.), ('prefix_k05', .5), ('prefix_k1', 1.)]
BETA = float(np.exp(.9623570742792675))
F0 = np.array([[.59, .5, .6856]])
PFX = ms.PREFIX_LISTENER_INDICES
ACTIVE = ms.ACTIVE_POS
ONEHOT = ms.ACTUAL_TOK_ONEHOT
MASKS = {}
OUTPUT_MASKS = {}
for support, ids in SUPPORTS.items():
    prefixes = {tuple(row[:i]) for row in SEQS[ids] for i in range(1,4) if row[i-1] >= 0}
    valid = np.array([tuple(a for a in row if a >= 0) in prefixes for row in SEQS])
    MASKS[support] = jnp.asarray(np.asarray(ms.CANDIDATE_MASK) & valid[np.asarray(PFX)])
    OUTPUT_MASKS[support] = jnp.asarray(np.isin(np.arange(15), ids))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def literal_all(scene, params, recursive):
    """Full referent vectors using the production lexical kernel exactly."""
    colour, k, wf = params
    prior = jnp.ones(scene.shape[0], dtype=jnp.float64)/scene.shape[0]
    sizes = scene[:, 0]
    idx = jnp.argsort(sizes)
    def size(post):
        return jnp.clip(ms.compute_size_semantics_fast_presorted(sizes, idx, sizes[idx], post, k, wf), 1e-8)
    fixed = size(prior)
    color = jnp.where(scene[:,1] == 1, colour, 1-colour) + 1e-8
    form = jnp.full_like(color, .5 + 1e-8)
    def interpret(tokens):
        def step(post, token):
            m = jnp.stack([size(post) if recursive else fixed, color, form])[jnp.maximum(token,0)]
            updated = post*m
            updated = updated/updated.sum()
            return jnp.where(token >= 0, updated, post), None
        return jax.lax.scan(step, prior, tokens[::-1])[0]
    return jax.vmap(interpret)(ms.utterance_list)

def path_parts(log_target, support):
    scores = log_target[PFX]
    scores = jnp.where(MASKS[support], scores, -jnp.inf)
    inert = jnp.full_like(scores, -jnp.inf).at[:,:,0].set(0.)
    scores = jnp.where(ACTIVE[:,:,None], scores, inert)
    chosen = jnp.sum(jnp.where(ONEHOT > 0, scores, 0.), axis=-1)
    norms = logsumexp(scores, axis=-1)
    chosen = jnp.where(ACTIVE, chosen, 0.)
    norms = jnp.where(ACTIVE, norms, 0.)
    probabilities = jnp.sum(jax.nn.softmax(scores, axis=-1)*ONEHOT, axis=-1)
    probabilities = jnp.where(ACTIVE, probabilities, 1.)
    floor = jnp.where(ACTIVE, jnp.log(jnp.clip(probabilities,1e-8)),0.) - (chosen-norms)
    return jnp.sum(chosen,axis=0), jnp.sum(norms-floor,axis=0)

def structural_probs(literal, support, rule, kappa, beta):
    logs = jnp.log(jnp.clip(literal.T, 1e-8))
    if rule == 'terminal':
        utility = logs
    else:
        raw, normalizers = jax.vmap(lambda v:path_parts(v,support))(logs)
        # Unsupported paths may have -inf utility; eliminate before arithmetic.
        raw = jnp.where(OUTPUT_MASKS[support], raw, 0.)
        normalizers = jnp.where(OUTPUT_MASKS[support], normalizers, 0.)
        utility = raw-kappa*normalizers
    utility = utility + beta*ms.LOG_LM_ORDER_ONLY_15
    return jax.nn.softmax(jnp.where(OUTPUT_MASKS[support],utility,-jnp.inf),axis=-1)

def old_probs_from_literal(literal, incremental):
    L1 = literal[jnp.array([0,5]),:].T
    L2 = literal[jnp.array([1,6]),:].T
    if not incremental:
        return jax.nn.softmax(jnp.log(jnp.clip(L2,1e-20,1.)),axis=-1)
    first = L1/jnp.clip(L1.sum(axis=-1,keepdims=True),1e-20)
    second = L2/jnp.clip(L2+L1,1e-20)
    return jax.nn.softmax(jnp.log(jnp.clip(first*second,1e-20,1.)),axis=-1)

def metric(prob, ids):
    p = prob[:,jnp.asarray(ids)]
    listener = p/jnp.sum(p,axis=0,keepdims=True)
    return jnp.array([listener[0,0],listener[0,1],p[0,0]/p[0].sum()])

def configs(with_order):
    rows=[]
    for sem in ['fixed','updating']:
        for variant in ['original_global','original_incremental','kernel_global','kernel_incremental']:
            rows.append(dict(semantics=sem,rule=variant,support='two',order='neutral'))
        for support in SUPPORTS:
            for rule,kappa in RULES:
                for order in (['neutral','stable'] if with_order else ['neutral']):
                    rows.append(dict(semantics=sem,rule=rule,support=support,order=order))
    return rows

def evaluate(scene, params, with_order):
    values=[]
    for rec in [False,True]:
        for inc in [False,True]:
            speaker = ('incremental_speaker' if inc else 'global_speaker')+('_static' if not rec else '')
            L = old.pragmatic_listener(scene,1.,0.,params[0],None,params[2],params[1],speaker,2)
            # The old conditional production score is obtained from the same rule.
            fn = (old.speaker_recursive if rec else old.speaker_recursive_frozen) if inc else (old.global_speaker if rec else old.global_speaker_static)
            args = (2,scene) if inc else (scene,)
            S = fn(*args,alpha=1.,bias=0.,color_semvalue=params[0],wf=params[2],k=params[1])
            values.append(jnp.array([L[0,0],L[1,0],S[0,0]]))
        literal = literal_all(scene,params,rec)
        for inc in [False,True]:
            values.append(metric(old_probs_from_literal(literal,inc),[0,1]))
        for support in SUPPORTS:
            for rule,kappa in RULES:
                for beta in ([0.,BETA] if with_order else [0.]):
                    values.append(metric(structural_probs(literal,support,rule,kappa,beta),[1,6]))
    return jnp.stack(values)

def validate(scene, params):
    errors={}
    for rec in [False,True]:
        L=np.asarray(literal_all(jnp.asarray(scene),jnp.asarray(params),rec))
        np.testing.assert_allclose(L.sum(axis=1),1,atol=1e-12)
        ref=np.asarray(ms.principled_prefix_log_listeners(jnp.asarray(scene),params[0],.5,params[1],params[2],rec))
        err=float(np.max(np.abs(np.log(np.clip(L[:,0],1e-8,None))-ref)))
        assert err < 1e-11
        errors[f'lexical_{rec}']=err
        for rule,kappa in RULES[1:]:
            actual=np.asarray(structural_probs(jnp.asarray(L),'fifteen',rule,kappa,0.))[0]
            expected=np.asarray(ms.incremental_speaker_principled_discovery(
                states=jnp.asarray(scene),sufficient_dim=-1,has_one_word_solution=0.,is_sharp=0.,is_colour_sufficient=0.,
                alpha=1.,beta_order=0.,kappa=kappa,nu_F=.5,color_semval=params[0],
                size_threshold_k=params[1],wf=params[2],epsilon=0.,recursive=rec,prefix_mode='B'))
            err=float(np.max(np.abs(actual-expected)))
            assert err < 1e-10,(rec,rule,err)
            errors[f'forward_{rec}_{rule}']=err
    return errors

def mean_se(x):
    return float(x.mean()),float(x.std(ddof=1)/np.sqrt(len(x)))

def summarize(array, cfg, nobj, spread, parameter_set, grid, rows, paired, cells):
    # array: parameter x scene x config x (listener DC,listener CD,speaker DC)
    adv=array[:,:,:,0]-array[:,:,:,1]
    for j,c in enumerate(cfg):
        for k,name in [(None,'listener_advantage'),(2,'speaker_size_first_given_set')]:
            y=adv[:,:,j] if k is None else array[:,:,j,k]
            by_scene=y.mean(axis=0)
            m,se=mean_se(by_scene)
            rows.append(dict(**c,nobj=nobj,spread=spread,parameter_set=parameter_set,metric=name,
                             mean=m,mcse=se,scenes=len(by_scene),parameter_combinations=len(grid)))
        for p,pars in enumerate(grid):
            m,se=mean_se(adv[p,:,j])
            cells.append(dict(**c,nobj=nobj,spread=spread,parameter_set=parameter_set,
                colour=float(pars[0]),k=float(pars[1]),wf=float(pars[2]),mean=m,mcse=se))
    lookup={tuple(c[k] for k in ['semantics','rule','support','order']):i for i,c in enumerate(cfg)}
    contrasts=[]
    for j,c in enumerate(cfg):
        key=tuple(c[k] for k in ['semantics','rule','support','order'])
        if c['semantics']=='updating':
            contrasts.append(('updating_minus_fixed',j,lookup[('fixed',*key[1:])]))
        if c['rule'] in ['kernel_global','kernel_incremental']:
            contrasts.append(('corrected_kernel_minus_original',j,lookup[(key[0],c['rule'].replace('kernel','original'),*key[2:])]))
        if c['rule']=='prefix_k0':
            contrasts.append(('prefix_sum_minus_terminal',j,lookup[(key[0],'terminal',*key[2:])]))
        if c['rule'] in ['prefix_k05','prefix_k1']:
            contrasts.append(('local_normalizers_minus_raw_prefix',j,lookup[(key[0],'prefix_k0',*key[2:])]))
        if c['support'] in ['four','fifteen']:
            prev='two' if c['support']=='four' else 'four'
            contrasts.append(('support_'+c['support']+'_minus_'+prev,j,lookup[(key[0],key[1],prev,key[3])]))
        if c['order']=='stable':
            contrasts.append(('stable_order_minus_neutral',j,lookup[(*key[:3],'neutral')]))
    for name,j,i in contrasts:
        y=(adv[:,:,j]-adv[:,:,i]).mean(axis=0)
        m,se=mean_se(y)
        paired.append(dict(contrast=name,**cfg[j],parameter_set=parameter_set,nobj=nobj,spread=spread,mean=m,mcse=se,scenes=len(y)))
    return adv.mean(axis=0)

def main(a):
    assert jax.default_backend()=='gpu' and jax.config.x64_enabled
    a.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    summaries=[];paired=[];cells=[];checks=[];input_hashes={}
    files=sorted(a.scenes.glob('scenes_n*_sd*.npz'), key=lambda p:(int(p.stem.split('_')[1][1:]),float(p.stem.split('_sd')[1])))
    assert len(files)==24
    # Identity against the previously frozen analysis manifest.
    frozen=json.loads((a.scenes.parent/'analysis_manifest.json').read_text())
    (a.output/'input_manifest_structure.json').write_text(json.dumps({'keys':list(frozen)},indent=2))
    funcs={}
    previous_nobj=None
    for file in files:
        digest=sha(file)
        assert digest in json.dumps(frozen),f'Not in frozen input manifest: {file}'
        input_hashes[str(file)]=digest
        nobj=int(file.stem.split('_')[1][1:]);spread=float(file.stem.split('_sd')[1])
        if previous_nobj != nobj:
            funcs.clear()
            jax.clear_caches()
            previous_nobj=nobj
        with np.load(file) as z:
            scenes=z['scenes'];grid=z['grid'];reference=z['results']
        assert scenes.shape==(1000,nobj,3)
        print(json.dumps(dict(validating=file.name)),flush=True)
        checks.append(dict(file=file.name,**validate(scenes[0],grid[0])))
        group_values={}
        for parameter_set,params,with_order in [('original_grid',grid,False),('fitted_constants',F0,True)]:
            key=(nobj,with_order)
            cfg=configs(with_order)
            saved=a.output/f'{file.stem}_{parameter_set}.npz'
            if saved.exists():
                with np.load(saved) as z:
                    assert str(z['source_scene_sha256'])==digest
                    assert json.loads(str(z['configs']))==cfg
                    np.testing.assert_array_equal(z['grid'],params)
                    result=z['metrics']
            else:
                print(json.dumps(dict(computing=file.name,parameter_set=parameter_set)),flush=True)
                if nobj==2 and with_order:
                    # This graph triggers a native compiler allocator error when
                    # fused. Keep the identical vmap on CUDA, with primitive-level
                    # compilation; no changes to scenes, parameters or formulas.
                    result=np.asarray(jax.vmap(lambda scene:evaluate(scene,jnp.asarray(params[0]),True))(jnp.asarray(scenes)))[None,...]
                else:
                    if key not in funcs:
                        funcs[key]=jax.jit(jax.vmap(jax.vmap(lambda s,p,flag=with_order:evaluate(s,p,flag),in_axes=(0,None)),in_axes=(None,0)))
                    fn=funcs[key]
                    result=np.concatenate([np.asarray(fn(jnp.asarray(scenes[b:b+a.batch]),jnp.asarray(params))) for b in range(0,1000,a.batch)],axis=1)
            assert result.shape==(len(params),1000,len(cfg),3)
            assert np.isfinite(result).all() and result.min()>=0 and result.max()<=1+1e-12
            if parameter_set=='original_grid':
                ids=[i for i,c in enumerate(cfg) if c['rule'].startswith('original')]
                err=float(np.max(np.abs(result[:,:,ids,:2].transpose(2,0,1,3)-reference)))
                assert err<1e-11,err
                checks[-1]['old_replay_max_error']=err
            for i,c in enumerate(cfg):
                if c['semantics']=='fixed' and (c['rule']=='terminal' or c['rule'].endswith('_global')):
                    err=float(np.max(np.abs(result[:,:,i,0]-result[:,:,i,1])))
                    assert err<1e-10,(c,err)
            adv=summarize(result,cfg,nobj,spread,parameter_set,params,summaries,paired,cells)
            group_values[parameter_set]=(adv,cfg)
            if not saved.exists():
                np.savez_compressed(saved,metrics=result,grid=params,configs=json.dumps(cfg),source_scene_sha256=digest)
        base,bcfg=group_values['original_grid'];new,ncfg=group_values['fitted_constants']
        lookup={tuple(c.values()):i for i,c in enumerate(ncfg)}
        for i,c in enumerate(bcfg):
            if c['rule'].startswith('original') or c['rule'].startswith('kernel'):continue
            j=lookup[tuple(c.values())]
            m,se=mean_se(new[:,j]-base[:,i])
            paired.append(dict(contrast='fitted_constants_minus_grid_average',**c,parameter_set='paired_constants',nobj=nobj,spread=spread,mean=m,mcse=se,scenes=1000))
        pd.DataFrame(summaries).to_csv(a.output/'context_summary.csv',index=False)
        pd.DataFrame(paired).to_csv(a.output/'paired_contrasts.csv',index=False)
        pd.DataFrame(cells).to_csv(a.output/'cell_summary.csv',index=False)
        print(json.dumps(dict(scene=file.name,seconds=time.monotonic()-start,replay_error=checks[-1]['old_replay_max_error'])),flush=True)
    for filename,keys in [('context_summary',['semantics','rule','support','order','parameter_set','metric']),('paired_contrasts',['contrast','semantics','rule','support','order','parameter_set'])]:
        frame=pd.read_csv(a.output/f'{filename}.csv')
        frame['variance']=frame.mcse**2
        out=frame.groupby(keys,dropna=False).agg(mean=('mean','mean'),variance=('variance','sum'),groups=('mean','size')).reset_index()
        out['mcse']=np.sqrt(out.variance)/out.groups
        out['lower_mc95']=out['mean']-1.96*out.mcse;out['upper_mc95']=out['mean']+1.96*out.mcse
        out.to_csv(a.output/f'{filename}_overall.csv',index=False)
    receipt=dict(complete=True,seconds=time.monotonic()-start,backend=jax.default_backend(),x64=jax.config.x64_enabled,
        devices=[str(d) for d in jax.devices()],seed_of_preserved_scenes=20260906,input_hashes=input_hashes,checks=checks,
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(ms.__file__),Path(old.__file__),ROOT/'models/production/discovery.py',ROOT/'models/production/principled_features.py']},
        plan_sha256=sha(a.output.parent/'plan.md'),new_fits=0,new_rental_usd=0,unplanned_analyses=[])
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','input_hashes','sources']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=25)
    main(p.parse_args())
