"""Plot saved controlled-simulation summaries without recomputing statistics."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'fixed':'#477A84','updating':'#AD534A'}
NAMES={'fixed':'Context-fixed','updating':'Context-updating'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})

def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.results/'context_summary_overall.csv')
    df=df[df.metric=='listener_advantage']
    stages=[('original_global','two','original_grid','neutral','Original terminal utility'),
        ('terminal','two','original_grid','neutral','Corrected terminal evaluation'),
        ('prefix_k0','two','original_grid','neutral','Add prefix evaluation'),
        ('prefix_k0','four','original_grid','neutral','Allow one-adjective responses'),
        ('prefix_k0','fifteen','original_grid','neutral','Expand to fifteen responses'),
        ('prefix_k0','fifteen','fitted_constants','neutral','Use production semantic constants'),
        ('prefix_k0','fifteen','fitted_constants','stable','Add stable-order score')]
    fig,ax=plt.subplots(figsize=(8.2,5))
    for si,sem in enumerate(COLORS):
        means=[];errors=[]
        for rule,support,param,order,label in stages:
            row=df[(df.rule==rule)&(df.support==support)&(df.parameter_set==param)&(df.order==order)&(df.semantics==sem)]
            assert len(row)==1
            means.append(100*row.iloc[0]['mean']);errors.append(196*row.iloc[0].mcse)
        ax.errorbar(means,np.arange(len(stages))+(si-.5)*.18,xerr=errors,fmt='o',capsize=3,
            color=COLORS[sem],label=NAMES[sem],markersize=5)
    ax.axvline(0,color='.5',lw=.7);ax.set_yticks(np.arange(len(stages)),[s[-1] for s in stages]);ax.invert_yaxis()
    ax.set_xlabel('Target-probability difference (size-first − colour-first), percentage points')
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2);ax.grid(axis='x',alpha=.15)
    fig.tight_layout();fig.savefig(a.output/'controlled_steps.pdf');fig.savefig(a.output/'controlled_steps.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
    settings=[('original_grid','neutral','Original parameter grid'),('fitted_constants','neutral','Production semantic constants'),('fitted_constants','stable','Constants + stable-order score')]
    for ax,(param,order,label) in zip(axes,settings):
        for si,sem in enumerate(COLORS):
            means=[];errors=[]
            for rule in ['terminal','prefix_k0','prefix_k05','prefix_k1']:
                row=df[(df.rule==rule)&(df.support=='fifteen')&(df.parameter_set==param)&(df.order==order)&(df.semantics==sem)].iloc[0]
                means.append(row['mean']*100);errors.append(row.mcse*196)
            ax.errorbar(np.arange(4)+(si-.5)*.12,means,yerr=errors,fmt='o',capsize=3,color=COLORS[sem],label=NAMES[sem])
        ax.axhline(0,color='.5',lw=.7);ax.set_xticks(np.arange(4),['Terminal','Prefix\nκ=0','Prefix\nκ=.5','Prefix\nκ=1']);ax.set_xlabel(label)
    axes[0].set_ylabel('Target-probability difference\n(percentage points)');axes[0].legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
    fig.tight_layout();fig.savefig(a.output/'controlled_architectures.pdf');fig.savefig(a.output/'controlled_architectures.png',dpi=180);plt.close(fig)
    d=pd.read_csv(a.results/'context_summary.csv');d=d[(d.metric=='listener_advantage')&(d.order=='neutral')&(d.parameter_set=='original_grid')]
    rows=[('original_global','two','Terminal'),('prefix_k0','two','Prefix, two responses'),('prefix_k0','fifteen','Prefix, fifteen responses')]
    fig,axes=plt.subplots(3,2,figsize=(9,8),sharex=True,sharey=True)
    palette=['#687EB0','#A65A51','#55866B']
    for r,(rule,support,label) in enumerate(rows):
        for j,sem in enumerate(COLORS):
            ax=axes[r,j]
            for color,spread in zip(palette,[2.,7.75,15.]):
                y=d[(d.rule==rule)&(d.support==support)&(d.semantics==sem)&(d.spread==spread)].sort_values('nobj')
                ax.errorbar(y.nobj,y['mean']*100,yerr=y.mcse*196,color=color,fmt='.-',capsize=2,label=str(spread))
            ax.axhline(0,color='.5',lw=.7)
            if j==0:ax.set_ylabel(label+'\nDifference (percentage points)')
            if r==0:ax.set_title(NAMES[sem]);ax.legend(title='Size spread',frameon=False)
            if r==2:ax.set_xlabel('Number of objects')
    fig.tight_layout();fig.savefig(a.output/'controlled_scene_dependence.pdf');fig.savefig(a.output/'controlled_scene_dependence.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
