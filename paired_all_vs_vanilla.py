"""Requested paired comparisons, primary L2 C=.1, all speakers retained."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/jasmin_six_signal_lr'
OUT=BASE/'paired_all_vs_vanilla'
METRICS=['roc_auc','ap_positive','pr_auc_positive']

def main():
 OUT.mkdir(exist_ok=True)
 reps=pd.read_csv(BASE/'bootstrap_replicates.csv');points=pd.read_csv(BASE/'summary.csv');previous=pd.read_csv(BASE/'paired_method_differences.csv')
 summaries=[];draws=[]
 for (variant,task,protocol),g in reps.groupby(['variant','task','protocol'],sort=False):
  baseline='lr_vanilla_C0.1';candidate='lr_all_without_contrast_C0.1' if task=='task2' else 'lr_all_with_word_contrast_C0.1'
  b=g[g.method.eq(baseline)].set_index('replicate');c=g[g.method.eq(candidate)].set_index('replicate')
  assert b.index.equals(c.index) and b.index.tolist()==list(range(1,2001))
  d=c[METRICS]-b[METRICS];assert np.isfinite(d.to_numpy()).all()
  pts=points[(points.variant==variant)&(points.task==task)&(points.protocol==protocol)].set_index('method')
  for metric in METRICS:
   lo,hi=np.quantile(d[metric],[.025,.975]);delta=float(pts.loc[candidate,metric]-pts.loc[baseline,metric])
   old=previous[(previous.variant==variant)&(previous.task==task)&(previous.protocol==protocol)&(previous.baseline==baseline)&(previous.candidate==candidate)&(previous.metric==metric)].iloc[0]
   np.testing.assert_allclose([delta,lo,hi],old[['delta','ci_lower','ci_upper']].to_numpy(float),atol=1e-12)
   summaries.append(dict(variant=variant,task=task,protocol=protocol,baseline=baseline,candidate=candidate,metric=metric,vanilla_score=pts.loc[baseline,metric],all_score=pts.loc[candidate,metric],delta_all_minus_vanilla=delta,ci_lower=lo,ci_upper=hi,n_bootstrap=2000,ci_excludes_zero=bool(lo>0 or hi<0)))
  d=d.reset_index().rename(columns={m:'delta_'+m for m in METRICS});d['variant']=variant;d['task']=task;d['protocol']=protocol;draws.append(d)
 s=pd.DataFrame(summaries);s.to_csv(OUT/'summary.csv',index=False);pd.concat(draws).to_csv(OUT/'paired_replicates.csv',index=False)
 (OUT/'metadata.json').write_text(json.dumps(dict(C=.1,seed=42,n_bootstrap=2000,difference='all minus vanilla',ci='95% percentile of within-replicate differences',sampling='Shared saved 30-speaker multiplicities; pooled word metrics; fixed models',excluded_speakers=[],task2='Both CCI retained and removed protocols, each comparing identical test words',task3='Filtered common word-contrast-valid cohort',multiplicity='Intervals are pointwise; no multiple-comparison adjustment',validation='Recomputed from paired replicates and matched previously saved comparisons'),indent=2)+'\n')
 lines=['# Paired bootstrap: LR all minus LR vanilla','', 'Primary L2 C=0.1. All speakers retained. Same words and speaker multiplicities for both methods within each comparison; 2000 draws, seed42, fixed models, pooled word metrics. Intervals are pointwise 95% percentile intervals, without multiplicity adjustment.','', '| Version | Task | Protocol | Metric | Vanilla | All | Difference [95% CI] |','|---|---|---|---|---|---|---|']
 for _,r in s.iterrows():lines.append(f'| {r.variant} | {r.task} | {r.protocol} | {r.metric} | {r.vanilla_score:.4f} | {r.all_score:.4f} | {r.delta_all_minus_vanilla:+.4f} [{r.ci_lower:.4f}, {r.ci_upper:.4f}] |')
 (OUT/'README.md').write_text('\n'.join(lines)+'\n')
 print('Verified and saved 18 metric comparisons, 12000 paired bootstrap replicates.')

if __name__=='__main__':main()
