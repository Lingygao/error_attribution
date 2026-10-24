"""Six-signal LR: all speakers, task2 CCI keep/remove, filtered task3."""
import hashlib
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from evaluate_jasmin_cci_only import ROOT,METRICS,metric_function,reference,add_pair
from six_signal_config import VANILLA,BASE,WORD,GROUPS

SOURCE=ROOT/'notebooks/jasmin_ctc_contrast_task3'
OUT=ROOT/'notebooks/jasmin_six_signal_lr'

def main():
 OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
 plan_path=SOURCE/'bootstrap_speaker_counts.csv';plan=pd.read_csv(plan_path);speakers=list(plan.columns[1:]);units_map={s:i for i,s in enumerate(speakers)};counts=plan.iloc[:,1:].to_numpy(int)
 assert len(counts)==2000 and len(speakers)==30 and (counts.sum(axis=1)==30).all()
 assert {'fn000117','fn000524','fn000531'}<=set(speakers)
 plan.to_csv(OUT/'bootstrap_speaker_counts.csv',index=False)
 paths={plan_path,Path(__file__).resolve(),ROOT/'scripts/six_signal_config.py',ROOT/'scripts/evaluate_jasmin_cci_only.py',ROOT/'scripts/batch_lm_word_scores_from_newsegments.py',SOURCE/'metadata.json'}
 summaries=[];replicates=[];cohorts=[];manifests=[];coefs=[];pairs=[];protocol_pairs=[];all_results={}
 for variant in ['v2','v3']:
  data={}
  for split in ['train','test']:
   p=SOURCE/f'{variant}_{split}_aligned.csv';paths.add(p);f=pd.read_csv(p,low_memory=False)
   confidence=f.lm_confidence.to_numpy(float)
   assert np.isfinite(confidence).all() and (confidence>0).all() and (confidence<=1+1e-12).all(), 'Cannot recover log probability from invalid confidence'
   f['vanilla_log_probability']=np.log(confidence)
   assert not f.duplicated(['speaker','asr_index']).any()
   allc=f.pr_mt_labels.fillna('').map(lambda s:bool(str(s).strip()) and all(t.strip().upper()=='C' for t in str(s).split('|')))
   f['is_cci']=f.mt_v_label.eq('C') & f.pr_v_label.eq('I') & allc
   data[split]=f
  for task,protocol in [('task2','cci_retained'),('task2','cci_removed'),('task3','filtered_word_common')]:
   selected={}
   for split,f in data.items():
    g=f[f[task].notna()].copy();valid=np.isfinite(g[BASE].to_numpy(float)).all(axis=1)
    if task=='task2' and protocol=='cci_removed':valid &= ~g.is_cci.to_numpy()
    if task=='task3':valid &= np.isfinite(g[WORD].to_numpy(float))
    m=g[['speaker','asr_index','asr_word',task,'is_cci']].rename(columns={task:'y_true'}).copy();m['variant']=variant;m['task']=task;m['protocol']=protocol;m['split']=split;m['retained']=valid;manifests.append(m)
    selected[split]=g.loc[valid].copy();h=selected[split]
    assert set(h.speaker)==set(f.speaker)
    cohorts.append(dict(variant=variant,task=task,protocol=protocol,split=split,n_before=len(g),n_words=len(h),n_positive=int(h[task].sum()),n_negative=int((h[task]==0).sum()),n_speakers=h.speaker.nunique(),n_cci_retained=int(h.is_cci.sum())))
   tr,te=selected['train'],selected['test'];assert not set(tr.speaker)&set(te.speaker);assert set(te.speaker)==set(speakers)
   y=te[task].to_numpy(int);units=te.speaker.map(units_map).to_numpy(int)
   pred=te[['speaker','asr_index','asr_word']].copy();pred['y_true']=y
   groups={k:v for k,v in GROUPS.items() if task=='task3' or WORD not in v};results={}
   for group,features in groups.items():
    assert not any(f in ['attn_entropy_mean','lm_margin','A_std','ctc_contrast_sentence','lm_confidence'] for f in features)
    for C in [.1,1.]:
     name=f'lr_{group}_C{C:g}'
     model=make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),LogisticRegression(C=C,class_weight=None,solver='lbfgs',max_iter=2000,random_state=42))
     model.fit(tr[features].to_numpy(float),tr[task].to_numpy(int));lr=model.named_steps['logisticregression'];assert lr.n_iter_.max()<2000
     score=model.predict_proba(te[features].to_numpy(float))[:,1];pred[name]=score
     model_path=OUT/'models'/f'{variant}_{task}_{protocol}_{name}.joblib';joblib.dump(dict(pipeline=model,features=features,transform='vanilla_log_probability = ln(lm_confidence)',variant=variant,task=task,protocol=protocol),model_path)
     np.testing.assert_allclose(joblib.load(model_path)['pipeline'].predict_proba(te[features].to_numpy(float))[:,1],score)
     coefs.extend(dict(variant=variant,task=task,protocol=protocol,method=name,feature=f,coefficient=float(v),intercept=float(lr.intercept_[0])) for f,v in zip(features,lr.coef_[0]))
     ev=metric_function(y,score,units);point=ev(np.ones(30,int));np.testing.assert_allclose(point,reference(y,score),atol=1e-12)
     values=np.array([ev(c) for c in counts])
     for i in [0,17]:np.testing.assert_allclose(values[i],reference(y,score,counts[i][units]),atol=1e-12)
     results[name]=(point,values)
     row=dict(variant=variant,task=task,protocol=protocol,method=name,C=C,n_features=len(features),n_train=len(tr),n_test=len(te),n_positive=int(y.sum()),n_speakers=30)
     for j,metric in enumerate(METRICS):
      v=values[:,j];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]);row.update({metric:point[j],metric+'_ci_lower':lo,metric+'_ci_upper':hi,metric+'_n_valid':len(v)})
     summaries.append(row);d=pd.DataFrame(values,columns=METRICS);d['replicate']=plan.replicate;d['variant']=variant;d['task']=task;d['protocol']=protocol;d['method']=name;replicates.append(d)
   comparison=[]
   for a,b in itertools.combinations(results,2):add_pair(comparison,variant,task,a,b,*results[a],*results[b])
   for row in comparison:row['protocol']=protocol
   pairs.extend(comparison);all_results[(variant,task,protocol)]=results
   pred.to_csv(OUT/f'{variant}_{task}_{protocol}_test_scores.csv',index=False)
   print('Completed',variant,task,protocol,flush=True)
 for variant in ['v2','v3']:
  keep=all_results[(variant,'task2','cci_retained')];remove=all_results[(variant,'task2','cci_removed')]
  for method in keep:
   rr=[];add_pair(rr,variant,'task2',method,method,*keep[method],*remove[method])
   for row in rr:row['comparison']='cci_removed_minus_retained'
   protocol_pairs.extend(rr)
 summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False);pd.concat(replicates).to_csv(OUT/'bootstrap_replicates.csv',index=False)
 pd.DataFrame(cohorts).to_csv(OUT/'cohort_counts.csv',index=False);pd.concat(manifests).to_csv(OUT/'cohort_manifest.csv',index=False)
 pd.DataFrame(coefs).to_csv(OUT/'coefficients.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_method_differences.csv',index=False);pd.DataFrame(protocol_pairs).to_csv(OUT/'paired_cci_protocol_differences.csv',index=False)
 meta=dict(signals=dict(vanilla_log_probability='ln(lm_confidence) = original mean token log probability',vanilla_entropy='lm_entropy',ctc_mean_probability='A_ctc',ctc_entropy='audio_entropy',decoding_consistency='decoding_consistency',prompt_aware_contrast='ctc_contrast_word (prompt minus ASR)'),groups=GROUPS,excluded_speakers=[],task2='CCI retained/removed from both train and test; all speakers retained in both variants',task3='Previous original-alignment similarity filter; all groups refit on identical word-contrast-complete training rows and evaluated on identical test rows',LR=dict(penalty='L2',C=[.1,1.],primary_C=.1,class_weight=None,solver='lbfgs',max_iter=2000,seed=42,preprocessing='training-only mean imputation and StandardScaler'),bootstrap=dict(n=2000,seed=42,unit='speaker',aggregation='pooled words; fixed fitted models',ci='95% percentile',plan='Same saved 30-speaker multiplicities for all tasks, variants, protocols and models'),protocol_comparison='CCI removal changes both train/test words; shared-speaker protocol comparison, not identical-word method comparison',missing='No zero fill; task3 complete-case coverage recorded',inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)])
 (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 lines=['# Six-signal LR results','', 'All speakers retained, including fn000117/fn000524/fn000531. Vanilla uses ln(lm_confidence) and lm_entropy; CTC mean=A_ctc; CTC entropy=audio_entropy; consistency=decoding_consistency; contrast=ctc_contrast_word only. Attention, lm_margin, A_std and sentence contrast excluded.','', 'L2 C=.1 primary, C=1 sensitivity; no class weighting; train-only preprocessing. 2000 common speaker bootstrap draws, seed42; fixed models, pooled word metrics, 95% percentile CI. Task2 CCI filtering applies to both splits. Task3 all models use the same filtered, word-contrast-complete cohorts.','']
 for (variant,task,protocol),g in summary.groupby(['variant','task','protocol'],sort=False):
  lines.extend([f'## {variant} {task} {protocol}','','| Model | ROC-AUC [95% CI] | AP positive [95% CI] | PR-AUC positive [95% CI] |','|---|---|---|---|'])
  for _,r in g[g.C.eq(.1)].iterrows():lines.append('| '+r.method+' | '+' | '.join(f'{r[m]:.4f} [{r[m+"_ci_lower"]:.4f}, {r[m+"_ci_upper"]:.4f}]' for m in METRICS)+' |')
  lines.append('')
 lines+=['Full C=1 sensitivity results and all metric CIs are in summary.csv. Paired method comparisons are in paired_method_differences.csv. CCI protocol differences change both training and test populations; see paired_cci_protocol_differences.csv.','']
 (OUT/'README.md').write_text('\n'.join(lines))
 print(summary[summary.C.eq(.1)][['variant','task','protocol','method']+METRICS].to_string(index=False))

if __name__=='__main__':main()
