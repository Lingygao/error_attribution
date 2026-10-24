"""Original-alignment Jasmin evaluation: CCI-only removal, all speakers retained."""
import ast
import hashlib
import itertools
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from bootstrap_single_signal_auc import ROOT, DIRECTIONS

OUT = ROOT/'notebooks/jasmin_cci_only_v2_v3'
METRICS = ['roc_auc', 'ap_positive', 'pr_auc_positive']
FEATURES = list(DIRECTIONS)
INTERNAL = [f for f in FEATURES if f not in ['A_ctc','A_std','audio_entropy','decoding_consistency']]
GROUPS = {'internal': INTERNAL,
          'internal_plus_ctc': [f for f in FEATURES if f!='decoding_consistency'],
          'internal_plus_decoding_consistency': [f for f in FEATURES if f not in ['A_ctc','A_std','audio_entropy']],
          'all': FEATURES}
SEED, N_BOOTSTRAP = 42, 2000


def metric_function(y, score, units):
    """Tie-aware pooled metrics with speaker multiplicities, high score => y=1."""
    y = np.asarray(y, int)
    _, rank = np.unique(-np.asarray(score), return_inverse=True)
    def evaluate(counts):
        w = counts[units]
        p = np.bincount(rank, weights=w*y)
        n = np.bincount(rank, weights=w*(1-y), minlength=len(p))
        present = (p+n)>0
        p,n = p[present],n[present]
        P,N = p.sum(),n.sum()
        if P==0 or N==0:
            return np.full(3,np.nan)
        tp,fp = np.cumsum(p),np.cumsum(n)
        precision = tp/(tp+fp)
        dr = p/P
        return np.array([np.sum(p*(N-fp+.5*n))/(P*N),
                         np.sum(dr*precision),
                         np.sum(dr*(precision+np.r_[1.,precision[:-1]])/2)])
    return evaluate


def reference(y,s,w=None):
    if w is not None:
        # This environment has an old sklearn that retains zero-weight thresholds.
        # Remove absent rows to match literal repeated-speaker resampling.
        keep=np.asarray(w)>0
        y,s,w=np.asarray(y)[keep],np.asarray(s)[keep],np.asarray(w)[keep]
    precision,recall,_ = precision_recall_curve(y,s,sample_weight=w)
    return np.array([roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w),auc(recall,precision)])


def load_data(variant,split,ns,lexicon,paths):
    speakers = (ROOT/f'notebooks/{split}.txt').read_text().split()
    frames=[]
    for speaker in speakers:
        path=ROOT/f'data/jasmin/output_final/{variant}_segment/{speaker}/{speaker}_pr_mt_{variant}_task_label_table_with_signals.csv'
        paths.add(path)
        f=pd.read_csv(path)
        assert f.speaker.eq(speaker).all()
        f['source_file']=str(path);f['source_csv_row']=np.arange(len(f))+2
        frames.append(f)
    f=pd.concat(frames,ignore_index=True)
    assert not f.duplicated(['speaker','asr_index']).any()
    assert np.isfinite(f[FEATURES].to_numpy(float)).all(), 'Nonfinite features: do not silently change paired cohorts'
    all_c=f.pr_mt_labels.fillna('').map(lambda x: bool(str(x).strip()) and all(t.strip().upper()=='C' for t in str(x).split('|')))
    f['cci_excluded']=f.mt_v_label.eq('C') & f.pr_v_label.eq('I') & all_c
    f['task2_original']=f.task2
    f['task3_original']=f.task3
    f.loc[f.cci_excluded,'task2']=np.nan
    f,_=ns['apply_task3_phonemic_filter'](f,f'{variant}/{split}',lexicon)
    f['variant']=variant;f['split']=split
    return f


def main():
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'models').mkdir(exist_ok=True)
    nb=ROOT/'notebooks/auc_ece.ipynb';cells=json.loads(nb.read_text())['cells']
    ns=dict(Path=Path,np=np,pd=pd,json=json,re=re)
    exec(''.join(cells[3]['source']),ns)
    nodes=[n for n in ast.parse(''.join(cells[9]['source'])).body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(nb),'exec'),ns)
    lp=ROOT/'notebooks'/ns['LEXICON_PATH'];lexicon=ns['load_lexicon'](lp)
    paths={nb,Path(__file__).resolve(),ROOT/'notebooks/train.txt',ROOT/'notebooks/test.txt'}
    if lp.exists():paths.add(lp)
    speakers=sorted((ROOT/'notebooks/test.txt').read_text().split())
    train_speakers=(ROOT/'notebooks/train.txt').read_text().split()
    assert not set(speakers)&set(train_speakers)
    assert len(speakers)==len(set(speakers)) and 'fn000117' in speakers
    k=len(speakers);unit_map={s:i for i,s in enumerate(speakers)}
    rng=np.random.default_rng(SEED)
    counts=np.array([np.bincount(rng.integers(k,size=k),minlength=k) for _ in range(N_BOOTSTRAP)])
    plan=pd.DataFrame(counts,columns=speakers);plan.insert(0,'replicate',np.arange(1,N_BOOTSTRAP+1));plan.to_csv(OUT/'bootstrap_speaker_counts.csv',index=False)
    summaries=[];replicates=[];pairs=[];cohorts=[];manifests=[];all_results={};coefficients=[]
    for variant in ['v2','v3']:
        data={s:load_data(variant,s,ns,lexicon,paths) for s in ['train','test']}
        for split,f in data.items():
            manifests.append(f[['variant','split','speaker','asr_index','asr_word','source_file','source_csv_row','task2_original','task2','task3_original','task3','cci_excluded','task3_phonemic_filter_excluded','task3_phonemic_similarity']])
        for task in ['task2','task3']:
            tr=data['train'].loc[data['train'][task].notna()].copy()
            te=data['test'].loc[data['test'][task].notna()].copy()
            for split,f in [('train',tr),('test',te)]:
                assert f[task].isin([0,1]).all()
                assert set(f.speaker)==set(train_speakers if split=='train' else speakers)
                cohorts.append(dict(variant=variant,task=task,split=split,n_words=len(f),n_positive=int(f[task].sum()),n_negative=int((f[task]==0).sum()),n_speakers=f.speaker.nunique()))
            y=te[task].to_numpy(int);units=te.speaker.map(unit_map).to_numpy(int)
            scores={f'signal_{f}':te[f].to_numpy(float)*sign for f,sign in DIRECTIONS.items()}
            for group,features in GROUPS.items():
                for C in [.1,1.]:
                    name=f'lr_{group}_C{C:g}'
                    model=make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),LogisticRegression(C=C,class_weight=None,solver='lbfgs',max_iter=2000,random_state=SEED))
                    model.fit(tr[features].to_numpy(float),tr[task].to_numpy(int))
                    lr=model.named_steps['logisticregression'];assert lr.n_iter_.max()<2000
                    scores[name]=model.predict_proba(te[features].to_numpy(float))[:,1]
                    path=OUT/'models'/f'{variant}_{task}_{name}.joblib'
                    joblib.dump(dict(pipeline=model,features=features,variant=variant,task=task),path)
                    np.testing.assert_allclose(joblib.load(path)['pipeline'].predict_proba(te[features].to_numpy(float))[:,1],scores[name])
                    coefficients.extend(dict(variant=variant,task=task,method=name,feature=f,coefficient=float(coef),intercept=float(lr.intercept_[0])) for f,coef in zip(features,lr.coef_[0]))
            prediction=te[['speaker','asr_index','asr_word','source_file','source_csv_row']].copy();prediction['y_true']=y
            results={}
            for method,score in scores.items():
                prediction[method]=score
                evaluate=metric_function(y,score,units);point=evaluate(np.ones(k,int))
                np.testing.assert_allclose(point,reference(y,score),atol=1e-12)
                values=np.array([evaluate(c) for c in counts])
                # Check real repeated-speaker samples against sklearn weighted metrics.
                for i in [0,1,17]:
                    if np.isfinite(values[i]).all():np.testing.assert_allclose(values[i],reference(y,score,counts[i][units]),atol=1e-12)
                results[method]=(point,values)
                row=dict(variant=variant,task=task,method=method,n_train=len(tr),n_test=len(te),n_positive=int(y.sum()),n_negative=int((1-y).sum()),n_speakers=k)
                for j,metric in enumerate(METRICS):
                    valid=values[:,j][np.isfinite(values[:,j])];lo,hi=np.quantile(valid,[.025,.975])
                    row.update({metric:point[j],metric+'_ci_lower':lo,metric+'_ci_upper':hi,metric+'_n_valid':len(valid)})
                summaries.append(row)
                rep=pd.DataFrame(values,columns=METRICS);rep.insert(0,'replicate',np.arange(1,N_BOOTSTRAP+1));rep['variant']=variant;rep['task']=task;rep['method']=method;replicates.append(rep)
            prediction.to_csv(OUT/f'{variant}_{task}_test_scores.csv',index=False)
            all_results[(variant,task)]=results
            for baseline,candidate in itertools.combinations(results,2):
                add_pair(pairs,variant,task,baseline,candidate,*results[baseline],*results[candidate])
            print(f'Completed {variant} {task}: train={len(tr)}, test={len(te)}, methods={len(scores)}',flush=True)
    for task in ['task2','task3']:
        for method in all_results[('v2',task)]:
            add_pair(pairs,'v3_minus_v2',task,method,method,*all_results[('v2',task)][method],*all_results[('v3',task)][method])
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    pd.concat(replicates,ignore_index=True).to_csv(OUT/'bootstrap_replicates.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    pd.DataFrame(cohorts).to_csv(OUT/'cohort_counts.csv',index=False)
    manifest=pd.concat(manifests,ignore_index=True);manifest.to_csv(OUT/'filter_manifest.csv',index=False)
    manifest[manifest.cci_excluded | manifest.task3_phonemic_filter_excluded].to_csv(OUT/'excluded_cases.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    metadata=dict(seed=SEED,n_bootstrap=N_BOOTSTRAP,ci='95% percentile',sampling='same 30-speaker multinomial plan across all versions/tasks/methods; pooled word metrics; fixed trained models',
                  excluded_speakers=[],task2_filter='Original alignment: remove nonempty all-C PR-MT AND MT-ASR C AND PR-ASR I from train and test',
                  task3_filter=dict(tau=ns['TASK3_FILTER_TAU'],lexicon_path=str(lp),lexicon_exists=lp.exists(),fallback='character similarity'),
                  feature_groups=GROUPS,single_signal_multipliers=DIRECTIONS,LR=dict(penalty='L2',C=[.1,1.],class_weight=None,solver='lbfgs',max_iter=2000,imputation='train mean',scaling='train StandardScaler'),
                  positive_class='task label 1',invalid_draw='All three metrics NaN if resampled pooled data contains only one class; preserve replicate ID',
                  cross_version_pairing='Shared speakers, but ASR words and labels can differ; not a matched-word comparison',
                  limitation='CCI exclusion selected following previous test diagnostics; post-hoc analysis',
                  numpy_version=np.__version__,sklearn_version=sklearn.__version__,
                  bootstrap_plan_sha256=hashlib.sha256((OUT/'bootstrap_speaker_counts.csv').read_bytes()).hexdigest(),
                  inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)])
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Jasmin CCI-only evaluation','', 'Original alignment; fn000117 retained. Task2 removes C/C/I in training and test. Task3 retains the previous similarity <0.2 filter (character fallback when the configured lexicon is absent).', '', 'All intervals are 95% percentile intervals from 2000 shared speaker-cluster bootstrap draws, seed 42. Models remain fixed; metrics pool all sampled words, not macro speaker AUC. Positive means task label 1. Single-signal directions are fixed, not optimized on test data. LR uses L2 C=0.1 (primary), C=1 (sensitivity), no class weights, train-only standardization.', '', 'bootstrap_speaker_counts.csv is the common sampling plan. bootstrap_replicates.csv can be joined on replicate; subtract candidate minus baseline within each replicate. paired_differences.csv already contains all within-version method comparisons and v3 minus v2 comparisons. Cross-version comparisons share speakers but have different word/label populations.','']
    for (v,t),f in summary.groupby(['variant','task']):
        lines.extend([f'## {v} {t}', '', '| Method | ROC-AUC [95% CI] | AP positive [95% CI] | PR-AUC positive [95% CI] |','|---|---|---|---|'])
        for _,r in f.iterrows():
            lines.append('| '+r['method']+' | '+' | '.join(f'{r[m]:.4f} [{r[m+"_ci_lower"]:.4f}, {r[m+"_ci_upper"]:.4f}]' for m in METRICS)+' |')
        lines.append('')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(cohorts).to_string(index=False))


def add_pair(rows,variant,task,baseline,candidate,base_point,base_values,point,values):
    for j,metric in enumerate(METRICS):
        d=values[:,j]-base_values[:,j];valid=d[np.isfinite(d)];lo,hi=np.quantile(valid,[.025,.975])
        rows.append(dict(variant=variant,task=task,baseline=baseline,candidate=candidate,metric=metric,delta=point[j]-base_point[j],ci_lower=lo,ci_upper=hi,n_valid=len(valid)))

if __name__=='__main__':main()
