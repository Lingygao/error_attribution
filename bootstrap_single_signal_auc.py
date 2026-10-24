"""Filtered Jasmin single-signal pooled ROC AUC with speaker-cluster bootstrap."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
# Fixed semantic directions; never optimized per speaker or bootstrap replicate.
DIRECTIONS = {'attn_entropy_mean': 1, 'A_ctc': -1, 'A_std': 1,
              'audio_entropy': 1, 'decoding_consistency': -1,
              'lm_confidence': -1, 'lm_entropy': 1, 'lm_margin': -1}


def weighted_auc_function(y, score, speaker):
    _, unit = np.unique(speaker, return_inverse=True)
    _, ranks = np.unique(score, return_inverse=True)
    y = np.asarray(y, dtype=int)
    def evaluate(counts):
        weights = counts[unit]
        pos = np.bincount(ranks, weights=weights*y)
        neg = np.bincount(ranks, weights=weights*(1-y), minlength=len(pos))
        denominator = pos.sum()*neg.sum()
        return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/denominator) if denominator else np.nan
    return evaluate


def load_filtered_test():
    notebook = ROOT/'notebooks/auc_ece.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    ns = dict(Path=Path, np=np, pd=pd, json=json, re=re)
    exec(''.join(cells[3]['source']), ns)
    # Reuse definitions only; do not run notebook training or output side effects.
    for index in [5, 9, 11]:
        tree = ast.parse(''.join(cells[index]['source']))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), 'exec'), ns)
    ns['NONE_STRINGS'] = {'', 'none', 'nan', 'na', 'n/a', 'null'}
    split = ROOT/'notebooks/test.txt'
    frames, paths = [], [notebook, split]
    for speaker in ns['read_speaker_list'](split):
        path = ns['find_speaker_file'](ns['ALL_SPEAKER_ROOT'], speaker)
        if path is None:
            raise FileNotFoundError(speaker)
        frame = pd.read_csv(path)
        if not frame.speaker.eq(speaker).all():
            raise ValueError(f'Speaker mismatch: {path}')
        frame['source_file'] = str(path)
        frames.append(frame)
        paths.append(path)
    frame = pd.concat(frames, ignore_index=True)
    lexicon_path = ROOT/'notebooks'/ns['LEXICON_PATH']
    lexicon = ns['load_lexicon'](lexicon_path)
    if lexicon_path.exists():
        paths.append(lexicon_path)
    frame, _ = ns['apply_task3_phonemic_filter'](frame, 'test', lexicon)
    reference_path = ROOT/'notebooks/logreg_task_test_predictions.csv'
    old = pd.read_csv(reference_path)
    paths.append(reference_path)
    for task in ns['TASK_COLS']:
        frame[task] = ns['normalize_task_label'](frame[task])
        current = frame[frame[task].notna()]
        ref = old[old.model_version.eq('conservative_C0.1_unweighted') & old.task.eq(task)]
        for col in ['speaker', 'asr_word', 'source_file']:
            if current[col].astype(str).tolist() != ref[col].astype(str).tolist():
                raise ValueError(f'Filtered test population mismatch: {task}/{col}')
        np.testing.assert_array_equal(current[task].astype(int), ref.y_true)
    return frame, paths, dict(task3_filter_tau=ns['TASK3_FILTER_TAU'],
                              lexicon_path=str(lexicon_path), lexicon_exists=lexicon_path.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'notebooks/jasmin_single_signal_speaker_bootstrap')
    args = parser.parse_args()
    if args.n_bootstrap < 2:
        raise ValueError('Require at least two bootstrap repetitions.')
    frame, paths, filter_info = load_filtered_test()
    summaries, draws, diagnostics = [], [], []
    for task in ['task1', 'task2', 'task3']:
        population = frame[frame[task].notna()]
        for feature, sign in DIRECTIONS.items():
            score = pd.to_numeric(population[feature], errors='coerce')
            mask = np.isfinite(score.to_numpy(dtype=float))
            group = population.loc[mask]
            y = group[task].astype(int).to_numpy()
            x = score.loc[mask].to_numpy(dtype=float)
            speakers = group.speaker.astype(str).to_numpy()
            k = len(np.unique(speakers))
            if k < 2:
                raise ValueError(f'{task}/{feature}: fewer than two speakers with valid scores.')
            for speaker, words in group.groupby('speaker'):
                diagnostics.append(dict(task=task, feature=feature, speaker=speaker,
                                        n_words=len(words), n_positive=int(words[task].sum()),
                                        single_class=words[task].nunique()==1))
            evaluate = weighted_auc_function(y, x, speakers)
            point = evaluate(np.ones(k, dtype=int))
            if len(np.unique(y))==2:
                np.testing.assert_allclose(point, roc_auc_score(y, x))
            rng = np.random.default_rng(args.seed)
            values = np.array([evaluate(np.bincount(rng.integers(k, size=k), minlength=k))
                               for _ in range(args.n_bootstrap)])
            aligned = values if sign==1 else 1-values
            valid = values[np.isfinite(values)]
            lo, hi = np.quantile(valid, [.025,.975]) if len(valid)>=2 else (np.nan,np.nan)
            oriented_lo, oriented_hi = (lo,hi) if sign==1 else (1-hi,1-lo)
            summaries.append(dict(task=task, feature=feature, score_multiplier=sign,
                                  raw_auc=point, raw_ci_lower=lo, raw_ci_upper=hi,
                                  oriented_auc=point if sign==1 else 1-point,
                                  oriented_ci_lower=oriented_lo, oriented_ci_upper=oriented_hi,
                                  n_words=len(group), n_missing_scores=int((~mask).sum()),
                                  n_positive=int(y.sum()), n_negative=int((1-y).sum()),
                                  n_speakers=k, n_speakers_no_valid_score=population.speaker.nunique()-k,
                                  n_single_class_speakers=int(group.groupby('speaker')[task].nunique().eq(1).sum()),
                                  n_bootstrap=args.n_bootstrap, n_valid=len(valid),
                                  n_invalid=args.n_bootstrap-len(valid), seed=args.seed,
                                  sampling_unit='speaker', aggregation='pooled_word_roc_auc',
                                  confidence_level=.95, ci_method='percentile'))
            for i in range(args.n_bootstrap):
                draws.append(dict(task=task, feature=feature, replicate=i+1,
                                  raw_auc=values[i], oriented_auc=aligned[i]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir/'summary.csv', index=False)
    pd.DataFrame(draws).to_csv(args.output_dir/'replicates.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir/'speaker_counts.csv', index=False)
    metadata = dict(sampling_unit='speaker', aggregation='pooled_word_roc_auc',
                    n_bootstrap=args.n_bootstrap, seed=args.seed, directions=DIRECTIONS,
                    filter=filter_info, single_class_speakers='Retained; only a single-class pooled replicate has undefined AUC.',
                    missing_scores='Drop nonfinite scores per signal; speakers with no valid scores excluded.',
                    confidence_level=.95, ci_method='percentile',
                    inputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
