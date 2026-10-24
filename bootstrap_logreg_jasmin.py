"""Word-row percentile bootstrap (optional speaker clusters) of saved, fixed-model test predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def metric_function(frame, n_bins, sampling_unit='word'):
    """Prepare tie-aware weighted AUC and equal-width ECE; weights preserve sampled unit multiplicity."""
    y = frame.y_true.to_numpy(dtype=float)
    p = frame.y_prob.to_numpy(dtype=float)
    if sampling_unit not in {'word', 'speaker'}:
        raise ValueError('sampling_unit must be word or speaker')
    if sampling_unit == 'speaker':
        _, unit = np.unique(frame.speaker.astype(str), return_inverse=True)
    else:
        unit = np.arange(len(frame))
    _, score = np.unique(p, return_inverse=True)
    bins = np.minimum(np.searchsorted(np.linspace(0, 1, n_bins + 1), p, side='right') - 1, n_bins - 1)

    def evaluate(counts):
        w = counts[unit]
        pos = np.bincount(score, weights=w * y)
        neg = np.bincount(score, weights=w * (1 - y), minlength=len(pos))
        denom = pos.sum() * neg.sum()
        auc = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / denom) if denom else np.nan
        ece = float(np.abs(np.bincount(bins, weights=w * (y - p), minlength=n_bins)).sum() / w.sum())
        return auc, ece
    return evaluate


def bootstrap_predictions(frame, n_bootstrap=2000, seed=42, n_bins=10, sampling_unit='word'):
    if sampling_unit not in {'word', 'speaker'}:
        raise ValueError('sampling_unit must be word or speaker')
    required = ['model_version', 'task', 'y_true', 'y_prob']
    if sampling_unit == 'speaker':
        required.append('speaker')
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Predictions must be nonempty with no missing required values.')
    if n_bootstrap < 2 or n_bins < 1:
        raise ValueError('Require n_bootstrap >= 2 and n_bins >= 1.')
    if not frame.y_true.isin([0, 1]).all() or not frame.y_prob.between(0, 1).all():
        raise ValueError('Require binary labels and finite probabilities in [0, 1].')
    summaries, replicates = [], []
    for (model, task), group in frame.groupby(['model_version', 'task'], sort=True):
        n_speakers = group.speaker.nunique() if 'speaker' in group else np.nan
        n_units = n_speakers if sampling_unit == 'speaker' else len(group)
        if n_units < 2:
            raise ValueError(f'{model}/{task}: at least two sampling units required.')
        evaluate = metric_function(group, n_bins, sampling_unit)
        point = evaluate(np.ones(n_units, dtype=int))
        rng = np.random.default_rng(seed)
        values = np.empty((n_bootstrap, 2))
        for i in range(n_bootstrap):
            counts = np.bincount(rng.integers(n_units, size=n_units), minlength=n_units)
            values[i] = evaluate(counts)
            replicates.append(dict(model_version=model, task=task, replicate=i + 1, sampling_unit=sampling_unit,
                                   auc=values[i, 0], ece=values[i, 1]))
        for j, metric in enumerate(['auc', 'ece']):
            valid = values[:, j][np.isfinite(values[:, j])]
            lo, hi = np.quantile(valid, [0.025, 0.975]) if len(valid) >= 2 else (np.nan, np.nan)
            if len(valid) < n_bootstrap:
                warnings.warn(f'{model}/{task}/{metric}: {len(valid)}/{n_bootstrap} valid replicates')
            summaries.append(dict(model_version=model, task=task, metric=metric,
                                  estimate=point[j], ci_lower=lo, ci_upper=hi,
                                  confidence_level=0.95, ci_method='percentile', sampling_unit=sampling_unit,
                                  n_bootstrap=n_bootstrap, n_valid=len(valid),
                                  n_invalid=n_bootstrap-len(valid), n_speakers=n_speakers,
                                  n_words=len(group), seed=seed, ece_n_bins=n_bins))
    return pd.DataFrame(summaries), pd.DataFrame(replicates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', type=Path, default=[
        ROOT/'notebooks/logreg_task_test_predictions.csv',
        ROOT/'notebooks/logreg_no_ctc_task_test_predictions.csv'])
    parser.add_argument('--output-dir', type=Path, default=ROOT/'notebooks/jasmin_word_bootstrap')
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-bins', type=int, default=10)
    parser.add_argument('--sampling-unit', choices=['word', 'speaker'], default='word')
    args = parser.parse_args()
    frames, sources, seen = [], [], set()
    for path in args.predictions:
        frame = pd.read_csv(path)
        keys = set(map(tuple, frame[['model_version', 'task']].drop_duplicates().to_numpy()))
        if keys & seen:
            raise ValueError('Duplicate model/task groups across input files.')
        seen.update(keys)
        frames.append(frame)
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summary, draws = bootstrap_predictions(pd.concat(frames, ignore_index=True), args.n_bootstrap, args.seed, args.n_bins, args.sampling_unit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir/'summary.csv', index=False)
    draws.to_csv(args.output_dir/'replicates.csv', index=False)
    metadata = dict(inputs=sources, n_bootstrap=args.n_bootstrap, seed=args.seed,
                    ece_n_bins=args.n_bins, confidence_level=0.95, ci_method='percentile', sampling_unit=args.sampling_unit,
                    sampling=('Within each model/task, sample N test word rows with replacement, without class stratification.' if args.sampling_unit == 'word' else 'Sample N speakers with replacement; retain all words and multiplicity.'),
                    estimand='Pooled word-level AUC/ECE conditional on fixed fitted models.',
                    numpy_version=np.__version__, pandas_version=pd.__version__)
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(summary.to_string(index=False))
    print(f'Saved to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
