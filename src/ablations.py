"""Ablation: training PCA variance, m ablation and exploratory paired statistics."""

import argparse
import csv
import hashlib
from itertools import combinations
import json
import numbers
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .classification import Euclidean1NN, classification_metrics
from .partition import DEFAULT_ROOT, DEFAULT_OUTPUT, load_partitions
from .dimensionality_reduction import PCA, LDA, PCALDA
from .experiments import PCA_DIMENSIONS, METHODS, _array_digest, _state_digest, best_configurations

M_VALUES = (40, 50, 60, 80, 100, 120, 140, 160, 180, 190, 195, 200)
OUTPUT = Path(__file__).resolve().parents[1]


def validate_m_values(values, maximum):
    values = tuple(values)
    if not values or any(isinstance(m, bool) or not isinstance(m, numbers.Integral)
                         or not 40 <= m <= maximum for m in values):
        raise ValueError(f'Intermediate dimensions must be integers in [40, {maximum}] for d=39')
    if tuple(sorted(set(values))) != values:
        raise ValueError('Intermediate dimensions must be unique and increasing')
    return values


def variance_thresholds(pca, thresholds=(.9, .95, .99)):
    result = {}
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError('Variance thresholds must be in (0, 1]')
        indices = np.flatnonzero(pca.cumulative_explained_variance_ >= threshold)
        if not len(indices):
            raise ValueError(f'Variance threshold {threshold} not attained at numerical rank')
        result[threshold] = int(indices[0] + 1)
    return result


def paired_comparison(truth, prediction_a, prediction_b):
    """B versus A: exact two-sided McNemar conditional on discordant counts."""
    # Reuse the classifier's validation of aligned nonempty label vectors.
    classification_metrics(truth, prediction_a)
    classification_metrics(truth, prediction_b)
    a, b = np.asarray(prediction_a) == truth, np.asarray(prediction_b) == truth
    corrected, new_errors = np.flatnonzero(~a & b), np.flatnonzero(a & ~b)
    discordant = len(corrected) + len(new_errors)
    pvalue = float(binomtest(len(corrected), discordant, .5).pvalue) if discordant else 1.0
    return dict(both_correct=int(np.count_nonzero(a & b)), both_wrong=int(np.count_nonzero(~a & ~b)),
                corrected=corrected.tolist(), new_errors=new_errors.tolist(), pvalue=pvalue)


def read_validation_results(results_dir):
    with (Path(results_dir) / 'validation_results.csv').open(encoding='utf-8', newline='') as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(dict(method=row['method'], retained_dimension=int(row['retained_dimension']),
                             intermediate_pca_dimension=int(row['intermediate_pca_dimension']) if row['intermediate_pca_dimension'] else None,
                             n_correct=int(row['n_correct']), n_validation=int(row['n_validation']),
                             validation_accuracy=float(row['validation_accuracy']), error_rate=float(row['error_rate'])))
    if len(rows) != 37 or any(r['n_validation'] != 80 or not 0 <= r['n_correct'] <= 80
                             or abs(r['validation_accuracy'] - r['n_correct']/80) > 1e-10
                             or abs(r['error_rate'] - (80-r['n_correct'])/80) > 1e-10 for r in rows):
        raise ValueError('Invalid main validation result counts/metrics')
    if tuple(r['retained_dimension'] for r in rows if r['method'] == 'PCA') != PCA_DIMENSIONS:
        raise ValueError('main validation PCA dimension grid differs from the approved grid')
    summary = json.loads((Path(results_dir) / 'experiment_summary.json').read_text(encoding='utf-8'))
    if best_configurations(rows) != summary['best_by_method']:
        raise ValueError('main validation results disagree with saved selection summary')
    return rows, summary


def analyse(train, validation, prior_rows, prior_summary, m_values=M_VALUES):
    """Reuse main validation accuracies; refit only training estimators needed here."""
    X, y, V, truth = train['X'], train['y'], validation['X'], validation['y']
    if X.shape != (240, 10304) or V.shape != (80, 10304):
        raise ValueError('Expected the approved ORL 240/80 feature matrices')
    for part, key in ((train, 'training'), (validation, 'validation')):
        if _array_digest(part['X']) != prior_summary[f'{key}_features_sha256'] or part['indices'].tolist() != prior_summary[f'{key}_indices']:
            raise ValueError(f'{key} differs from main validation samples/features')
    if np.intersect1d(train['indices'], validation['indices']).size:
        raise ValueError('Training and validation overlap')
    before = tuple(_array_digest(a) for a in (X, y, V, truth))
    pca = PCA().fit(X)
    m_values = validate_m_values(m_values, min(pca.rank_, len(y) - len(np.unique(y))))
    pca_state = _state_digest(pca)
    variance_rows = [dict(retained_dimension=r['retained_dimension'],
                          cumulative_explained_variance=float(pca.cumulative_explained_variance_[r['retained_dimension'] - 1]),
                          validation_accuracy=r['validation_accuracy'], n_correct=r['n_correct'], n_validation=80)
                     for r in prior_rows if r['method'] == 'PCA']
    thresholds = variance_thresholds(pca)
    ablation_rows, ablation_predictions = [], {}
    for m in m_values:
        model = PCALDA(m, 39, ridge=0).fit(X, y, fitted_pca=pca)
        state = _state_digest(model)
        Z, ZV = model.transform(X), model.transform(V)
        if Z.shape != (240, 39) or ZV.shape != (80, 39) or not np.isfinite(Z).all() or not np.isfinite(ZV).all():
            raise RuntimeError(f'm={m}: invalid projection')
        prediction = Euclidean1NN().fit(Z, y).predict(ZV)
        if _state_digest(model) != state:
            raise RuntimeError('Transform modified fitted parameters')
        metrics = classification_metrics(truth, prediction)
        ablation_rows.append(dict(intermediate_pca_dimension=m, retained_dimension=39,
                                  validation_accuracy=metrics['accuracy'], error_rate=metrics['error_rate'],
                                  n_correct=metrics['n_correct'], n_validation=80,
                                  within_rank=model.lda_.within_rank_, within_condition=model.lda_.within_condition_))
        ablation_predictions[m] = prediction
        if not np.isfinite(model.lda_.within_condition_) or model.lda_.within_rank_ != m:
            raise RuntimeError(f'm={m}: within scatter is not full rank/finite')
        print(f'm={m:3d}, d=39: {metrics["n_correct"]}/80 = {metrics["accuracy"]:.2%}, '
              f'within condition={model.lda_.within_condition_:.6g}')
    selected = min(ablation_rows, key=lambda r: (-r['n_correct'], r['intermediate_pca_dimension']))
    best = best_configurations(prior_rows)
    selected_predictions = {}
    for method in METHODS[:3]:
        d = best[method]['retained_dimension']
        if method == 'Unreduced':
            Z, ZV = X, V
        else:
            model = pca if method == 'PCA' else LDA(d, ridge=1e-3).fit(X, y)
            state = _state_digest(model)
            Z, ZV = model.transform(X, d), model.transform(V, d)
            if _state_digest(model) != state:
                raise RuntimeError('Selected transform modified fitted parameters')
        prediction = Euclidean1NN().fit(Z, y).predict(ZV)
        if _array_digest(prediction.astype('<i8')) != prior_summary['prediction_sha256'][f'{method}:{d}']:
            raise RuntimeError(f'{method}: selected predictions differ from main validation')
        selected_predictions[method] = prediction
    selected_predictions['PCA→LDA'] = ablation_predictions[selected['intermediate_pca_dimension']]
    if 200 in ablation_predictions and _array_digest(ablation_predictions[200].astype('<i8')) != prior_summary['prediction_sha256']['PCA→LDA:39']:
        raise RuntimeError('m=200,d=39 predictions differ from main validation')
    if _state_digest(pca) != pca_state or before != tuple(_array_digest(a) for a in (X, y, V, truth)):
        raise RuntimeError('Analysis modified fitted PCA or input observations')
    statistics = {}
    for name, prediction in selected_predictions.items():
        metrics = classification_metrics(truth, prediction)
        ci = binomtest(metrics['n_correct'], len(truth)).proportion_ci(method='wilson')
        statistics[name] = dict(**metrics, n_incorrect=len(truth)-metrics['n_correct'], ci_low=float(ci.low), ci_high=float(ci.high))
    pairs = {(a, b): paired_comparison(truth, selected_predictions[a], selected_predictions[b])
             for a, b in combinations(METHODS, 2)}
    return dict(variance=variance_rows, thresholds=thresholds, ablation=ablation_rows,
                selected=selected, predictions=selected_predictions, ablation_predictions=ablation_predictions,
                statistics=statistics, pairs=pairs)


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(analysis, baseline_accuracy, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}
    with plt.rc_context(style):
        rows = analysis['variance']
        dimensions = [r['retained_dimension'] for r in rows]
        variance = np.asarray([r['cumulative_explained_variance'] for r in rows]) * 100
        accuracy = np.asarray([r['validation_accuracy'] for r in rows]) * 100
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout='constrained')
        axes[0].scatter(dimensions, variance, color='#0072B2', s=25)
        axes[0].set(xlabel='Retained PCA dimension, d', ylabel='Training cumulative variance (%)', title='(a) Retained training variance', xlim=(0, 245), ylim=(0, 103))
        axes[0].set_xticks([1, 50, 100, 150, 200, 239])
        for level, d in analysis['thresholds'].items():
            axes[0].axhline(level*100, color='#888888', ls=':', lw=.7)
        threshold_text = '\n'.join(f'{level:.0%}: d={d}' for level, d in analysis['thresholds'].items())
        axes[0].text(.53, .17, threshold_text, transform=axes[0].transAxes, fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        axes[1].scatter(variance, accuracy, facecolors='none', edgecolors='#D55E00', s=30)
        axes[1].axhline(baseline_accuracy*100, color='#555555', ls='--', lw=1, label='Unreduced baseline')
        for d, label, offset in ((1, 'd=1', (5, 4)), (50, 'd=50', (-27, -17)), (239, 'd=239', (-38, -20))):
            row = next(r for r in rows if r['retained_dimension'] == d)
            axes[1].annotate(label, (row['cumulative_explained_variance']*100, row['validation_accuracy']*100),
                             xytext=offset, textcoords='offset points', fontsize=8,
                             arrowprops=dict(arrowstyle='-', color='#777777', lw=.6) if d != 1 else None)
        axes[1].set(xlabel='Training cumulative variance (%)', ylabel='Validation accuracy (%)', title='(b) Variance versus identification', ylim=(0, 105), xlim=(0, 105))
        axes[1].legend(loc='lower right', fontsize=7.5)
        for ax in axes:
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
        fig.savefig(output/'pca_variance_accuracy.pdf', dpi=300)
        plt.close(fig)

        rows = analysis['ablation']
        m = [r['intermediate_pca_dimension'] for r in rows]
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.5), sharex=True, layout='constrained')
        axes[0].scatter(m, [r['validation_accuracy']*100 for r in rows], color='#0072B2', s=30, label='PCA→LDA, d=39, ridge=0')
        axes[0].axhline(baseline_accuracy*100, color='#555555', ls='--', lw=1, label='Unreduced baseline')
        axes[0].set(ylabel='Validation accuracy (%)', ylim=(0, 104), title='Intermediate PCA dimension • fixed final d=39')
        axes[0].legend(loc='lower left', fontsize=8)
        axes[1].scatter(m, [r['within_condition'] for r in rows], color='#D55E00', marker='s', s=28)
        axes[1].set(xlabel='Intermediate PCA dimension, m', ylabel='Within-scatter condition\n(log scale)', yscale='log', xlim=(35, 205))
        axes[1].set_xticks([40, 60, 80, 100, 120, 140, 160, 180, 200])
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        fig.savefig(output/'pcalda_intermediate_ablation.pdf', dpi=300)
        plt.close(fig)
def run(data_root=DEFAULT_ROOT, split_file=DEFAULT_OUTPUT/'split_indices.json', previous_results=DEFAULT_OUTPUT, output_dir=OUTPUT):
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the original dataset')
    split_bytes = Path(split_file).read_bytes()
    rows, summary = read_validation_results(previous_results)
    if hashlib.sha256(split_bytes).hexdigest() != summary['split_sha256']:
        raise ValueError('Saved split differs from main validation')
    parts = load_partitions(root, split_file, names=('train', 'validation'))
    analysis = analyse(parts['train'], parts['validation'], rows, summary)
    if Path(split_file).read_bytes() != split_bytes:
        raise RuntimeError('Saved split changed during analysis')
    results = output/'results'
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results/'pca_variance_accuracy.csv', analysis['variance'])
    _write_csv(results/'pcalda_intermediate_ablation.csv', analysis['ablation'])
    prediction_rows = [dict(index=int(index), path=str(path), true_label=int(label),
                            **{name: int(prediction[i]) for name, prediction in analysis['predictions'].items()})
                       for i, (index, path, label) in enumerate(zip(parts['validation']['indices'], parts['validation']['paths'], parts['validation']['y']))]
    _write_csv(results/'selected_validation_predictions.csv', prediction_rows)
    plot_analysis(analysis, next(r['validation_accuracy'] for r in rows if r['method'] == 'Unreduced'), output/'figures')
    print('Variance thresholds:', analysis['thresholds'])
    print('Selected PCA→LDA:', analysis['selected'])
    print('Statistical assessment:', analysis['statistics'])
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--split-file', type=Path, default=DEFAULT_OUTPUT/'split_indices.json')
    parser.add_argument('--previous-results', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.data_root, args.split_file, args.previous_results, args.output_dir)


if __name__ == '__main__':
    main()
