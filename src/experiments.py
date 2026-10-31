"""Validation comparisons and weighted-SPCA extension; no test evaluation."""

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np

from .classification import Euclidean1NN, classification_metrics
from .partition import DEFAULT_ROOT, DEFAULT_OUTPUT, load_partitions
from .dimensionality_reduction import PCA, SPCA, LDA, PCALDA

PCA_DIMENSIONS = (1, 2, 5, 10, 15, 20, 25, 30, 35, 39, 50, 75, 100, 150, 200, 239)
LDA_DIMENSIONS = (1, 2, 5, 10, 15, 20, 25, 30, 35, 39)
METHODS = ('Unreduced', 'PCA', 'Direct ridge LDA', 'PCA→LDA')
DEFAULT_EXPERIMENT_OUTPUT = Path(__file__).resolve().parents[1]
SPCA_WEIGHTS = (1, 2, 5, 10)


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_digest(model):
    """Check that transformation never mutates learned estimator state."""
    def state(obj):
        return {name: _array_digest(value) if isinstance(value, np.ndarray)
                else state(value) if hasattr(value, '__dict__') else value
                for name, value in sorted(vars(obj).items())}
    return hashlib.sha256(json.dumps(state(model), sort_keys=True).encode()).hexdigest()


def best_configurations(rows):
    """Per-method validation selection: most correct, then smallest final d."""
    return {method: min((r for r in rows if r['method'] == method),
                        key=lambda r: (-r['n_correct'], r['retained_dimension']))
            for method in METHODS}


def evaluate(train, validation):
    """Evaluate the fixed grids using only supplied training/validation data.

    Fits PCA once, sharing it with E3 through its verified fitted-PCA interface.
    No estimator is fitted to validation observations or labels.
    """
    X, y = train['X'], train['y']
    V, truth = validation['X'], validation['y']
    if X.shape != (240, 10304) or V.shape != (80, 10304):
        raise ValueError('Expected the approved ORL 240/80 feature matrices')
    if np.intersect1d(train['indices'], validation['indices']).size:
        raise ValueError('Training and validation indices overlap')
    before = (_array_digest(X), _array_digest(y), _array_digest(V), _array_digest(truth))
    rows, predictions = [], {}

    def record(method, d, m, training_features, validation_features):
        for features, n in ((training_features, len(X)), (validation_features, len(V))):
            if features.shape != (n, d) or not np.isfinite(features).all():
                raise RuntimeError(f'{method}, d={d}: invalid projected shape or values')
        classifier = Euclidean1NN().fit(training_features, y)
        prediction = classifier.predict(validation_features)
        metrics = classification_metrics(truth, prediction)
        rows.append(dict(method=method, retained_dimension=d, intermediate_pca_dimension=m,
                         validation_accuracy=metrics['accuracy'], error_rate=metrics['error_rate'],
                         n_correct=metrics['n_correct'], n_validation=metrics['n_samples']))
        predictions[f'{method}:{d}'] = prediction

    record(METHODS[0], X.shape[1], None, X, V)
    print(f'E0: {rows[-1]["n_correct"]}/80 validation predictions correct')
    pca = PCA(max(PCA_DIMENSIONS)).fit(X)
    direct = LDA(max(LDA_DIMENSIONS), ridge=1e-3).fit(X, y)
    pipeline = PCALDA(200, max(LDA_DIMENSIONS), ridge=0).fit(X, y, fitted_pca=pca)
    for method, model, dimensions, m in (
        (METHODS[1], pca, PCA_DIMENSIONS, None),
        (METHODS[2], direct, LDA_DIMENSIONS, None),
        (METHODS[3], pipeline, LDA_DIMENSIONS, 200),
    ):
        state = _state_digest(model)
        # Transform to the largest requested d once, then take prefix columns.
        # Each dimension is exactly the same fitted ordered projection basis.
        training_features = model.transform(X, max(dimensions))
        validation_features = model.transform(V, max(dimensions))
        for d in dimensions:
            record(method, d, m, training_features[:, :d], validation_features[:, :d])
        if _state_digest(model) != state:
            raise RuntimeError(f'{method}: transform changed fitted parameters')
        print(f'{method}: evaluated {len(dimensions)} dimensions')
    after = (_array_digest(X), _array_digest(y), _array_digest(V), _array_digest(truth))
    if before != after:
        raise RuntimeError('Experiment modified input features/labels')
    diagnostics = dict(
        pca_rank=pca.rank_, direct_within_rank=direct.within_rank_,
        direct_ridge=direct.ridge, direct_lambda=direct.regularization_,
        direct_regularized_condition=direct.regularized_condition_,
        pipeline_pca_dimension=200, pipeline_ridge=0,
        pipeline_within_rank=pipeline.lda_.within_rank_,
        pipeline_within_condition=pipeline.lda_.within_condition_,
        fitted_decompositions=dict(pca=1, direct_lda=1, projected_lda=1),
    )
    return rows, predictions, diagnostics


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    baseline = next(r for r in rows if r['method'] == METHODS[0])['validation_accuracy'] * 100
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        for extended, name in ((False, 'validation_accuracy_dimensions'), (True, 'pca_validation_accuracy_extended')):
            fig, ax = plt.subplots(figsize=(6.8, 3.8), layout='constrained')
            for method, marker, color, label in (
                (METHODS[1], 'o', '#0072B2', 'PCA'),
                (METHODS[2], 's', '#D55E00', 'Direct ridge LDA'),
                (METHODS[3], '^', '#009E73', 'PCA→LDA (m=200)'),
            ):
                if extended and method != METHODS[1]:
                    continue
                selected = [r for r in rows if r['method'] == method and (extended or r['retained_dimension'] <= 39)]
                ax.scatter([r['retained_dimension'] for r in selected],
                           [r['validation_accuracy'] * 100 for r in selected],
                           marker=marker, edgecolors=color, facecolors='none', s=48, linewidths=1.3, label=label)
            ax.axhline(baseline, color='#555555', ls='--', lw=1.1,
                       label=f'Unreduced (10,304D): {baseline:g}%')
            ax.set(xlabel='Retained dimension, d', ylabel='Validation accuracy (%)',
                   ylim=(0, 103), xlim=(-3, 245) if extended else (0, 40.5))
            ax.set_xticks([1, 39, 75, 100, 150, 200, 239] if extended else [1, 5, 10, 15, 20, 25, 30, 35, 39])
            ax.set_yticks(np.arange(0, 101, 20))
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.legend(loc='lower right', frameon=True, fontsize=8.5)
            ax.set_title('ORL validation set (n=80) • Euclidean 1-NN', fontsize=11)
            fig.savefig(output / f'{name}.pdf', dpi=300)
            plt.close(fig)


def run(data_root=DEFAULT_ROOT, split_file=DEFAULT_OUTPUT / 'split_indices.json', output_dir=DEFAULT_EXPERIMENT_OUTPUT):
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the original dataset')
    split_file = Path(split_file)
    split_bytes = split_file.read_bytes()
    parts = load_partitions(root, split_file, names=('train', 'validation'))
    rows, predictions, diagnostics = evaluate(parts['train'], parts['validation'])
    if split_file.read_bytes() != split_bytes:
        raise RuntimeError('Saved partition definition changed')
    results = output / 'results'
    results.mkdir(parents=True, exist_ok=True)
    with (results / 'validation_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '' if value is None else f'{value:.6f}' if isinstance(value, float) else value
                             for key, value in row.items()})
    summary = dict(
        evaluation_partition='validation', n_training=240, n_validation=80,
        split_sha256=hashlib.sha256(split_bytes).hexdigest(),
        seed=json.loads(split_bytes)['seed'],
        training_indices=parts['train']['indices'].tolist(), validation_indices=parts['validation']['indices'].tolist(),
        training_features_sha256=_array_digest(parts['train']['X']),
        validation_features_sha256=_array_digest(parts['validation']['X']),
        dimension_grids=dict(PCA=PCA_DIMENSIONS, LDA=LDA_DIMENSIONS),
        tie_rules=dict(neighbor='first training row for exactly equal squared distances',
                       configuration='highest validation correct count, then smallest retained dimension'),
        best_by_method=best_configurations(rows), diagnostics=diagnostics,
        prediction_sha256={key: _array_digest(value.astype('<i8')) for key, value in predictions.items()},
        verification=dict(same_samples_all_methods=True, fitting_training_only=True,
                          transforms_do_not_mutate_parameters=True, input_arrays_unchanged=True,
                          output_shapes_and_finite_values=True, test_not_classified_or_evaluated=True),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      matplotlib=version('matplotlib'), scikit_learn=version('scikit-learn')),
    )
    (results / 'experiment_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    plot_results(rows, output / 'figures')
    for method, best in summary['best_by_method'].items():
        print(f'{method}: best validation {best["n_correct"]}/80 = '
              f'{best["validation_accuracy"]:.2%}, d={best["retained_dimension"]}')
    return rows, predictions, summary


def evaluate_spca(train, validation, previous_rows, previous_summary):
    """Only the predefined SPCA extension and its required ordinary-PCA control."""
    X, y, V, truth = train['X'], train['y'], validation['X'], validation['y']
    if X.shape != (240, 10304) or V.shape != (80, 10304):
        raise ValueError('Expected the approved ORL 240/80 feature matrices')
    for part, name in ((train, 'training'), (validation, 'validation')):
        if (_array_digest(part['X']) != previous_summary[f'{name}_features_sha256']
                or part['indices'].tolist() != previous_summary[f'{name}_indices']):
            raise ValueError(f'{name} differs from approved main validation observations')
    if np.intersect1d(train['indices'], validation['indices']).size:
        raise ValueError('Training and validation overlap')
    input_state = tuple(_array_digest(a) for a in (X, y, V, truth))
    reference = PCA().fit(X)
    reference_state = _state_digest(reference)
    P, PV = reference.transform(X), reference.transform(V)
    control_predictions = {}
    for d in PCA_DIMENSIONS:
        prediction = Euclidean1NN().fit(P[:, :d], y).predict(PV[:, :d])
        if _array_digest(prediction.astype('<i8')) != previous_summary['prediction_sha256'][f'PCA:{d}']:
            raise RuntimeError(f'PCA control differs from main validation at d={d}')
        control_predictions[d] = prediction
    rows = [dict(method='PCA', supervision_weight=None, retained_dimension=r['retained_dimension'],
                 validation_accuracy=r['validation_accuracy'], error_rate=r['error_rate'],
                 n_correct=r['n_correct'], n_validation=80) for r in previous_rows if r['method'] == 'PCA']
    predictions, ranks, excluded = {}, {}, {}
    for eta in SPCA_WEIGHTS:
        model = SPCA(eta=eta).fit(X, y)
        state = _state_digest(model)
        dimensions = tuple(d for d in PCA_DIMENSIONS if d <= model.rank_)
        if not dimensions:
            raise RuntimeError(f'eta={eta}: no requested dimension is valid')
        ranks[eta] = model.rank_
        excluded[eta] = [d for d in PCA_DIMENSIONS if d > model.rank_]
        Z, ZV = model.transform(X, max(dimensions)), model.transform(V, max(dimensions))
        if Z.shape != (240, max(dimensions)) or ZV.shape != (80, max(dimensions)) or not np.isfinite(Z).all() or not np.isfinite(ZV).all():
            raise RuntimeError('Invalid SPCA feature dimensions/values')
        if eta == 1:
            if model.rank_ != reference.rank_ or not np.allclose(model.eigenvalues_, reference.explained_variance_ * (len(X)-1)/len(X), atol=1e-11, rtol=1e-10):
                raise RuntimeError('eta=1 eigenvalues do not equal total population covariance')
            overlap = reference.components_ @ model.components_.T
        for d in dimensions:
            prediction = Euclidean1NN().fit(Z[:, :d], y).predict(ZV[:, :d])
            if eta == 1:
                rotation = overlap[:d, :d]
                if (not np.allclose(rotation @ rotation.T, np.eye(d), atol=1e-9)
                        or not np.allclose(Z[:, :d], P[:, :d] @ rotation, atol=1e-8)
                        or not np.allclose(ZV[:, :d], PV[:, :d] @ rotation, atol=1e-8)
                        or not np.array_equal(prediction, control_predictions[d])):
                    raise RuntimeError(f'eta=1 PCA subspace/projection/prediction control failed at d={d}')
            if d == reference.rank_ and not np.array_equal(prediction, control_predictions[d]):
                raise RuntimeError('Full-rank orthogonal SPCA should preserve PCA 1-NN predictions')
            metrics = classification_metrics(truth, prediction)
            rows.append(dict(method='SPCA', supervision_weight=eta, retained_dimension=d,
                             validation_accuracy=metrics['accuracy'], error_rate=metrics['error_rate'],
                             n_correct=metrics['n_correct'], n_validation=80))
            predictions[(eta, d)] = prediction
        if _state_digest(model) != state:
            raise RuntimeError('SPCA transformation modified learned parameters')
        best = min((r for r in rows if r['supervision_weight'] == eta), key=lambda r: (-r['n_correct'], r['retained_dimension']))
        print(f'SPCA eta={eta}: rank={model.rank_}; best {best["n_correct"]}/80, d={best["retained_dimension"]}')
    if _state_digest(reference) != reference_state or input_state != tuple(_array_digest(a) for a in (X, y, V, truth)):
        raise RuntimeError('SPCA extension changed reference PCA or input observations')
    selected = min((r for r in rows if r['method'] == 'SPCA'),
                   key=lambda r: (-r['n_correct'], r['retained_dimension'], r['supervision_weight']))
    return dict(rows=rows, predictions=predictions, control_predictions=control_predictions,
                selected=selected, ranks=ranks, excluded_dimensions=excluded)


def plot_spca(analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(7.2, 4), layout='constrained')
        ordinary = [r for r in analysis['rows'] if r['method'] == 'PCA']
        ax.scatter([r['retained_dimension'] for r in ordinary], [r['validation_accuracy']*100 for r in ordinary],
                   marker='o', s=80, facecolors='none', edgecolors='#333333', lw=1.2, label='Ordinary PCA')
        for eta, marker, color, size in ((1, '+', '#777777', 32), (2, 's', '#0072B2', 32),
                                        (5, '^', '#D55E00', 36), (10, 'D', '#009E73', 22)):
            rows = [r for r in analysis['rows'] if r['supervision_weight'] == eta]
            ax.scatter([r['retained_dimension'] for r in rows], [r['validation_accuracy']*100 for r in rows],
                       marker=marker, color=color, s=size, label=f'SPCA η={eta}' + (' (control)' if eta == 1 else ''))
        ax.set(xscale='log', xlabel='Retained dimension, d (log scale)', ylabel='Validation accuracy (%)',
               xlim=(.85, 280), ylim=(0, 104), title='Class-weighted SPCA • ORL validation (n=80) • Euclidean 1-NN')
        ticks = [1, 2, 5, 10, 20, 39, 75, 150, 239]
        ax.set_xticks(ticks, labels=[str(d) for d in ticks])
        ax.minorticks_off()
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output/'spca_validation_accuracy.pdf', dpi=300)
        plt.close(fig)


def run_spca(data_root=DEFAULT_ROOT, split_file=DEFAULT_OUTPUT/'split_indices.json',
             output_dir=DEFAULT_EXPERIMENT_OUTPUT, previous_results=DEFAULT_OUTPUT):
    from .ablations import read_validation_results
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the source dataset')
    split_bytes = Path(split_file).read_bytes()
    prior_rows, prior_summary = read_validation_results(previous_results)
    if hashlib.sha256(split_bytes).hexdigest() != prior_summary['split_sha256']:
        raise ValueError('Saved split differs from approved main validation')
    parts = load_partitions(root, split_file, names=('train', 'validation'))
    analysis = evaluate_spca(parts['train'], parts['validation'], prior_rows, prior_summary)
    if Path(split_file).read_bytes() != split_bytes:
        raise RuntimeError('SPCA extension changed the saved partition')
    results = output/'results'
    results.mkdir(parents=True, exist_ok=True)
    with (results/'spca_validation_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis['rows'][0]))
        writer.writeheader()
        writer.writerows(analysis['rows'])
    plot_spca(analysis, output/'figures')
    print('Selected SPCA:', analysis['selected'])
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--split-file', type=Path, default=DEFAULT_OUTPUT / 'split_indices.json')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_EXPERIMENT_OUTPUT)
    parser.add_argument('--spca', action='store_true', help='Run only the supervised-PCA extension and its PCA control')
    parser.add_argument('--previous-results', type=Path, default=DEFAULT_OUTPUT, help='Prior saved validation results for --spca')
    args = parser.parse_args()
    if args.spca:
        run_spca(args.data_root, args.split_file, args.output_dir, args.previous_results)
    else:
        run(args.data_root, args.split_file, args.output_dir)


if __name__ == '__main__':
    main()
