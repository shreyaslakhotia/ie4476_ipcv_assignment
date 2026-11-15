"""Update the dimension comparison with the already selected PCA→LDA m=40.

Run: python -B -m src.selected_validation_figure
Original m=200 experiments and all final-test artifacts remain unchanged.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from .ablations import read_validation_results
from .classification import Euclidean1NN, classification_metrics
from .dimensionality_reduction import PCALDA
from .error_analysis import load_review_partitions, read_csv
from .experiments import LDA_DIMENSIONS, METHODS, _array_digest, _state_digest
from .partition import DEFAULT_ROOT, DEFAULT_OUTPUT


def run(root=DEFAULT_ROOT, results=DEFAULT_OUTPUT, preview=None):
    results = Path(results)
    previous, metadata = read_validation_results(results)
    parts = load_review_partitions(root, results/'split_indices.json', metadata)
    train, validation = parts['train'], parts['validation']
    model = PCALDA(40, 39, ridge=0).fit(train['X'], train['y'])
    state = _state_digest(model)
    Z, V = model.transform(train['X']), model.transform(validation['X'])
    rows = []
    stored = read_csv(results/'selected_validation_predictions.csv')
    if [r['path'] for r in stored] != validation['paths'].tolist():
        raise ValueError('Saved validation prediction order differs')
    for d in LDA_DIMENSIONS:
        prediction = Euclidean1NN().fit(Z[:, :d], train['y']).predict(V[:, :d])
        metrics = classification_metrics(validation['y'], prediction)
        rows.append(dict(method=METHODS[3], retained_dimension=d, intermediate_pca_dimension=40,
                         validation_accuracy=metrics['accuracy'], error_rate=metrics['error_rate'],
                         n_correct=metrics['n_correct'], n_validation=80,
                         prediction_sha256=_array_digest(prediction.astype('<i8'))))
        if d == 39:
            expected = next(r for r in read_csv(results/'pcalda_intermediate_ablation.csv')
                            if int(r['intermediate_pca_dimension']) == 40)
            if metrics['n_correct'] != int(expected['n_correct']):
                raise ValueError('Selected endpoint differs from saved ablation')
            if not np.array_equal(prediction, [int(r[METHODS[3]]) for r in stored]):
                raise ValueError('Selected endpoint predictions differ from saved vector')
    if _state_digest(model) != state:
        raise RuntimeError('Transform changed fitted parameters')
    csv_path = results/'pcalda_m40_validation_dimensions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_rows = [r for r in previous if r['method'] != METHODS[3]] + rows
    output = results.parent/'figures'/'validation_accuracy_dimensions.pdf'
    plot(plot_rows, output, preview)
    print(f'Updated {output}; m=40, d=39 reproduces 78/80 and saved predictions.')
    print(f'Curve data: {csv_path}; only training/validation images opened.')


def plot(rows, output, preview=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    baseline = next(r for r in rows if r['method'] == METHODS[0])['validation_accuracy']*100
    style = {'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.8, 3.8), layout='constrained')
        for method, marker, color, label in (
            (METHODS[1], 'o', '#0072B2', 'PCA'),
            (METHODS[2], 's', '#D55E00', 'Direct ridge LDA'),
            (METHODS[3], '^', '#009E73', 'PCA→LDA (m=40)'),
        ):
            maximum = 50 if method == METHODS[1] else 39
            selected = [r for r in rows if r['method'] == method and r['retained_dimension'] <= maximum]
            ax.scatter([r['retained_dimension'] for r in selected],
                       [r['validation_accuracy']*100 for r in selected],
                       marker=marker, edgecolors=color, facecolors='none', s=48, linewidths=1.3, label=label)
        ax.axhline(baseline, color='#555555', ls='--', lw=1.1,
                   label=f'Unreduced (10,304D): {baseline:g}%')
        ax.set(xlabel='Retained dimension, d', ylabel='Validation accuracy (%)',
               ylim=(0, 103), xlim=(0, 52))
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 39, 50])
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', frameon=True, fontsize=8.5)
        ax.set_title('ORL validation set (n=80) • Euclidean 1-NN', fontsize=11)
        fig.savefig(output, dpi=300)
        if preview is not None:
            fig.savefig(preview, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--preview', type=Path, help='Optional PNG for visual inspection')
    args = parser.parse_args()
    run(args.data_root, args.results_dir, args.preview)


if __name__ == '__main__':
    main()
