"""Frozen, training-only held-out evaluation; never silently repeat a test run."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from .classification import Euclidean1NN, classification_metrics
from .dimensionality_reduction import PCA, LDA, PCALDA, SPCA
from .error_analysis import selected_configurations
from .experiments import _array_digest, _state_digest
from .partition import DEFAULT_ROOT, DEFAULT_OUTPUT, load_partitions, validate_indices, make_split_indices
from .preprocessing import preprocess_images
from .wavelets import HaarDWT


CONFIGURATIONS = [
    dict(method='Unreduced', configuration='Original pixels', d=10304, validation_correct=76),
    dict(method='PCA', configuration='d=50', d=50, validation_correct=79),
    dict(method='Direct ridge LDA', configuration='d=39; ridge=0.001', d=39, ridge=0.001, validation_correct=78),
    dict(method='PCA→LDA', configuration='m=40; d=39; ridge=0', m=40, d=39, ridge=0, validation_correct=78),
    dict(method='SPCA', configuration='eta=2; d=50', eta=2, d=50, validation_correct=79),
    dict(method='Haar DWT', configuration='level=3; LL; mode=symmetric', level=3, d=168, validation_correct=78),
]
SOURCES = ('validation_results.csv', 'pcalda_intermediate_ablation.csv',
           'spca_validation_results.csv', 'dwt_validation_results.csv',
           'experiment_summary.json', 'split_indices.json')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_selection(results):
    """Verify the requested constants against existing validation selections only."""
    chosen, metadata = selected_configurations(results)
    aliases = {'SPCA': 'Supervised SPCA', 'Haar DWT': 'DWT'}
    for config in CONFIGURATIONS:
        saved = chosen[aliases.get(config['method'], config['method'])]
        for key in ('d', 'm', 'eta', 'level'):
            if key in config and saved.get(key) != config[key]:
                raise ValueError(f'{config["method"]}: saved selection differs for {key}')
        if saved['expected_correct'] != config['validation_correct']:
            raise ValueError('Validation count differs from frozen selection')
    # Ridge and DWT boundary conventions are fixed by the existing implementations.
    diagnostics = metadata['diagnostics']
    if diagnostics['direct_ridge'] != 0.001 or diagnostics['pipeline_ridge'] != 0:
        raise ValueError('Saved ridge conventions differ')
    extractor = HaarDWT(3)
    if extractor.wavelet != 'haar' or extractor.mode != 'symmetric':
        raise ValueError('DWT conventions differ')
    split = json.loads((results/'split_indices.json').read_text(encoding='utf-8'))
    indices = {name: np.asarray(part['indices'], dtype=np.int64)
               for name, part in split['partitions'].items()}
    labels = np.repeat(np.arange(1, 41), 10)
    validate_indices(labels, indices)
    expected = make_split_indices(labels, split['seed'])
    paths = [f's{s}/{i}.pgm' for s in range(1, 41) for i in range(1, 11)]
    for name, part in split['partitions'].items():
        if (not np.array_equal(indices[name], expected[name])
                or part['paths'] != [paths[i] for i in indices[name]]):
            raise ValueError('Saved split paths/order differ from approved partition')
    if sha(results/'split_indices.json') != metadata['split_sha256']:
        raise ValueError('Saved split differs from validation experiments')
    return metadata


def freeze(results):
    metadata = verify_selection(results)
    protocol = dict(schema_version=1, frozen_at_utc=timestamp(), configurations=CONFIGURATIONS,
                    training_count=240, validation_count=80, test_count=80,
                    preprocessing='uint8 grayscale; row-major flatten; float32 / 255',
                    classifier='Euclidean 1-NN; float64 distances; first training row breaks ties',
                    fit_partition='train only; no training/validation refit',
                    selection='Existing validation results only; no post-test selection',
                    selection_ties='Highest validation correct count, then smaller dimension; '
                                   'PCA→LDA: smaller m; SPCA: smaller d then eta, restricted to eta>1',
                    source_sha256={name: sha(results/name) for name in SOURCES},
                    code_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                    training_features_sha256=metadata['training_features_sha256'])
    path = results/'final_test_protocol.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return protocol


def load_training(root, results, protocol):
    """Read only original training images, before any test pixels are opened."""
    root = Path(root).resolve()
    saved = json.loads((results/'split_indices.json').read_text(encoding='utf-8'))
    images, labels = [], []
    for relative in saved['partitions']['train']['paths']:
        path = (root/relative).resolve()
        if root not in path.parents:
            raise ValueError('Image path escapes dataset root')
        with Image.open(path) as image:
            if image.mode != 'L' or image.size != (92, 112):
                raise ValueError(f'Unexpected image representation: {relative}')
            images.append(np.asarray(image).copy())
        labels.append(int(path.parent.name[1:]))
    X, y = preprocess_images(np.stack(images)), np.asarray(labels, dtype=np.int64)
    if len(X) != 240 or _array_digest(X) != protocol['training_features_sha256']:
        raise ValueError('Training samples differ from approved experiment inputs')
    return X, y


def fit_models(X, y, configurations):
    """This function has no validation/test inputs; complete every fit first."""
    models = []
    for config in configurations:
        name = config['method']
        if name == 'Unreduced':
            model = None
        elif name == 'PCA':
            model = PCA(config['d']).fit(X)
        elif name == 'Direct ridge LDA':
            model = LDA(config['d'], ridge=config['ridge']).fit(X, y)
        elif name == 'PCA→LDA':
            model = PCALDA(config['m'], config['d'], ridge=config['ridge']).fit(X, y)
        elif name == 'SPCA':
            model = SPCA(config['d'], eta=config['eta']).fit(X, y)
        elif name == 'Haar DWT':
            model = HaarDWT(config['level'])
        else:
            raise ValueError(f'Unknown frozen method: {name}')
        Z = X if model is None else model.transform(X)
        if Z.shape != (len(X), config['d']):
            raise ValueError('Unexpected training feature dimensions')
        models.append((model, Euclidean1NN().fit(Z, y)))
    return models


def evaluate(root, results):
    protocol_path = results/'final_test_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['configurations'] != CONFIGURATIONS:
        raise ValueError('Frozen configurations changed')
    verify_selection(results)
    for name, digest in protocol['source_sha256'].items():
        if sha(results/name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    for name, digest in protocol['code_sha256'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError(f'Frozen code changed: {name}')
    # Exclusive creation refuses a second run, including an interrupted partial run.
    audit_path = results/'final_test_audit.json'
    audit = dict(status='started', started_at_utc=timestamp(), protocol_sha256=sha(protocol_path),
                 python=platform.python_version(),
                 versions={name: version(name) for name in ('numpy', 'scipy', 'Pillow', 'PyWavelets')},
                 predictions={})
    with audit_path.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2)
    X, y = load_training(root, results, protocol)
    before = (_array_digest(X), _array_digest(y))
    models = fit_models(X, y, protocol['configurations'])
    audit['all_fits_completed_at_utc'] = timestamp()
    write_json(audit_path, audit)
    # Existing loader verifies the complete dataset fingerprint and saved split.
    # Its integrity scan opens all source files only AFTER every model has fitted.
    parts = load_partitions(root, results/'split_indices.json', names=('train', 'test'))
    if not (np.array_equal(X, parts['train']['X']) and np.array_equal(y, parts['train']['y'])):
        raise ValueError('Fitting inputs differ from verified training partition')
    test = parts['test']
    audit.update(test_loaded_at_utc=timestamp(), test_indices=test['indices'].tolist(),
                 test_paths=test['paths'].tolist(), test_labels=test['y'].tolist(),
                 test_features_sha256=_array_digest(test['X']),
                 verification=dict(split_disjoint_exhaustive_stratified=True,
                                   dataset_matches_original_fingerprint=True,
                                   all_fits_before_test_loading=True, training_count=240,
                                   validation_used_for_fitting=False, test_used_for_fitting=False))
    write_json(audit_path, audit)
    for config, (model, classifier) in zip(protocol['configurations'], models):
        state = (None if model is None else _state_digest(model), _state_digest(classifier))
        features = test['X'] if model is None else model.transform(test['X'])
        if features.shape != (80, config['d']):
            raise ValueError('Unexpected test feature dimensions')
        prediction = classifier.predict(features)  # The sole real-test predict call for this method.
        audit['predictions'][config['method']] = dict(labels=prediction.tolist(), predict_calls=1,
                                                       sha256=_array_digest(prediction.astype('<i8')))
        write_json(audit_path, audit)  # Preserve completed predictions even if a later method fails.
        if state != (None if model is None else _state_digest(model), _state_digest(classifier)):
            raise RuntimeError('Prediction or transformation changed fitted model state')
        print(f'{config["method"]}: predictions saved', flush=True)
    if before != (_array_digest(X), _array_digest(y)):
        raise RuntimeError('Evaluation changed training inputs')
    audit.update(status='completed', completed_at_utc=timestamp())
    write_json(audit_path, audit)
    render(results)


def render(results):
    """Rebuild report artifacts from saved predictions without loading images."""
    protocol_path = results/'final_test_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    audit = json.loads((results/'final_test_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'completed' or sha(protocol_path) != audit['protocol_sha256']:
        raise ValueError('A completed evaluation matching the frozen protocol is required')
    truth = np.asarray(audit['test_labels'], dtype=np.int64)
    if len(truth) != 80:
        raise ValueError('Expected 80 saved test labels')
    rows = []
    for config in protocol['configurations']:
        saved = audit['predictions'][config['method']]
        prediction = np.asarray(saved['labels'], dtype=np.int64)
        if saved['predict_calls'] != 1 or _array_digest(prediction.astype('<i8')) != saved['sha256']:
            raise ValueError('Invalid saved prediction record')
        metrics = classification_metrics(truth, prediction)
        rows.append(dict(method=config['method'], configuration=config['configuration'],
                         feature_dimensionality=config['d'], test_accuracy=metrics['accuracy'],
                         error_rate=metrics['error_rate'], n_correct=metrics['n_correct'],
                         n_incorrect=80-metrics['n_correct'], n_test=80))
    with (results/'final_test_results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (results/'final_test_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        names = [r['method'] for r in rows]
        writer = csv.writer(handle)
        writer.writerow(['global_index', 'path', 'true_identity'] + names)
        for i in range(80):
            writer.writerow([audit['test_indices'][i], audit['test_paths'][i], int(truth[i])]
                            + [audit['predictions'][name]['labels'][i] for name in names])
    lines = ['# Final held-out test evaluation', '',
             'Six configurations were frozen against saved validation results before test access. '
             'Every learned transformation and Euclidean 1-NN used only the original 240 training '
             'observations. Validation was not added to training. Each method predicted the 80 test '
             'observations once; no configuration was changed afterward.', '',
             '| Method | Frozen configuration | Features | Validation | Test | Test errors | Δ test−validation (pp) |',
             '|---|---|---:|---:|---:|---:|---:|']
    latex = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Frozen configurations: validation selection and one-time held-out test accuracy. '
             r'All models use only 240 training images; each evaluation partition contains 80 images.}',
             r'\label{tab:final-test}', r'\begin{tabular}{lrrrr}', r'\hline',
             r'Method & Features & Validation (\%) & Test (\%) & Test correct \\', r'\hline']
    for config, row in zip(protocol['configurations'], rows):
        validation = config['validation_correct']/80
        delta = 100*(row['test_accuracy']-validation)
        lines.append(f'| {row["method"]} | {row["configuration"]} | {config["d"]:,} | '
                     f'{100*validation:.2f}% ({config["validation_correct"]}/80) | '
                     f'{100*row["test_accuracy"]:.2f}% ({row["n_correct"]}/80) | {row["n_incorrect"]} | {delta:+.2f} |')
        name = row['method'].replace('→', r'$\rightarrow$')
        latex.append(f'{name} & {config["d"]:,} & {100*validation:.2f} & '
                     f'{100*row["test_accuracy"]:.2f} & {row["n_correct"]}/80 ' + r'\\')
    best = max(row['n_correct'] for row in rows)
    leaders = ', '.join(row['method'] for row in rows if row['n_correct'] == best)
    lines += ['', f'The highest observed test accuracy is {100*best/80:.2f}% ({best}/80): {leaders}. '
              'This is a descriptive comparison of already frozen methods, not a new model-selection step.', '',
              'SPCA uses the specified multiclass weighted covariance Sw + ηSb, with η=2; '
              'the η=1 ordinary-PCA equivalence control is not an additional final-test method. '
              'Haar DWT uses level-3 raw LL coefficients (14×12), symmetric boundaries, and no learned transform.', '',
              'Validation and test contain different images of the same 40 identities. Differences may reflect '
              'sample variation and validation-selection optimism; they do not isolate a causal explanation. '
              'One image changes accuracy by 1.25 percentage points. These small, dependent observations '
              'from one fixed split do not establish statistical superiority or generalization to unseen identities.', '',
              '## Verification and reproduction', '',
              '`final_test_protocol.json` records the pre-test freeze, source/code hashes and validation counts. '
              '`final_test_audit.json` records package versions, timestamps, partition checks and each prediction vector. '
              'The existing dataset fingerprint and exhaustive, disjoint 6/2/2-per-identity split were verified. '
              'All fitting completed before test images were loaded; transform/classifier states and training inputs '
              'remained unchanged during prediction. Validation CSVs and the saved partition were preserved.', '',
              'The execution sequence was `python -B -m src.final_evaluation --freeze` followed by '
              '`python -B -m src.final_evaluation`. The run record blocks a second evaluation, including after interruption. '
              'To regenerate CSVs, this summary and the portable LaTeX table **without fitting or test prediction**, run:', '',
              '```powershell', 'conda activate ie4476', 'python -B -m src.final_evaluation --render-only', '```', '',
              'Report table: `results/final_test_table.tex` (standard LaTeX, no additional packages). '
              'Per-image predictions: `results/final_test_predictions.csv`. Accuracies/error rates in the results CSV '
              'are fractions; percentages are used here and in the report table.', '']
    (results/'final_test_findings.md').write_text('\n'.join(lines), encoding='utf-8')
    latex += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    (results/'final_test_table.tex').write_text('\n'.join(latex), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true', help='Document approved configurations without opening images')
    mode.add_argument('--render-only', action='store_true', help='Rebuild reports from saved predictions only')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.results_dir)
        print('Configurations verified and frozen; no images loaded.')
    elif args.render_only:
        render(args.results_dir)
        print('Reports regenerated from saved predictions; no images loaded.')
    else:
        try:
            evaluate(args.data_root, args.results_dir)
        except FileExistsError:
            parser.exit(1, 'Evaluation already started or completed. Refusing another test evaluation. '
                        'Use --render-only for completed results.\n')


if __name__ == '__main__':
    main()
