"""Selected-model validation errors and within-representation neighbor evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .ablations import read_validation_results
from .classification import Euclidean1NN
from .dimensionality_reduction import PCA, SPCA, LDA, PCALDA
from .experiments import _array_digest, _state_digest, best_configurations
from .partition import DEFAULT_ROOT, DEFAULT_OUTPUT
from .preprocessing import preprocess_images
from .wavelets import HaarDWT

OUTPUT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def selected_configurations(results):
    rows, metadata = read_validation_results(results)
    chosen = best_configurations(rows)
    configurations = {name: dict(d=r['retained_dimension'], expected_correct=r['n_correct'])
                      for name, r in chosen.items() if name != 'PCA→LDA'}
    pipeline = min(read_csv(results/'pcalda_intermediate_ablation.csv'),
                   key=lambda r: (-int(r['n_correct']), int(r['intermediate_pca_dimension'])))
    configurations['PCA→LDA'] = dict(d=int(pipeline['retained_dimension']), m=int(pipeline['intermediate_pca_dimension']), expected_correct=int(pipeline['n_correct']))
    spca = [r for r in read_csv(results/'spca_validation_results.csv') if r['method'] == 'SPCA']
    key = lambda r: (-int(r['n_correct']), int(r['retained_dimension']), float(r['supervision_weight']))
    for name, pool in (('SPCA control', spca), ('Supervised SPCA', [r for r in spca if float(r['supervision_weight']) > 1])):
        row = min(pool, key=key)
        configurations[name] = dict(d=int(row['retained_dimension']), eta=float(row['supervision_weight']), expected_correct=int(row['n_correct']))
    # DWT levels 2/3 tie: preserve the already reported smaller representation.
    row = min(read_csv(results/'dwt_validation_results.csv'), key=lambda r: (-int(r['n_correct']), int(r['retained_dimension'])))
    configurations['DWT'] = dict(d=int(row['retained_dimension']), level=int(row['level']), expected_correct=int(row['n_correct']))
    return configurations, metadata


def load_review_partitions(root, split_file, metadata):
    """Load ONLY train/validation paths; do not open or hash test image files.

    Reuse approved preprocessing and compare resulting feature/index hashes to
    the existing experiment metadata. This avoids the full-dataset integrity
    scan in load_partitions(), which would also open held-out test images.
    """
    root, split_file = Path(root).resolve(), Path(split_file)
    content = split_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata['split_sha256']:
        raise ValueError('Saved split differs from approved experiments')
    saved, result = json.loads(content), {}
    for name, key, per_class in (('train', 'training', 6), ('validation', 'validation', 2)):
        part = saved['partitions'][name]
        paths, indices = part['paths'], np.asarray(part['indices'], dtype=np.int64)
        images, labels = [], []
        for relative in paths:
            path = (root/relative).resolve()
            if root not in path.parents:
                raise ValueError('Image path escapes dataset root')
            with Image.open(path) as image:
                if image.mode != 'L' or image.size != (92, 112):
                    raise ValueError(f'Unexpected image representation: {relative}')
                images.append(np.array(image))
            labels.append(int(path.parent.name[1:]))
        X, y = preprocess_images(np.stack(images)), np.asarray(labels, dtype=np.int64)
        if (_array_digest(X) != metadata[f'{key}_features_sha256']
                or indices.tolist() != metadata[f'{key}_indices']
                or any(np.count_nonzero(y == label) != per_class for label in range(1, 41))):
            raise ValueError(f'{name} samples/features differ from approved results')
        result[name] = dict(X=X, y=y, paths=np.asarray(paths), indices=indices)
    if np.intersect1d(result['train']['indices'], result['validation']['indices']).size:
        raise ValueError('Training and validation overlap')
    return result


def neighbors(training_features, query, training_labels, true_label):
    """Stable ranking agrees with 1-NN; distances are ordinary Euclidean units."""
    difference = np.asarray(training_features, dtype=float) - np.asarray(query, dtype=float)
    squared = np.einsum('ij,ij->i', difference, difference)
    order = np.argsort(squared, kind='stable')
    correct = next(int(i) for i in order if training_labels[i] == true_label)
    wrong = next(int(i) for i in order if training_labels[i] != true_label)
    nearest = int(order[0])
    return dict(nearest=nearest, correct=correct, wrong=wrong,
                nearest_distance=float(np.sqrt(squared[nearest])),
                correct_distance=float(np.sqrt(squared[correct])), wrong_distance=float(np.sqrt(squared[wrong])),
                correct_rank=int(np.flatnonzero(order == correct)[0] + 1))


def analyse(parts, configurations, metadata, results):
    train, validation = parts['train'], parts['validation']
    X, y, V = train['X'], train['y'], validation['X']
    before = tuple(_array_digest(a) for a in (X, y, V, validation['y']))
    pca = PCA(configurations['PCA']['d']).fit(X)
    projections, predictions = {}, {}
    for name, config in configurations.items():
        if name == 'Unreduced':
            Z, ZV = X, V
        else:
            if name == 'PCA':
                model = pca
            elif name == 'Direct ridge LDA':
                model = LDA(config['d'], ridge=1e-3).fit(X, y)
            elif name == 'PCA→LDA':
                model = PCALDA(config['m'], config['d'], ridge=0).fit(X, y, fitted_pca=pca)
            elif name in ('SPCA control', 'Supervised SPCA'):
                model = SPCA(config['d'], eta=config['eta']).fit(X, y)
            elif name == 'DWT':
                model = HaarDWT(config['level'])
            else:
                raise ValueError(f'Unknown method: {name}')
            state = _state_digest(model)
            Z, ZV = model.transform(X), model.transform(V)
            if state != _state_digest(model):
                raise RuntimeError('Transform changed fitted parameters')
        prediction = Euclidean1NN().fit(Z, y).predict(ZV)
        if Z.shape != (len(X), config['d']) or ZV.shape != (len(V), config['d']):
            raise RuntimeError(f'{name}: projection dimension differs from saved configuration')
        if np.count_nonzero(prediction == validation['y']) != config['expected_correct']:
            raise RuntimeError(f'{name}: selected accuracy differs from saved result')
        projections[name], predictions[name] = (Z, ZV), prediction
    # Check all previously stored vectors, not only their aggregate accuracies.
    stored = read_csv(results/'selected_validation_predictions.csv')
    if [r['path'] for r in stored] != validation['paths'].tolist():
        raise ValueError('Stored prediction order differs')
    for name in ('Unreduced', 'PCA', 'Direct ridge LDA', 'PCA→LDA'):
        if not np.array_equal(predictions[name], [int(r[name]) for r in stored]):
            raise RuntimeError(f'{name}: predictions differ from saved vector')
    if not np.array_equal(predictions['SPCA control'], predictions['PCA']):
        raise RuntimeError('SPCA equivalence control differs from PCA')
    if not np.array_equal(predictions['Supervised SPCA'], predictions['PCA']):
        raise RuntimeError('Selected supervised SPCA differs from previously observed PCA prediction tie')
    # Check selected DWT predictions against the exact previous hash.
    stored_dwt = next(r for r in read_csv(results/'dwt_validation_results.csv') if int(r['level']) == configurations['DWT']['level'])
    if _array_digest(predictions['DWT'].astype('<i8')) != stored_dwt['prediction_sha256']:
        raise RuntimeError('DWT predictions differ from saved result')
    if before != tuple(_array_digest(a) for a in (X, y, V, validation['y'])):
        raise RuntimeError('Analysis changed input samples')
    errors = {name: np.flatnonzero(pred != validation['y']).tolist() for name, pred in predictions.items()}
    scores = {name: len(V)-len(indices) for name, indices in errors.items()}
    best_score = max(scores.values())
    top = [name for name in scores if scores[name] == best_score]
    next_score = max(score for score in scores.values() if score < best_score)
    next_group = [name for name in scores if scores[name] == next_score]
    # PCA is a representative of the top tied group, not a fabricated winner.
    first = errors['PCA'][0]
    focal = next(name for name in next_group if predictions[name][first] == validation['y'][first])
    changed = [i for i in errors[focal] if predictions['PCA'][i] == validation['y'][i]]
    illustrated = [first] + changed[:1]
    relevant = sorted(set(errors['PCA']) | {i for name in next_group for i in errors[name] if predictions['PCA'][i] == validation['y'][i]})
    details = {i: {name: neighbors(Z, ZV[i], y, validation['y'][i])
                   for name, (Z, ZV) in projections.items()} for i in relevant}
    for i in relevant:
        for name in predictions:
            if y[details[i][name]['nearest']] != predictions[name][i]:
                raise RuntimeError('Neighbor ranking disagrees with the existing classifier')
    return dict(configurations=configurations, predictions=predictions, errors=errors,
                scores=scores, top=top, next_group=next_group, details=details,
                illustrated=illustrated, focal=focal)


def plot_comparison(analysis, parts, path):
    """Two complementary cases, original pixels, fixed display scale [0,1]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train, validation = parts['train'], parts['validation']
    focal = analysis['focal']
    short = {'PCA': 'PCA', 'Direct ridge LDA': 'Ridge LDA', 'PCA→LDA': 'PCA→LDA', 'DWT': 'DWT'}
    with plt.rc_context({'font.size': 9}):
        fig, axes = plt.subplots(len(analysis['illustrated']), 4, figsize=(7.2, 2.65*len(analysis['illustrated'])), squeeze=False, layout='constrained')
        for row, i in enumerate(analysis['illustrated']):
            true = int(validation['y'][i])
            error_method = 'PCA' if analysis['predictions']['PCA'][i] != true else focal
            correct_details = analysis['details'][i][error_method]
            panels = [(validation['X'][i], f'Query: true ID {true}\n{validation["paths"][i]}', '#555555')]
            for method in ('PCA', focal):
                d = analysis['details'][i][method]
                k = d['nearest']
                predicted = int(train['y'][k])
                panels.append((train['X'][k], f'{short[method]}: chosen ID {predicted}\n{train["paths"][k]}\ndistance {d["nearest_distance"]:.4g}',
                               '#00836C' if predicted == true else '#B64B00'))
            k = correct_details['correct']
            panels.append((train['X'][k], f'{short[error_method]}: nearest ID {true}\n{train["paths"][k]}\ndistance {correct_details["correct_distance"]:.4g}; rank {correct_details["correct_rank"]}', '#00836C'))
            for col, (image, title, color) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(image.reshape(112, 92), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.set_title(title, fontsize=8.2, color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(1.5)
        fig.suptitle('Complementary validation errors: PCA (50D) and ridge LDA (39D)', fontsize=11)
        fig.supxlabel('Original images; common [0,1] display. Distances/ranks are within each representation only.', fontsize=8)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)


def write_summary(analysis, parts, path):
    train, validation = parts['train'], parts['validation']
    lines = ['# Qualitative validation error analysis', '',
             '## Selection and ties', '',
             'PCA(d=50) and supervised SPCA(eta=2,d=50) tie at **79/80 (98.75%)**, with identical predictions. '
             'SPCA eta=1 is the equivalent PCA control, also identical; it is not counted as a distinct competing representation. '
             'The overall within-SPCA rule selects eta=1; the previously reported best eta>1 setting is retained separately here. '
             'Direct ridge LDA(d=39,ridge=1e-3), PCA→LDA(m=40,d=39,ridge=0), and Haar LL(level=3,168D,symmetric) '
             'tie at **78/80 (97.50%)**. Thus there is neither a unique best method nor a unique runner-up. '
             'DWT levels 2/3 tie; the reported smaller representation, level 3, is retained. Unreduced pixels give 76/80.', '',
             'Previously defined within-method selection rules are unchanged. No across-method ranking is invented. '
             'The figure uses ridge LDA as an illustrative member of the next accuracy group because it corrects the PCA failure '
             'and also makes an error PCA avoids.', '',
             '## All observed errors', '', '| Method | Misclassified validation images (true → predicted ID) |', '|---|---|']
    for name, indices in analysis['errors'].items():
        cases = ', '.join(f'`{validation["paths"][i]}` ({validation["y"][i]}→{analysis["predictions"][name][i]})' for i in indices) or 'none'
        lines.append(f'| {name} | {cases} |')
    lines += ['', 'Every top-group error is `s5/10.pgm` (5→40). It is shared with unreduced pixels, '
              'PCA→LDA and DWT, but **ridge LDA correctly identifies it**. '
              'Errors in the next accuracy group that PCA/SPCA avoid are `s1/10.pgm` '
              '(LDA and PCA→LDA predict 17; DWT predicts 16) and `s16/8.pgm` (ridge LDA predicts 1). '
              'PCA and ridge LDA therefore have **no shared error images**, despite only one correct-count difference.', '',
              '## Nearest-neighbor evidence', '',
              'Distances below are Euclidean distances in the unchanged method representation. '
              'Their absolute magnitudes **must not be compared across methods**: the feature bases and LDA scaling differ. '
              'Correct rank is the first correct-class training sample’s rank among all 240 training images. '
              'Equal-distance ties retain saved training-row order.', '',
              '| Query | Method | Chosen training neighbor (ID) | Distance | Nearest correct-class neighbor | Distance | Correct rank |',
              '|---|---|---|---:|---|---:|---:|']
    for i, methods in analysis['details'].items():
        for name in ('PCA', 'Supervised SPCA', 'Direct ridge LDA', 'PCA→LDA', 'DWT'):
            d = methods[name]
            nn, correct = d['nearest'], d['correct']
            lines.append(f'| {validation["paths"][i]} | {name} | {train["paths"][nn]} ({train["y"][nn]}) | '
                         f'{d["nearest_distance"]:.4f} | {train["paths"][correct]} | {d["correct_distance"]:.4f} | {d["correct_rank"]} |')
    lines += ['', '### Visual observations and interpretation', '',
              '**s5/10.pgm (illustrated, top row).** The query and the wrongly chosen s40/5 image are approximately frontal '
              'with slightly open mouths. The correct-identity training images show different head angles and smile shapes. '
              'These visible pose/expression differences are plausible sources of pixel variation, not established causes of error. '
              'PCA prefers s40/5 at distance 7.1249 to its nearest identity-5 image at 8.8335 (rank 5). '
              'Ridge LDA instead ranks s5/3 first (1135.5704), ahead of the nearest wrong class (1365.0608). '
              'The learned representations reverse the class preference. DWT and both SPCA settings still choose s40/5.', '',
              '**s1/10.pgm (illustrated, bottom row).** Head angle and gaze differ between the query and the frontal '
              'identity-1 training examples. The LDA-selected s17/1 neighbor visibly has glasses and facial hair, '
              'unlike the query; the error is not adequately explained by an obvious visual lookalike. '
              'PCA chooses the correct s1/8 at 11.3913, versus the nearest wrong-class distance 11.8683. '
              'In ridge LDA, s17/1 is closer (1648.7620) than every identity-1 sample; the closest correct image '
              's1/1 is at 1953.8415 and rank 31. PCA→LDA also chooses s17/1 (correct-class rank 7). '
              'DWT chooses s16/7 at 12.2924 while the correct s1/8 is second at 12.7275. '
              'This is quantitative evidence of changed distance order, not proof that glasses, pose or a specific filter caused it.', '',
              '**s16/8.pgm (tabulated, not added to the compact figure).** The query and correct identity-16 samples '
              'show modest head-angle changes. PCA selects the correct s16/2, whereas ridge LDA selects s1/1; '
              'its closest identity-16 image, s16/7, falls to rank 7. PCA’s closest correct/wrong distances '
              'are 11.6744/11.8153; ridge LDA’s are 1721.9663/1654.1279. '
              'Visible within-identity pose variation alone cannot explain why one representation succeeds and the other fails.', '',
              'The reviewed queries show no obvious large occluding object. Tonal/shading differences are visible, '
              'but illumination was not measured or manipulated. No causal illumination or occlusion explanation is established. '
              'PCA preserves high total-variance directions, LDA rescales training discriminants, and DWT keeps low-frequency '
              'spatial coefficients; these definitions explain why the metrics can differ, but the photographs do not identify '
              'which particular visual factor produced an error. In particular, a greater aggregate accuracy does not make '
              'PCA correct on every image that a lower-scoring method gets right.', '',
              '## Reproducibility and scope', '',
              'Only the saved 240 training and 80 validation image paths are opened. Test images are neither opened nor evaluated; '
              'the saved split and prior feature/index hashes verify alignment. Fits use training observations/labels only. '
              'Existing prediction vectors/hashes and correct counts are checked before collecting neighbors. '
              'The classifier and selected settings are unchanged. '
              'The 80-row `classification_error_predictions.csv` records paths, true IDs and every selected prediction/correctness flag. '
              'Case selection is qualitative and post hoc; these two examples cannot establish a population-level cause or superiority.', '',
              'Reproduce: `python -B -m src.error_analysis`. Figure: `../figures/classification_error_comparison.png`.']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(data_root=DEFAULT_ROOT, split_file=DEFAULT_OUTPUT/'split_indices.json', previous_results=DEFAULT_OUTPUT, output_dir=OUTPUT):
    root, results, output = Path(data_root).resolve(), Path(previous_results), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the original dataset')
    config, metadata = selected_configurations(results)
    split_bytes = Path(split_file).read_bytes()
    parts = load_review_partitions(root, split_file, metadata)
    analysis = analyse(parts, config, metadata, results)
    if Path(split_file).read_bytes() != split_bytes:
        raise RuntimeError('Analysis changed the saved split')
    destination = output/'results'
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (relative, truth) in enumerate(zip(parts['validation']['paths'], parts['validation']['y'])):
        row = dict(index=int(parts['validation']['indices'][i]), path=str(relative), true_label=int(truth))
        for name, predictions in analysis['predictions'].items():
            row[name+' prediction'] = int(predictions[i])
            row[name+' correct'] = bool(predictions[i] == truth)
        rows.append(row)
    with (destination/'classification_error_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(analysis, parts, output/'figures/classification_error_comparison.png')
    write_summary(analysis, parts, destination/'classification_error_analysis.md')
    print('Top tied group:', ', '.join(analysis['top']))
    print('Next tied accuracy group:', ', '.join(analysis['next_group']))
    print('Illustrated:', ', '.join(parts['validation']['paths'][i] for i in analysis['illustrated']))
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
