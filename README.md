# Dimensionality Reduction for Face Identification

**NTU IE4476 — Image Processing and Computer Vision**

Does reducing image dimensionality improve face-identification accuracy? This project compares PCA, direct ridge LDA, PCA→LDA, supervised PCA and fixed Haar wavelet features against unreduced pixels, using the same Euclidean 1-nearest-neighbor classifier throughout.

## Dataset and approach

The ORL/AT&T Face Database contains **400 grayscale images of 40 subjects**, each at 112×92 pixels. Images are scaled to [0,1] and flattened into 10,304 features.

A fixed, identity-stratified split uses **240 training, 80 validation and 80 test images**. All learned transformations use training data only; validation selects configurations. **The test set has not been evaluated.** The task is identification of known subjects.

## Validation results

| Method | Selected configuration | Correct / 80 | Accuracy |
|---|---|---:|---:|
| Unreduced pixels | 10,304 features | 76 | 95.00% |
| PCA | d=50 | 79 | **98.75%** |
| Direct ridge LDA | d=39, ridge=1e-3 | 78 | 97.50% |
| PCA→LDA | m=40, d=39 | 78 | 97.50% |
| Supervised PCA (η>1) | η=2, d=50 | 79 | **98.75%** |
| Haar DWT (LL) | level 3, 168 features | 78 | 97.50% |

PCA achieves the highest observed accuracy with substantially fewer features. Retaining more variance does not necessarily improve identification. PCA→LDA is sensitive to its intermediate dimension: m=200 gives only 45.00%, alongside much poorer within-class scatter conditioning.

Our SPCA is the specified multiclass weighted-covariance variant **Sw + ηSb**, not an exact reproduction of Jiang’s complete research algorithms. **η=1 is the ordinary-PCA equivalence control.** The best genuinely supervised setting, η=2, matches PCA’s predictions rather than improving them.

These are validation-selected findings on a small dataset. One image changes accuracy by 1.25 percentage points; PCA’s three-image gain over the baseline is not statistically conclusive (exploratory exact McNemar p=0.25).

## Repository

```text
src/       Data handling, reduction methods, 1-NN and experiment execution
tests/     Mathematical, reference and reproducibility checks
results/   Full measured CSVs, saved split and prediction metadata
figures/   Report figures and representative face images
```

The experiments include accuracy-versus-dimension comparisons, PCA variance analysis, the PCA→LDA intermediate-dimension ablation and the SPCA supervision-weight comparison. See [results/](results/) for numerical tables and [figures/](figures/) for publication-quality plots.

## Wavelet extension

`src/wavelets.py` extracts raw low-frequency (LL) coefficients using 2D Haar DWT with **symmetric boundary extension**, without learning parameters or using labels.

| Level | LL shape | Features | Feature reduction | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 56×46 | 2,576 | 75.00% | 95.00% (76/80) |
| 2 | 28×23 | 644 | 93.75% | 97.50% (78/80) |
| 3 | 14×12 | 168 | 98.37% | 97.50% (78/80) |
| 4 | 7×6 | 42 | 99.59% | 95.00% (76/80) |

Low-frequency information preserves substantial identity information: levels 2–3 match the LDA methods, while PCA remains one image better. Accuracy falls at level 4 as more spatial detail is discarded. These small validation differences do not establish superiority. Haar and PCA use different measured dimensions; the comparison does not imply matched-dimension performance.

The [comparison](figures/dwt_validation_accuracy.pdf), [multiresolution illustration](figures/dwt_multiresolution.pdf) and [four-configuration CSV](results/dwt_validation_results.csv) are reproducible with `python -B -m src.experiments --dwt`. The illustration uses LL/2^level on a common [0,1] display scale; classification uses raw coefficients. The CSV records package versions and prediction hashes.

## Reproduce

From the repository root, with the dataset at `../dataset-faces`:

```powershell
conda activate ie4476
python -m pip install -r requirements.txt
python -B -m src.experiments
python -B -m src.ablations
python -B -m src.experiments --spca
python -B -m src.experiments --dwt
python -B -m unittest discover -s tests -v
```

Run `python -B -m src.figures` to regenerate the face grid. Command-line options are available through `--help`.

For the [focused validation error analysis](results/classification_error_analysis.md), run `python -B -m src.error_analysis`; it reproduces the [two-case comparison](figures/classification_error_comparison.png) and a CSV of selected-method predictions.

## References

- F. Samaria and A. Harter, “Parameterisation of a stochastic model for human face identification,” IEEE Workshop on Applications of Computer Vision, 1994. Dataset credit: **Olivetti Research Laboratory**.
- IE4476 course lecture materials on PCA, LDA and supervised/asymmetric PCA.
- [PyWavelets 2D DWT documentation](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html); DWT is an exploration beyond the main PCA/LDA material.
