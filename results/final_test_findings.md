# Final held-out test evaluation

Six configurations were frozen against saved validation results before test access. Every learned transformation and Euclidean 1-NN used only the original 240 training observations. Validation was not added to training. Each method predicted the 80 test observations once; no configuration was changed afterward.

| Method | Frozen configuration | Features | Validation | Test | Test errors | Δ test−validation (pp) |
|---|---|---:|---:|---:|---:|---:|
| Unreduced | Original pixels | 10,304 | 95.00% (76/80) | 96.25% (77/80) | 3 | +1.25 |
| PCA | d=50 | 50 | 98.75% (79/80) | 97.50% (78/80) | 2 | -1.25 |
| Direct ridge LDA | d=39; ridge=0.001 | 39 | 97.50% (78/80) | 98.75% (79/80) | 1 | +1.25 |
| PCA→LDA | m=40; d=39; ridge=0 | 39 | 97.50% (78/80) | 98.75% (79/80) | 1 | +1.25 |
| SPCA | eta=2; d=50 | 50 | 98.75% (79/80) | 96.25% (77/80) | 3 | -2.50 |
| Haar DWT | level=3; LL; mode=symmetric | 168 | 97.50% (78/80) | 97.50% (78/80) | 2 | +0.00 |

The highest observed test accuracy is 98.75% (79/80): Direct ridge LDA, PCA→LDA. This is a descriptive comparison of already frozen methods, not a new model-selection step.

SPCA uses the specified multiclass weighted covariance Sw + ηSb, with η=2; the η=1 ordinary-PCA equivalence control is not an additional final-test method. Haar DWT uses level-3 raw LL coefficients (14×12), symmetric boundaries, and no learned transform.

Validation and test contain different images of the same 40 identities. Differences may reflect sample variation and validation-selection optimism; they do not isolate a causal explanation. One image changes accuracy by 1.25 percentage points. These small, dependent observations from one fixed split do not establish statistical superiority or generalization to unseen identities.

## Verification and reproduction

`final_test_protocol.json` records the pre-test freeze, source/code hashes and validation counts. `final_test_audit.json` records package versions, timestamps, partition checks and each prediction vector. The existing dataset fingerprint and exhaustive, disjoint 6/2/2-per-identity split were verified. All fitting completed before test images were loaded; transform/classifier states and training inputs remained unchanged during prediction. Validation CSVs and the saved partition were preserved.

The execution sequence was `python -B -m src.final_evaluation --freeze` followed by `python -B -m src.final_evaluation`. The run record blocks a second evaluation, including after interruption. To regenerate CSVs, this summary and the portable LaTeX table **without fitting or test prediction**, run:

```powershell
conda activate ie4476
python -B -m src.final_evaluation --render-only
```

Report table: `results/final_test_table.tex` (standard LaTeX, no additional packages). Per-image predictions: `results/final_test_predictions.csv`. Accuracies/error rates in the results CSV are fractions; percentages are used here and in the report table.
