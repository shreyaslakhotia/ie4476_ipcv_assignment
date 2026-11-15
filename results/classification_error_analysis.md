# Qualitative validation error analysis

## Selection and ties

PCA(d=50) and supervised SPCA(eta=2,d=50) tie at **79/80 (98.75%)**, with identical predictions. SPCA eta=1 is the equivalent PCA control, also identical; it is not counted as a distinct competing representation. The overall within-SPCA rule selects eta=1; the previously reported best eta>1 setting is retained separately here. Direct ridge LDA(d=39,ridge=1e-3), PCA→LDA(m=40,d=39,ridge=0), and Haar LL(level=3,168D,symmetric) tie at **78/80 (97.50%)**. Thus there is neither a unique best method nor a unique runner-up. DWT levels 2/3 tie; the reported smaller representation, level 3, is retained. Unreduced pixels give 76/80.

Previously defined within-method selection rules are unchanged. No across-method ranking is invented. The figure uses ridge LDA as an illustrative member of the next accuracy group because it corrects the PCA failure and also makes an error PCA avoids.

## All observed errors

| Method | Misclassified validation images (true → predicted ID) |
|---|---|
| Unreduced | `s5/10.pgm` (5→40), `s10/9.pgm` (10→23), `s16/8.pgm` (16→1), `s28/9.pgm` (28→17) |
| PCA | `s5/10.pgm` (5→40) |
| Direct ridge LDA | `s1/10.pgm` (1→17), `s16/8.pgm` (16→1) |
| PCA→LDA | `s1/10.pgm` (1→17), `s5/10.pgm` (5→40) |
| SPCA control | `s5/10.pgm` (5→40) |
| Supervised SPCA | `s5/10.pgm` (5→40) |
| DWT | `s1/10.pgm` (1→16), `s5/10.pgm` (5→40) |

Every top-group error is `s5/10.pgm` (5→40). It is shared with unreduced pixels, PCA→LDA and DWT, but **ridge LDA correctly identifies it**. Errors in the next accuracy group that PCA/SPCA avoid are `s1/10.pgm` (LDA and PCA→LDA predict 17; DWT predicts 16) and `s16/8.pgm` (ridge LDA predicts 1). PCA and ridge LDA therefore have **no shared error images**, despite only one correct-count difference.

## Nearest-neighbor evidence

Distances below are Euclidean distances in the unchanged method representation. Their absolute magnitudes **must not be compared across methods**: the feature bases and LDA scaling differ. Correct rank is the first correct-class training sample’s rank among all 240 training images. Equal-distance ties retain saved training-row order.

| Query | Method | Chosen training neighbor (ID) | Distance | Nearest correct-class neighbor | Distance | Correct rank |
|---|---|---|---:|---|---:|---:|
| s1/10.pgm | PCA | s1/8.pgm (1) | 11.3913 | s1/8.pgm | 11.3913 | 1 |
| s1/10.pgm | Supervised SPCA | s1/8.pgm (1) | 11.4810 | s1/8.pgm | 11.4810 | 1 |
| s1/10.pgm | Direct ridge LDA | s17/1.pgm (17) | 1648.7620 | s1/1.pgm | 1953.8415 | 31 |
| s1/10.pgm | PCA→LDA | s17/1.pgm (17) | 10.8319 | s1/1.pgm | 13.7235 | 7 |
| s1/10.pgm | DWT | s16/7.pgm (16) | 12.2924 | s1/8.pgm | 12.7275 | 2 |
| s5/10.pgm | PCA | s40/5.pgm (40) | 7.1249 | s5/6.pgm | 8.8335 | 5 |
| s5/10.pgm | Supervised SPCA | s40/5.pgm (40) | 7.0752 | s5/6.pgm | 8.8043 | 5 |
| s5/10.pgm | Direct ridge LDA | s5/3.pgm (5) | 1135.5704 | s5/3.pgm | 1135.5704 | 1 |
| s5/10.pgm | PCA→LDA | s40/5.pgm (40) | 7.3995 | s5/2.pgm | 7.9395 | 2 |
| s5/10.pgm | DWT | s40/5.pgm (40) | 7.4001 | s5/6.pgm | 7.8879 | 3 |
| s16/8.pgm | PCA | s16/2.pgm (16) | 11.6744 | s16/2.pgm | 11.6744 | 1 |
| s16/8.pgm | Supervised SPCA | s16/2.pgm (16) | 11.6012 | s16/2.pgm | 11.6012 | 1 |
| s16/8.pgm | Direct ridge LDA | s1/1.pgm (1) | 1654.1279 | s16/7.pgm | 1721.9663 | 7 |
| s16/8.pgm | PCA→LDA | s16/7.pgm (16) | 11.7590 | s16/7.pgm | 11.7590 | 1 |
| s16/8.pgm | DWT | s16/2.pgm (16) | 11.7579 | s16/2.pgm | 11.7579 | 1 |

### Visual observations and interpretation

**s5/10.pgm (illustrated, top row).** The query and the wrongly chosen s40/5 image are approximately frontal with slightly open mouths. The correct-identity training images show different head angles and smile shapes. These visible pose/expression differences are plausible sources of pixel variation, not established causes of error. PCA prefers s40/5 at distance 7.1249 to its nearest identity-5 image at 8.8335 (rank 5). Ridge LDA instead ranks s5/3 first (1135.5704), ahead of the nearest wrong class (1365.0608). The learned representations reverse the class preference. DWT and both SPCA settings still choose s40/5.

**s1/10.pgm (illustrated, bottom row).** Head angle and gaze differ between the query and the frontal identity-1 training examples. The LDA-selected s17/1 neighbor visibly has glasses and facial hair, unlike the query; the error is not adequately explained by an obvious visual lookalike. PCA chooses the correct s1/8 at 11.3913, versus the nearest wrong-class distance 11.8683. In ridge LDA, s17/1 is closer (1648.7620) than every identity-1 sample; the closest correct image s1/1 is at 1953.8415 and rank 31. PCA→LDA also chooses s17/1 (correct-class rank 7). DWT chooses s16/7 at 12.2924 while the correct s1/8 is second at 12.7275. This is quantitative evidence of changed distance order, not proof that glasses, pose or a specific filter caused it.

**s16/8.pgm (tabulated, not added to the compact figure).** The query and correct identity-16 samples show modest head-angle changes. PCA selects the correct s16/2, whereas ridge LDA selects s1/1; its closest identity-16 image, s16/7, falls to rank 7. PCA’s closest correct/wrong distances are 11.6744/11.8153; ridge LDA’s are 1721.9663/1654.1279. Visible within-identity pose variation alone cannot explain why one representation succeeds and the other fails.

The reviewed queries show no obvious large occluding object. Tonal/shading differences are visible, but illumination was not measured or manipulated. No causal illumination or occlusion explanation is established. PCA preserves high total-variance directions, LDA rescales training discriminants, and DWT keeps low-frequency spatial coefficients; these definitions explain why the metrics can differ, but the photographs do not identify which particular visual factor produced an error. In particular, a greater aggregate accuracy does not make PCA correct on every image that a lower-scoring method gets right.

## Reproducibility and scope

Only the saved 240 training and 80 validation image paths are opened. Test images are neither opened nor evaluated; the saved split and prior feature/index hashes verify alignment. Fits use training observations/labels only. Existing prediction vectors/hashes and correct counts are checked before collecting neighbors. The classifier and selected settings are unchanged. The 80-row `classification_error_predictions.csv` records paths, true IDs and every selected prediction/correctness flag. Case selection is qualitative and post hoc; these two examples cannot establish a population-level cause or superiority.

Reproduce: `python -B -m src.error_analysis`. Figure: `../figures/classification_error_comparison.png`.
