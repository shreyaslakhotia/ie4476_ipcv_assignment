import copy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from sklearn.decomposition import PCA as ReferencePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.partition import load_partitions
from src.dimensionality_reduction import PCA, LDA, PCALDA


def data(p=6, per_class=24):
    rng = np.random.default_rng(123)
    y = np.repeat(np.arange(4), per_class)
    X = rng.normal(size=(len(y), p)) + rng.normal(size=(4, p))[y] * 2
    return X, y


def scatters(X, y):
    residual = X - np.asarray([X[y == label].mean(0) for label in y])
    B = np.asarray([np.sqrt(np.mean(y == label)) * (X[y == label].mean(0) - X.mean(0))
                    for label in np.unique(y)]).T
    return residual.T @ residual / len(X), B @ B.T


def projector(A):
    Q, _ = np.linalg.qr(A)
    return Q @ Q.T


class RidgeCovariance:
    """Reference estimator giving sklearn exactly the same ridge covariance."""
    def __init__(self, ridge):
        self.ridge = ridge

    def fit(self, X):
        centered = X - X.mean(0)
        self.covariance_ = centered.T @ centered / len(X) + self.ridge * np.eye(X.shape[1])
        return self


class ReductionTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = data()

    def test_pca_variance_orthogonality_and_reference(self):
        model = PCA(3).fit(self.X)
        reference = ReferencePCA(svd_solver='full').fit(self.X)
        assert_allclose(model.components_ @ model.components_.T, np.eye(6), atol=1e-13)
        self.assertTrue(np.all(np.diff(model.explained_variance_) <= 0))
        assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-12)
        assert_allclose(model.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-12)
        assert_allclose(model.cumulative_explained_variance_, np.cumsum(reference.explained_variance_ratio_))
        assert_allclose(np.var(model.transform(self.X, 6), axis=0, ddof=1), model.explained_variance_)
        assert_allclose(projector(model.components_[:3].T), projector(reference.components_[:3].T), atol=1e-12)
        Z = model.transform(self.X, 3)
        reconstruction = Z @ model.components_[:3] + model.mean_
        assert_allclose(np.sum((self.X - reconstruction) ** 2),
                        (len(self.X)-1) * model.explained_variance_[3:].sum(), rtol=1e-12)

    def test_lda_scatter_dense_reference_and_sklearn(self):
        for ridge in (0.0, 0.05):
            with self.subTest(ridge=ridge):
                model = LDA(3, ridge).fit(self.X, self.y)
                Sw, Sb = scatters(self.X, self.y)
                M = Sw + model.regularization_ * np.eye(6)
                values, vectors = eigh(Sb, M)
                W = model.scalings_
                assert_allclose(W.T @ M @ W, np.eye(3), atol=1e-12)
                assert_allclose(W.T @ Sb @ W, np.diag(model.eigenvalues_), atol=1e-11)
                assert_allclose(Sb @ W, (M @ W) * model.eigenvalues_, atol=1e-11)
                assert_allclose(model.eigenvalues_, values[-3:][::-1], rtol=1e-12)
                assert_allclose(projector(W), projector(vectors[:, -3:]), atol=1e-12)
                reference = LinearDiscriminantAnalysis(solver='eigen', n_components=3,
                            covariance_estimator=RidgeCovariance(model.regularization_)).fit(self.X, self.y)
                assert_allclose(projector(W), projector(reference.scalings_[:, :3]), atol=1e-11)
                # Distances are invariant to eigenvector signs/orthogonal changes.
                Z, Zref = model.transform(self.X), reference.transform(self.X)
                assert_allclose(np.sum((Z[:, None] - Z[None, :])**2, axis=2),
                                np.sum((Zref[:, None] - Zref[None, :])**2, axis=2), atol=1e-10)

    def test_direct_high_dimensional_ridge_retains_nullspace(self):
        X, y = data(p=80, per_class=8)
        model = LDA(3, ridge=0.01).fit(X, y)
        self.assertEqual(model.within_rank_, len(X) - len(np.unique(y)))
        Sw, Sb = scatters(X, y)
        M = Sw + model.regularization_ * np.eye(80)
        values, vectors = eigh(Sb, M)
        assert_allclose(model.eigenvalues_, values[-3:][::-1], rtol=1e-10)
        assert_allclose(model.scalings_.T @ M @ model.scalings_, np.eye(3), atol=1e-10)
        assert_allclose(projector(model.scalings_), projector(vectors[:, -3:]), atol=1e-9)
        _, _, vt = np.linalg.svd(X - np.asarray([X[y == label].mean(0) for label in y]), full_matrices=False)
        basis = vt[:model.within_rank_]
        null_part = model.scalings_ - basis.T @ (basis @ model.scalings_)
        self.assertGreater(np.linalg.norm(null_part), 1.0)
        with self.assertRaisesRegex(ValueError, 'singular'):
            LDA(3, ridge=0).fit(X, y)

    def test_pipeline_exact_sequential_equivalence(self):
        pipeline = PCALDA(5, 3).fit(self.X, self.y)
        pca = PCA(5).fit(self.X)
        lda = LDA(3, ridge=0).fit(pca.transform(self.X), self.y)
        probe = np.random.default_rng(55).normal(size=(7, 6))
        assert_allclose(pipeline.transform(probe), lda.transform(pca.transform(probe)), atol=1e-12)
        self.assertEqual(pipeline.lda_.within_rank_, 5)
        shared = PCA().fit(self.X)
        reused = PCALDA(5, 3).fit(self.X, self.y, fitted_pca=shared)
        self.assertIs(reused.pca_, shared)
        assert_allclose(reused.transform(probe), pipeline.transform(probe), atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'exactly these training'):
            PCALDA(5, 3).fit(self.X + 1, self.y, fitted_pca=shared)

    def test_transform_is_read_only_and_deterministic(self):
        for model in (PCA(3), LDA(3), PCALDA(5, 3)):
            if isinstance(model, PCA):
                model.fit(self.X)
            else:
                model.fit(self.X, self.y)
            before = copy.deepcopy(model)
            probe = np.random.default_rng(18).normal(size=(5, 6)) + 100
            first = model.transform(probe)
            assert_array_equal(first, model.transform(probe))
            self.assertEqual(first.shape, (5, 3))
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(model.transform(probe, 1).shape, (5, 1))
            for current, old in ((model, before),) + (((model.pca_, before.pca_), (model.lda_, before.lda_)) if isinstance(model, PCALDA) else ()):
                for name, value in vars(current).items():
                    if isinstance(value, np.ndarray):
                        assert_array_equal(value, getattr(old, name))
            if isinstance(model, PCA):
                model.fit(self.X)
            else:
                model.fit(self.X, self.y)
            assert_array_equal(first, model.transform(probe))

    def test_invalid_input_dimension_and_rank(self):
        for d in (0, -1, 7, 1.5, True):
            with self.assertRaises(ValueError):
                PCA(d).fit(self.X)
        with self.assertRaises(ValueError):
            LDA(4).fit(self.X, self.y)
        for ridge in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                LDA(3, ridge).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            PCA().fit(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            PCALDA(2, 3).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            PCA().transform(self.X)
        with self.assertRaises(ValueError):
            PCA().fit(self.X).transform(np.ones((2, 7)))
        with self.assertRaises(ValueError):
            LDA().fit(self.X, self.y[:-1])
        with self.assertRaises(ValueError):
            PCA().fit(np.full((3, 2), np.nan))
        rank_one = np.arange(12.0).reshape(4, 3)
        self.assertEqual(PCA().fit(rank_one).rank_, 1)
        with self.assertRaises(ValueError):
            PCA(2).fit(rank_one)
        # m <= n-C is necessary but not sufficient: actual rank is checked.
        y = np.repeat([0, 1, 2], 4)
        X = np.column_stack((np.tile(np.arange(4), 3), y))
        with self.assertRaisesRegex(ValueError, 'singular'):
            PCALDA(2, 1).fit(X, y)


class TrainingSmokeTest(unittest.TestCase):
    def test_orl_training_only(self):
        train = load_partitions(names=('train',))['train']
        X, y = train['X'], train['y']
        original = X.copy()
        pca = PCA(39).fit(X)
        direct = LDA(39, ridge=1e-3).fit(X, y)
        pipeline = PCALDA(200, 39, ridge=0).fit(X, y)
        for model in (pca, direct, pipeline):
            Z = model.transform(X)
            self.assertEqual(Z.shape, (240, 39))
            self.assertTrue(np.isfinite(Z).all())
        self.assertEqual(pca.rank_, 239)
        self.assertEqual(direct.within_rank_, 200)
        self.assertEqual(pipeline.lda_.within_rank_, 200)
        # Within-scatter metric check without ever allocating a p×p matrix.
        residual = X.astype(float) - direct.class_means_[np.searchsorted(direct.classes_, y)]
        RW = residual @ direct.scalings_
        metric = RW.T @ RW / len(X) + direct.regularization_ * (direct.scalings_.T @ direct.scalings_)
        assert_allclose(metric, np.eye(39), atol=1e-8)
        assert_array_equal(X, original)
        print(f'\nORL training only: PCA rank={pca.rank_}; direct within rank={direct.within_rank_}; '
              f'ridge lambda={direct.regularization_:.9g}; PCA(200)→LDA within rank={pipeline.lda_.within_rank_}')
        print(f'Within-scatter condition after PCA(200): {pipeline.lda_.within_condition_:.6g}; '
              f'direct regularized condition: {direct.regularized_condition_:.6g}')



if __name__ == '__main__':
    unittest.main()
