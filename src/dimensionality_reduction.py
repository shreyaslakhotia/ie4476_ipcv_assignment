"""Training-fitted PCA, weighted multiclass SPCA, LDA, and sequential PCA→LDA.

All algebra uses float64. No estimator selects settings or accesses partitions.
"""

import hashlib
import numbers
import numpy as np


def _matrix(X):
    if np.iscomplexobj(X):
        raise ValueError("Expected real features")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite 2D feature matrix")
    return X


def _dimension(d, maximum):
    if d is None:
        return maximum
    if isinstance(d, bool) or not isinstance(d, numbers.Integral) or not 1 <= d <= maximum:
        raise ValueError(f"n_components must be an integer in [1, {maximum}], got {d}")
    return int(d)


def _rank(s, shape):
    tolerance = np.finfo(np.float64).eps * max(shape) * s[0] if len(s) else 0.0
    return int(np.count_nonzero(s > tolerance)), tolerance


def _sign_rows(vectors):
    """Fix arbitrary signs by making the largest-magnitude entry positive."""
    signs = np.sign(vectors[np.arange(len(vectors)), np.argmax(np.abs(vectors), axis=1)])
    return vectors * signs[:, None]


def _training_digest(X):
    """Bind optional decomposition reuse to the exact training observations."""
    return hashlib.sha256(np.asarray(X, dtype='<f8').tobytes()).hexdigest()


class PCA:
    """Economy SVD PCA without whitening; keep the full positive-rank basis.

    n_components selects the default transform size. transform(X, d) reuses
    the same fitted decomposition for another d without fitting again.
    components_, explained_variance_, explained_variance_ratio_ and
    cumulative_explained_variance_ contain the full numerical-rank spectrum.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = _matrix(X)
        if len(X) < 2:
            raise ValueError("PCA requires at least two training samples")
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        rank, tolerance = _rank(s, X.shape)
        if rank == 0:
            raise ValueError("Training data have no numerically positive variance")
        d = _dimension(self.n_components, rank)
        self.mean_, self.n_features_in_ = mean, X.shape[1]
        self.training_digest_ = _training_digest(X)
        self.rank_, self.rank_tolerance_, self.n_components_ = rank, tolerance, d
        self.components_ = _sign_rows(vt[:rank])
        self.singular_values_ = s[:rank]
        self.explained_variance_ = s[:rank] ** 2 / (len(X) - 1)
        self.explained_variance_ratio_ = s[:rank] ** 2 / np.sum(s ** 2)
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X, n_components=None):
        if not hasattr(self, 'components_'):
            raise ValueError("Fit PCA before transforming")
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training data")
        d = self.n_components_ if n_components is None else _dimension(n_components, self.rank_)
        return (X - self.mean_) @ self.components_[:d].T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SPCA:
    """Specified multiclass weighted-covariance supervised PCA variant.

    Maximize trace(W.T @ (Sw + eta*Sb) @ W), subject to W.T @ W = I.
    Class priors are n_c/n; class covariance uses divisor n_c (not n_c-1),
    giving Sw=R.T@R/n and Sb=B@B.T, where R is class-centered data and
    B[:,c]=sqrt(n_c/n)*(mean_c-global_mean). Thus eta=1 recovers total
    population covariance. PCA's sample-covariance eigenvalues differ only
    by n/(n-1); its eigenspaces/projections are identical.

    Economy SVD of [R/sqrt(n); sqrt(eta)*B.T] avoids a pixel-space matrix.
    eta must be finite and nonnegative. Eigenvalues/components retain all
    positive-rank directions; transform(X,d) reuses the fitted decomposition.
    No whitening. This is not a claim to reproduce Jiang's full algorithms.
    """

    def __init__(self, n_components=None, eta=1.0):
        self.n_components, self.eta = n_components, eta

    def fit(self, X, y):
        X = _matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('Labels must be a 1D vector aligned with training samples')
        if y.dtype.kind in 'fc' and not np.isfinite(y).all():
            raise ValueError('Labels must be finite')
        if isinstance(self.eta, bool) or not isinstance(self.eta, numbers.Real) or not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError('eta must be finite and nonnegative')
        classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        if len(classes) < 2:
            raise ValueError('SPCA requires at least two classes')
        mean = X.mean(axis=0)
        means = np.asarray([X[inverse == c].mean(axis=0) for c in range(len(classes))])
        residual = X - means[inverse]
        B = (means - mean).T * np.sqrt(counts / len(X))
        with np.errstate(over='raise', invalid='raise'):
            try:
                augmented = np.vstack((residual / np.sqrt(len(X)), np.sqrt(self.eta) * B.T))
            except FloatingPointError as exc:
                raise ValueError('SPCA scatter scale overflow; check eta/features') from exc
        _, s, vt = np.linalg.svd(augmented, full_matrices=False)
        rank, tolerance = _rank(s, augmented.shape)
        if rank == 0:
            raise ValueError('SPCA scatter has no numerically positive directions')
        d = _dimension(self.n_components, rank)
        with np.errstate(over='raise', invalid='raise'):
            try:
                eigenvalues = s[:rank] ** 2
            except FloatingPointError as exc:
                raise ValueError('SPCA eigenvalue overflow; check eta/features') from exc
        self.mean_, self.class_means_, self.classes_ = mean, means, classes
        self.class_priors_, self.eta_ = counts / len(X), float(self.eta)
        self.n_features_in_, self.n_components_ = X.shape[1], d
        self.rank_, self.rank_tolerance_ = rank, tolerance
        self.components_, self.eigenvalues_ = _sign_rows(vt[:rank]), eigenvalues
        return self

    def transform(self, X, n_components=None):
        if not hasattr(self, 'components_'):
            raise ValueError('Fit SPCA before transforming')
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from training data')
        d = self.n_components_ if n_components is None else _dimension(n_components, self.rank_)
        return (X - self.mean_) @ self.components_[:d].T

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class LDA:
    """Fisher directions solving Sb w = eigenvalue * (Sw + lambda I) w.

    Sw = R.T R / n, R = class-centered training observations.
    Sb = B B.T, B[:,c] = sqrt(n_c/n) * (mean_c - global_mean).
    lambda = ridge * trace(Sw) / p, default ridge=1e-3, fixed a priori.

    SVD applies (Sw + lambda I)^(-1/2) without forming a p×p scatter matrix.
    Crucially, ridge>0 retains the within-scatter nullspace; this is direct
    regularized LDA, not a preliminary PCA truncation. ridge=0 requires full
    within-class column rank and otherwise raises an explicit error.

    scalings_ columns are metric-normalized: W.T (Sw + lambda I) W = I.
    The full positive discriminant spectrum is retained for reuse at other d.
    """

    def __init__(self, n_components=None, ridge=1e-3):
        self.n_components, self.ridge = n_components, ridge

    def fit(self, X, y):
        X = _matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Labels must be a 1D vector aligned with training samples")
        if y.dtype.kind in 'fc' and not np.isfinite(y).all():
            raise ValueError("Labels must be finite")
        classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        if len(classes) < 2 or len(X) <= len(classes):
            raise ValueError("Need at least two classes and positive within-class degrees of freedom")
        if not isinstance(self.ridge, numbers.Real) or not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        n, p = X.shape
        mean = X.mean(axis=0)
        means = np.asarray([X[inverse == c].mean(axis=0) for c in range(len(classes))])
        residual = X - means[inverse]
        _, s, vt = np.linalg.svd(residual, full_matrices=False)
        rank, tolerance = _rank(s, residual.shape)
        if self.ridge == 0 and rank < p:
            raise ValueError(f"Unregularized within-class scatter is singular: rank {rank} < {p}; "
                             "reduce PCA dimension or set an explicit positive ridge")
        scale = np.sum(s ** 2) / (n * p)
        if scale <= 0:
            raise ValueError("Zero within-class scatter; relative ridge scale is undefined")
        regularization = float(self.ridge * scale)
        if self.ridge > 0 and regularization == 0:
            raise ValueError("ridge underflow: choose a larger positive value")
        basis = vt[:rank]
        denominators = np.sqrt(s[:rank] ** 2 / n + regularization)

        def inverse_sqrt(A):
            projected = basis @ A
            result = basis.T @ (projected / denominators[:, None])
            if regularization > 0 and rank < p:
                result += (A - basis.T @ projected) / np.sqrt(regularization)
            return result

        B = (means - mean).T * np.sqrt(counts / n)
        whitened = inverse_sqrt(B)
        u, discriminant_s, _ = np.linalg.svd(whitened, full_matrices=False)
        between_rank, between_tolerance = _rank(discriminant_s, whitened.shape)
        maximum = min(len(classes) - 1, between_rank, p)
        if maximum == 0:
            raise ValueError("Class means have no numerically distinct discriminant directions")
        d = _dimension(self.n_components, maximum)
        directions = _sign_rows(inverse_sqrt(u[:, :maximum]).T).T
        if not np.isfinite(directions).all():
            raise ValueError("Nonfinite LDA directions; check feature scale and ridge")
        self.mean_, self.class_means_, self.classes_ = mean, means, classes
        self.n_features_in_, self.n_components_, self.max_components_ = p, d, maximum
        self.within_rank_, self.within_rank_tolerance_ = rank, tolerance
        self.between_rank_tolerance_ = between_tolerance
        self.regularization_, self.within_scale_ = regularization, float(scale)
        smallest = s[p-1] ** 2 / n if rank == p else 0.0
        self.within_condition_ = float((s[0] ** 2 / n) / smallest) if smallest > 0 else np.inf
        self.regularized_condition_ = float((s[0] ** 2 / n + regularization) / (smallest + regularization))
        self.scalings_, self.eigenvalues_ = directions, discriminant_s[:maximum] ** 2
        return self

    def transform(self, X, n_components=None):
        if not hasattr(self, 'scalings_'):
            raise ValueError("Fit LDA before transforming")
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training data")
        d = self.n_components_ if n_components is None else _dimension(n_components, self.max_components_)
        return (X - self.mean_) @ self.scalings_[:, :d]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class PCALDA:
    """Explicit PCA(m) followed by LDA(d), unregularized by default.

    Actual projected within-class rank is checked by LDA, not assumed from
    m <= n-C. Positive ridge may be requested explicitly for a regularized
    two-stage variant; it must be reported as such in subsequent comparisons.
    """

    def __init__(self, pca_components=200, n_components=39, ridge=0.0):
        self.pca_components = pca_components
        self.n_components, self.ridge = n_components, ridge

    def fit(self, X, y, *, fitted_pca=None):
        """Optionally reuse PCA fitted to these exact training observations."""
        if self.pca_components is None:
            raise ValueError("Specify the intermediate PCA dimension explicitly")
        X = _matrix(X)
        if fitted_pca is None:
            pca = PCA(self.pca_components).fit(X)
        else:
            pca = fitted_pca
            if not isinstance(pca, PCA) or not hasattr(pca, 'components_'):
                raise ValueError('fitted_pca must be a fitted PCA instance')
            if pca.n_features_in_ != X.shape[1] or pca.training_digest_ != _training_digest(X):
                raise ValueError('Reused PCA must have been fitted to exactly these training features')
        m = _dimension(self.pca_components, pca.rank_)
        Z = pca.transform(X, m)
        lda = LDA(self.n_components, self.ridge).fit(Z, y)
        self.pca_, self.lda_ = pca, lda
        self.pca_components_ = m
        self.n_components_, self.n_features_in_ = lda.n_components_, pca.n_features_in_
        return self

    def transform(self, X, n_components=None):
        if not hasattr(self, 'pca_'):
            raise ValueError("Fit PCA→LDA before transforming")
        return self.lda_.transform(self.pca_.transform(X, self.pca_components_), n_components)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
