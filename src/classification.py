"""Euclidean 1-nearest neighbor with deterministic training-row tie breaking."""

import numpy as np


def _features(X):
    if np.iscomplexobj(X):
        raise ValueError("Expected real features")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite 2D feature matrix")
    return X


def _labels(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("Expected a 1D label vector aligned with samples")
    if y.dtype.kind in 'fc' and not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    return y


def classification_metrics(y_true, y_pred):
    """Return accuracy/error as fractions and integer prediction counts."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("Expected nonempty 1D true labels")
    y_true = _labels(y_true, len(y_true))
    y_pred = _labels(y_pred, len(y_true))
    correct = int(np.count_nonzero(y_true == y_pred))
    return dict(accuracy=correct / len(y_true), error_rate=(len(y_true) - correct) / len(y_true),
                n_correct=correct, n_samples=len(y_true))


class Euclidean1NN:
    """Store training observations and select minimum squared distance.

    Exact equal-distance ties select the earliest row supplied to fit. Squared
    differences are summed directly in float64 to avoid the cancellation in
    ||a||² + ||b||² - 2a.b. Query rows are processed individually, bounding
    working memory to one training-sized difference matrix.
    """

    def fit(self, X, y):
        X = _features(X)
        y = _labels(y, len(X))
        self.X_, self.y_ = X.copy(), y.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, 'X_'):
            raise ValueError("Fit 1-NN before predicting")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training data")
        indices = np.empty(len(X), dtype=np.int64)
        for i, row in enumerate(X):
            with np.errstate(over='raise', invalid='raise'):
                try:
                    difference = self.X_ - row
                    distances = np.einsum('ij,ij->i', difference, difference)
                except FloatingPointError as exc:
                    raise ValueError("Squared distances overflow; check feature magnitudes") from exc
            if not np.isfinite(distances).all():
                raise ValueError("Squared distances are not finite")
            indices[i] = np.argmin(distances)
        return self.y_[indices]

    def score(self, X, y):
        return classification_metrics(y, self.predict(X))['accuracy']
