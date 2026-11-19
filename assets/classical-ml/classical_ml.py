import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal


def majority(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[counts.argmax()]


def splits(X, features):
    for j in features:
        values = np.unique(X[:, j])
        for t in values[:-1] / 2 + values[1:] / 2:
            left = X[:, j] <= t
            if left.any() and not left.all():
                yield j, t, left


class LinearRegression:
    def __init__(self, l2=0.0):
        self.l2 = l2

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        penalty = np.sqrt(self.l2) * np.eye(X.shape[1])
        A = np.vstack([X - x_mean, penalty])
        z = np.r_[y - y_mean, np.zeros(X.shape[1])]
        self.w = np.linalg.lstsq(A, z, rcond=None)[0]
        self.b = y_mean - x_mean @ self.w
        return self

    def predict(self, X):
        return np.asarray(X) @ self.w + self.b


class RidgeRegression(LinearRegression):
    def __init__(self, l2=1.0):
        super().__init__(l2=l2)


class LogisticRegression:
    def fit(self, X, y, steps=300):
        A = np.c_[X, np.ones(len(X))]
        y = np.asarray(y)
        self.w = np.zeros(A.shape[1])
        step = 4 / np.linalg.norm(A, ord=2)**2
        for _ in range(steps):
            self.w -= step * (A.T @ (expit(A @ self.w) - y))
        return self

    def predict_proba(self, X):
        return expit(np.asarray(X) @ self.w[:-1] + self.w[-1])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class GaussianNaiveBayes:
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes, counts = np.unique(y, return_counts=True)
        groups = [X[y == c] for c in self.classes]
        self.mean = np.array([group.mean(axis=0) for group in groups])
        self.var = np.maximum([group.var(axis=0) for group in groups], 1e-9)
        self.log_prior = np.log(counts / len(y))
        return self

    def predict(self, X):
        diff = np.asarray(X)[:, None, :] - self.mean
        log_joint = self.log_prior - 0.5 * (np.log(self.var) + diff**2 / self.var).sum(axis=2)
        return self.classes[log_joint.argmax(axis=1)]


class SVM:
    def fit(self, X, y, l2=0.05, steps=600, lr=0.5):
        X, y = np.asarray(X, float), np.asarray(y)
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for t in range(steps):
            active = y * (X @ self.w + self.b) < 1
            signed = y * active
            eta = lr / np.sqrt(t + 1)
            self.w -= eta * (l2 * self.w - X.T @ signed / len(y))
            self.b += eta * signed.mean()
        return self

    def predict(self, X):
        return np.where(np.asarray(X) @ self.w + self.b >= 0, 1, -1)


class DecisionTree:
    def __init__(self, depth=3, regression=False, max_features=None, seed=0):
        self.depth, self.regression = depth, regression
        self.max_features, self.seed = max_features, seed

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        if not self.regression:
            self.classes, y = np.unique(y, return_inverse=True)
        self.rng = np.random.default_rng(self.seed)
        self.root = self._grow(X, y, weights, self.depth)
        return self

    def _leaf(self, y, w):
        if self.regression:
            return np.average(y, weights=w)
        return np.bincount(y, weights=w).argmax()

    def _loss(self, y, w):
        if self.regression:
            return w @ (y - self._leaf(y, w))**2
        p = np.bincount(y, weights=w) / w.sum()
        return w.sum() * (1 - p @ p)

    def _grow(self, X, y, w, depth):
        leaf = self._leaf(y, w)
        if depth == 0 or np.all(y == y[0]):
            return leaf
        d = X.shape[1]
        features = range(d)
        if self.max_features is not None:
            features = self.rng.choice(d, self.max_features, replace=False)
        best, loss = None, self._loss(y, w)
        for j, t, left in splits(X, features):
            right = ~left
            total = self._loss(y[left], w[left]) + self._loss(y[right], w[right])
            if total < loss - 1e-12:
                best, loss = (j, t, left), total
        if best is None:
            return leaf
        j, t, left = best
        right = ~left
        return (j, t, self._grow(X[left], y[left], w[left], depth - 1),
                self._grow(X[right], y[right], w[right], depth - 1))

    def predict(self, X):
        output = []
        for x in X:
            node = self.root
            while isinstance(node, tuple):
                j, t, left, right = node
                node = left if x[j] <= t else right
            output.append(node)
        if self.regression:
            return np.asarray(output, float)
        return self.classes[np.asarray(output, int)]


class KNN:
    def __init__(self, k=5, regression=False):
        self.k, self.regression = k, regression

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        return self

    def predict(self, X):
        d2 = ((np.asarray(X)[:, None, :] - self.X)**2).sum(axis=2)
        labels = self.y[np.argsort(d2, axis=1, kind="stable")[:, :self.k]]
        if self.regression:
            return labels.mean(axis=1)
        return np.array([majority(row) for row in labels])


class KMeans:
    def __init__(self, k=3, steps=50, seed=0):
        self.k, self.steps, self.seed = k, steps, seed

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        self.centers = X[rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.steps):
            labels = self.predict(X)
            for j in range(self.k):
                cluster = X[labels == j]
                if len(cluster):
                    self.centers[j] = cluster.mean(axis=0)
        return self

    def predict(self, X):
        return ((np.asarray(X)[:, None, :] - self.centers)**2).sum(axis=2).argmin(axis=1)


class PCA:
    def fit(self, X, k=1):
        X = np.asarray(X, float)
        self.mean = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.components = Vt[:k]
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return np.asarray(Z) @ self.components + self.mean


class GaussianMixture:
    def __init__(self, k=3, steps=40, seed=0):
        self.k, self.steps, self.seed = k, steps, seed

    def fit(self, X):
        X = np.asarray(X, float)
        eye = np.eye(X.shape[1])
        self.mean = KMeans(self.k, seed=self.seed).fit(X).centers
        self.cov = np.repeat(eye[None], self.k, axis=0)
        self.weights = np.full(self.k, 1 / self.k)
        for _ in range(self.steps):
            r = self.predict_proba(X)
            mass = np.maximum(r.sum(axis=0), 1e-12)
            self.weights = mass / mass.sum()
            self.mean = r.T @ X / mass[:, None]
            for j in range(self.k):
                diff = X - self.mean[j]
                self.cov[j] = (diff.T * r[:, j]) @ diff / mass[j] + 1e-6 * eye
        return self

    def predict_proba(self, X):
        log_p = np.column_stack([multivariate_normal.logpdf(X, m, c)
                                 for m, c in zip(self.mean, self.cov)])
        log_p += np.log(self.weights)
        return np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class RandomForest:
    def __init__(self, trees=30, depth=4, regression=False, seed=0):
        self.trees, self.depth = trees, depth
        self.regression, self.seed = regression, seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        m = max(1, int(np.sqrt(X.shape[1])))
        self.models = []
        for _ in range(self.trees):
            rows = rng.integers(len(X), size=len(X))
            tree = DecisionTree(self.depth, self.regression, m, seed=rng.integers(2**32))
            self.models.append(tree.fit(X[rows], y[rows]))
        return self

    def predict(self, X):
        votes = np.column_stack([tree.predict(X) for tree in self.models])
        if self.regression:
            return votes.mean(axis=1)
        return np.array([majority(row) for row in votes])


class AdaBoost:
    def fit(self, X, y, rounds=30):
        y = np.asarray(y)
        w = np.full(len(y), 1 / len(y))
        self.models = []
        for _ in range(rounds):
            stump = DecisionTree(depth=1).fit(X, y, sample_weight=w)
            h = stump.predict(X)
            error = w[h != y].sum()
            if error >= 0.5:
                break
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-12))
            self.models.append((alpha, stump))
            if error == 0:
                break
            w *= np.exp(-alpha * y * h)
            w /= w.sum()
        return self

    def predict(self, X):
        score = sum((a * h.predict(X) for a, h in self.models), np.zeros(len(X)))
        return np.where(score >= 0, 1, -1)


class GradientBoosting:
    def __init__(self, trees=30, lr=0.1, depth=2):
        self.trees, self.lr, self.depth = trees, lr, depth

    def _tree(self, X, residual):
        return DecisionTree(self.depth, regression=True).fit(X, residual).predict

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.base, self.models = y.mean(), []
        prediction = np.full(len(y), self.base)
        for _ in range(self.trees):
            tree = self._tree(X, y - prediction)
            prediction += self.lr * tree(X)
            self.models.append(tree)
        return self

    def predict(self, X):
        return self.base + self.lr * sum((tree(X) for tree in self.models), np.zeros(len(X)))


class XGBoost(GradientBoosting):
    def __init__(self, trees=30, lr=0.1, l2=1.0, gamma=0.0):
        super().__init__(trees=trees, lr=lr)
        self.l2, self.gamma = l2, gamma

    def _tree(self, X, residual):
        def value(r):
            return r.sum() / (len(r) + self.l2)

        def score(r):
            return r.sum() * value(r)

        best = (0, np.inf, value(residual), 0.0)
        parent, gain = score(residual), 0.0
        for j, t, left in splits(X, range(X.shape[1])):
            a, b = residual[left], residual[~left]
            improvement = 0.5 * (score(a) + score(b) - parent) - self.gamma
            if improvement > gain:
                best, gain = (j, t, value(a), value(b)), improvement
        j, t, a, b = best
        return lambda Z: np.where(np.asarray(Z)[:, j] <= t, a, b)
