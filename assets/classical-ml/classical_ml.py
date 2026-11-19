import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


class LinearRegression:
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        center, target_mean = X.mean(axis=0), y.mean()
        self.coef_ = self._solve(X - center, y - target_mean)
        self.intercept_ = target_mean - center @ self.coef_
        return self

    def _solve(self, X, y):
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = alpha

    def _solve(self, X, y):
        d = X.shape[1]
        design = np.vstack((X, np.sqrt(self.alpha) * np.eye(d)))
        target = np.r_[y, np.zeros(d)]
        return super()._solve(design, target)


class LogisticRegression:
    def __init__(self, l2=0.02, steps=250):
        if l2 < 0 or steps < 1:
            raise ValueError("expected nonnegative l2 and at least one step")
        self.l2, self.steps = l2, steps

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, target = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("expected two classes")
        design = np.column_stack((X, np.ones(len(X))))
        weights = np.zeros(design.shape[1])
        lipschitz = np.linalg.norm(design, ord=2)**2 / (4 * len(X)) + self.l2
        for _ in range(self.steps):
            residual = expit(design @ weights) - target
            gradient = design.T @ residual / len(X)
            gradient[:-1] += self.l2 * weights[:-1]
            weights -= gradient / lipschitz
        self.coef_, self.intercept_ = weights[:-1], weights[-1]
        return self

    def predict_proba(self, X):
        p = expit(np.asarray(X) @ self.coef_ + self.intercept_)
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class GaussianNaiveBayes:
    def __init__(self, min_variance=1e-9):
        if min_variance <= 0:
            raise ValueError("min_variance must be positive")
        self.min_variance = min_variance

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        groups = [X[y == label] for label in self.classes_]
        self.mean_ = np.array([group.mean(axis=0) for group in groups])
        variances = np.array([group.var(axis=0) for group in groups])
        self.var_ = np.maximum(variances, self.min_variance)
        self.log_prior_ = np.log(counts / len(X))
        return self

    def _log_joint(self, X):
        diff = np.asarray(X)[:, None, :] - self.mean_
        terms = np.log(2 * np.pi * self.var_) + diff**2 / self.var_
        return self.log_prior_ - 0.5 * terms.sum(axis=2)

    def predict_proba(self, X):
        log_joint = self._log_joint(X)
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[self._log_joint(X).argmax(axis=1)]


class SVM:
    def __init__(self, C=3.0, kernel="rbf", gamma=1.0):
        if C <= 0 or gamma <= 0 or kernel not in {"linear", "rbf"}:
            raise ValueError("expected positive C and gamma and a linear or RBF kernel")
        self.C, self.kernel, self.gamma = C, kernel, gamma

    def _kernel(self, X, Z):
        dot = X @ Z.T
        if self.kernel == "linear":
            return dot
        distance = (X**2).sum(axis=1)[:, None] + (Z**2).sum(axis=1) - 2 * dot
        return np.exp(-self.gamma * np.maximum(distance, 0))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, target = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("expected two classes")
        signs = 2.0 * target - 1
        kernel = self._kernel(X, X)
        gram = signs[:, None] * kernel * signs

        def objective(alpha):
            product = gram @ alpha
            return 0.5 * (alpha @ product) - alpha.sum(), product - 1

        result = minimize(
            objective, np.zeros(len(X)), method="SLSQP", jac=True,
            bounds=[(0, self.C)] * len(X),
            constraints={"type": "eq", "fun": lambda a: signs @ a,
                         "jac": lambda a: signs},
            options={"ftol": 1e-9, "maxiter": 300},
        )
        if not result.success:
            raise RuntimeError(result.message)
        alpha = result.x
        tol = min(1e-6, self.C / 4)
        at_zero, at_limit = alpha <= tol, alpha >= self.C - tol
        free = ~(at_zero | at_limit)
        offsets = signs - kernel @ (alpha * signs)
        if free.any():
            self.intercept_ = offsets[free].mean()
        else:
            positive = signs > 0
            lower = (at_zero & positive) | (at_limit & ~positive)
            upper = (at_zero & ~positive) | (at_limit & positive)
            self.intercept_ = (offsets[lower].max() + offsets[upper].min()) / 2
        self.support_ = X[~at_zero]
        self.dual_coef_ = (alpha * signs)[~at_zero]
        return self

    def decision_function(self, X):
        return self._kernel(np.asarray(X, dtype=float), self.support_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]


class DecisionTree:
    def __init__(self, kind="classification", max_depth=4, min_leaf=3,
                 criterion="gini", max_features=None, seed=0, min_gain=0.0):
        if min_leaf < 1:
            raise ValueError("min_leaf must be positive")
        self.kind, self.criterion = kind, criterion
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.max_features, self.seed, self.min_gain = max_features, seed, min_gain

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        weight = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if np.any(weight < 0) or not np.any(weight > 0):
            raise ValueError("expected nonnegative weights with positive total")
        positive = weight > 0
        X, y, weight = X[positive], y[positive], weight[positive]
        if self.kind == "regression":
            y = y.astype(float)
            stats = weight[:, None] * np.column_stack((np.ones(len(y)), y, y**2))
        else:
            self.classes_, target = np.unique(y, return_inverse=True)
            stats = weight[:, None] * (target[:, None] == np.arange(len(self.classes_)))
        return self._fit_stats(X, stats)

    def _fit_stats(self, X, stats):
        self.rng_ = np.random.default_rng(self.seed)
        self.root_ = self._grow(X, stats, 0)
        return self

    def _leaf(self, stats):
        if self.kind == "regression":
            return stats[1] / stats[0]
        return stats.argmax()

    def _cost(self, stats):
        if self.kind == "regression":
            weight, target, square = np.moveaxis(stats, -1, 0)
            return np.maximum(square - target**2 / weight, 0)
        weight = stats.sum(axis=-1)
        if self.criterion == "error":
            return weight - stats.max(axis=-1)
        p = stats / weight[..., None]
        if self.criterion == "entropy":
            log_p = np.log(np.where(p > 0, p, 1))
            return -(stats * log_p).sum(axis=-1)
        return weight * (1 - (p**2).sum(axis=-1))

    def _grow(self, X, stats, depth):
        total = stats.sum(axis=0)
        if depth >= self.max_depth or len(X) < 2 * self.min_leaf:
            return self._leaf(total)
        features = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < len(features):
            features = self.rng_.choice(features, self.max_features, replace=False)
        best, best_gain = None, max(1e-12, self.min_gain)
        parent = self._cost(total)
        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            x = X[order, feature]
            sizes = np.arange(1, len(x))
            valid = (x[:-1] < x[1:]) & (sizes >= self.min_leaf) & (len(x) - sizes >= self.min_leaf)
            if not valid.any():
                continue
            ordered = stats[order]
            left = np.cumsum(ordered, axis=0)[:-1]
            right = np.cumsum(ordered[::-1], axis=0)[::-1][1:]
            gains = parent - self._cost(left) - self._cost(right)
            gains[~valid] = -np.inf
            cut = gains.argmax()
            if gains[cut] > best_gain:
                best_gain = gains[cut]
                best = feature, x[cut], x[cut + 1], order[:cut + 1], order[cut + 1:]
        if best is None:
            return self._leaf(total)
        feature, lower, upper, left, right = best
        threshold = np.clip(lower / 2 + upper / 2, lower, np.nextafter(upper, lower))
        return (feature, threshold, self._grow(X[left], stats[left], depth + 1),
                self._grow(X[right], stats[right], depth + 1))

    def predict(self, X):
        predictions = []
        for row in np.asarray(X):
            node = self.root_
            while isinstance(node, tuple):
                feature, threshold, left, right = node
                node = left if row[feature] <= threshold else right
            predictions.append(node)
        if self.kind == "regression":
            return np.asarray(predictions, dtype=float)
        return self.classes_[np.asarray(predictions, dtype=int)]


class KNN:
    def __init__(self, k=7, kind="classification"):
        self.k, self.kind = k, kind

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        if not 1 <= self.k <= len(self.X_):
            raise ValueError("k must be between 1 and the sample count")
        if self.kind == "regression":
            self.target_ = np.asarray(y, dtype=float)
        else:
            self.classes_, self.target_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        distance = ((np.asarray(X)[:, None, :] - self.X_)**2).sum(axis=-1)
        indices = np.argsort(distance, axis=1, kind="stable")[:, :self.k]
        labels = self.target_[indices]
        if self.kind == "regression":
            return labels.mean(axis=1)
        votes = (labels[:, :, None] == np.arange(len(self.classes_))).sum(axis=1)
        return self.classes_[votes.argmax(axis=1)]


class KMeans:
    def __init__(self, k=3, steps=100, tol=1e-6, seed=0):
        self.k, self.steps, self.tol, self.seed = k, steps, tol, seed

    def fit(self, X, init=None):
        X = np.asarray(X, dtype=float)
        if init is None:
            indices = np.random.default_rng(self.seed).choice(len(X), self.k, replace=False)
            init = X[indices]
        self.centers_ = np.array(init, dtype=float, copy=True)
        if self.k < 1 or self.centers_.shape != (self.k, X.shape[1]):
            raise ValueError("expected a nonempty array of k initial centers")
        self.history_ = [self.centers_.copy()]
        for _ in range(self.steps):
            assignment = self.predict(X)
            updated = self.centers_.copy()
            for k in range(self.k):
                cluster = X[assignment == k]
                if len(cluster):
                    updated[k] = cluster.mean(axis=0)
            change = np.max(np.linalg.norm(updated - self.centers_, axis=1))
            self.centers_ = updated
            self.history_.append(updated.copy())
            if change <= self.tol:
                break
        return self

    def predict(self, X):
        distance = ((np.asarray(X)[:, None, :] - self.centers_)**2).sum(axis=-1)
        return distance.argmin(axis=1)


class PCA:
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if not 1 <= self.n_components <= min(X.shape):
            raise ValueError("n_components must fit the reduced SVD")
        self.mean_ = X.mean(axis=0)
        _, values, right = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = right[:self.n_components]
        self.variance_ = values[:self.n_components]**2 / len(X)
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return np.asarray(Z) @ self.components_ + self.mean_


class GaussianMixture:
    def __init__(self, k=3, steps=100, tol=1e-6, min_covar=1e-3, seed=0):
        if k < 1 or min_covar <= 0:
            raise ValueError("k and min_covar must be positive")
        self.k, self.steps, self.tol = k, steps, tol
        self.min_covar, self.seed = min_covar, seed

    def _stabilize(self, cov):
        values, vectors = np.linalg.eigh(cov)
        return (vectors * np.maximum(values, self.min_covar)) @ vectors.T

    def _log_joint(self, X):
        log_joint = np.full((len(X), self.k), -np.inf)
        for k in np.flatnonzero(self.weights_ > 0):
            chol = np.linalg.cholesky(self.covariances_[k])
            whitened = np.linalg.solve(chol, (X - self.means_[k]).T)
            log_normalizer = X.shape[1] * np.log(2 * np.pi) + 2 * np.log(chol.diagonal()).sum()
            log_joint[:, k] = np.log(self.weights_[k]) - 0.5 * (log_normalizer + (whitened**2).sum(axis=0))
        return log_joint

    def _snapshot(self):
        return self.weights_.copy(), self.means_.copy(), self.covariances_.copy()

    def fit(self, X, init=None):
        X = np.asarray(X, dtype=float)
        if init is None:
            indices = np.random.default_rng(self.seed).choice(len(X), self.k, replace=False)
            init = X[indices]
        self.means_ = np.array(init, dtype=float, copy=True)
        if self.means_.shape != (self.k, X.shape[1]):
            raise ValueError("expected an array of k initial centers")
        self.weights_ = np.full(self.k, 1 / self.k)
        centered = X - X.mean(axis=0)
        covariance = self._stabilize(centered.T @ centered / len(X))
        self.covariances_ = np.repeat(covariance[None], self.k, axis=0)
        self.history_ = [self._snapshot()]
        log_joint = self._log_joint(X)
        log_density = logsumexp(log_joint, axis=1, keepdims=True)
        for _ in range(self.steps):
            responsibility = np.exp(log_joint - log_density)
            mass = responsibility.sum(axis=0)
            active = np.flatnonzero(mass > 0)
            self.weights_ = mass / mass.sum()
            for k in active:
                weight = responsibility[:, k]
                self.means_[k] = weight @ X / mass[k]
                diff = X - self.means_[k]
                covariance = (diff.T * weight) @ diff / mass[k]
                self.covariances_[k] = self._stabilize(covariance)
            self.history_.append(self._snapshot())
            log_joint = self._log_joint(X)
            updated = logsumexp(log_joint, axis=1, keepdims=True)
            improvement = (updated - log_density).mean()
            log_density = updated
            if improvement <= self.tol:
                break
        return self

    def predict_proba(self, X):
        log_joint = self._log_joint(np.asarray(X, dtype=float))
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class RandomForest:
    def __init__(self, n_estimators=30, kind="classification", max_depth=5,
                 min_leaf=3, max_features=None, seed=0):
        self.n_estimators, self.kind = n_estimators, kind
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.max_features, self.seed = max_features, seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y) if self.kind == "classification" else None
        features = self.max_features or max(1, int(np.sqrt(X.shape[1])))
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            rows = rng.integers(len(X), size=len(X))
            tree = DecisionTree(kind=self.kind, max_depth=self.max_depth,
                                min_leaf=self.min_leaf, max_features=features,
                                seed=rng.integers(2**32))
            self.trees_.append(tree.fit(X[rows], y[rows]))
        return self

    def predict(self, X):
        votes = np.stack([tree.predict(X) for tree in self.trees_])
        if self.kind == "regression":
            return votes.mean(axis=0)
        counts = (votes[:, :, None] == self.classes_).sum(axis=0)
        return self.classes_[counts.argmax(axis=1)]


class AdaBoost:
    def __init__(self, n_estimators=35):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, target = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("expected two classes")
        signs = 2 * target - 1
        weights = np.full(len(X), 1 / len(X))
        self.rounds_ = []
        for _ in range(self.n_estimators):
            stump = DecisionTree(max_depth=1, min_leaf=1, criterion="error")
            tree = stump.fit(X, signs, weights)
            prediction = tree.predict(X)
            error = weights[prediction != signs].sum()
            if error >= 0.5 - 1e-12:
                break
            safe_error = max(error, 1e-12)
            vote = 0.5 * np.log((1 - safe_error) / safe_error)
            self.rounds_.append((vote, tree))
            if error == 0:
                break
            weights *= np.exp(-vote * signs * prediction)
            weights /= weights.sum()
        return self

    def decision_function(self, X):
        score = np.zeros(len(X))
        for vote, tree in self.rounds_:
            score += vote * tree.predict(X)
        return score

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class GradientBoosting:
    def __init__(self, n_estimators=40, learning_rate=0.13, max_depth=2, min_leaf=4):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_depth, self.min_leaf = max_depth, min_leaf

    def _fit_tree(self, X, residual):
        tree = DecisionTree(kind="regression", max_depth=self.max_depth, min_leaf=self.min_leaf)
        return tree.fit(X, residual)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.base_ = y.mean()
        self.trees_ = []
        prediction = np.full(len(X), self.base_)
        for _ in range(self.n_estimators):
            tree = self._fit_tree(X, y - prediction)
            prediction += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        prediction = np.full(len(X), self.base_)
        for tree in self.trees_:
            prediction += self.learning_rate * tree.predict(X)
        return prediction


class XGBoost(GradientBoosting):
    class Tree(DecisionTree):
        def __init__(self, max_depth, min_leaf, l2, gamma):
            super().__init__(kind="regression", max_depth=max_depth,
                             min_leaf=min_leaf, min_gain=gamma)
            self.l2 = l2

        def _leaf(self, stats):
            gradient, hessian = stats
            return -gradient / (hessian + self.l2)

        def _cost(self, stats):
            gradient, hessian = np.moveaxis(stats, -1, 0)
            return -0.5 * gradient**2 / (hessian + self.l2)

    def __init__(self, n_estimators=40, learning_rate=0.13, max_depth=2,
                 min_leaf=4, l2=2.0, gamma=0.02):
        super().__init__(n_estimators, learning_rate, max_depth, min_leaf)
        if l2 < 0 or gamma < 0:
            raise ValueError("l2 and gamma must be nonnegative")
        self.l2, self.gamma = l2, gamma

    def _fit_tree(self, X, residual):
        tree = self.Tree(self.max_depth, self.min_leaf, self.l2, self.gamma)
        stats = np.column_stack((-residual, np.ones(len(residual))))
        return tree._fit_stats(X, stats)
