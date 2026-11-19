---
title: "classical ML"
category: "Machine Learning"
order: 1
description: "Short derivations and Python implementations of classical machine learning algorithms."
---

In supervised learning, we observe independent training pairs $$(x_i,y_i)$$ from an unknown distribution. Regression predicts a real value and classification predicts a label. Clustering and dimensionality reduction use the inputs alone. [Rosasco, Chapter 1](https://lcsl.unige.it/courses/ml/1617/MLNotes.pdf#page=5)

$$
R(f)=\mathbb E[\ell(y,f(x))],\qquad
\widehat R(f)=\frac1n\sum_{i=1}^n\ell(y_i,f(x_i)).
$$

The loss $$\ell$$ measures prediction error. We fit on training data and use held-out data to estimate performance on new samples.

* TOC
{:toc}

The examples use NumPy and SciPy. Each row of $$X$$ is one sample. [Download the Python source]({{ '/assets/classical-ml/classical_ml.py' | relative_url }})

~~~python
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
~~~

## Supervised learning

### Linear and ridge regression

Fit $$y_i=w^\top x_i+b+\epsilon_i$$ by least squares, which is maximum likelihood under independent Gaussian noise of fixed variance. Ridge adds a squared penalty to $$w$$.

![Noisy observations around a linear function]({{ '/assets/classical-ml/linear-regression.png' | relative_url }}){: width="420"}

Noisy samples around a line. [Rosasco, Chapter 1, Figure 2](https://lcsl.unige.it/courses/ml/1617/MLNotes.pdf#page=6)

$$
\begin{aligned}
L(w,b)&=\tfrac12\|Xw+b\mathbf1-y\|^2+\tfrac{\lambda}{2}\|w\|^2,\\
\partial_b L=0&\implies b=\bar y-\bar x^\top w,\\
\nabla_w L=0&\implies (X_c^\top X_c+\lambda I)w=X_c^\top y_c.
\end{aligned}
$$

Here $$X_c$$ and $$y_c$$ are centered. Ordinary least squares uses $$\lambda=0$$.

~~~python
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
~~~

~~~python
class RidgeRegression(LinearRegression):
    def __init__(self, l2=1.0):
        super().__init__(l2=l2)
~~~

### Logistic regression

For labels $$y_i\in\{0,1\}$$, fit a Bernoulli model with $$p_i=\sigma(w^\top x_i)$$, where $$x_i$$ includes a constant feature for the intercept.

$$
\begin{aligned}
z_i &= w^\top x_i,\qquad p_i=\frac{1}{1+e^{-z_i}},\\
\ell_i &= -y_i\ln p_i-(1-y_i)\ln(1-p_i)
        =\ln(1+e^{z_i})-y_i z_i,\\
\partial_{z_i}\ell_i&=p_i-y_i,\qquad
w\leftarrow w-\eta\sum_i(p_i-y_i)x_i.
\end{aligned}
$$

~~~python
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
~~~

### Gaussian naive Bayes

Assume the features are independent given the class. Fit a Gaussian to each class-feature pair and predict the class with the largest score.

$$
\begin{aligned}
p(c\mid x)&\propto \pi_c\prod_j \mathcal N(x_j;\mu_{cj},v_{cj}),\\
\partial_{\mu_{cj}}\log p(X\mid y)
&=\frac1{v_{cj}}\sum_{i:y_i=c}(x_{ij}-\mu_{cj}),\\
\hat\pi_c&=\frac{n_c}{n},\qquad \hat\mu_{cj}=\frac1{n_c}\sum_{i:y_i=c}x_{ij},\\
\hat v_{cj}&=\frac1{n_c}\sum_{i:y_i=c}(x_{ij}-\hat\mu_{cj})^2.
\end{aligned}
$$

~~~python
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
~~~

### Support vector machine

For $$y_i\in\{-1,1\}$$, set $$f_i=w^\top x_i+b$$. On separable data we can scale $$w,b$$ to make $$\min_i y_i f_i=1$$, giving a width $$2/\|w\|$$ between the two margin planes.

![A decision boundary and its two margin lines]({{ '/assets/classical-ml/svm-margin.png' | relative_url }}){: width="370"}

The circled points are support vectors. [Rosasco, Chapter 9, Figure 1](https://lcsl.unige.it/courses/ml/1617/MLNotes.pdf#page=38)

$$
\begin{aligned}
\text{Hard margin}\quad &\min_{w,b}\ \tfrac12\|w\|^2
                     \quad\text{subject to }y_i f_i\ge1,\\
\text{Soft margin}\quad &y_i f_i\ge1-\xi_i,\qquad
                  \xi_i^*=\max(0,1-y_i f_i),\\
L(w,b)&=\tfrac{\lambda}{2}\|w\|^2+\tfrac1n\sum_i\max(0,1-y_i f_i).
\end{aligned}
$$

The linear implementation updates on points with $$y_i f_i<1$$. A kernel SVM uses inner products $$K(x,z)=\phi(x)^\top\phi(z)$$ in feature space.

~~~python
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
~~~

### Decision tree

Try thresholds on each feature and keep the split with the smallest total child loss. A leaf predicts its majority class or weighted mean.

$$
\begin{aligned}
G(p)&=1-\sum_c p_c^2,\qquad H(p)=-\sum_c p_c\ln p_c,\\
L_{\rm cls}(D)&=\Big(\sum_{i\in D}a_i\Big)G(p_D),\\
a^*&=\arg\min_a\sum_{i\in D}a_i(y_i-a)^2
    =\frac{\sum_{i\in D}a_i y_i}{\sum_{i\in D}a_i},\\
L_{\rm reg}(D)&=\sum_{i\in D}a_i(y_i-a^*)^2.
\end{aligned}
$$

Here $$a_i$$ are sample weights. With entropy, the reduction in average impurity is $$H(Y)-H(Y\mid S)$$ for split result $$S$$.

~~~python
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
~~~

### K-nearest neighbors

Find the $$k$$ closest points $$N_k(x)$$. Their vote minimizes local classification error and their mean minimizes local squared error.

$$
\begin{aligned}
\hat y_{\rm cls}(x)&=\arg\max_c\sum_{i\in N_k(x)}\mathbf1_{\{y_i=c\}},\\
\hat y_{\rm reg}(x)&=\arg\min_a\sum_{i\in N_k(x)}(y_i-a)^2
                    =\frac1k\sum_{i\in N_k(x)}y_i.
\end{aligned}
$$

![A jagged one-nearest-neighbor decision boundary]({{ '/assets/classical-ml/nearest-neighbor.png' | relative_url }}){: width="480"}

The decision boundary for one nearest neighbor. [Rosasco, Chapter 2, Figure 1](https://lcsl.unige.it/courses/ml/1617/MLNotes.pdf#page=10)

~~~python
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
~~~

## Unsupervised learning

### K-means

Alternate between assigning points to their nearest center and updating each center to its cluster mean.

$$
\begin{aligned}
J(c,\mu)&=\sum_i\|x_i-\mu_{c_i}\|^2,\\
c_i&\leftarrow\arg\min_j\|x_i-\mu_j\|^2,\\
\nabla_{\mu_j}J&=2\sum_{i:c_i=j}(\mu_j-x_i)=0
\quad\Rightarrow\quad \mu_j=\frac1{n_j}\sum_{i:c_i=j}x_i.
\end{aligned}
$$

~~~python
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
~~~

### Principal component analysis

Center the data and project onto the directions of largest variance. The code gets these directions from the SVD.

$$
\begin{aligned}
\Sigma&=X_c^\top X_c/n,\qquad \operatorname{Var}(u^\top x)=u^\top\Sigma u,\\
\nabla_u[u^\top\Sigma u-\lambda(u^\top u-1)]=0
 &\implies\Sigma u=\lambda u,\\
X_c=USV^\top&\implies \Sigma=VS^\top S V^\top/n.
\end{aligned}
$$

The first principal direction has the largest eigenvalue. For a unit vector $$u$$, reconstruction error per point is $$\operatorname{tr}(\Sigma)-u^\top\Sigma u$$.

~~~python
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
~~~

### Gaussian mixtures and EM

Start from K-means centers. EM uses Bayes' rule for the component probabilities, then fits the Gaussians using those probabilities as weights.

$$
\begin{aligned}
p(x_i)&=\sum_j\pi_j\mathcal N(x_i;\mu_j,\Sigma_j),\\
r_{ij}&=\frac{\pi_j\mathcal N(x_i;\mu_j,\Sigma_j)}{p(x_i)},\\
N_j&=\sum_i r_{ij},\qquad
\pi_j\leftarrow\frac{N_j}{n},\qquad
\mu_j\leftarrow\frac{\sum_i r_{ij}x_i}{N_j},\\
\Sigma_j&\leftarrow\frac{\sum_i r_{ij}(x_i-\mu_j)(x_i-\mu_j)^\top}{N_j}.
\end{aligned}
$$

~~~python
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
~~~

## Ensembles

### Random forest

Fit trees on bootstrap samples and try a random subset of features at each split. Average predictions for regression and vote for classification.

$$
\hat f(x)=\frac1T\sum_{t=1}^T f_t(x),\qquad
\operatorname{Var}(\hat f)=\rho\sigma^2+\frac{1-\rho}{T}\sigma^2.
$$

The variance expression assumes tree predictions have common variance $$\sigma^2$$ and pairwise correlation $$\rho$$.

~~~python
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
~~~

### AdaBoost

Fit a weighted stump $$h(x)\in\{-1,1\}$$ to labels $$y_i\in\{-1,1\}$$, then give more weight to mistakes.

$$
\begin{aligned}
\epsilon&=\sum_i a_i\mathbf1_{\{h(x_i)\ne y_i\}},\\
L(\alpha)&=(1-\epsilon)e^{-\alpha}+\epsilon e^\alpha,\\
L'(\alpha)=0 &\implies \alpha=\tfrac12\ln\frac{1-\epsilon}{\epsilon},\\
a_i'&\propto a_i e^{-\alpha y_i h(x_i)},\qquad
F(x)=\sum_t \alpha_t h_t(x).
\end{aligned}
$$

~~~python
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
~~~

### Gradient boosting

Fit each new tree to the negative gradient of the loss. For squared loss, these targets are the residuals.

$$
\ell(y,F)=\tfrac12(y-F)^2,\quad
r_i=-\partial_F\ell(y_i,F(x_i))=y_i-F(x_i),\quad
F\leftarrow F+\eta h.
$$

~~~python
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
~~~

### XGBoost

Use a quadratic loss approximation to score each leaf. This example uses squared loss and depth-one trees.

$$
\begin{aligned}
g_i&=F(x_i)-y_i,\qquad h_i=1,\qquad
G=\sum_{\rm leaf}g_i,\quad H=\sum_{\rm leaf}h_i,\\
L(w)&=Gw+\tfrac12(H+\lambda)w^2
\quad\Rightarrow\quad w^*=-\frac G{H+\lambda},\\
\mathrm{Gain}
 &=\tfrac12\left(\frac{G_L^2}{H_L+\lambda}
               +\frac{G_R^2}{H_R+\lambda}
               -\frac{G^2}{H+\lambda}\right)-\gamma .
\end{aligned}
$$

~~~python
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
~~~

## Bias and variance

At a fixed test input, write $$Y=f(x)+\epsilon$$, with independent zero-mean noise. Average over training sets $$D$$ and test noise to get

$$
\mathbb E[(Y-\hat f_D(x))^2]
=\big(f(x)-\mathbb E_D[\hat f_D(x)]\big)^2
+\operatorname{Var}_D(\hat f_D(x))
+\operatorname{Var}(\epsilon).
$$
