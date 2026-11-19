---
title: "classical ML"
category: "Machine Learning"
order: 1
description: "经典机器学习的数学推导、Python 实现与同任务动画。"
---

* TOC
{:toc}

样本按行排列，$$X\in\mathbb R^{n\times d}$$，$$x_i$$ 为第 $$i$$ 个样本的列向量。代码使用 NumPy 和 SciPy。

~~~python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
~~~

[Python 类]({{ '/assets/classical-ml/classical_ml.py' | relative_url }}) · [动画生成脚本]({{ '/assets/classical-ml/animate.py' | relative_url }})

## 监督学习

### Linear Regression

$$
\begin{aligned}
y_i\mid x_i &\sim \mathcal N(x_i^\top w+b,\sigma^2),\\
-\log p(y\mid X,w,b,\sigma^2)
 &=\frac n2\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\|y-Xw-b\mathbf1\|_2^2 .
\end{aligned}
$$

对 $$\tfrac12\|y-Xw-b\mathbf1\|_2^2$$ 求导。记 $$X_c=X-\mathbf1\bar x^\top$$，$$y_c=y-\bar y\mathbf1$$。

$$
\begin{aligned}
\partial_b L=0 &\implies b=\bar y-\bar x^\top w,\\
\nabla_w L=X_c^\top(X_cw-y_c)=0
 &\implies X_c^\top X_cw=X_c^\top y_c,\\
\hat w&=X_c^\dagger y_c,\qquad \hat b=\bar y-\bar x^\top\hat w.
\end{aligned}
$$

~~~python
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
~~~

### Ridge Regression

取先验 $$w\sim\mathcal N(0,\tau^2 I)$$，截距自由，$$\lambda=\sigma^2/\tau^2$$。

$$
\begin{aligned}
\hat w_{\lambda}
 &=\arg\min_w\left(\tfrac12\|y_c-X_cw\|_2^2
                    +\tfrac\lambda2\|w\|_2^2\right)\\
 &=(X_c^\top X_c+\lambda I)^{-1}X_c^\top y_c,\qquad \lambda>0,\\[2pt]
\hat w_\lambda
 &=\arg\min_w\left\|
 \begin{bmatrix} X_c\\ \sqrt{\lambda}I\end{bmatrix}w
 -\begin{bmatrix}y_c\\0\end{bmatrix}\right\|_2^2.
\end{aligned}
$$

若 $$X_c=U\operatorname{diag}(s_j)V^\top$$，则

$$
 \hat w_\lambda
 =V\operatorname{diag}\!\left(\frac{s_j}{s_j^2+\lambda}\right)U^\top y_c.
$$

~~~python
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
~~~

### Logistic Regression

记 $$z_i=x_i^\top w+b$$，$$p_i=\sigma(z_i)=(1+e^{-z_i})^{-1}$$。由 $$Y_i\mid x_i\sim\operatorname{Bernoulli}(p_i)$$ 得

$$
\begin{aligned}
\ell_i
 &=-y_i\ln p_i-(1-y_i)\ln(1-p_i)\\
 &=\ln(1+e^{z_i})-y_i z_i,\\
\partial_{z_i}\ell_i&=p_i-y_i,\qquad
\partial_{z_i}^2\ell_i=p_i(1-p_i)\in(0,\tfrac14].
\end{aligned}
$$

令 $$A=[X,\mathbf1]$$，$$u=(w,b)$$，$$R=\operatorname{diag}(1,\ldots,1,0)$$。

$$
\begin{aligned}
L(u)&=\frac1n\sum_i\ell_i+\frac{\lambda}{2}u^\top Ru,\\
\nabla L(u)&=A^\top(p-y)/n+\lambda Ru,\\
\nabla^2L(u)&=A^\top\operatorname{diag}\!\big(p_i(1-p_i)\big)A/n+\lambda R,\\
\|\nabla^2L(u)\|_2&\le L_0=\frac{\|A\|_2^2}{4n}+\lambda,\\
u_{t+1}&=u_t-L_0^{-1}\nabla L(u_t),\qquad
\hat y(x)=\mathbf1_{\{w^\top x+b\ge0\}} .
\end{aligned}
$$

~~~python
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
~~~

### Gaussian Naive Bayes

$$
\begin{aligned}
p(c\mid x)&=\frac{\pi_c\,p(x\mid c)}{\sum_a\pi_a\,p(x\mid a)},\\
p(x\mid c)&=\prod_{j=1}^d\mathcal N(x_j\mid\mu_{cj},v_{cj}),\\
g_c(x)&=\ln\pi_c-\frac12\sum_j
 \left[\ln(2\pi v_{cj})+\frac{(x_j-\mu_{cj})^2}{v_{cj}}\right],\\
\hat c(x)&=\arg\max_c g_c(x).
\end{aligned}
$$

令 $$D_c=\{i:y_i=c\}$$，$$n_c=\lvert D_c\rvert$$。对数似然的分量为

$$
\begin{aligned}
\partial_{\mu_{cj}}\log p(D)
 &=v_{cj}^{-1}\sum_{i\in D_c}(x_{ij}-\mu_{cj})=0,\\
\partial_{v_{cj}}\log p(D)
 &=-\frac{n_c}{2v_{cj}}
   +\frac{\sum_{i\in D_c}(x_{ij}-\mu_{cj})^2}{2v_{cj}^2}=0,\\
\widehat\mu_{cj}&=\frac1{n_c}\sum_{i\in D_c}x_{ij},\\
\widehat v_{cj}
 &=\max\!\left(\delta,\frac1{n_c}\sum_{i\in D_c}(x_{ij}-\widehat\mu_{cj})^2\right),\\
\partial_{\pi_c}\bigg(\sum_a n_a\ln\pi_a-\nu\Big(\sum_a\pi_a-1\Big)\bigg)&=0
 \ \Longrightarrow\ \hat\pi_c=\frac{n_c}{n}.
\end{aligned}
$$

~~~python
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
~~~

### Support Vector Machine

设 $$y_i\in\{-1,1\}$$，$$f_i=\langle w,\phi(x_i)\rangle+b$$。可分情形缩放到 $$\min_i y_if_i=1$$，两侧间隔宽度为 $$2/\|w\|_2$$。

$$
\begin{aligned}
\max_{w,b}\ \min_i\frac{y_if_i}{\|w\|_2}
 &\iff \min_{w,b}\ \tfrac12\|w\|_2^2
          \quad\mathrm{s.t.}\quad y_if_i\ge1,\\
\min_{w,b,\xi}\ \tfrac12\|w\|_2^2+C\sum_i\xi_i
 &\quad\mathrm{s.t.}\quad y_if_i\ge1-\xi_i,\quad \xi_i\ge0,\\
\xi_i^*=[1-y_if_i]_+
 &\implies\min_{w,b}\ \tfrac12\|w\|_2^2+C\sum_i[1-y_if_i]_+ .
\end{aligned}
$$

$$
\begin{aligned}
\mathcal L
 &=\tfrac12\|w\|_2^2+C\sum_i\xi_i
 -\sum_i\alpha_i(y_if_i-1+\xi_i)-\sum_i\nu_i\xi_i,\\
\nabla_w\mathcal L=0 &\implies w=\sum_i\alpha_i y_i\phi(x_i),\\
\partial_b\mathcal L=0 &\implies \sum_i\alpha_i y_i=0,\\
\partial_{\xi_i}\mathcal L=0 &\implies C=\alpha_i+\nu_i.
\end{aligned}
$$

代回 $$\mathcal L$$，令 $$K_{ij}=\langle\phi(x_i),\phi(x_j)\rangle$$。

$$
\begin{aligned}
\hat\alpha
 &=\arg\max_{\substack{0\le\alpha\le C\\y^\top\alpha=0}}
 \left\{\mathbf1^\top\alpha-\frac12(\alpha\odot y)^\top K(\alpha\odot y)\right\},\\
f(x)&=\sum_{i:\hat\alpha_i>0}\hat\alpha_i y_i K(x_i,x)+b,\\
K_{\mathrm{linear}}(x,z)&=x^\top z,\qquad
K_{\mathrm{RBF}}(x,z)=e^{-\gamma\|x-z\|_2^2}.
\end{aligned}
$$

$$
y_if_i
\begin{cases}
\ge 1 & \hat\alpha_i=0,\\
= 1 & 0<\hat\alpha_i<C,\\
\le 1 & \hat\alpha_i=C.
\end{cases}
$$

自由支持向量给出 $$b=y_i-\sum_j\hat\alpha_jy_jK(x_j,x_i)$$。无自由支持向量时，取 KKT 给出的截距区间中点。

~~~python
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
~~~

### Decision Tree

在当前节点令 $$a_i\ge0$$ 为样本权重，$$W=\sum_i a_i$$，$$p_c=W^{-1}\sum_i a_i\mathbf1_{\{y_i=c\}}$$。候选划分 $$S=\mathbf1_{\{x_j>t\}}$$。

$$
\begin{aligned}
H(p)&=-\sum_c p_c\ln p_c,\\
I(Y;S)&=H(Y)-H(Y\mid S)
       =H(p)-\sum_{s=0}^1\frac{W_s}{W}H(p_s),\\
G(p)&=\Pr(Y_1\ne Y_2)=1-\sum_c p_c^2,\qquad Y_1,Y_2\overset{\mathrm{i.i.d.}}{\sim}p.
\end{aligned}
$$

$$
\begin{aligned}
L_{\mathrm{node}}^{\mathrm{cls}} &\in
 \{WH(p),\,WG(p),\,W(1-\max_c p_c)\},\\
\partial_a\sum_i a_i(y_i-a)^2&=2\sum_i a_i(a-y_i)=0
 \quad\Longrightarrow\quad a^*=\frac{\sum_i a_i y_i}{W},\\
L_{\mathrm{node}}^{\mathrm{reg}}&=\sum_i a_i y_i^2-\frac{(\sum_i a_i y_i)^2}{W},\\
(j^*,t^*)&=\arg\max_{j,t}[L_{\mathrm{node}}-L_{\mathrm L}-L_{\mathrm R}].
\end{aligned}
$$

在排序后相邻不同取值之间枚举阈值。深度、最小叶样本数和增益决定停止。

~~~python
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
~~~

### K-Nearest Neighbours

令 $$N_k(x)$$ 为欧氏距离下最近的 $$k$$ 个训练点。

$$
\begin{aligned}
\arg\min_c\sum_{i\in N_k(x)}\mathbf1_{\{y_i\ne c\}}
 &=\arg\max_c\sum_{i\in N_k(x)}\mathbf1_{\{y_i=c\}},\\
\partial_a\sum_{i\in N_k(x)}(y_i-a)^2&=2ka-2\sum_{i\in N_k(x)}y_i,\\
\hat y_{\mathrm{reg}}(x)&=\frac1k\sum_{i\in N_k(x)}y_i.
\end{aligned}
$$

均匀分布于单位立方体时，内部边长为 $$r$$ 的小立方体满足 $$\mathbb E[\#\text{邻居}]=nr^d$$，期望容纳 $$k$$ 个点所需的边长为 $$r=(k/n)^{1/d}$$。

~~~python
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
~~~

## 无监督学习

### K-Means

$$
\begin{aligned}
J(c,\mu)&=\sum_{i=1}^n\|x_i-\mu_{c_i}\|_2^2,\\
c_i^{(t+1)}&=\arg\min_j\|x_i-\mu_j^{(t)}\|_2^2,\\
\partial_{\mu_j}J&=2\sum_{i:c_i=j}(\mu_j-x_i)=0,\\
\mu_j^{(t+1)}&=\frac{1}{|\{i:c_i^{(t+1)}=j\}|}
  \sum_{i:c_i^{(t+1)}=j} x_i .
\end{aligned}
$$

$$
0\le J(c^{(t+1)},\mu^{(t+1)})
\le J(c^{(t+1)},\mu^{(t)})\le J(c^{(t)},\mu^{(t)}).
$$

空簇保留旧中心。目标值单调收敛，所得分组依赖初始化。

~~~python
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
~~~

### Principal Component Analysis

令 $$X_c=X-\mathbf1\bar x^\top$$，$$\Sigma=X_c^\top X_c/n$$。

$$
\begin{aligned}
\operatorname{Var}(u^\top x)&=u^\top\Sigma u,\quad u^\top u=1,\\
\nabla_u\!\left[u^\top\Sigma u-\lambda(u^\top u-1)\right]&=0
 \quad\Longrightarrow\quad\Sigma u=\lambda u,\\
u^\top\Sigma u&=\lambda.
\end{aligned}
$$

令 $$U^\top U=I_r$$，$$P=UU^\top$$。若 $$\Sigma v_j=\lambda_jv_j$$，$$\lambda_1\ge\cdots\ge\lambda_d$$，则

$$
\begin{aligned}
\frac1n\|X_c-X_cP\|_F^2
   &=\operatorname{tr}\Sigma-\operatorname{tr}(U^\top\Sigma U),\\
\operatorname{tr}(U^\top\Sigma U)&=\sum_j\lambda_j a_j,\qquad
 a_j=\|U^\top v_j\|_2^2,\quad 0\le a_j\le1,\quad \sum_j a_j=r,\\
\operatorname{tr}(U^\top\Sigma U)&\le\sum_{j=1}^r\lambda_j ,
 \qquad U=[v_1,\ldots,v_r]\ \text{取等号},\\
X_c=A\operatorname{diag}(s_j)V^\top
 &\implies v_j=V_{:j},\quad \lambda_j=s_j^2/n .
\end{aligned}
$$

~~~python
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
~~~

### Gaussian Mixture Model 与 EM

$$
p_\theta(x,z=k)=\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),\qquad
\ell(\theta)=\sum_i\ln\sum_k p_\theta(x_i,z=k).
$$

对每个 $$i$$，令 $$r_i$$ 为分量上的概率向量。

$$
\begin{aligned}
\ell(\theta)
 &=\mathcal F(r,\theta)
   +\sum_i D_{\mathrm{KL}}\bigl(r_i\,\|\,p_\theta(z\mid x_i)\bigr),\\
\mathcal F(r,\theta)
 &=\sum_{i,k}r_{ik}\ln\frac{p_\theta(x_i,z=k)}{r_{ik}},\\
\ell(\theta)&\ge\mathcal F(r,\theta).
\end{aligned}
$$

E-step 取等号。

$$
r_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}
             {\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)},\qquad
N_k=\sum_i r_{ik}.
$$

M-step 固定 $$r$$，令 $$A_k=\Sigma_k^{-1}$$，$$S_k=N_k^{-1}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$$。

$$
\begin{aligned}
\partial_{\pi_k}\Big(\sum_jN_j\ln\pi_j-\nu(\textstyle\sum_j\pi_j-1)\Big)&=0
 \ \Longrightarrow\ \pi_k=\frac{N_k}{n},\\
\partial_{\mu_k}\mathcal F=\sum_i r_{ik}A_k(x_i-\mu_k)&=0
 \ \Longrightarrow\ \mu_k=\frac{\sum_i r_{ik}x_i}{N_k},\\
\partial_{A_k}\left(\frac{N_k}{2}[\ln\det A_k-\operatorname{tr}(A_kS_k)]\right)
 &=\frac{N_k}{2}(A_k^{-1}-S_k).
\end{aligned}
$$

无约束时，一个分量可令 $$\mu_k=x_i$$、$$\Sigma_k=\sigma^2I$$，使样本 $$x_i$$ 的密度随 $$\sigma^{-d}$$ 增大，其余样本由其他分量覆盖。加上 $$\Sigma_k\succeq\delta I$$ 后，在 $$S_k$$ 的特征基上，Hadamard 不等式使固定对角元下的 $$\det A_k$$ 在对角矩阵上取最大值。

$$
\begin{aligned}
S_k=V\operatorname{diag}(s_j)V^\top,\qquad
\max_{0<a_j\le1/\delta}\{\ln a_j-a_js_j\}
 &\implies a_j^{-1}=\max(s_j,\delta),\\
\ell(\theta^{(t+1)})\ \ge\ \mathcal F(r^{(t)},\theta^{(t+1)})
 &\ge\ \mathcal F(r^{(t)},\theta^{(t)})=\ell(\theta^{(t)}).
\end{aligned}
$$

球形等方差、固定正混合权重、唯一最近中心时，

$$
\begin{aligned}
r_{ik}&=\operatorname{softmax}_k
     \left(\ln\pi_k-\frac{\|x_i-\mu_k\|_2^2}{2\sigma^2}\right)\\
&\xrightarrow[\sigma^2\downarrow0]{}\mathbf1_{\{k=\arg\min_j\|x_i-\mu_j\|_2^2\}}.
\end{aligned}
$$

代入均值更新即得 K-Means 的中心更新。

~~~python
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
~~~

## 集成学习

### Random Forest

每棵树对 $$n$$ 个样本有放回地抽样 $$n$$ 次。每个节点重新抽取候选特征。

$$
\hat f_{\mathrm{reg}}(x)=\frac1T\sum_{t=1}^T f_t(x),\qquad
\hat f_{\mathrm{cls}}(x)=\arg\max_c\sum_{t=1}^T\mathbf1_{\{f_t(x)=c\}}.
$$

固定 $$x$$，若 $$\operatorname{Var}(f_t)=\sigma^2$$，$$\operatorname{Cov}(f_s,f_t)=\rho\sigma^2$$（$$s\ne t$$），则

$$
\begin{aligned}
\operatorname{Var}\!\left(\frac1T\sum_t f_t\right)
 &=\frac{T\sigma^2+T(T-1)\rho\sigma^2}{T^2}\\
 &=\rho\sigma^2+(1-\rho)\frac{\sigma^2}{T}.
\end{aligned}
$$

~~~python
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
~~~

### AdaBoost

$$
\begin{aligned}
F_t(x)&=F_{t-1}(x)+\alpha h(x),\qquad y_i,h(x_i)\in\{-1,1\},\\
a_i&=\frac{e^{-y_i F_{t-1}(x_i)}}{\sum_j e^{-y_j F_{t-1}(x_j)}},
  \qquad\epsilon=\sum_i a_i\mathbf1_{\{y_i\ne h(x_i)\}},\\
L_t(\alpha)&\propto
 \sum_i a_i e^{-\alpha y_ih(x_i)}=(1-\epsilon)e^{-\alpha}+\epsilon e^\alpha,\\
L_t'(\alpha^*)=0&\implies\alpha^*=\tfrac12\ln\frac{1-\epsilon}{\epsilon},\\
a_i^{\mathrm{new}}&=\frac{a_i e^{-\alpha^* y_i h(x_i)}}{\sum_j a_j e^{-\alpha^* y_j h(x_j)}} .
\end{aligned}
$$

当 $$\epsilon\in(0,\tfrac12)$$，最小值为 $$2\sqrt{\epsilon(1-\epsilon)}$$，随 $$\epsilon$$ 递增。代码用加权分类错误挑选树桩。

~~~python
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
~~~

### Gradient Boosting

$$
\begin{aligned}
L(F)&=\sum_i\ell(y_i,F(x_i)),\\
r_{it}&=-\left.\partial_f\ell(y_i,f)\right|_{f=F_{t-1}(x_i)},\\
h_t&=\arg\min_{h\in\mathcal H_{\mathrm{tree}}}\sum_i(r_{it}-h(x_i))^2,\\
\rho_t&=\arg\min_\rho\sum_i\ell\!\left(y_i,F_{t-1}(x_i)+\rho h_t(x_i)\right),\\
F_t&=F_{t-1}+\eta\rho_t h_t .
\end{aligned}
$$

对平方损失 $$\ell(y,f)=\tfrac12(y-f)^2$$，$$F_0=\bar y$$，$$r_{it}=y_i-F_{t-1}(x_i)$$。

$$
\begin{aligned}
h_t(x)&=\bar r_j\quad(x\in \text{叶 }j),\\
\sum_i r_{it}h_t(x_i)&=\sum_j n_j\bar r_j^2=\sum_i h_t(x_i)^2
 \quad\Longrightarrow\quad\rho_t=1\quad(h_t\not\equiv0).
\end{aligned}
$$

函数空间更新见 [Friedman, Greedy Function Approximation](https://doi.org/10.1214/aos/1013203451)。

~~~python
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
~~~

### XGBoost

令 $$g_i=\partial_f\ell(y_i,f_i)$$，$$h_i=\partial_f^2\ell(y_i,f_i)$$。拟合一棵有 $$J$$ 个叶、叶值 $$w_j$$ 的树。

$$
\begin{aligned}
\widetilde L(w)
 &=\sum_i\left(g_iw_{q(x_i)}+\tfrac12 h_iw_{q(x_i)}^2\right)
             +\tfrac{\lambda}{2}\sum_j w_j^2+\gamma J\\
 &=\sum_j\left(G_jw_j+\tfrac12(H_j+\lambda)w_j^2\right)+\gamma J,\\
G_j&=\sum_{i:q(x_i)=j}g_i,\qquad H_j=\sum_{i:q(x_i)=j}h_i,\\
w_j^*&=-\frac{G_j}{H_j+\lambda},\qquad
\widetilde L^*=-\frac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma J .
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{Gain}
 &=\frac12\left(\frac{G_{\mathrm L}^2}{H_{\mathrm L}+\lambda}
     +\frac{G_{\mathrm R}^2}{H_{\mathrm R}+\lambda}
     -\frac{(G_{\mathrm L}+G_{\mathrm R})^2}{H_{\mathrm L}+H_{\mathrm R}+\lambda}\right)
       -\gamma,\\
g_i&=f_i-y_i,\qquad h_i=1\quad\text{（平方损失）}.
\end{aligned}
$$

代码实现平方损失、正则化叶值与精确贪心分裂。二阶目标见 [Chen & Guestrin, XGBoost](https://arxiv.org/abs/1603.02754)。

~~~python
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
~~~

## 偏差–方差与线性自编码器

固定测试点 $$x$$，训练集 $$D$$ 随机，$$Y=f(x)+\varepsilon$$，$$\varepsilon$$ 与 $$D$$ 独立，$$\mathbb E[\varepsilon]=0$$。记 $$m(x)=\mathbb E_D[\hat f_D(x)]$$。

$$
\begin{aligned}
\mathbb E_{D,Y\mid x}[(Y-\hat f_D(x))^2]
 &=\mathbb E_{D,Y\mid x}[(\varepsilon+f(x)-m(x)+m(x)-\hat f_D(x))^2]\\
 &=\operatorname{Var}(\varepsilon)+[f(x)-m(x)]^2
                  +\operatorname{Var}_D(\hat f_D(x)).
\end{aligned}
$$

中心化数据上，线性编码器 $$B:\mathbb R^d\to\mathbb R^r$$、解码器 $$A:\mathbb R^r\to\mathbb R^d$$。对 $$S=\operatorname{im}A$$，

$$
\|x-ABx\|_2^2
=\|x-P_S x\|_2^2+\|P_Sx-ABx\|_2^2
\ \ge\ \|x-P_Sx\|_2^2.
$$

取 $$S$$ 为前 $$r$$ 个主成分的子空间、$$A=U$$、$$B=U^\top$$ 时达到最小重建误差。

## 同任务动画 {#shared-tasks}

### 一维回归

$$
\begin{aligned}
X&\sim{\rm Unif}[-3,3],\qquad Z\sim\mathcal N(0,1),\quad Z\perp X,\\
Y&=f(X)+0.22Z,\qquad f(x)=0.55x+0.8\sin(1.65x).
\end{aligned}
$$

每帧用同一训练序列的前 $$n$$ 个样本重新拟合。虚线为 $$f$$，RMSE 在同一均匀网格上相对 $$f$$ 计算。Ridge 取 $$\lambda=12$$。

<video controls playsinline preload="none" style="display:block;width:100%;height:auto" aria-label="同一样本上的回归模型比较" poster="{{ '/assets/classical-ml/regression.png' | relative_url }}">
  <source src="{{ '/assets/classical-ml/regression.mp4' | relative_url }}" type="video/mp4">
</video>

### 二维二分类

$$
\begin{aligned}
X&\sim{\rm Unif}([-2.4,2.4]^2),\\
B(x_1)&=0.7\sin(2.2x_1)+0.25x_1,\\
Y\mid X=x&\sim{\rm Bernoulli}\!\left(\sigma\big(4[x_2-B(x_1)]\big)\right).
\end{aligned}
$$

每帧复用前 $$n$$ 个训练样本和独立的固定测试集。底色是预测类别，虚线是 Bayes 边界。

<video controls playsinline preload="none" style="display:block;width:100%;height:auto" aria-label="同一样本上的二分类决策区域比较" poster="{{ '/assets/classical-ml/classification.png' | relative_url }}">
  <source src="{{ '/assets/classical-ml/classification.mp4' | relative_url }}" type="video/mp4">
</video>

### 二维聚类

$$
\begin{aligned}
X_{k,i}&\sim\mathcal N(\mu_k,\Sigma_k),\qquad i=1,\ldots,70,\\
\mu_1&=(-1.8,-1.3),\quad \mu_2=(1.55,-0.55),\quad \mu_3=(-0.2,1.70),\\
\Sigma_1&=\begin{pmatrix}.72&.49\\.49&.59\end{pmatrix},\quad
\Sigma_2=\begin{pmatrix}.78&-.47\\-.47&.63\end{pmatrix},\quad
\Sigma_3=\begin{pmatrix}1.05&.29\\.29&.28\end{pmatrix}.
\end{aligned}
$$

两种算法共享初始中心与 210 个无标签样本。右图点色按后验责任度混合，椭圆为马氏距离等于 2 的等密度线。

<video controls playsinline preload="none" style="display:block;width:100%;height:auto" aria-label="K-Means 与 Gaussian Mixture 的实际迭代" poster="{{ '/assets/classical-ml/clustering.png' | relative_url }}">
  <source src="{{ '/assets/classical-ml/clustering.mp4' | relative_url }}" type="video/mp4">
</video>

### 二维投影至一维

$$
X=(Z_1,\,0.85Z_1+0.40Z_2),\qquad (Z_1,Z_2)\sim\mathcal N(0,I_2).
$$

扫过不同投影角度，最后落在 SVD 求出的第一主轴。

<video controls playsinline preload="none" style="display:block;width:100%;height:auto" aria-label="PCA 投影方向、方差与重建误差" poster="{{ '/assets/classical-ml/pca.png' | relative_url }}">
  <source src="{{ '/assets/classical-ml/pca.mp4' | relative_url }}" type="video/mp4">
</video>

同目录运行 `python animate.py» 可复现四组动画，额外依赖 Matplotlib 与 imageio-ffmpeg。
