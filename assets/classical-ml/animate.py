from pathlib import Path
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory

from imageio_ffmpeg import get_ffmpeg_exe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import expit, logsumexp

from classical_ml import (
    AdaBoost, DecisionTree, GaussianMixture, GaussianNaiveBayes, GradientBoosting,
    KMeans, KNN, LinearRegression, LogisticRegression, PCA, RandomForest,
    RidgeRegression, SVM, XGBoost,
)


PAPER = "#f9f8f3"
INK = "#25343b"
MUTED = "#596872"
BLUE = "#215981"
GOLD = "#9a4e23"
PALETTE = [BLUE, GOLD, "#247260"]
CLUSTER_RGB = np.array([to_rgb(color) for color in PALETTE])
SIZES = (12, 20, 28, 40, 56, 72, 90, 112, 136, 160)
SEED = 41

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": PAPER,
    "axes.facecolor": "#ffffff",
    "axes.labelcolor": INK,
    "axes.edgecolor": "#ced2ca",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def canvas(rows, columns, title, subtitle, shared=True):
    height = 7.0 if rows == 2 else 5.6
    fig, axes = plt.subplots(rows, columns, figsize=(12, height), squeeze=False,
                             sharex=shared, sharey=shared)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.815, bottom=0.12,
                        wspace=0.24, hspace=0.50)
    fig.text(0.075, 0.943, title, fontsize=24, weight="bold", va="center")
    fig.text(0.075, 0.880, subtitle, fontsize=11, color=MUTED, va="center")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(color="#e4e7e1", linewidth=0.6)
        ax.tick_params(labelsize=9, length=2)
    return fig, axes.flat


def panel(ax, title, metric):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=7)
    ax.text(0.97, 0.04, metric, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=INK, zorder=6,
            bbox={"facecolor": "white", "edgecolor": "#e1e5df", "pad": 3})


def regression_frames():
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-3, 3, (SIZES[-1], 1))

    def mean(x):
        return 0.55 * x + 0.8 * np.sin(1.65 * x)

    y = mean(X[:, 0]) + 0.22 * rng.normal(size=len(X))
    x = np.linspace(-3, 3, 361)
    truth = mean(x)
    groups = [
        ("OLS and Ridge", [LinearRegression(), RidgeRegression(alpha=12)]),
        ("KNN", [KNN(k=7, kind="regression")]),
        ("Regression tree", [DecisionTree(kind="regression", max_depth=5)]),
        ("Random forest", [RandomForest(kind="regression", seed=SEED)]),
        ("Gradient boosting", [GradientBoosting()]),
        ("XGBoost objective", [XGBoost()]),
    ]
    for n in SIZES:
        note = f"{n} training samples · dashed curve: true mean · RMSE against the true mean"
        fig, axes = canvas(2, 3, "Regression", note)
        for ax, (title, models) in zip(axes, groups):
            ax.plot(x, truth, "--", color=MUTED, alpha=0.75, linewidth=1.2)
            ax.scatter(X[:n, 0], y[:n], s=12, color=INK, alpha=0.32, linewidths=0)
            errors = []
            for label, color, model in zip(("OLS", "Ridge"), (BLUE, GOLD), models):
                prediction = model.fit(X[:n], y[:n]).predict(x[:, None])
                ax.plot(x, prediction, linewidth=1.7, color=color, label=label)
                errors.append(np.sqrt(np.mean((prediction - truth)**2)))
            ax.set(xlim=(-3, 3), ylim=(-2.25, 2.25), xticks=[-3, -1.5, 0, 1.5, 3])
            if len(models) == 2:
                ax.legend(loc="upper left", fontsize=8, frameon=False)
            panel(ax, title, " / ".join(f"{v:.3f}" for v in errors))
        fig.supxlabel("x", y=0.050)
        fig.supylabel("y", x=0.030)
        yield fig


def classification_frames():
    rng = np.random.default_rng(SEED)

    def boundary(x):
        return 0.7 * np.sin(2.2 * x) + 0.25 * x

    def sample(n):
        X = rng.uniform(-2.4, 2.4, (n, 2))
        probability = expit(4 * (X[:, 1] - boundary(X[:, 0])))
        return X, (rng.random(n) < probability).astype(int)

    X, y = sample(SIZES[-1])
    X_test, y_test = sample(1800)
    coords = np.linspace(-2.4, 2.4, 101)
    xx, yy = np.meshgrid(coords, coords)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    regions = ListedColormap(["#dce9f2", "#f4e7d9"])
    models = [
        ("Logistic regression", LogisticRegression()),
        ("Gaussian Naive Bayes", GaussianNaiveBayes()),
        ("Linear SVM", SVM(kernel="linear")),
        ("RBF SVM", SVM()),
        ("Decision tree", DecisionTree()),
        ("KNN", KNN()),
        ("Random forest", RandomForest(seed=SEED)),
        ("AdaBoost", AdaBoost()),
    ]
    for n in SIZES:
        note = f"{n} training samples · dashed curve: Bayes boundary · accuracy on the same test set"
        fig, axes = canvas(2, 4, "Binary classification", note)
        for ax, (name, model) in zip(axes, models):
            model.fit(X[:n], y[:n])
            prediction = model.predict(grid).reshape(xx.shape)
            accuracy = (model.predict(X_test) == y_test).mean()
            ax.contourf(xx, yy, prediction, levels=[-0.5, 0.5, 1.5], cmap=regions)
            ax.plot(coords, boundary(coords), "--", linewidth=1.1, color=MUTED, alpha=0.8)
            for label, marker in enumerate(("o", "^")):
                points = X[:n][y[:n] == label]
                ax.scatter(*points.T, color=PALETTE[label], s=12,
                           marker=marker, linewidths=0, alpha=0.9)
            ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            panel(ax, name, f"{accuracy:.1%}")
        fig.supxlabel("x₁", y=0.050)
        fig.supylabel("x₂", x=0.030)
        yield fig


def clustering_frames():
    rng = np.random.default_rng(SEED)
    means = [(-1.8, -1.3), (1.55, -0.55), (-0.2, 1.70)]
    covariances = [[[0.72, 0.49], [0.49, 0.59]],
                   [[0.78, -0.47], [-0.47, 0.63]],
                   [[1.05, 0.29], [0.29, 0.28]]]
    X = np.vstack([rng.multivariate_normal(mean, cov, 70)
                   for mean, cov in zip(means, covariances)])
    init = np.array([[-0.6, -1.3], [0.35, -0.80], [-0.85, 0.60]])
    kmeans = KMeans(steps=60).fit(X, init=init)
    mixture = GaussianMixture(steps=60).fit(X, init=init)
    last = max(len(kmeans.history_), len(mixture.history_)) - 1
    rounds = sorted({*range(min(last + 1, 11)), *np.linspace(min(10, last), last, 12, dtype=int)})

    for step in rounds:
        km_step = min(step, len(kmeans.history_) - 1)
        em_step = min(step, len(mixture.history_) - 1)
        note = f"210 shared samples · update {step} · identical initial centers"
        fig, (left, right) = canvas(1, 2, "Clustering", note)
        kmeans.centers_ = kmeans.history_[km_step]
        mixture.weights_, mixture.means_, mixture.covariances_ = mixture.history_[em_step]
        assignment = kmeans.predict(X)
        inertia = ((X - kmeans.centers_[assignment])**2).sum()
        log_density = logsumexp(mixture._log_joint(X), axis=1)
        left.scatter(*X.T, c=CLUSTER_RGB[assignment], s=21, alpha=0.62, linewidths=0)
        right.scatter(*X.T, c=mixture.predict_proba(X) @ CLUSTER_RGB,
                      s=21, alpha=0.67, linewidths=0)
        for index, color in enumerate(PALETTE):
            for ax, center in ((left, kmeans.centers_[index]), (right, mixture.means_[index])):
                ax.scatter(*center, c=color, marker="X", s=105, edgecolors="white", linewidths=0.85)
            values, vectors = np.linalg.eigh(mixture.covariances_[index])
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
            width, height = 4 * np.sqrt(values[::-1])
            right.add_patch(Ellipse(mixture.means_[index], width, height, angle=angle,
                                    facecolor="none", edgecolor=color, linewidth=1.7))
        for ax in (left, right):
            ax.set(xlim=(-4.5, 4.7), ylim=(-3.8, 4.0), xlabel="x₁")
            ax.set_aspect("equal", adjustable="box")
        left.set_ylabel("x₂")
        panel(left, "K-Means · hard assignments", f"step {km_step}  |  SSE {inertia:.1f}")
        panel(right, "GMM · EM responsibilities", f"step {em_step}  |  mean log p(x) {log_density.mean():.3f}")
        yield fig


def pca_frames():
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=(95, 2))
    X = np.column_stack((noise[:, 0], 0.85 * noise[:, 0] + 0.40 * noise[:, 1]))
    pca = PCA().fit(X)
    centered = X - pca.mean_
    best = pca.components_[0]
    optimal_angle = np.arctan2(best[1], best[0]) % np.pi
    arc = np.linspace(-90, 90, 181)
    angles = optimal_angle + np.radians(arc)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    variance = ((centered @ directions.T)**2).mean(axis=0)
    offsets = np.r_[np.linspace(-84, 45, 28), np.linspace(45, 0, 15)[1:]]

    for offset in offsets:
        theta = optimal_angle + np.radians(offset)
        direction = np.array([np.cos(theta), np.sin(theta)])
        projected = centered @ direction
        reconstruction = pca.mean_ + projected[:, None] * direction
        mse = ((X - reconstruction)**2).sum(axis=1).mean()
        fig, (left, right) = canvas(
            1, 2, "Principal component analysis",
            "One direction in two dimensions · orthogonal projection · 95 shared points",
            shared=False,
        )
        axis = pca.mean_ + np.array([-5, 5])[:, None] * direction
        left.plot(*axis.T, color=BLUE, linewidth=1.6)
        left.add_collection(LineCollection(np.stack((X, reconstruction), axis=1),
                                           colors="#8b9c9c", alpha=0.50, linewidths=0.8))
        left.scatter(*X.T, color=INK, s=13, alpha=0.59, linewidths=0)
        left.scatter(*reconstruction.T, color=BLUE, s=14, alpha=0.74, linewidths=0)
        left.set(xlim=(-3.4, 3.4), ylim=(-3.1, 3.1), xlabel="x₁", ylabel="x₂")
        left.set_aspect("equal", adjustable="box")
        panel(left, "Points and their projections", f"mean squared error {mse:.3f}")
        right.plot(arc, variance, color=BLUE, linewidth=2)
        right.axvline(0, color=MUTED, linestyle="--", linewidth=1)
        right.scatter(offset, (projected**2).mean(), color=GOLD, s=58, zorder=4)
        right.set(xlim=(-90, 90), ylim=(0, variance.max() * 1.17),
                  xlabel="Angle relative to PC1 (degrees)", ylabel="Projected variance")
        panel(right, "Variance along the direction", f"PC1 variance {pca.variance_[0]:.3f}")
        yield fig


def save_animation(name, frames, seconds, destination):
    with TemporaryDirectory(prefix=f"{name}-", dir=destination) as temporary:
        folder = Path(temporary)
        playlist = []
        for i, figure in enumerate(frames):
            frame = folder / f"{i:03}.png"
            figure.savefig(frame, dpi=110)
            plt.close(figure)
            playlist.extend([f"file '{frame.name}'", f"duration {seconds}"])
        playlist[-1] = "duration 2.0"
        playlist.append(f"file '{frame.name}'")
        copyfile(frame, destination / f"{name}.png")
        (folder / "frames.txt").write_text("\n".join(playlist), encoding="ascii")
        subprocess.run([
            get_ffmpeg_exe(), "-v", "error", "-y", "-f", "concat", "-i", "frames.txt",
            "-vf", "fps=20,format=yuv420p", "-an", "-c:v", "libx264",
            "-crf", "20", "-movflags", "+faststart", str(destination / f"{name}.mp4"),
        ], cwd=folder, check=True)


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent
    for name, frames, seconds in (
        ("regression", regression_frames, 0.74),
        ("classification", classification_frames, 0.74),
        ("clustering", clustering_frames, 0.42),
        ("pca", pca_frames, 0.12),
    ):
        save_animation(name, frames(), seconds, destination)
        print(name, flush=True)
