import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse

def quadrant_corr(x, y):
    med_x = np.median(x)
    med_y = np.median(y)

    x_c = x - med_x
    y_c = y - med_y

    n1 = np.sum((x_c > 0) & (y_c > 0))
    n2 = np.sum((x_c < 0) & (y_c > 0))
    n3 = np.sum((x_c < 0) & (y_c < 0))
    n4 = np.sum((x_c > 0) & (y_c < 0))

    return (n1 + n3 - n2 - n4) / len(x)



def generate_mixture(rng, size):
    cov1 = [[1, 0.9], [0.9, 1]]
    cov2 = [[10, -9], [-9, 10]]

    s1 = rng.multivariate_normal([0, 0], cov1, size=size)
    s2 = rng.multivariate_normal([0, 0], cov2, size=size)

    mask = rng.random(size) < 0.9

    mixed = np.zeros((size, 2))
    mixed[mask] = s1[mask]
    mixed[~mask] = s2[~mask]

    return mixed[:, 0], mixed[:, 1]


def draw_ellipse(ax, x, y, color):
    cov = np.cov(x, y)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    width, height = 2 * 3 * np.sqrt(eigenvalues)

    center_x = np.mean(x)
    center_y = np.mean(y)

    ellipse = Ellipse(
        (center_x, center_y),
        width,
        height,
        angle=angle,
        edgecolor=color,
        facecolor='none',
        linewidth=2
    )

    ax.add_patch(ellipse)


def plot_data_block(size, data_dict, title):
    fig, axes = plt.subplots(1, len(data_dict), figsize=(5 * len(data_dict), 5))

    if len(data_dict) == 1:
        axes = [axes]

    for ax, (label, (x, y)) in zip(axes, data_dict.items()):
        ax.scatter(x, y, s=15, alpha=0.4)

        draw_ellipse(ax, x, y, 'red')

        ax.axhline(0, lw=0.5)
        ax.axvline(0, lw=0.5)

        ax.set_title(label)
        ax.grid(True, linestyle=':')

    plt.suptitle(f"{title}, n = {size}")
    plt.tight_layout()
    plt.show()


def print_summary(stats_data, dist_name):
    print("\n" + "=" * 60)
    print(f"{dist_name.upper()}")
    print("=" * 60)

    for method in stats_data:
        print(f"\n{method}")

        for size in stats_data[method]:
            for rho in stats_data[method][size]:
                mean_val = np.mean(stats_data[method][size][rho])
                var_val = np.var(stats_data[method][size][rho])

                print(f"n={size}, rho={rho}: M={mean_val:.3f}, D={var_val:.3f}")



def run_experiment():
    rng = np.random.default_rng(4)

    sizes = [20, 60, 100]
    rhos = [0, 0.5, 0.9]
    iterations = 1000

    methods = ["Пирсон", "Спирмен", "Квадрантный"]

    normal_stats = {m: {n: {rho: [] for rho in rhos} for n in sizes} for m in methods}
    mix_stats = {m: {n: {0: []} for n in sizes} for m in methods}

    # для графиков — сохраняем одну выборку
    saved_samples = {}

    for size in sizes:
        for rho in rhos:
            cov = [[1, rho], [rho, 1]]
            sample = rng.multivariate_normal([0, 0], cov, size=size)
            saved_samples[(size, rho)] = sample

        mx, my = generate_mixture(rng, size)
        saved_samples[(size, "mix")] = np.column_stack([mx, my])

    # основной цикл
    for _ in range(iterations):
        for size in sizes:
            for rho in rhos:
                cov = [[1, rho], [rho, 1]]
                sample = rng.multivariate_normal([0, 0], cov, size=size)

                x, y = sample[:, 0], sample[:, 1]

                normal_stats["Пирсон"][size][rho].append(stats.pearsonr(x, y)[0])
                normal_stats["Спирмен"][size][rho].append(stats.spearmanr(x, y)[0])
                normal_stats["Квадрантный"][size][rho].append(quadrant_corr(x, y))

            mx, my = generate_mixture(rng, size)

            mix_stats["Пирсон"][size][0].append(stats.pearsonr(mx, my)[0])
            mix_stats["Спирмен"][size][0].append(stats.spearmanr(mx, my)[0])
            mix_stats["Квадрантный"][size][0].append(quadrant_corr(mx, my))

    # вывод
    print_summary(normal_stats, "Нормальное распределение")
    print_summary(mix_stats, "Смешанное распределение")

    # графики
    for size in sizes:
        normal_block = {
            f"ρ={rho}": (
                saved_samples[(size, rho)][:, 0],
                saved_samples[(size, rho)][:, 1]
            )
            for rho in rhos
        }

        plot_data_block(size, normal_block, "Нормальное")

        mix_block = {
            "Смесь": (
                saved_samples[(size, "mix")][:, 0],
                saved_samples[(size, "mix")][:, 1]
            )
        }

        plot_data_block(size, mix_block, "Смешанное")


if __name__ == "__main__":
    run_experiment()
