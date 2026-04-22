import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def chi_square_analysis(sample, alpha=0.05):
    n = len(sample)

    mu_hat = np.mean(sample)
    sigma_hat = np.std(sample, ddof=0)

    k = int(np.round(1 + 3.322 * np.log10(n)))
    if k < 4:
        k = 4

    observed, bins = np.histogram(sample, bins=k)

    edges = bins.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    p_i = np.zeros(k)

    for i in range(k):
        p_i[i] = stats.norm.cdf(edges[i+1], mu_hat, sigma_hat) - \
                 stats.norm.cdf(edges[i], mu_hat, sigma_hat)

    expected = n * p_i

    chi2_parts = (observed - expected) ** 2 / expected
    chi2_stat = np.sum(chi2_parts)

    df = k - 3
    chi2_crit = stats.chi2.ppf(1 - alpha, df) if df > 0 else 0.0

    return mu_hat, sigma_hat, chi2_stat, chi2_crit, bins, observed, expected, p_i, chi2_parts



def chi2_detail_report(name, sample, bins, obs, exp, p_i, parts):
    n = len(sample)

    print("\n" + "=" * 110)
    print(f"Детализация χ²: {name} (n={n})")
    print("=" * 110)

    print(f"{'i':<3} | {'Интервал':<28} | {'n_i':<5} | {'p_i':<10} | {'n*p_i':<12} | {'n_i-n*p_i':<15} | {'χ² вклад':<10}")
    print("-" * 110)

    k = len(obs)

    for i in range(k):
        if i == 0:
            interval = f"(-∞, {bins[1]:.3f}]"
        elif i == k - 1:
            interval = f"({bins[i]:.3f}, +∞)"
        else:
            interval = f"({bins[i]:.3f}, {bins[i+1]:.3f}]"

        diff = obs[i] - exp[i]

        print(f"{i+1:<3} | {interval:<28} | {obs[i]:<5} | {p_i[i]:<10.4f} | {exp[i]:<12.3f} | {diff:<15.3f} | {parts[i]:<10.3f}")

    print("-" * 110)
    print(f"{'Σ':<33} | {sum(obs):<5} | {sum(p_i):<10.4f} | {sum(exp):<12.3f} | {'~0':<15} | {sum(parts):<10.3f}")
    print()


def plot_distribution(sample, title):
    mu = np.mean(sample)
    sigma = np.std(sample)

    plt.figure(figsize=(7, 5))

    plt.hist(sample, bins=10, density=True, alpha=0.6, edgecolor='black')

    x = np.linspace(min(sample), max(sample), 200)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r', label='N(μ̂, σ̂)')

    plt.title(title)
    plt.grid(True, linestyle=':')
    plt.legend()

    plt.show()




def main():
    rng = np.random.default_rng(10)
    alpha = 0.05

    samples = [
        ("Нормальное N(0,1)", rng.normal(0, 1, 100)),
        ("Равномерное", rng.uniform(-np.sqrt(3), np.sqrt(3), 20)),
        ("Лаплас", rng.laplace(0, 1/np.sqrt(2), 20))
    ]

    print("=" * 100)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗЫ О НОРМАЛЬНОСТИ (α = 0.05)")
    print("=" * 100)

    print(f"{'Распределение':<22} | {'n':<5} | {'μ̂':<6} | {'σ̂':<6} | {'χ²':<8} | {'χ² крит':<8} | {'Результат':<15}")
    print("-" * 100)

    storage = []

    for name, sample in samples:
        mu, sigma, chi2, crit, bins, obs, exp, p_i, parts = chi_square_analysis(sample, alpha)

        result = "Принимается" if chi2 < crit else "ОТКЛОНЯЕТСЯ"

        print(f"{name:<22} | {len(sample):<5} | {mu:<6.2f} | {sigma:<6.2f} | {chi2:<8.2f} | {crit:<8.2f} | {result:<15}")

        storage.append((name, sample, bins, obs, exp, p_i, parts))

    print("\n")

    for item in storage:
        chi2_detail_report(*item)

    # графики ОТДЕЛЬНО
    for name, sample, *_ in storage:
        plot_distribution(sample, name)



if __name__ == "__main__":
    main()
