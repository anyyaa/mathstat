import numpy as np
from scipy import stats


def generate_samples(seed=10):
    rng = np.random.default_rng(seed)

    n1, n2 = 20, 100

    sample1 = rng.normal(0, 1, n1)
    sample2 = rng.normal(0, 1, n2)

    return sample1, sample2


def confidence_intervals(sample, alpha=0.05):
    n = len(sample)

    mean = np.mean(sample)
    var = np.var(sample, ddof=1)
    std = np.sqrt(var)

    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    ci_mean = (mean - margin, mean + margin)

    chi2_left = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_right = stats.chi2.ppf(1 - alpha/2, df=n-1)

    ci_var = (
        (n-1)*var / chi2_right,
        (n-1)*var / chi2_left
    )

    return mean, var, ci_mean, ci_var



def fisher_test(sample1, sample2, alpha=0.05):
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)

    n1, n2 = len(sample1), len(sample2)

    if var1 >= var2:
        F = var1 / var2
        df1, df2 = n1 - 1, n2 - 1
        label = "Выборка 1"
    else:
        F = var2 / var1
        df1, df2 = n2 - 1, n1 - 1
        label = "Выборка 2"

    F_crit = stats.f.ppf(1 - alpha, df1, df2)

    decision = "ОТКЛОНЯЕТСЯ" if F > F_crit else "Принимается"

    return F, F_crit, decision, label


def main():
    alpha = 0.05

    sample1, sample2 = generate_samples()

    samples = [
        ("Выборка 1 (n=20)", sample1),
        ("Выборка 2 (n=100)", sample2)
    ]

    print("=" * 95)
    print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (α = 0.05)")
    print("=" * 95)

    print(f"{'Выборка':<20} | {'μ̂':<8} | {'ДИ для μ':<25} | {'s²':<10} | {'ДИ для σ²':<25}")
    print("-" * 95)

    for name, sample in samples:
        mean, var, ci_mean, ci_var = confidence_intervals(sample, alpha)

        ci_mean_str = f"[{ci_mean[0]:.3f}, {ci_mean[1]:.3f}]"
        ci_var_str = f"[{ci_var[0]:.3f}, {ci_var[1]:.3f}]"

        print(f"{name:<20} | {mean:<8.3f} | {ci_mean_str:<25} | {var:<10.3f} | {ci_var_str:<25}")


    print("\n" + "=" * 95)
    print("F-ТЕСТ (КРИТЕРИЙ ФИШЕРА)")
    print("=" * 95)

    F, F_crit, decision, label = fisher_test(sample1, sample2, alpha)

    print("H0: σ1² = σ2²")
    print("H1: дисперсии различаются")
    print("-" * 95)
    print(f"Большая дисперсия      : {label}")
    print(f"F наблюдаемое          : {F:.4f}")
    print(f"F критическое          : {F_crit:.4f}")
    print(f"Условие F > F_crit     : {F > F_crit}")
    print(f"Результат              : {decision}")
    print("=" * 95)


if __name__ == "__main__":
    main()
