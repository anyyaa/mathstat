import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_sample(with_outliers=False, seed=10):
    np.random.seed(seed)
    
    x = np.arange(-1.8, 2.01, 0.2)  
    noise = np.random.normal(0, 1, len(x))
    y = 2 + 2 * x + noise 
    if with_outliers:
        y = y.copy()
        y[0] += 10
        y[-1] -= 10
    
    return x, y


def least_squares_method(x, y):
    A = np.vstack([np.ones(len(x)), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def least_absolute_deviation_method(x, y):
    def objective(params):
        a, b = params
        return np.sum(np.abs(y - (a + b * x)))
    
    result = minimize(objective, [0, 0])
    return result.x


def compute_errors(a_est, b_est, a_true=2, b_true=2):
    da = abs(a_true - a_est)
    db = abs(b_true - b_est)
    
    da_rel = da / abs(a_true) * 100
    db_rel = db / abs(b_true) * 100
    
    return da, da_rel, db, db_rel



def print_results(title, mnk_params, mnm_params):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Метод':<10} {'a':<10} {'Δa':<10} {'δa%':<10} {'b':<10} {'Δb':<10} {'δb%':<10}")
    print("-" * 70)
    
    da, da_p, db, db_p = compute_errors(*mnk_params)
    print(f"{'МНК':<10} {mnk_params[0]:<10.3f} {da:<10.3f} {da_p:<10.2f} {mnk_params[1]:<10.3f} {db:<10.3f} {db_p:<10.2f}")
    
    da, da_p, db, db_p = compute_errors(*mnm_params)
    print(f"{'МНМ':<10} {mnm_params[0]:<10.3f} {da:<10.3f} {da_p:<10.2f} {mnm_params[1]:<10.3f} {db:<10.3f} {db_p:<10.2f}")
    
    print("=" * 70)


def draw_plot(x, y, mnk_params, mnm_params, title, show_outliers=False):
    y_true = 2 + 2 * x
    
    plt.figure(figsize=(8, 5))

    if show_outliers:
        plt.scatter(x[1:-1], y[1:-1], label="Данные")
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], 
                    color='red', marker='x', s=100, label="Выбросы")
    else:
        plt.scatter(x, y, label="Данные")

    plt.plot(x, y_true, '--', label="Истинная модель")
    plt.plot(x, mnk_params[0] + mnk_params[1] * x, label="МНК")
    plt.plot(x, mnm_params[0] + mnm_params[1] * x, label="МНМ")
    
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def run_experiment():

    x1, y1 = generate_sample(False)
    mnk1 = least_squares_method(x1, y1)
    mnm1 = least_absolute_deviation_method(x1, y1)
    
    print_results("БЕЗ ВЫБРОСОВ", mnk1, mnm1)
    draw_plot(x1, y1, mnk1, mnm1, "Регрессия без выбросов")
    
    x2, y2 = generate_sample(True)
    mnk2 = least_squares_method(x2, y2)
    mnm2 = least_absolute_deviation_method(x2, y2)
    
    print_results("С ВЫБРОСАМИ", mnk2, mnm2)
    draw_plot(x2, y2, mnk2, mnm2, "Регрессия с выбросами", show_outliers=True)


if __name__ == "__main__":
    run_experiment()
