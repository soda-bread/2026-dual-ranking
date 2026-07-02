
import numpy as np

def coverage(y_test, pred_mean, pred_std, k=1.0):
    err = np.abs(y_test - pred_mean)
    inside = err <= k * pred_std
    per_dim = inside.mean(axis=0)
    overall = inside.mean()
    return per_dim, overall

def find_alpha(X_val, y_val, model_kriging, target_coverage, alpha_max=50, alpha_step=0.01):
    mean, std = model_kriging.predict(X_val)

    alpha = 0
    best_alpha = alpha_max
    while alpha < alpha_max:
        f_upper = mean + alpha * std
        current_coverage = np.mean(y_val <= f_upper)

        if current_coverage >= target_coverage:
            best_alpha = alpha
            print(f"coverage={current_coverage*100:.3f}%")
            break

        alpha += alpha_step

    return best_alpha


def coverage_upper(y_true, y_upper):
    y_true = np.asarray(y_true)
    y_upper = np.asarray(y_upper)
    inside = y_true <= y_upper
    per_dim = inside.mean(axis=0)
    overall = inside.mean()
    return per_dim, overall
