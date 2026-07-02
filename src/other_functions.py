import numpy as np

def mean_std(arr):
    return np.mean(arr), np.std(arr)

def print_gpr_params(model_f1, model_f2):
    print("f1 lengthscale:", model_f1.model.kern.lengthscale.values)
    print("f1 kernel variance:", model_f1.model.kern.variance.values)
    print("f1 noise:", model_f1.model.Gaussian_noise.variance.values)
    print("f2 lengthscale:", model_f2.model.kern.lengthscale.values)
    print("f2 kernel variance:", model_f2.model.kern.variance.values)
    print("f2 noise:", model_f2.model.Gaussian_noise.variance.values)


def compute_bias_variance(preds, f_true):
    preds = np.asarray(preds)
    f_true = np.asarray(f_true)
    
    mean_pred = preds.mean(axis=0)
    bias = mean_pred - f_true
    variance = ((preds - mean_pred) ** 2).mean(axis=0)
    return {"bias": bias,"variance": variance}