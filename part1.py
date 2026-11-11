"""Part 1: Descriptive Statistics, Inference, and Regression Analysis.
Flipkart E-commerce Decision Intelligence.

This script implements:
1.1 Summary statistics and 95% confidence interval for order value.
1.2 Hypothesis testing for population mean order value (H0: mu = 8.0).
1.3 Sample size determination for margin of error <= 0.5 (INR 50).
1.4 Closed-form linear regression of order value on X4 (avg_order_value_inr_hundreds).
1.5 Batch Gradient Descent for linear regression on standardized X4.
1.6 Heteroskedasticity diagnostic & log-log transformation.
1.7 Multiple linear regression of ln(value) on predictors via normal equations.

Outputs: outputs/part1_results.txt
"""

import math
from collections import OrderedDict
import numpy as np

from common import (
    load_data,
    sample_var,
    sample_sd,
    sample_cov,
    write_results,
)


def run_part1():
    # Load dataset
    data = load_data("data/records.csv")
    y = data["y_value"]
    X1 = data["X1"]
    X2 = data["X2"]
    X3 = data["X3"]
    X4 = data["X4"]
    X5 = data["X5"]
    X6 = data["X6"]
    X7 = data["X7"]

    results = OrderedDict()

    # ---------------- 1.1 Summary Statistics & 95% CI ----------------
    n = len(y)
    mean_val = float(np.mean(y))
    sd_val = float(sample_sd(y))
    se_val = float(sd_val / math.sqrt(n))
    ci95_low = float(mean_val - 1.96 * se_val)
    ci95_high = float(mean_val + 1.96 * se_val)

    results["n"] = n
    results["mean_value"] = mean_val
    results["sd_value"] = sd_val
    results["se_value"] = se_val
    results["ci95_low"] = ci95_low
    results["ci95_high"] = ci95_high

    # ---------------- 1.2 Hypothesis Testing ----------------
    # H0: mu = 8.0 (two-tailed, alpha = 0.05)
    mu_0 = 8.0
    z_stat = float((mean_val - mu_0) / se_val)
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    results["z_stat"] = z_stat
    results["reject_h0"] = reject_h0

    # ---------------- 1.3 Required Sample Size ----------------
    # Smallest integer n >= (1.96 * sd_value / 0.5)^2 (ceil)
    n_required = int(math.ceil((1.96 * sd_val / 0.5) ** 2))
    results["n_required"] = n_required

    # ---------------- 1.4 Simple Linear Regression (Closed Form) ----------------
    # Regress value on X4
    mean_X4 = float(np.mean(X4))
    cov_X4_y = float(sample_cov(X4, y))
    var_X4 = float(sample_var(X4))

    reg_slope = float(cov_X4_y / var_X4)
    reg_intercept = float(mean_val - reg_slope * mean_X4)

    fit_val = reg_intercept + reg_slope * X4
    residuals_raw = y - fit_val

    tss = float(np.sum((y - mean_val) ** 2))
    ess = float(np.sum((fit_val - mean_val) ** 2))
    rss = float(np.sum((y - fit_val) ** 2))

    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = float(1.0 - rss / tss)
    rmse = float(math.sqrt(rss / n))
    mae = float(np.mean(np.abs(y - fit_val)))

    results["reg_slope"] = reg_slope
    results["reg_intercept"] = reg_intercept
    results["tss"] = tss
    results["ess"] = ess
    results["rss"] = rss
    results["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
    results["r2"] = r2
    results["rmse"] = rmse
    results["mae"] = mae

    # ---------------- 1.5 Gradient Descent on Standardized X4 ----------------
    sd_X4 = float(sample_sd(X4))
    z = (X4 - mean_X4) / sd_X4

    m = 0.0
    c = 0.0
    eta = 0.1
    gd_costs = {}

    for step in range(501):
        yhat = m * z + c
        e = y - yhat
        cost = float((1.0 / n) * np.sum(e ** 2))
        if step in [0, 1, 10, 100, 500]:
            gd_costs[step] = cost

        if step < 500:
            dJ_dm = float(-(2.0 / n) * np.sum(z * e))
            dJ_dc = float(-(2.0 / n) * np.sum(e))
            m -= eta * dJ_dm
            c -= eta * dJ_dc

    gd_slope = float(m / sd_X4)
    gd_intercept = float(c - m * mean_X4 / sd_X4)
    gd_slope_gap = float(abs(gd_slope - reg_slope))

    results["gd_J0"] = gd_costs[0]
    results["gd_J1"] = gd_costs[1]
    results["gd_J10"] = gd_costs[10]
    results["gd_J100"] = gd_costs[100]
    results["gd_J500"] = gd_costs[500]
    results["gd_slope"] = gd_slope
    results["gd_intercept"] = gd_intercept
    results["gd_slope_gap"] = gd_slope_gap

    # ---------------- 1.6 Heteroskedasticity Diagnostic & Log-Log Model ----------------
    # Sort rows by Task-1.4 fitted value ascending with stable sort
    order_raw = np.argsort(fit_val, kind="stable")
    lower_idx_raw = order_raw[:400]
    upper_idx_raw = order_raw[400:]

    resid_sd_lower_raw = float(sample_sd(residuals_raw[lower_idx_raw]))
    resid_sd_upper_raw = float(sample_sd(residuals_raw[upper_idx_raw]))
    resid_sd_ratio_raw = float(resid_sd_upper_raw / resid_sd_lower_raw)

    results["resid_sd_lower_raw"] = resid_sd_lower_raw
    results["resid_sd_upper_raw"] = resid_sd_upper_raw
    results["resid_sd_ratio_raw"] = resid_sd_ratio_raw

    # Closed-form regression of ln(value) on ln(X4)
    ln_y = np.log(y)
    ln_X4 = np.log(X4)

    mean_lny = float(np.mean(ln_y))
    mean_lnX4 = float(np.mean(ln_X4))
    cov_log = float(sample_cov(ln_X4, ln_y))
    var_log = float(sample_var(ln_X4))

    loglog_slope = float(cov_log / var_log)
    loglog_intercept = float(mean_lny - loglog_slope * mean_lnX4)

    fit_log = loglog_intercept + loglog_slope * ln_X4
    residuals_log = ln_y - fit_log

    tss_log = float(np.sum((ln_y - mean_lny) ** 2))
    rss_log = float(np.sum(residuals_log ** 2))
    loglog_r2 = float(1.0 - rss_log / tss_log)
    loglog_rmse = float(math.sqrt(rss_log / n))

    results["loglog_intercept"] = loglog_intercept
    results["loglog_slope"] = loglog_slope
    results["loglog_r2"] = loglog_r2
    results["loglog_rmse"] = loglog_rmse

    # Split by ITS OWN fitted values with stable sort
    order_log = np.argsort(fit_log, kind="stable")
    lower_idx_log = order_log[:400]
    upper_idx_log = order_log[400:]

    resid_sd_lower_log = float(sample_sd(residuals_log[lower_idx_log]))
    resid_sd_upper_log = float(sample_sd(residuals_log[upper_idx_log]))
    resid_sd_ratio_log = float(resid_sd_upper_log / resid_sd_lower_log)

    results["resid_sd_lower_log"] = resid_sd_lower_log
    results["resid_sd_upper_log"] = resid_sd_upper_log
    results["resid_sd_ratio_log"] = resid_sd_ratio_log

    # ---------------- 1.7 Multiple Linear Regression ----------------
    # Regress ln(value) on [ones, ln(X4), X1, X2, X3, X5, X6, X7]
    ones = np.ones(n, dtype=np.float64)
    X_mr = np.column_stack([ones, ln_X4, X1, X2, X3, X5, X6, X7])
    Y_mr = ln_y

    # Solve normal equations: beta = solve(X.T @ X, X.T @ Y)
    beta = np.linalg.solve(X_mr.T @ X_mr, X_mr.T @ Y_mr)

    for i in range(8):
        results[f"mr_beta_{i}"] = float(beta[i])

    fit_mr = X_mr @ beta
    tss_mr = float(np.sum((Y_mr - mean_lny) ** 2))
    rss_mr = float(np.sum((Y_mr - fit_mr) ** 2))

    mr_r2 = float(1.0 - rss_mr / tss_mr)
    # p = 7 predictors
    p_mr = 7
    mr_adj_r2 = float(1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1))

    # loglog_adj_r2 with p=1
    p_log = 1
    loglog_adj_r2 = float(1.0 - (1.0 - loglog_r2) * (n - 1) / (n - p_log - 1))

    results["mr_r2"] = mr_r2
    results["mr_adj_r2"] = mr_adj_r2
    results["loglog_adj_r2"] = loglog_adj_r2

    # Save outputs
    output_path = "outputs/part1_results.txt"
    write_results(output_path, results)

    # Verification checkpoints
    print(f"CHECKPOINT: n = {n}")
    print(f"CHECKPOINT: mean_value = {mean_val:.4f}")
    print(f"CHECKPOINT: gd_slope_gap = {gd_slope_gap:.6f}")
    assert n == 800, f"Expected n=800, got {n}"
    assert abs(mean_val - 8.9382) < 1e-4, f"Expected mean_value=8.9382, got {mean_val:.4f}"
    assert gd_slope_gap < 0.001, f"gd_slope_gap {gd_slope_gap} not < 0.001"

    return results


if __name__ == "__main__":
    res = run_part1()
    print(f"Successfully wrote {len(res)} keys to outputs/part1_results.txt")

