"""Part 2: Supervised Classification, Evaluation Metrics, and Decision Intelligence.
Flipkart E-commerce Decision Intelligence.

This script implements:
2.1 Stratified 70/30 train/test split preserving file order & feature standardisation.
2.2 Logistic regression from scratch (binary cross-entropy via batch gradient descent).
2.3 Learning rate comparison (eta in {0.01, 0.1, 1.0} over 200 epochs).
2.4 Odds ratios for predictors X1..X7.
2.5 Classification performance on test set at tau=0.5, tau=0.3, and blind accuracy.
2.6 Threshold sweep (tau in 0.1..0.9) producing outputs/part2_thresholds.csv.
2.7 K-Nearest Neighbours with 5-fold CV producing outputs/part2_knn_cv.csv and test metrics.
2.8 Decision-theoretic cost evaluation (INR 50 per FP, INR 400 per FN) on train and test.

Outputs:
- outputs/part2_results.txt
- outputs/part2_thresholds.csv
- outputs/part2_knn_cv.csv
"""

import math
from collections import OrderedDict
import numpy as np

from common import (
    load_data,
    stratified_split,
    standardise,
    train_logistic_regression,
    predict_proba_logistic,
    knn_predict,
    compute_metrics,
    write_results,
    write_csv,
    binary_cross_entropy,
)


def run_part2():
    data = load_data("data/records.csv")
    X = data["X"]
    y = data["y_churn"]

    results = OrderedDict()

    # ---------------- 2.1 Stratified Split & Standardisation ----------------
    train_idx, test_idx = stratified_split(X, y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_rows = len(train_idx)
    train_positives = int(np.sum(y_train))
    test_rows = len(test_idx)
    test_positives = int(np.sum(y_test))

    # Verify check
    assert (train_rows, train_positives, test_rows, test_positives) == (560, 119, 240, 51), (
        f"Split counts mismatch: {(train_rows, train_positives, test_rows, test_positives)}"
    )

    results["train_rows"] = train_rows
    results["train_positives"] = train_positives
    results["test_rows"] = test_rows
    results["test_positives"] = test_positives

    # Standardise features with train parameters
    X_train_std, X_test_std, mean_train, sd_train = standardise(X_train, X_test)

    # ---------------- 2.2 Logistic Regression from Scratch ----------------
    w, b, history = train_logistic_regression(X_train_std, y_train, eta=0.1, epochs=1000)

    # Compute test BCE
    p_test = predict_proba_logistic(X_test_std, w, b)
    test_bce = binary_cross_entropy(y_test, p_test)

    results["loss_epoch_0"] = history[0]
    results["loss_epoch_1"] = history[1]
    results["loss_epoch_10"] = history[10]
    results["loss_epoch_100"] = history[100]
    results["loss_epoch_1000"] = history[1000]

    for j in range(7):
        results[f"weight_X{j + 1}"] = float(w[j])
    results["bias"] = float(b)
    results["test_bce"] = test_bce

    # ---------------- 2.3 Learning Rate Comparison ----------------
    for eta_val in [0.01, 0.1, 1.0]:
        _, _, hist_eta = train_logistic_regression(X_train_std, y_train, eta=eta_val, epochs=200)
        results[f"loss_epoch_200_eta_{eta_val}"] = hist_eta[200]

    # ---------------- 2.4 Odds Ratios ----------------
    for j in range(7):
        results[f"odds_ratio_X{j + 1}"] = float(math.exp(w[j]))

    # ---------------- 2.5 Test Classification Performance ----------------
    # tau = 0.5
    pred_05 = (p_test >= 0.5).astype(int)
    m_05 = compute_metrics(y_test, pred_05)
    results["tau_0.5_tp"] = m_05["tp"]
    results["tau_0.5_fp"] = m_05["fp"]
    results["tau_0.5_tn"] = m_05["tn"]
    results["tau_0.5_fn"] = m_05["fn"]
    results["tau_0.5_accuracy"] = m_05["accuracy"]
    results["tau_0.5_precision"] = m_05["precision"]
    results["tau_0.5_recall"] = m_05["recall"]
    results["tau_0.5_f1"] = m_05["f1"]
    results["tau_0.5_specificity"] = m_05["specificity"]

    # tau = 0.3
    pred_03 = (p_test >= 0.3).astype(int)
    m_03 = compute_metrics(y_test, pred_03)
    results["tau_0.3_tp"] = m_03["tp"]
    results["tau_0.3_fp"] = m_03["fp"]
    results["tau_0.3_tn"] = m_03["tn"]
    results["tau_0.3_fn"] = m_03["fn"]
    results["tau_0.3_accuracy"] = m_03["accuracy"]
    results["tau_0.3_precision"] = m_03["precision"]
    results["tau_0.3_recall"] = m_03["recall"]
    results["tau_0.3_f1"] = m_03["f1"]
    results["tau_0.3_specificity"] = m_03["specificity"]

    # Blind accuracy (predicting 0 for every test record)
    blind_pred = np.zeros(test_rows, dtype=int)
    m_blind = compute_metrics(y_test, blind_pred)
    results["blind_accuracy"] = m_blind["accuracy"]

    # ---------------- 2.6 Threshold Sweep CSV ----------------
    threshold_rows = []
    for k in range(1, 10):
        tau = k / 10.0
        p_pred = (p_test >= tau).astype(int)
        m_tau = compute_metrics(y_test, p_pred)
        threshold_rows.append([
            tau,
            m_tau["tp"],
            m_tau["fp"],
            m_tau["tn"],
            m_tau["fn"],
            m_tau["tpr"],
            m_tau["fpr"],
        ])

    write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], threshold_rows)

    # ---------------- 2.7 K-Nearest Neighbours & 5-Fold CV ----------------
    k_candidates = list(range(1, 23, 2))  # 1, 3, ..., 21 (11 values)
    knn_cv_rows = []
    best_f1 = -1.0
    best_k = 1

    for k in k_candidates:
        f1_folds = []
        for f in range(5):
            val_mask = np.array([i % 5 == f for i in range(train_rows)])
            ref_mask = ~val_mask
            X_ref, y_ref = X_train_std[ref_mask], y_train[ref_mask]
            X_val, y_val = X_train_std[val_mask], y_train[val_mask]

            fold_preds = knn_predict(X_ref, y_ref, X_val, k)
            f_metrics = compute_metrics(y_val, fold_preds)
            f1_folds.append(f_metrics["f1"])

        mean_f1 = float(np.mean(f1_folds))
        knn_cv_rows.append([k, mean_f1])

        # Best K: highest mean F1, ties broken to smaller K
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], knn_cv_rows)

    results["knn_best_k"] = best_k

    # Refit with all 560 training rows as reference and evaluate on test
    test_knn_preds = knn_predict(X_train_std, y_train, X_test_std, best_k)
    m_knn = compute_metrics(y_test, test_knn_preds)
    results["knn_tp"] = m_knn["tp"]
    results["knn_fp"] = m_knn["fp"]
    results["knn_tn"] = m_knn["tn"]
    results["knn_fn"] = m_knn["fn"]
    results["knn_accuracy"] = m_knn["accuracy"]
    results["knn_precision"] = m_knn["precision"]
    results["knn_recall"] = m_knn["recall"]
    results["knn_f1"] = m_knn["f1"]

    # ---------------- 2.8 Decision Cost Evaluation ----------------
    p_train = predict_proba_logistic(X_train_std, w, b)

    best_cost_train = float("inf")
    best_cost_tau = 0.5

    for k in range(1, 20):
        tau = k / 20.0
        pred_tr = (p_train >= tau).astype(int)
        tp_tr, fp_tr, tn_tr, fn_tr = compute_metrics(y_train, pred_tr)["tp"], compute_metrics(y_train, pred_tr)["fp"], compute_metrics(y_train, pred_tr)["tn"], compute_metrics(y_train, pred_tr)["fn"]
        cost_tr = 50 * fp_tr + 400 * fn_tr

        # Lowest training cost; ties to smaller tau
        if cost_tr < best_cost_train:
            best_cost_train = cost_tr
            best_cost_tau = tau

    results["cost_tau"] = best_cost_tau
    results["cost_train_at_tau"] = int(best_cost_train)

    # Evaluate on test rows
    pred_test_tau = (p_test >= best_cost_tau).astype(int)
    m_test_tau = compute_metrics(y_test, pred_test_tau)
    cost_test_at_tau = 50 * m_test_tau["fp"] + 400 * m_test_tau["fn"]
    results["cost_test_at_tau"] = int(cost_test_at_tau)

    # Cost at tau = 0.5 on test
    cost_test_at_05 = 50 * m_05["fp"] + 400 * m_05["fn"]
    results["cost_test_at_0.5"] = int(cost_test_at_05)

    # Cost of flagging nobody on test: INR 400 * number of test positives
    cost_test_flag_nobody = 400 * test_positives
    results["cost_test_flag_nobody"] = int(cost_test_flag_nobody)

    # Write results
    write_results("outputs/part2_results.txt", results)

    # Verification checkpoints
    print(f"CHECKPOINT: train_rows={train_rows}, train_pos={train_positives}, test_rows={test_rows}, test_pos={test_positives}")
    print(f"CHECKPOINT: loss_epoch_0={history[0]:.4f}")
    print(f"CHECKPOINT: knn_best_k={best_k}")
    print(f"CHECKPOINT: cost_tau={best_cost_tau:.4f}")

    return results


if __name__ == "__main__":
    res = run_part2()
    print(f"Successfully wrote {len(res)} keys to outputs/part2_results.txt")

