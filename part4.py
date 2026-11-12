"""Part 4: Tree-Based Methods, Ensembles, Deep Learning Architectures, and Decision Intelligence.
Flipkart E-commerce Decision Intelligence.

This script implements:
4.1 Decision tree split evaluation (Entropy, Gini, Information Gain, Gini Gain) on 560 train rows.
    Produces outputs/part4_splits.csv.
4.2 Decision stump and ensemble voting (logistic, KNN, stump, hard vote, soft vote) on 240 test rows.
    Produces outputs/part4_voting.csv and prints decision costs.
4.3 AdaBoost by hand on 8 sample points across 2 rounds.
    Produces outputs/part4_adaboost.csv.
4.4 Out-of-bag (OOB) sample fraction calculation.
4.5 Neural network architecture parameter computations (MLP, CNN, FC vs Conv, RNN).
    Produces outputs/part4_architecture.csv.

Outputs:
- outputs/part4_results.txt
- outputs/part4_splits.csv
- outputs/part4_voting.csv
- outputs/part4_adaboost.csv
- outputs/part4_architecture.csv
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
    compute_metrics,
    write_results,
    write_csv,
)


def run_part4():
    data = load_data("data/records.csv")
    X = data["X"]
    y = data["y_churn"]

    # Stratified split imported from common.py
    train_idx, test_idx = stratified_split(X, y)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    n_train = len(y_train)
    n_test = len(y_test)

    results = OrderedDict()

    # ---------------- 4.1 Decision Tree Split Evaluation ----------------
    p1_root = float(np.mean(y_train))
    p0_root = 1.0 - p1_root

    root_entropy = -(p1_root * math.log2(p1_root) + p0_root * math.log2(p0_root))
    root_gini = 1.0 - (p1_root ** 2 + p0_root ** 2)

    assert abs(root_entropy - 0.7462) < 1e-3, f"root_entropy verification failed: {root_entropy}"

    results["root_entropy"] = root_entropy
    results["root_gini"] = root_gini

    def calc_entropy(pos, total):
        if total == 0 or pos == 0 or pos == total:
            return 0.0
        p1 = pos / total
        p0 = 1.0 - p1
        return -(p1 * math.log2(p1) + p0 * math.log2(p0))

    def calc_gini(pos, total):
        if total == 0:
            return 0.0
        p1 = pos / total
        p0 = 1.0 - p1
        return 1.0 - (p1 ** 2 + p0 ** 2)

    candidate_splits = [
        ("S1", X_train[:, 5] == 1),   # X6 is cod_flag (col index 5)
        ("S2", X_train[:, 6] == 1),   # X7 is plus_member_flag (col index 6)
        ("S3", X_train[:, 2] > 0.5),  # X3 is discount_order_share (col index 2)
        ("S4", X_train[:, 0] <= 24),  # X1 is customer_tenure_months (col index 0)
    ]

    splits_rows = []
    best_info_gain = -1.0
    best_info_split = None
    best_gini_gain = -1.0
    best_gini_split = None

    for name, mask in candidate_splits:
        n_yes = int(np.sum(mask))
        pos_yes = int(np.sum(y_train[mask]))
        n_no = n_train - n_yes
        pos_no = int(np.sum(y_train[~mask]))

        e_yes = calc_entropy(pos_yes, n_yes)
        e_no = calc_entropy(pos_no, n_no)
        child_entropy = (n_yes / n_train) * e_yes + (n_no / n_train) * e_no
        info_gain = root_entropy - child_entropy

        g_yes = calc_gini(pos_yes, n_yes)
        g_no = calc_gini(pos_no, n_no)
        child_gini = (n_yes / n_train) * g_yes + (n_no / n_train) * g_no
        gini_gain = root_gini - child_gini

        splits_rows.append([name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain])

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_info_split = name

        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_gini_split = name

    write_csv(
        "outputs/part4_splits.csv",
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        splits_rows,
    )

    results["best_split_by_info_gain"] = best_info_split
    results["best_split_by_gini_gain"] = best_gini_split

    # ---------------- 4.2 Stump & Ensemble Voting ----------------
    # Best split is S4 (X1 <= 24)
    mask_train_best = (X_train[:, 0] <= 24)
    stump_rate_yes = float(np.mean(y_train[mask_train_best]))
    stump_rate_no = float(np.mean(y_train[~mask_train_best]))

    results["stump_rate_yes"] = stump_rate_yes
    results["stump_rate_no"] = stump_rate_no

    # Standardised features for Logistic and KNN
    X_train_std, X_test_std, _, _ = standardise(X_train, X_test)

    # 1. Logistic regression (Task 2.2 model)
    w_log, b_log, _ = train_logistic_regression(X_train_std, y_train, eta=0.1, epochs=1000)
    p_log = predict_proba_logistic(X_test_std, w_log, b_log)
    pred_log = (p_log >= 0.3).astype(int)

    # 2. KNN (Task 2.7 chosen K=5)
    K = 5
    dists = np.sqrt(np.sum((X_test_std[:, np.newaxis, :] - X_train_std[np.newaxis, :, :]) ** 2, axis=2))
    p_knn = np.zeros(n_test, dtype=np.float64)
    pred_knn = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        order = np.argsort(dists[i], kind="stable")
        k_y = y_train[order[:K]]
        p_knn[i] = float(np.mean(k_y))
        pred_knn[i] = 1 if np.sum(k_y) >= (K + 1) // 2 else 0

    # 3. Decision Stump on test rows
    mask_test_best = (X_test[:, 0] <= 24)
    p_stump = np.where(mask_test_best, stump_rate_yes, stump_rate_no)
    pred_stump = (p_stump >= 0.3).astype(int)

    # 4. Hard vote: positive if at least 2 of 3 models predict positive
    pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)

    # 5. Soft vote: positive if mean of 3 probabilities >= 0.3
    p_soft = (p_log + p_knn + p_stump) / 3.0
    pred_soft = (p_soft >= 0.3).astype(int)

    voting_models = [
        ("logistic", pred_log),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft),
    ]

    voting_rows = []
    print("\n--- Test Set Decision Costs (INR 50 per FP, INR 400 per FN) ---")
    for name, pred in voting_models:
        m = compute_metrics(y_test, pred)
        cost = 50 * m["fp"] + 400 * m["fn"]
        print(f"Model {name:10s} -> TP: {m['tp']:2d}, FP: {m['fp']:2d}, TN: {m['tn']:3d}, FN: {m['fn']:2d} | F1: {m['f1']:.4f} | Total Cost: INR {cost}")
        voting_rows.append([
            name,
            m["tp"],
            m["fp"],
            m["tn"],
            m["fn"],
            m["precision"],
            m["recall"],
            m["f1"],
        ])

    write_csv(
        "outputs/part4_voting.csv",
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )

    # ---------------- 4.3 AdaBoost by Hand (8 Points, 2 Rounds) ----------------
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.int64)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=np.int64)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=np.int64)

    adaboost_rows = []

    # Round 1
    w0 = np.ones(8, dtype=np.float64) / 8.0
    mis1 = (h1 != y_ada)
    eps1 = float(np.sum(w0[mis1]))
    assert abs(eps1 - 0.2500) < 1e-4, f"round1_error verification failed: {eps1}"
    alpha1 = 0.5 * math.log((1.0 - eps1) / eps1)
    unnorm1 = w0 * np.exp(-alpha1 * y_ada * h1)
    w1 = unnorm1 / np.sum(unnorm1)

    for pt in range(8):
        adaboost_rows.append([1, pt + 1, w0[pt], unnorm1[pt], w1[pt]])

    weight_misclassified_r1 = float(np.sum(w1[mis1]))
    print(f"AdaBoost: total normalised weight of misclassified points after round 1 = {weight_misclassified_r1:.4f}")

    # Round 2
    mis2 = (h2 != y_ada)
    eps2 = float(np.sum(w1[mis2]))
    alpha2 = 0.5 * math.log((1.0 - eps2) / eps2)
    unnorm2 = w1 * np.exp(-alpha2 * y_ada * h2)
    w2 = unnorm2 / np.sum(unnorm2)

    for pt in range(8):
        adaboost_rows.append([2, pt + 1, w1[pt], unnorm2[pt], w2[pt]])

    # Ensemble accuracy
    ensemble_score = alpha1 * h1 + alpha2 * h2
    ensemble_pred = np.sign(ensemble_score).astype(int)
    ensemble_acc = float(np.mean(ensemble_pred == y_ada))

    write_csv(
        "outputs/part4_adaboost.csv",
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        adaboost_rows,
    )

    results["round1_error"] = eps1
    results["round1_alpha"] = alpha1
    results["round2_error"] = eps2
    results["round2_alpha"] = alpha2
    results["ensemble_accuracy"] = ensemble_acc

    # ---------------- 4.4 Out-of-Bag (OOB) Fraction ----------------
    oob_fraction = float((1.0 - 1.0 / 560.0) ** 560)
    results["oob_fraction"] = oob_fraction

    # ---------------- 4.5 Deep Learning Architecture Calculations ----------------
    # Option 1: MLP 784 -> 16 -> 16 -> 10
    mlp_h1 = 784 * 16 + 16
    mlp_h2 = 16 * 16 + 16
    mlp_out = 16 * 10 + 10
    mlp_total = mlp_h1 + mlp_h2 + mlp_out

    # Option 2: CNN
    # Input: 28 x 28 x 1
    # conv1: 5x5, 6 filters, pad 2, stride 1 -> size = (28 - 5 + 4)/1 + 1 = 28 -> 28x28x6
    cnn_conv1 = (5 * 5 * 1 + 1) * 6
    # pool1: 2x2 stride 2 -> 14x14x6
    # conv2: 5x5, 16 filters, pad 0, stride 1 -> size = (14 - 5 + 0)/1 + 1 = 10 -> 10x10x16
    cnn_conv2 = (5 * 5 * 6 + 1) * 16
    # pool2: 2x2 stride 2 -> 5x5x16 = 400
    # dense1: 400 -> 32
    cnn_dense1 = 400 * 32 + 32
    # dense2: 32 -> 10
    cnn_dense2 = 32 * 10 + 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    architecture_rows = [
        ["mlp_hidden1", mlp_h1],
        ["mlp_hidden2", mlp_h2],
        ["mlp_output", mlp_out],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2],
    ]

    write_csv("outputs/part4_architecture.csv", ["layer", "parameters"], architecture_rows)

    results["mlp_total"] = mlp_total
    results["cnn_total"] = cnn_total

    # Large image comparison: input 128x128x3 = 49152
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    conv_layer_params = (3 * 3 * 3 + 1) * 64
    results["fc_layer_params"] = fc_layer_params
    results["conv_layer_params"] = conv_layer_params

    # Model B RNN (review text, 16 topics): input 64, hidden 128, output 16
    # H_t = tanh(W_xh X_t + W_hh H_{t-1} + b_h), Y = W_hy H_T + b_y
    w_xh = 64 * 128
    w_hh = 128 * 128
    b_h = 128
    w_hy = 128 * 16
    b_y = 16
    rnn_total = w_xh + w_hh + b_h + w_hy + b_y
    results["rnn_total"] = rnn_total

    print("\n--- CNN Layer Output Shapes (H x W x Depth) ---")
    print("Input:        28 x 28 x 1")
    print("conv1:        28 x 28 x 6  (5x5, pad 2, stride 1)")
    print("pool1:        14 x 14 x 6  (maxpool 2x2, stride 2)")
    print("conv2:        10 x 10 x 16 (5x5, pad 0, stride 1)")
    print("pool2:         5 x  5 x 16 (maxpool 2x2, stride 2)")
    print("flatten:      400")
    print("dense1:        32")
    print("dense2:        10")

    write_results("outputs/part4_results.txt", results)
    print(f"\nSuccessfully wrote {len(results)} keys to outputs/part4_results.txt")
    return results


if __name__ == "__main__":
    run_part4()

