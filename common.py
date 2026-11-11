"""Shared helpers for Flipkart E-commerce Decision Intelligence capstone.
Importing this module will NOT execute any Part calculations.
Only standard library and NumPy are used.
"""

import os
import csv
import math
import numpy as np


def load_data(filepath="data/records.csv"):
    """Loads records.csv and returns raw headers and structured numpy arrays.
    
    Columns in order:
      0: record_id
      1: customer_tenure_months (X1)
      2: complaint_count_90d (X2)
      3: discount_order_share (X3)
      4: avg_order_value_inr_hundreds (X4)
      5: loyalty_score (X5)
      6: cod_flag (X6)
      7: plus_member_flag (X7)
      8: churned (outcome)
      9: order_value_inr_hundreds (value)
    """
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    record_ids = [r[0] for r in rows]
    # Predictors X1 through X7
    X = np.array([[float(val) for val in r[1:8]] for r in rows], dtype=np.float64)
    # Outcome: churned
    y_churn = np.array([int(r[8]) for r in rows], dtype=np.int64)
    # Target value: order_value_inr_hundreds
    y_value = np.array([float(r[9]) for r in rows], dtype=np.float64)

    return {
        "header": header,
        "record_ids": record_ids,
        "X": X,
        "y_churn": y_churn,
        "y_value": y_value,
        "X1": X[:, 0],
        "X2": X[:, 1],
        "X3": X[:, 2],
        "X4": X[:, 3],
        "X5": X[:, 4],
        "X6": X[:, 5],
        "X7": X[:, 6],
    }


def sample_var(x):
    """Computes sample variance using divisor (n - 1)."""
    n = len(x)
    if n <= 1:
        return 0.0
    mean_x = np.mean(x)
    return np.sum((x - mean_x) ** 2) / (n - 1)


def sample_sd(x):
    """Computes sample standard deviation using divisor (n - 1)."""
    return math.sqrt(sample_var(x))


def sample_cov(x, y):
    """Computes sample covariance between x and y using divisor (n - 1)."""
    n = len(x)
    if n <= 1:
        return 0.0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)


def format_value(val):
    """Formats a value according to project rules:
    - floats printed with 4 decimals
    - integers printed as integers
    - booleans printed as yes/no
    - strings printed as given
    """
    if isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        return str(int(val))
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    elif isinstance(val, str):
        return val
    else:
        return str(val)


def write_results(filepath, results_dict):
    """Writes results to a text file with one line per key as `key = value`."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for k, v in results_dict.items():
            formatted = format_value(v)
            f.write(f"{k} = {formatted}\n")


def write_csv(filepath, header, rows):
    """Writes a CSV file with comma separator and floats formatted with 4 decimals."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            formatted_row = [format_value(item) for item in row]
            writer.writerow(formatted_row)


def stratified_split(X, y, seed=None):
    """Stratified 70/30 split according to Task 2.1 specifications:
    Takes outcome-0 rows in file order, first floor(7*count/10) to train, rest to test.
    Same for outcome-1 rows. Both sets keep file order.
    Returns (train_idx, test_idx).
    """
    n = len(y)
    idx_0 = [i for i in range(n) if y[i] == 0]
    idx_1 = [i for i in range(n) if y[i] == 1]

    n0_train = (7 * len(idx_0)) // 10
    n1_train = (7 * len(idx_1)) // 10

    train_idx = sorted(idx_0[:n0_train] + idx_1[:n1_train])
    test_idx = sorted(idx_0[n0_train:] + idx_1[n1_train:])

    return train_idx, test_idx


def standardise(X_train, X_test):
    """Standardises features using train mean and train sample SD (ddof=1).
    Applies the exact same parameters to test.
    Returns (X_train_std, X_test_std, mean_train, sd_train).
    """
    mean_train = np.mean(X_train, axis=0)
    sd_train = np.std(X_train, axis=0, ddof=1)
    X_train_std = (X_train - mean_train) / sd_train
    X_test_std = (X_test - mean_train) / sd_train
    return X_train_std, X_test_std, mean_train, sd_train


def binary_cross_entropy(y_true, p):
    """Computes Binary Cross-Entropy loss safely without log(0)."""
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -float(np.mean(y_true * np.log(p_safe) + (1.0 - y_true) * np.log(1.0 - p_safe)))


def train_logistic_regression(X_train, y_train, eta=0.1, epochs=1000):
    """Full-batch logistic regression training from scratch.
    Starting weights = zeros, bias = 0.0.
    Returns (w, b, history_dict).
    """
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    history = {}

    for epoch in range(epochs + 1):
        z = X_train @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        loss = binary_cross_entropy(y_train, p)
        history[epoch] = loss

        if epoch < epochs:
            error = p - y_train
            dw = (1.0 / n_samples) * (X_train.T @ error)
            db = float(np.mean(error))
            w -= eta * dw
            b -= eta * db

    return w, b, history


def predict_proba_logistic(X, w, b):
    """Calculates predicted probabilities for logistic regression."""
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def knn_predict(X_ref, y_ref, X_query, k):
    """Predicts binary labels using K-Nearest Neighbours with Euclidean distance.
    Ties in distance broken by position in file order among reference rows via stable sort.
    Predicts positive (1) if at least (k + 1) // 2 neighbours are positive.
    """
    dists = np.sqrt(np.sum((X_query[:, np.newaxis, :] - X_ref[np.newaxis, :, :]) ** 2, axis=2))
    threshold = (k + 1) // 2
    preds = np.zeros(len(X_query), dtype=np.int64)
    for i in range(len(X_query)):
        order = np.argsort(dists[i], kind="stable")
        k_nearest_y = y_ref[order[:k]]
        preds[i] = 1 if np.sum(k_nearest_y) >= threshold else 0
    return preds


def compute_confusion_matrix(y_true, y_pred):
    """Calculates tp, fp, tn, fn based on positive=1."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def compute_metrics(y_true, y_pred):
    """Calculates classification metrics adhering to conventions:
    If a denominator is 0, the quantity is 0.0.
    """
    tp, fp, tn, fn = compute_confusion_matrix(y_true, y_pred)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "tpr": recall,
    }
