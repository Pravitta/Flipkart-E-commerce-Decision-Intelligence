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
