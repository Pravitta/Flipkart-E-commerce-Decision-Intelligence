"""Part 3: Unsupervised Learning, Clustering, and Dimensionality Reduction.
Flipkart E-commerce Decision Intelligence.

This script implements:
3.1 K-Means clustering from scratch on standardized continuous predictors X1..X5.
    Evaluates K in {2, 3, 4, 5, 6} with pinned initial centroids.
3.2 Elbow method detection for optimal K.
3.3 Customer segmentation analysis with K=3 (Segments A, B, C ordered by mean X3).
    Produces outputs/part3_segments.csv.
3.4 Agglomerative Hierarchical Clustering from scratch (single, complete, average linkages).
    Produces outputs/part3_hierarchical.csv and determines cut clusters.
3.5 Principal Component Analysis (PCA) via eigendecomposition of sample covariance matrix.

Outputs:
- outputs/part3_results.txt
- outputs/part3_segments.csv
- outputs/part3_hierarchical.csv
"""

from collections import OrderedDict
import numpy as np

from common import (
    load_data,
    write_results,
    write_csv,
)


def run_part3():
    data = load_data("data/records.csv")
    X5 = data["X"][:, :5]  # Continuous predictors X1..X5
    n = len(X5)
    y_churn = data["y_churn"]
    y_value = data["y_value"]

    results = OrderedDict()

    # ---------------- 3.1 Standardisation & K-Means from Scratch ----------------
    mean_X5 = np.mean(X5, axis=0)
    sd_X5 = np.std(X5, axis=0, ddof=1)
    X5_std = (X5 - mean_X5) / sd_X5

    wcss_dict = {}
    passes_dict = {}
    k_models = {}

    for K in [2, 3, 4, 5, 6]:
        # Pinned starting centroids
        step = n // K
        init_indices = [i * step for i in range(K)]
        centroids = X5_std[init_indices].copy()

        passes = 0
        prev_labels = None
        while True:
            passes += 1
            # Squared Euclidean distance to each centroid: shape (n, K)
            dists = np.sum((X5_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            # Ties broken to lower-numbered centroid via np.argmin
            labels = np.argmin(dists, axis=1)

            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()

            # Update centroids: mean of members, empty stays put
            for k in range(K):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    centroids[k] = np.mean(X5_std[mask], axis=0)

        wcss = float(np.sum((X5_std - centroids[labels]) ** 2))
        wcss_dict[K] = wcss
        passes_dict[K] = passes
        k_models[K] = {"centroids": centroids, "labels": labels}
        print(f"K-Means K={K}: assignment passes = {passes}, WCSS = {wcss:.4f}")

    results["wcss_k2"] = wcss_dict[2]
    results["wcss_k3"] = wcss_dict[3]
    results["wcss_k4"] = wcss_dict[4]
    results["wcss_k5"] = wcss_dict[5]
    results["wcss_k6"] = wcss_dict[6]

    # Verify checkpoint
    assert abs(wcss_dict[2] - 2262.6753) < 1e-3, f"wcss_k2 verification failed: {wcss_dict[2]}"
    assert passes_dict[4] == 21, f"K=4 passes expected 21, got {passes_dict[4]}"
    assert passes_dict[5] == 15, f"K=5 passes expected 15, got {passes_dict[5]}"
    assert passes_dict[6] == 27, f"K=6 passes expected 27, got {passes_dict[6]}"

    # ---------------- 3.2 Elbow Method ----------------
    elbow_k = None
    for K in [3, 4, 5]:
        drop_next = wcss_dict[K] - wcss_dict[K + 1]
        drop_prev = wcss_dict[K - 1] - wcss_dict[K]
        if drop_next < 0.5 * drop_prev:
            elbow_k = K
            break

    results["elbow_k"] = int(elbow_k)

    # ---------------- 3.3 Customer Segmentation Analysis (K=3) ----------------
    k3_labels = k_models[3]["labels"]
    # Order clusters by ascending mean of X3 in original units
    cluster_means_X3 = [float(np.mean(X5[k3_labels == k, 2])) for k in range(3)]
    cluster_order = np.argsort(cluster_means_X3)
    segment_names = ["A", "B", "C"]

    segment_rows = []
    seg_masks = {}

    for i, orig_c in enumerate(cluster_order):
        seg_name = segment_names[i]
        mask = (k3_labels == orig_c)
        seg_masks[seg_name] = mask
        size = int(np.sum(mask))
        means_orig = np.mean(X5[mask], axis=0)
        outcome_rate = float(np.mean(y_churn[mask]))
        mean_val = float(np.mean(y_value[mask]))

        results[f"size_segment_{seg_name}"] = size
        segment_rows.append([
            seg_name,
            size,
            float(means_orig[0]),
            float(means_orig[1]),
            float(means_orig[2]),
            float(means_orig[3]),
            float(means_orig[4]),
            outcome_rate,
            mean_val,
        ])

    assert sum(r[1] for r in segment_rows) == 800, "Segment sizes must sum to 800"

    write_csv(
        "outputs/part3_segments.csv",
        ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
        segment_rows,
    )

    # ---------------- 3.4 Agglomerative Hierarchical Clustering ----------------
    items = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ], dtype=np.float64)

    def get_member_key(item_str):
        return int(item_str.replace("E", ""))

    def format_cluster(cluster_members):
        sorted_m = sorted(cluster_members, key=lambda x: get_member_key(x))
        return "+".join(sorted_m)

    def min_member_num(cluster_members):
        return min(get_member_key(x) for x in cluster_members)

    def cluster_distance(c1, c2, linkage):
        pair_dists = [dist_matrix[items.index(m1), items.index(m2)] for m1 in c1 for m2 in c2]
        if linkage == "single":
            return min(pair_dists)
        elif linkage == "complete":
            return max(pair_dists)
        elif linkage == "average":
            return sum(pair_dists) / len(pair_dists)

    hierarchical_rows = []
    clusters_at_cut = {}
    cut_members = {}

    for linkage in ["single", "complete", "average"]:
        clusters = [[item] for item in items]
        merges = []
        cluster_history = [list(clusters)]

        for merge_no in range(1, 6):
            best_dist = float("inf")
            best_pair = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = cluster_distance(clusters[i], clusters[j], linkage)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)

            c_i, c_j = clusters[best_pair[0]], clusters[best_pair[1]]
            if min_member_num(c_i) < min_member_num(c_j):
                ca, cb = c_i, c_j
            else:
                ca, cb = c_j, c_i

            str_ca = format_cluster(ca)
            str_cb = format_cluster(cb)
            height = float(best_dist)
            merges.append((merge_no, str_ca, str_cb, height))
            hierarchical_rows.append([linkage, merge_no, str_ca, str_cb, height])

            new_c = ca + cb
            clusters = [c for idx, c in enumerate(clusters) if idx not in best_pair] + [new_c]
            cluster_history.append(list(clusters))

        # Determine cut
        heights = [m[3] for m in merges]
        gaps = [heights[i] - heights[i - 1] for i in range(1, len(heights))]
        max_gap_idx = int(np.argmax(gaps))
        merges_below_cut = max_gap_idx + 1
        n_clusters_at_cut = 6 - merges_below_cut
        clusters_at_cut[linkage] = n_clusters_at_cut
        cut_members[linkage] = [format_cluster(c) for c in cluster_history[merges_below_cut]]

    write_csv(
        "outputs/part3_hierarchical.csv",
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hierarchical_rows,
    )

    results["single_clusters_at_cut"] = clusters_at_cut["single"]
    results["complete_clusters_at_cut"] = clusters_at_cut["complete"]
    results["average_clusters_at_cut"] = clusters_at_cut["average"]

    print("Agglomerative Single Linkage clusters at cut:", cut_members["single"])
    print("Agglomerative Complete Linkage clusters at cut:", cut_members["complete"])

    # ---------------- 3.5 Principal Component Analysis (PCA) ----------------
    # Step 1: Mean vector (zeros)
    mean_vector = np.mean(X5_std, axis=0)
    print("PCA Step 1: Mean vector:\n", mean_vector)

    # Step 2: Centre
    X_centred = X5_std - mean_vector

    # Step 3: Sample covariance matrix S = X.T @ X / (n - 1)
    S = (X_centred.T @ X_centred) / (n - 1)

    # Step 4: Eigendecomposition via numpy.linalg.eigh
    eigvals, eigvecs = np.linalg.eigh(S)

    # Step 5: Sort eigenvalues descending
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Sign convention: flip each eigenvector so its entry of largest absolute value is positive
    for j in range(eigvecs.shape[1]):
        col = eigvecs[:, j]
        max_abs_idx = int(np.argmax(np.abs(col)))
        if col[max_abs_idx] < 0:
            eigvecs[:, j] = -col

    # Step 6: Keep top two eigenvectors
    W = eigvecs[:, :2]

    # Step 7: Z = X W
    Z = X_centred @ W

    eigval_sum = float(np.sum(eigvals))
    results["eigenvalue_sum"] = eigval_sum
    assert abs(eigval_sum - 5.0) < 1e-4, f"Eigenvalue sum verification failed: {eigval_sum}"

    for i in range(5):
        results[f"eigenvalue_{i + 1}"] = float(eigvals[i])

    explained_ratio = eigvals / eigval_sum
    cum_ratio = np.cumsum(explained_ratio)

    for i in range(5):
        results[f"explained_ratio_{i + 1}"] = float(explained_ratio[i])

    for i in range(5):
        results[f"cumulative_ratio_{i + 1}"] = float(cum_ratio[i])

    components_for_80pct = int(np.argmax(cum_ratio >= 0.80) + 1)
    results["components_for_80pct"] = components_for_80pct

    # Loadings (eigenvector columns)
    for j in range(5):
        results[f"pc1_loading_X{j + 1}"] = float(W[j, 0])
    for j in range(5):
        results[f"pc2_loading_X{j + 1}"] = float(W[j, 1])

    # Projected coordinates for rows 0, 1, 2
    results["row1_pc1"] = float(Z[0, 0])
    results["row1_pc2"] = float(Z[0, 1])
    results["row2_pc1"] = float(Z[1, 0])
    results["row2_pc2"] = float(Z[1, 1])
    results["row3_pc1"] = float(Z[2, 0])
    results["row3_pc2"] = float(Z[2, 1])

    # Mean PC1 per segment
    pc1 = Z[:, 0]
    for seg in ["A", "B", "C"]:
        results[f"mean_pc1_segment_{seg}"] = float(np.mean(pc1[seg_masks[seg]]))

    write_results("outputs/part3_results.txt", results)
    print(f"Successfully wrote {len(results)} keys to outputs/part3_results.txt")
    return results


if __name__ == "__main__":
    run_part3()

