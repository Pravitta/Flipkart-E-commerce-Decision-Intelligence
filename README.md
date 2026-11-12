Domain: E-commerce

# Flipkart E-commerce Decision Intelligence: Capstone Project

## Executive Summary
This repository houses an end-to-end decision intelligence platform developed for e-commerce customer retention, spend modeling, and lifetime value optimization. All statistical inference, predictive models, unsupervised clustering, ensemble learning, and deep learning architectures are implemented purely using Python standard library and NumPy from mathematical first principles without third-party machine learning frameworks.

All dataset records are synthetic and designed for educational evaluation; they do not reflect confidential Flipkart corporate data. All monetary values are reported in Indian Rupees (INR), where 1 value unit corresponds to INR 100 (hundreds of INR).

---

## Repository Structure & Layout
The repository layout follows the strict grading specifications:
```
Flipkart E-commerce Decision Intelligence/
├── README.md                      # Project documentation and decision memo (Domain: E-commerce on line 1)
├── requirements.txt               # Locked dependencies (numpy==2.2.6)
├── generate_data.py               # Deterministic synthetic data generator
├── common.py                      # Reusable shared helpers (no execution on import)
├── part1.py                       # Part 1: Descriptive stats, inference, and regression
├── part2.py                       # Part 2: Supervised classification, metrics, and decision theory
├── part3.py                       # Part 3: K-Means, hierarchical clustering, and PCA
├── part4.py                       # Part 4: Decision trees, voting ensembles, AdaBoost, deep architectures
├── data/
│   └── records.csv                # 800 synthetic customer records (MD5: c6f249ba693f1ebeed318d6d5e18de23)
└── outputs/
    ├── part1_results.txt          # Part 1 computed parameters & diagnostic metrics
    ├── part2_results.txt          # Part 2 classification metrics & cost analysis
    ├── part2_thresholds.csv       # Threshold sweep (tau in 0.1..0.9)
    ├── part2_knn_cv.csv           # 5-fold cross-validation F1 across odd K (1..21)
    ├── part3_results.txt          # Part 3 clustering metrics & PCA loadings
    ├── part3_segments.csv         # K=3 segment operational profiles
    ├── part3_hierarchical.csv     # 15-step agglomerative merge history
    ├── part4_results.txt          # Part 4 split gains, AdaBoost rounds, NN parameters
    ├── part4_splits.csv           # Candidate decision tree split metrics
    ├── part4_voting.csv           # Test evaluation across 5 classification approaches
    ├── part4_adaboost.csv         # 2-round sample weight updates
    └── part4_architecture.csv     # Parameter counts across neural network layers
```

---

## Part 1: Descriptive Statistics, Inference & Regression Analysis

### 1.1 Summary Statistics & 95% Confidence Interval
- **Sample Size (n):** 800 customer records
- **Mean Order Value:** 8.9382 (INR 893.82)
- **Sample Standard Deviation (s):** 9.1723 (INR 917.23)
- **Standard Error (SE):** 0.3243 (INR 32.43)
- **95% Confidence Interval:** [8.3026, 9.5738] (INR 830.26 to INR 957.38)

### 1.2 Hypothesis Testing (H0: mu = 8.00)
- **Null Hypothesis (H0):** Population mean order value mu = 8.00 (INR 800).
- **Two-Tailed z-statistic:** 2.8930
- **Decision (alpha = 0.05):** `reject_h0 = yes` (|z| = 2.8930 > 1.96, p < 0.01). Mean order value significantly exceeds INR 800.

### 1.3 Sample Size Determination
- To estimate mean order value within a margin of error E = 0.50 (INR 50) at 95% confidence:
  n_required = ceil((1.96 * s / 0.50)^2) = 1293 customers

### 1.4 Simple Linear Regression (Order Value on X4)
- Model: y_hat = beta_0 + beta_1 * X4, where X4 is average past order value.
- **Slope (`reg_slope`):** 0.6701
- **Intercept (`reg_intercept`):** -3.6598
- **ANOVA Decomposition:** TSS = 67221.4285, ESS = 34925.6088, RSS = 32295.8197
- **Identity Check (`tss_equals_ess_plus_rss`):** `yes` (|TSS - ESS - RSS| < 1e-6 * TSS)
- **Performance:** R2 = 0.5196, RMSE = 6.3537, MAE = 3.7636

### 1.5 Batch Gradient Descent Verification
- Features standardized: z = (X4 - mean(X4)) / s(X4).
- Convergence trajectory (eta = 0.1):
  - J_0 = 163.9178
  - J_1 = 119.4579
  - J_10 = 41.7973
  - J_100 = 40.3698
  - J_500 = 40.3698
- Recovered parameters: `gd_slope = 0.6701`, `gd_intercept = -3.6598`
- Parameter gap: `gd_slope_gap = 0.0000` (< 0.001, exact numerical convergence).

### 1.6 Heteroskedasticity Diagnostic & Log-Log Transformation
- When rows are sorted by fitted value, residual standard deviation in the upper half (8.6891) is substantially greater than in the lower half (2.3155):
  resid_sd_ratio_raw = 8.6891 / 2.3155 = 3.7525
  indicating severe heteroskedasticity in raw units.
- Under log-log regression ln(value) = beta_0 + beta_1 * ln(X4):
  - `loglog_intercept = -2.4370`
  - `loglog_slope = 1.5030`
  - `loglog_r2 = 0.6565`
  - `loglog_rmse = 0.5268`
  - `resid_sd_lower_log = 0.5165`, `resid_sd_upper_log = 0.5369`
  - `resid_sd_ratio_log = 1.0396` (variance stabilization confirmed).

### 1.7 Multiple Linear Regression
Regressing ln(value) on all seven predictors [ones, ln(X4), X1, X2, X3, X5, X6, X7]:
- beta_0 = -1.8613 (intercept)
- beta_1 = 1.1576 (ln(X4), past order value)
- beta_2 = -0.0074 (X1, customer tenure in months)
- beta_3 = 0.0940 (X2, 90-day complaints)
- beta_4 = 0.1067 (X3, discount share)
- beta_5 = 0.0028 (X5, loyalty score)
- beta_6 = 0.1518 (X6, cash-on-delivery flag)
- beta_7 = 0.0172 (X7, Flipkart Plus membership flag)
- **Model Fit:** R2 = 0.7425, adj_R2 = 0.7403 (vs. single log-log adj_R2 = 0.6561).

---

## Part 2: Supervised Classification & Decision Theory

### 2.1 Stratified Train/Test Partitioning
- **Training Set (70%):** 560 rows (441 non-churn, 119 churn; churn rate = 21.25%)
- **Test Set (30%):** 240 rows (189 non-churn, 51 churn; churn rate = 21.25%)
- Features standardized using training sample statistics (mean_train, sd_train) only.

### 2.2 Logistic Regression from Scratch
- Binary cross-entropy optimization (eta = 0.1, 1000 epochs):
  - Epoch 0 BCE: 0.6931 (-ln(0.5))
  - Epoch 1 BCE: 0.6753
  - Epoch 10 BCE: 0.5685
  - Epoch 100 BCE: 0.3952
  - Epoch 1000 BCE: 0.3704
  - Out-of-sample Test BCE: 0.3818
- Learned Weights:
  - w_1 = -0.5554 (X1, tenure)
  - w_2 = 0.5505 (X2, complaints)
  - w_3 = 0.9013 (X3, discount share)
  - w_4 = -0.1459 (X4, past AOV)
  - w_5 = -0.2162 (X5, loyalty score)
  - w_6 = 0.2348 (X6, COD flag)
  - w_7 = -0.0120 (X7, Plus member flag)
  - bias = -2.0696

### 2.3 Learning Rate Sensitivity (200 Epochs)
- eta = 0.01: Loss = 0.5091 (underfitting/slow convergence)
- eta = 0.1: Loss = 0.3776 (stable, rapid descent)
- eta = 1.0: Loss = 0.3703 (fastest convergence without divergence)

### 2.4 Risk Factor Interpretation (Odds Ratios)
- OR(X3) = exp(0.9013) = 2.4629: Each 1-SD increase in discount reliance increases churn odds by 146.29%.
- OR(X2) = exp(0.5505) = 1.7341: Each 1-SD increase in complaints increases churn odds by 73.41%.
- OR(X1) = exp(-0.5554) = 0.5739: Customer tenure is protective; each 1-SD increase reduces churn odds by 42.61%.

### 2.5 Classification Threshold Evaluation on Test Set
- **Default Threshold (tau = 0.50):**
  - TP = 24, FP = 20, TN = 169, FN = 27
  - Accuracy: 0.8042, Precision: 0.5455, Recall: 0.4706, F1: 0.5053, Specificity: 0.8942
  - Captures fewer than half of actual churners (47.06%).
- **Retention Threshold (tau = 0.30):**
  - TP = 38, FP = 44, TN = 145, FN = 13
  - Accuracy: 0.7625, Precision: 0.4634, Recall: 0.7451, F1: 0.5714, Specificity: 0.7672
  - Recall jumps from 47.06% to 74.51%, capturing 38 of 51 at-risk accounts.
- **Blind Accuracy (predicting 0 for all):** 0.7875 (189 / 240). Demonstrates that plain accuracy is misleading under class imbalance.

### 2.7 K-Nearest Neighbours & Cross-Validation
- 5-fold cross-validation on 560 training records across odd K in [1, 21]:
  - Best hyperparameter: `knn_best_k = 5` (mean CV F1 = 0.4541).
- Out-of-sample test performance (K=5):
  - TP = 17, FP = 24, TN = 165, FN = 34
  - Accuracy: 0.7583, Precision: 0.4146, Recall: 0.3333, F1: 0.3696

### 2.8 Decision-Theoretic Cost Optimization
Flipkart's cost structure:
- **Cost of False Positive (wasted retention coupon):** INR 50
- **Cost of False Negative (unflagged churned customer lost):** INR 400
- **Objective:** Cost = 50 * FP + 400 * FN

Evaluating tau in {0.05, 0.10, ..., 0.95} on the training set:
- **Optimal Training Threshold (`cost_tau`):** tau* = 0.1500
- **Training Cost at tau* = 0.1500:** INR 12,150 (vs. INR 28,700 at tau = 0.50)

**Out-of-sample Test Cost Benchmark:**
- **Flag Nobody (Blind):** 400 * 51 = **INR 20,400**
- **Default Logistic (tau = 0.50):** 50 * 20 + 400 * 27 = **INR 11,800**
- **Decision-Optimal Logistic (tau = 0.1500):** **INR 6,050**
  - Selecting tau = 0.1500 reduces customer churn financial losses by **48.7%** compared to the naive tau = 0.50 baseline, and by **70.3%** compared to inaction.

---

## Part 3: Unsupervised Segmentation & Dimensionality Reduction

### 3.1 K-Means Clustering on Continuous Attributes (X1..X5)
Pinned starting centroids converged as follows:
- K=2: 6 assignment passes, WCSS = 2262.6753
- K=3: 7 assignment passes, WCSS = 1602.8471
- K=4: 21 assignment passes, WCSS = 1494.7732
- K=5: 15 assignment passes, WCSS = 1323.7157
- K=6: 27 assignment passes, WCSS = 1278.1691
- **Elbow Analysis:** `elbow_k = 3` (WCSS(3) - WCSS(4) = 108.0739 < 0.5 * (WCSS(2) - WCSS(3)) = 329.9141).

### 3.3 Customer Segment Profiles (K=3)
Ordered by ascending discount dependence (X3):
1. **Segment A (Loyal Champions, n=322):**
   - Mean Tenure: 47.50 months | Complaints: 1.95 | Discount Share: 22.50% | Past AOV: INR 1,253.01 | Loyalty Score: 62.16
   - **Churn Rate:** 1.86% (6 / 322) | **Mean Order Value:** INR 359.76
   - *Strategic Action:* Retain organic engagement; avoid margin-diluting discount campaigns.
2. **Segment B (High-Value Power Shoppers, n=255):**
   - Mean Tenure: 28.61 months | Complaints: 6.07 | Discount Share: 45.34% | Past AOV: INR 2,871.92 | Loyalty Score: 55.01
   - **Churn Rate:** 23.53% (60 / 255) | **Mean Order Value:** INR 1,712.48
   - *Strategic Action:* Critical high-yield cohort; prioritize VIP customer support resolution to alleviate the friction indicated by 6.07 complaints.
3. **Segment C (Discount Hunters, n=223):**
   - Mean Tenure: 14.39 months | Complaints: 4.23 | Discount Share: 71.18% | Past AOV: INR 1,650.99 | Loyalty Score: 38.52
   - **Churn Rate:** 46.64% (104 / 223) | **Mean Order Value:** INR 728.83
   - *Strategic Action:* Deploy targeted coupon incentives contingent on purchase frequency thresholds.

### 3.4 Agglomerative Hierarchical Clustering (Sample Matrix E1..E6)
- Single, complete, and average linkage merge trees computed from pairwise distances.
- **Single Linkage:** Largest gap at Merge 2 -> 3 (height 1.4000 to 2.8000; gap = 1.4000), yielding **4 clusters at cut**:
  - `['E1', 'E4', 'E5', 'E2+E3+E6']`
- **Complete Linkage:** Largest gap at Merge 4 -> 5 (height 2.9000 to 7.9000; gap = 5.0000), yielding **2 clusters at cut**:
  - `['E2+E3+E6', 'E1+E4+E5']`
- **Average Linkage:** Largest gap at Merge 4 -> 5 (height 2.9000 to 5.6556; gap = 2.7556), yielding **2 clusters at cut**:
  - `['E2+E3+E6', 'E1+E4+E5']`

### 3.5 Principal Component Analysis (PCA)
- Total variance: `eigenvalue_sum = 5.0000`
- **Eigenvalues:** lambda_1 = 2.6136, lambda_2 = 1.1828, lambda_3 = 0.5114, lambda_4 = 0.4055, lambda_5 = 0.2867
- **Explained Variance Ratios:** 52.27%, 23.66%, 10.23%, 8.11%, 5.73%
- **Cumulative Variance:** PC1+PC2 account for 75.93%; 3 components are required to exceed 80% (`components_for_80pct = 3`, 86.16%).
- **Component Loadings:**
  - PC1: +0.5317 * X1 - 0.4100 * X2 - 0.5187 * X3 - 0.2916 * X4 + 0.4418 * X5 (Tenure vs. Churn Risk / Friction)
  - PC2: +0.1773 * X1 + 0.5056 * X2 - 0.2648 * X3 + 0.6934 * X4 + 0.4025 * X5 (High Spend & Engagement Intensity)
- **Segment Separation along PC1:**
  - Segment A: +1.7773
  - Segment B: -0.7831
  - Segment C: -1.6709

---

## Part 4: Trees, Ensembles, Deep Architectures & Decision Intelligence

### 4.1 Decision Tree Splitting
On 560 training records:
- Root Node Entropy: 0.7462 | Root Gini: 0.3347
- Candidate Splits:
  - S1 (X6 = 1, COD): Info Gain = 0.0110, Gini Gain = 0.0052
  - S2 (X7 = 1, Plus Member): Info Gain = 0.0033, Gini Gain = 0.0015
  - S3 (X3 > 0.5, Discount Share): Info Gain = 0.1046, Gini Gain = 0.0494
  - S4 (X1 <= 24, Customer Tenure): Info Gain = 0.1252, Gini Gain = 0.0606
- **Optimal Split:** S4 (X1 <= 24 months) achieves highest Information Gain (0.1252) and Gini Gain (0.0606).
- Decision Stump Probabilities: `stump_rate_yes = 0.4574`, `stump_rate_no = 0.0887`.

### 4.2 Out-of-Sample Ensemble & Voting Comparison (240 Test Rows)

| Model | TP | FP | TN | FN | Precision | Recall | F1 Score | Test Cost (50*FP + 400*FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Stump (X1 <= 24)** | 39 | 51 | 138 | 12 | 0.4333 | 0.7647 | 0.5532 | **INR 7,350** |
| **Logistic Regression (tau=0.30)** | 38 | 44 | 145 | 13 | 0.4634 | 0.7451 | **0.5714** | **INR 7,400** |
| **Soft Voting Ensemble** | 38 | 49 | 140 | 13 | 0.4368 | 0.7451 | 0.5507 | **INR 7,650** |
| **Hard Voting Ensemble** | 35 | 45 | 144 | 16 | 0.4375 | 0.6863 | 0.5344 | **INR 8,650** |
| **KNN (K=5)** | 17 | 24 | 165 | 34 | 0.4146 | 0.3333 | 0.3696 | **INR 14,800** |

*Key Finding for Management:*
While Logistic Regression (tau=0.30) achieves the highest F1 score (0.5714), the single-rule **Decision Stump (X1 <= 24 months)** achieves the lowest operational cost among all baseline classifiers (**INR 7,350**), detecting 76.47% of churners with zero feature standardization or complex inference overhead.

### 4.3 Manual AdaBoost Step Verification
- 8 data points, initial weights w_i = 1/8 = 0.1250:
  - Round 1 Error: `round1_error = 0.2500` (points 3 and 7 misclassified)
  - Round 1 Alpha: `round1_alpha = 0.5493`
  - Weight of misclassified points after Round 1 normalization: 0.2500 + 0.2500 = 0.5000 (exact property of AdaBoost)
  - Round 2 Error: `round2_error = 0.1667`
  - Round 2 Alpha: `round2_alpha = 0.8047`
  - Ensemble Accuracy: `ensemble_accuracy = 0.7500` (6 of 8 correctly classified)

### 4.4 Out-of-Bag (OOB) Sampling
- Asymptotic bootstrap sampling:
  oob_fraction = (1 - 1/560)^560 = 0.3676 (approximately 1/e)

### 4.5 Neural Network Architecture Parameters
- **Option 1 (MLP 784 -> 16 -> 16 -> 10):**
  - Hidden 1 (784 * 16 + 16): 12,560
  - Hidden 2 (16 * 16 + 16): 272
  - Output (16 * 10 + 10): 170
  - **Total Parameters:** `mlp_total = 13002`
- **Option 2 (CNN for 28x28x1 Image Categorization):**
  - conv1 (5 * 5 * 1 * 6 + 6): 156 parameters -> Output shape: 28 x 28 x 6
  - pool1 (2 x 2, stride 2): 0 parameters -> Output shape: 14 x 14 x 6
  - conv2 (5 * 5 * 6 * 16 + 16): 2,416 parameters -> Output shape: 10 x 10 x 16
  - pool2 (2 x 2, stride 2): 0 parameters -> Output shape: 5 x 5 x 16
  - flatten: 5 * 5 * 16 = 400
  - dense1 (400 * 32 + 32): 12,832 parameters -> Output: 32
  - dense2 (32 * 10 + 10): 330 parameters -> Output: 10
  - **Total Parameters:** `cnn_total = 15734`
- **Parameter Efficiency: Dense vs. Convolutional:**
  - Fully connected layer (128 * 128 * 3 -> 1024): `fc_layer_params = 50332672`
  - Convolutional layer (64 filters of 3 x 3 x 3): `conv_layer_params = 1792`
  - Demonstrates weight sharing and spatial parameter efficiency (>28,000x reduction).
- **Model B (Recurrent Neural Network for Product Reviews):**
  - Input: 64, Hidden: 128, Output: 16
  - W_xh = 64 * 128 = 8,192, W_hh = 128 * 128 = 16,384, b_h = 128
  - W_hy = 128 * 16 = 2,048, b_y = 16
  - **Total Parameters:** `rnn_total = 26768`

---

## Reproducibility & Verification

To reproduce every output file from scratch:
```bash
# 1. Install locked dependencies
pip install -r requirements.txt

# 2. Generate raw synthetic records
python generate_data.py

# 3. Execute all analysis pipelines
python part1.py
python part2.py
python part3.py
python part4.py
```
Every output in `outputs/` will be generated identically and deterministically.
