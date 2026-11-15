Domain: E-commerce

## How to run

To reproduce all results from a clean clone, run the following exact commands in order:

pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py

Expected print output from generate_data.py:
800 10 170 c6f249ba693f1ebeed318d6d5e18de23
(representing 800 rows, 10 columns, 170 positive churn records, and the exact MD5 hash of data/records.csv).

## Part 1 findings

### R1.1 Hypothesis testing and practical significance
The null hypothesis H0 (population mean order value mu = 8.00 hundreds of INR, or INR 800) was rejected because the computed two-tailed z-statistic of 2.8930 exceeds the critical value of 1.96 (reject_h0 = yes). In e-commerce business terms, a Type I error for this claim would occur if Flipkart falsely concluded that average order value has risen above INR 800 when the true population mean remains unchanged at INR 800, leading leadership to inappropriately elevate revenue forecasts, over-purchase higher-ticket inventory, or expand operational fulfillment cost commitments based on an illusory spend gain. The observed difference mean_value - 8.0 equals 8.9382 - 8.0 = 0.9382 in column units (which corresponds to INR 93.82 per transaction). While this INR 93.82 increase is statistically significant due to the sample size of 800 records, in practical business operations an 11.7% increase represents a modest revenue expansion that could easily be eroded by return logistics, packaging overheads, or marketing acquisition costs.

### R1.2 Sample size determination
The required sample size to guarantee a margin of error within 0.50 hundreds of INR (INR 50) at a 95% confidence level is n_required = 1293 customers. Because the current dataset contains only 800 records, the planning team must recognize that the available data is underpowered for estimating population mean spend at the desired precision, requiring 493 additional customer observations (a 61.6% increase in sample volume) before executing margin-sensitive commercial budgeting decisions.

### R1.3 Gradient descent verification
The parameter difference between gradient descent and ordinary least squares was gd_slope_gap = 0.0000. Because linear regression with mean squared error over properly standardized features forms a strictly convex quadratic cost bowl with a single global optimum, batch gradient descent with a well-calibrated learning rate of eta = 0.1 converged monotonically and reached the exact closed-form analytical line within 500 iterations.

### R1.4 Heteroskedasticity and log transformation
When records were sorted by fitted order value, the residual standard deviation of the upper 400 rows was 8.6891 compared to 2.3155 for the lower 400 rows, yielding resid_sd_ratio_raw = 3.7525. This ratio confirms an expanding funnel pattern (severe heteroskedasticity) where prediction error scales proportionally with customer expenditure. Applying the log-log regression model compressed this residual standard deviation ratio to resid_sd_ratio_log = 1.0396, successfully removing the funnel and stabilizing error variance across all spending tiers.

### R1.5 Multiple regression predictors
Applying the mechanical rule |beta| > 0.05 to the coefficients in Task 1.7 (excluding the intercept beta_0 = -1.8613) selects four predictors: ln(X4) (1.1576), X6 (0.1518), X3 (0.1067), and X2 (0.0940). This mechanical threshold is an unstandardized coefficient magnitude filter rather than a true feature importance ranking because the underlying predictors exist on disparate scales (counts, shares, and log-amounts). Comparing model fits, mr_adj_r2 = 0.7403 demonstrates a substantial improvement over loglog_adj_r2 = 0.6561, indicating that adding tenure, complaints, discount reliance, loyalty score, cash-on-delivery, and membership flags explains an additional 8.42% of spending variance beyond past order value alone, proving that these extra predictors earn their place in the model.

## Part 2 findings

### R2.1 Learning rate sensitivity
The binary cross-entropy losses on the 560 training records after 200 iterations, ordered from lowest to highest, are:
1. eta = 1.0: loss_epoch_200_eta_1.0 = 0.3703
2. eta = 0.1: loss_epoch_200_eta_0.1 = 0.3776
3. eta = 0.01: loss_epoch_200_eta_0.01 = 0.5091
A too-small learning rate (eta = 0.01) makes minimal progress along the error surface and remains heavily underfitted after 200 epochs, whereas a large learning rate (eta = 1.0) achieves the fastest optimization without diverging because all seven input features were standardized using training statistics.

### R2.2 Predictive risk factors and odds ratios
The three predictors with the largest absolute standardized weights in the logistic regression model are:
1. X3 (discount_order_share): weight_X3 = 0.9013
2. X1 (customer_tenure_months): weight_X1 = -0.5554
3. X2 (complaint_count_90d): weight_X2 = 0.5505
The largest odds ratio is odds_ratio_X3 = 2.4629, which means in customer terms that for every one-standard-deviation increase in a customer's share of discounted orders, the odds of that customer churning multiply by 2.4629 (a 146.29% increase in churn odds), revealing that heavy discount reliance is the single strongest indicator of customer fragility and churn risk.

### R2.3 Classification threshold and metric trade-offs
The test set blind_accuracy obtained by predicting non-churn for every single customer is 0.7875, which sits uncomfortably close to the standard model accuracy of 0.8042 at threshold tau = 0.50. Accuracy alone is an inadequate evaluation metric because a naive model that flags nobody achieves nearly 79% accuracy solely due to class imbalance (189 non-churners vs 51 churners), while failing to detect 52.94% of actual churners. Lowering the decision threshold from tau = 0.50 to tau = 0.30 increased recall from 0.4706 (24 of 51 churners caught) to 0.7451 (38 of 51 churners caught), while precision decreased from 0.5455 to 0.4634.

### R2.4 Logistic regression versus K-Nearest Neighbours
At threshold tau = 0.30, Logistic Regression achieved an f1 score of 0.5714 and a recall of 0.7451 on the 240 test records. In contrast, K-Nearest Neighbours (using the cross-validated best parameter K = 5) achieved an f1 score of only 0.3696 and a recall of 0.3333. Logistic regression captured more than twice the proportion of churners as KNN while maintaining superior overall balance between precision and recall.

### R2.5 Model deployment and decision cost
The recommended model for production deployment is Logistic Regression operating at decision threshold tau = 0.1500. Under Flipkart's asymmetric loss function (INR 50 per wasted retention coupon and INR 400 per unflagged lost customer), deploying at tau = 0.1500 yields an out-of-sample cost_test_at_tau of INR 6050, which slashes losses relative to the naive cost_test_at_0.5 of INR 11800 and the inactive baseline cost_test_flag_nobody of INR 20400. The decision threshold was chosen strictly by minimizing empirical cost on training rows to protect the integrity of the evaluation and prevent optimistic data leakage from the test set.

## Part 3 findings

### R3.1 K-Means convergence and elbow analysis
The sequential drops in within-cluster sum of squares across the evaluated cluster counts are:
- WCSS(2) - WCSS(3) = 2262.6753 - 1602.8471 = 659.8282
- WCSS(3) - WCSS(4) = 1602.8471 - 1494.7732 = 108.0739
- WCSS(4) - WCSS(5) = 1494.7732 - 1323.7157 = 171.0575
- WCSS(5) - WCSS(6) = 1323.7157 - 1278.1691 = 45.5466
The mathematical elbow criterion supports elbow_k = 3 because the decrease in error from K=3 to K=4 (108.0739) is less than half the decrease from K=2 to K=3 (329.9141). Because the subsequent drop increases to 171.0575 between K=4 and K=5 rather than shrinking monotonically, the pinned starting row positions provide a deterministic, reproducible local clustering solution rather than a guaranteed global optimum. In Step 1 of the PCA procedure, the mean vector across the standardized continuous attributes X1 through X5 was verified to be identically all zeros.

### R3.2 Customer segment operational profiles
Based on part3_segments.csv, the three K=3 customer segments exhibit distinct behavioral profiles:
- Segment A (size = 322, mean_X1 = 47.5028, mean_X2 = 1.9503, mean_X3 = 0.2250, mean_X4 = 12.5301, mean_X5 = 62.1584, outcome_rate = 0.0186, mean_value = 3.5976): Established Core Loyalists. Action: Maintain regular brand loyalty engagement without offering unnecessary discount vouchers that dilute organic margins.
- Segment B (size = 255, mean_X1 = 28.6071, mean_X2 = 6.0706, mean_X3 = 0.4534, mean_X4 = 28.7192, mean_X5 = 55.0082, outcome_rate = 0.2353, mean_value = 17.1248): High-Yield Friction Shoppers. Action: Provide white-glove logistics routing and priority customer care resolution to eliminate delivery friction and protect their substantial INR 1712.48 average basket value.
- Segment C (size = 223, mean_X1 = 14.3892, mean_X2 = 4.2287, mean_X3 = 0.7118, mean_X4 = 16.5099, mean_X5 = 38.5215, outcome_rate = 0.4664, mean_value = 7.2883): Bargain-Focused At-Risk Shoppers. Segment C has the highest outcome_rate of 0.4664 (a 46.64% churn rate). Action: Issue targeted re-engagement coupons tied to minimum basket sizes to stimulate repeat transactions while mitigating churn.

### R3.3 Hierarchical linkage comparison
Cutting the agglomerative dendrogram at the largest consecutive merge height gap yields:
- Single linkage (4 clusters at cut): E1, E4, E5, and E2+E3+E6
- Complete linkage (2 clusters at cut): E1+E4+E5 and E2+E3+E6
These two linkage criteria disagree because single linkage evaluates the nearest neighbor distance between clusters and exhibits chaining, leaving outliers E1, E4, and E5 as singletons when a substantial gap precedes their incorporation into the main chain. Conversely, complete linkage evaluates the maximum pairwise distance, enforcing strict diameter boundaries that merge mutually close subsets into two balanced, coherent clusters.

### R3.4 Principal Component Analysis
The first principal component (PC1) explains 52.27% of total feature variance and is driven positively by customer tenure X1 (+0.5317) and loyalty score X5 (+0.4418), and driven negatively by discount share X3 (-0.5187), complaint count X2 (-0.4100), and past order value X4 (-0.2916). Retaining 80% or more of cumulative variance requires 3 components (components_for_80pct = 3, which collectively capture 86.16% of total variance). The mean PC1 projection across segments reveals sharp linear separation: mean_pc1_segment_A = 1.7773, mean_pc1_segment_B = -0.7831, and mean_pc1_segment_C = -1.6709, proving that PC1 functions as a composite index of customer relationship health, ordering segments from healthy organic loyalty to severe friction and discount dependence.

## Part 4 findings

### R4.1 Decision tree splitting and regularization
Split S4 (customer_tenure_months <= 24) is the best split because it maximizes both information gain (0.1252) and Gini gain (0.0606) by isolating newer accounts (45.74% churn rate) from tenured accounts (8.87% churn rate). A tree grown without restrictions until every leaf is pure inevitably overfits because it memorizes idiosyncratic noise and random sample flukes in the training data, leading to severe leaf fragmentation and degraded out-of-sample generalization. One established structural rule to prevent a tree from growing too deep is imposing a maximum depth limit (max_depth) or enforcing a minimum leaf sample threshold (min_samples_leaf).

### R4.2 Model comparison and ensembling
Among the five structured models evaluated on the 240 test records:
- logistic: f1 = 0.5714, recall = 0.7451
- knn: f1 = 0.3696, recall = 0.3333
- stump: f1 = 0.5532, recall = 0.7647
- hard_vote: f1 = 0.5344, recall = 0.6863
- soft_vote: f1 = 0.5507, recall = 0.7451
Flipkart should ship standalone Logistic Regression if seeking balanced performance (f1 = 0.5714) or the Decision Stump if seeking maximal churn capture (recall = 0.7647). Ensembling did not help performance; voting combinations diluted logistic regression's strong signal by incorporating poor predictions from KNN, dropping hard voting f1 to 0.5344 and soft voting f1 to 0.5507.

### R4.3 AdaBoost sample reweighting
Following round 1 normalisation, the two misclassified training instances (points 3 and 7) received an updated weight of 0.2500 each, producing a total combined weight of 0.5000. Round 2 placed greater confidence in h2 (alpha2 = 0.8047) than round 1 placed in h1 (alpha1 = 0.5493) because h2 committed fewer errors on the reweighted distribution, achieving a lower weighted classification error of round2_error = 0.1667 compared to round1_error = 0.2500.

### R4.4 Deep learning architectures and gradient dynamics
Model A Option 2 (Convolutional Neural Network) layer architecture:

| Layer | Output Shape (H x W x Depth) | Parameter Count |
| :--- | :--- | :--- |
| conv1 | 28 x 28 x 6 | 156 |
| pool1 | 14 x 14 x 6 | 0 |
| conv2 | 10 x 10 x 16 | 2416 |
| pool2 | 5 x 5 x 16 | 0 |
| flatten | 400 | 0 |
| dense1 | 32 | 12832 |
| dense2 | 10 | 330 |

For Model B (Recurrent Neural Network), the transition tensor dimensions are:
- W_xh: 128 x 64 (input to hidden)
- W_hh: 128 x 128 (hidden to hidden)
- W_hy: 16 x 128 (hidden to output)
The total parameter count rnn_total = 26768 remains invariant to sequence length because recurrent units reuse identical transition matrices and bias terms at every time step regardless of review token length. For a 10-class image categorization network, the hidden activation function is ReLU, the final output layer activation is Softmax, and the training objective is Categorical Cross-Entropy loss. In very deep plain feedforward networks, repeated multiplication of backpropagated gradients through successive layers with derivative magnitudes less than one causes the gradient signal to vanish exponentially, freezing learning in early feature extractors. A residual skip connection mitigates this breakdown by creating an unhindered identity highway that routes gradient signals directly backward without attenuation.

## Model-choice memo

To: Head of Retention Team  
From: Decision Intelligence Lead  
Subject: Production Deployment Recommendation for Churn Prevention  

I recommend deploying the Logistic Regression model for Flipkart's customer retention intervention system. When evaluated on our out-of-sample test cohort of 240 records, Logistic Regression operating at decision threshold tau = 0.30 delivers superior performance, achieving the highest test f1 score of 0.5714 and a recall of 0.7451 by correctly identifying 38 of 51 at-risk customers. Under our operational cost framework of INR 50 per wasted retention coupon (false positive) and INR 400 per lost customer (false negative), this model produces an out-of-sample cost of INR 7400.

If executive leadership prioritizes direct expenditure reduction over F1 balance, deploying this same logistic model at the training-derived optimal threshold of tau = 0.1500 achieves our lowest test cost of INR 6050 (cost_test_at_tau). This substantially outperforms the default threshold tau = 0.50 cost of INR 11800 (cost_test_at_0.5), as well as the unmitigated baseline loss of INR 20400 incurred by flagging nobody (cost_test_flag_nobody). Ensembling proved counterproductive: hard voting degraded f1 to 0.5344, while soft voting achieved 0.5507. If absolute operational simplicity is mandated, our single-rule Decision Stump (customer_tenure_months <= 24) incurs an out-of-sample cost of INR 7350, but logistic regression remains our primary recommendation due to its granular calibrated probabilities.

A notable risk of this evidence is that our test set contains only 51 positive churn events, introducing potential sampling variance in real-world deployment. Neither Model A (product image classification) nor Model B (review text processing) is worth building before the structured tabular model is operational in production, as behavioral transaction attributes already deliver actionable risk detection at negligible computational expense.

Memo word count: 273 words.

## Limitations

1. Synthetic Data Generation: The dataset is generated from synthetic parametric distributions that cannot fully capture complex real-world supply chain anomalies, festive sales spikes, or competitor discounting shifts.
2. Single Train-Test Partition: Model validation relies on a single 70/30 stratified train-test split rather than repeated k-fold cross-validation across all models, increasing sensitivity to partition composition.
3. Limited Positive Test Samples: The test partition contains only 51 actual churners, meaning individual prediction shifts carry an outsized 1.96% impact on observed recall and cost estimates.
4. Pinned Centroid Initialization: K-means clustering utilized fixed deterministic row indices for starting centroids; while completely reproducible, it does not explore multiple stochastic restarts to confirm global convergence.
