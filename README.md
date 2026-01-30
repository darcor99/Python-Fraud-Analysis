# Credit Card Fraud Detection

Binary classification of credit card transactions as fraudulent or legitimate using the [Kaggle Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (simulated data, ~1.3M training rows, ~556K test rows).

## Objective

Build and compare machine learning models that can reliably identify fraudulent transactions despite severe class imbalance (~0.58% fraud rate). The project focuses on addressing the practical challenges of fraud detection: class imbalance handling, appropriate evaluation metrics, and feature engineering from raw transactional data.

## Pipeline

1. **Data ingestion** — Download from Kaggle, load `fraudTrain.csv` and `fraudTest.csv`
2. **Exploratory analysis** — Visualize class distribution to quantify the imbalance
3. **Feature engineering** — Extract meaningful features from raw data:
   - `amt`, `log_amt` — transaction amount and log-transform
   - `hour`, `day_of_week`, `is_weekend` — cyclical temporal patterns from timestamps (raw `unix_time` buries these signals)
   - `age` — cardholder age at time of transaction
   - `category_encoded` — merchant category (label-encoded)
   - `lat_dif`, `long_dif`, `geo_distance` — cardholder-to-merchant distance
   - `city_pop` — city population
4. **Feature scaling** — `StandardScaler` to normalize all features
5. **SMOTE resampling** — Synthetic minority oversampling (`sampling_strategy=0.1`) on training set only
6. **Model training** — Three models with class imbalance handling:
   - Logistic Regression (`LogisticRegressionCV`, scored by AUC-ROC, `class_weight='balanced'`)
   - Random Forest (`class_weight='balanced_subsample'`)
   - XGBoost (`scale_pos_weight` auto-calculated from class ratio)
7. **Evaluation** — Accuracy, AUC-ROC, PR-AUC, F1, confusion matrices, ROC curves, precision-recall curves, feature importance

## Results

| Model | AUC-ROC | PR-AUC | F1 | Fraud Recall | Fraud Precision |
|---|---|---|---|---|---|
| Logistic Regression | 0.877 | 0.146 | 0.118 | 74% | 6% |
| Random Forest | 0.995 | 0.810 | 0.369 | 94% | 23% |
| XGBoost | 0.997 | 0.866 | 0.484 | 93% | 33% |

**XGBoost** is the strongest overall model with the best precision-recall balance — it catches 93% of fraud (1,994 of 2,145 cases) with the fewest false positives (4,098). **Random Forest** has the highest recall (94%) but lower precision. **Logistic Regression** is limited by its linear decision boundary, producing many false positives despite reasonable recall.

### Top Features

Both tree models rank features identically:
1. `amt` (~36% RF, ~52% XGBoost) — transaction amount dominates
2. `log_amt` (~33% RF, ~15% XGBoost)
3. `hour` (~18% RF, ~11% XGBoost) — fraud concentrates at specific hours
4. `category_encoded` (~9% both) — certain merchant categories see more fraud
5. `day_of_week` and `age` contribute modestly

Geographic features (`lat_dif`, `long_dif`, `geo_distance`) had near-zero importance.

## Conclusions

- **Ensemble models significantly outperform logistic regression** on this dataset — the nonlinear patterns in fraud data require tree-based approaches
- **Transaction amount and time of day are the strongest fraud signals**, while geographic distance between cardholder and merchant provides little discriminative value
- **Class imbalance handling is essential** — SMOTE resampling combined with class weighting enables all models to actively detect fraud rather than defaulting to the majority class
- **Precision remains a challenge** — even XGBoost flags ~2 legitimate transactions for every true fraud case; threshold tuning based on the business cost of missed fraud vs. false alerts would improve operational deployment

### Future Work

- Threshold optimization for precision-recall trade-off
- Hyperparameter tuning with `RandomizedSearchCV` or `Optuna`
- Transaction velocity features (frequency per cardholder per day)
- Amount deviation from cardholder spending patterns

## Setup

**Google Colab (original environment):**
- Requires Kaggle credentials stored in Colab secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
- The notebook downloads and unzips the dataset automatically

**Local:**
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place `fraudTrain.csv` / `fraudTest.csv` in the working directory
- Update `TrainingSet` and `TestSet` path variables in the notebook
- Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost kaggle`
