# alrIEEEna-26 | Binary Classification Challenge

## What's this about

This is a ML challenge where we had to predict a binary class label from 47 numerical features. Dataset had ~44k training samples and ~11k test samples. Pretty standard tabular classification problem but the class distribution was imbalanced (60-40 split roughly).

## Approach

Tried a bunch of things before settling on the final pipeline:

1. **Data Cleaning** — checked for nulls (none), duplicates (738 found), and class imbalance
2. **Feature Engineering** — added 15 row-level statistical features (mean, std, skew, kurtosis, quantiles, energy, etc). This helped a LOT actually, bumped accuracy by ~1%
3. **Scaling** — went with RobustScaler instead of StandardScaler because some features had outliers
4. **Model** — XGBoost. Tried Random Forest, Extra Trees, LightGBM, Gradient Boosting too but XGBoost consistently gave the best F1 score
5. **Threshold Tuning** — instead of using default 0.5 cutoff, searched for the optimal threshold that maximizes F1. Ended up at 0.39 which makes sense given the imbalance
6. **Validation** — 5-fold stratified cross-validation with out-of-fold predictions for honest evaluation

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.86% |
| Precision | 98.76% |
| Recall | 98.35% |
| F1 Score | 98.56% |
| ROC-AUC | 99.92% |

Confusion matrix was pretty clean — only 213 false positives and 286 false negatives out of 43k+ samples.

## Files

- `model copy.ipynb` — main notebook with the full pipeline
- `TRAIN.csv` — training data (43776 rows, 48 cols)
- `TEST.csv` — test data (10944 rows, 48 cols)  
- `FINAL.csv` — predictions on test set (ID + CLASS)

## How to run

```
pip install numpy pandas scikit-learn xgboost lightgbm
```

Open `model copy.ipynb` and run all cells. Takes around 5-6 min mainly because of the cross-validation step.

## What we learned

- Feature engineering matters more than hyperparameter tuning honestly. The statistical features gave a bigger boost than anything else
- Threshold tuning is underrated — moving from 0.5 to 0.39 improved F1 by a good margin
- RobustScaler > StandardScaler when you have outlier-ish data
- XGBoost is still king for tabular data, LightGBM was close but XGBoost edged it out

## Tech stack

- Python 3.11
- XGBoost, scikit-learn, pandas, numpy
