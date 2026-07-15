# Project: Decision Tree Classification - ShopSmart Purchase Prediction
Date: [Fill in today's date]
Context: Course Assignment (Day 31, Assignment 4 - Supervised ML)

## Problem Statement
ShopSmart, an e-commerce company, wants to predict whether a website visitor will
complete a purchase (`Revenue`: True/False) based on their browsing session behavior —
pages visited, time spent, bounce/exit rates, traffic source, and more. Built as a
Decision Tree classification model with a required F1 score benchmark of 0.55
(imbalanced dataset, so F1 rather than accuracy was specified as the evaluation metric).

## Honest Note on Origin
This project was completed by following a Decision Tree template/code provided by my
course instructor as Assignment 4. I did not design the pipeline architecture from
scratch — I wrote out each line myself, ran it, and worked to understand what each
step does and why. This document reflects my own understanding of the code after the
fact, including gaps and questions I still have. I'm documenting it honestly as an
assignment-based learning project, not an independently-designed one — the value here
was in understanding the full ML pipeline (ColumnTransformer, Pipeline, GridSearchCV)
deeply enough to explain it, not in original architecture design.

## Dataset
- Source: Online Shoppers Purchasing Intention dataset (`online_shoppers.csv`)
- Rows: 12,330 individual user sessions (1 year, one row per unique visitor session)
- Target variable: `Revenue` (binary: True = purchase made, False = no purchase)
- Class imbalance: ~15.5% positive class (based on confusion matrix support: 382 out
  of 2,466 test rows) — genuinely imbalanced, which is why F1 (not accuracy) was
  specified as the benchmark metric

## Features (17 total)
**Numerical (14):** Administrative, Administrative_Duration, Informational,
Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates,
ExitRates, PageValues, SpecialDay, OperatingSystems, Browser, Region, TrafficType

**Categorical (2):** Month, VisitorType

**⚠️ Not used by the model:** `Weekend` (boolean) — see "What I'd Do Differently" below.

## Key Decisions Made (As Understood From the Provided Code)

### Feature Type Separation
```python
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns
```
Columns were split by data type to route them to different preprocessing steps.

### Preprocessing Pipeline (ColumnTransformer)
- Numerical features → `StandardScaler()` (mean=0, std=1)
- Categorical features → `OneHotEncoder(handle_unknown="ignore")` — the
  `handle_unknown="ignore"` parameter prevents errors if the test set contains a
  category not seen during training
- Both bundled into a single `ColumnTransformer`, then combined with the model into
  one `Pipeline` object — this means preprocessing and model training happen together
  with one `.fit()` call, and the exact same transformations automatically apply to
  test data via `.predict()`

### Model: Decision Tree Classifier
```python
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
```
- `max_depth=6` — limits tree depth to prevent the tree from growing too complex and
  memorizing training data (overfitting guard)
- `min_samples_leaf=30` — requires at least 30 samples in each leaf node, which
  smooths the decision boundaries and further reduces overfitting risk
- `class_weight="balanced"` — automatically adjusts weights inversely proportional
  to class frequencies, addressing the ~15.5% positive class imbalance

### Train-Test Split
- 80/20 split, `stratify=y` (ensures both train and test sets maintain the same
  ~15.5% purchase ratio, important given the imbalance), `random_state=42`

### Hyperparameter Tuning (GridSearchCV)
```python
param_grid = {
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [20, 30, 50]
}
grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
```
- Searched 9 combinations (3×3 grid) across 5-fold cross-validation, scoring by F1
  (matching the assignment's benchmark metric)
- The `model__` prefix in parameter names is required because these parameters belong
  to the `"model"` step inside the Pipeline, not the Pipeline itself

## Model Performance

| Metric | Initial Model (max_depth=6, min_samples_leaf=30) | After GridSearchCV Tuning |
|--------|-----|-----|
| F1 Score | 0.628 | **0.634 (best)** |
| Best Params | — | max_depth=4, min_samples_leaf=50 |

**Benchmark:** 0.55 — both the initial and tuned models exceeded this comfortably.

### Classification Report (Initial Model)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Purchase) | 0.97 | 0.85 | 0.90 | 2,084 |
| 1 (Purchase) | 0.50 | 0.83 | 0.63 | 382 |

**Overall Accuracy:** 0.85

### Confusion Matrix (Initial Model)
```
              Predicted: No   Predicted: Yes
Actual: No        1,771            313
Actual: Yes           64            318
```

## Interpreting the Results
The model catches 83% of actual purchasers (recall for class 1 = 0.83) — good for
ShopSmart's business goal of not missing likely buyers. But precision for class 1 is
only 0.50, meaning half of the visitors flagged as "likely to purchase" don't
actually purchase. This tradeoff comes directly from `class_weight="balanced"`,
which deliberately prioritizes catching the minority class over precision. Whether
this tradeoff is correct depends on the business cost — if targeting a non-buyer
with marketing is cheap, this tradeoff is good; if it's expensive, precision might
need more weight.

## What I'd Do Differently Next Time
- **Fix the dropped `Weekend` column** — `select_dtypes(include=["int64", "float64"])`
  and `select_dtypes(include=["object", "category"])` don't capture boolean columns,
  so `Weekend` was silently excluded from the `ColumnTransformer` entirely (it doesn't
  appear in either the "num" or "cat" transformer lists). This is worth re-running
  with `Weekend` explicitly included (likely via a third transformer, or converting it
  to int before the split) to see if it improves the F1 score.
- Try `RandomizedSearchCV` or expand the `param_grid` range to see if there's a better
  combination outside the tested 3×3 grid
- Try comparing against Random Forest to see if the ensemble improves on precision
  for class 1 without sacrificing recall
- Re-attempt building a similar pipeline from scratch without the instructor template,
  to test my own independent understanding

## What I Learned
- How `ColumnTransformer` + `Pipeline` bundle preprocessing and modeling into one
  object, so the same transformations automatically apply to train and test data
  without manual repetition (and reduce risk of data leakage)
- Why `class_weight="balanced"` matters for imbalanced classification — and the
  precision/recall tradeoff it introduces
- How `GridSearchCV` searches hyperparameter combinations systematically using
  cross-validation, and why parameter names need the `model__` prefix when tuning
  parameters inside a Pipeline step
- That `select_dtypes()` needs to be checked carefully — boolean columns can silently
  fall through both numeric and categorical selections if not explicitly handled
- The difference between `y_test` (ground truth, held back) and `y_pred` (the model's
  actual guesses) and why evaluation metrics only make sense as a comparison between
  the two, not either one alone

## Interview Talking Points
"I worked through a Decision Tree classification pipeline predicting e-commerce
purchase intent on an imbalanced dataset (~15.5% positive class), using
ColumnTransformer and Pipeline to bundle preprocessing with modeling, and
GridSearchCV for hyperparameter tuning — improving F1 from 0.628 to 0.634 against
a 0.55 benchmark. While working through it, I noticed the feature selection logic
was silently dropping a boolean column from preprocessing, which taught me to be
more careful auditing exactly which columns a ColumnTransformer actually touches."
