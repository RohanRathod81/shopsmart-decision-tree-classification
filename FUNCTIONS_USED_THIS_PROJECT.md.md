# 🧰 Functions & Techniques Used — This Project
### ShopSmart Decision Tree Classification (Purchase Prediction)

*(Project-specific file — add matching entries to your MASTER_FUNCTIONS_CHEATSHEET.md kept locally at ML_Series root)*

---

## 📚 PANDAS FUNCTIONS

| Function | What It Does | Where I Used It |
|----------|---------------|-------------------|
| `df.drop(columns=[...])` | Removes specified column(s) | Separated `X` (features) from `Revenue` (target) |
| `series.astype(int)` | Converts data type | Converted `Revenue` (True/False) to `int` (0/1) for the model |
| `df.select_dtypes(include=[...])` | Selects columns matching data type(s) | Splitting numeric vs categorical columns — ⚠️ learned this misses boolean columns |

---

## 🤖 SCIKIT-LEARN — PREPROCESSING & PIPELINE (New This Project!)

| Function | What It Does | Where I Used It |
|----------|---------------|-------------------|
| `ColumnTransformer(transformers=[...])` | Applies different preprocessing to different column groups in one object | Numeric columns → StandardScaler, categorical → OneHotEncoder, combined |
| `Pipeline(steps=[...])` | Chains preprocessing + model into a single object | `.fit()` and `.predict()` now handle both preprocessing AND modeling in one call |
| `StandardScaler()` | Scales features to mean=0, std=1 | Applied to numeric features only, within the ColumnTransformer |
| `OneHotEncoder(handle_unknown="ignore")` | Converts categorical text to binary columns | Applied to `Month` and `VisitorType`; `handle_unknown="ignore"` prevents errors on unseen categories in test data |

---

## 🤖 SCIKIT-LEARN — MODEL

| Function | What It Does | Where I Used It |
|----------|---------------|-------------------|
| `DecisionTreeClassifier(max_depth=, min_samples_leaf=, class_weight=, random_state=)` | Builds a decision tree classification model | Core model — parameters tuned to fight overfitting and class imbalance |
| `class_weight="balanced"` | Auto-adjusts class weights inversely to their frequency | Addressed the ~15.5% positive class imbalance in the target |

---

## 🤖 SCIKIT-LEARN — HYPERPARAMETER TUNING (New This Project!)

| Function | What It Does | Where I Used It |
|----------|---------------|-------------------|
| `GridSearchCV(estimator, param_grid, scoring=, cv=, n_jobs=)` | Exhaustively tests every combination in a parameter grid, using cross-validation | Tested 9 combinations of `max_depth` × `min_samples_leaf`, scored by F1, 5-fold CV |
| `"model__paramname"` syntax | Targets a parameter inside a specific Pipeline step | Needed because `max_depth`/`min_samples_leaf` belong to the `"model"` step, not the Pipeline itself |
| `grid.best_score_` | Returns the best cross-validated score found | Best F1 = 0.634 |
| `grid.best_params_` | Returns the parameter combination that achieved the best score | `{max_depth: 4, min_samples_leaf: 50}` |
| `n_jobs=-1` | Uses all available CPU cores to run the search in parallel | Speeds up the grid search |

---

## 📊 SCIKIT-LEARN — CLASSIFICATION EVALUATION METRICS

| Function | What It Does | Why It Matters Here |
|----------|---------------|-------------------|
| `f1_score(y_test, y_pred)` | Harmonic mean of Precision and Recall | The specified benchmark metric (target: beat 0.55) since the dataset is imbalanced |
| `classification_report(y_test, y_pred)` | Full per-class Precision/Recall/F1/Support table | Revealed the precision/recall tradeoff — high recall (0.83), lower precision (0.50) for the minority class |
| `confusion_matrix(y_test, y_pred)` | 2×2 table of prediction outcomes | Showed exactly how many actual purchasers were correctly caught (318) vs missed (64) |

---

## 💡 KEY CONCEPTS LEARNED (New This Project)

1. **Pipeline + ColumnTransformer combo** — bundling preprocessing and modeling into
   one object means the exact same transformations automatically apply to train and
   test data, with no manual re-encoding step and less risk of data leakage

2. **`select_dtypes()` has blind spots** — boolean columns don't match `["int64",
   "float64"]` or `["object", "category"]`, so they can silently disappear from a
   ColumnTransformer without any error — always double-check by comparing your
   feature list length to your original column count

3. **`class_weight="balanced"` and its tradeoff** — auto-balancing class weights
   improves recall on the minority class but typically at some cost to precision;
   the right balance depends on the real-world cost of false positives vs false
   negatives

4. **GridSearchCV's `model__` naming convention** — when tuning parameters inside a
   named Pipeline step, you must prefix the parameter with the step's name and two
   underscores (e.g., `"model__max_depth"`) so sklearn knows which step's parameter
   you're referring to

5. **Cross-validation inside hyperparameter search** — GridSearchCV doesn't just try
   each parameter combination once; it evaluates each combination across multiple
   folds (`cv=5` here) and averages the score, giving a more robust "best" choice
   than testing on a single train/test split alone

---

*Course Assignment Project (Day 31) — Decision Tree Classification, ShopSmart Purchase Prediction.
See MASTER_FUNCTIONS_CHEATSHEET.md (local, ML_Series root) for the cumulative
across-all-projects version.*
