# 🛒 ShopSmart Purchase Prediction | Decision Tree Classification

**Course Assignment Project (Day 31) — worked through as part of my ML Learning Series.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📌 Project Overview

ShopSmart, an e-commerce company, wants to predict whether a website visitor will complete a purchase based on their browsing session behavior — pages visited, time spent, bounce/exit rates, traffic source, and more.

**Note on origin:** This project was built by working through a Decision Tree pipeline template provided by my course instructor as a graded assignment. I wrote and ran every line myself and documented my own understanding of each step below, including a gap I found in the provided feature-selection logic (see "Key Findings" section). I'm labeling this honestly as coursework rather than an independently-designed project — the goal here was deep understanding of a full sklearn Pipeline + hyperparameter tuning workflow, not original architecture design.

**Goal:** Build a Decision Tree classifier to predict purchase intent (`Revenue`), evaluated by F1 score against a required benchmark of 0.55 (dataset is imbalanced, so F1 rather than accuracy was specified).

---

## 📊 Dataset

- **Source:** Online Shoppers Purchasing Intention dataset
- **Size:** 12,330 individual user sessions over 1 year
- **Target variable:** `Revenue` (binary: purchase made or not)
- **Class balance:** ~15.5% positive class — genuinely imbalanced

| Feature Type | Columns |
|---|---|
| Numerical (14) | Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, OperatingSystems, Browser, Region, TrafficType |
| Categorical (2) | Month, VisitorType |
| Not used by model | Weekend (see Key Findings below) |

---

## 🛠️ Pipeline Architecture

```
Raw Data → Train-Test Split (80/20, stratified) → ColumnTransformer
   ├── Numerical → StandardScaler
   └── Categorical → OneHotEncoder (handle_unknown="ignore")
→ Decision Tree Classifier (class_weight="balanced")
→ GridSearchCV hyperparameter tuning (5-fold CV, scored by F1)
```

All wrapped in a single `sklearn.pipeline.Pipeline` object, so preprocessing and modeling happen together with one `.fit()` call.

---

## 🤖 Model & Results

**Initial Model:** `max_depth=6, min_samples_leaf=30, class_weight="balanced"`

| Metric | Initial Model | After GridSearchCV Tuning |
|--------|---------------|----------------------------|
| F1 Score | 0.628 | **0.634 (best)** |
| Best Params | — | `max_depth=4, min_samples_leaf=50` |

**Benchmark: 0.55** — both models comfortably exceeded this.

### Classification Report (Initial Model)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Purchase) | 0.97 | 0.85 | 0.90 | 2,084 |
| 1 (Purchase) | 0.50 | 0.83 | 0.63 | 382 |

**Overall Accuracy:** 0.85

---

## 🔍 Key Finding

While reviewing the feature-selection code, I noticed:
```python
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns
```

The `Weekend` column is a **boolean** type, which matches neither selection — so it was silently excluded from the `ColumnTransformer` entirely and never reached the model. This is a common, easy-to-miss gotcha with `select_dtypes()`. Re-including `Weekend` (via a third transformer or explicit type conversion) is a natural next experiment.

---

## 🔑 Key Learnings

1. **ColumnTransformer + Pipeline** bundle preprocessing and modeling into one object — the same transformations automatically apply to train and test data with a single `.fit()`/`.predict()` call, reducing data leakage risk
2. **`class_weight="balanced"`** improves recall on a minority class but trades off precision — the classification report makes this tradeoff visible and quantifiable
3. **`select_dtypes()` has blind spots** — boolean columns don't match typical numeric/categorical `include` filters and can silently disappear from preprocessing
4. **GridSearchCV's `model__` prefix** is required to target hyperparameters nested inside a named Pipeline step
5. Understanding *why* a benchmark metric (F1, not accuracy) was chosen — matters directly with imbalanced data

## 🔮 Future Improvements
- Fix and re-test with the `Weekend` column properly included
- Expand the GridSearchCV parameter range beyond the tested 3×3 grid
- Compare against Random Forest for a precision/recall tradeoff comparison
- Attempt a similar pipeline independently, without an instructor template, as a self-test

---

## 🛠️ Tech Stack
`Python` `pandas` `numpy` `scikit-learn` `Jupyter Notebook`

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook shop_smart.ipynb
```

---

## 📬 Connect
Part of my ongoing ML learning series — a mix of self-directed projects and course assignments, all documented as I work through fundamentals toward Data Scientist roles.
