# CreditLens

An internal bank-staff analytics tool covering three stages of the customer
lifecycle: **Acquire → Lend → Retain**. Win the customer, decide whether to
lend to them, then keep them.

It trains five scikit-learn classifiers on three public banking datasets (all in US dollars) and
serves them through a FastAPI backend with a plain HTML/JS frontend. Pick a
dataset, pick a model, enter the customer's details, and get a prediction with
the model's estimated probability. The result box is green when the outcome is
good for the bank (offer accepted, loan repaid) and red when it is bad
(default, customer leaving).

**Not a production credit tool.** It is trained on public research datasets and
has never been validated for real lending or customer decisions.

## What it predicts

| Stage | Dataset | Target | Rows used | Features |
|---|---|---|---|---|
| Acquire | Kaggle Bank Personal Loan Modelling (Thera Bank) | Accepts a personal loan offer | 5,000 | 5 |
| Lend | Kaggle Credit Risk Dataset | Loan defaults | 32,409 (after cleaning) | 6 |
| Retain | Kaggle Credit Card Customers (BankChurners) | Card customer leaves the bank | 10,127 | 6 |

Every row is used. Features were chosen by permutation importance computed on the training split only
(`python -m creditlens.ml.feature_analysis`).

Each dataset is served by five models, and you choose which one at request time:
Logistic Regression, Random Forest, Histogram Gradient Boosting, an MLP neural
network, and a soft-voting ensemble of those four. Class weights are balanced
where the model supports it.

Hyperparameters come from a randomized search (20 candidates, 5-fold CV,
scored by ROC-AUC) on the training split. Each model's yes/no cut-off is the
probability that maximises F1 on out-of-fold training predictions, rather
than a fixed 0.5. The search runs once with `--tune` and is saved to
`creditlens/tuned_params.json`, so a normal training run reuses it without
searching.

## How it fits together

```
datasets/raw/ -> preprocess.py -> datasets/*.csv -> train.py -> models/
                                                                  |
                                            main.py (FastAPI) <---+
                                                  |
                                            static/ (browser)
```

## Running it

```bash
pip install -r requirements.txt
python -m creditlens.ml.preprocess
python -m creditlens.ml.train
uvicorn creditlens.main:app --reload
```

To redo the hyperparameter and cut-off search (about 5 minutes), run
`python -m creditlens.ml.train --tune` and commit the updated
`creditlens/tuned_params.json`.

Then open <http://127.0.0.1:8000>.

## Model performance

These scores come from a held-out 20% test split. CV Accuracy is 5-fold
stratified cross-validation on the training split only. Cut-off is the tuned
decision threshold each model uses for yes/no. Every number below comes from
`models/metrics.json`.

- **Gini** (2·AUC − 1) and **KS** (the maximum gap between the cumulative
  positive and negative rates) are the standard credit-scoring measures of how
  well a model ranks customers.
- **PR-AUC** is more informative than ROC-AUC when positives are rare.
- **Brier** score measures how good the probabilities are. Lower is better.
- The **majority-class baseline** always predicts the most common class. On
  imbalanced data its accuracy looks high, which is why accuracy alone is
  misleading here.

### Acquire - Personal Loan Offer (5,000 rows, 5 features)

| Model | Cut-off | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini | KS | PR-AUC | Brier | CV Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Majority-class baseline | - | 0.904 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.096 | 0.087 | - |
| Logistic Regression | 0.833 | 0.941 | 0.676 | 0.740 | 0.706 | 0.963 | 0.926 | 0.824 | 0.789 | 0.080 | 0.940 ± 0.008 |
| Random Forest | 0.290 | 0.989 | 0.913 | 0.979 | 0.945 | 0.999 | 0.998 | 0.976 | 0.993 | 0.007 | 0.988 ± 0.003 |
| Gradient Boosting (HistGB) | 0.783 | 0.992 | 0.949 | 0.969 | 0.959 | 0.999 | 0.998 | 0.979 | 0.993 | 0.009 | 0.987 ± 0.005 |
| Neural Network (MLP) | 0.279 | 0.990 | 0.930 | 0.969 | 0.949 | 0.997 | 0.993 | 0.962 | 0.984 | 0.007 | 0.987 ± 0.002 |
| Ensemble (soft voting) | 0.465 | 0.990 | 0.939 | 0.958 | 0.948 | 0.999 | 0.999 | 0.983 | 0.994 | 0.012 | 0.988 ± 0.005 |

### Lend - Loan Default (32,409 rows, 6 features)

| Model | Cut-off | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini | KS | PR-AUC | Brier | CV Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Majority-class baseline | - | 0.781 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.219 | 0.171 | - |
| Logistic Regression | 0.646 | 0.834 | 0.609 | 0.669 | 0.638 | 0.861 | 0.722 | 0.578 | 0.683 | 0.150 | 0.838 ± 0.006 |
| Random Forest | 0.578 | 0.934 | 0.953 | 0.736 | 0.830 | 0.939 | 0.878 | 0.738 | 0.893 | 0.067 | 0.930 ± 0.003 |
| Gradient Boosting (HistGB) | 0.727 | 0.937 | 0.973 | 0.731 | 0.835 | 0.948 | 0.896 | 0.756 | 0.904 | 0.069 | 0.934 ± 0.001 |
| Neural Network (MLP) | 0.402 | 0.928 | 0.954 | 0.705 | 0.811 | 0.924 | 0.847 | 0.707 | 0.868 | 0.063 | 0.921 ± 0.003 |
| Ensemble (soft voting) | 0.575 | 0.934 | 0.957 | 0.732 | 0.829 | 0.937 | 0.874 | 0.739 | 0.889 | 0.071 | 0.931 ± 0.004 |

### Retain - Card Customer Attrition (10,127 rows, 6 features)

| Model | Cut-off | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini | KS | PR-AUC | Brier | CV Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Majority-class baseline | - | 0.840 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.160 | 0.135 | - |
| Logistic Regression | 0.674 | 0.876 | 0.597 | 0.698 | 0.644 | 0.906 | 0.811 | 0.642 | 0.674 | 0.119 | 0.865 ± 0.007 |
| Random Forest | 0.429 | 0.954 | 0.860 | 0.849 | 0.854 | 0.983 | 0.966 | 0.885 | 0.912 | 0.037 | 0.961 ± 0.006 |
| Gradient Boosting (HistGB) | 0.740 | 0.960 | 0.869 | 0.880 | 0.875 | 0.987 | 0.975 | 0.893 | 0.938 | 0.037 | 0.963 ± 0.007 |
| Neural Network (MLP) | 0.338 | 0.958 | 0.848 | 0.895 | 0.871 | 0.988 | 0.976 | 0.895 | 0.940 | 0.031 | 0.956 ± 0.007 |
| Ensemble (soft voting) | 0.515 | 0.959 | 0.862 | 0.883 | 0.872 | 0.984 | 0.969 | 0.887 | 0.931 | 0.038 | 0.962 ± 0.007 |

### Why some scores are so high

- **Acquire:** the Thera Bank data is very clean (likely synthetic), and
  income alone separates most acceptors, so near-perfect scores are expected.
- **Lend:** `loan_grade` is the lender's own risk grade, so the model partly
  learns from an existing assessment rather than from scratch.
- **Retain:** the transaction totals cover the same 12 months in which the
  customer left, so they describe the leaving as much as predict it.

## The API

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"attrition","model_type":"ensemble","Total_Trans_Ct":40,"Total_Trans_Amt":2000,"Total_Relationship_Count":2,"Total_Revolving_Bal":0,"Total_Amt_Chng_Q4_Q1":0.5,"Total_Ct_Chng_Q4_Q1":0.4}'
```

`GET /api/datasets` lists the datasets, their input fields and bounds.
`GET /api/metrics` returns the tables above as JSON. `GET /docs` is the
auto-generated API reference.

## Layout

```
creditlens/       config, FastAPI app, model loading
creditlens/ml/    preprocessing, feature analysis, training, evaluation
static/           the frontend
tests/            pytest suite
datasets/raw/     the original source files
```

## Credits

Datasets: Kaggle Bank Personal Loan Modelling, Kaggle Credit Risk Dataset,
and Kaggle Credit Card Customers.
