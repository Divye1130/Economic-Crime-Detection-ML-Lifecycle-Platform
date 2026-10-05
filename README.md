# Economic Crime Detection & ML Lifecycle Platform

A complete transaction-risk machine-learning lifecycle covering **model development, threshold governance, serving, monitoring, automated testing and stakeholder reporting**.

The project is designed around a banking data-science problem where fraud is rare, investigation capacity is limited, and a model is only useful if its output can be translated into an operational policy and monitored after release.

## Project at a glance

- **60,000** time-ordered transaction records
- **491** fraud events (**0.82%** overall event rate)
- chronological **60/20/20** train-validation-test design
- Logistic Regression baseline and Gradient Boosting challenger
- model selection using validation **PR-AUC**
- separate operating-threshold selection using review capacity and a cost function
- selected model: **Gradient Boosting**
- held-out **ROC-AUC 0.892**
- held-out **PR-AUC 0.271**
- held-out precision **29.7%** and recall **33.3%** at the selected threshold
- validation alert rate **1.36%**; later held-out alert rate **1.77%**
- feature drift monitoring using **Population Stability Index (PSI)**
- FastAPI scoring service, Docker packaging and GitHub Actions CI
- **15/15 automated tests passing**
- static monitoring dashboard ready for GitHub Pages

## Why I built it this way

A basic fraud-classification notebook can report a strong ROC-AUC and still be unusable. Economic-crime teams have finite investigation capacity, false positives create operational cost and customer friction, and fraud patterns can change over time. I therefore treated the project as an **ML lifecycle** rather than a single modelling task.

The central design questions were:

1. Can the model rank rare fraud events better than an interpretable baseline?
2. What threshold should be used when only a limited percentage of transactions can be reviewed?
3. What happens to alert volume and model behaviour when the transaction population changes?
4. How should the model be packaged and monitored so it can be integrated into an application rather than living in a notebook?
5. What controls are needed so the score remains decision support rather than an opaque automatic customer decision?

## Data design

The repository includes a deterministic transaction-data builder. It does **not** contain customer or bank data.

Each record includes transaction, velocity, account, device and risk-context fields such as:

- transaction amount
- hour of day
- account age
- device age
- transactions in the previous hour and 24 hours
- distance from home
- merchant risk score
- card-present indicator
- international indicator
- new-payee indicator
- recent authentication failures
- amount relative to the customer's typical amount
- transaction channel
- country risk score

The later time period contains a deliberate behaviour change in channel mix, device age and new-payee activity. That lets the monitoring layer detect genuine population change instead of producing a dashboard where all features remain perfectly stationary.

The current generated dataset contains **60,000 transactions and 491 fraud events**.

## Architecture

```text
Transaction data builder
        |
        v
Chronological 60/20/20 split
        |
        +--------------------+
        |                    |
        v                    v
Logistic Regression   Gradient Boosting
   baseline              challenger
        |                    |
        +---------+----------+
                  |
                  v
       validation PR-AUC comparison
                  |
                  v
           selected model
                  |
                  v
       operating-threshold scan
  review cost + missed-fraud cost
       + maximum alert capacity
                  |
         +--------+---------+
         |                  |
         v                  v
  FastAPI scorer       monitoring layer
                       - PSI feature drift
                       - weekly alert rate
                       - weekly fraud rate
                       - average risk score
         |                  |
         +--------+---------+
                  v
       stakeholder dashboard
```

The static dashboard is independent of the API and therefore works on GitHub Pages. The FastAPI scoring service can be run locally or in Docker.

## 1. Temporal validation

Transactions are ordered by timestamp and split into:

- **36,000 training records**
- **12,000 validation records**
- **12,000 held-out test records**

I used a chronological split rather than a random split because a forward-looking fraud system should learn from earlier behaviour and be evaluated on later behaviour. It also makes drift visible: the held-out period contains a different population mix, so the model must operate under a modest distribution shift.

The validation set is used for:

- comparing candidate models
- selecting the operating threshold

The test set is used once for final evaluation and monitoring outputs.

## 2. Model design

### Logistic Regression baseline

The baseline provides a transparent benchmark. Numerical fields are standardised and transaction channel is one-hot encoded. It is important to retain this baseline because extra model complexity should have to earn its place.

### Gradient Boosting challenger

The challenger uses Histogram Gradient Boosting to capture nonlinear risk relationships and interactions without requiring manual rule expansion.

### Validation results

| Model | PR-AUC | ROC-AUC | Brier score |
|---|---:|---:|---:|
| Gradient Boosting | **0.222** | **0.869** | **0.00963** |
| Logistic Regression | 0.194 | 0.848 | 0.00972 |

Gradient Boosting therefore became the selected model. The choice was based on validation PR-AUC, not on model complexity or the held-out test result.

## 3. Why PR-AUC matters

Fraud represents less than 1% of the data. Raw accuracy is therefore a poor metric: a model that labels almost everything legitimate could still look highly accurate.

I report both ROC-AUC and PR-AUC, but use **PR-AUC for model selection** because it focuses attention on the quality of positive-event ranking under class imbalance.

The held-out model achieved:

- **ROC-AUC: 0.892**
- **PR-AUC: 0.271**
- **Brier score: 0.0138**

These are ranking and probability-quality metrics. They do not by themselves define the operational alert policy.

## 4. Threshold governance

A fraud model outputs a probability. Turning that probability into an alert is a separate decision.

I evaluated a grid of thresholds and calculated, for each operating point:

- precision
- recall
- F1
- alert rate
- expected review cost
- expected missed-fraud loss

The review-loss function is:

```text
review loss = false positives * review cost
            + sum(missed fraud amount * loss fraction + fixed miss cost)
```

The selected threshold is the lowest-cost validation policy among thresholds that keep the alert rate within a **1.5% validation capacity constraint**.

### Selected operating point

- threshold: **0.0457**
- validation precision: **29.4%**
- validation recall: **36.9%**
- validation alert rate: **1.36%**

On the later held-out population:

- precision: **29.7%**
- recall: **33.3%**
- alert rate: **1.77%**
- true positives: **63**
- false positives: **149**
- false negatives: **126**

The increase in alert rate from 1.36% to 1.77% is useful rather than something to hide: it demonstrates why a model needs post-release monitoring. A threshold calibrated to one period can produce a different operational workload when the population changes.

## 5. Drift monitoring

The monitoring layer uses **Population Stability Index (PSI)** for selected input features.

Thresholds are intentionally explicit:

- **Stable:** PSI < 0.10
- **Watch:** 0.10 <= PSI < 0.25
- **Material:** PSI >= 0.25

Current held-out drift results:

| Feature | PSI | Status |
|---|---:|---|
| device_age_days | **0.394** | Material |
| country_risk_score | **0.172** | Watch |
| merchant_risk_score | 0.0015 | Stable |
| amount_vs_customer_median | 0.0009 | Stable |
| transactions_last_24h | 0.0007 | Stable |
| amount_gbp | 0.0002 | Stable |

The strongest change is device age, which is consistent with the later-period behavioural shift. PSI is not proof that the model is wrong; it is a trigger for investigation, performance review and possible recalibration or retraining.

## 6. Weekly monitoring

The held-out period is also grouped into weekly monitoring windows. Each week records:

- transaction count
- observed fraud rate
- alert rate
- mean model score

This makes it possible to distinguish a one-off fluctuation from a sustained change in model workload or risk mix.

## 7. Serving layer

`src/api.py` exposes a FastAPI service with two endpoints:

### `GET /health`

Returns service status and confirms whether the model bundle is loaded.

### `POST /score`

Accepts a typed transaction schema and returns:

```json
{
  "fraud_probability": 0.183421,
  "threshold": 0.045687,
  "alert": true
}
```

Pydantic validates input ranges before scoring. The persisted model bundle contains:

- fitted preprocessing/model pipeline
- selected threshold
- expected feature list
