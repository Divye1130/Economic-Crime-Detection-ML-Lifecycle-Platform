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

That keeps model and decision policy versioned together.

## 8. Docker and CI

The repository includes a Dockerfile:

```bash
docker build -t economic-crime-ml .
docker run --rm -p 8000:8000 economic-crime-ml
```

It also includes a GitHub Actions workflow that installs dependencies and runs the test suite on pushes and pull requests.

## 9. Responsible AI and model-risk thinking

The model score is treated as a **review-prioritisation signal**, not an automatic adverse customer decision.

Controls demonstrated in the project include:

- chronological holdout periods
- validation-only model and threshold choices
- class-imbalance-aware metrics
- explicit investigation-capacity constraint
- input validation
- feature drift monitoring
- versioned model artifact
- documented human-review boundary

A real bank deployment would additionally require:

- governed data lineage and data-owner approvals
- independent model validation
- privacy and information-security review
- access controls and audit logging
- customer-outcome and fairness analysis where appropriate
- investigator feedback loops
- delayed-label performance monitoring
- controlled release, rollback and model-change governance
- incident management and retraining approval criteria

See `RESPONSIBLE_AI.md` and `MODEL_CARD.md` for the shorter governance artefacts.

## Repository structure

```text
.
├── index.html
├── README.md
├── MODEL_CARD.md
├── RESPONSIBLE_AI.md
├── Dockerfile
├── requirements.txt
├── TEST_RESULTS.txt
├── .github/
│   └── workflows/ci.yml
├── docs/
│   └── architecture.md
├── src/
│   ├── data_builder.py
│   ├── modeling.py
│   ├── monitoring.py
│   ├── api.py
│   └── run_pipeline.py
├── tests/
├── artifacts/
│   ├── fraud_model.joblib
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   ├── test_scored_transactions.csv
│   ├── feature_drift.csv
│   ├── weekly_monitoring.csv
│   └── monitoring_summary.json
└── data/
    └── outputs/
```

## Run from a clean environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_pipeline
pytest -q
```

Run the scoring service:

```bash
uvicorn src.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

for FastAPI's interactive API documentation.

## GitHub Pages deployment

The website is already arranged for a project repository:

```text
repository-root/
├── index.html
├── .nojekyll
├── README.md
└── ...
```

To deploy:

1. Upload or push the repository contents to GitHub.
2. Open **Settings -> Pages**.
3. Set **Source** to **Deploy from a branch**.
4. Select `main` and `/(root)`.
5. Save.

The dashboard is static and therefore does not require Python, FastAPI or Docker to be running on GitHub Pages.

## Automated verification

The current suite contains **15 tests** covering:

- deterministic row generation
- unique transaction identifiers
- chronological ordering
- feature-domain checks
- rare-event target-rate checks
- train/validation/test split sizes and temporal ordering
- non-negative review-loss calculations
- threshold alert-capacity behaviour
- PSI stability and shift detection
- drift-status boundaries
- model-artifact metadata
- model metric integrity
- API request validation and scoring
- monitoring-output reconciliation
- threshold-curve integrity

Current result:

```text
15 passed
```

## Interview explanation - 60 seconds

> I built an end-to-end economic-crime ML lifecycle rather than stopping at a fraud-classification notebook. I created 60,000 time-ordered transaction records, compared Logistic Regression with Gradient Boosting using validation PR-AUC, and kept Gradient Boosting because it performed better on the validation period. I then treated the probability threshold as a separate operational decision: I selected it using investigation cost and missed-fraud loss while constraining validation alerts to 1.5% of transactions. The held-out period achieved ROC-AUC 0.892 and PR-AUC 0.271. I packaged the model behind FastAPI and Docker, added CI and 15 automated tests, then monitored PSI and weekly alert behaviour. Device-age PSI reached 0.394 and the later alert rate rose from 1.36% to 1.77%, which gave me a concrete example of why deployment and monitoring matter as much as initial model accuracy.

## Interview explanation - 3 minutes

### Business framing
Fraud detection is a rare-event ranking and operations problem. I wanted to avoid building a model that looked good statistically but generated an impractical investigation queue.

### Data and validation
I generated a reproducible transaction dataset with amount, account age, device age, transaction velocity, merchant and country risk, channel and payment-context variables. I used time order and a 60/20/20 split so that all model choices were made before the final period.

### Modelling
I deliberately used an interpretable Logistic Regression baseline and a nonlinear Gradient Boosting challenger. Gradient Boosting reached validation PR-AUC 0.222 versus 0.194 for Logistic Regression, so I selected it without looking at the test result.

### Threshold selection
I did not use 0.5. I scanned operating thresholds using precision, recall, alert volume and an explicit review-loss function. I imposed a 1.5% validation alert-capacity constraint. The selected threshold was 0.0457, with 29.4% precision and 36.9% recall on validation.

### Held-out behaviour
On the later test period, ROC-AUC was 0.892 and PR-AUC was 0.271. Precision remained 29.7%, while recall was 33.3%. Alert rate rose to 1.77%, showing that operational workload can move after deployment even with a fixed threshold.

### Monitoring
I calculate PSI for selected features and weekly model statistics. Device age showed PSI 0.394, which I classify as material drift, and country-risk score showed PSI 0.172, a watch condition. That would trigger investigation rather than automatic retraining.

### Engineering
The selected pipeline and threshold are saved together, the scorer is exposed through a typed FastAPI endpoint, Docker packages the service, GitHub Actions runs tests, and the static dashboard can be served through GitHub Pages.

### Governance
I explicitly treat the score as investigator decision support. In a real bank I would expect independent validation, approved data lineage, security controls, customer-outcome testing, audit logging, controlled release and delayed-label performance monitoring before production use.

## Questions I would expect in an interview

### Why did you use a time split instead of stratified random splitting?
Because deployment is forward-looking. Random splitting can mix later transaction patterns into training and make the model appear more stable than it is. Time splitting also gave the monitoring layer a genuine later-period population to evaluate.

### Why choose PR-AUC?
The event rate is below 1%. Accuracy is dominated by legitimate transactions, and ROC-AUC can also look strong when the positive class is very rare. PR-AUC focuses directly on how well high-risk predictions separate true fraud from false alerts.

### Why is the threshold only 0.0457?
The threshold is not a statement that a transaction is definitely fraud. It is the point at which the transaction enters a review queue. With rare events, calibrated risk probabilities can be well below 0.5 even for high-ranking cases. The right threshold depends on cost and investigation capacity.

### Why did the held-out alert rate exceed the 1.5% validation constraint?
The constraint was deliberately applied only to validation data. The later population changed, particularly device-age and country-risk distributions. With the threshold held fixed, those changes pushed alert volume to 1.77%. That is exactly the kind of post-release behaviour monitoring should detect.

### Does PSI mean the model needs retraining?
No. PSI is a population-change indicator, not proof of model failure. I would combine it with precision/recall once labels mature, alert-volume changes, calibration, investigator feedback and business context before deciding whether to recalibrate or retrain.

### Why retain Logistic Regression if Gradient Boosting won?
It provides an interpretable challenger baseline. If a future boosted model stops materially outperforming it, the simpler model may be preferable. It also helps diagnose whether performance comes from genuine nonlinear structure or basic linear risk relationships.

### What would you add next?
A delayed-label monitoring process, investigator feedback capture, probability calibration monitoring, champion/challenger release logic, feature attribution for investigator context, secure authentication and audit logging, and a proper deployment/rollback path on a managed cloud service.

## Five numbers to remember

1. **60,000 transactions / 491 frauds / 0.82% event rate**
2. **Validation PR-AUC: 0.222; Gradient Boosting selected**
3. **Held-out ROC-AUC: 0.892; PR-AUC: 0.271**
4. **Selected threshold: 0.0457; validation alert rate: 1.36%**
5. **Device-age PSI: 0.394; held-out alert rate: 1.77%**
