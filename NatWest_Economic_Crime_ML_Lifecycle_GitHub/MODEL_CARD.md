# Model Card - Economic Crime Detection Model

## Intended use
Prioritise transactions for investigation using a fraud-risk score. The output is a decision-support signal rather than an autonomous customer decision.

## Model candidates
- Logistic Regression baseline
- Histogram Gradient Boosting challenger

Model selection uses validation PR-AUC because the positive class is rare. A separate operating threshold is selected on validation data under a capped alert-rate constraint and an explicit review/missed-fraud cost function.

## Data
The repository includes a deterministic transaction-data builder used to exercise the full lifecycle. It contains no customer or bank data. The builder creates transaction, device, velocity, geography-proxy and channel risk variables and introduces a later-period behaviour shift for monitoring.

## Key controls
- chronological train/validation/test split
- threshold selection isolated to validation data
- class-imbalance-aware metrics
- feature drift monitoring using PSI
- human review implied by alert workflow
- versioned model artifact and metrics

## Limitations
The included data builder is designed for engineering and evaluation of the lifecycle rather than representing a specific bank's portfolio. Real deployment would require approved production data, feature governance, independent model-risk validation, customer-outcome analysis, security review and ongoing outcome monitoring.
