# Responsible AI and Model-Risk Controls

The fraud score is treated as a review-prioritisation signal, not an automatic adverse customer decision.

Controls demonstrated:
- forward-looking chronological holdout periods
- PR-AUC, precision, recall and false-positive monitoring rather than accuracy alone
- cost-aware thresholding constrained by alert capacity
- feature-drift monitoring with transparent PSI thresholds
- fixed input schema through Pydantic
- model/version metadata stored with the artifact
- documented human-review boundary

Before any real deployment, additional controls would include governed data lineage, privacy/security review, independent validation, customer-outcome testing, audit logging, access controls, change approval and incident response.
