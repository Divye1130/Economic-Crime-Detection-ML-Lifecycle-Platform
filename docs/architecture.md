# Architecture

```text
Transaction data builder
        |
        v
Chronological 60/20/20 split
        |
        +--> Logistic Regression baseline
        |
        +--> Gradient Boosting challenger
                  |
                  v
        Validation PR-AUC selection
                  |
                  v
       Cost-aware threshold selection
                  |
          +-------+-------+
          |               |
          v               v
   FastAPI scorer     Monitoring layer
                      - PSI feature drift
                      - weekly score/alert rate
                      - outcome metrics
          |               |
          +-------+-------+
                  v
       GitHub Pages dashboard
```

The static dashboard uses precomputed project outputs, so it can be hosted on GitHub Pages without a backend. The FastAPI service is separately runnable locally or in a container for transaction scoring.
