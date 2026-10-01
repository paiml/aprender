# Spike 012 — no-argument bitwise invariance

## A. Baseline: same input twice through `forecast()` (determinism)

| case | signature | repeat | identical | fit s |
|---|---|---|---|---|
| peyton/prophet/default | `aa669c2352dd376a` | `aa669c2352dd376a` | yes | 1.39 |
| air/prophet/multiplicative | `3053244dcb27492c` | `3053244dcb27492c` | yes | 0.01 |
| retail/prophet/default | `73b171523eb3fa3b` | `73b171523eb3fa3b` | yes | 0.19 |
| wp_log_R/prophet/logistic | `331d67a8f8c924bb` | `331d67a8f8c924bb` | yes | 0.76 |
| peyton/prophet/holidays+windows | `e0a13f9a10e3c3cc` | `e0a13f9a10e3c3cc` | yes | 1.29 |
| peyton/neuralprophet/lag0 | `d5a583795591a3db` | `d5a583795591a3db` | yes | 0.16 |
| peyton/neuralprophet/lag7 | `726043759dde45dc` | `726043759dde45dc` | yes | 0.47 |
| wp_log_R/neuralprophet/lag0 | `72528357591ad4d6` | `72528357591ad4d6` | yes | 0.15 |

All repeat calls identical: **yes**

## B. Mutation proof — the signature detects a 1-ULP change

| mutation | signature | detected |
|---|---|---|
| yhat[0] += 1 ULP (8.56059168341344368e0 -> 8.56059168341344545e0) | `5b5d2076f95dc573` | **yes** |
| trend[last] += 1 ULP | `89c76ef81da8296d` | **yes** |
| one extra component key | `a06b8e1b24a62d90` | **yes** |
| all mutations reverted | `aa669c2352dd376a` | back to clean |

## C. Mechanism — spike-011 regressor plumbing with ZERO regressors

Compares the crate's untouched `predict` against the prototype path that carries
the regressor machinery but is handed an empty regressor list.

| dataset | design identical | yhat bits | trend bits | components | verdict |
|---|---|---|---|---|---|
| peyton_manning.csv | yes | identical | identical | 3 identical: yes | INERT ✓ |
| retail_sales.csv | yes | identical | identical | 3 identical: yes | INERT ✓ |
| air_passengers.csv | yes | identical | identical | 3 identical: yes | INERT ✓ |
