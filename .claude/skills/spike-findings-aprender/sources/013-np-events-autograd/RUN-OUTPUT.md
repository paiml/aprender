# Spike 013 — NeuralProphet events on the autograd

Synthetic daily series, 1200 points. Known event effect **+8.0** per active
indicator; 6 event columns (2 events x windows). Series scale = 30.43.

## A/B — does the block train, and is the known effect recovered?

| run | params | epochs | steps | final train loss | MAE vs truth |
|---|---|---|---|---|---|
| events ON | 37 | 110 | 4180 | 0.00035 | 0.403 |
| events OFF | 31 | 110 | 4180 | 0.00237 | 0.697 |

| event column | learned weight (normalised) | denormalised | truth |
|---|---|---|---|
| promo_-1 | +0.25532 | **+7.769** | +8.0 |
| promo_+0 | +0.27257 | **+8.293** | +8.0 |
| promo_+1 | +0.26260 | **+7.990** | +8.0 |
| blackfriday_+0 | +0.26433 | **+8.043** | +8.0 |
| blackfriday_+1 | +0.27111 | **+8.249** | +8.0 |
| blackfriday_+2 | +0.27812 | **+8.462** | +8.0 |

Worst |denormalised - truth| = **0.462** (5.8% of the true effect)

## C — determinism at a fixed seed with the new block

| check | result |
|---|---|
| seed 42 twice: predictions bit-identical | **yes** |
| seed 42 twice: event weights bit-identical | **yes** |
| seed 43 differs (the check can fail) | **yes** |
| tape length per step (events ON / OFF) | 25 / 20 |

## D — the D-10 train-cost bound with new input dimensions

`train_cost(n_samples, epochs, n_lags) = epochs * n_samples * (n_lags + 1)`
has no term for event columns. Measured cost per step, events ON vs OFF:

| run | event cols | priced cost | seconds | us/step | ratio vs OFF |
|---|---|---|---|---|---|
| events OFF | 0 | 132000 | 0.053 | 12.6 | 1.00x |
| events ON | 6 | 132000 | 0.068 | 16.3 | **1.29x** |
| events ON (E=7) | 7 | 132000 | 0.065 | 15.6 | **1.24x** |
| events ON (E=28) | 28 | 132000 | 0.079 | 18.9 | **1.50x** |
| events ON (E=84) | 84 | 132000 | 0.090 | 21.6 | **1.71x** |
| events ON (E=210) | 210 | 132000 | 0.134 | 31.9 | **2.53x** |
| events ON (E=504) | 504 | 132000 | 0.242 | 58.0 | **4.59x** |
| events ON (E=1001) | 1001 | 132000 | 0.418 | 100.1 | **7.93x** |

Priced cost is IDENTICAL on every row (132000), because the formula has no
event term. Measured work is linear in E: us/step ~= 12.9 + 0.085*E on this box,
so at E = MAX_HOLIDAY_COLUMNS (1000) a request buys ~7.6x the work it was priced at.
