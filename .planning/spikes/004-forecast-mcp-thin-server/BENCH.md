| points | model | fit s | predict s | total s | L-BFGS rounds/iters/evals |
|---|---|---|---|---|---|
| 1000 | prophet | 0.136 | 0.015 | 0.152 | 4/471/3163 |
| 1000 | neuralprophet | 0.135 | 0.000 | 0.135 | epochs 110 steps 3520 |
| 1000 | neuralprophet n_lags=30 | 1.219 | 0.006 | 1.224 | epochs 110 steps 3410 |
| 3000 | prophet | 0.200 | 0.015 | 0.215 | 4/794/1530 |
| 3000 | neuralprophet | 0.158 | 0.001 | 0.159 | epochs 80 steps 3760 |
| 3000 | neuralprophet n_lags=30 | 2.747 | 0.012 | 2.759 | epochs 80 steps 3760 |
| 10000 | prophet | 1.174 | 0.013 | 1.188 | 4/1850/2708 |
| 10000 | neuralprophet | 0.303 | 0.002 | 0.305 | epochs 60 steps 4740 |
| 10000 | neuralprophet n_lags=30 | 6.935 | 0.036 | 6.971 | epochs 60 steps 9360 |
| 20000 | prophet | 8.191 | 0.013 | 8.207 | 3/6000/9302 |
