# Spike 011 — Prophet external regressors, parity ladder

## retail_sales + 4 regressors

### rung 0 — standardisation constants (history rows only)

| regressor | mode | rust mu | python mu | rust std | python std | d mu | d std |
|---|---|---|---|---|---|---|---|
| promo | Additive | 0.0000000000 | 0.0000000000 | 1.0000000000 | 1.0000000000 | 0.0e0 | 0.0e0 |
| price | Additive | 107.3772162309 | 107.3772162309 | 6.5114415813 | 6.5114415813 | 0.0e0 | 2.7e-15 |
| discount | Multiplicative | 0.0999146758 | 0.0999146758 | 0.0354459296 | 0.0354459296 | 8.3e-17 | 2.8e-17 |
| weather | Additive | 0.0096738043 | 0.0096738043 | 5.7754536319 | 5.7754536319 | 0.0e0 | 3.6e-15 |

### rung 1 — data prep exact

| quantity | result |
|---|---|
| base columns (no regressors) | 20 |
| columns with regressors | 24 (python 24) |
| column ORDER identical to python | yes |
| max abs d prior_scales | 0.00e0 |
| max abs d s_a | 0.00e0 |
| max abs d s_m | 0.00e0 |
| max abs d X (first3 + last3 rows) | 3.33e-15 |
| max abs d standardisation constants | 3.55e-15 |

### rung 3 — predict at Python MAP (y_scale 518253.000)

| quantity | max abs diff | relative to y_scale |
|---|---|---|
| yhat | 2.33e-10 | 4.49e-16 |
| trend | 1.75e-10 | 3.37e-16 |

| component | max abs diff | compared |
|---|---|---|
| additive_terms | 3.64e-11 | yes |
| discount | 1.39e-17 | yes |
| extra_regressors_additive | 2.18e-11 | yes |
| extra_regressors_multiplicative | 1.39e-17 | yes |
| multiplicative_terms | 1.39e-17 | yes |
| price | 2.18e-11 | yes |
| promo | 0.00e0 | yes |
| weather | 9.09e-13 | yes |
| yearly | 2.18e-11 | yes |

9 components compared, worst 3.64e-11

### rung 4 — Rust MAP fit with regressor columns

| quantity | value |
|---|---|
| status | Stalled |
| rounds / iters | 2 / 939 |
| fit seconds | 0.02 |
| max abs d yhat vs python forecast | 5214.2050 (1.01% of y_scale) |
| objective at python MAP (unscaled) | -1022.4943 |
| objective at rust fit (unscaled) | -1025.9109 |
| **slack f_rust - f_python** | **-3.4166** (contract bar 0.5; negative = rust found a BETTER optimum) |

| regressor | python beta | rust beta | r vs t (trend) | max r vs other X col | identifiable? |
|---|---|---|---|---|---|
| promo | +0.001404 | +0.001427 | -0.006 | +0.022 (`weather`) | yes |
| price | +0.036035 | +0.183202 | +0.647 | +0.759 (`yearly_delim_1`) | yes |
| discount | -0.004205 | -0.009709 | -0.013 | +0.999 (`yearly_delim_4`) | **no — collinear** |
| weather | -0.001322 | -0.001268 | -0.021 | +0.169 (`yearly_delim_1`) | yes |

## retail_sales + 2 holidays (windows) + 4 regressors

### rung 0 — standardisation constants (history rows only)

| regressor | mode | rust mu | python mu | rust std | python std | d mu | d std |
|---|---|---|---|---|---|---|---|
| promo | Additive | 0.0000000000 | 0.0000000000 | 1.0000000000 | 1.0000000000 | 0.0e0 | 0.0e0 |
| price | Additive | 107.3772162309 | 107.3772162309 | 6.5114415813 | 6.5114415813 | 0.0e0 | 2.7e-15 |
| discount | Multiplicative | 0.0999146758 | 0.0999146758 | 0.0354459296 | 0.0354459296 | 8.3e-17 | 2.8e-17 |
| weather | Additive | 0.0096738043 | 0.0096738043 | 5.7754536319 | 5.7754536319 | 0.0e0 | 3.6e-15 |

### rung 1 — data prep exact

| quantity | result |
|---|---|
| base columns (no regressors) | 26 |
| columns with regressors | 30 (python 30) |
| column ORDER identical to python | yes |
| max abs d prior_scales | 0.00e0 |
| max abs d s_a | 0.00e0 |
| max abs d s_m | 0.00e0 |
| max abs d X (first3 + last3 rows) | 3.33e-15 |
| max abs d standardisation constants | 3.55e-15 |

### rung 3 — predict at Python MAP (y_scale 518253.000)

| quantity | max abs diff | relative to y_scale |
|---|---|---|
| yhat | 1.75e-10 | 3.37e-16 |
| trend | 1.16e-10 | 2.25e-16 |

| component | max abs diff | compared |
|---|---|---|
| additive_terms | 2.91e-11 | yes |
| blackfriday | 0.00e0 | yes |
| discount | 1.47e-17 | yes |
| extra_regressors_additive | 2.18e-11 | yes |
| extra_regressors_multiplicative | 1.47e-17 | yes |
| holidays | 0.00e0 | yes |
| multiplicative_terms | 1.47e-17 | yes |
| newyear | 0.00e0 | yes |
| price | 2.18e-11 | yes |
| promo | 0.00e0 | yes |
| weather | 9.09e-13 | yes |
| yearly | 2.91e-11 | yes |

12 components compared, worst 2.91e-11

### rung 4 — Rust MAP fit with regressor columns

| quantity | value |
|---|---|
| status | Stalled |
| rounds / iters | 7 / 1342 |
| fit seconds | 0.05 |
| max abs d yhat vs python forecast | 3726.2207 (0.72% of y_scale) |
| objective at python MAP (unscaled) | -1022.4623 |
| objective at rust fit (unscaled) | -1024.6849 |
| **slack f_rust - f_python** | **-2.2226** (contract bar 0.5; negative = rust found a BETTER optimum) |

| regressor | python beta | rust beta | r vs t (trend) | max r vs other X col | identifiable? |
|---|---|---|---|---|---|
| promo | +0.001415 | +0.001413 | -0.006 | +0.022 (`weather`) | yes |
| price | +0.034187 | +0.085746 | +0.647 | +0.759 (`yearly_delim_1`) | yes |
| discount | -0.004352 | -0.005994 | -0.013 | +0.999 (`yearly_delim_4`) | **no — collinear** |
| weather | -0.001312 | -0.001321 | -0.021 | +0.169 (`yearly_delim_1`) | yes |
