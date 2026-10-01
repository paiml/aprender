# Spike 014 — regressor values on NeuralProphet's imputed grid days

Re-scoped: the door refuses `freq != "D"` for neuralprophet (`forecast.rs:421`),
so there is no weekly/month-start mapping. The daily-gap case is what remains.

Daily series: **790 observed** rows over a **899-day** grid -> **109 imputed days**
(12.1% of the grid), including three 15-day blackouts. Drivers: `promo` (binary,
true effect +9.0) and `price` (continuous, true effect -0.6 per unit).

## Does the imputed-day value get read at all?

Trains with the imputed days filled by each rule and compares against `linear interp`.
`garbage (1e3)` is the falsification probe: if it changes nothing, the value is unread.

| n_lags | rule | pred bit-identical to linear | max abs d pred | promo w | price w |
|---|---|---|---|---|---|
| 0 | linear interp | **yes** | 0.000e0 | +0.2546 | -0.0493 |
| 0 | zero fill | **yes** | 0.000e0 | +0.2546 | -0.0493 |
| 0 | carry forward | **yes** | 0.000e0 | +0.2546 | -0.0493 |
| 0 | garbage (1e3) | **yes** | 0.000e0 | +0.2546 | -0.0493 |
| 7 | linear interp | **yes** | 0.000e0 | +0.2640 | -0.0048 |
| 7 | zero fill | no | 7.978e0 | +0.2707 | -0.0018 |
| 7 | carry forward | no | 1.048e1 | +0.2453 | -0.0086 |
| 7 | garbage (1e3) | no | 1.922e2 | -0.1920 | +0.5711 |

Scale 35.32; a `promo` weight w denormalises to w x scale.
