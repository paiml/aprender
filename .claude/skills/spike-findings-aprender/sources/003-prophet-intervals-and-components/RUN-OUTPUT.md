# Spike 003 — Prophet intervals, components, holidays, logistic growth, multiplicative seasonality


---

## `peyton_manning` — mode **default**, growth linear, seasonality additive (2905 rows, K = 26)

1. Design parity: columns "[yearly_delim_1 … weekly_delim_6]" identical (26 cols), s_a/s_m/prior_scales identical, t/y_scaled/changepoints max diff 0.0e0, X first/last rows 0.0e0
2. Gradient vs central FD at a perturbed MAP (all 54 params incl. linear trend): worst rel 1.5e-7 (param 44), worst abs 1.3e-5
3. Fit: 6 restarts, 1312 iters [r0:Stalled/203→-7999.572618 r1:Stalled/187→-8004.137545 r2:Stalled/176→-8004.806819 r3:Stalled/230→-8004.925279 r4:Stalled/516→-8004.928690 r5:Stalled/0→-8004.928690], 3.106s, f = -8004.9287 vs Python -8004.7980 (Δ -0.131); Python fit 0.32s
   Forecast vs Python (y range 7.58): yhat hist max|Δ| 0.0041, future max|Δ| 0.0030, trend future 0.0025; Python params through Rust predict(): yhat 5.3e-15, trend 3.6e-15
4. Components (Python params through Rust): max|Δ| per component vs Python — yearly 5.6e-16, weekly 1.1e-16, additive_terms 6.7e-16, multiplicative_terms 0.0e0; trend·(1+multiplicative_terms)+additive_terms reconstructs yhat to 0.0e0
5. 80% intervals (1000 draws) — mean band width: history Rust 1.2377 vs Python 1.2376; future Rust 1.2907 (seed 7: 1.3092) vs Python 1.2936; last 30 days Rust 1.4229 vs Python 1.4273; future trend band Rust 0.2610 vs Python 0.2716
   In-sample coverage of the 80% band: Rust 0.882
6. Holdout (train 2540 / test 365): Rust coverage ["0.816", "0.833", "0.811"] (mean width ["1.309", "1.309", "1.316"], MAE 0.4131) vs Python coverage ["0.827", "0.825", "0.814"] (mean width ["1.317", "1.310", "1.317"], MAE 0.4125)

---

## `peyton_manning` — mode **holidays**, growth linear, seasonality additive (2905 rows, K = 30)

1. Design parity: columns "[yearly_delim_1 … superbowl_delim_+1]" identical (30 cols), s_a/s_m/prior_scales identical, t/y_scaled/changepoints max diff 0.0e0, X first/last rows 0.0e0, 25 holiday-indicator ones in X
2. Gradient vs central FD at a perturbed MAP (all 58 params incl. linear trend): worst rel 4.4e-8 (param 55), worst abs 1.2e-5
3. Fit: 4 restarts, 1193 iters [r0:Stalled/665→-8155.635077 r1:Stalled/357→-8155.755699 r2:Stalled/171→-8155.755729 r3:Stalled/0→-8155.755729], 1.808s, f = -8155.7557 vs Python -8155.2205 (Δ -0.535); Python fit 0.32s
   Forecast vs Python (y range 7.58): yhat hist max|Δ| 0.0230, future max|Δ| 0.0207, trend future 0.0050; Python params through Rust predict(): yhat 7.1e-15, trend 7.1e-15
4. Components (Python params through Rust): max|Δ| per component vs Python — yearly 4.4e-16, weekly 1.1e-16, playoff 0.0e0, superbowl 0.0e0, holidays 0.0e0, additive_terms 6.7e-16, multiplicative_terms 0.0e0; trend·(1+multiplicative_terms)+additive_terms reconstructs yhat to 0.0e0
5. 80% intervals (1000 draws) — mean band width: history Rust 1.1725 vs Python 1.1748; future Rust 1.2352 (seed 7: 1.2556) vs Python 1.2353; last 30 days Rust 1.3882 vs Python 1.3790; future trend band Rust 0.2783 vs Python 0.2759
   In-sample coverage of the 80% band: Rust 0.873

---

## `wp_log_R` — mode **logistic**, growth logistic, seasonality additive (2863 rows, K = 26)

1. Design parity: columns "[yearly_delim_1 … weekly_delim_6]" identical (26 cols), s_a/s_m/prior_scales identical, t/y_scaled/changepoints max diff 0.0e0, X first/last rows 0.0e0
2. Gradient vs central FD at a perturbed MAP (all 54 params incl. logistic γ-recursion): worst rel 4.5e-8 (param 22), worst abs 1.1e-5
3. Fit: 8 restarts, 815 iters [r0:Stalled/38→-9004.135721 r1:Stalled/209→-9019.576143 r2:Stalled/50→-9019.672735 r3:Stalled/166→-9019.828262 r4:Stalled/74→-9019.855823 r5:Stalled/130→-9019.862670 r6:Stalled/58→-9019.864538 r7:Stalled/90→-9019.865960], 0.929s, f = -9019.8660 vs Python -9019.6702 (Δ -0.196); Python fit 0.20s
   Forecast vs Python (y range 4.98): yhat hist max|Δ| 0.0028, future max|Δ| 0.0018, trend future 0.0016; Python params through Rust predict(): yhat 2.7e-15, trend 2.7e-15
4. Components (Python params through Rust): max|Δ| per component vs Python — yearly 1.1e-16, weekly 1.7e-16, additive_terms 2.2e-16, multiplicative_terms 0.0e0; trend·(1+multiplicative_terms)+additive_terms reconstructs yhat to 0.0e0
5. 80% intervals (1000 draws) — mean band width: history Rust 0.5979 vs Python 0.5983; future Rust 0.5980 (seed 7: 0.5992) vs Python 0.6004; last 30 days Rust 0.5940 vs Python 0.6024; future trend band Rust 0.0024 vs Python 0.0024
   In-sample coverage of the 80% band: Rust 0.918

---

## `air_passengers` — mode **multiplicative**, growth linear, seasonality multiplicative (144 rows, K = 20)

1. Design parity: columns "[yearly_delim_1 … yearly_delim_20]" identical (20 cols), s_a/s_m/prior_scales identical, t/y_scaled/changepoints max diff 0.0e0, X first/last rows 0.0e0
2. Gradient vs central FD at a perturbed MAP (all 48 params incl. linear trend): worst rel 2.1e-9 (param 29), worst abs 2.0e-6
3. Fit: 8 restarts, 678 iters [r0:Stalled/99→-502.678107 r1:Stalled/103→-503.397108 r2:Stalled/110→-503.427815 r3:Stalled/85→-503.438685 r4:Stalled/52→-503.439112 r5:Stalled/79→-503.439292 r6:Stalled/41→-503.439293 r7:Stalled/109→-503.439293], 0.042s, f = -503.4393 vs Python -503.3958 (Δ -0.044); Python fit 0.02s
   Forecast vs Python (y range 518.00): yhat hist max|Δ| 0.8039, future max|Δ| 26.2541, trend future 0.2985; Python params through Rust predict(): yhat 3.4e-13, trend 1.7e-13
4. Components (Python params through Rust): max|Δ| per component vs Python — yearly 1.1e-16, additive_terms 0.0e0, multiplicative_terms 1.1e-16; trend·(1+multiplicative_terms)+additive_terms reconstructs yhat to 0.0e0
5. 80% intervals (1000 draws) — mean band width: history Rust 26.2119 vs Python 26.3067; future Rust 26.3064 (seed 7: 26.2645) vs Python 26.3773; last 30 days Rust 26.4748 vs Python 26.4695; future trend band Rust 1.4315 vs Python 1.4186
   In-sample coverage of the 80% band: Rust 0.771

Wrote results.json and report.html (four panels: y, Rust yhat, Rust 80% band, Python band edges).
