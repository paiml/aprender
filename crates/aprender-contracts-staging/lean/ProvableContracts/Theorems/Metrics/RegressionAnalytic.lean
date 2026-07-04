import Mathlib

/-!
# Regression Metrics — Analytic Correctness (Mathlib-backed)

The three genuinely-analytic obligations of `metrics-regression-v1.yaml` that
are **infeasible core-only** (they require real analysis over `ℝ`, `Finset`
sums and the real square root). Companion to the sorry-free, Mathlib-free
`Regression.lean`, which discharges the four algebraic obligations.

Metrics are modelled over an arbitrary index `Finset s` and real-valued
targets `y`, predictions `ŷ`, residuals `eᵢ = yᵢ - ŷᵢ`:

* `SSres = Σ (yᵢ - ŷᵢ)²`     (residual sum of squares, `= n · MSE`)
* `SStot = Σ (yᵢ - ȳ)²`      (total sum of squares, `> 0` by hypothesis)
* `R²    = 1 - SSres / SStot`
* `MAE   = (Σ |eᵢ|) / n`,  `RMSE = √((Σ eᵢ²) / n)`   with `n = #s`.

All three theorems are proved sorry-free against Mathlib `master`.
-/

namespace ProvableContracts.Metrics.RegressionAnalytic

open Finset

/-! ## Obligation — R² upper bound (`R² ≤ 1`)

`R² = 1 - SSres / SStot`. Since `SSres = Σ (yᵢ - ŷᵢ)² ≥ 0` (a sum of squares)
and `SStot > 0`, the quotient is non-negative, so `1 - (nonneg) ≤ 1`.
Uses `Finset.sum_nonneg`, `sq_nonneg`, `div_nonneg` and `sub_le_self`. -/
theorem r_squared_le_one {ι : Type*} (s : Finset ι) (y yh : ι → ℝ)
    (sstot : ℝ) (h_tot : 0 < sstot) :
    1 - (∑ i ∈ s, (y i - yh i) ^ 2) / sstot ≤ 1 := by
  have hres : 0 ≤ ∑ i ∈ s, (y i - yh i) ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  have hquot : 0 ≤ (∑ i ∈ s, (y i - yh i) ^ 2) / sstot :=
    div_nonneg hres (le_of_lt h_tot)
  exact sub_le_self 1 hquot

/-! ## Obligation — perfect fit ⇒ `R² = 1`

When `ŷᵢ = yᵢ` for all `i`, every residual is `0`, so `SSres = 0` and
`R² = 1 - 0 / SStot = 1`. This is the `R² = 1` conjunct of the bundled
perfect-prediction identity that `Regression.lean` could not close core-only
(it needs real division). -/
theorem r_squared_perfect {ι : Type*} (s : Finset ι) (y : ι → ℝ)
    (sstot : ℝ) (_h_tot : 0 < sstot) :
    1 - (∑ i ∈ s, (y i - y i) ^ 2) / sstot = 1 := by
  have hzero : (∑ i ∈ s, (y i - y i) ^ 2) = 0 := by
    apply Finset.sum_eq_zero
    intro i _
    simp
  rw [hzero]
  simp

/-! ## Obligation — MAE ≤ RMSE (Cauchy–Schwarz / QM–AM)

By Cauchy–Schwarz (`sq_sum_le_card_mul_sum_sq`, the `f = g` case of Chebyshev):
`(Σ |eᵢ|)² ≤ #s · Σ |eᵢ|² = n · Σ eᵢ²`. Dividing by `n²` gives
`MAE² = (Σ|eᵢ|/n)² ≤ (Σ eᵢ²)/n = MSE`, and `Real.sqrt` monotonicity
(`Real.le_sqrt_of_sq_le`) yields `MAE ≤ √MSE = RMSE`. -/
theorem mae_le_rmse {ι : Type*} (s : Finset ι) (e : ι → ℝ)
    (hpos : 0 < (s.card : ℝ)) :
    (∑ i ∈ s, |e i|) / (s.card : ℝ)
      ≤ Real.sqrt ((∑ i ∈ s, (e i) ^ 2) / (s.card : ℝ)) := by
  set n : ℝ := (s.card : ℝ) with hn
  -- Cauchy–Schwarz on the absolute residuals.
  have hcs : (∑ i ∈ s, |e i|) ^ 2 ≤ (s.card : ℝ) * ∑ i ∈ s, |e i| ^ 2 :=
    sq_sum_le_card_mul_sum_sq
  -- |eᵢ|² = eᵢ², so the right factor is the SSE.
  have hsq : (∑ i ∈ s, |e i| ^ 2) = ∑ i ∈ s, (e i) ^ 2 := by
    apply Finset.sum_congr rfl
    intro i _
    exact sq_abs (e i)
  have hcs' : (∑ i ∈ s, |e i|) ^ 2 ≤ n * ∑ i ∈ s, (e i) ^ 2 := by
    rw [← hn] at hcs
    rw [hsq] at hcs
    exact hcs
  -- MAE ≤ √MSE via sq monotonicity of sqrt.
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, div_le_div_iff₀ (by positivity) hpos]
  nlinarith [hcs', hpos]

-- #check surface
#check @r_squared_le_one
#check @r_squared_perfect
#check @mae_le_rmse

end ProvableContracts.Metrics.RegressionAnalytic
