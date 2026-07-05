import Mathlib

/-!
# Preprocessing Normalization — Analytic Correctness (Mathlib-backed)

The analytic obligations of `preprocessing-normalization-v1.yaml`.

We model a feature column as an index `Finset s` and a real-valued sample
`x : ι → ℝ`.  `n = #s`, `μ = mean s x = (Σ xᵢ)/n`, `σ > 0` the (population)
standard deviation, and the **standardized** value

  `z i = (x i - μ) / σ`.

Obligations discharged sorry-free against Mathlib `master`:

* **StandardScaler zero mean**   — `mean(z) = 0`  (`standardize_mean_zero`)
* **StandardScaler unit variance** — `Var(z) = 1` when `σ² = Var(x) > 0`
  (`standardize_variance_one`)
* **StandardScaler inverse**      — `z·σ + μ = x`  (`standardize_inverse`,
  an exact roundtrip / left-inverse of the affine transform)
* **MinMaxScaler extremes**       — `scale(xmin) = lo`, `scale(xmax) = hi`
  (`minmax_scale_min`, `minmax_scale_max`)
* **MinMaxScaler bounded**        — `xmin ≤ x ≤ xmax ⇒ lo ≤ scale(x) ≤ hi`
  (`minmax_scale_bounded`)

The remaining obligation `OBLIG-SCALER-F64-ACCUM` (f64 vs f32 accumulator
drift under catastrophic cancellation) is a runtime IEEE-754 floating-point
precision claim, not an analytic identity over ℝ, and is marked
`l4_not_applicable` in the contract — it cannot be stated over the reals.

## References
- Scikit-learn: Preprocessing data (StandardScaler, MinMaxScaler)
- Bishop (2006) Pattern Recognition and Machine Learning, §1.1
-/

namespace ProvableContracts.Preprocessing.Normalization

open Finset

variable {ι : Type*}

/-- Sample mean of a feature column over index set `s`. -/
noncomputable def mean (s : Finset ι) (x : ι → ℝ) : ℝ :=
  (∑ i ∈ s, x i) / (s.card : ℝ)

/-- Standardization transform `z i = (x i - μ) / σ`. -/
noncomputable def standardize (μ σ : ℝ) (x : ι → ℝ) : ι → ℝ :=
  fun i => (x i - μ) / σ

/-! ## Obligation — StandardScaler zero mean

`Σᵢ (xᵢ - μ) = (Σ xᵢ) - n·μ = (Σ xᵢ) - (Σ xᵢ) = 0` (since `μ = (Σ xᵢ)/n` and
`n > 0`), hence the mean of the standardized column is `(0/σ)/n = 0`. -/
theorem standardize_sum_centered_zero (s : Finset ι) (x : ι → ℝ)
    (hn : 0 < s.card) :
    ∑ i ∈ s, (x i - mean s x) = 0 := by
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
  unfold mean
  field_simp
  ring

/-- Mean of the standardized column is exactly `0`. -/
theorem standardize_mean_zero (s : Finset ι) (x : ι → ℝ) (σ : ℝ)
    (hn : 0 < s.card) :
    mean s (standardize (mean s x) σ x) = 0 := by
  have hsum : (∑ i ∈ s, standardize (mean s x) σ x i) = 0 := by
    unfold standardize
    rw [← Finset.sum_div, standardize_sum_centered_zero s x hn, zero_div]
  show (∑ i ∈ s, standardize (mean s x) σ x i) / (s.card : ℝ) = 0
  rw [hsum, zero_div]

/-! ## Obligation — StandardScaler unit variance

With `μ = mean s x`, the population variance of `x` is
`Var(x) = (Σ (xᵢ-μ)²)/n`.  The standardized column has mean `0`
(`standardize_mean_zero`), so its variance is its second moment:

  `Var(z) = (Σ zᵢ²)/n = (Σ (xᵢ-μ)²/σ²)/n = Var(x)/σ²`.

When `σ² = Var(x)` (the fitted std) and `σ ≠ 0`, this is exactly `1`. -/
theorem standardize_variance_one (s : Finset ι) (x : ι → ℝ) (σ : ℝ)
    (hn : 0 < s.card) (hσ : σ ≠ 0)
    (hvar : σ ^ 2 = (∑ i ∈ s, (x i - mean s x) ^ 2) / (s.card : ℝ)) :
    (∑ i ∈ s, (standardize (mean s x) σ x i
        - mean s (standardize (mean s x) σ x)) ^ 2) / (s.card : ℝ) = 1 := by
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  -- mean of z is 0
  rw [standardize_mean_zero s x σ hn]
  unfold standardize
  -- Σ ((xᵢ-μ)/σ - 0)² = Σ (xᵢ-μ)²/σ² = (Σ (xᵢ-μ)²)/σ²
  have hterm : ∀ i ∈ s, ((x i - mean s x) / σ - 0) ^ 2
      = (x i - mean s x) ^ 2 / σ ^ 2 := by
    intro i _
    rw [sub_zero, div_pow]
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_div]
  -- Σ (xᵢ-μ)² = σ² · n  from hvar
  have hnσ : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have hsum : (∑ i ∈ s, (x i - mean s x) ^ 2) = σ ^ 2 * (s.card : ℝ) := by
    rw [hvar]; field_simp
  rw [hsum]
  field_simp

/-! ## Obligation — StandardScaler inverse (exact roundtrip)

The inverse transform `x = z·σ + μ` recovers the original value exactly:
`((x - μ)/σ)·σ + μ = (x - μ) + μ = x`, for any `σ ≠ 0`. -/
theorem standardize_inverse (μ σ x : ℝ) (hσ : σ ≠ 0) :
    standardize μ σ (fun _ => x) 0 * σ + μ = x := by
  unfold standardize
  field_simp
  ring

/-- Pointwise inverse identity, independent of any index. -/
theorem standardize_roundtrip (μ σ x : ℝ) (hσ : σ ≠ 0) :
    ((x - μ) / σ) * σ + μ = x := by
  field_simp
  ring

/-! ## MinMaxScaler

`scale(x) = (x - xmin)/(xmax - xmin) · (hi - lo) + lo`, target range `[lo, hi]`. -/
noncomputable def minmax_scale (xmin xmax lo hi x : ℝ) : ℝ :=
  (x - xmin) / (xmax - xmin) * (hi - lo) + lo

/-! ## Obligation — MinMaxScaler extremes

`xmin ↦ lo` (the numerator vanishes) and `xmax ↦ hi` (the ratio is `1`,
requiring `xmax ≠ xmin`). -/
theorem minmax_scale_min (xmin xmax lo hi : ℝ) :
    minmax_scale xmin xmax lo hi xmin = lo := by
  unfold minmax_scale
  simp

theorem minmax_scale_max (xmin xmax lo hi : ℝ) (h : xmax ≠ xmin) :
    minmax_scale xmin xmax lo hi xmax = hi := by
  unfold minmax_scale
  have hne : xmax - xmin ≠ 0 := sub_ne_zero.mpr h
  field_simp
  ring

/-! ## Obligation — MinMaxScaler bounded

For training-range input `xmin ≤ x ≤ xmax` (with `xmin < xmax`) and a valid
target range `lo ≤ hi`, the scaled value lies in `[lo, hi]`.  The ratio
`t = (x-xmin)/(xmax-xmin) ∈ [0,1]`, so `scale(x) = t·(hi-lo) + lo ∈ [lo, hi]`. -/
theorem minmax_scale_bounded (xmin xmax lo hi x : ℝ)
    (hx : xmin < xmax) (hlohi : lo ≤ hi)
    (hlo : xmin ≤ x) (hhi : x ≤ xmax) :
    lo ≤ minmax_scale xmin xmax lo hi x ∧ minmax_scale xmin xmax lo hi x ≤ hi := by
  unfold minmax_scale
  have hden : 0 < xmax - xmin := sub_pos.mpr hx
  set t : ℝ := (x - xmin) / (xmax - xmin) with ht
  have ht0 : 0 ≤ t := div_nonneg (by linarith) (le_of_lt hden)
  have ht1 : t ≤ 1 := by
    rw [ht, div_le_one hden]; linarith
  have hrange : 0 ≤ hi - lo := by linarith
  constructor
  · nlinarith [mul_nonneg ht0 hrange]
  · nlinarith [mul_le_of_le_one_left hrange ht1]

/-! ## MinMaxScaler inverse roundtrip (FALSIFY-PP-006, bonus)

`inverse(scale(x)) = x`:  `((scale(x) - lo)/(hi-lo))·(xmax-xmin) + xmin = x`. -/
theorem minmax_inverse (xmin xmax lo hi x : ℝ)
    (hx : xmax ≠ xmin) (hr : hi ≠ lo) :
    ((minmax_scale xmin xmax lo hi x - lo) / (hi - lo)) * (xmax - xmin) + xmin = x := by
  unfold minmax_scale
  have hden : xmax - xmin ≠ 0 := sub_ne_zero.mpr hx
  have hran : hi - lo ≠ 0 := sub_ne_zero.mpr hr
  field_simp
  ring

-- #check surface
#check @standardize_mean_zero
#check @standardize_variance_one
#check @standardize_roundtrip
#check @minmax_scale_min
#check @minmax_scale_max
#check @minmax_scale_bounded
#check @minmax_inverse

end ProvableContracts.Preprocessing.Normalization
