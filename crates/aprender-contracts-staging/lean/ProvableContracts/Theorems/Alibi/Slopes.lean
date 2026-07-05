import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# ALiBi Head-Slope Exponent (`alibi-slopes-v1`)

The ALiBi (Attention with Linear Biases) per-head slope is

  `m[h] = 2 ^ (-8 * (h + 1) / n)`,   `h ∈ {0, …, n-1}`, `n ≥ 1`

(Press, Smith & Lewis 2021, *"Train Short, Test Long"*; llama.cpp ggml
`soft_max_ext`: `m0 = 2^(-8/n)`, `slope = m0^(h+1)`).

These are the analytic obligations of the contract, proved over ℝ with
`Real.rpow`:

* **AS-001 (equivalence)** — head-0 slope equals `m0 = 2^(-8/n)`, and in
  particular `m[0] = 1/2` when `n = 8`.
* **AS-002 (bound)** — every slope is strictly below `1`
  (`m[h] < 1` for `h ≥ 0`, `n > 0`).
* **AS-003 (equivalence)** — the closed form matches the ggml reference
  `m[h] = m0 ^ (h+1)`.

Bonus analytic facts (the "positive, strictly-decreasing geometric sequence"
framing): slopes are strictly positive and strictly decreasing in `h`.
-/

namespace ProvableContracts.Alibi

open Real

/-- ALiBi base ratio `m0 = 2 ^ (-8 / n)`. -/
noncomputable def alibiM0 (n : ℝ) : ℝ := (2 : ℝ) ^ (-8 / n)

/-- ALiBi per-head slope `m[h] = 2 ^ (-8 * (h + 1) / n)`. -/
noncomputable def alibiSlope (n h : ℝ) : ℝ := (2 : ℝ) ^ (-8 * (h + 1) / n)

/-! ## AS-001 — head-zero slope equals `m0` -/

-- Status: proved
/-- The head-0 slope equals the base ratio `m0 = 2^(-8/n)`; the `(h+1)`
    offset collapses to `1`. -/
theorem alibi_head_zero (n : ℝ) : alibiSlope n 0 = alibiM0 n := by
  unfold alibiSlope alibiM0
  congr 1
  ring

-- Status: proved
/-- Concrete: for `n = 8` the head-0 slope is `1/2` (NOT `1.0`). -/
theorem alibi_head_zero_eight : alibiSlope 8 0 = 1 / 2 := by
  unfold alibiSlope
  have h : (-8 * ((0 : ℝ) + 1) / 8) = -1 := by norm_num
  rw [h, Real.rpow_neg_one]
  norm_num

/-! ## AS-002 — slopes are strictly below one -/

-- Status: proved
/-- Every ALiBi slope is strictly positive (base `2 > 0`). -/
theorem alibi_slope_pos (n h : ℝ) : 0 < alibiSlope n h := by
  unfold alibiSlope
  exact Real.rpow_pos_of_pos (by norm_num) _

-- Status: proved
/-- Every ALiBi slope is strictly below `1`: the exponent
    `-8(h+1)/n` is negative for `h ≥ 0`, `n > 0`, and base `2 > 1`. -/
theorem alibi_slope_lt_one {n h : ℝ} (hn : 0 < n) (hh : 0 ≤ h) :
    alibiSlope n h < 1 := by
  unfold alibiSlope
  apply Real.rpow_lt_one_of_one_lt_of_neg (by norm_num : (1 : ℝ) < 2)
  apply div_neg_of_neg_of_pos _ hn
  linarith

/-! ## AS-003 — matches the ggml reference exponent -/

-- Status: proved
/-- The closed form equals the ggml recurrence `m[h] = m0 ^ (h+1)`
    with `m0 = 2^(-8/n)`, via `(x^y)^z = x^(y·z)`. -/
theorem alibi_slope_ggml (n h : ℝ) :
    alibiSlope n h = (alibiM0 n) ^ (h + 1) := by
  unfold alibiSlope alibiM0
  rw [← Real.rpow_mul (by norm_num : (0 : ℝ) ≤ 2)]
  congr 1
  ring

/-! ## Bonus — strictly decreasing geometric sequence -/

-- Status: proved
/-- Slopes are strictly decreasing in the head index `h` (`m0` largest):
    `h₁ < h₂ → m[h₂] < m[h₁]`. -/
theorem alibi_slope_strict_anti {n h₁ h₂ : ℝ} (hn : 0 < n) (h : h₁ < h₂) :
    alibiSlope n h₂ < alibiSlope n h₁ := by
  unfold alibiSlope
  rw [Real.rpow_lt_rpow_left_iff (by norm_num : (1 : ℝ) < 2)]
  rw [div_lt_div_iff_of_pos_right hn]
  linarith

-- Tests
#check @alibi_head_zero
#check @alibi_head_zero_eight
#check @alibi_slope_pos
#check @alibi_slope_lt_one
#check @alibi_slope_ggml
#check @alibi_slope_strict_anti

example : alibiSlope 8 0 = 1 / 2 := alibi_head_zero_eight
example : (0 : ℝ) < alibiSlope 8 3 := alibi_slope_pos 8 3
example : alibiSlope 8 0 = alibiM0 8 := alibi_head_zero 8

end ProvableContracts.Alibi
