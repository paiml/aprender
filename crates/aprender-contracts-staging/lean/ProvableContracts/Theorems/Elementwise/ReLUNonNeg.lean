import ProvableContracts.Defs.Elementwise

/-!
# ReLU Non-Negativity

Proves that ReLU outputs are always non-negative: relu(x) >= 0.

## Obligation

`EW-INV-001`: ∀ x ∈ ℝ, relu(x) ≥ 0

Since relu(x) = max(0, x), the result is at least 0 by definition.
-/

namespace ProvableContracts.Elementwise

-- Status: proved
/-- ReLU is non-negative: max(0, x) ≥ 0. -/
theorem relu_nonneg (x : ℝ) : relu x ≥ 0 := by
  unfold relu
  exact le_max_left 0 x

-- Status: proved
/-- ReLU preserves non-negative inputs: relu(x) = x when x ≥ 0. -/
theorem relu_of_nonneg (x : ℝ) (hx : x ≥ 0) : relu x = x := by
  unfold relu
  exact max_eq_right hx

-- Status: proved
/-- ReLU of non-positive input is zero: relu(x) = 0 when x ≤ 0. -/
theorem relu_of_nonpos (x : ℝ) (hx : x ≤ 0) : relu x = 0 := by
  unfold relu
  exact max_eq_left hx

-- Tests
#check @relu_nonneg
#check @relu_of_nonneg
#check @relu_of_nonpos

example : relu 5.0 ≥ 0 := relu_nonneg 5.0
example : relu (-3.0) ≥ 0 := relu_nonneg (-3.0)
example : relu 0 ≥ 0 := relu_nonneg 0
example : relu 5 = 5 := relu_of_nonneg 5 (by norm_num)
example : relu (-3) = 0 := relu_of_nonpos (-3) (by norm_num)

end ProvableContracts.Elementwise
