import ProvableContracts.Defs.Elementwise

/-!
# Leaky ReLU Identities

Proves the two defining branches of Leaky ReLU:

  leaky_relu(α, x) = x       when x ≥ 0
  leaky_relu(α, x) = α · x   when x < 0

## Obligation

`EW-LEAKY-001`: ∀ α x ∈ ℝ, x ≥ 0 ⟹ leaky_relu α x = x
`EW-LEAKY-002`: ∀ α x ∈ ℝ, x < 0 ⟹ leaky_relu α x = α · x

These match the Rust `nn::functional::leaky_relu`, which computes
`if x > 0 { x } else { negative_slope * x }`; the two formulations agree
everywhere (at `x = 0` both branches yield `0`).

## References

- Maas et al. (2013) Rectifier Nonlinearities Improve Neural Network Acoustic Models
-/

namespace ProvableContracts.Elementwise

-- Status: proved
/-- Leaky ReLU is the identity on non-negative inputs: `x ≥ 0 ⟹ leaky_relu α x = x`. -/
theorem leaky_relu_of_nonneg (α x : ℝ) (hx : x ≥ 0) : leaky_relu α x = x := by
  unfold leaky_relu
  exact if_pos hx

-- Status: proved
/-- Leaky ReLU scales negative inputs by `α`: `x < 0 ⟹ leaky_relu α x = α · x`. -/
theorem leaky_relu_of_neg (α x : ℝ) (hx : x < 0) : leaky_relu α x = α * x := by
  unfold leaky_relu
  exact if_neg (not_le.mpr hx)

-- Status: proved
/-- Leaky ReLU with a non-negative slope is non-negative on non-negative inputs:
    a direct corollary of `leaky_relu_of_nonneg`. -/
theorem leaky_relu_nonneg_of_nonneg (α x : ℝ) (hx : x ≥ 0) : leaky_relu α x ≥ 0 := by
  rw [leaky_relu_of_nonneg α x hx]; exact hx

-- Tests
#check @leaky_relu_of_nonneg
#check @leaky_relu_of_neg
#check @leaky_relu_nonneg_of_nonneg

example : leaky_relu 0.01 5 = 5 := leaky_relu_of_nonneg 0.01 5 (by norm_num)
example : leaky_relu 0.5 (-4) = 0.5 * (-4) := leaky_relu_of_neg 0.5 (-4) (by norm_num)

end ProvableContracts.Elementwise
