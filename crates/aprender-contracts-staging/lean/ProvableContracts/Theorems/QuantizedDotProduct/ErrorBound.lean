import ProvableContracts.Theorems.Quantization.RoundtripBound
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Quantized Dot-Product Error Bound

Proves the central correctness property of a quantized dot-product kernel:
dequantizing a quantized weight vector and taking its dot product with a
(full-precision) activation vector deviates from the exact dot product by at
most `(scale / 2) · ‖y‖₁`.

## Obligation

`QDOT-BND-001`:
  |⟨dequant(quant(x)), y⟩ − ⟨x, y⟩| ≤ (scale / 2) · Σᵢ |yᵢ|

## Proof strategy (analytic)

Purely `analytic`: no runtime measurement is involved.

1. **Linearity** — combine the two sums into a single sum of
   `(dequant(quant(xᵢ)) − xᵢ) · yᵢ` (distributivity of `Finset.sum`).
2. **Triangle inequality** — `|Σ …| ≤ Σ |…|` (`Finset.abs_sum_le_sum_abs`).
3. **Per-element round-trip bound** — for each term,
   `|(dequant(quant(xᵢ)) − xᵢ) · yᵢ| = |dequant(quant(xᵢ)) − xᵢ| · |yᵢ|
      ≤ (scale / 2) · |yᵢ|`, reusing the already-proved
   `roundtrip_bound : |dequant(quant(x)) − x| ≤ scale / 2`.
4. **Factor the constant** out of the sum (`Finset.mul_sum`).
-/

namespace ProvableContracts.QuantizedDotProduct

open ProvableContracts.Quantization

/-- Quantized dot-product error bound.

Let `x`, `y : ℕ → ℝ` be the weight and activation coordinate functions and
`scale > 0` the (positive) quantization scale.  Then the dot product computed
from the dequantized weights differs from the exact dot product by at most
`(scale / 2) · Σᵢ |yᵢ|`. -/
theorem quant_dot_error_bound
    (n : ℕ) (x y : ℕ → ℝ) (scale : ℝ) (hs : scale > 0) :
    |(∑ i ∈ Finset.range n, dequantize (quantize (x i) scale) scale * y i)
        - (∑ i ∈ Finset.range n, x i * y i)|
      ≤ (scale / 2) * ∑ i ∈ Finset.range n, |y i| := by
  -- Step 1: linearity — fold both sums into one.
  have hcombine :
      (∑ i ∈ Finset.range n, dequantize (quantize (x i) scale) scale * y i)
          - (∑ i ∈ Finset.range n, x i * y i)
        = ∑ i ∈ Finset.range n,
            (dequantize (quantize (x i) scale) scale - x i) * y i := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hcombine]
  calc
    |∑ i ∈ Finset.range n,
        (dequantize (quantize (x i) scale) scale - x i) * y i|
        ≤ ∑ i ∈ Finset.range n,
            |(dequantize (quantize (x i) scale) scale - x i) * y i| := by
          apply Finset.abs_sum_le_sum_abs
      -- Step 2 + 3: triangle inequality then per-element round-trip bound.
      _ ≤ ∑ i ∈ Finset.range n, (scale / 2) * |y i| := by
          apply Finset.sum_le_sum
          intro i _
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_right
            (roundtrip_bound (x i) scale hs) (abs_nonneg _)
      -- Step 4: factor the constant out.
      _ = (scale / 2) * ∑ i ∈ Finset.range n, |y i| := by
          rw [Finset.mul_sum]

#check @quant_dot_error_bound

end ProvableContracts.QuantizedDotProduct
