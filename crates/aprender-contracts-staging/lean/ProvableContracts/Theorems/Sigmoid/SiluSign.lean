import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SigmoidBounded

/-!
# SiLU Sign Preservation

Proves that `sign(SiLU(x)) = sign(x)`: SiLU is positive on the positives,
negative on the negatives, and zero at zero.

## Obligation

`SI-SIGN-001`: x > 0 → SiLU(x) > 0 ∧ x < 0 → SiLU(x) < 0

Since SiLU(x) = x · σ(x) and σ(x) > 0 for all x, the sign of SiLU(x)
is exactly the sign of x.
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- SiLU is positive on positive inputs: x > 0 → SiLU(x) > 0. -/
theorem silu_pos_of_pos {x : ℝ} (hx : 0 < x) : 0 < silu x := by
  unfold silu
  exact mul_pos hx (sigmoid_pos x)

-- Status: proved
/-- SiLU is negative on negative inputs: x < 0 → SiLU(x) < 0. -/
theorem silu_neg_of_neg {x : ℝ} (hx : x < 0) : silu x < 0 := by
  unfold silu
  exact mul_neg_of_neg_of_pos hx (sigmoid_pos x)

-- Status: proved
/-- Sign preservation: SiLU shares the sign of its argument. -/
theorem silu_sign (x : ℝ) :
    (0 < x → 0 < silu x) ∧ (x < 0 → silu x < 0) :=
  ⟨fun h => silu_pos_of_pos h, fun h => silu_neg_of_neg h⟩

-- Tests
#check @silu_pos_of_pos
#check @silu_neg_of_neg
#check @silu_sign

end ProvableContracts.Sigmoid
