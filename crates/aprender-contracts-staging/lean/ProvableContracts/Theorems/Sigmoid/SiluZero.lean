import ProvableContracts.Defs.Sigmoid

/-!
# SiLU Zero Preservation

Proves that SiLU(0) = 0.

## Obligation

`SI-INV-001`: SiLU(0) = 0

Since SiLU(x) = x · σ(x), at x = 0 we get 0 · σ(0) = 0.
-/

namespace ProvableContracts.Sigmoid

-- Status: proved
/-- SiLU preserves zero: SiLU(0) = 0 · σ(0) = 0. -/
theorem silu_zero : silu 0 = 0 := by
  unfold silu
  ring

-- Tests
#check @silu_zero

example : silu 0 = 0 := silu_zero

end ProvableContracts.Sigmoid
