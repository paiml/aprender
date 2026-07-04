import ProvableContracts.Defs.Gelu

/-!
# GELU Zero Preservation

Proves that `GELU(0) = 0`.

## Obligation

`ACT-GELU-ZERO`: GELU(0) = 0

The tanh-approximation GELU is `0.5·x·(1 + tanh(…))`. At `x = 0` the leading
factor `0.5·x` is zero, so the whole product is zero regardless of the tanh
argument. Closed by `ring` (which treats `tanh …` as an opaque atom).
-/

namespace ProvableContracts.Gelu

-- Status: proved
/-- GELU preserves zero: `GELU(0) = 0.5·0·(1 + tanh …) = 0`. -/
theorem gelu_zero : gelu 0 = 0 := by
  unfold gelu
  ring

-- Tests
#check @gelu_zero

example : gelu 0 = 0 := gelu_zero

end ProvableContracts.Gelu
