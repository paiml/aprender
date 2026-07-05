import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SwigluZero

/-!
# SwiGLU Gating Identity

Proves the gating-decomposition invariant of the SwiGLU unit from
`swiglu-kernel-v1.yaml` (equation `swiglu`, invariant "Decomposable as
gate * value where gate = SiLU(xW+b)"):

`SwiGLU(g, v) = SiLU(g) · v`

i.e. every output component factors as the SiLU-activated gate
pre-activation times the value pre-activation. This is the algebraic
contract the fused kernel must honour componentwise.

## Obligation

`SG-GATE-001` (Gating identity): the fused SwiGLU output equals
`SiLU(gate) · value`.
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Gating identity: a SwiGLU output component is the SiLU-activated gate
    pre-activation times the value pre-activation. -/
theorem swiglu_gating_identity (g v : ℝ) : swigluElem g v = silu g * v := rfl

-- Status: proved
/-- Consequence: when the value pre-activation is `0`, the output is `0`
    for any gate (value-side annihilation), complementing `swiglu_gate_zero`. -/
theorem swiglu_value_zero (g : ℝ) : swigluElem g 0 = 0 := by
  rw [swiglu_gating_identity]; ring

-- Tests
#check @swiglu_gating_identity
#check @swiglu_value_zero

example (g v : ℝ) : swigluElem g v = silu g * v := swiglu_gating_identity g v
example (g : ℝ) : swigluElem g 0 = 0 := swiglu_value_zero g

end ProvableContracts.Sigmoid
