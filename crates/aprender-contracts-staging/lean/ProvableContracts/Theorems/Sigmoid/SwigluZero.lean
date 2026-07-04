import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SiluZero

/-!
# SwiGLU Zero Preservation

Proves the zero-preservation invariant of the SwiGLU gated linear unit
from `swiglu-kernel-v1.yaml`.

## Equation

`SwiGLU(x, W, V, b, c) = SiLU(x·W + b) ⊙ (x·V + c)`

Per output component `i`, writing `g = (x·W + b)_i` for the gate
pre-activation and `v = (x·V + c)_i` for the value pre-activation:

`SwiGLU_i = SiLU(g) · v`.

## Obligation

`SG-INV-001` (Zero preservation): `SwiGLU(0, W, V, 0, 0) = 0`.

When the input `x = 0` and both biases `b = c = 0`, each affine
projection collapses to `0` (linearity: `0·W = 0`, `0·V = 0`), so every
gate pre-activation `g = 0` and every value pre-activation `v = 0`.
Hence each output component is `SiLU(0) · 0 = 0 · 0 = 0`.

This file proves the algebraic core of that invariant over ℝ. The
matrix-vector reduction `0·W = 0` is the standard linear-map-at-zero
identity; the nontrivial content is that a zero gate pre-activation
forces the output component to zero, which follows from `silu_zero`.
-/

namespace ProvableContracts.Sigmoid

open Real

/-- A single SwiGLU output component as a function of its gate
    pre-activation `g` and value pre-activation `v`:
    `swigluElem g v = SiLU(g) · v`. -/
noncomputable def swigluElem (g v : ℝ) : ℝ :=
  silu g * v

-- Status: proved
/-- Gate-zero collapse: if the gate pre-activation is `0`, the SwiGLU
    output component is `0` for **any** value pre-activation `v`, since
    `SiLU(0) = 0` annihilates the product. -/
theorem swiglu_gate_zero (v : ℝ) : swigluElem 0 v = 0 := by
  unfold swigluElem
  rw [silu_zero]
  ring

-- Status: proved
/-- Zero preservation (`SG-INV-001`): with zero input and zero biases,
    both pre-activations are `0`, so the SwiGLU output component is `0`.
    `SwiGLU(0, W, V, 0, 0)_i = SiLU(0) · 0 = 0`. -/
theorem swiglu_zero_preservation : swigluElem 0 0 = 0 :=
  swiglu_gate_zero 0

-- Tests
#check @swiglu_gate_zero
#check @swiglu_zero_preservation

example : swigluElem 0 0 = 0 := swiglu_zero_preservation
example (v : ℝ) : swigluElem 0 v = 0 := swiglu_gate_zero v
example : swigluElem 0 3.14 = 0 := swiglu_gate_zero 3.14

end ProvableContracts.Sigmoid
