import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import ProvableContracts.Basic

/-!
# Absolute Positional Encoding Definitions

Definitions for absolute positional encoding as used by transformer models
(GPT-2 / BERT class, `absolute-position-v1.yaml`). Two flavours are modeled:

1. **Learned additive** encoding — `output[t] = token_embed[t] + pos_embed[t]`
   (the on-disk kernel `abs_position_scalar`).
2. **Sinusoidal** encoding (Vaswani et al. 2017) —
   `PE(pos, 2i)   = sin(pos / 10000^(2i/d))`,
   `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`.

The sinusoidal components are `Real.sin` / `Real.cos` of the angle
`pos * ω(i)` where `ω(i) = 1 / 10000^(2i/d)` is the per-dimension angular
frequency. These definitions back the analytic proof obligations:
bounded components, known zero-position value, and relative-position
linear rotation (angle-addition).

## References

- Vaswani et al. (2017) Attention Is All You Need
-/

namespace ProvableContracts.AbsolutePosition

open ProvableContracts

/-- Learned additive positional encoding on a single position vector:
`output = token + pos`, elementwise. Shape (index type `Fin n`) is preserved
by construction. -/
def abs_add {n : ℕ} (token pos : RVec n) : RVec n :=
  fun i => token i + pos i

/-- Per-dimension angular frequency for the sinusoidal encoding:
`ω(i) = 1 / 10000^(2i/d)`. -/
noncomputable def omega (d : ℕ) (i : ℕ) : ℝ :=
  1 / (10000 : ℝ) ^ ((2 * (i : ℝ)) / (d : ℝ))

/-- Sinusoidal PE even component: `PE(pos, 2i) = sin(pos · ω(i))`. -/
noncomputable def pe_even (d : ℕ) (pos : ℝ) (i : ℕ) : ℝ :=
  Real.sin (pos * omega d i)

/-- Sinusoidal PE odd component: `PE(pos, 2i+1) = cos(pos · ω(i))`. -/
noncomputable def pe_odd (d : ℕ) (pos : ℝ) (i : ℕ) : ℝ :=
  Real.cos (pos * omega d i)

end ProvableContracts.AbsolutePosition
