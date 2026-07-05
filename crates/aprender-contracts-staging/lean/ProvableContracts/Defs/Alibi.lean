import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.EReal.Basic

/-!
# ALiBi (Attention with Linear Biases) — Definitions

Mathematical model of the ALiBi positional-encoding kernel, matching the
`alibi-kernel-v1.yaml` contract equations:

  bias(i, j) = -m_h · |i - j|              (linear in the integer distance)
  m_h        = 2^(-8h/H)                    (geometric per-head slope schedule)

Positions `i, j` are integers (token offsets), so the *distance* is exact
integer arithmetic; the slope `m_h` and the resulting bias live in `ℝ`.
Causal masking sends future positions to `-∞`, faithfully modelled by the
bottom element `⊥` of the extended reals `EReal` (so that `exp(⊥) = 0` under a
subsequent softmax, i.e. future tokens receive zero attention weight).

## References

- Press, Smith, Lewis (2022) *Train Short, Test Long: Attention with Linear
  Biases Enables Input Length Extrapolation*.
-/

namespace ProvableContracts.Alibi

open Real

/-- Integer distance between two token positions `i` and `j`. -/
def dist (i j : ℤ) : ℤ := |i - j|

/-- ALiBi additive bias: `bias(i,j) = -m · |i - j|`.
    Linear in the (integer) distance with per-head slope `m`. -/
noncomputable def alibiBias (m : ℝ) (i j : ℤ) : ℝ := -m * (dist i j : ℝ)

/-- Per-head ALiBi slope `m_h = 2^(-8h/H)` — a geometric schedule with ratio
    `2^(-8/H) ∈ (0,1)` across heads `h = 0 … H-1`. -/
noncomputable def slope (H h : ℕ) : ℝ := (2 : ℝ) ^ (-(8 * (h : ℝ)) / (H : ℝ))

/-- Causal-masked ALiBi bias over the extended reals: future positions
    (`j > i`) are masked to `-∞` (`⊥`); non-future positions keep the finite
    linear bias. -/
noncomputable def causalBias (m : ℝ) (i j : ℤ) : EReal :=
  if j > i then ⊥ else ((alibiBias m i j : ℝ) : EReal)

end ProvableContracts.Alibi
