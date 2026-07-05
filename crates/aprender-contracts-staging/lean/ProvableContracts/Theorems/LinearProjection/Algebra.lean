import ProvableContracts.Defs.LinearProjection
import Mathlib.Data.Matrix.Basic

/-!
# Linear Projection — Analytic Obligations

Proofs for the analytic proof obligations of `linear-projection-v1`:

- `LP-SHP-001` Output shape / element correctness — `element_formula`
- `LP-LIN-001` Homogeneity without bias — `homogeneity`
- `LP-BIA-001` Bias additivity — `bias_additivity`
- `LP-ZER-001` Zero input produces bias — `zero_input_bias`

Supporting linearity lemmas (`additivity`, `zero_no_bias`) and the
`composition is a projection` corollary are included as the full linear-map
characterisation.

The SIMD/scalar ULP-equivalence obligation is NOT proved here: it is an
empirical floating-point (IEEE-754 accumulation-order) property, not an
algebraic identity, and is marked `l4_not_applicable` in the contract.
-/

namespace ProvableContracts.LinearProjection

open Matrix
open scoped BigOperators

variable {batch d_in d_out : ℕ}

-- Status: proved
/-- **LP-LIN-001 (homogeneity).** The no-bias projection is homogeneous:
    `(c • x) Wᵀ = c • (x Wᵀ)`. -/
theorem homogeneity
    (c : ℝ)
    (x : Matrix (Fin batch) (Fin d_in) ℝ)
    (W : Matrix (Fin d_out) (Fin d_in) ℝ) :
    linearNoBias (c • x) W = c • linearNoBias x W := by
  unfold linearNoBias
  exact Matrix.smul_mul c x Wᵀ

-- Status: proved
/-- **Additivity.** The no-bias projection distributes over input addition:
    `(x + x') Wᵀ = x Wᵀ + x' Wᵀ`. Together with `homogeneity` this establishes
    that `linearNoBias · W` is a linear map. -/
theorem additivity
    (x x' : Matrix (Fin batch) (Fin d_in) ℝ)
    (W : Matrix (Fin d_out) (Fin d_in) ℝ) :
    linearNoBias (x + x') W = linearNoBias x W + linearNoBias x' W := by
  unfold linearNoBias
  exact Matrix.add_mul x x' Wᵀ

-- Status: proved
/-- **Zero preservation (no bias).** `0 · Wᵀ = 0`. -/
theorem zero_no_bias
    (W : Matrix (Fin d_out) (Fin d_in) ℝ) :
    linearNoBias (0 : Matrix (Fin batch) (Fin d_in) ℝ) W = 0 := by
  unfold linearNoBias
  exact Matrix.zero_mul Wᵀ

-- Status: proved
/-- **LP-BIA-001 (bias additivity).** The full projection is exactly the
    no-bias projection offset by the broadcast bias — the bias is added
    independently of the matmul. -/
theorem bias_additivity
    (x : Matrix (Fin batch) (Fin d_in) ℝ)
    (W : Matrix (Fin d_out) (Fin d_in) ℝ)
    (b : Fin d_out → ℝ) :
    linearForward x W b = linearNoBias x W + biasBroadcast b :=
  rfl

-- Status: proved
/-- **LP-ZER-001 (zero input produces bias).** With zero input the projection
    collapses to the broadcast bias: `linearForward 0 W b = b`. -/
theorem zero_input_bias
    (W : Matrix (Fin d_out) (Fin d_in) ℝ)
    (b : Fin d_out → ℝ) :
    linearForward (0 : Matrix (Fin batch) (Fin d_in) ℝ) W b
      = biasBroadcast b := by
  unfold linearForward
  rw [zero_no_bias, zero_add]

-- Status: proved
/-- **LP-SHP-001 (output shape / element correctness).** Every output element
    at position `(i, k)` (with `i : Fin batch`, `k : Fin d_out`) is well-defined
    and equals `∑ⱼ xᵢⱼ Wₖⱼ + bₖ`. The dependent type
    `Matrix (Fin batch) (Fin d_out) ℝ` witnesses the `(batch, d_out)` shape;
    this theorem pins the value at each valid index, so no index outside
    `Fin batch × Fin d_out` is ever produced. -/
theorem element_formula
    (x : Matrix (Fin batch) (Fin d_in) ℝ)
    (W : Matrix (Fin d_out) (Fin d_in) ℝ)
    (b : Fin d_out → ℝ)
    (i : Fin batch) (k : Fin d_out) :
    linearForward x W b i k = (∑ j, x i j * W k j) + b k := by
  unfold linearForward linearNoBias biasBroadcast
  simp [Matrix.add_apply, Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply]

-- Status: proved
/-- **Composition is a projection.** Composing two no-bias projections
    `x ↦ x Wᵀ ↦ (x Wᵀ) Vᵀ` is itself a single no-bias projection with weight
    `(V * W)`, i.e. `linearNoBias (linearNoBias x W) V = linearNoBias x (V * W)`.
    (`(x Wᵀ) Vᵀ = x (Wᵀ Vᵀ) = x (V W)ᵀ`.) -/
theorem composition_is_projection
    (x : Matrix (Fin batch) (Fin d_in) ℝ)
    {d_mid : ℕ}
    (W : Matrix (Fin d_mid) (Fin d_in) ℝ)
    (V : Matrix (Fin d_out) (Fin d_mid) ℝ) :
    linearNoBias (linearNoBias x W) V = linearNoBias x (V * W) := by
  unfold linearNoBias
  rw [Matrix.transpose_mul, Matrix.mul_assoc]

-- Tests
#check @homogeneity
#check @additivity
#check @zero_no_bias
#check @bias_additivity
#check @zero_input_bias
#check @element_formula
#check @composition_is_projection

end ProvableContracts.LinearProjection
