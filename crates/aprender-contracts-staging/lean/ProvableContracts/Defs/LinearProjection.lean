import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# Linear Projection Definitions

Dense-layer forward pass `y = x Wᵀ + b`.

- `x : Matrix (Fin batch) (Fin d_in) ℝ`   input activations (row per sample)
- `W : Matrix (Fin d_out) (Fin d_in) ℝ`   weight matrix (row per output feature)
- `b : Fin d_out → ℝ`                       bias, broadcast across the batch

The projection is `x * Wᵀ` (row-major matmul against the transposed weight),
optionally offset by the broadcast bias. These definitions reuse Mathlib's
`Matrix.mul` / `Matrix.transpose`, so the algebraic obligations (linearity,
additivity, zero preservation, bias offset, element/shape formula) are all
discharged by the existing `Matrix` module lemmas.

## References

- Bishop (2006) Pattern Recognition and Machine Learning, §5.1 (linear layers).
-/

namespace ProvableContracts.LinearProjection

open Matrix

variable {batch d_in d_out : ℕ}

/-- No-bias linear projection: `y = x * Wᵀ`. -/
noncomputable def linearNoBias
    (x : Matrix (Fin batch) (Fin d_in) ℝ)
    (W : Matrix (Fin d_out) (Fin d_in) ℝ) :
    Matrix (Fin batch) (Fin d_out) ℝ :=
  x * Wᵀ

/-- Bias broadcast: replicate the `d_out`-vector `b` across every one of the
    `batch` rows. -/
def biasBroadcast (b : Fin d_out → ℝ) :
    Matrix (Fin batch) (Fin d_out) ℝ :=
  Matrix.of (fun _ (j : Fin d_out) => b j)

/-- Full linear projection: `y = x * Wᵀ + b` (bias broadcast over the batch). -/
noncomputable def linearForward
    (x : Matrix (Fin batch) (Fin d_in) ℝ)
    (W : Matrix (Fin d_out) (Fin d_in) ℝ)
    (b : Fin d_out → ℝ) :
    Matrix (Fin batch) (Fin d_out) ℝ :=
  linearNoBias x W + biasBroadcast b

end ProvableContracts.LinearProjection
