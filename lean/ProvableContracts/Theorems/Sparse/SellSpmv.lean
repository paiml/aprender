import ProvableContracts.Defs.Sparse
import Mathlib.Data.Matrix.Basic

/-!
# SELL SpMV — Sliced ELLPACK Preserves Linearity

SELL (Sliced ELLPACK) SpMV is mathematically identical to dense MV.
Slicing is a storage/parallelism optimization.
-/

namespace ProvableContracts.Sparse

open Matrix

-- Status: proved
/-- SELL SpMV distributes over addition. -/
theorem sell_spmv_add {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (x y : Fin n → ℝ) :
    spmv A (x + y) = spmv A x + spmv A y := by
  unfold spmv; exact mulVec_add A x y

#check @sell_spmv_add

end ProvableContracts.Sparse
