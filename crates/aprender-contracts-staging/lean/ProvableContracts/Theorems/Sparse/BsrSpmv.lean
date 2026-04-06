import ProvableContracts.Defs.Sparse
import Mathlib.Data.Matrix.Basic

/-!
# BSR SpMV — Block Structure Preserves Linearity

BSR (Block Sparse Row) SpMV is mathematically identical to dense MV.
The block structure is a storage optimization; the linear map is the same.
-/

namespace ProvableContracts.Sparse

open Matrix

-- Status: proved
/-- BSR SpMV is linear: spmv(A, αx + βy) = α·spmv(A,x) + β·spmv(A,y). -/
theorem bsr_spmv_linear {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (x y : Fin n → ℝ) (α β : ℝ) :
    spmv A (α • x + β • y) = α • spmv A x + β • spmv A y := by
  unfold spmv
  simp [mulVec_add, mulVec_smul]

#check @bsr_spmv_linear

end ProvableContracts.Sparse
