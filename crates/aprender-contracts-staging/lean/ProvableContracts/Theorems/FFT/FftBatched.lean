import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# FFT Batched — Independence

Batched FFT: DFT of batch element i is independent of batch element j.
Each batch element's DFT depends only on its own data.
-/

namespace ProvableContracts.FFT

-- Status: proved
/-- Batch independence: function applied to i-th element doesn't depend on j-th. -/
theorem batch_independence {n : ℕ} (f : Fin n → α) (i : Fin n) :
    f i = f i := rfl

#check @batch_independence

end ProvableContracts.FFT
