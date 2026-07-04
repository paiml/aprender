import ProvableContracts.Defs.Conv1D
import Mathlib.Data.List.Basic

/-!
# Conv1d Output-Length (shape) correctness

Proves the contract's output-shape formula for 1-D convolution:

  `L_out = ⌊(L + 2·pad − K) / stride⌋ + 1`.

## Obligation

`CV-INV-001` (contract `conv1d-kernel-v1.yaml`, proof obligation
"Output shape correctness"): `L_out = floor((L + 2*pad - K) / stride) + 1`.

Two theorems discharge it:

* `outLen_valid` — the general nat formula specialises, at `pad = 0`,
  `stride = 1` (valid convolution), to `L − K + 1`.
* `conv1d_valid_length` — the concrete `List`-valued valid convolution
  actually **produces** that many output elements, and this equals
  `outLen`.
-/

namespace ProvableContracts.Conv1D

open Finset

-- Status: proved
/-- The general output-length formula, at `pad = 0` and unit stride, reduces to
the valid-convolution length `L − K + 1`. -/
theorem outLen_valid (L K : ℕ) : outLen L K 0 1 = L - K + 1 := by
  simp [outLen]

-- Status: proved
/-- **Output-shape correctness.** The concrete valid convolution of a length-`L`
input with a length-`K` kernel produces exactly `L − K + 1` outputs. -/
theorem conv1d_valid_length (w x : List ℝ) :
    (conv1d_valid w x).length = x.length - w.length + 1 := by
  simp [conv1d_valid]

-- Status: proved
/-- The produced length matches the contract's `L_out` formula (valid mode). -/
theorem conv1d_valid_length_eq_outLen (w x : List ℝ) :
    (conv1d_valid w x).length = outLen x.length w.length 0 1 := by
  rw [conv1d_valid_length, outLen_valid]

#check @outLen_valid
#check @conv1d_valid_length
#check @conv1d_valid_length_eq_outLen

end ProvableContracts.Conv1D
