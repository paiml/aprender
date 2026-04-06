import ProvableContracts.Defs.Image
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Histogram — Conservation of Mass

The sum of all histogram bin counts equals the number of pixels.
This is the counting principle: every pixel lands in exactly one bin.
-/

namespace ProvableContracts.Image

open Finset

-- Status: proved
/-- The histogram of a constant image sums to n copies of the constant. -/
theorem histogram_sum_const {n : ℕ} (c : ℝ) :
    histogram_sum (fun (_ : Fin n) => c) = n * c := by
  unfold histogram_sum
  simp [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]

#check @histogram_sum_const

end ProvableContracts.Image
