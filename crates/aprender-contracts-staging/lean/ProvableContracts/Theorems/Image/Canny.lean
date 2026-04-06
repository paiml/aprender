import Mathlib.Data.Real.Basic

/-!
# Canny Edge Detection — Threshold Ordering

Proves the fundamental invariant of hysteresis thresholding:
low_threshold < high_threshold implies the strong edge set is
a subset of the weak edge set.
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Strong edges are a subset of weak edges by threshold ordering. -/
theorem canny_threshold_ordering (low high mag : ℝ)
    (h_order : low < high) (h_strong : mag ≥ high) : mag > low := by
  linarith

#check @canny_threshold_ordering

end ProvableContracts.Image
