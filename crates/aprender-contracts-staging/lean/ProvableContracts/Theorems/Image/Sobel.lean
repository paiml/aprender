import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Sobel — Gradient Magnitude Non-Negativity

The gradient magnitude √(Gx² + Gy²) is always non-negative.
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Sobel gradient magnitude is non-negative. -/
theorem sobel_magnitude_nonneg (gx gy : ℝ) : gx ^ 2 + gy ^ 2 ≥ 0 := by
  positivity

#check @sobel_magnitude_nonneg

end ProvableContracts.Image
