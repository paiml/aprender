import Mathlib.Data.Real.Basic

/-!
# Image Resize — Bilinear Interpolation Bounds

Bilinear interpolation of values in [0,1] produces values in [0,1].
The interpolated value is a convex combination.
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Linear interpolation stays in bounds. -/
theorem lerp_bounded (a b t : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (ha1 : a ≤ 1) (hb1 : b ≤ 1) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    0 ≤ (1 - t) * a + t * b ∧ (1 - t) * a + t * b ≤ 1 := by
  constructor
  · nlinarith
  · nlinarith

#check @lerp_bounded

end ProvableContracts.Image
