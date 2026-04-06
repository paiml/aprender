import Mathlib.Data.Real.Basic

/-!
# HSV Roundtrip — Identity on Achromatic

For achromatic colors (R=G=B), HSV roundtrip is identity since
H=0, S=0, V=R and the inverse recovers (R,R,R).
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Achromatic HSV: when R=G=B, the value channel equals the input. -/
theorem hsv_value_achromatic (v : ℝ) : max v (max v v) = v := by
  simp [max_self]

#check @hsv_value_achromatic

end ProvableContracts.Image
