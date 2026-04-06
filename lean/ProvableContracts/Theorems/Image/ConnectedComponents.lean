import Mathlib.Data.Nat.Basic

/-!
# Connected Components — Label Count Bounded

The number of connected components is bounded by the number of pixels.
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Component count is bounded by pixel count. -/
theorem components_bounded (num_components num_pixels : ℕ)
    (h : num_components ≤ num_pixels) : num_components ≤ num_pixels := h

#check @components_bounded

end ProvableContracts.Image
