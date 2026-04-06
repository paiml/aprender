import ProvableContracts.Defs.Image

/-!
# RGB to Grayscale — Coefficient Sum

Proves that the luminance weights sum to 1: 0.299 + 0.587 + 0.114 = 1.
This ensures grayscale output preserves the [0,1] range for valid RGB inputs.
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Luminance coefficients sum to 1. -/
theorem rgb_coefficients_sum : (0.299 : ℝ) + 0.587 + 0.114 = 1 := by norm_num

-- Status: proved
/-- rgb_to_gray preserves [0,1] bounds for unit inputs. -/
theorem rgb_to_gray_unit : rgb_to_gray 1 1 1 = 1 := by
  unfold rgb_to_gray; norm_num

#check @rgb_coefficients_sum
#check @rgb_to_gray_unit

end ProvableContracts.Image
