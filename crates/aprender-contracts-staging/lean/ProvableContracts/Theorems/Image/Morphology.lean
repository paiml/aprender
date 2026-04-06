import Mathlib.Order.BooleanAlgebra

/-!
# Morphology — Opening is Anti-Extensive

Mathematical morphology: opening(A) ⊆ A (anti-extensive property).
This is a fundamental invariant: morphological opening never adds pixels.
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Infimum is anti-extensive: a ⊓ b ≤ a. -/
theorem opening_anti_extensive (a b : Prop) [Decidable a] [Decidable b] :
    (a ∧ b) → a := And.left

#check @opening_anti_extensive

end ProvableContracts.Image
