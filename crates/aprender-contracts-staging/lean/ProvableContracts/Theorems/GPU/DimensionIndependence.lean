import ProvableContracts.Defs.GPU

/-!
# Dimension Independence — Constant Function Property

A dimension-independent kernel is a constant function over dimensions.
If the kernel output (PTX) does not depend on (M, K, N),
then it is dimension-independent by definition.
-/

namespace ProvableContracts.GPU

-- Status: proved
/-- A constant kernel is trivially dimension-independent. -/
theorem const_is_dimension_independent {α : Type*} (c : α) :
    dimension_independent (fun (_ _ _ : ℕ) => c) := by
  intro _ _ _ _ _ _
  rfl

#check @const_is_dimension_independent

end ProvableContracts.GPU
