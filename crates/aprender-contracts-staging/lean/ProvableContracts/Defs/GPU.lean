import Mathlib.Data.Real.Basic

/-!
# GPU Kernel Definitions

Definitions for GPU kernel contracts (dimension independence).
-/

namespace ProvableContracts.GPU

/-- A dimension-independent kernel produces the same PTX regardless of
    input dimensions. We model this as: the kernel function is
    parametric in dimensions (they are runtime params, not compile-time). -/
def dimension_independent (kernel : ℕ → ℕ → ℕ → α) : Prop :=
  ∀ m₁ k₁ n₁ m₂ k₂ n₂, kernel m₁ k₁ n₁ = kernel m₂ k₂ n₂

end ProvableContracts.GPU
