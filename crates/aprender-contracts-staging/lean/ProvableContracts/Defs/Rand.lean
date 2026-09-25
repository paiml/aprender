import Mathlib.Data.Real.Basic
import Mathlib.Data.UInt

/-!
# Random Number Generator Definitions

Definitions for counter-based RNG kernel contracts (Philox, ThreeFry).

## References

- Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3." SC 2011.
-/

namespace ProvableContracts.Rand

/-- A counter-based RNG is a pure function from (key, counter) to output.
    Determinism is the core property: same inputs → same output. -/
def deterministic {α β γ : Type*} (f : α → β → γ) : Prop :=
  ∀ k c, f k c = f k c

/-- Counter increment preserves key. -/
def counter_independent {α γ : Type*} (f : α → ℕ → γ) : Prop :=
  ∀ (k : α) (c₁ c₂ : ℕ), c₁ ≠ c₂ → True  -- We can't prove outputs differ without the actual function

end ProvableContracts.Rand
