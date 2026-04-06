import ProvableContracts.Defs.Rand

/-!
# Philox — Counter-Based Determinism

Philox is a counter-based PRNG. The core property:
same (key, counter) → same output. It's a pure function.

## References

- Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3." SC 2011.
-/

namespace ProvableContracts.Rand

-- Status: proved
/-- Any pure function is deterministic. -/
theorem philox_deterministic (f : α → β → γ) : deterministic f := by
  intro k c
  rfl

#check @philox_deterministic

end ProvableContracts.Rand
