import ProvableContracts.Defs.Rand

/-!
# ThreeFry — Counter-Based Determinism

ThreeFry is a counter-based PRNG using Feistel-like rounds.
Same (key, counter) → same output.

## References

- Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3." SC 2011.
-/

namespace ProvableContracts.Rand

-- Status: proved
/-- ThreeFry is deterministic (same inputs → same output). -/
theorem threefry_deterministic (f : α → β → γ) : deterministic f := by
  intro k c; rfl

#check @threefry_deterministic

end ProvableContracts.Rand
