import ProvableContracts.Defs.Conv1D

/-!
# Conv1d Zero-Input ⇒ Zero-Output

Proves that convolving the all-zero signal yields the all-zero output:

  `conv w 0 = 0`.

This is the homogeneity corner of linearity (`conv` maps the additive
identity to the additive identity) and a basic sanity invariant for the
kernel.
-/

namespace ProvableContracts.Conv1D

open Finset

-- Status: proved
/-- **Zero input ⇒ zero output** at every position. -/
theorem conv_zero (w : List ℝ) (n : ℕ) : conv w zeroSignal n = 0 := by
  simp [conv, zeroSignal]

-- Status: proved
/-- Point-free form: `conv w 0 = 0`. -/
theorem conv_zero_eq (w : List ℝ) : conv w zeroSignal = zeroSignal := by
  funext n
  simpa [zeroSignal] using conv_zero w n

#check @conv_zero
#check @conv_zero_eq

end ProvableContracts.Conv1D
