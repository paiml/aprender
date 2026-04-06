import ProvableContracts.Defs.FFT
import Mathlib.Data.Complex.Basic

/-!
# Bluestein — Chirp-Z Identity

Bluestein's algorithm rewrites DFT as a convolution via the identity:
  jk = -(j-k)²/2 + j²/2 + k²/2
This means the twiddle factor factorizes into chirp components.
-/

namespace ProvableContracts.FFT

-- Status: proved
/-- Bluestein identity: 2jk = j² + k² - (j-k)². -/
theorem bluestein_identity (j k : ℤ) :
    2 * j * k = j ^ 2 + k ^ 2 - (j - k) ^ 2 := by ring

#check @bluestein_identity

end ProvableContracts.FFT
