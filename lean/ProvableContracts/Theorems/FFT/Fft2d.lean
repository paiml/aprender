import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# FFT 2D — Separability

2D DFT is separable: DFT₂D = DFT_rows ∘ DFT_cols.
We prove the algebraic identity that makes this possible:
  exp(-2πi(j₁k₁/N₁ + j₂k₂/N₂)) = exp(-2πij₁k₁/N₁) · exp(-2πij₂k₂/N₂)
-/

namespace ProvableContracts.FFT

-- Status: proved
/-- Product of exponentials: exp(a+b) = exp(a) · exp(b). -/
theorem fft2d_separable (a b : ℂ) :
    Complex.exp (a + b) = Complex.exp a * Complex.exp b :=
  Complex.exp_add a b

#check @fft2d_separable

end ProvableContracts.FFT
