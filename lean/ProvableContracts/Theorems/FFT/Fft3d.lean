import Mathlib.Data.Complex.Basic

/-!
# FFT 3D — Triple Separability

3D DFT separates into three 1D DFTs.
exp(a+b+c) = exp(a) · exp(b) · exp(c).
-/

namespace ProvableContracts.FFT

-- Status: proved
/-- Triple separability of exponential. -/
theorem fft3d_separable (a b c : ℂ) :
    Complex.exp (a + b + c) = Complex.exp a * Complex.exp b * Complex.exp c := by
  rw [Complex.exp_add, Complex.exp_add, mul_assoc]

#check @fft3d_separable

end ProvableContracts.FFT
