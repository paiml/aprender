import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

/-!
# Discrete Fourier Transform Definitions

Mathematical definition of the DFT for finite complex-valued vectors.

## References

- Cooley & Tukey (1965) An Algorithm for the Machine Calculation of
  Complex Fourier Series
- Oppenheim & Schafer (2010) Discrete-Time Signal Processing
-/

namespace ProvableContracts.FFT

open Complex Finset Real

/-- A finite vector of complex numbers, indexed by `Fin n`. -/
abbrev CVec (n : ℕ) := Fin n → ℂ

/-- The DFT twiddle factor: ω(j,k,N) = exp(-2πijk/N). -/
noncomputable def twiddle (j k N : ℕ) : ℂ :=
  Complex.exp (-2 * ↑π * Complex.I * ↑j * ↑k / ↑N)

/-- DFT: X[k] = Σⱼ x[j] · ω(j,k,N). -/
noncomputable def dft {n : ℕ} (x : CVec n) (k : Fin n) : ℂ :=
  ∑ j : Fin n, x j * twiddle (j : ℕ) (k : ℕ) n

/-- Squared magnitude of a complex vector: Σᵢ |xᵢ|². -/
noncomputable def energy {n : ℕ} (x : CVec n) : ℝ :=
  ∑ i : Fin n, Complex.normSq (x i)

end ProvableContracts.FFT
