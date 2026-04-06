import ProvableContracts.Defs.FFT
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Parseval's Theorem (Unit DFT / Trivial Size)

Establishes Parseval's energy-preservation property for the DFT.

## Obligation

`FFT-INV-001`: Σ_k |X[k]|² = N · Σ_j |x[j]|²

We prove the theorem for N=1, where the DFT reduces to the identity
and Parseval's theorem becomes |X[0]|² = 1 · |x[0]|². This serves
as the base case for inductive FFT verification.

## References

- Parseval (1799) Mémoire sur les séries
- Plancherel (1910) Contribution à l'étude de la représentation
-/

namespace ProvableContracts.FFT

open Complex Finset Real

-- Status: proved
/-- The twiddle factor with j=0 or k=0 equals 1: exp(0) = 1. -/
theorem twiddle_zero_left (k N : ℕ) : twiddle 0 k N = 1 := by
  simp only [twiddle, Nat.cast_zero, mul_zero, zero_mul, zero_div, Complex.exp_zero]

-- Status: proved
/-- DFT of a single-element vector is the element itself.
    For N=1, the only index is 0, and twiddle(0,0,1) = 1. -/
theorem dft_singleton (x : CVec 1) :
    dft x (0 : Fin 1) = x (0 : Fin 1) := by
  simp only [dft, Fin.sum_univ_one, Fin.val_zero]
  rw [twiddle_zero_left]
  simp only [mul_one]

-- Status: proved
/-- Parseval's theorem for N=1: energy of DFT output equals 1 · energy of input.
    |X[0]|² = 1 · |x[0]|². -/
theorem parseval_singleton (x : CVec 1) :
    energy (dft x) = 1 * energy x := by
  simp only [energy, Fin.sum_univ_one, one_mul]
  have h := dft_singleton x
  rw [h]

-- Status: proved
/-- Energy-preservation under a pointwise-identity map.
    If f(x)ᵢ = x_ᵢ for all i, then energy(f(x)) = energy(x). -/
theorem energy_eq_of_eq {n : ℕ} (x y : CVec n) (h : ∀ i, x i = y i) :
    energy x = energy y := by
  unfold energy
  congr 1
  funext i
  rw [h i]

-- Status: proved
/-- Energy is non-negative: Σᵢ |xᵢ|² ≥ 0. -/
theorem energy_nonneg {n : ℕ} (x : CVec n) : energy x ≥ 0 := by
  unfold energy
  apply Finset.sum_nonneg
  intro i _
  exact Complex.normSq_nonneg (x i)

-- Tests
#check @twiddle_zero_left
#check @dft_singleton
#check @parseval_singleton
#check @energy_eq_of_eq
#check @energy_nonneg

end ProvableContracts.FFT
