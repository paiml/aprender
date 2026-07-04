import ProvableContracts.Defs.Conv1D

/-!
# Conv1d Shift Equivariance

Proves that 1-D convolution **commutes with translation** of the input:

  `conv w (shift x s) = shift (conv w x) s`,

i.e. delaying/advancing the input by `s` samples delays/advances the output
by the same `s`. This is the defining structural property of a convolution
(a shift-invariant linear system) and appears among the contract's
`equations.invariants`.
-/

namespace ProvableContracts.Conv1D

open Finset

-- Status: proved
/-- **Shift equivariance.** Convolving a left-shifted signal equals shifting the
convolution output: `conv w (shift x s) n = conv w x (n + s)`. -/
theorem conv_shift (w : List ℝ) (x : Signal) (s n : ℕ) :
    conv w (shift x s) n = conv w x (n + s) := by
  simp only [conv, shift]
  apply Finset.sum_congr rfl
  intro k _
  have h : n + k + s = n + s + k := by omega
  rw [h]

-- Status: proved
/-- Point-free form: `conv w (shift x s) = shift (conv w x) s`. -/
theorem conv_shift_eq (w : List ℝ) (x : Signal) (s : ℕ) :
    conv w (shift x s) = shift (conv w x) s := by
  funext n
  exact conv_shift w x s n

#check @conv_shift
#check @conv_shift_eq

end ProvableContracts.Conv1D
