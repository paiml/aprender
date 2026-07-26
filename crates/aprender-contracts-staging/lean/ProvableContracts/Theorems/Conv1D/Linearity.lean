import ProvableContracts.Defs.Conv1D
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring

/-!
# Conv1d Linearity

Proves that 1-D convolution is a **linear** operator in its input signal:

  `conv w (a·x + b·z) = a · conv w x + b · conv w z`.

## Obligation

`CV-LIN-001` (contract `conv1d-kernel-v1.yaml`, proof obligation
"Convolution linearity"): `|conv(a·x + b·z) − (a·conv(x) + b·conv(z))| < eps`.
Here we prove the exact (eps = 0) real-number identity, which is strictly
stronger than the floating-point tolerance form.
-/

namespace ProvableContracts.Conv1D

open Finset

-- Status: proved
/-- **Linearity of conv1d in the input signal.** For any kernel `w`, scalars
`a, b`, signals `x, z` and position `n`:
`conv w (a·x + b·z) n = a · (conv w x n) + b · (conv w z n)`. -/
theorem conv_linear (w : List ℝ) (a b : ℝ) (x z : Signal) (n : ℕ) :
    conv w (lin a b x z) n = a * conv w x n + b * conv w z n := by
  simp only [conv, lin]
  rw [mul_sum, mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  ring

-- Status: proved
/-- **Additivity** (special case `a = b = 1`). -/
theorem conv_add (w : List ℝ) (x z : Signal) (n : ℕ) :
    conv w (fun i => x i + z i) n = conv w x n + conv w z n := by
  simp only [conv]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  ring

-- Status: proved
/-- **Homogeneity** (special case `b = 0`, `z = 0`). -/
theorem conv_smul (w : List ℝ) (a : ℝ) (x : Signal) (n : ℕ) :
    conv w (fun i => a * x i) n = a * conv w x n := by
  simp only [conv]
  rw [mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  ring

#check @conv_linear
#check @conv_add
#check @conv_smul

end ProvableContracts.Conv1D
