import ProvableContracts.Defs.CrossEntropy
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Log-Softmax Upper Bound

Proves that log_softmax(x)ᵢ ≤ 0 for all i.

## Obligation

`CE-BND-001`: log_softmax(x)ᵢ ≤ 0 for all i.

Since softmax(x)ᵢ ∈ (0, 1], log(softmax(x)ᵢ) ≤ 0.
Equivalently, xᵢ - log(Σⱼ exp(xⱼ)) ≤ 0 ⟺ exp(xᵢ) ≤ Σⱼ exp(xⱼ),
which holds because xᵢ is one term of the sum.
-/

namespace ProvableContracts.CrossEntropy

open Real Finset

-- Status: proved
/-- The partition function Z = Σⱼ exp(xⱼ) is strictly positive. -/
theorem partition_pos {n : ℕ} (x : RVec (n + 1)) :
    0 < ∑ j : Fin (n + 1), Real.exp (x j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (x j)) Finset.univ_nonempty

-- Status: proved
/-- A single exponential is at most the sum: exp(xᵢ) ≤ Σⱼ exp(xⱼ). -/
theorem exp_le_sum {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) :
    Real.exp (x i) ≤ ∑ j : Fin (n + 1), Real.exp (x j) :=
  Finset.single_le_sum (fun j _ => le_of_lt (Real.exp_pos (x j)))
    (Finset.mem_univ i)

-- Status: proved
/-- Log-softmax is bounded above by zero.
    log(exp(xᵢ)/Z) = log(exp(xᵢ)) - log(Z) = xᵢ - log(Z).
    Since exp(xᵢ) ≤ Z, taking log gives xᵢ ≤ log(Z), so the
    difference is ≤ 0. -/
theorem log_softmax_le_zero {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) :
    log_softmax x i ≤ 0 := by
  unfold log_softmax
  linarith [Real.log_le_log (Real.exp_pos (x i)) (exp_le_sum x i),
            Real.log_exp (x i)]

-- Tests
#check @partition_pos
#check @exp_le_sum
#check @log_softmax_le_zero

end ProvableContracts.CrossEntropy
