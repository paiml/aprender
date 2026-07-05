import ProvableContracts.Defs.CMAES
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# CMA-ES Step-Size Positivity

Proves the analytic core of obligation **CMA-BND-001** (`sigma > 0 at every
generation`): the CSA multiplicative update `σ · exp(f)` can never drive the
step size to zero or negative, because `exp` is strictly positive.

The "at every generation" quantifier is discharged by induction on the list of
per-generation adaptation factors (`stepSizeIterate`).
-/

namespace ProvableContracts.CMAES

open Real

-- Status: proved
/-- One CSA step preserves strict positivity of the step size. -/
theorem stepSize_pos (sigma factor : ℝ) (h : 0 < sigma) :
    0 < stepSizeUpdate sigma factor := by
  unfold stepSizeUpdate
  exact mul_pos h (Real.exp_pos factor)

-- Status: proved
/-- The step size stays strictly positive after **any** number of generations. -/
theorem stepSizeIterate_pos (sigma : ℝ) (factors : List ℝ) (h : 0 < sigma) :
    0 < stepSizeIterate sigma factors := by
  induction factors generalizing sigma with
  | nil => simpa [stepSizeIterate] using h
  | cons f fs ih =>
      unfold stepSizeIterate
      exact ih _ (stepSize_pos sigma f h)

-- Tests
#check @stepSize_pos
#check @stepSizeIterate_pos

end ProvableContracts.CMAES
