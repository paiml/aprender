import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Task Vector — additive roundtrip

Contract: `lora-algebra-v1`, equation `task_vector`.

The task vector `δ = W_fine - W_base` reconstructs the fine-tuned weights
additively: `W_base + δ = W_fine` (contract invariant "Additive: W_base + delta
== W_fine (roundtrip)"). Proved element-wise; the matrix statement is the
element-wise identity applied entrywise.
-/

namespace ProvableContracts.LoRA.TaskVector

/-- Task vector (element-wise): `δ = W_fine - W_base`. -/
def task_vector (w_fine w_base : ℝ) : ℝ := w_fine - w_base

-- Status: proved (core algebraic)
/-- Roundtrip: adding the task vector back to the base recovers `W_fine`. -/
theorem task_vector_roundtrip (w_fine w_base : ℝ) :
    w_base + task_vector w_fine w_base = w_fine := by
  unfold task_vector; ring

#check @task_vector_roundtrip

end ProvableContracts.LoRA.TaskVector
