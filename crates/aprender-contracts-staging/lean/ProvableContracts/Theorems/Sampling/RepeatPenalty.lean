import ProvableContracts.Defs.Sampling
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith

/-!
# Repeat-Penalty Theorems

Proves the analytic content of the `repeat_penalty` equation of
`apr-cli-sampling-v1.yaml`: a penalty factor of `1.0` is the identity
transform (no logit is changed), and a penalty `> 1` never increases the
logit of a repeated token (it pushes positive logits down toward zero and
negative logits further down), which is exactly the intended de-repetition
effect.

## Obligation

`apr-cli-sampling-v1 / repeat_penalty`: penalty `1.0` is identity;
penalty `> 1.0` reduces the (probability contribution of) repeated tokens.
-/

namespace ProvableContracts.Sampling

/-- **Penalty 1.0 is the identity.** Applying the repeat penalty with factor
    `1.0` leaves every logit unchanged, regardless of sign. -/
theorem penalty_one_identity (l : ℝ) : applyPenalty l 1 = l := by
  unfold applyPenalty
  split_ifs <;> simp

/-- A penalty `p ≥ 1` never increases a positive logit: `l / p ≤ l`. Combined
    with `penalty_one_identity`, this shows the penalty is monotone in `p` at
    the identity point and de-emphasises already-generated tokens. -/
theorem penalty_reduces_positive (l p : ℝ) (hl : 0 < l) (hp : 1 ≤ p) :
    applyPenalty l p ≤ l := by
  unfold applyPenalty
  rw [if_pos hl]
  exact div_le_self (le_of_lt hl) hp

/-- A penalty `p ≥ 1` never increases a negative logit: `l * p ≤ l`
    (it becomes more negative), further suppressing repeated tokens. -/
theorem penalty_reduces_negative (l p : ℝ) (hl : l < 0) (hp : 1 ≤ p) :
    applyPenalty l p ≤ l := by
  unfold applyPenalty
  rw [if_neg (not_lt.mpr (le_of_lt hl)), if_pos hl]
  nlinarith [mul_nonneg (sub_nonneg.mpr hp) (neg_nonneg.mpr hl.le)]

-- Tests
#check @penalty_one_identity
#check @penalty_reduces_positive
#check @penalty_reduces_negative

end ProvableContracts.Sampling
