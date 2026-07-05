import ProvableContracts.Defs.Gelu

/-!
# GELU Sign Behaviour

Proves the sign / non-negativity obligations of `gelu-kernel-v1.yaml`:

* `GE-BND-001` (obligation `bound`): `x > 0 → GELU(x) ≥ 0`
* sign behaviour on the negative side: `x < 0 → GELU(x) ≤ 0`

## Mechanism

The tanh-approximation GELU factors as

  `GELU(x) = (0.5·x) · (1 + tanh g(x))`,  `g(x) = √(2/π)·(x + 0.044715·x³)`.

The right factor `1 + tanh g(x)` is *strictly positive* for every real `x`
because `tanh` is bounded below by `-1` (`Real.neg_one_lt_tanh`). Hence the sign
of `GELU(x)` is exactly the sign of the left factor `0.5·x`:

* `x > 0`  ⟹ `0.5·x ≥ 0` and the right factor `> 0` ⟹ product `≥ 0`;
* `x < 0`  ⟹ `0.5·x ≤ 0` and the right factor `> 0` ⟹ product `≤ 0`.

Both are closed by `mul_nonneg` / `mul_nonpos` on the two factors.
-/

namespace ProvableContracts.Gelu

open Real

/-- The right factor `1 + tanh(…)` of the GELU product is strictly positive,
    since `tanh > -1` everywhere. -/
theorem gelu_right_factor_pos (x : ℝ) :
    0 < 1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3)) := by
  have h := Real.neg_one_lt_tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3))
  linarith

-- Status: proved
/-- **GE-BND-001 / non-negativity for positive inputs.**
    `GELU(x) ≥ 0` for `x > 0`. -/
theorem gelu_nonneg_of_pos {x : ℝ} (hx : 0 < x) : 0 ≤ gelu x := by
  unfold gelu
  have hleft : 0 ≤ 0.5 * x := by linarith
  have hright : 0 ≤ 1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3)) :=
    le_of_lt (gelu_right_factor_pos x)
  exact mul_nonneg hleft hright

-- Status: proved
/-- **Sign behaviour on the negative side.**
    `GELU(x) ≤ 0` for `x < 0`. -/
theorem gelu_nonpos_of_neg {x : ℝ} (hx : x < 0) : gelu x ≤ 0 := by
  unfold gelu
  have hleft : 0.5 * x ≤ 0 := by linarith
  have hright : 0 ≤ 1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3)) :=
    le_of_lt (gelu_right_factor_pos x)
  exact mul_nonpos_of_nonpos_of_nonneg hleft hright

-- Tests
#check @gelu_nonneg_of_pos
#check @gelu_nonpos_of_neg

example {x : ℝ} (hx : 0 < x) : 0 ≤ gelu x := gelu_nonneg_of_pos hx
example {x : ℝ} (hx : x < 0) : gelu x ≤ 0 := gelu_nonpos_of_neg hx

end ProvableContracts.Gelu
