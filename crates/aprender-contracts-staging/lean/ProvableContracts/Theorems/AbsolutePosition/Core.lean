import ProvableContracts.Defs.AbsolutePosition

/-!
# Absolute Positional Encoding — Analytic Theorems

Proves the analytic proof obligations of `absolute-position-v1.yaml`.

## Obligations discharged

- `AP-INV-001` Shape preservation — `abs_add` is elementwise, preserving the
  index type `Fin n` (no broadcast, no dimension change).
- `AP-INV-002` Additive identity — a zero position embedding is the additive
  identity: `abs_add token 0 = token`.
- `AP-BND-005` Sinusoidal component bound — every PE component lies in `[-1, 1]`
  (`Real.sin` / `Real.cos` bounds).
- `AP-INV-006` Zero-position value — `PE(0, 2i) = 0`, `PE(0, 2i+1) = 1`.
- `AP-LIN-007` Relative-position rotation — `PE(pos + k)` is a linear rotation
  of `PE(pos)` by angle `k·ω(i)` (angle-addition identities), the analytic core
  that lets self-attention read relative positions.

All proofs are `sorry`-free and use only Mathlib trigonometric lemmas.
-/

namespace ProvableContracts.AbsolutePosition

open ProvableContracts

-- ─────────────────────────────────────────────────────────────────
-- Learned additive encoding
-- ─────────────────────────────────────────────────────────────────

-- Status: proved
/-- AP-INV-001 — Shape preservation. `abs_add token pos` is a total function on
the same index type `Fin n` as its inputs and equals the pointwise sum; there is
no broadcast or dimension change. -/
theorem abs_add_shape {n : ℕ} (token pos : RVec n) (i : Fin n) :
    abs_add token pos i = token i + pos i := by
  rfl

-- Status: proved
/-- AP-INV-002 — Additive identity. A zero position embedding preserves the token
embedding: `abs_add token 0 = token`. -/
theorem abs_add_identity {n : ℕ} (token : RVec n) :
    abs_add token (fun _ => (0 : ℝ)) = token := by
  funext i
  simp only [abs_add, add_zero]

-- ─────────────────────────────────────────────────────────────────
-- Sinusoidal encoding
-- ─────────────────────────────────────────────────────────────────

-- Status: proved
/-- AP-BND-005 — Even sinusoidal component is bounded in `[-1, 1]`. -/
theorem pe_even_bound (d : ℕ) (pos : ℝ) (i : ℕ) :
    -1 ≤ pe_even d pos i ∧ pe_even d pos i ≤ 1 :=
  ⟨Real.neg_one_le_sin _, Real.sin_le_one _⟩

-- Status: proved
/-- AP-BND-005 — Odd sinusoidal component is bounded in `[-1, 1]`. -/
theorem pe_odd_bound (d : ℕ) (pos : ℝ) (i : ℕ) :
    -1 ≤ pe_odd d pos i ∧ pe_odd d pos i ≤ 1 :=
  ⟨Real.neg_one_le_cos _, Real.cos_le_one _⟩

-- Status: proved
/-- AP-INV-006 — Zero-position even value: `PE(0, 2i) = sin(0) = 0`. -/
theorem pe_even_at_zero (d : ℕ) (i : ℕ) :
    pe_even d 0 i = 0 := by
  simp only [pe_even, zero_mul, Real.sin_zero]

-- Status: proved
/-- AP-INV-006 — Zero-position odd value: `PE(0, 2i+1) = cos(0) = 1`. -/
theorem pe_odd_at_zero (d : ℕ) (i : ℕ) :
    pe_odd d 0 i = 1 := by
  simp only [pe_odd, zero_mul, Real.cos_zero]

-- Status: proved
/-- AP-LIN-007 — Relative-position rotation of the even component. Shifting the
position by `k` rotates `(PE_even, PE_odd)` by angle `k·ω(i)`:
`PE_even(pos+k) = PE_even(pos)·cos(k·ω) + PE_odd(pos)·sin(k·ω)`. Follows from the
sine angle-addition identity. -/
theorem pe_even_rotation (d : ℕ) (pos k : ℝ) (i : ℕ) :
    pe_even d (pos + k) i
      = pe_even d pos i * Real.cos (k * omega d i)
        + pe_odd d pos i * Real.sin (k * omega d i) := by
  simp only [pe_even, pe_odd, add_mul, Real.sin_add]

-- Status: proved
/-- AP-LIN-007 — Relative-position rotation of the odd component:
`PE_odd(pos+k) = PE_odd(pos)·cos(k·ω) − PE_even(pos)·sin(k·ω)`. Follows from the
cosine angle-addition identity. -/
theorem pe_odd_rotation (d : ℕ) (pos k : ℝ) (i : ℕ) :
    pe_odd d (pos + k) i
      = pe_odd d pos i * Real.cos (k * omega d i)
        - pe_even d pos i * Real.sin (k * omega d i) := by
  simp only [pe_even, pe_odd, add_mul, Real.cos_add]

-- Tests
#check @abs_add_shape
#check @abs_add_identity
#check @pe_even_bound
#check @pe_odd_bound
#check @pe_even_at_zero
#check @pe_odd_at_zero
#check @pe_even_rotation
#check @pe_odd_rotation

end ProvableContracts.AbsolutePosition
