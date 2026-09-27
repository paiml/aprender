/-!
# LoRA Shape Compatibility

Contract: `lora-algebra-v1`, equation `lora_shape`.

For `A : m×r` and `B : r×n`, the product `A·B` has shape `m×n` — exactly the
base weight `W`'s shape — so the LoRA-merged weight `W + A·B` is dimensionally
well-formed (contract invariant "A @ B has same shape as original weight").
Shapes are modelled as `(rows, cols) : Nat × Nat`; the proof is core Lean (no
Mathlib).

## L4 simulation (ONT-10)

`matmul_len` and `merge_len` transcribe the Rust that decides these shapes, on
the flat lengths the Rust actually stores (`Tensor` is a 1-D `Array1<f32>`):
`entrenar::autograd::matmul`'s two size asserts, and `LoRALayer::merge`'s walk
of the base weight that indexes `B @ A`. `none` is a Rust panic. The float
VALUES are out of scope; this module claims only which calls panic and what
length the result has. Witness: `crates/aprender-train/src/lora/layer/
lean_witness_tests.rs`, against the golden table `#eval golden_matmul` /
`#eval golden_merge` print.
-/

namespace ProvableContracts.LoRA.Shape

/-- A tensor shape as `(rows, cols)`. -/
abbrev Shape := Nat × Nat

/-- Shape of the product `A·B`: the inner dimension cancels, leaving
    `(rows A, cols B)`. -/
def matmul_shape (a b : Shape) : Shape := (a.1, b.2)

-- Status: proved (core Lean)
/-- LoRA product `A(m×r) · B(r×n)` has shape `(m×n)`, the base weight's shape. -/
theorem lora_shape (m r n : Nat) :
    matmul_shape (m, r) (r, n) = (m, n) := rfl

-- Status: proved (core Lean)
/-- The merged shape `A·B` matches the base weight `W : (m×n)` exactly. -/
theorem lora_shape_matches_base (m r n : Nat) :
    matmul_shape (m, r) (r, n) = (m, n) := rfl

/-- Transcribes `entrenar::autograd::matmul`
    (crates/aprender-train/src/autograd/ops/matmul.rs:453-456) on lengths:
    `assert_eq!(a.len(), m * k)`, then `assert_eq!(b.len(), k * n)`, then a
    result of `m * n` elements. `none` = one of the asserts panicked. -/
def matmul_len (a_len b_len m k n : Nat) : Option Nat :=
  if a_len = m * k then
    if b_len = k * n then some (m * n) else none
  else none

/-- Transcribes `LoRALayer::merge` (crates/aprender-train/src/lora/layer/core.rs:291-305)
    on lengths. Already merged: return, weight untouched. Otherwise
    `ba = matmul(lora_b, lora_a, d_out, rank, d_in)` and then
    `for i in 0..base_len { base[i] += scale * ba[i] }`, which panics exactly
    when some `i < base_len` is out of `ba`'s bounds, i.e. `ba_len < base_len`.
    Returns the merged weight's length; `none` = panic. -/
def merge_len (merged : Bool) (base_len a_len b_len d_out d_in rank : Nat) : Option Nat :=
  if merged then some base_len
  else
    match matmul_len b_len a_len d_out rank d_in with
    | none => none
    | some ba_len => if base_len ≤ ba_len then some base_len else none

/-- The lengths `LoRALayer::new` (core.rs:94-109) leaves: base asserted
    `d_out * d_in`, A built as `rank * d_in`, B as `d_out * rank`. -/
def new_lens (d_out d_in rank : Nat) : Nat × Nat × Nat :=
  (d_out * d_in, rank * d_in, d_out * rank)

-- Status: proved (core Lean)
/-- `matmul` returns exactly when both operand lengths match their dims, and
    then with `m * n` elements. -/
theorem matmul_len_eq_some_iff (a_len b_len m k n c : Nat) :
    matmul_len a_len b_len m k n = some c ↔
      a_len = m * k ∧ b_len = k * n ∧ c = m * n := by
  unfold matmul_len
  by_cases h1 : a_len = m * k <;> by_cases h2 : b_len = k * n <;> simp [h1, h2]
  exact ⟨Eq.symm, Eq.symm⟩

-- Status: proved (core Lean)
/-- `lora_shape` over the model: `A(m×r) @ B(r×n)` stored flat never panics and
    has `m * n` elements, the base weight's length. -/
theorem lora_shape_sim (m r n : Nat) :
    matmul_len (m * r) (r * n) m r n = some (m * n) :=
  (matmul_len_eq_some_iff _ _ _ _ _ _).mpr ⟨rfl, rfl, rfl⟩

-- Status: proved (core Lean)
/-- An unmerged `merge` returns exactly when B and A have their LoRA lengths and
    the base fits in `B @ A`; the merged weight then keeps the base's length. -/
theorem merge_len_unmerged_eq_some_iff (base_len a_len b_len d_out d_in rank c : Nat) :
    merge_len false base_len a_len b_len d_out d_in rank = some c ↔
      b_len = d_out * rank ∧ a_len = rank * d_in ∧ base_len ≤ d_out * d_in ∧ c = base_len := by
  unfold merge_len matmul_len
  by_cases hb : b_len = d_out * rank <;> by_cases ha : a_len = rank * d_in <;>
    by_cases hl : base_len ≤ d_out * d_in <;> simp [hb, ha, hl]
  exact ⟨Eq.symm, Eq.symm⟩

-- Status: proved (core Lean)
/-- Every layer `LoRALayer::new` builds merges without panicking, and the merged
    weight has the base's `d_out * d_in` elements, merged or not. -/
theorem lora_merge_preserves_shape (merged : Bool) (d_out d_in rank : Nat) :
    merge_len merged (new_lens d_out d_in rank).1 (new_lens d_out d_in rank).2.1
      (new_lens d_out d_in rank).2.2 d_out d_in rank = some (d_out * d_in) := by
  cases merged
  · exact (merge_len_unmerged_eq_some_iff _ _ _ _ _ _ _).mpr
      ⟨rfl, rfl, Nat.le_refl _, rfl⟩
  · rfl

-- Status: proved (core Lean)
/-- A LoRA A matrix of the wrong length makes an unmerged `merge` panic,
    whatever else holds. -/
theorem merge_len_none_of_a_mismatch (base_len a_len b_len d_out d_in rank : Nat)
    (h : a_len ≠ rank * d_in) :
    merge_len false base_len a_len b_len d_out d_in rank = none := by
  cases hm : merge_len false base_len a_len b_len d_out d_in rank with
  | none => rfl
  | some c => exact absurd ((merge_len_unmerged_eq_some_iff _ _ _ _ _ _ _).mp hm).2.1 h

/-- Golden table for the witness: `matmul_len a b m k n` over `m k n ∈ 1..3`,
    `a b ∈ 0..9`, loops nested m, k, n, a, b; one char per call, `.` = panic,
    else the digit `m * n`. -/
def golden_matmul : String := Id.run do
  let mut s := ""
  for m in [1, 2, 3] do
    for k in [1, 2, 3] do
      for n in [1, 2, 3] do
        for a in List.range 10 do
          for b in List.range 10 do
            s := s.push (match matmul_len a b m k n with
              | none => '.'
              | some c => Char.ofNat (48 + c))
  return s

/-- Golden table for the witness: `merge_len false` on a layer from `new_lens`
    with A's length offset by `da` and B's by `db` (`0 = −1, 1 = 0, 2 = +1`);
    loops nested d_out, d_in, rank ∈ 1..3, da, db. `.` = panic, `k` = kept the
    base length. -/
def golden_merge : String := Id.run do
  let mut s := ""
  for d_out in [1, 2, 3] do
    for d_in in [1, 2, 3] do
      for rank in [1, 2, 3] do
        for da in [0, 1, 2] do
          for db in [0, 1, 2] do
            let (base, a, b) := new_lens d_out d_in rank
            s := s.push (match merge_len false base (a + da - 1) (b + db - 1) d_out d_in rank with
              | none => '.'
              | some c => if c = base then 'k' else '?')
  return s

#check @lora_shape

end ProvableContracts.LoRA.Shape
