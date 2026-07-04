/-!
# LoRA Shape Compatibility

Contract: `lora-algebra-v1`, equation `lora_shape`.

For `A : m×r` and `B : r×n`, the product `A·B` has shape `m×n` — exactly the
base weight `W`'s shape — so the LoRA-merged weight `W + A·B` is dimensionally
well-formed (contract invariant "A @ B has same shape as original weight").
Shapes are modelled as `(rows, cols) : Nat × Nat`; the proof is core Lean (no
Mathlib).
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

#check @lora_shape

end ProvableContracts.LoRA.Shape
