/-!
# Merge Shape Preservation

Contract: `lora-algebra-v1`, equation `shape_preservation`.

Any merge that combines the base tensor with a delta of the *same* shape yields
a tensor of the base's shape — the merge never changes tensor shapes (contract
invariant "Merge never changes tensor shapes"). Shapes are modelled as
`(rows, cols) : Nat × Nat`; the proof is core Lean (no Mathlib).
-/

namespace ProvableContracts.LoRA.ShapePreservation

/-- A tensor shape as `(rows, cols)`. -/
abbrev Shape := Nat × Nat

/-- A shape-preserving merge: defined only when `delta` has the base's shape,
    and returns that shape. -/
def merge_shape (base delta : Shape) (_h : delta = base) : Shape := base

-- Status: proved (core Lean)
/-- The merged tensor has the base tensor's shape — and, since the delta shares
    it, the delta's shape too. Shape is preserved. -/
theorem shape_preservation (base delta : Shape) (h : delta = base) :
    merge_shape base delta h = base ∧ merge_shape base delta h = delta :=
  ⟨rfl, h.symm⟩

#check @shape_preservation

end ProvableContracts.LoRA.ShapePreservation
