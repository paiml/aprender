import ProvableContracts.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# RoPE (Rotary Position Embedding) — Analytic Theorems

Machine-checked proofs of the analytic proof-obligations of the
`rope-kernel-v1.yaml` contract (Su et al. 2021, RoFormer).

RoPE applies, to each disjoint coordinate pair `(x_{2k}, x_{2k+1})` of a
vector, the planar rotation by angle `m·θ_k`:

```
RoPE(x, m)_{2k}   = x_{2k}·cos(mθ_k) − x_{2k+1}·sin(mθ_k)
RoPE(x, m)_{2k+1} = x_{2k}·sin(mθ_k) + x_{2k+1}·cos(mθ_k)
```

Because RoPE is a direct sum of 2×2 rotations, every analytic guarantee
reduces to a fact about one planar rotation `rot`, proved here and then
lifted to the whole vector-of-pairs `ropePairs`.

## Obligations discharged (contract `rope-kernel-v1`)

* `invariant`  — norm preservation  `‖RoPE(x,m)‖ = ‖x‖`  → `ropePairs_normSq`
* `invariant`  — relative position  `⟨RoPE(q,m),RoPE(k,n)⟩ = f(q,k,m−n)` → `rope_relative`
* `postcondition` — length preservation `len(out) = len(x)` → `ropeList_length`
* `bound`      — `‖RoPE(x,m)‖ ≤ ‖x‖ + ε` → `ropePairs_norm_le`

Supporting identities (task-requested): `rope_identity` (m=0 identity) and
`rope_angle_add` (angle addition / composition of rotations).

All proofs are over `ℝ`; the f32/IEEE-finiteness and SIMD-ULP obligations are
runtime concerns and are marked `l4_not_applicable` in the contract, together
with the frame (buffer-non-mutation) obligation.
-/

namespace ProvableContracts.Rope

open Real Finset

/-- First coordinate of the planar rotation of `(x0, x1)` by angle `θ`. -/
noncomputable def rot0 (x0 x1 θ : ℝ) : ℝ := x0 * Real.cos θ - x1 * Real.sin θ

/-- Second coordinate of the planar rotation of `(x0, x1)` by angle `θ`. -/
noncomputable def rot1 (x0 x1 θ : ℝ) : ℝ := x0 * Real.sin θ + x1 * Real.cos θ

/-- Euclidean inner product of two planar vectors. -/
def dot2 (a0 a1 b0 b1 : ℝ) : ℝ := a0 * b0 + a1 * b1

/-!
## Pair-level (atomic) rotation facts
-/

/-- **Norm preservation (pair).** A planar rotation preserves the squared
    length of a coordinate pair — the atomic invariant behind RoPE norm
    preservation. Proof: expand and use `sin²+cos² = 1`. -/
theorem rope_norm_sq (x0 x1 θ : ℝ) :
    (rot0 x0 x1 θ) ^ 2 + (rot1 x0 x1 θ) ^ 2 = x0 ^ 2 + x1 ^ 2 := by
  unfold rot0 rot1
  have h : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1 := Real.sin_sq_add_cos_sq θ
  linear_combination (x0 ^ 2 + x1 ^ 2) * h

/-- **Identity at position 0.** `RoPE(x, 0) = x`: rotation by `0` is the
    identity because `cos 0 = 1` and `sin 0 = 0`. -/
theorem rope_identity (x0 x1 : ℝ) :
    rot0 x0 x1 0 = x0 ∧ rot1 x0 x1 0 = x1 := by
  unfold rot0 rot1
  constructor <;> simp [Real.cos_zero, Real.sin_zero]

/-- **Angle addition / composition.** Rotating by `a` then by `b` equals
    rotating by `a + b` — i.e. `R(b) ∘ R(a) = R(a+b)`. Proof: `cos_add`,
    `sin_add`, then `ring`. -/
theorem rope_angle_add (x0 x1 a b : ℝ) :
    rot0 (rot0 x0 x1 a) (rot1 x0 x1 a) b = rot0 x0 x1 (a + b) ∧
    rot1 (rot0 x0 x1 a) (rot1 x0 x1 a) b = rot1 x0 x1 (a + b) := by
  unfold rot0 rot1
  constructor <;> · rw [Real.cos_add, Real.sin_add]; ring

/-- **Relative-position property.** The inner product of two rotated pairs
    depends only on the *difference* of their rotation angles:
    `⟨R(a)q, R(b)k⟩ = ⟨q, R(b−a)k⟩`. This is the RoPE relative-position
    guarantee (with `a = mθ`, `b = nθ`, so the RHS depends only on `n−m`).
    Proof: `cos_sub`, `sin_sub`, then `ring` (a pure polynomial identity). -/
theorem rope_relative (q0 q1 k0 k1 a b : ℝ) :
    dot2 (rot0 q0 q1 a) (rot1 q0 q1 a) (rot0 k0 k1 b) (rot1 k0 k1 b)
      = dot2 q0 q1 (rot0 k0 k1 (b - a)) (rot1 k0 k1 (b - a)) := by
  unfold dot2 rot0 rot1
  rw [Real.cos_sub, Real.sin_sub]
  ring

/-!
## Vector-level lifting

A RoPE'd vector is the direct sum of planar rotations, one per coordinate
pair. We model it two ways: as `Fin n`-indexed pairs (for the algebraic
norm invariant) and as a `List` of pairs (for length preservation).
-/

/-- RoPE on a vector of `n` coordinate pairs, position-dependent angle `θ i`. -/
noncomputable def ropePairs {n : ℕ} (x : Fin n → ℝ × ℝ) (θ : Fin n → ℝ) :
    Fin n → ℝ × ℝ :=
  fun i => (rot0 (x i).1 (x i).2 (θ i), rot1 (x i).1 (x i).2 (θ i))

/-- Squared Euclidean norm of a vector of pairs. -/
noncomputable def normSq {n : ℕ} (v : Fin n → ℝ × ℝ) : ℝ :=
  ∑ i, ((v i).1 ^ 2 + (v i).2 ^ 2)

/-- **Norm preservation (vector).** `‖RoPE(x)‖² = ‖x‖²`: the whole-vector
    norm is preserved because each pair's norm is (via `rope_norm_sq`). -/
theorem ropePairs_normSq {n : ℕ} (x : Fin n → ℝ × ℝ) (θ : Fin n → ℝ) :
    normSq (ropePairs x θ) = normSq x := by
  unfold normSq ropePairs
  apply Finset.sum_congr rfl
  intro i _
  exact rope_norm_sq (x i).1 (x i).2 (θ i)

/-- **Norm bound (vector).** `‖RoPE(x)‖ ≤ ‖x‖`: immediate from the norm
    equality (with `ε ≥ 0` the contract's `‖RoPE(x)‖ ≤ ‖x‖ + ε` follows). -/
theorem ropePairs_norm_le {n : ℕ} (x : Fin n → ℝ × ℝ) (θ : Fin n → ℝ) :
    Real.sqrt (normSq (ropePairs x θ)) ≤ Real.sqrt (normSq x) := by
  rw [ropePairs_normSq]

/-- RoPE on a `List` of coordinate pairs at a single position angle `θ`. -/
noncomputable def ropeList (xs : List (ℝ × ℝ)) (θ : ℝ) : List (ℝ × ℝ) :=
  xs.map (fun p => (rot0 p.1 p.2 θ, rot1 p.1 p.2 θ))

/-- **Length preservation (postcondition).** `len(RoPE(x)) = len(x)`: RoPE
    maps `ℝ^d → ℝ^d`, so the output has exactly the input length. -/
theorem ropeList_length (xs : List (ℝ × ℝ)) (θ : ℝ) :
    (ropeList xs θ).length = xs.length := by
  unfold ropeList
  simp

-- Sanity checks
#check @rope_norm_sq
#check @rope_identity
#check @rope_angle_add
#check @rope_relative
#check @ropePairs_normSq
#check @ropePairs_norm_le
#check @ropeList_length

end ProvableContracts.Rope
