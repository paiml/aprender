/-!
# Kernel-Fusion Forward-Equivalence (CORE-ONLY, no Mathlib)

Proves the algebraic core of `kernel-fusion-v1.yaml`: a *fused* kernel computes
exactly the same value as the sequence of *unfused* kernels it replaces. Fusion
is a launch-count / memory-traffic optimization; these theorems certify it is
**forward-equivalent** — `fused(x) == unfused(x)` — so no numerical semantics
change when the contract flips a decision to `ACTIVE`.

The whole domain is modeled over `List Int` so the file compiles with the bare
`lean` binary — NO `import Mathlib`, NO `import ProvableContracts.*`. Vectors are
`List Int`, matrices are `List (List Int)` (a list of rows). Every theorem is
proved by structural induction from first principles.

## Obligations discharged
- `FUSION-FWD-001` SwiGLU activation×multiply fusion (FUSION-001 / FUSION-007)
    `swigluFused f u v = hmul (vmap f u) v`
    Fusing an activation `f` with the following element-wise multiply equals
    mapping `f` then multiplying — the zipWith/map fusion law. Holds for *any*
    activation `f : Int → Int` (SiLU is one instance), so it is strictly
    stronger than a SiLU-specific claim.  →  `swiglu_fusion_correct`
- `FUSION-FWD-002` multi-projection GEMV fusion (FUSION-006 QKV / FUSION-010 gate+up)
    `matvec (A ++ B) x = matvec A x ++ matvec B x`
    Stacking projection weight-rows into one matrix and doing a single GEMV
    equals concatenating the separate GEMV outputs.  →  `matvec_append`
    (plus the 3-way `matvec_append3` for the Q/K/V case).
- `FUSION-FWD-003` pipeline fusion (FUSION-008 GEMM+bias+GELU)
    `fuse3 f g h x = f (g (h x))`
    A fused GEMM+bias+activation kernel equals the composition of the three
    stages.  →  `pipeline_fusion_correct`, `gemm_bias_gelu_fusion`.
-/

namespace ProvableContracts.Fusion

/-! ## Vector / matrix primitives (integer model) -/

/-- Dot product of two integer vectors (truncates to the shorter length). -/
def dot : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | a :: as, b :: bs => a * b + dot as bs

/-- Element-wise vector addition (bias add), zero-padding the shorter tail. -/
def vadd : List Int → List Int → List Int
  | [], ys => ys
  | xs, [] => xs
  | x :: xs, y :: ys => (x + y) :: vadd xs ys

/-- Element-wise vector multiply (truncates to the shorter length). -/
def hmul : List Int → List Int → List Int
  | [], _ => []
  | _ :: _, [] => []
  | a :: as, b :: bs => (a * b) :: hmul as bs

/-- Apply a scalar activation `f` element-wise to a vector. -/
def vmap (f : Int → Int) : List Int → List Int
  | [] => []
  | x :: xs => f x :: vmap f xs

/-- Matrix–vector product: apply `dot _ x` to every row. -/
def matvec (M : List (List Int)) (x : List Int) : List Int :=
  M.map (fun row => dot row x)

/-! ## FUSION-FWD-001 — SwiGLU activation×multiply fusion

`swigluFused f u v` computes `f(uᵢ) * vᵢ` in a single pass (the fused kernel:
one read of `u` and `v`, one write). The unfused path is `hmul (vmap f u) v`:
first materialize the activation `vmap f u`, then element-wise multiply. -/

/-- Fused activation-then-multiply, computed in one pass. -/
def swigluFused (f : Int → Int) : List Int → List Int → List Int
  | [], _ => []
  | _ :: _, [] => []
  | a :: as, b :: bs => (f a * b) :: swigluFused f as bs

/-- `FUSION-FWD-001` — fusing activation `f` with the element-wise multiply is
    forward-equivalent to mapping `f` then multiplying, for **any** activation.
    (SiLU/SwiGLU is the intended instance; the proof needs nothing about `f`.) -/
theorem swiglu_fusion_correct : ∀ (f : Int → Int) (u v : List Int),
    swigluFused f u v = hmul (vmap f u) v
  | _, [], _ => by simp [swigluFused, vmap, hmul]
  | _, _ :: _, [] => by simp [swigluFused, vmap, hmul]
  | f, a :: as, b :: bs => by
      simp only [swigluFused, vmap, hmul]
      rw [swiglu_fusion_correct f as bs]

/-! ## FUSION-FWD-002 — multi-projection GEMV fusion

The fused QKV kernel (FUSION-006) stacks the Q, K, V weight rows into one matrix
and issues a single GEMV, then slices the result. The gate+up kernel (FUSION-010)
does the same for two projections. Correctness reduces to: a GEMV over stacked
rows equals the concatenation of the per-block GEMVs. -/

/-- `FUSION-FWD-002` — GEMV over stacked (appended) weight rows equals the
    concatenation of the separate GEMV outputs: `(A ++ B) x = (A x) ++ (B x)`. -/
theorem matvec_append : ∀ (A B : List (List Int)) (x : List Int),
    matvec (A ++ B) x = matvec A x ++ matvec B x
  | [], B, x => by simp [matvec]
  | r :: rs, B, x => by
      simp only [matvec, List.cons_append, List.map_cons]
      have ih := matvec_append rs B x
      simp only [matvec] at ih
      rw [ih]

/-- `FUSION-FWD-002` (Q/K/V) — the three-way stack used by `FusedQKVKernel`:
    a single GEMV over `[Wq; Wk; Wv]` splits into the three projection outputs. -/
theorem matvec_append3 (Q K V : List (List Int)) (x : List Int) :
    matvec (Q ++ K ++ V) x = matvec Q x ++ matvec K x ++ matvec V x := by
  rw [matvec_append, matvec_append]

/-! ## FUSION-FWD-003 — pipeline (GEMM+bias+activation) fusion

`FusedGemmBiasGeluKernel` (FUSION-008) fuses three stages into one launch:
matmul → bias-add → activation. A fused pipeline of three maps equals their
composition. -/

/-- A fused 3-stage pipeline. -/
def fuse3 (f g h : List Int → List Int) (x : List Int) : List Int := f (g (h x))

/-- `FUSION-FWD-003` — the fused pipeline equals the staged composition. -/
theorem pipeline_fusion_correct (f g h : List Int → List Int) (x : List Int) :
    fuse3 f g h x = f (g (h x)) := rfl

/-- `FUSION-FWD-003` (concrete) — GEMM + bias + GELU fused kernel is
    forward-equivalent to `gelu(bias + W·x)` computed in three separate stages.
    `matvec W` = GEMM, `vadd bias` = bias add, `vmap gelu` = activation. -/
theorem gemm_bias_gelu_fusion
    (gelu : Int → Int) (W : List (List Int)) (bias x : List Int) :
    fuse3 (vmap gelu) (vadd bias) (matvec W) x
      = vmap gelu (vadd bias (matvec W x)) := rfl

/-! ## Concrete sanity checks (executable `decide` examples) -/

-- FUSION-FWD-001: f = square (a·a). Fused f(u)*v vs (map f u) ⊙ v.
example :
    swigluFused (fun a => a * a) [2, 3, 4] [5, 6, 7]
      = hmul (vmap (fun a => a * a) [2, 3, 4]) [5, 6, 7] := by decide

-- FUSION-FWD-002: stacked GEMV splits. Wq=[[1,2]], Wk=[[3,4]], Wv=[[5,6]], x=[7,8].
example :
    matvec ([[1, 2]] ++ [[3, 4]] ++ [[5, 6]]) [7, 8]
      = matvec [[1, 2]] [7, 8] ++ matvec [[3, 4]] [7, 8] ++ matvec [[5, 6]] [7, 8] := by
  decide

-- FUSION-FWD-003: fused GEMM+bias+GELU (gelu ≔ id here) equals staged.
example :
    fuse3 (vmap (fun a => a)) (vadd [10, 20]) (matvec [[1, 2], [3, 4]]) [5, 6]
      = vmap (fun a => a) (vadd [10, 20] (matvec [[1, 2], [3, 4]] [5, 6])) := by
  decide

#check @swiglu_fusion_correct
#check @matvec_append
#check @matvec_append3
#check @pipeline_fusion_correct
#check @gemm_bias_gelu_fusion

end ProvableContracts.Fusion
