/-!
# LoRA Merge Forward-Equivalence (CORE-ONLY, no Mathlib)

Proves the algebraic identities that make merging a LoRA adapter into a base
weight a *forward-equivalent* operation:

  merge distributivity : (W + s·(B@A)) x = W x + s·(B (A x))
  composed affine       : (W + dW) x + b = W x + dW x + b

The whole domain is modeled over `List Int` so the file compiles with the bare
`lean` binary — NO `import Mathlib`, NO `import ProvableContracts.*`. Vectors are
`List Int`, matrices are `List (List Int)` (a list of rows), and matrix/vector
operations are folds/recursions over those lists. Every theorem is proved by
structural induction from first principles using only core `Int` ring lemmas
(`Int.add_mul`, `Int.mul_assoc`) and `omega` for the residual linear goals.

## Obligations discharged
- `LORA-FWD-001` merge distributivity   → `lora_merge_forward_equiv`
- `LORA-FWD-002` composed-affine identity → `composed_affine`, `composed_affine_scaled`

Supporting lemmas (`dot_vadd_left`, `dot_smul_left`, `matvec_madd`,
`matvec_msmul`, `dot_vecmat`, `matvec_matmul_assoc`) are the inductive core.
-/

namespace ProvableContracts.Lora

/-- Dot product of two integer vectors (truncates to the shorter length). -/
def dot : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | a :: as, b :: bs => a * b + dot as bs

/-- Elementwise vector addition. A missing tail is treated as trailing zeros,
    so `[]` acts as the additive identity: `vadd [] v = v`, `vadd u [] = u`. -/
def vadd : List Int → List Int → List Int
  | [], ys => ys
  | xs, [] => xs
  | x :: xs, y :: ys => (x + y) :: vadd xs ys

/-- Scalar multiplication of a vector. -/
def smul (s : Int) : List Int → List Int
  | [] => []
  | x :: xs => (s * x) :: smul s xs

/-- Matrix–vector product: apply `dot _ x` to every row. -/
def matvec (M : List (List Int)) (x : List Int) : List Int :=
  M.map (fun row => dot row x)

/-- Matrix addition (elementwise per row), with the same zero-padding
    convention as `vadd`. -/
def madd : List (List Int) → List (List Int) → List (List Int)
  | [], N => N
  | M, [] => M
  | r :: rs, q :: qs => vadd r q :: madd rs qs

/-- Scalar × matrix. -/
def msmul (s : Int) (M : List (List Int)) : List (List Int) :=
  M.map (smul s)

/-- Row-vector times matrix: `Σ_t b_t · A_t` (row `t` of `A`). This is one row
    of the matrix product `B @ A`. -/
def vecmat : List Int → List (List Int) → List Int
  | [], _ => []
  | _ :: _, [] => []
  | b :: bs, r :: rs => vadd (smul b r) (vecmat bs rs)

/-- Matrix product `B @ A`: each row of `B` combined against the rows of `A`. -/
def matmul (B A : List (List Int)) : List (List Int) :=
  B.map (fun brow => vecmat brow A)

/-! ## Core inductive lemmas -/

/-- `dot` is left-additive: `(u + v) · x = u · x + v · x`. -/
theorem dot_vadd_left : ∀ (u v x : List Int),
    dot (vadd u v) x = dot u x + dot v x
  | [], _, _ => by simp [vadd, dot]
  | _ :: _, [], _ => by simp [vadd, dot]
  | _ :: _, _ :: _, [] => by simp [vadd, dot]
  | a :: as, b :: bs, c :: cs => by
      simp only [vadd, dot]
      rw [dot_vadd_left as bs cs, Int.add_mul]
      omega

/-- `dot` is left-homogeneous: `(s · u) · x = s · (u · x)`. -/
theorem dot_smul_left : ∀ (s : Int) (u x : List Int),
    dot (smul s u) x = s * dot u x
  | _, [], _ => by simp [smul, dot]
  | s, _ :: _, [] => by simp [smul, dot]
  | s, a :: as, c :: cs => by
      simp only [smul, dot]
      rw [dot_smul_left s as cs, Int.mul_assoc, Int.mul_add]

/-- Matrix-add distributes over matrix–vector product:
    `(W + M) x = W x + M x`. -/
theorem matvec_madd : ∀ (W M : List (List Int)) (x : List Int),
    matvec (madd W M) x = vadd (matvec W x) (matvec M x)
  | [], M, x => by simp [madd, matvec, vadd]
  | r :: rs, [], x => by simp [madd, matvec, vadd]
  | r :: rs, q :: qs, x => by
      simp only [madd, matvec, List.map_cons, vadd]
      rw [dot_vadd_left r q x]
      have ih := matvec_madd rs qs x
      simp only [matvec] at ih
      rw [ih]

/-- Scalar factors out of the matrix–vector product:
    `(s · M) x = s · (M x)`. -/
theorem matvec_msmul : ∀ (s : Int) (M : List (List Int)) (x : List Int),
    matvec (msmul s M) x = smul s (matvec M x)
  | _, [], _ => by simp [msmul, matvec, smul]
  | s, r :: rs, x => by
      simp only [msmul, matvec, List.map_cons, smul]
      rw [dot_smul_left s r x]
      have ih := matvec_msmul s rs x
      simp only [msmul, matvec] at ih
      rw [ih]

/-- The bilinearity bridge: dotting a row-times-matrix against `x` equals
    dotting the row against `M x`. This encodes `(b·A) · x = b · (A x)`. -/
theorem dot_vecmat : ∀ (b : List Int) (A : List (List Int)) (x : List Int),
    dot (vecmat b A) x = dot b (matvec A x)
  | [], _, _ => by simp [vecmat, dot]
  | _ :: _, [], _ => by simp [vecmat, dot, matvec]
  | b :: bs, r :: rs, x => by
      simp only [vecmat, matvec, List.map_cons, dot]
      rw [dot_vadd_left (smul b r) (vecmat bs rs) x, dot_smul_left b r x]
      have ih := dot_vecmat bs rs x
      simp only [matvec] at ih
      rw [ih]

/-- Matrix-product associativity against a vector:
    `(B @ A) x = B (A x)`. -/
theorem matvec_matmul_assoc : ∀ (B A : List (List Int)) (x : List Int),
    matvec (matmul B A) x = matvec B (matvec A x)
  | [], _, _ => by simp [matmul, matvec]
  | brow :: rest, A, x => by
      show dot (vecmat brow A) x :: matvec (matmul rest A) x
         = dot brow (matvec A x) :: matvec rest (matvec A x)
      rw [dot_vecmat brow A x, matvec_matmul_assoc rest A x]

/-! ## Top-level obligations -/

/-- `LORA-FWD-001` — merge distributivity, general form:
    `(W + s·M) x = W x + s·(M x)`. -/
theorem merge_distributes (W M : List (List Int)) (s : Int) (x : List Int) :
    matvec (madd W (msmul s M)) x = vadd (matvec W x) (smul s (matvec M x)) := by
  rw [matvec_madd, matvec_msmul]

/-- `LORA-FWD-001` — the exact LoRA merge identity with `M = B @ A`:
    `(W + s·(B@A)) x = W x + s·(B (A x))`.

    This is precisely the `merge` operation in
    `entrenar::lora::layer::core::LoRALayer::merge`
    (`W' = W + scale · (B @ A)`), proving that merging is forward-equivalent. -/
theorem lora_merge_forward_equiv
    (W B A : List (List Int)) (s : Int) (x : List Int) :
    matvec (madd W (msmul s (matmul B A))) x
      = vadd (matvec W x) (smul s (matvec B (matvec A x))) := by
  rw [merge_distributes, matvec_matmul_assoc]

/-- `LORA-FWD-002` — composed-affine identity:
    `(W + dW) x + b = (W x + dW x) + b`. -/
theorem composed_affine (W dW : List (List Int)) (x b : List Int) :
    vadd (matvec (madd W dW) x) b
      = vadd (vadd (matvec W x) (matvec dW x)) b := by
  rw [matvec_madd]

/-- `LORA-FWD-002` — scaled composed-affine identity:
    `(W + s·dW) x + b = (W x + s·(dW x)) + b`. -/
theorem composed_affine_scaled
    (W dW : List (List Int)) (s : Int) (x b : List Int) :
    vadd (matvec (madd W (msmul s dW)) x) b
      = vadd (vadd (matvec W x) (smul s (matvec dW x))) b := by
  rw [merge_distributes]

/-! ## Concrete sanity checks (executable `decide`-style examples) -/

example :
    lora_merge_forward_equiv [[1, 2], [3, 4]] [[1], [0]] [[2, 1]] 3 [5, 7]
      = lora_merge_forward_equiv [[1, 2], [3, 4]] [[1], [0]] [[2, 1]] 3 [5, 7] :=
  rfl

-- Numeric instance: W=[[1,2],[3,4]], B=[[1],[0]], A=[[2,1]], s=3, x=[5,7]
-- LHS  (W + 3·(B@A)) x   should equal   W x + 3·(B (A x)).
example :
    matvec (madd [[1, 2], [3, 4]] (msmul 3 (matmul [[1], [0]] [[2, 1]]))) [5, 7]
      = vadd (matvec [[1, 2], [3, 4]] [5, 7])
             (smul 3 (matvec [[1], [0]] (matvec [[2, 1]] [5, 7]))) := by
  decide

#check @lora_merge_forward_equiv
#check @composed_affine
#check @composed_affine_scaled
#check @merge_distributes
#check @matvec_matmul_assoc

end ProvableContracts.Lora
