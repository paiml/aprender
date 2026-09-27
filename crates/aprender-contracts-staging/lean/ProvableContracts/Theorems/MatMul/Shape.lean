/-!
# Matrix Multiplication Output Shape

Contract: `matmul-kernel-v1`, proof-obligation *Output shape correctness*
(`formal: shape(A @ B) = (rows(A), cols(B))`).

For `A : m×k` and `B : k×n`, the product `A·B` has shape `(m, n)`: the inner
dimension `k` cancels. We additionally prove the row-major flattened result has
length `m * n`, matching the contract postcondition `result.len() == m * n` and
precondition `a.len() == m*k`, `b.len() == k*n`.

The proof is core Lean (no Mathlib): shapes are `(rows, cols) : Nat × Nat` and
a flattened buffer length is `rows * cols`.
-/

namespace ProvableContracts.MatMul.Shape

/-- A tensor shape as `(rows, cols)`. -/
abbrev Shape := Nat × Nat

/-- Shape of the product `A·B`: the inner dimension cancels, leaving
    `(rows A, cols B)`. -/
def matmul_shape (a b : Shape) : Shape := (a.1, b.2)

/-- Flattened row-major buffer length of a shape `(rows, cols)`. -/
def buffer_len (s : Shape) : Nat := s.1 * s.2

-- Status: proved (core Lean)
/-- Output shape correctness: `matmul(A[m,k], B[k,n])` has shape `(m, n)`. -/
theorem matmul_output_shape (m k n : Nat) :
    matmul_shape (m, k) (k, n) = (m, n) := rfl

-- Status: proved (core Lean)
/-- The row-major result buffer has length `m * n` (contract postcondition
    `result.len() == m * n`). -/
theorem matmul_output_len (m k n : Nat) :
    buffer_len (matmul_shape (m, k) (k, n)) = m * n := rfl

-- Status: proved (core Lean)
/-- The inner dimension `k` does not appear in the output shape: the product of
    an `m×k` and a `k×n` matrix is independent of `k` in its shape. -/
theorem matmul_shape_inner_cancels (m k k' n : Nat) :
    matmul_shape (m, k) (k, n) = matmul_shape (m, k') (k', n) := rfl

/-! ## L4 simulation model of `Tensor::matmul`

The defs above are an abstract `(rows, cols)` algebra. The model below
transcribes the shape path of the Rust item itself,
`crates/aprender-core/src/autograd/ops/activation.rs:274-299`
(`impl Tensor { pub fn matmul(&self, other: &Tensor) -> Tensor }`):

* `:275-276` `assert_eq!(self.ndim(), 2)` / `assert_eq!(other.ndim(), 2)`
  — any shape whose length is not 2 panics (`none`);
* `:278-280` `(m, k1)`, `(k2, n)`, `assert_eq!(k1, k2)` — mismatch panics;
* `:282-296` `data` is the GEMV branch (`m == 1`, `vec![0.0; n]`) or the trueno
  `Matrix` branch (`m × n` row-major);
* `:298` `Tensor::from_vec(data, &[m, n])`.

A Rust panic is `none`. The witness is
`crates/aprender-core/src/autograd/ops/tests_matmul_shape_l4.rs`, which checks
the Rust fn against `rust_matmul_golden` (generated from `#eval` below). -/

/-- `Tensor::matmul`'s output shape, `activation.rs:275-280,298`. -/
def tensorMatmulShape (a b : List Nat) : Option (List Nat) :=
  match a, b with
  | [m, k1], [k2, n] => if k1 = k2 then some [m, n] else none
  | _, _ => none

/-- Length of `data` in `Tensor::matmul`, `activation.rs:282-296`: the GEMV
    branch allocates `n`, the general branch returns `m * n`. -/
def tensorMatmulDataLen (m n : Nat) : Nat :=
  if m = 1 then n else m * n

/-- Both data branches produce a buffer of exactly `m * n` elements, so
    `Tensor::from_vec(data, &[m, n])` is always well-formed. -/
theorem tensor_matmul_data_len (m n : Nat) :
    tensorMatmulDataLen m n = m * n := by
  unfold tensorMatmulDataLen
  split
  · subst_vars; simp
  · rfl

/-- The model returns a shape exactly when both inputs are 2-D and the inner
    dimensions agree, and that shape is `[m, n]`. -/
theorem tensor_matmul_shape_some_iff (a b out : List Nat) :
    tensorMatmulShape a b = some out ↔
      ∃ m k n, a = [m, k] ∧ b = [k, n] ∧ out = [m, n] := by
  constructor
  · intro h
    unfold tensorMatmulShape at h
    split at h
    · rename_i m k1 k2 n
      split at h
      · rename_i hk
        subst hk
        cases h
        exact ⟨m, k1, n, rfl, rfl, rfl⟩
      · cases h
    · cases h
  · rintro ⟨m, k, n, rfl, rfl, rfl⟩
    simp [tensorMatmulShape]

/-- Output shape correctness over the Rust model. -/
theorem tensor_matmul_output_shape (m k n : Nat) :
    tensorMatmulShape [m, k] [k, n] = some [m, n] :=
  (tensor_matmul_shape_some_iff _ _ _).2 ⟨m, k, n, rfl, rfl, rfl⟩

/-- An inner-dimension mismatch panics (`assert_eq!(k1, k2)`). -/
theorem tensor_matmul_mismatch_panics (m k1 k2 n : Nat) (h : k1 ≠ k2) :
    tensorMatmulShape [m, k1] [k2, n] = none := by
  simp [tensorMatmulShape, h]

/-- A left operand that is not 2-D panics (`assert_eq!(self.ndim(), 2)`). -/
theorem tensor_matmul_left_not_2d_panics (a b : List Nat) (h : a.length ≠ 2) :
    tensorMatmulShape a b = none := by
  match a, b, h with
  | [_, _], _, h => exact absurd rfl h
  | [], _, _ => rfl
  | [_], _, _ => rfl
  | _ :: _ :: _ :: _, _, _ => rfl

/-- A right operand that is not 2-D panics (`assert_eq!(other.ndim(), 2)`). -/
theorem tensor_matmul_right_not_2d_panics (a b : List Nat) (h : b.length ≠ 2) :
    tensorMatmulShape a b = none := by
  cases hs : tensorMatmulShape a b with
  | none => rfl
  | some out =>
    obtain ⟨_, _, _, _, rfl, _⟩ := (tensor_matmul_shape_some_iff a b out).1 hs
    exact absurd rfl h

/-- The abstract algebra above is the Rust model on its defined domain: where
    `Tensor::matmul` returns, its shape is `matmul_shape` and its buffer length
    is `buffer_len (matmul_shape ..)`. -/
theorem tensor_matmul_refines_abstract (m k n : Nat) :
    tensorMatmulShape [m, k] [k, n] =
        some [(matmul_shape (m, k) (k, n)).1, (matmul_shape (m, k) (k, n)).2] ∧
      tensorMatmulDataLen m n = buffer_len (matmul_shape (m, k) (k, n)) :=
  ⟨tensor_matmul_output_shape m k n, tensor_matmul_data_len m n⟩

/-- Every shape of rank 1..3 with dims in 1..3 (39 shapes). Dim 0 is excluded:
    the Rust general branch hands it to `trueno::Matrix::from_vec`. -/
def goldenShapes : List (List Nat) :=
  let d := [1, 2, 3]
  (d.map fun a => [a]) ++
  (d.flatMap fun a => d.map fun b => [a, b]) ++
  (d.flatMap fun a => d.flatMap fun b => d.map fun c => [a, b, c])

/-- One golden line per (a, b) pair: `a;b;out` with `out` = `panic` or dims. -/
def rustMatmulGolden : List String :=
  let fmt (s : List Nat) : String := ",".intercalate (s.map toString)
  goldenShapes.flatMap fun a => goldenShapes.map fun b =>
    match tensorMatmulShape a b with
    | none => s!"{fmt a};{fmt b};panic"
    | some o => s!"{fmt a};{fmt b};{fmt o}"

#check @matmul_output_shape
#check @matmul_output_len
#check @matmul_shape_inner_cancels
#check @tensor_matmul_shape_some_iff
#check @tensor_matmul_refines_abstract

end ProvableContracts.MatMul.Shape
