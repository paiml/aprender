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

#check @matmul_output_shape
#check @matmul_output_len
#check @matmul_shape_inner_cancels

end ProvableContracts.MatMul.Shape
