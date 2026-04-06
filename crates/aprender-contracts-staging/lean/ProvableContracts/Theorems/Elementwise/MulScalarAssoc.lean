import ProvableContracts.Defs.Elementwise

/-!
# Scalar Multiplication Associativity

Proves that scalar multiplication is associative:
mul_scalar(a, mul_scalar(b, x)) = mul_scalar(a*b, x).

## Obligation

`EW-INV-003`: ∀ a, b ∈ ℝ, ∀ x ∈ ℝⁿ, (a·b)·x = a·(b·x)

This follows pointwise from associativity of real multiplication.
-/

namespace ProvableContracts.Elementwise

-- Status: proved
/-- Scalar multiplication is associative:
    mul_scalar(a, mul_scalar(b, x)) = mul_scalar(a*b, x). -/
theorem mul_scalar_assoc {n : ℕ} (a b : ℝ) (x : RVec n) :
    mul_scalar a (mul_scalar b x) = mul_scalar (a * b) x := by
  funext i
  simp only [mul_scalar]
  exact (mul_assoc a b (x i)).symm

-- Status: proved
/-- Scalar multiplication distributes over vector addition:
    mul_scalar(α, add(a, b)) = add(mul_scalar(α, a), mul_scalar(α, b)). -/
theorem mul_scalar_add_distrib {n : ℕ} (α : ℝ) (a b : RVec n) :
    mul_scalar α (add a b) = add (mul_scalar α a) (mul_scalar α b) := by
  funext i
  simp only [mul_scalar, add]
  exact mul_add α (a i) (b i)

-- Tests
#check @mul_scalar_assoc
#check @mul_scalar_add_distrib

example : mul_scalar 2 (mul_scalar 3 (fun _ : Fin 2 => (1 : ℝ))) =
    mul_scalar (2 * 3) (fun _ => 1) :=
  mul_scalar_assoc 2 3 (fun _ => 1)

end ProvableContracts.Elementwise
