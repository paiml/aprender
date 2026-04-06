import Mathlib.Data.Real.Basic
import ProvableContracts.Basic

/-!
# Elementwise Operation Definitions

Mathematical definitions of elementwise operations on real-valued vectors:
ReLU, addition, scalar multiplication, Leaky ReLU, subtraction, division,
and negation.

## References

- Nair & Hinton (2010) Rectified Linear Units Improve Restricted Boltzmann Machines
- Maas et al. (2013) Rectifier Nonlinearities Improve Neural Network Acoustic Models
-/

namespace ProvableContracts.Elementwise

/-- ReLU activation: relu(x) = max(0, x). -/
noncomputable def relu (x : ℝ) : ℝ :=
  max 0 x

/-- Leaky ReLU: leaky_relu(α, x) = x if x ≥ 0, else α·x. -/
noncomputable def leaky_relu (α : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x else α * x

/-- Elementwise vector addition: (a + b)ᵢ = aᵢ + bᵢ. -/
def add {n : ℕ} (a b : RVec n) : RVec n :=
  fun i => a i + b i

/-- Elementwise vector subtraction: (a - b)ᵢ = aᵢ - bᵢ. -/
def sub {n : ℕ} (a b : RVec n) : RVec n :=
  fun i => a i - b i

/-- Scalar-vector multiplication: (α * x)ᵢ = α · xᵢ. -/
def mul_scalar (α : ℝ) {n : ℕ} (x : RVec n) : RVec n :=
  fun i => α * x i

/-- Elementwise vector division: (a / b)ᵢ = aᵢ / bᵢ. -/
noncomputable def div {n : ℕ} (a b : RVec n) : RVec n :=
  fun i => a i / b i

/-- Elementwise negation: (-x)ᵢ = -(xᵢ). -/
def neg {n : ℕ} (x : RVec n) : RVec n :=
  fun i => -(x i)

end ProvableContracts.Elementwise
