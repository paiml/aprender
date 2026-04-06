import ProvableContracts.Defs.Sigmoid
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Sigmoid Symmetry

Proves that σ(-x) = 1 - σ(x).

## Obligation

`SI-INV-SYMMETRY`: sigmoid(-x) = 1 - sigmoid(x)

This is a standard identity. Proof:
σ(-x) = 1/(1+eˣ) and σ(x) = 1/(1+e⁻ˣ) = eˣ/(eˣ+1).
So 1 - σ(x) = 1/(eˣ+1) = σ(-x).
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Sigmoid symmetry: σ(-x) = 1 - σ(x).
    Both sides equal 1/(1 + eˣ). -/
theorem sigmoid_symmetry (x : ℝ) :
    sigmoid (-x) = 1 - sigmoid x := by
  unfold sigmoid
  simp only [neg_neg]
  have he : Real.exp x > 0 := Real.exp_pos x
  have hem : Real.exp (-x) > 0 := Real.exp_pos (-x)
  have h1 : 1 + Real.exp x > 0 := by linarith
  have h2 : 1 + Real.exp (-x) > 0 := by linarith
  field_simp
  rw [Real.exp_neg]
  field_simp
  ring

-- Tests
#check @sigmoid_symmetry

end ProvableContracts.Sigmoid
