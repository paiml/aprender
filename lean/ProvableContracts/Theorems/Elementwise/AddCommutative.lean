import ProvableContracts.Defs.Elementwise

/-!
# Elementwise Addition Commutativity

Proves that elementwise vector addition is commutative: add(a, b) = add(b, a).

## Obligation

`EW-INV-002`: ∀ a, b ∈ ℝⁿ, add(a, b) = add(b, a)

This follows pointwise from commutativity of real addition.
-/

namespace ProvableContracts.Elementwise

-- Status: proved
/-- Elementwise addition is commutative: add(a, b) = add(b, a). -/
theorem add_comm_vec {n : ℕ} (a b : RVec n) :
    add a b = add b a := by
  funext i
  simp only [add]
  exact _root_.add_comm (a i) (b i)

-- Status: proved
/-- Elementwise addition is associative: add(add(a,b),c) = add(a,add(b,c)). -/
theorem add_assoc_vec {n : ℕ} (a b c : RVec n) :
    add (add a b) c = add a (add b c) := by
  funext i
  simp only [add]
  exact _root_.add_assoc (a i) (b i) (c i)

-- Tests
#check @add_comm_vec
#check @add_assoc_vec

example : add (fun _ : Fin 2 => (1 : ℝ)) (fun _ => 2) = add (fun _ => 2) (fun _ => 1) :=
  add_comm_vec (fun _ => 1) (fun _ => 2)

end ProvableContracts.Elementwise
