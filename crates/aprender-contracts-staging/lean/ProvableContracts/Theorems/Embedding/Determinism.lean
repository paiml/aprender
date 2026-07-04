import ProvableContracts.Defs.Embedding

/-!
# Embedding Lookup — Deterministic Output

Contract obligation: *Deterministic output* —
`lookup(W, ids) = lookup(W, ids)` for identical `W` and `ids`.

The lookup is a pure total function of `(table, ids, dflt)`, so equal inputs
produce bit-identical outputs. We state this as an input-congruence: any two
calls whose arguments are pairwise equal yield equal results. There is no
hidden state, uninitialized memory, or concurrency in the model — the only
source of the output is the arguments.
-/

namespace ProvableContracts.Embedding

/-- **Determinism (input congruence).** Two lookups with pairwise-equal
    arguments produce identical output. -/
theorem gather_deterministic {α : Type _} (t₁ t₂ : List α) (i₁ i₂ : List ℕ)
    (d₁ d₂ : α) (ht : t₁ = t₂) (hi : i₁ = i₂) (hd : d₁ = d₂) :
    gather t₁ i₁ d₁ = gather t₂ i₂ d₂ := by
  subst ht; subst hi; subst hd; rfl

/-- Specialisation: the same call is equal to itself (reflexive determinism),
    the exact "two identical calls agree" statement in the contract. -/
theorem gather_self_eq {α : Type _} (table : List α) (ids : List ℕ) (dflt : α) :
    gather table ids dflt = gather table ids dflt :=
  gather_deterministic table table ids ids dflt dflt rfl rfl rfl

end ProvableContracts.Embedding
