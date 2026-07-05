import ProvableContracts.Defs.KLDivergence
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Forward-KL Knowledge-Distillation Loss — Core Analytic Theorems

Proves the analytic core of `kd-loss-forward-kl-v1.yaml`:

* `kl_forward_def`   — the soft-target loss is the FORWARD KL form
  `Σ p_t·(log p_t − log p_s)` (definitional, `KD-POST-001`).
* `kl_self_zero`     — `KL(p ‖ p) = 0`: identical distributions give zero
  loss (`KD-INV-001`).
* `kl_nonneg`        — Gibbs' inequality `KL(p ‖ q) ≥ 0` for probability
  distributions (`KD-BND-001`).
* `kl_soft_nonneg_and_forward` — the combined postcondition
  `KL_soft ≥ 0 ∧ KL_soft = Σ p_t·(log p_t − log p_s)` (`KD-POST-001`).

The non-negativity proof is the classic Gibbs argument via
`Real.log_le_sub_one_of_pos` (`log x ≤ x − 1`):

  `Σ p_i (log p_i − log q_i)` = `− Σ p_i log(q_i/p_i)`
    `≥ − Σ p_i (q_i/p_i − 1)` = `− Σ (q_i − p_i)` = `−(1 − 1)` = `0`.

## References

- Kullback & Leibler (1951); Gibbs' inequality.
- `Real.log_le_sub_one_of_pos` (Mathlib).
-/

namespace ProvableContracts.KLDivergence

open Real Finset

/-- **`KD-POST-001` (definitional half).** The soft-target loss is the
    forward-KL form `Σ_i p_t_i·(log p_t_i − log p_s_i)` — the teacher
    distribution `p_t` is the outer measure.  Definitional (`rfl`). -/
theorem kl_forward_def {n : ℕ} (p q : RVec n) :
    kl p q = ∑ i : Fin n, p i * (Real.log (p i) - Real.log (q i)) := rfl

/-- **`KD-INV-001`.** Zero soft-target loss exactly when the student
    distribution equals the teacher: `KL(p ‖ p) = 0`.  Each summand is
    `p_i·(log p_i − log p_i) = 0`. -/
theorem kl_self_zero {n : ℕ} (p : RVec n) : kl p p = 0 := by
  unfold kl
  apply Finset.sum_eq_zero
  intro i _
  ring

/-- **`KD-BND-001` (Gibbs' inequality).** Forward KL is a non-negative
    divergence: for probability distributions `p, q` (strictly positive
    entries summing to one), `KL(p ‖ q) ≥ 0`.

    Proof: `log(q_i/p_i) ≤ q_i/p_i − 1`, so
    `p_i·(log p_i − log q_i) ≥ p_i − q_i`, and summing gives
    `KL(p ‖ q) ≥ Σ(p_i − q_i) = 1 − 1 = 0`. -/
theorem kl_nonneg {n : ℕ} (p q : RVec n)
    (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hsp : ∑ i : Fin n, p i = 1) (hsq : ∑ i : Fin n, q i = 1) :
    0 ≤ kl p q := by
  have hsum0 : ∑ i : Fin n, (p i - q i) = 0 := by
    rw [Finset.sum_sub_distrib, hsp, hsq]; ring
  have hpt : ∀ i ∈ (univ : Finset (Fin n)),
      p i - q i ≤ p i * (Real.log (p i) - Real.log (q i)) := by
    intro i _
    have hpi := hp i
    have hqi := hq i
    -- log p_i − log q_i = log (p_i / q_i)
    have hlog : Real.log (p i) - Real.log (q i) = Real.log (p i / q i) :=
      (Real.log_div (ne_of_gt hpi) (ne_of_gt hqi)).symm
    rw [hlog]
    -- log(q_i/p_i) ≤ q_i/p_i − 1
    have hx : 0 < q i / p i := div_pos hqi hpi
    have hle : Real.log (q i / p i) ≤ q i / p i - 1 := Real.log_le_sub_one_of_pos hx
    -- log(q_i/p_i) = − log(p_i/q_i)
    have hswap : Real.log (q i / p i) = - Real.log (p i / q i) := by
      rw [← Real.log_inv, inv_div]
    rw [hswap] at hle
    -- ⇒ log(p_i/q_i) ≥ 1 − q_i/p_i
    have hge : 1 - q i / p i ≤ Real.log (p i / q i) := by linarith
    have hmul := mul_le_mul_of_nonneg_left hge (le_of_lt hpi)
    calc p i - q i = p i * (1 - q i / p i) := by field_simp
      _ ≤ p i * Real.log (p i / q i) := hmul
  calc (0 : ℝ) = ∑ i : Fin n, (p i - q i) := hsum0.symm
    _ ≤ ∑ i : Fin n, p i * (Real.log (p i) - Real.log (q i)) := Finset.sum_le_sum hpt
    _ = kl p q := rfl

/-- **`KD-POST-001` (full postcondition).** The soft-target KL term is
    non-negative AND equals the forward-KL form
    `Σ_i p_t_i·(log p_t_i − log p_s_i)` (teacher `p` is the outer
    measure). -/
theorem kl_soft_nonneg_and_forward {n : ℕ} (p q : RVec n)
    (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hsp : ∑ i : Fin n, p i = 1) (hsq : ∑ i : Fin n, q i = 1) :
    0 ≤ kl p q ∧
      kl p q = ∑ i : Fin n, p i * (Real.log (p i) - Real.log (q i)) :=
  ⟨kl_nonneg p q hp hq hsp hsq, rfl⟩

-- Tests
#check @kl_forward_def
#check @kl_self_zero
#check @kl_nonneg
#check @kl_soft_nonneg_and_forward

end ProvableContracts.KLDivergence
