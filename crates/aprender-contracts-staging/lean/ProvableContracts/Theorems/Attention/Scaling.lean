import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.NonNegativity
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.Bounded
import ProvableContracts.Theorems.Softmax.ShiftInvariance
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Attention Scaling — Analytic Obligations of `attention-scaling-v1.yaml`

Scaled dot-product attention normalizes the pre-softmax scores by `1/√d_k`:

    score(Q, K) = Q Kᵀ / √d_k

This file discharges the **analytic** proof obligations of the
`attention-scaling-v1` contract, reusing the sovereign `Softmax` definitions
(`ProvableContracts.Defs.Softmax`) and their theorems.

## Obligations discharged (see `proof_obligations` in the contract)

* **"Score shape correctness"** `shape(Q Kᵀ / √d_k) = [n, m]` —
  `scaledScores_apply` gives the closed-form entry of the scaled score matrix,
  whose Lean type `Matrix (Fin n) (Fin m) ℝ` *is* the `[n, m]` shape witness.
* **"Variance preservation"** `Var(score) ≈ 1 for unit-variance inputs` —
  the exact algebraic variance identity `variance_smul` (`Var(c·x) = c²·Var(x)`)
  and its corollary `variance_scaled_by_inv_sqrt_normalizes`: scaling by `1/√d`
  the pre-scaled score vector (whose variance is `d`, the contract's own
  "without scaling: Var = d_k" premise) yields variance exactly `1`.
* **"Score bound with QK-norm"** `|score_ij| ≤ √d_k` —
  `scaled_score_abs_le_sqrt`, a Cauchy–Schwarz bound: with unit-norm `q, k`
  (QK-norm), `|q·k| ≤ 1`, so `|q·k / √d| ≤ 1/√d ≤ √d` for `d ≥ 1`.
* **"Attention entropy non-negative"** `∀ i, H(attn_i) ≥ 0` —
  `attention_entropy_nonneg`, since each softmax weight lies in `[0, 1]` and
  `negMulLog` is non-negative there.
* **"Attention entropy upper bound"** `∀ i, H(attn_i) ≤ log m` —
  `attention_entropy_le_log_card`, Jensen's inequality against the concave
  `log`: `H(p) = Σ pⱼ log(1/pⱼ) ≤ log(Σ pⱼ / pⱼ) = log m`.
* **"Max-subtraction equivalence"** `softmax(x - max x) = softmax(x)` —
  `softmax_max_subtraction_invariant`, a direct corollary of the softmax
  `shift_invariance`.

## Obligation NOT proved (honest ceiling)

The obligation **"Scaling prevents saturation"**
`H(softmax(QKᵀ/√d_k)) > H(softmax(QKᵀ)) for large d_k` asserts strict
monotonicity of softmax entropy in the temperature `T = √d_k`. This is a genuine
analytic statement (its proof is the derivative identity
`dH/dT = Var_p(energy)/T² > 0`), but it is **not** discharged here; it is left
UNCOVERED (analytic-unproven), which is why the contract honestly remains L3.

## References

- Vaswani et al. "Attention Is All You Need." NeurIPS, 2017. Eq. 1.
- Henry et al. "Query-Key Normalization for Transformers." 2020.
-/

namespace ProvableContracts.AttentionScaling

open Real Finset Matrix
open ProvableContracts
open ProvableContracts.Softmax

/-! ## Score shape correctness -/

/-- The scaled score matrix `score(Q, K) = scale · (Q Kᵀ)`. Its Lean type
    `Matrix (Fin n) (Fin m) ℝ` is exactly the `[n, m]` output shape. -/
noncomputable def scaledScores {n m d : ℕ}
    (Q : Matrix (Fin n) (Fin d) ℝ) (K : Matrix (Fin m) (Fin d) ℝ) (scale : ℝ) :
    Matrix (Fin n) (Fin m) ℝ :=
  scale • (Q * Kᵀ)

/-- **Shape correctness (entry form).** The `(i, j)` entry of the scaled score
    matrix is `scale · Σ_l Q_{il} K_{jl}`. The very typing
    `scaledScores … : Matrix (Fin n) (Fin m) ℝ` certifies the `[n, m]` shape. -/
theorem scaledScores_apply {n m d : ℕ}
    (Q : Matrix (Fin n) (Fin d) ℝ) (K : Matrix (Fin m) (Fin d) ℝ) (scale : ℝ)
    (i : Fin n) (j : Fin m) :
    scaledScores Q K scale i j = scale * ∑ l : Fin d, Q i l * K j l := by
  unfold scaledScores
  rw [Matrix.smul_apply, Matrix.mul_apply]
  simp [Matrix.transpose_apply, smul_eq_mul]

/-! ## Variance preservation (exact algebraic identity) -/

/-- Population mean of a real vector. -/
noncomputable def mean {n : ℕ} (x : RVec n) : ℝ := (∑ i, x i) / (n : ℝ)

/-- Population variance of a real vector. -/
noncomputable def variance {n : ℕ} (x : RVec n) : ℝ :=
  (∑ i, (x i - mean x) ^ 2) / (n : ℝ)

/-- Mean is linear under scalar multiplication: `mean(c·x) = c·mean x`. -/
theorem mean_smul {n : ℕ} (c : ℝ) (x : RVec n) :
    mean (fun i => c * x i) = c * mean x := by
  unfold mean
  rw [← Finset.mul_sum, mul_div_assoc]

/-- **Scaling is quadratic on variance:** `Var(c·x) = c²·Var(x)`. This is the
    algebraic heart of `1/√d_k` normalization. -/
theorem variance_smul {n : ℕ} (c : ℝ) (x : RVec n) :
    variance (fun i => c * x i) = c ^ 2 * variance x := by
  unfold variance
  rw [mean_smul]
  have hterm : ∀ i : Fin n, (c * x i - c * mean x) ^ 2 = c ^ 2 * (x i - mean x) ^ 2 :=
    fun i => by ring
  simp_rw [hterm]
  rw [← Finset.mul_sum, mul_div_assoc]

/-- **Variance normalization.** If the *unscaled* score vector `s` has variance
    `d` (the contract's "without scaling: Var(Q Kᵀ) = d_k" premise), then scaling
    by `1/√d` gives variance exactly `1`:
    `Var(s / √d) = (1/√d)² · d = (1/d) · d = 1`. -/
theorem variance_scaled_by_inv_sqrt_normalizes {n : ℕ} (s : RVec n) (d : ℝ)
    (hd : 0 < d) (hvar : variance s = d) :
    variance (fun i => (1 / Real.sqrt d) * s i) = 1 := by
  rw [variance_smul, hvar, div_pow, one_pow, Real.sq_sqrt hd.le,
    one_div_mul_cancel (ne_of_gt hd)]

/-! ## Score bound with QK-norm (Cauchy–Schwarz) -/

/-- Dot product of two real vectors. -/
noncomputable def dot {d : ℕ} (q k : RVec d) : ℝ := ∑ i, q i * k i

/-- Squared Euclidean norm of a real vector. -/
noncomputable def sqNorm {d : ℕ} (q : RVec d) : ℝ := ∑ i, (q i) ^ 2

/-- **Cauchy–Schwarz:** `(q·k)² ≤ ‖q‖² · ‖k‖²`. -/
theorem dot_sq_le {d : ℕ} (q k : RVec d) : (dot q k) ^ 2 ≤ sqNorm q * sqNorm k := by
  unfold dot sqNorm
  exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ q k

/-- **Score bound with QK-norm.** With unit-norm queries and keys
    (`‖q‖² ≤ 1`, `‖k‖² ≤ 1`, from QK-norm) and `d_k ≥ 1`, the scaled score
    satisfies `|q·k / √d_k| ≤ √d_k`. -/
theorem scaled_score_abs_le_sqrt {d : ℕ} (q k : RVec d) (D : ℝ)
    (hD : 1 ≤ D) (hq : sqNorm q ≤ 1) (hk : sqNorm k ≤ 1) :
    |dot q k / Real.sqrt D| ≤ Real.sqrt D := by
  have hDpos : 0 < D := lt_of_lt_of_le one_pos hD
  have hsqrt_pos : 0 < Real.sqrt D := Real.sqrt_pos.mpr hDpos
  have hnormk : 0 ≤ sqNorm k := Finset.sum_nonneg (fun i _ => sq_nonneg _)
  have hdot_sq : (dot q k) ^ 2 ≤ 1 :=
    calc (dot q k) ^ 2 ≤ sqNorm q * sqNorm k := dot_sq_le q k
      _ ≤ 1 * 1 := mul_le_mul hq hk hnormk (by norm_num)
      _ = 1 := by norm_num
  have habs2 : |dot q k| ^ 2 ≤ 1 := by rw [sq_abs]; exact hdot_sq
  have hdot_abs : |dot q k| ≤ 1 := by nlinarith [habs2, abs_nonneg (dot q k)]
  rw [abs_div, abs_of_pos hsqrt_pos, div_le_iff₀ hsqrt_pos, Real.mul_self_sqrt hDpos.le]
  exact le_trans hdot_abs hD

/-! ## Attention entropy -/

/-- Shannon entropy of a distribution `p`: `H(p) = Σ_j negMulLog(p_j)`. -/
noncomputable def entropy {n : ℕ} (p : RVec n) : ℝ := ∑ i, Real.negMulLog (p i)

/-- Each softmax weight is at most `1` (it is one term of a sum of non-negatives
    equal to `1`). -/
theorem softmax_le_one {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) :
    softmax x i ≤ 1 := by
  have hsum : ∑ j, softmax x j = 1 := partition_of_unity x
  calc softmax x i ≤ ∑ j, softmax x j :=
        Finset.single_le_sum (fun j _ => (softmax_pos x j).le) (Finset.mem_univ i)
    _ = 1 := hsum

/-- **Attention entropy non-negative.** `H(attn_i) ≥ 0`, since each attention
    weight is a softmax value in `[0, 1]` and `negMulLog` is non-negative there. -/
theorem attention_entropy_nonneg {n : ℕ} (scores : RVec (n + 1)) :
    0 ≤ entropy (fun j => softmax scores j) := by
  unfold entropy
  apply Finset.sum_nonneg
  intro j _
  exact Real.negMulLog_nonneg (softmax_pos scores j).le (softmax_le_one scores j)

/-- **Attention entropy upper bound.** `H(attn_i) ≤ log m` for `m = n + 1` keys,
    with equality at the uniform distribution. Proof by Jensen's inequality
    against the concave `log`:
    `H(p) = Σ_j p_j·log(1/p_j) ≤ log(Σ_j p_j·(1/p_j)) = log(Σ_j 1) = log m`. -/
theorem attention_entropy_le_log_card {n : ℕ} (scores : RVec (n + 1)) :
    entropy (fun j => softmax scores j) ≤ Real.log ((n : ℝ) + 1) := by
  set p : Fin (n + 1) → ℝ := fun j => softmax scores j with hp
  have hpos : ∀ j, 0 < p j := fun j => softmax_pos scores j
  have hsum : ∑ j, p j = 1 := partition_of_unity scores
  -- entropy p = Σ p_j · log(1/p_j)
  have hentropy_eq : entropy p = ∑ j, p j * Real.log ((p j)⁻¹) := by
    unfold entropy
    refine Finset.sum_congr rfl (fun j _ => ?_)
    simp only [Real.negMulLog, Real.log_inv]
    ring
  rw [hentropy_eq]
  -- Jensen: Σ p_j • log(x_j) ≤ log(Σ p_j • x_j) with x_j = (p_j)⁻¹.
  have hjensen :=
    (strictConcaveOn_log_Ioi.concaveOn).le_map_sum
      (t := (Finset.univ : Finset (Fin (n + 1)))) (w := p) (p := fun j => (p j)⁻¹)
      (fun j _ => (hpos j).le) (by simpa using hsum)
      (fun j _ => Set.mem_Ioi.mpr (inv_pos.mpr (hpos j)))
  simp only [smul_eq_mul] at hjensen
  refine le_trans hjensen ?_
  have hsimpl : ∑ j : Fin (n + 1), p j * (p j)⁻¹ = ((n : ℝ) + 1) := by
    rw [Finset.sum_congr rfl (fun j _ => mul_inv_cancel₀ (hpos j).ne'),
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
    push_cast; ring
  rw [hsimpl]

/-! ## Max-subtraction equivalence -/

/-- **Max-subtraction equivalence.** Subtracting any constant `M` (in practice
    `M = max x`) from every score leaves the softmax unchanged:
    `softmax(x - M) = softmax(x)`. Direct corollary of `shift_invariance`. -/
theorem softmax_max_subtraction_invariant {n : ℕ} (x : RVec (n + 1)) (M : ℝ)
    (i : Fin (n + 1)) :
    softmax (fun j => x j - M) i = softmax x i := by
  have h := shift_invariance x (-M) i
  simpa only [shift, sub_eq_add_neg] using h

-- Tests
#check @scaledScores_apply
#check @variance_smul
#check @variance_scaled_by_inv_sqrt_normalizes
#check @scaled_score_abs_le_sqrt
#check @attention_entropy_nonneg
#check @attention_entropy_le_log_card
#check @softmax_max_subtraction_invariant

end ProvableContracts.AttentionScaling
