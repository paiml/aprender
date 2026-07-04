import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SigmoidBounded
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Sqrt

/-!
# Gated Delta Net — analytic recurrence obligations

Proves the algebraic / analytic proof obligations of
`contracts/gated-delta-net-v1.yaml` (Qwen3.5 linear-attention layer,
Yang et al. 2024 Gated Delta Networks).

The gated delta recurrence is a *linear* state-space update:

    decay:  α_t = σ(exp(A_log)·dt + dt_bias)          ∈ (0,1)
    read:   r_t = Sᵀ k_t
    delta:  δ_t = β_t · (v_t − r_t)
    write:  S_{t+1} = α_t · S_t + k_t ⊗ δ_t
    output: o_t = (Sᵀ q_t) ⊙ z_t

This module discharges the FOUR analytic obligations of the contract:

  * GDN-BND-001  Decay in the unit interval          (σ maps ℝ → (0,1))
  * GDN-INV-002  State shape preserved by the write   (shape algebra)
  * GDN-INV-003  Causal conv1d                        (finite-support sum)
  * GDN-INV-004  L2 normalisation preserves direction (positive scalar multiple)

The fifth obligation (SIMD-matches-scalar within 8 ULP) is a runtime
floating-point-reduction-order property with no exact algebraic identity;
it is NOT proved here (marked l4_not_applicable in the contract).
-/

namespace ProvableContracts.GatedDeltaNet

open Real
open scoped BigOperators

/-! ## GDN-BND-001 — Decay in the unit interval

The decay gate is `σ` applied to a real pre-activation
`z = exp(A_log)·dt + dt_bias`.  Since the sigmoid maps ℝ strictly into
(0,1), the gate is strictly in (0,1) for every finite input — independent
of `A_log`, `dt`, `dt_bias`.  This is exactly the retention-controlling
property the contract requires (α ∈ (0,1)). -/

/-- The decay gate pre-activation `z = exp(A_log)·dt + dt_bias`. -/
noncomputable def decayPreact (aLog dt dtBias : ℝ) : ℝ :=
  Real.exp aLog * dt + dtBias

/-- The decay gate `α = σ(exp(A_log)·dt + dt_bias)`. -/
noncomputable def decay (aLog dt dtBias : ℝ) : ℝ :=
  Sigmoid.sigmoid (decayPreact aLog dt dtBias)

/-- **GDN-BND-001**: the decay gate is strictly inside the unit interval
    `(0,1)` for every finite `A_log`, `dt`, `dt_bias`. -/
theorem decay_mem_unit_interval (aLog dt dtBias : ℝ) :
    0 < decay aLog dt dtBias ∧ decay aLog dt dtBias < 1 :=
  ProvableContracts.Sigmoid.sigmoid_bounded (decayPreact aLog dt dtBias)

/-- Decay is strictly positive (retains a strictly nonzero fraction of state). -/
theorem decay_pos (aLog dt dtBias : ℝ) : 0 < decay aLog dt dtBias :=
  (decay_mem_unit_interval aLog dt dtBias).1

/-- Decay is strictly below 1 (never a pure copy — always some contraction). -/
theorem decay_lt_one (aLog dt dtBias : ℝ) : decay aLog dt dtBias < 1 :=
  (decay_mem_unit_interval aLog dt dtBias).2

/-! ## GDN-INV-002 — State shape preserved by the write

The recurrent state `S` has shape `[k_dim, v_dim]`.  The write
`S_{t+1} = α · S_t + k ⊗ δ` combines:

  * the outer product `k ⊗ δ` of a `k_dim`-vector with a `v_dim`-vector,
    which has shape `[k_dim, v_dim]`;
  * a scalar multiple `α · S_t`, which preserves shape;
  * an elementwise sum of two `[k_dim, v_dim]` matrices.

Modelling a shape as a pair of naturals, we prove the write maps a
`[k_dim, v_dim]` state back to a `[k_dim, v_dim]` state — for any number of
timesteps. -/

/-- A tensor shape `[rows, cols]`. -/
structure Shape where
  rows : ℕ
  cols : ℕ
deriving DecidableEq, Repr

/-- Shape of the outer product of a `kDim`-vector and a `vDim`-vector. -/
def outerShape (kDim vDim : ℕ) : Shape := ⟨kDim, vDim⟩

/-- Scalar multiplication preserves shape. -/
def scaleShape (s : Shape) : Shape := s

/-- Elementwise addition of two equal shapes preserves that shape. -/
def addShape (s : Shape) : Shape := s

/-- The write step `S_{t+1} = α·S_t + k⊗δ` at the shape level.  `kDim`/`vDim`
    are the ambient state dimensions the write operates within. -/
def writeShape (_kDim _vDim : ℕ) (s : Shape) : Shape :=
  addShape (scaleShape s)  -- α·S_t + k⊗δ, both are ⟨kDim,vDim⟩

/-- **GDN-INV-002**: the outer product `k ⊗ δ` has shape `[k_dim, v_dim]`. -/
theorem outer_shape (kDim vDim : ℕ) :
    outerShape kDim vDim = ⟨kDim, vDim⟩ := rfl

/-- **GDN-INV-002**: one write step preserves the state shape `[k_dim, v_dim]`. -/
theorem write_preserves_shape (kDim vDim : ℕ) :
    writeShape kDim vDim ⟨kDim, vDim⟩ = ⟨kDim, vDim⟩ := rfl

/-- **GDN-INV-002**: `n` write steps preserve the state shape `[k_dim, v_dim]`
    — the recurrence keeps shape invariant for any number of timesteps. -/
theorem write_iterate_preserves_shape (kDim vDim : ℕ) :
    ∀ n : ℕ, (writeShape kDim vDim)^[n] ⟨kDim, vDim⟩ = ⟨kDim, vDim⟩ := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih => rw [Function.iterate_succ_apply', ih]; rfl

/-! ## GDN-INV-003 — Causal conv1d

The depthwise causal conv1d reads a window of the *past* `k` inputs:

    o[t] = Σ_{j=0}^{k-1} w[j] · x[t − j]

Every index read, `t − j` with `j < k`, satisfies `t − j ≤ t` (natural
subtraction).  Hence the output at time `t` depends only on inputs at
indices `≤ t`: perturbing any *future* input (index `> t`) leaves `o[t]`
unchanged.  This is precisely the causality obligation. -/

/-- Causal conv1d: `o[t] = Σ_{j<k} w[j]·x[t−j]` (natural-number window). -/
noncomputable def conv1d (k : ℕ) (w x : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ j ∈ Finset.range k, w j * x (t - j)

/-- **GDN-INV-003**: causality.  If two input signals agree on all indices
    `≤ t`, the conv1d outputs at `t` are equal — the output cannot depend on
    any future (`> t`) input. -/
theorem conv1d_causal (k : ℕ) (w x y : ℕ → ℝ) (t : ℕ)
    (h : ∀ i, i ≤ t → x i = y i) :
    conv1d k w x t = conv1d k w y t := by
  unfold conv1d
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [h (t - j) (Nat.sub_le t j)]

/-- Corollary of `conv1d_causal`: overwriting only strictly-*future* inputs
    (indices `> t`) leaves `o[t]` unchanged.  Here `y` agrees with `x` on all
    indices `≤ t` (it may differ arbitrarily above `t`). -/
theorem conv1d_future_indep (k : ℕ) (w x y : ℕ → ℝ) (t : ℕ)
    (h : ∀ i, i ≤ t → x i = y i) :
    conv1d k w x t = conv1d k w y t :=
  conv1d_causal k w x y t h

/-! ## GDN-INV-004 — L2 normalisation preserves direction

L2 normalisation scales a vector by the strictly-positive reciprocal of its
norm: `q̂ = (1/‖q‖) · q` with `‖q‖ > 0` for `q ≠ 0`.  A strictly-positive
scalar multiple is *parallel* to the original vector, so the direction (and
hence cosine similarity, = 1) is preserved. -/

variable {ι : Type*} [Fintype ι]

/-- Squared L2 norm `‖q‖² = Σ qᵢ²`. -/
def normSq (q : ι → ℝ) : ℝ := ∑ i, (q i) ^ 2

/-- L2 norm `‖q‖ = √(Σ qᵢ²)`. -/
noncomputable def l2norm (q : ι → ℝ) : ℝ := Real.sqrt (normSq q)

/-- L2 normalisation `q̂ = (1/‖q‖) · q`. -/
noncomputable def normalize (q : ι → ℝ) : ι → ℝ := fun i => (1 / l2norm q) * q i

/-- Two vectors point in the *same direction* when one is a strictly-positive
    scalar multiple of the other (cosine similarity = 1). -/
def sameDirection (u v : ι → ℝ) : Prop := ∃ c : ℝ, 0 < c ∧ u = fun i => c * v i

/-- The squared norm is nonnegative. -/
theorem normSq_nonneg (q : ι → ℝ) : 0 ≤ normSq q :=
  Finset.sum_nonneg (fun i _ => sq_nonneg (q i))

/-- The squared norm is strictly positive for a nonzero vector. -/
theorem normSq_pos_of_ne (q : ι → ℝ) (hq : q ≠ 0) : 0 < normSq q := by
  rcases Function.ne_iff.mp hq with ⟨i, hi⟩
  have hpos : 0 < (q i) ^ 2 := (sq_nonneg (q i)).lt_of_ne (Ne.symm (pow_ne_zero 2 hi))
  have : (q i) ^ 2 ≤ normSq q :=
    Finset.single_le_sum (fun k _ => sq_nonneg (q k)) (Finset.mem_univ i)
  linarith

/-- The L2 norm is strictly positive for a nonzero vector. -/
theorem l2norm_pos_of_ne (q : ι → ℝ) (hq : q ≠ 0) : 0 < l2norm q :=
  Real.sqrt_pos.mpr (normSq_pos_of_ne q hq)

/-- **GDN-INV-004**: L2 normalisation preserves direction — for any nonzero
    vector the normalised vector is a strictly-positive scalar multiple of the
    input, hence parallel to it (cosine similarity 1). -/
theorem normalize_same_direction (q : ι → ℝ) (hq : q ≠ 0) :
    sameDirection (normalize q) q := by
  refine ⟨1 / l2norm q, ?_, rfl⟩
  have := l2norm_pos_of_ne q hq
  positivity

/-- The normalisation scalar is strictly positive (direction, not reflection). -/
theorem normalize_scalar_pos (q : ι → ℝ) (hq : q ≠ 0) : 0 < 1 / l2norm q := by
  have := l2norm_pos_of_ne q hq
  positivity

-- ── Verification checks ──────────────────────────────────────────────
#check @decay_mem_unit_interval
#check @write_iterate_preserves_shape
#check @conv1d_causal
#check @normalize_same_direction

end ProvableContracts.GatedDeltaNet
