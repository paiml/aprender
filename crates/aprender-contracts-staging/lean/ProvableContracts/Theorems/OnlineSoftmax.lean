import ProvableContracts.Defs.Softmax
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Exponential

/-!
# Online Softmax — Analytic Correctness Theorems

Verified analytic obligations for `contracts/online-softmax-v1.yaml`
(Milakov & Gimelshein 2018, "Online normalizer calculation for softmax";
Rabe & Staats 2022 flash-attention core, PILLAR-2/4).

The online / streaming normalizer scans the score vector once, maintaining a
running pair `(mᵢ, dᵢ)` where

  mᵢ = max(x₁,…,xᵢ),   dᵢ = Σ_{j≤i} exp(xⱼ − mᵢ),

updated by the rescale recurrence

  mᵢ = max(m_{i-1}, xᵢ),   dᵢ = d_{i-1}·exp(m_{i-1} − mᵢ) + exp(xᵢ − mᵢ).

The final output is `softmax(x)ⱼ = exp(xⱼ − mₙ) / dₙ`.

## What is proved here (ALL over ℝ — exact, not ε-approximate)

* `denom_rescale`         — the exact rescale identity Σexp(·−b) = Σexp(·−a)·exp(a−b).
* `online_update_denom`   — OBLIGATION `old_state`: one recurrence step equals full
                            recomputation of the shifted partial sum.
* `foldl_step_snd`        — OBLIGATION `loop_invariant` (partial sum): the running
                            denominator after the whole scan equals Σ exp(xⱼ − mₙ).
* `foldl_max_ge` / `foldl_max_mem` — OBLIGATION `loop_invariant` (running max):
                            the running maximum dominates the start and every element.
* `loop_variant_decreases` — OBLIGATION `loop_variant`: V(i)=n−i is ≥0 and strictly
                            decreasing, so the scan terminates.
* `softmax_sub_const`     — max-subtraction / shift is EXACT over ℝ: subtracting any
                            constant m before exponentiating leaves softmax unchanged.
* `online_eq_standard`    — HEADLINE, OBLIGATION `equivalence`: the online output
                            equals the standard (max-subtraction) softmax EXACTLY.
* `softmax_sum_one`       — OBLIGATION `invariant`: outputs sum to 1.
* `softmax_pos`           — OBLIGATION `invariant`: every output is strictly positive.
* `softmax_strict_mono`   — OBLIGATION `monotonicity`: order preservation.
* `softmax_shift_invariant` — OBLIGATION `invariant`: softmax(x+c)=softmax(x).

The remaining obligation `Two-pass (not three)` is a memory-access / source-structure
claim (reads the array exactly twice) with no algebraic content over ℝ — it is a
genuine (b)-class runtime property, marked `l4_not_applicable` in the contract.
-/

namespace ProvableContracts.OnlineSoftmax

open Real Finset ProvableContracts ProvableContracts.Softmax

/-! ## Streaming denominator model (List ℝ) -/

/-- The shifted partial denominator `Σ_{x ∈ l} exp(x − M)`. This is exactly the
`dᵢ` the online scan maintains, with `M` the running maximum. -/
noncomputable def denom (l : List ℝ) (M : ℝ) : ℝ :=
  (l.map (fun x => Real.exp (x - M))).sum

@[simp] theorem denom_nil (M : ℝ) : denom [] M = 0 := rfl

@[simp] theorem denom_cons (x : ℝ) (xs : List ℝ) (M : ℝ) :
    denom (x :: xs) M = Real.exp (x - M) + denom xs M := rfl

/-- **Exact rescale identity.** Shifting the reference point from `a` to `b`
multiplies every shifted denominator by the single scalar `exp(a − b)`:
`Σ exp(x − b) = (Σ exp(x − a))·exp(a − b)`. This is the algebraic heart of the
online normalizer — it is what makes a running scalar `dᵢ` sufficient. -/
theorem denom_rescale (l : List ℝ) (a b : ℝ) :
    denom l b = denom l a * Real.exp (a - b) := by
  induction l with
  | nil => simp
  | cons y ys ih =>
    simp only [denom_cons, ih, add_mul]
    congr 1
    rw [← Real.exp_add]
    congr 1
    ring

/-- **OBLIGATION `old_state`.** One online-normalizer update reproduces the full
recomputation of the shifted partial sum: appending `xᵢ` and moving the reference
from the old running max `mold` to the new one `mnew` satisfies

  denom (l ++ [xᵢ]) mnew = denom l mold · exp(mold − mnew) + exp(xᵢ − mnew).

Holds for ALL `mold, mnew` (the identity does not even need `mnew = max mold xᵢ`),
so the rescaling factor `exp(m_{i-1} − mᵢ)` is exactly correct. -/
theorem online_update_denom (l : List ℝ) (xi mold mnew : ℝ) :
    denom (l ++ [xi]) mnew
      = denom l mold * Real.exp (mold - mnew) + Real.exp (xi - mnew) := by
  unfold denom
  rw [List.map_append, List.sum_append]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  have := denom_rescale l mold mnew
  unfold denom at this
  rw [this]

/-! ## The scan fold -/

/-- One online step: update the running `(max, denom)` pair on a new score `x`. -/
noncomputable def step (s : ℝ × ℝ) (x : ℝ) : ℝ × ℝ :=
  let mnew := max s.1 x
  (mnew, s.2 * Real.exp (s.1 - mnew) + Real.exp (x - mnew))

/-- The running-max component of the fold is exactly `List.foldl max`. -/
theorem step_fst (l : List ℝ) : ∀ s : ℝ × ℝ,
    (l.foldl step s).1 = l.foldl max s.1 := by
  induction l with
  | nil => intro s; rfl
  | cons x xs ih => intro s; simp only [List.foldl_cons]; rw [ih (step s x)]; rfl

/-- **OBLIGATION `loop_invariant` (partial sum), generalized.** Folding `step` over
`l` from any starting state `s = (m₀, d₀)` yields a running denominator equal to

  d₀·exp(m₀ − M) + denom l M,   where M = foldl max m₀ l

is the final running maximum. In particular the `denom l M` term is precisely the
correct shifted partial sum `Σ_{x∈l} exp(x − M)`. -/
theorem foldl_step_snd (l : List ℝ) : ∀ s : ℝ × ℝ,
    (l.foldl step s).2
      = s.2 * Real.exp (s.1 - l.foldl max s.1) + denom l (l.foldl max s.1) := by
  induction l with
  | nil =>
    intro s
    simp only [List.foldl_nil, denom_nil, add_zero, sub_self, Real.exp_zero, mul_one]
  | cons x xs ih =>
    intro s
    simp only [List.foldl_cons]
    set M : ℝ := xs.foldl max (step s x).1 with hM
    rw [ih (step s x)]
    show (s.2 * Real.exp (s.1 - max s.1 x) + Real.exp (x - max s.1 x))
          * Real.exp (max s.1 x - M) + denom xs M
        = s.2 * Real.exp (s.1 - M) + denom (x :: xs) M
    rw [denom_cons]
    have e1 : Real.exp (s.1 - max s.1 x) * Real.exp (max s.1 x - M)
                = Real.exp (s.1 - M) := by
      rw [← Real.exp_add]; congr 1; ring
    have e2 : Real.exp (x - max s.1 x) * Real.exp (max s.1 x - M)
                = Real.exp (x - M) := by
      rw [← Real.exp_add]; congr 1; ring
    calc
      (s.2 * Real.exp (s.1 - max s.1 x) + Real.exp (x - max s.1 x))
          * Real.exp (max s.1 x - M) + denom xs M
        = s.2 * (Real.exp (s.1 - max s.1 x) * Real.exp (max s.1 x - M))
            + Real.exp (x - max s.1 x) * Real.exp (max s.1 x - M) + denom xs M := by
              ring
      _ = s.2 * Real.exp (s.1 - M) + Real.exp (x - M) + denom xs M := by
              rw [e1, e2]
      _ = s.2 * Real.exp (s.1 - M) + (Real.exp (x - M) + denom xs M) := by ring

/-- **Full-scan denominator.** Starting the scan on a non-empty vector `x₀ :: xs`
from state `(x₀, 1)` (i.e. `m = x₀`, `d = exp(x₀−x₀) = 1`) computes exactly the
standard shifted denominator `Σ_{v ∈ x₀::xs} exp(v − M)` at the true maximum
`M = foldl max x₀ xs`. -/
theorem scan_denom (x0 : ℝ) (xs : List ℝ) :
    (xs.foldl step (x0, 1)).2 = denom (x0 :: xs) (xs.foldl max x0) := by
  have h := foldl_step_snd xs (x0, 1)
  simp only at h
  rw [h, denom_cons, one_mul]

/-- The running-max state after the non-empty scan is the true list maximum. -/
theorem scan_max (x0 : ℝ) (xs : List ℝ) :
    (xs.foldl step (x0, 1)).1 = xs.foldl max x0 := by
  have := step_fst xs (x0, 1); simpa using this

/-! ## Running-max loop invariant -/

/-- The running maximum dominates the starting accumulator. -/
theorem foldl_max_ge (l : List ℝ) : ∀ m0 : ℝ, m0 ≤ l.foldl max m0 := by
  induction l with
  | nil => intro m0; exact le_refl _
  | cons x xs ih =>
    intro m0
    simp only [List.foldl_cons]
    exact le_trans (le_max_left m0 x) (ih (max m0 x))

/-- **OBLIGATION `loop_invariant` (running max).** After processing the whole list
the running maximum is ≥ every element seen, i.e. `mᵢ ≥ xₖ` for all `k ≤ i`. Together
with `foldl_max_ge` this establishes `mᵢ = max(x₁,…,xᵢ)`. -/
theorem foldl_max_mem (l : List ℝ) :
    ∀ (m0 : ℝ), ∀ x ∈ l, x ≤ l.foldl max m0 := by
  induction l with
  | nil => intro m0 x hx; cases hx
  | cons y ys ih =>
    intro m0 x hx
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp hx with rfl | hin
    · exact le_trans (le_max_right m0 x) (foldl_max_ge ys (max m0 x))
    · exact ih (max m0 y) x hin

/-! ## Loop variant (termination) -/

/-- **OBLIGATION `loop_variant`.** The variant `V(i) = n − i` is bounded below by 0
and strictly decreases on every iteration while `i < n`; hence the scan terminates
in exactly `n` steps. Pure elementary arithmetic. -/
theorem loop_variant_decreases (n i : ℕ) (h : i < n) :
    0 ≤ n - (i + 1) ∧ n - (i + 1) < n - i := by
  constructor
  · exact Nat.zero_le _
  · omega

/-! ## Shift / max-subtraction exactness and the standard-softmax identity -/

/-- **Max-subtraction is EXACT over ℝ.** For ANY reference `m`, the shifted-softmax
expression `exp(xᵢ − m) / Σⱼ exp(xⱼ − m)` equals `softmax(x)ᵢ = exp(xᵢ)/Σⱼ exp(xⱼ)`.
Taking `m = max x` gives the numerically-safe standard softmax; taking `m = −c`
gives shift invariance. The whole family of "subtract a constant" tricks is one
lemma. -/
theorem softmax_sub_const {n : ℕ} (x : RVec n) (m : ℝ) (i : Fin n) :
    Real.exp (x i - m) / (∑ j : Fin n, Real.exp (x j - m)) = softmax x i := by
  unfold softmax
  have hj : ∀ j : Fin n, Real.exp (x j - m) = Real.exp (x j) * Real.exp (-m) := by
    intro j; rw [sub_eq_add_neg, Real.exp_add]
  simp only [hj]
  rw [← Finset.sum_mul]
  rw [mul_div_mul_right _ _ (Real.exp_ne_zero (-m))]

/-- **HEADLINE — OBLIGATION `equivalence`.** The online output and the standard
(max-subtraction) output are EXACTLY equal element-wise over ℝ (not merely within
`ε`): both equal `softmax(x)ᵢ`. `mOnline` is the scan's running max and `mStd` the
independently-computed max; the theorem holds for any two reference points since
`softmax_sub_const` collapses both to the unshifted softmax. -/
theorem online_eq_standard {n : ℕ} (x : RVec n) (mOnline mStd : ℝ) (i : Fin n) :
    Real.exp (x i - mOnline) / (∑ j : Fin n, Real.exp (x j - mOnline))
      = Real.exp (x i - mStd) / (∑ j : Fin n, Real.exp (x j - mStd)) := by
  rw [softmax_sub_const x mOnline i, softmax_sub_const x mStd i]

/-- **OBLIGATION `invariant` (shift invariance).** `softmax(x + c) = softmax(x)`
for any scalar `c`. Immediate corollary of `softmax_sub_const` with `m = −c`. -/
theorem softmax_shift_invariant {n : ℕ} (x : RVec n) (c : ℝ) (i : Fin n) :
    softmax (fun j => x j + c) i = softmax x i := by
  have h := softmax_sub_const x (-c) i
  simpa [softmax, sub_neg_eq_add] using h

/-! ## Distribution invariants (sum-to-one, positivity, monotonicity) -/

variable {n : ℕ} [NeZero n]

/-- The softmax denominator `Σⱼ exp(xⱼ)` is strictly positive on a non-empty index. -/
theorem softmax_denom_pos (x : RVec n) : 0 < ∑ j : Fin n, Real.exp (x j) := by
  apply Finset.sum_pos
  · intro j _; exact Real.exp_pos _
  · exact ⟨(Classical.arbitrary (Fin n)), Finset.mem_univ _⟩

/-- **OBLIGATION `invariant` (sum-to-one).** `Σᵢ softmax(x)ᵢ = 1`. -/
theorem softmax_sum_one (x : RVec n) : (∑ i : Fin n, softmax x i) = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (softmax_denom_pos x))

/-- **OBLIGATION `invariant` (positivity).** Every softmax output is strictly
positive: `softmax(x)ᵢ > 0`. -/
theorem softmax_pos (x : RVec n) (i : Fin n) : 0 < softmax x i := by
  unfold softmax
  exact div_pos (Real.exp_pos _) (softmax_denom_pos x)

/-- **OBLIGATION `monotonicity` (order preservation).** Strictly larger logits map
to strictly larger softmax probabilities: `xᵢ > xⱼ ⟹ softmax(x)ᵢ > softmax(x)ⱼ`. -/
theorem softmax_strict_mono (x : RVec n) (i j : Fin n) (h : x j < x i) :
    softmax x j < softmax x i := by
  unfold softmax
  have hZ : 0 < ∑ k : Fin n, Real.exp (x k) := softmax_denom_pos x
  rw [div_lt_div_iff_of_pos_right hZ]
  exact Real.exp_lt_exp.mpr h

/-! ## Sanity checks -/

#check @denom_rescale
#check @online_update_denom
#check @foldl_step_snd
#check @scan_denom
#check @foldl_max_mem
#check @loop_variant_decreases
#check @softmax_sub_const
#check @online_eq_standard
#check @softmax_shift_invariant
#check @softmax_sum_one
#check @softmax_pos
#check @softmax_strict_mono

end ProvableContracts.OnlineSoftmax
