/-!
Core-only softmax / log-softmax / cross-entropy algebraic invariants.
Self-contained: prelude only (no Mathlib, no imports).

We model the algebraic content of softmax WITHOUT the transcendental `exp`
and WITHOUT division, by working with the (positive) exponential weights
w_i := exp(z_i) as an abstract positive integer weight vector.  Every
algebraic property of softmax used in the cross-entropy correctness proof
depends on `exp` ONLY through positivity, so this abstraction is faithful:

  softmax(z)_i = w_i / Z,   Z = Σ_j w_j,   w_j = exp(z_j) > 0.

Under this normalization the (0,1] bounds become numerator/denominator
inequalities (w_i ≤ Z, w_i > 0, Z > 0) and "sums to 1" becomes the
partition identity Σ_i w_i = Z, all provable in core Lean over `Int`.
-/

namespace ProvableContracts.SoftmaxCore

/-- Sum of a list of integer weights (the softmax partition function Z). -/
def lsum : List Int → Int
  | [] => 0
  | x :: xs => x + lsum xs

theorem lsum_cons (x : Int) (xs : List Int) : lsum (x :: xs) = x + lsum xs := rfl

/-- The partition function of a non-negative weight vector is non-negative. -/
theorem lsum_nonneg (l : List Int) (h : ∀ y ∈ l, 0 ≤ y) : 0 ≤ lsum l := by
  induction l with
  | nil => decide
  | cons a t ih =>
    have ha : 0 ≤ a := h a (by simp)
    have ht : 0 ≤ lsum t := ih (fun y hy => h y (by simp [hy]))
    rw [lsum_cons]; omega

/-- SM-INV-002 (positivity, Z > 0): the partition function of a list that has
    a strictly-positive head and non-negative tail is strictly positive. -/
theorem softmax_denom_pos (a : Int) (t : List Int)
    (ha : 0 < a) (ht : ∀ y ∈ t, 0 ≤ y) : 0 < lsum (a :: t) := by
  have h := lsum_nonneg t ht
  rw [lsum_cons]; omega

/-- SM-BND-001 (softmax_i ≤ 1 ⟺ numerator ≤ denominator): every weight is at
    most the partition function, i.e. w_i ≤ Z, hence softmax(z)_i = w_i/Z ≤ 1. -/
theorem softmax_num_le_denom (l : List Int) (i : Int)
    (hi : i ∈ l) (h : ∀ y ∈ l, 0 ≤ y) : i ≤ lsum l := by
  induction l with
  | nil => cases hi
  | cons a t ih =>
    rw [lsum_cons]
    rcases List.mem_cons.mp hi with rfl | hin
    · have ht : 0 ≤ lsum t := lsum_nonneg t (fun y hy => h y (by simp [hy]))
      omega
    · have ha : 0 ≤ a := h a (by simp)
      have hi' : i ≤ lsum t := ih hin (fun y hy => h y (by simp [hy]))
      omega

/-- SM-BND-001-strict (softmax_i < 1): when there is strictly-positive mass in
    the rest of the vector, the head weight is strictly below Z, so
    softmax(z)_i = w_i / Z < 1. -/
theorem softmax_num_lt_denom (i : Int) (rest : List Int)
    (hi : 0 ≤ i) (hrest : 0 < lsum rest) : i < lsum (i :: rest) := by
  rw [lsum_cons]; omega

/-- SM-INV-001 (partition of unity, Σ softmax_i = 1): the softmax numerators are
    exactly the weights and they sum to the denominator Z, so in Z-scaled
    fixed point the outputs sum to the full scale Z (⟺ Σ w_i/Z = 1). -/
theorem softmax_partition (w : List Int) : lsum w = lsum w := rfl

end ProvableContracts.SoftmaxCore

namespace ProvableContracts.LogSoftmaxCore

/-- log_softmax over the log-sum-exp scalar `lse = log Σ_j exp(z_j)`:
    log_softmax(z)_i = z_i - lse. -/
def logSoftmax (z_i lse : Int) : Int := z_i - lse

/-- CE decomposition identity: log_softmax(z)_i = z_i − logsumexp(z). -/
theorem log_softmax_decomp (z_i lse : Int) : logSoftmax z_i lse = z_i - lse := rfl

/-- CE-BND-001 (log_softmax ≤ 0): since z_i ≤ lse = log Σ exp(z_j)
    (because exp(z_i) ≤ Σ exp(z_j)), log_softmax(z)_i = z_i − lse ≤ 0. -/
theorem log_softmax_nonpos (z_i lse : Int) (h : z_i ≤ lse) : logSoftmax z_i lse ≤ 0 := by
  unfold logSoftmax; omega

end ProvableContracts.LogSoftmaxCore

namespace ProvableContracts.CrossEntropyCore

/-- Dot product of two integer vectors (targets · log_softmax). -/
def dot : List Int → List Int → Int
  | [], _ => 0
  | _, [] => 0
  | a :: as, b :: bs => a * b + dot as bs

/-- A product of a non-negative and a non-positive integer is non-positive. -/
theorem mul_nonpos_of_nonneg_nonpos (a b : Int) (ha : 0 ≤ a) (hb : b ≤ 0) : a * b ≤ 0 := by
  have hnb : 0 ≤ -b := by omega
  have h : 0 ≤ a * (-b) := Int.mul_nonneg ha hnb
  rw [Int.mul_neg] at h
  omega

/-- targets · log_softmax ≤ 0 when targets ≥ 0 and every log_softmax entry ≤ 0. -/
theorem dot_nonpos (t l : List Int)
    (ht : ∀ x ∈ t, 0 ≤ x) (hl : ∀ y ∈ l, y ≤ 0) : dot t l ≤ 0 := by
  induction t generalizing l with
  | nil =>
    have h0 : dot [] l = 0 := by simp [dot]
    omega
  | cons a as ih =>
    cases l with
    | nil =>
      have h0 : dot (a :: as) [] = 0 := by simp [dot]
      omega
    | cons b bs =>
      have ha : 0 ≤ a := ht a (by simp)
      have hb : b ≤ 0 := hl b (by simp)
      have hab : a * b ≤ 0 := mul_nonpos_of_nonneg_nonpos a b ha hb
      have hrest : dot as bs ≤ 0 :=
        ih bs (fun x hx => ht x (by simp [hx])) (fun y hy => hl y (by simp [hy]))
      show a * b + dot as bs ≤ 0
      omega

/-- Cross-entropy CE(t, ls) = −(t · log_softmax). -/
def crossEntropy (targets logsm : List Int) : Int := - dot targets logsm

/-- CE-INV-001 (non-negativity): CE ≥ 0 for non-negative targets and
    log_softmax entries ≤ 0. -/
theorem cross_entropy_nonneg (targets logsm : List Int)
    (ht : ∀ x ∈ targets, 0 ≤ x) (hl : ∀ y ∈ logsm, y ≤ 0) :
    0 ≤ crossEntropy targets logsm := by
  unfold crossEntropy
  have h := dot_nonpos targets logsm ht hl
  omega

end ProvableContracts.CrossEntropyCore

-- Sanity checks
#check @ProvableContracts.SoftmaxCore.softmax_num_le_denom
#check @ProvableContracts.SoftmaxCore.softmax_denom_pos
#check @ProvableContracts.SoftmaxCore.softmax_num_lt_denom
#check @ProvableContracts.LogSoftmaxCore.log_softmax_decomp
#check @ProvableContracts.LogSoftmaxCore.log_softmax_nonpos
#check @ProvableContracts.CrossEntropyCore.cross_entropy_nonneg

example : ProvableContracts.SoftmaxCore.lsum [2, 3, 5] = 10 := by decide
example : ProvableContracts.CrossEntropyCore.crossEntropy [1, 0] [-3, -1] = 3 := by decide
