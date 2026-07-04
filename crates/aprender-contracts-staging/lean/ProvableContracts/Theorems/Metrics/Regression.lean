/-!
# Regression Metrics — Core Algebraic Correctness

Sorry-free, **Mathlib-free** proofs for the core-feasible obligations of
`metrics-regression-v1.yaml`.

We model targets `y` and predictions `ŷ` as `List Int`, the residual
`rᵢ = yᵢ - ŷᵢ`, and the metric numerators:

* `sse y ŷ = Σ (yᵢ - ŷᵢ)²`   (equals `n · MSE`; the fixed factor `1/n`, n>0,
                              preserves sign, zero-ness and symmetry)
* `sae y ŷ = Σ |yᵢ - ŷᵢ|`    (equals `n · MAE`)

`RMSE = √MSE` is modelled by its *defining property* (`IsSqrt`): the principal
(non-negative) root of the MSE, which is all that the non-negativity and
perfect-prediction obligations require. No real analysis is used.

The analytic obligations (R² ≤ 1 via Cauchy–Schwarz, MAE ≤ RMSE via QM–AM,
R² = 1 at perfect prediction) genuinely need real-analysis / Mathlib and are
intentionally **not** discharged here.
-/

namespace ProvableContracts.Metrics.Regression

/-- Single-residual absolute value, staying in `Int`. -/
def iabs (a : Int) : Int := if 0 ≤ a then a else -a

/-- Element-wise residuals `yᵢ - ŷᵢ`. Truncates to the shorter list. -/
def residuals : List Int → List Int → List Int
  | y :: ys, yh :: yhs => (y - yh) :: residuals ys yhs
  | _, _ => []

/-- Sum of squared residuals over an explicit residual list. -/
def sseR : List Int → Int
  | []      => 0
  | r :: rs => r * r + sseR rs

/-- Sum of absolute residuals over an explicit residual list. -/
def saeR : List Int → Int
  | []      => 0
  | r :: rs => iabs r + saeR rs

/-- MSE numerator: `Σ (yᵢ - ŷᵢ)²`  (= n · MSE). -/
def sse (y yh : List Int) : Int := sseR (residuals y yh)

/-- MAE numerator: `Σ |yᵢ - ŷᵢ|`  (= n · MAE). -/
def sae (y yh : List Int) : Int := saeR (residuals y yh)

/-- `s` is the principal (non-negative) square root of `x`.
    Models `RMSE = √MSE` using only its defining property. -/
structure IsSqrt (s x : Int) : Prop where
  nonneg : 0 ≤ s
  sq     : s * s = x

/-! ## Elementary lemmas -/

/-- A square is non-negative in `Int` (no `Mathlib`, no `positivity`). -/
theorem int_sq_nonneg (a : Int) : 0 ≤ a * a := by
  rcases Int.le_total 0 a with h | h
  · exact Int.mul_nonneg h h
  · have h2 : 0 ≤ -a := by omega
    have hp : 0 ≤ (-a) * (-a) := Int.mul_nonneg h2 h2
    simpa [Int.neg_mul_neg] using hp

/-- Absolute value is non-negative. -/
theorem iabs_nonneg (a : Int) : 0 ≤ iabs a := by
  unfold iabs; split <;> omega

/-- `iabs 0 = 0`. -/
@[simp] theorem iabs_zero : iabs 0 = 0 := by decide

/-! ## Obligation #2 — MSE non-negativity -/

theorem sseR_nonneg (rs : List Int) : 0 ≤ sseR rs := by
  induction rs with
  | nil => decide
  | cons r rs ih =>
      have := int_sq_nonneg r
      simp only [sseR]; omega

/-- MSE ≥ 0. -/
theorem mse_nonneg (y yh : List Int) : 0 ≤ sse y yh := sseR_nonneg _

/-! ## Obligation #6 — MAE non-negativity -/

theorem saeR_nonneg (rs : List Int) : 0 ≤ saeR rs := by
  induction rs with
  | nil => decide
  | cons r rs ih =>
      have := iabs_nonneg r
      simp only [saeR]; omega

/-- MAE ≥ 0. -/
theorem mae_nonneg (y yh : List Int) : 0 ≤ sae y yh := saeR_nonneg _

/-! ## Obligation #5 — MSE symmetry: MSE(y,ŷ) = MSE(ŷ,y) -/

/-- Swapping arguments negates every residual. -/
theorem residuals_swap (y yh : List Int) :
    residuals yh y = (residuals y yh).map (fun r => -r) := by
  induction y generalizing yh with
  | nil => cases yh <;> rfl
  | cons a ys ih =>
      cases yh with
      | nil => rfl
      | cons b yhs =>
          simp only [residuals, List.map_cons, ih yhs]
          rw [show b - a = -(a - b) from by omega]

/-- `sseR` is invariant under element-wise negation. -/
theorem sseR_map_neg (rs : List Int) :
    sseR (rs.map (fun r => -r)) = sseR rs := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
      simp only [List.map_cons, sseR, ih]
      have : (-r) * (-r) = r * r := Int.neg_mul_neg r r
      rw [this]

/-- MSE symmetry. -/
theorem mse_symm (y yh : List Int) : sse y yh = sse yh y := by
  unfold sse
  rw [residuals_swap y yh, sseR_map_neg]

/-! ## Obligation #4 (feasible part) — perfect prediction ⇒ MSE=MAE=RMSE=0 -/

/-- When `ŷ = y`, every residual is zero. -/
theorem residuals_self (y : List Int) :
    residuals y y = List.replicate y.length 0 := by
  induction y with
  | nil => rfl
  | cons a ys ih =>
      simp only [residuals, List.length_cons, List.replicate, ih]
      rw [show a - a = (0 : Int) from by omega]

theorem sseR_replicate_zero (n : Nat) : sseR (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [List.replicate, sseR, ih]; decide

theorem saeR_replicate_zero (n : Nat) : saeR (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [List.replicate, saeR, iabs_zero, ih]; omega

/-- Perfect prediction ⇒ MSE = 0. -/
theorem mse_self_zero (y : List Int) : sse y y = 0 := by
  unfold sse; rw [residuals_self, sseR_replicate_zero]

/-- Perfect prediction ⇒ MAE = 0. -/
theorem mae_self_zero (y : List Int) : sae y y = 0 := by
  unfold sae; rw [residuals_self, saeR_replicate_zero]

/-- `0` is the principal square root of the perfect-prediction MSE (`= 0`),
    hence RMSE = 0. -/
theorem rmse_self_zero (y : List Int) : IsSqrt 0 (sse y y) :=
  ⟨by omega, by rw [mse_self_zero]; decide⟩

/-- The principal square root of `0` is unique: any `IsSqrt s 0` has `s = 0`. -/
theorem sqrt_zero_unique {s : Int} (h : IsSqrt s 0) : s = 0 := by
  rcases h with ⟨hs, hsq⟩
  rcases (by omega : s = 0 ∨ 0 < s) with h0 | hpos
  · exact h0
  · have hprod : 0 < s * s := Int.mul_pos hpos hpos
    omega

/-! ## Obligation #7 — RMSE non-negativity (definitional) -/

/-- RMSE ≥ 0: the principal root is non-negative by definition. -/
theorem rmse_nonneg {s x : Int} (h : IsSqrt s x) : 0 ≤ s := h.nonneg

-- #check surface
#check @mse_nonneg
#check @mae_nonneg
#check @mse_symm
#check @mse_self_zero
#check @mae_self_zero
#check @rmse_self_zero
#check @rmse_nonneg

end ProvableContracts.Metrics.Regression
