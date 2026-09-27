import Mathlib.Analysis.Normed.Field.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# F16 input rounding — the error of a dot product over F16-rounded inputs

Contract: `fp16-cublas-gemm-v1`, equation `f16_input_rounding_bound`.

Moved here from `MatMul/CooperativeTiling.lean` (#4502): that module's only binding was the dead
WGSL cooperative-matrix shader, deleted with it. The lemmas are about the rounding model, not that
kernel, so they now back the live FP16 GEMM contract. Over ℝ under a rounding-model hypothesis:
a statement about the math the kernel approximates (L3), not a refinement of an f32 Rust item.
-/

namespace ProvableContracts.F16Rounding

/-- The standard rounding model: `fl` rounds every real with relative error at most `u`,
    `|fl x - x| ≤ u·|x|`. For F16 round-to-nearest, `u = 2⁻¹¹` (half the machine epsilon 2⁻¹⁰).
    A hypothesis of the theorems below, never an axiom: no model of IEEE rounding is assumed true. -/
def RoundingModel (fl : ℝ → ℝ) (u : ℝ) : Prop :=
  ∀ x : ℝ, |fl x - x| ≤ u * |x|

/-- The hypothesis is satisfiable (exact arithmetic meets it with `u = 0`), so the theorems below
    are not vacuous. -/
theorem roundingModel_id : RoundingModel id 0 := by
  intro x; simp

/-- One product of rounded inputs: `|fl a·fl b - a·b| ≤ (2u + u²)·|a|·|b|`. -/
theorem rounded_product_error {fl : ℝ → ℝ} {u : ℝ} (hu : 0 ≤ u) (hfl : RoundingModel fl u)
    (a b : ℝ) : |fl a * fl b - a * b| ≤ (2 * u + u ^ 2) * (|a| * |b|) := by
  have ha := hfl a
  have hb := hfl b
  have hflb : |fl b| ≤ (1 + u) * |b| := by
    have := abs_sub_abs_le_abs_sub (fl b) b
    nlinarith [abs_nonneg b]
  have split : fl a * fl b - a * b = (fl a - a) * fl b + a * (fl b - b) := by ring
  rw [split]
  calc |(fl a - a) * fl b + a * (fl b - b)|
      ≤ |fl a - a| * |fl b| + |a| * |fl b - b| := by
        simpa [abs_mul] using abs_add_le ((fl a - a) * fl b) (a * (fl b - b))
    _ ≤ (u * |a|) * ((1 + u) * |b|) + |a| * (u * |b|) := by
        gcongr
    _ = (2 * u + u ^ 2) * (|a| * |b|) := by ring

/-- The F16-input part of an FP16 GEMM: a length-`k` dot product of F16-rounded inputs, accumulated
    exactly, is within `k·(2u + u²)·maxA·maxB` of the exact one. With `u = 2⁻¹¹` the factor is
    `2⁻¹⁰ + 2⁻²²`. The F32 accumulation rounding is not modelled here.

    This replaces the axiom `f16_accumulation_error_bound`, which asserted only that the number
    `k·2⁻¹⁰·maxA·maxB` exists and is nonnegative -- a statement about no rounding at all (#4347). -/
theorem f16_input_rounding_error_bound {k : ℕ} {fl : ℝ → ℝ} {u : ℝ}
    (hu : 0 ≤ u) (hfl : RoundingModel fl u)
    (a b : Fin k → ℝ) (maxA maxB : ℝ)
    (hA : ∀ i, |a i| ≤ maxA) (hB : ∀ i, |b i| ≤ maxB) :
    |∑ i, fl (a i) * fl (b i) - ∑ i, a i * b i| ≤ k * (2 * u + u ^ 2) * maxA * maxB := by
  have hc : 0 ≤ 2 * u + u ^ 2 := by positivity
  rw [← Finset.sum_sub_distrib]
  calc |∑ i, (fl (a i) * fl (b i) - a i * b i)|
      ≤ ∑ i, |fl (a i) * fl (b i) - a i * b i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin k, (2 * u + u ^ 2) * (maxA * maxB) := by
        apply Finset.sum_le_sum
        intro i _
        calc |fl (a i) * fl (b i) - a i * b i|
            ≤ (2 * u + u ^ 2) * (|a i| * |b i|) := rounded_product_error hu hfl _ _
          _ ≤ (2 * u + u ^ 2) * (maxA * maxB) := by
              gcongr
              · exact le_trans (abs_nonneg _) (hA i)
              · exact hA i
              · exact hB i
    _ = k * (2 * u + u ^ 2) * maxA * maxB := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

#check @f16_input_rounding_error_bound

end ProvableContracts.F16Rounding
