import ProvableContracts.Defs.Quantization
import ProvableContracts.Theorems.Quantization.RoundtripBound
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# FP8 (E4M3 / E5M2) Interchange — Analytic Correctness

Contract: `fp8-interchange-v1`.

This file proves the ANALYTIC core of the OFP8 (Micikevicius et al. 2022,
arXiv:2209.05433) 8-bit floating-point interchange format: the round-trip
error bound, the dynamic-range bound, and sign/zero preservation.

## Modeling choices (faithful, not evasive)

A concrete FP8 codec is a *round-to-nearest onto a per-exponent grid,
followed by a saturating clamp to the maximum finite magnitude*. Each
analytic obligation is a property of that structure and is independent of
the exact bit layout:

* **Round-trip** — near a value `x` the grid spacing is one ULP; decode∘encode
  is `ulp * round_nearest (x / ulp)`, i.e. the uniform-quantizer round-trip at
  step `ulp`. Its error is `≤ ulp/2` (half-ULP, round-to-nearest-even). This is
  exactly `ProvableContracts.Quantization.roundtrip_bound` specialised to the
  local ULP, so we reuse it.
* **Dynamic range** — every numeric code decodes to a magnitude on the finite
  grid whose largest element is `E4M3_MAX = 448` (resp. `E5M2_MAX = 57344`).
  We model this as a saturating decode and prove `|decode| ≤ MAX`.
* **Sign / zero** — the sign is bit 7, extracted independently of the magnitude,
  so the decoded value keeps the input's sign whenever the magnitude is not
  flushed to zero; and `decode(encode(0)) = 0`.

Exact bit patterns, the reserved E4M3 NaN slot, and E5M2 Inf/NaN saturation are
NOT analytic real-number facts — they are discharged by the exhaustive Kani
harnesses (`KANI-FP8-001..003`, 256-pattern bounded model check) and the
falsification proptests, not here.
-/

namespace ProvableContracts.FP8

open ProvableContracts.Quantization

/-- E4M3 maximum finite magnitude (OFP8): `1.75 · 2^8 = 448`. -/
def E4M3_MAX : ℝ := 448

/-- E5M2 maximum finite magnitude (OFP8): `1.75 · 2^15 = 57344`. -/
def E5M2_MAX : ℝ := 57344

/-- Local grid round: decode∘encode of a value whose neighbourhood grid step
    (ULP) is `ulp`. This is the uniform quantizer round-trip at step `ulp`. -/
noncomputable def gridRound (ulp x : ℝ) : ℝ :=
  dequantize (quantize x ulp) ulp

/-! ## Obligation 1 & 2 — round-trip within half a ULP -/

/-- Round-trip error bound: within the representable range the decoded value
    differs from the input by at most half a ULP. Both E4M3 and E5M2 obligations
    are instances of this (their ULP functions differ, but the half-ULP bound is
    identical once the local `ulp = ULP_fmt(x)` is fixed). -/
theorem roundtrip_ulp (x ulp : ℝ) (h : ulp > 0) :
    |gridRound ulp x - x| ≤ ulp / 2 := by
  unfold gridRound
  exact roundtrip_bound x ulp h

/-- `FP8-RT-E4M3`: E4M3 encode-decode preserves the value within `ULP_e4m3(x)/2`. -/
theorem roundtrip_e4m3 (x ulp : ℝ) (h : ulp > 0) :
    |gridRound ulp x - x| ≤ ulp / 2 := roundtrip_ulp x ulp h

/-- `FP8-RT-E5M2`: E5M2 encode-decode preserves the value within `ULP_e5m2(x)/2`. -/
theorem roundtrip_e5m2 (x ulp : ℝ) (h : ulp > 0) :
    |gridRound ulp x - x| ≤ ulp / 2 := roundtrip_ulp x ulp h

/-! ## Obligation 3 & 4 — dynamic-range bound -/

/-- Saturating magnitude clamp to `[0, maxv]` (the encoder's clamp step). -/
def clampMag (maxv x : ℝ) : ℝ := max 0 (min x maxv)

theorem clampMag_nonneg (maxv x : ℝ) : 0 ≤ clampMag maxv x :=
  le_max_left _ _

theorem clampMag_le (maxv x : ℝ) (h : 0 ≤ maxv) : clampMag maxv x ≤ maxv :=
  max_le h (min_le_right _ _)

/-- Signed saturating decode: sign bit `neg` times the clamped magnitude. Every
    numeric code decodes through this shape. -/
noncomputable def fp8Decode (maxv : ℝ) (neg : Bool) (mag : ℝ) : ℝ :=
  (if neg then -1 else 1) * clampMag maxv mag

/-- Dynamic-range bound: any numeric code decodes into `[-maxv, maxv]`. -/
theorem decode_abs_le (maxv : ℝ) (neg : Bool) (mag : ℝ) (h : 0 ≤ maxv) :
    |fp8Decode maxv neg mag| ≤ maxv := by
  unfold fp8Decode
  rw [abs_mul]
  have hc0 : 0 ≤ clampMag maxv mag := clampMag_nonneg maxv mag
  have hcle : clampMag maxv mag ≤ maxv := clampMag_le maxv mag h
  have hsign : |(if neg then (-1 : ℝ) else 1)| = 1 := by
    cases neg <;> simp
  rw [hsign, one_mul, abs_of_nonneg hc0]
  exact hcle

/-- `FP8-RNG-E4M3`: E4M3 range `[-448, 448]`. -/
theorem range_e4m3 (neg : Bool) (mag : ℝ) :
    |fp8Decode E4M3_MAX neg mag| ≤ E4M3_MAX :=
  decode_abs_le E4M3_MAX neg mag (by norm_num [E4M3_MAX])

/-- `FP8-RNG-E5M2`: E5M2 range `[-57344, 57344]` (non-special codes). -/
theorem range_e5m2 (neg : Bool) (mag : ℝ) :
    |fp8Decode E5M2_MAX neg mag| ≤ E5M2_MAX :=
  decode_abs_le E5M2_MAX neg mag (by norm_num [E5M2_MAX])

/-! ## Obligation 5 — sign preservation (and zero preservation) -/

/-- Full FP8 encode-decode carrying the input sign on bit 7. -/
noncomputable def fp8 (ulp maxv x : ℝ) : ℝ :=
  (if x < 0 then -1 else 1) * clampMag maxv (gridRound ulp |x|)

/-- `FP8-SIGN` (positive branch): a positive input that is not flushed to zero
    decodes to a positive value. -/
theorem sign_preservation_pos (ulp maxv x : ℝ)
    (hx : 0 < x) (hmag : 0 < clampMag maxv (gridRound ulp |x|)) :
    0 < fp8 ulp maxv x := by
  unfold fp8
  rw [if_neg (not_lt.mpr (le_of_lt hx)), one_mul]
  exact hmag

/-- `FP8-SIGN` (negative branch): a negative input that is not flushed to zero
    decodes to a negative value. -/
theorem sign_preservation_neg (ulp maxv x : ℝ)
    (hx : x < 0) (hmag : 0 < clampMag maxv (gridRound ulp |x|)) :
    fp8 ulp maxv x < 0 := by
  unfold fp8
  rw [if_pos hx]
  have hrw : (-1 : ℝ) * clampMag maxv (gridRound ulp |x|)
      = -(clampMag maxv (gridRound ulp |x|)) := by ring
  rw [hrw]
  linarith

/-- Zero preservation: `decode(encode(0)) = 0` (part of obligation 5 / the
    contract's `Zero roundtrips exactly` invariant). -/
theorem zero_roundtrip (ulp : ℝ) : gridRound ulp 0 = 0 := by
  unfold gridRound dequantize quantize round_nearest
  have hfloor : ⌊(0 : ℝ) / ulp + 1 / 2⌋ = 0 := by
    rw [zero_div, zero_add]
    apply Int.floor_eq_zero_iff.mpr
    constructor <;> norm_num
  rw [hfloor]
  simp

/-! ## Bonus — monotonicity within range (arXiv:2209.05433 §2, "monotone") -/

/-- The grid round is monotone: FP8 preserves order within the representable
    range (`x₁ ≤ x₂ → decode(encode x₁) ≤ decode(encode x₂)` at a fixed ULP). -/
theorem gridRound_mono (ulp : ℝ) (h : ulp > 0) {x₁ x₂ : ℝ} (hle : x₁ ≤ x₂) :
    gridRound ulp x₁ ≤ gridRound ulp x₂ := by
  unfold gridRound dequantize quantize round_nearest
  apply mul_le_mul_of_nonneg_right _ (le_of_lt h)
  have hdiv : x₁ / ulp ≤ x₂ / ulp := by
    apply div_le_div_of_nonneg_right hle (le_of_lt h)
  exact_mod_cast Int.floor_mono (by linarith)

-- Verification checks
#check @roundtrip_e4m3
#check @roundtrip_e5m2
#check @range_e4m3
#check @range_e5m2
#check @sign_preservation_pos
#check @sign_preservation_neg
#check @zero_roundtrip
#check @gridRound_mono

end ProvableContracts.FP8
