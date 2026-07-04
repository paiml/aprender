/-!
# IEEE 754 Half-Precision (F16) → Single-Precision (F32) — Analytic Invariants

Contract: `f16-conversion-v1`

A *normal* IEEE-754 half-precision value is modelled by its three bit fields:

* sign     `s ∈ {0, 1}`
* exponent `e ∈ [1, 30]`   (biased; f16 bias = 15)
* mantissa `m ∈ [0, 1023]` (10 explicit bits)

The "bias-trick" widening to single precision packs those fields into an F32
bit pattern by rebiasing the exponent (`+112 = 127 − 15`) and zero-padding the
mantissa (`<< 13 = 23 − 10`):

    f32_bits = (s << 31) | ((e + 112) << 23) | (m << 13)

This file proves — over the **entire** normal-f16 domain — the analytic
proof-obligations of the contract:

* `sign_preserved`      — F16-CO-003 (sign preservation)
* `bias_trick_correct`  — F16-CO-001 (bias-trick correctness)
* `roundtrip_identity`  — F16-CO-002 (round-trip identity)

Plus a strengthening lemma used implicitly by the round-trip:

* `mant_padding_zero`   — the low 13 mantissa bits of the widened f32 are 0
* `toF32Bits_monotone`  — the widening is order-preserving on the packed field
                          (the integer/float total-order agreement that makes
                          `x ≤ y → f16→f32 x ≤ f16→f32 y` a Nat identity)

These are **exact `Nat` identities** — no rounding and no reals are needed —
because widening f16 → f32 is *lossless*: F32 has strictly more exponent range
and more mantissa bits, so every normal f16 embeds injectively and the fields
decode back bit-for-bit.

The remaining contract obligations are genuinely runtime / empirical and are
marked `l4_not_applicable` in the contract's `verification_summary`:

* SIMD conversion equivalence — AVX2-lane vs scalar behaviour on real silicon.
* F32→F16 round-to-nearest-even bit-exact parity vs the `half` crate over all
  2³² inputs, including subnormals, overflow-to-Inf and NaN payloads (IEEE
  runtime rounding, not an algebraic identity).
-/

set_option maxRecDepth 4000

namespace ProvableContracts.F16

/-- A normal half-precision value, as its three bit-fields with domain bounds. -/
structure F16Normal where
  s : Nat
  e : Nat
  m : Nat
  hs : s ≤ 1
  he_lo : 1 ≤ e
  he_hi : e ≤ 30
  hm : m ≤ 1023

/-- Bias-trick widening: pack sign (bit 31), rebiased exponent (`e + 112`,
    bits 23–30) and mantissa (`m << 13`, bits 13–22). `2147483648 = 2^31`,
    `8388608 = 2^23`, `8192 = 2^13`. -/
def toF32Bits (h : F16Normal) : Nat :=
  h.s * 2147483648 + (h.e + 112) * 8388608 + h.m * 8192

/-- Decode the F32 sign bit (bit 31). -/
def signBit (bits : Nat) : Nat := bits / 2147483648

/-- Decode the 8-bit F32 exponent field (bits 23–30). -/
def expField (bits : Nat) : Nat := (bits / 8388608) % 256

/-- Decode the f16 mantissa: the upper 10 bits of the 23-bit F32 mantissa. -/
def mantF16 (bits : Nat) : Nat := (bits / 8192) % 1024

/-- Decode the low 13 bits of the F32 mantissa (must be 0 after a bias-trick). -/
def mantLow13 (bits : Nat) : Nat := bits % 8192

/-! ## Field-decode lemmas (each proved by `omega` from the domain bounds). -/

/-- **Sign preservation** (F16-CO-003): the widened f32 sign bit equals the f16
    sign bit. The exponent+mantissa contribution stays below `2^31`, so the
    sign occupies bit 31 exactly. -/
theorem sign_preserved (h : F16Normal) : signBit (toF32Bits h) = h.s := by
  obtain ⟨s, e, m, hs, he_lo, he_hi, hm⟩ := h
  simp only [signBit, toF32Bits]
  omega

/-- The rebiased exponent field decodes to `e + 112`. -/
theorem exp_rebiased (h : F16Normal) : expField (toF32Bits h) = h.e + 112 := by
  obtain ⟨s, e, m, hs, he_lo, he_hi, hm⟩ := h
  simp only [expField, toF32Bits]
  omega

/-- The mantissa decodes back to `m` exactly (10 bits preserved). -/
theorem mant_preserved (h : F16Normal) : mantF16 (toF32Bits h) = h.m := by
  obtain ⟨s, e, m, hs, he_lo, he_hi, hm⟩ := h
  simp only [mantF16, toF32Bits]
  omega

/-- The low 13 mantissa bits of the widened f32 are zero (zero-padding). -/
theorem mant_padding_zero (h : F16Normal) : mantLow13 (toF32Bits h) = 0 := by
  obtain ⟨s, e, m, hs, he_lo, he_hi, hm⟩ := h
  simp only [mantLow13, toF32Bits]
  omega

/-! ## Contract obligations. -/

/-- **Bias-trick correctness** (F16-CO-001): the bit-manipulation widening
    agrees, field-by-field, with the arithmetic definition — the sign is at bit
    31, the exponent is `e + 112`, and the mantissa `m` sits in the top 10 bits
    with a zero-padded 13-bit tail. -/
theorem bias_trick_correct (h : F16Normal) :
    signBit (toF32Bits h) = h.s ∧
    expField (toF32Bits h) = h.e + 112 ∧
    mantF16 (toF32Bits h) = h.m ∧
    mantLow13 (toF32Bits h) = 0 :=
  ⟨sign_preserved h, exp_rebiased h, mant_preserved h, mant_padding_zero h⟩

/-- **Round-trip identity** (F16-CO-002): decoding the widened f32 fields — and
    un-rebiasing the exponent by `−112` — recovers the original `(s, e, m)`
    triple exactly. That is, `f32→f16 ∘ f16→f32 = id` on every normal f16. -/
theorem roundtrip_identity (h : F16Normal) :
    (signBit (toF32Bits h), expField (toF32Bits h) - 112, mantF16 (toF32Bits h))
      = (h.s, h.e, h.m) := by
  have h1 := sign_preserved h
  have h2 := exp_rebiased h
  have h3 := mant_preserved h
  simp only [h1, h2, h3, Nat.add_sub_cancel]

/-- **Order-preserving widening** (monotonicity core): if two same-sign normals
    have `f16` field-packings in order, their widened f32 packings are in the
    same order. The packed field is an affine, strictly-increasing image of the
    f16 field, so the integer order — and hence the float order — is preserved,
    giving `x ≤ y → f16→f32 x ≤ f16→f32 y`. -/
theorem toF32Bits_monotone (a b : F16Normal)
    (hsign : a.s = b.s)
    (hle : a.e * 1024 + a.m ≤ b.e * 1024 + b.m) :
    toF32Bits a ≤ toF32Bits b := by
  obtain ⟨sa, ea, ma, _, _, _, _⟩ := a
  obtain ⟨sb, eb, mb, _, _, _, _⟩ := b
  simp only [toF32Bits] at *
  subst hsign
  omega

#check @sign_preserved
#check @exp_rebiased
#check @mant_preserved
#check @mant_padding_zero
#check @bias_trick_correct
#check @roundtrip_identity
#check @toF32Bits_monotone

end ProvableContracts.F16
