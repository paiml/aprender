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

/-! ## Refinement: simulation of `trueno::activations::f16_to_f32` (normal branch)

The theorems above are about the *field model* `F16Normal`. This section ties
that model to the Rust function in `crates/aprender-compute/src/activations.rs`
(re-exported as `trueno::activations::f16_to_f32`), whose normal branch is

```rust
let sign = (bits >> 15) & 0x1;
let exponent = (bits >> 10) & 0x1F;
let mantissa = bits & 0x3FF;
if exponent != 0 && exponent != 31 {
    let f32_exp = (exponent as u32 + 112) as u32;
    let f32_mant = (mantissa as u32) << 13;
    let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mant;
    return f32::from_bits(f32_bits);
}
```

`rustSign`/`rustExp`/`rustMant`/`rustNormalBits` transcribe those lines
operator-for-operator (`>>` ↦ `>>>`, `&` ↦ `&&&`, `<<` ↦ `<<<`, `|` ↦ `|||`,
same association) over `Nat`, with the input a `u16` (`bits < 2^16`).
`rustNormalBits_lt_u32` proves no intermediate exceeds `u32`, so `Nat`
arithmetic agrees with the Rust `u16`/`u32` arithmetic (no truncation), and
`f16_to_f32_normal_simulates` proves the Rust bits equal `toF32Bits` of the
decoded `F16Normal`. Every theorem above therefore holds of the Rust output.

**Scope: normal inputs only** (`exponent ∉ {0, 31}`). The subnormal, ±0,
±inf and NaN branches that follow in the Rust are NOT modelled here. -/

/-- Rust: `(bits >> 15) & 0x1`. -/
def rustSign (bits : Nat) : Nat := (bits >>> 15) &&& 0x1

/-- Rust: `(bits >> 10) & 0x1F`. -/
def rustExp (bits : Nat) : Nat := (bits >>> 10) &&& 0x1F

/-- Rust: `bits & 0x3FF`. -/
def rustMant (bits : Nat) : Nat := bits &&& 0x3FF

/-- Rust: the branch guard `exponent != 0 && exponent != 31`. -/
def RustNormal (bits : Nat) : Prop := rustExp bits ≠ 0 ∧ rustExp bits ≠ 31

/-- Rust: `((sign as u32) << 31) | (f32_exp << 23) | f32_mant`, with
    `f32_exp = exponent + 112` and `f32_mant = mantissa << 13`. -/
def rustNormalBits (bits : Nat) : Nat :=
  (rustSign bits <<< 31 ||| (rustExp bits + 112) <<< 23) ||| rustMant bits <<< 13

theorem rustSign_eq (b : Nat) : rustSign b = b / 32768 % 2 := by
  have h := Nat.and_two_pow_sub_one_eq_mod (b >>> 15) 1
  simp only [Nat.reducePow, Nat.reduceSub] at h
  unfold rustSign
  rw [h, Nat.shiftRight_eq_div_pow]

theorem rustExp_eq (b : Nat) : rustExp b = b / 1024 % 32 := by
  have h := Nat.and_two_pow_sub_one_eq_mod (b >>> 10) 5
  simp only [Nat.reducePow, Nat.reduceSub] at h
  unfold rustExp
  rw [h, Nat.shiftRight_eq_div_pow]

theorem rustMant_eq (b : Nat) : rustMant b = b % 1024 := by
  have h := Nat.and_two_pow_sub_one_eq_mod b 10
  simp only [Nat.reducePow, Nat.reduceSub] at h
  simp only [rustMant, h]

/-- Disjoint bit fields: the Rust `|`-packing equals the `+`-packing. -/
theorem or_pack_eq_add (s e m : Nat) (he : e < 256) (hm : m < 1024) :
    (s <<< 31 ||| e <<< 23) ||| m <<< 13
      = s * 2147483648 + e * 8388608 + m * 8192 := by
  have h1 : s <<< 31 ||| e <<< 23 = (s * 256 + e) <<< 23 := by
    rw [← Nat.shiftLeft_add_eq_or_of_lt
      (by simp only [Nat.shiftLeft_eq, Nat.reducePow]; omega)]
    simp only [Nat.shiftLeft_eq, Nat.reducePow]
    omega
  rw [h1, ← Nat.shiftLeft_add_eq_or_of_lt
    (by simp only [Nat.shiftLeft_eq, Nat.reducePow]; omega)]
  simp only [Nat.shiftLeft_eq, Nat.reducePow]
  omega

/-- Decode a `u16` taking the Rust normal branch into the field model. -/
def ofBits (b : Nat) (hn : RustNormal b) : F16Normal where
  s := rustSign b
  e := rustExp b
  m := rustMant b
  hs := by rw [rustSign_eq]; omega
  he_lo := by have := hn.1; omega
  he_hi := by have := hn.2; rw [rustExp_eq] at *; omega
  hm := by rw [rustMant_eq]; omega

/-- **Simulation** (normal branch): for every `u16` input on which
    `f16_to_f32` takes the normal branch, the bit pattern it hands to
    `f32::from_bits` is `toF32Bits` of the decoded field model. -/
theorem f16_to_f32_normal_simulates (b : Nat) (hn : RustNormal b) :
    rustNormalBits b = toF32Bits (ofBits b hn) := by
  have hx := hn.2
  rw [rustExp_eq] at hx
  simp only [rustNormalBits, toF32Bits, ofBits]
  rw [or_pack_eq_add _ _ _
    (by rw [rustExp_eq]; omega)
    (by rw [rustMant_eq]; omega)]

/-- No `u32` overflow: the Rust packing of any `u16` fits in 32 bits, so the
    `Nat` model and the `u32` computation coincide. -/
theorem rustNormalBits_lt_u32 (b : Nat) (hn : RustNormal b) :
    rustNormalBits b < 4294967296 := by
  rw [f16_to_f32_normal_simulates b hn]
  have h := (ofBits b hn).hs
  have h2 := (ofBits b hn).he_hi
  have h3 := (ofBits b hn).hm
  simp only [toF32Bits]
  omega

/-- The branch guard is decided on the same field `ofBits` stores, so the
    exponent the model sees is exactly the one the Rust tested. -/
theorem ofBits_fields (b : Nat) (hb : b < 65536) (hn : RustNormal b) :
    (ofBits b hn).s = b / 32768 ∧ (ofBits b hn).e = b / 1024 % 32
      ∧ (ofBits b hn).m = b % 1024 := by
  dsimp only [ofBits]
  rw [rustSign_eq, rustExp_eq, rustMant_eq]
  omega

#check @sign_preserved
#check @exp_rebiased
#check @mant_preserved
#check @mant_padding_zero
#check @bias_trick_correct
#check @roundtrip_identity
#check @toF32Bits_monotone
#check @or_pack_eq_add
#check @f16_to_f32_normal_simulates
#check @rustNormalBits_lt_u32
#check @ofBits_fields

end ProvableContracts.F16
