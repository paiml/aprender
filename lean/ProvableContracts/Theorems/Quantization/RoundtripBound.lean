import ProvableContracts.Defs.Quantization
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Quantization Round-Trip Bound

Proves that the dequantize-after-quantize round-trip error is bounded
by scale/2: |dequant(quant(x)) - x| ≤ scale/2.

## Obligation

`QT-BND-001`: |dequantize(quantize(x, s), s) - x| ≤ s/2

The quantization round-trip satisfies:
  dequant(quant(x)) = ⌊x/s + 1/2⌋ · s

The rounding error |⌊y + 1/2⌋ - y| ≤ 1/2 for any y,
so multiplying by |s| gives the bound.
-/

namespace ProvableContracts.Quantization

-- Status: proved
/-- Floor rounding error: |⌊y + 1/2⌋ - y| ≤ 1/2.
    This is the key lemma: the floor of y + 1/2 is within 1/2 of y. -/
theorem round_nearest_error (y : ℝ) :
    |↑(round_nearest y) - y| ≤ 1 / 2 := by
  unfold round_nearest
  have h1 := Int.floor_le (y + 1 / 2)
  have h2 := Int.lt_floor_add_one (y + 1 / 2)
  rw [abs_le]
  constructor
  · linarith
  · linarith

-- Status: proved
/-- Quantization round-trip bound for positive scale:
    |dequantize(quantize(x, s), s) - x| ≤ s / 2. -/
theorem roundtrip_bound (x : ℝ) (scale : ℝ) (hs : scale > 0) :
    |dequantize (quantize x scale) scale - x| ≤ scale / 2 := by
  unfold dequantize quantize
  have hne : scale ≠ 0 := ne_of_gt hs
  -- dequant(quant(x)) - x = round_nearest(x/s) * s - x
  -- = (round_nearest(x/s) - x/s) * s
  have key : ↑(round_nearest (x / scale)) * scale - x =
      (↑(round_nearest (x / scale)) - x / scale) * scale := by
    field_simp
  rw [key, abs_mul, abs_of_pos hs]
  have herr := round_nearest_error (x / scale)
  have h1 : (1 : ℝ) / 2 * scale = scale / 2 := by ring
  linarith [mul_le_mul_of_nonneg_right herr (le_of_lt hs)]

-- Tests
#check @round_nearest_error
#check @roundtrip_bound

end ProvableContracts.Quantization
