import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Quantization Definitions

Mathematical definitions of uniform affine quantization and dequantization,
modeling the round-trip property of INT8/INT4 quantization kernels.

## References

- Jacob et al. (2018) Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference
- Nagel et al. (2021) A White Paper on Neural Network Quantization
-/

namespace ProvableContracts.Quantization

/-- Round-to-nearest: ⌊x + 0.5⌋. -/
noncomputable def round_nearest (x : ℝ) : ℤ :=
  ⌊x + 1 / 2⌋

/-- Quantize: round(x / scale). -/
noncomputable def quantize (x : ℝ) (scale : ℝ) : ℤ :=
  round_nearest (x / scale)

/-- Dequantize: q * scale. -/
noncomputable def dequantize (q : ℤ) (scale : ℝ) : ℝ :=
  (q : ℝ) * scale

end ProvableContracts.Quantization
