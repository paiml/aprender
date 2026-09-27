/// R8 (la-0.72): GatedDeltaNet tensors that are never block-quantized.
///
/// Contract: `contracts/gdn-quantize-policy-v1.yaml`. Three tensors in every
/// Qwen3.5 linear-attention layer stay F32 under Int8/Int4/Q4K:
///
/// | tensor   | HF name                        | GGUF name              | why |
/// |----------|--------------------------------|------------------------|-----|
/// | conv1d   | `linear_attn.conv1d.weight`    | `blk.N.ssm_conv1d.weight` | K=4 taps per channel; a 4-bit step is a large fraction of each tap |
/// | A_log    | `linear_attn.A_log`            | `blk.N.ssm_a`          | enters as `-exp(A_log)`, so an absolute error becomes a relative decay error |
/// | dt_bias  | `linear_attn.dt_bias`          | `blk.N.ssm_dt.bias`    | sits inside `softplus` at ~-5, where softplus ≈ exp |
///
/// Before this policy the three converter rules disagreed: `conv1d` was
/// quantized by all of them (it matches no norm/bias pattern and is ≥ 256
/// elements), and the export path fake-quantized `A_log` too.
///
/// Matching is by dotted segment, so `ssm_alpha` (a projection, quantizable)
/// is not `ssm_a`, and Mamba's `ssm_dt.weight` (a projection) is not
/// `ssm_dt.bias`.
pub(crate) fn gdn_keeps_full_precision(name: &str) -> bool {
    name.split('.')
        .any(|s| s.contains("conv1d") || matches!(s, "A_log" | "ssm_a" | "dt_bias" | "ssm_dt_bias"))
        || name.contains("ssm_dt.bias")
}
