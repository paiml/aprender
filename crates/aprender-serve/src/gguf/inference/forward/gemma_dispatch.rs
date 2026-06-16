// PMAT-809: Gemma-v1 architecture dispatch helpers.
//
// These thin wrappers select the Gemma-specific math (`(1 + weight)` RMSNorm,
// GeGLU gate activation) when the loaded model is Gemma-v1, and the standard
// LLaMA-style math otherwise. Gating lives HERE (one place) so every forward
// variant can call the same method and stay byte-identical for non-Gemma archs.
//
// The three Gemma behaviors:
//   (a) GeGLU FFN          — `gemma_gate_activation` → gelu instead of silu
//   (b) (1+w) RMSNorm      — `rms_norm_arch` / `rms_norm_into_arch`
//   (c) sqrt(hidden) embed  — handled in `embed`/`embed_into` (matmul_fused.rs)
//
// Gemma2/Gemma3 (softcapping) are NOT handled — `GGUFConfig::is_gemma1()` is
// false for them, so they never reach these paths and remain fail-loud at the
// contract gate.

impl OwnedQuantizedModel {
    /// Allocating RMSNorm, arch-dispatched.
    ///
    /// Gemma-v1 uses `(1 + weight)` (PMAT-809 b); all other RMSNorm families use
    /// the standard `* weight`. Byte-identical to `ops::rms_norm` for non-Gemma.
    #[inline]
    pub(crate) fn rms_norm_arch(&self, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        if self.config.rmsnorm_unit_offset() {
            ops::rms_norm_unit_offset(input, weight, eps)
        } else {
            ops::rms_norm(input, weight, eps)
        }
    }

    /// Zero-allocation RMSNorm into a buffer, arch-dispatched. See `rms_norm_arch`.
    #[inline]
    pub(crate) fn rms_norm_into_arch(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) {
        if self.config.rmsnorm_unit_offset() {
            ops::rms_norm_unit_offset_into(input, weight, eps, output);
        } else {
            ops::rms_norm_into(input, weight, eps, output);
        }
    }

    /// Gate-branch activation for a gated FFN, arch-dispatched.
    ///
    /// Gemma-v1 GatedMlp uses GeGLU — `gelu_tanh(gate)` (PMAT-809 a). All other
    /// gated families (LLaMA/Qwen/Mistral SwiGLU) use `silu(gate)`. In-place.
    /// Byte-identical to `ops::silu` for non-Gemma.
    #[inline]
    pub(crate) fn gemma_gate_activation(&self, gate: &mut [f32]) {
        if self.config.geglu_ffn() {
            ops::gelu(gate);
        } else {
            ops::silu(gate);
        }
    }
}
