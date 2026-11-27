# Agent rules: history and design detail

Moved verbatim from CLAUDE.md (#7444). These are design notes and history, not rules. The rules they support stay in CLAUDE.md.

### FFN Gate+Up Kernel Fusion (PMAT-FFN-FUSION)

The SwiGLU FFN block fuses gate and up projections into a single rayon dispatch via
`generic_fused_gate_up_matvec_into<F>` (`crates/aprender-serve/src/quantize/fused_gate_up.rs:63`). This halves
rayon spawn overhead (56→28 dispatches/token on 28-layer models) and improves L1/L2 cache
reuse by loading the activation vector once per midi-tile instead of twice.

- **Fused path**: Q4K, Q5K, Q6K when both gate+up weights share the same qtype and dims
- **Fallback**: `rayon::join` with two separate `fused_matmul_into` for mixed types
- **Q8K path**: Existing `fused_q4k_q8k_ffn_up_gate_into` still used when Q8K activations available
- **Key files**: `crates/aprender-serve/src/quantize/fused_gate_up.rs`,
  `crates/aprender-serve/src/gguf/inference/fused_matmul_into.rs` (`fused_gate_up_matmul_into`)

