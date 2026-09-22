//! #3809 — the CPU half of `gpu_cpu_trace_compare`'s fixture, with no GPU.
//!
//! WHY THIS EXISTS. `gpu_cpu_trace_compare` asserts CPU/GPU agreement and fails
//! at 91.92% L2 divergence. Before that number can mean "the GPU is wrong", the
//! fixture has to be capable of a meaningful comparison at all: the target could
//! not COMPILE for months, so `create_test_model()` was never updated alongside
//! the code it exercises. This runs only the CPU side — no `cuda` feature, no
//! card — so the first elimination costs nothing and blocks on nothing.
//!
//! WHAT IT ESTABLISHES: the fixture builds a model the CPU traced path accepts,
//! and that path returns a full set of finite, non-degenerate logits. Measured:
//! 256 logits, 256 finite, 256 non-zero, L2 372.95, range ±32.48.
//!
//! WHAT IT DOES NOT ESTABLISH, and this matters more:
//!   * NOT that the GPU adapter receives what it expects. The divergence may
//!     still be a fixture/adapter mismatch; this only rules out "the CPU side
//!     produces garbage".
//!   * NOT that the fixture is a GOOD discriminator. `token_embedding` is
//!     `sin(i * 0.01)`, so token 42's row spans 0.64 radians — the embedding is a
//!     NEARLY CONSTANT vector (mean 0.867, std 0.088). That is fine for a smoke
//!     fixture and poor for finding a numerical bug, because a near-constant
//!     input hides per-channel errors. If this row ever concludes "no divergence
//!     detectable", the fixture's inability to detect one is part of the answer.
//!   * NOT anything about correctness. There is no reference to compare against;
//!     the assertions are well-formedness only, deliberately.
//!
//! KNOWN WEAKNESS: the fixture below is a COPY of the one in
//! `gpu_cpu_trace_compare.rs`, because that one lives inside a
//! `#[cfg(all(test, feature = "cuda"))]` module and Rust integration tests
//! cannot import from one another. A copy that drifts stops testing the thing it
//! exists for. The fix is to lift the fixture into a shared `tests/common/`
//! module used by both — deliberately NOT done here, because the cuda target is
//! mid-repair on another branch (#3809) and restructuring it now would collide.
//! Whoever folds that repair should do the consolidation.
#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod tests {
    use realizar::apr_transformer::{
        AprTransformer, AprTransformerConfig, AprTransformerLayer, TracedForward,
    };

    /// Create a minimal test model
    fn create_test_model() -> AprTransformer {
        let hidden_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = 128;
        let vocab_size = 256;

        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        };

        let token_embedding: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();

        let output_norm_weight = vec![1.0f32; hidden_dim];

        let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i as f32) * 0.001).cos())
            .collect();

        let qkv_out_dim = hidden_dim + 2 * kv_dim;
        let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();

        let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();

        let attn_norm_weight = vec![1.0f32; hidden_dim];

        let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.1)
            .collect();
        let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
            .map(|i| ((i as f32) * 0.04).cos() * 0.1)
            .collect();
        let ffn_gate_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.1)
            .collect();
        let ffn_norm_weight = vec![1.0f32; hidden_dim];

        let layer = AprTransformerLayer {
            qkv_weight,
            qkv_bias: None,
            attn_output_weight,
            attn_output_bias: None,
            attn_norm_weight,
            attn_norm_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_gate_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        };

        AprTransformer {
            config,
            token_embedding,
            layers: vec![layer],
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            lm_head_tied: false,
            q4k_layers: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
        }
    }

    /// The CPU traced forward returns a full set of finite, non-degenerate logits.
    ///
    /// Each assertion names a distinct way the fixture could be dead, because
    /// "it ran" is not a result: a model returning all zeros, or NaNs, or a
    /// short vector, would each let a 92% divergence mean nothing at all.
    #[test]
    fn cpu_half_of_the_trace_fixture_is_well_formed() {
        let mut model = create_test_model();
        let trace = TracedForward::forward_traced(&mut model, &[42u32])
            .expect("CPU traced forward failed on the fixture");

        let logits = &trace.logits;
        let finite = logits.iter().filter(|x| x.is_finite()).count();
        let nonzero = logits.iter().filter(|x| x.abs() > 1e-12).count();
        let l2: f32 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert_eq!(
            logits.len(),
            model.config.vocab_size,
            "the CPU path returned {} logits for a vocab of {}",
            logits.len(),
            model.config.vocab_size
        );
        assert_eq!(
            finite,
            logits.len(),
            "{} of {} CPU logits are not finite — a divergence percentage against \
             NaN or inf is meaningless",
            logits.len() - finite,
            logits.len()
        );
        assert!(
            nonzero > 0,
            "every CPU logit is zero — the fixture computes nothing, so any \
             divergence figure is an artefact"
        );
        assert!(
            l2.is_finite() && l2 > 0.0,
            "CPU logit L2 is {l2}, so the denominator of the divergence ratio is dead"
        );

        // One layer, and its activations must be finite too: a divergence that
        // appears only at the logits means something different from one that is
        // already present at the layer boundary, and that reading needs both
        // sides to be well formed.
        assert!(
            !trace.layer_activations.is_empty(),
            "the traced forward recorded no layers"
        );
        for layer in &trace.layer_activations {
            for (name, stat) in [
                ("qkv", &layer.qkv_stats),
                ("attn_out", &layer.attn_out_stats),
                ("ffn_out", &layer.ffn_out_stats),
            ] {
                assert!(
                    stat.mean.is_finite() && stat.std_dev.is_finite(),
                    "layer {} {name} stats are not finite (mean={}, std={})",
                    layer.layer_idx,
                    stat.mean,
                    stat.std_dev
                );
            }
        }
    }
}
