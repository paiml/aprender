//! Contract tier and kernel equivalence class lookups.
//!
//! Derived from `docs/specifications/sub/registry.md`. Contracts not
//! explicitly listed default to Tier 7 (performance/KAIZEN).

/// Contract tier (1-7). Lower tier = more foundational.
pub fn tier_of(stem: &str) -> u8 {
    // Tier 1: Foundation kernels
    const TIER1: &[&str] = &[
        "softmax-kernel-v1",
        "rmsnorm-kernel-v1",
        "rope-kernel-v1",
        "silu-kernel-v1",
        "swiglu-kernel-v1",
        "gelu-kernel-v1",
        "layernorm-kernel-v1",
        "batchnorm-kernel-v1",
        "embedding-lookup-v1",
        "cross-entropy-kernel-v1",
        "linear-projection-v1",
        "dropout-v1",
        "activation-kernel-v1",
        "bias-add-v1",
        "transpose-kernel-v1",
    ];
    // Tier 2: Composite kernels
    const TIER2: &[&str] = &[
        "attention-kernel-v1",
        "gqa-kernel-v1",
        "matmul-kernel-v1",
        "flash-attention-v1",
        "sliding-window-attention-v1",
        "qk-norm-v1",
        "attention-scaling-v1",
        "bidirectional-attention-v1",
    ];
    // Tier 3: System kernels
    const TIER3: &[&str] = &[
        "kv-cache-equivalence-v1",
        "kv-cache-sizing-v1",
        "sampling-algorithms-v1",
        "inference-pipeline-v1",
        "streaming-tpot-v1",
        "backend-dispatch-v1",
        "conversation-generation-v1",
        "safetensors-cpu-dispatch-v1",
        "safetensors-header-v1",
        "weight-loading-v1",
        "validated-tensor-v1",
    ];
    // Tier 4: Training kernels
    const TIER4: &[&str] = &[
        "adamw-kernel-v1",
        "loss-functions-v1",
        "lora-algebra-v1",
        "classification-finetune-v1",
        "optimization-v1",
        "lbfgs-kernel-v1",
        "cmaes-kernel-v1",
        "gradient-clipping-v1",
        "learning-rate-scheduling-v1",
    ];
    // Tier 5: Classical ML
    const TIER5: &[&str] = &[
        "kmeans-kernel-v1",
        "pagerank-kernel-v1",
        "pca-v1",
        "svm-v1",
        "decision-tree-v1",
        "random-forest-v1",
        "naive-bayes-v1",
        "gbm-v1",
        "arima-v1",
        "bayesian-v1",
        "calibration-v1",
        "dbscan-v1",
        "gaussian-mixture-v1",
        "isotonic-regression-v1",
        "knn-v1",
        "logistic-regression-v1",
        "linear-probe-classifier-v1",
        "active-learning-v1",
    ];
    // Tier 6: Model-specific
    const TIER6: &[&str] = &[
        "qwen2-shapes-v1",
        "qwen2-e2e-verification-v1",
        "qwen3-shapes-v1",
        "qwen3-e2e-verification-v1",
        "qwen3moe-shapes-v1",
        "qwen3moe-e2e-verification-v1",
        "qwen35-shapes-v1",
        "qwen35-hybrid-forward-v1",
    ];

    if TIER1.contains(&stem) {
        1
    } else if TIER2.contains(&stem) {
        2
    } else if TIER3.contains(&stem) {
        3
    } else if TIER4.contains(&stem) {
        4
    } else if TIER5.contains(&stem) {
        5
    } else if TIER6.contains(&stem) {
        6
    } else {
        7
    }
}

/// Kernel equivalence class (A-E). Returns None for contracts that
/// don't belong to any class (training, classical ML, etc.).
pub fn class_of(stem: &str) -> Option<char> {
    // Class A: Llama / Mistral / Yi — GQA + RMSNorm + SiLU + SwiGLU + RoPE
    const CLASS_A: &[&str] = &[
        "gqa-kernel-v1",
        "rmsnorm-kernel-v1",
        "silu-kernel-v1",
        "swiglu-kernel-v1",
        "rope-kernel-v1",
    ];
    // Class B: GPT-2 / BERT — MHA + LayerNorm + GELU + AbsPos
    const CLASS_B: &[&str] = &[
        "attention-kernel-v1",
        "layernorm-kernel-v1",
        "gelu-kernel-v1",
        "absolute-position-v1",
        "bidirectional-attention-v1",
    ];
    // Class C: BLOOM / MPT — MHA + LayerNorm + GELU + ALiBi
    const CLASS_C: &[&str] = &["alibi-kernel-v1"];
    // Class D: Gemma — LayerNorm + GELU + SiLU + GQA
    // (shares contracts with A and B; D-unique contracts are few)
    const CLASS_D: &[&str] = &[];
    // Class E: Qwen — RMSNorm + SwiGLU + GQA + model-specific
    const CLASS_E: &[&str] = &[
        "qwen2-shapes-v1",
        "qwen2-e2e-verification-v1",
        "qwen3-shapes-v1",
        "qwen3-e2e-verification-v1",
        "qwen3moe-shapes-v1",
        "qwen3moe-e2e-verification-v1",
        "qwen35-shapes-v1",
        "qwen35-hybrid-forward-v1",
    ];

    // A contract can belong to multiple classes. Return the primary one.
    if CLASS_A.contains(&stem) {
        Some('A')
    } else if CLASS_B.contains(&stem) {
        Some('B')
    } else if CLASS_C.contains(&stem) {
        Some('C')
    } else if CLASS_D.contains(&stem) {
        Some('D')
    } else if CLASS_E.contains(&stem) {
        Some('E')
    } else {
        None
    }
}

/// All classes a contract belongs to (a contract can be in multiple classes).
pub fn classes_of(stem: &str) -> Vec<char> {
    let mut result = Vec::new();
    // Class A
    if matches!(
        stem,
        "gqa-kernel-v1"
            | "rmsnorm-kernel-v1"
            | "silu-kernel-v1"
            | "swiglu-kernel-v1"
            | "rope-kernel-v1"
    ) {
        result.push('A');
    }
    // Class B
    if matches!(
        stem,
        "attention-kernel-v1"
            | "layernorm-kernel-v1"
            | "gelu-kernel-v1"
            | "absolute-position-v1"
            | "bidirectional-attention-v1"
    ) {
        result.push('B');
    }
    // Class C (shares attention, layernorm, gelu with B)
    if matches!(
        stem,
        "attention-kernel-v1" | "layernorm-kernel-v1" | "gelu-kernel-v1" | "alibi-kernel-v1"
    ) {
        result.push('C');
    }
    // Class D (shares layernorm, gelu, silu, gqa with A/B)
    if matches!(
        stem,
        "layernorm-kernel-v1" | "gelu-kernel-v1" | "silu-kernel-v1" | "gqa-kernel-v1"
    ) {
        result.push('D');
    }
    // Class E
    if matches!(
        stem,
        "rmsnorm-kernel-v1"
            | "swiglu-kernel-v1"
            | "gqa-kernel-v1"
            | "qwen2-shapes-v1"
            | "qwen2-e2e-verification-v1"
            | "qwen3-shapes-v1"
            | "qwen3-e2e-verification-v1"
            | "qwen3moe-shapes-v1"
            | "qwen3moe-e2e-verification-v1"
            | "qwen35-shapes-v1"
            | "qwen35-hybrid-forward-v1"
    ) {
        result.push('E');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier1_foundation() {
        assert_eq!(tier_of("softmax-kernel-v1"), 1);
        assert_eq!(tier_of("rmsnorm-kernel-v1"), 1);
        assert_eq!(tier_of("dropout-v1"), 1);
    }

    #[test]
    fn tier2_composite() {
        assert_eq!(tier_of("attention-kernel-v1"), 2);
        assert_eq!(tier_of("flash-attention-v1"), 2);
    }

    #[test]
    fn tier3_system() {
        assert_eq!(tier_of("kv-cache-equivalence-v1"), 3);
        assert_eq!(tier_of("sampling-algorithms-v1"), 3);
    }

    #[test]
    fn tier4_training() {
        assert_eq!(tier_of("adamw-kernel-v1"), 4);
        assert_eq!(tier_of("lora-algebra-v1"), 4);
    }

    #[test]
    fn tier5_classical() {
        assert_eq!(tier_of("kmeans-kernel-v1"), 5);
        assert_eq!(tier_of("pagerank-kernel-v1"), 5);
    }

    #[test]
    fn tier6_model_specific() {
        assert_eq!(tier_of("qwen2-shapes-v1"), 6);
        assert_eq!(tier_of("qwen35-shapes-v1"), 6);
    }

    #[test]
    fn tier7_default() {
        assert_eq!(tier_of("some-unknown-contract-v1"), 7);
        assert_eq!(tier_of("encoder-forward-v1"), 7);
    }

    #[test]
    fn class_a_llama() {
        assert_eq!(class_of("gqa-kernel-v1"), Some('A'));
        assert_eq!(class_of("rmsnorm-kernel-v1"), Some('A'));
        assert_eq!(class_of("rope-kernel-v1"), Some('A'));
    }

    #[test]
    fn class_b_gpt2() {
        assert_eq!(class_of("attention-kernel-v1"), Some('B'));
        assert_eq!(class_of("layernorm-kernel-v1"), Some('B'));
    }

    #[test]
    fn class_c_bloom() {
        assert_eq!(class_of("alibi-kernel-v1"), Some('C'));
    }

    #[test]
    fn class_e_qwen() {
        assert_eq!(class_of("qwen2-shapes-v1"), Some('E'));
        assert_eq!(class_of("qwen35-shapes-v1"), Some('E'));
    }

    #[test]
    fn class_none_for_non_arch() {
        assert_eq!(class_of("adamw-kernel-v1"), None);
        assert_eq!(class_of("kmeans-kernel-v1"), None);
    }

    #[test]
    fn multi_class_membership() {
        // gqa-kernel-v1 is in A, D, E
        let classes = classes_of("gqa-kernel-v1");
        assert!(classes.contains(&'A'));
        assert!(classes.contains(&'D'));
        assert!(classes.contains(&'E'));
    }

    #[test]
    fn multi_class_layernorm() {
        // layernorm is in B, C, D
        let classes = classes_of("layernorm-kernel-v1");
        assert!(classes.contains(&'B'));
        assert!(classes.contains(&'C'));
        assert!(classes.contains(&'D'));
    }
}
