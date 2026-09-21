

/// #3477: which pair of forwards `apr parity` measures an architecture with.
///
/// No GPU and no model file: the dispatch is a pure function of the GGUF
/// architecture tag, so the regression that produced this ticket — `qwen35`
/// admitted by the refusal but still sent to the dense loader — is catchable on
/// any host.
#[cfg(all(test, feature = "inference"))]
mod parity_arm_tests {
    use super::{parity_arm, ParityArm};

    #[test]
    fn qwen35_is_measured_by_the_hybrid_arm() {
        assert_eq!(
            parity_arm("qwen35"),
            ParityArm::Hybrid,
            "the hybrid has both forwards (#3090/#3091); the dense loader refuses its layers"
        );
    }

    /// #3714 R2: the spellings the runtime dispatches to the routed-expert
    /// forward take the MoE arm; `qwen35moe` (the GGUF Qwen3.5-MoE tag) reaches
    /// no MoE forward and does not.
    #[test]
    fn qwen3moe_is_measured_by_the_moe_arm() {
        for arch in ["qwen3moe", "qwen3_moe"] {
            assert_eq!(
                parity_arm(arch),
                ParityArm::Moe,
                "{arch}: both routed-expert forwards exist (#3367 CPU, #3714 CUDA)"
            );
        }
        assert_ne!(parity_arm("qwen35moe"), ParityArm::Moe);
    }

    #[test]
    fn every_other_architecture_keeps_the_dense_arm() {
        for arch in ["llama", "qwen2", "qwen3", "gemma", "phi3", ""] {
            assert_eq!(
                parity_arm(arch),
                ParityArm::Dense,
                "{arch} is measured by the dense CPU/GPU loop"
            );
        }
    }

    /// The spellings the runtime does NOT dispatch to the hybrid forward are
    /// refused by `parity_refusal_for` before any arm is chosen — so the arm
    /// they would otherwise take is the dense one, not a hybrid loop that would
    /// fail to load their layers.
    #[test]
    fn undispatched_hybrid_spellings_do_not_take_the_hybrid_arm() {
        for arch in ["qwen3.5", "qwen3_5", "QWEN3_5"] {
            assert_eq!(parity_arm(arch), ParityArm::Dense, "{arch}");
        }
    }
}

/// #3477: a hybrid file the GPU upload cannot take is REFUSED from the header,
/// not failed halfway through the upload.
///
/// Header-only inputs — `(tensor name, GGML type)` pairs — so the whole rule is
/// falsifiable with no GPU, no model and no file.
#[cfg(all(test, feature = "inference"))]
mod hybrid_quant_refusal_tests {
    use super::{hybrid_quant_refusal, PARITY_REFUSED_EXIT};

    /// GGML type 23 is IQ4_XS (`gguf::ggml_type_table`) — what
    /// `Qwen3.5-0.8B-IQ4_XS.gguf`, the file the C14 manifest row resolves to on
    /// this box, stores `blk.0.attn_gate.weight` as. Before this refusal that
    /// row read `FAIL … Operation 'qwen35_cuda_upload' not supported` — a red
    /// row naming the MODEL for a missing kernel.
    const IQ4_XS: u32 = 23;
    /// Q4_K: a type the GPU GEMV kernels do cover.
    const Q4_K: u32 = 12;

    #[test]
    fn a_deltanet_projection_with_no_gpu_kernel_is_refused() {
        let r = hybrid_quant_refusal("qwen35", [("blk.0.ssm_beta.weight", IQ4_XS)])
            .expect("no GPU GEMV kernel for IQ4_XS: parity cannot build the GPU half");
        assert_eq!(r.architecture, "qwen35");
        assert!(
            r.reason.contains("blk.0.ssm_beta.weight") && r.reason.contains("23"),
            "the refusal must name the tensor and its type: {}",
            r.reason
        );
        assert!(
            !r.reason.starts_with("Architecture 'qwen35'"),
            "the line already prints architecture=qwen35: {}",
            r.reason
        );
        assert_eq!(r.issue, "PMAT-785");
        assert!(r.line().starts_with("parity: REFUSED architecture=qwen35 — "));
        assert_eq!(PARITY_REFUSED_EXIT, 12, "UNMEASURED-TOOL, not FAIL");
    }

    #[test]
    fn a_hybrid_the_gpu_can_take_is_not_refused() {
        assert!(
            hybrid_quant_refusal(
                "qwen35",
                [
                    ("token_embd.weight", Q4_K),
                    ("blk.0.ssm_beta.weight", Q4_K),
                    ("blk.0.attn_qkv.weight", Q4_K),
                ]
            )
            .is_none(),
            "Q4_K has a verified kernel — this file is measurable and must be measured"
        );
    }

    /// The dense quant gate already speaks for every other architecture; a
    /// second refusal here would refuse dense models parity measures today.
    #[test]
    fn a_dense_architecture_is_never_refused_by_the_hybrid_quant_gate() {
        for arch in ["llama", "qwen2", "qwen3", "qwen3.5"] {
            assert!(
                hybrid_quant_refusal(arch, [("blk.0.attn_q.weight", IQ4_XS)]).is_none(),
                "{arch}"
            );
        }
    }
}

/// The end-to-end contract of the hybrid arm, measured on the real 0.8B file.
///
/// Needs a CUDA device AND the model, so it is `cuda`-gated and skips when the
/// file is absent (the same shape every model-file test in this tree uses); the
/// authoritative run is the C14 row, which fails loudly when the model is
/// missing. What it pins is the measured #3517 contract: the CPU forward
/// quantizes its activations, so the two are NOT bit-identical — what must hold
/// is the same DISTRIBUTION (cosine) and the same CHOICE (argmax) at every
/// position, over enough positions for the I8 ≥ 64 rule.
#[cfg(all(test, feature = "cuda"))]
mod hybrid_e2e_tests {
    use super::measure_hybrid;

    const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";
    /// Long enough that the tokenizer yields well over the 64 positions the I8
    /// admission rule wants, without making the test a benchmark.
    const PROMPT_REPEATS: usize = 9;

    #[test]
    fn qwen35_hybrid_cpu_and_cuda_agree_over_at_least_64_positions() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("skipped: {MODEL} is not on this host");
            return;
        }
        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(PROMPT_REPEATS);
        let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map the model");
        let tokens = mapped.model.encode(&prompt).expect("tokenize the prompt");
        assert!(
            tokens.len() >= 64,
            "the prompt must exercise at least 64 positions, got {}",
            tokens.len()
        );

        let measured = measure_hybrid(&mapped, &tokens, false).expect("measure the hybrid");
        assert_eq!(measured.metrics.len(), tokens.len());
        assert!(
            measured.metrics.len() >= 64,
            "the I8 admission rule wants at least 64 positions"
        );

        for m in &measured.metrics {
            assert!(
                m.cosine_similarity >= 0.996,
                "position {}: cosine {} is below the measured #3517 floor",
                m.position,
                m.cosine_similarity
            );
            assert_eq!(
                m.cpu_argmax, m.gpu_argmax,
                "position {}: the two backends chose different tokens",
                m.position
            );
        }
    }
}
