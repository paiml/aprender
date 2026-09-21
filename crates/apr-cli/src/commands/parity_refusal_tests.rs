// PMAT-1098 — case table for the `apr parity` architecture refusal.
//
// Included from parity.rs. Names start with `parity_refus` so
// `cargo nextest run -p apr-cli --lib parity_refus` selects exactly this table.

#[cfg(all(test, feature = "inference"))]
mod parity_refusal_case_table {
    use super::{capability_refusal, parity_refusal_for, PARITY_REFUSED_EXIT};

    /// The three architectures the 0.68.0 T-2 dogfood measured as FAIL rows.
    /// Each one is a TOOL refusal, not a model defect.
    #[test]
    fn parity_refusal_moe_architecture_is_refused_with_issue_3367() {
        // `qwen3moe` is the RAW general.architecture string Qwen3-Coder-30B-A3B
        // and Qwen3-30B-A3B ship; `qwen3_moe` is the canonical key. Every
        // spelling `ArchConstraints` calls MoE must be refused, canonical or not.
        for raw in ["qwen3moe", "qwen3_moe", "qwen3_5_moe", "qwen3_5moe"] {
            let r = parity_refusal_for(raw, Vec::<&str>::new()).unwrap_or_else(|| {
                panic!("{raw} must be refused: the dense parity loop cannot route it")
            });
            assert_eq!(r.issue, "#3367");
            assert!(
                r.reason.contains("MoE placeholder"),
                "reason must name WHY the dense loop is wrong here: {}",
                r.reason
            );
        }
    }

    /// The refusal names the canonical key when `normalize_architecture` knows
    /// the spelling, and the RAW tag when it does not.
    ///
    /// `normalize_architecture` is NOT total over the MoE spellings
    /// `ArchConstraints` accepts: `qwen3_5moe` falls through its `_ => "llama"`
    /// arm. Naming that refusal `architecture=llama` would blame a dense
    /// architecture parity runs fine — the same laundering the `qwen35 -> qwen3`
    /// fold would do. When in doubt the refusal reports what the file said.
    #[test]
    fn parity_refusal_moe_names_the_canonical_key_only_when_the_normalizer_knows_it() {
        for (raw, named) in [
            ("qwen3moe", "qwen3_moe"),
            ("qwen3_moe", "qwen3_moe"),
            ("qwen3_5_moe", "qwen3_moe"),
            ("qwen3_5moe", "qwen3_5moe"),
        ] {
            let r = parity_refusal_for(raw, Vec::<&str>::new()).expect("refused");
            assert_eq!(r.architecture, named, "arch named for raw tag {raw}");
            assert_ne!(
                r.architecture, "llama",
                "never launder a refusal as a dense arch"
            );
        }
    }

    /// #3477 (operator ruling 2026-09-19): `qwen35` — the ONE spelling the
    /// runtime dispatches to the hybrid forward — now has a GPU forward
    /// (`Qwen35CudaModel`, #3090) as well as the CPU one (#3091), so parity can
    /// measure GPU-vs-CPU for it and must stop refusing it. Every undispatched
    /// hybrid spelling keeps the refusal: no forward reaches them, so parity
    /// would still be comparing nothing.
    #[test]
    fn parity_refusal_qwen35_is_admitted_and_undispatched_spellings_are_refused() {
        assert!(
            parity_refusal_for("qwen35", Vec::<&str>::new()).is_none(),
            "the GPU forward exists (#3090) — parity must run, not refuse"
        );
        assert!(
            parity_refusal_for("qwen35", ["blk.0.ssm_a", "blk.0.attn_qkv.weight"]).is_none(),
            "the SSM tensors are exactly what the hybrid forward consumes"
        );
        for raw in ["qwen3.5", "qwen3_5", "QWEN3_5"] {
            let r = parity_refusal_for(raw, Vec::<&str>::new())
                .unwrap_or_else(|| panic!("{raw} reaches no forward: it must be refused"));
            assert_eq!(r.issue, "#3090");
            assert!(
                r.reason.contains("GPU forward"),
                "reason must name the missing half: {}",
                r.reason
            );
        }
    }

    /// `normalize_architecture("qwen3_5") == "qwen3"`, so naming the refusal with
    /// the shared normalizer would print `architecture=qwen3` — a dense arch that
    /// parity runs happily. The refusal must name the architecture that was
    /// actually refused.
    #[test]
    fn parity_refusal_qwen35_line_does_not_launder_the_arch_as_qwen3() {
        let r = parity_refusal_for("qwen3_5", Vec::<&str>::new()).expect("refused");
        assert_eq!(r.architecture, "qwen3_5");
        assert!(
            !r.line().contains("architecture=qwen3 "),
            "the refusal must not report qwen3_5 as plain qwen3: {}",
            r.line()
        );
    }

    /// An SSM/Gated-DeltaNet GGUF is refused on the TENSOR evidence too, even
    /// when its architecture tag is one the dense loop would otherwise accept —
    /// this is `realizar::gguf::unsupported_architecture_reason`, the predicate
    /// `apr check` and `apr ptx-map` already share.
    #[test]
    fn parity_refusal_ssm_tensors_are_refused_whatever_the_arch_tag_says() {
        let r = parity_refusal_for("llama", ["blk.0.ssm_a", "blk.0.attn_q.weight"])
            .expect("an SSM tensor is a refusal on any arch tag");
        assert_eq!(r.issue, "#3090");
    }

    /// The dense archs are UNCHANGED: the predicate is false for them, so
    /// `apr parity` still takes the ordinary CPU-vs-GPU path.
    #[test]
    fn parity_refusal_predicate_is_false_for_every_dense_arch() {
        for raw in [
            "qwen2", "qwen2.5", "qwen3", "llama", "mistral", "gemma2", "phi3",
        ] {
            assert!(
                parity_refusal_for(raw, ["blk.0.attn_q.weight", "output.weight"]).is_none(),
                "{raw} is a dense arch — parity must run it, not refuse it"
            );
        }
    }

    /// The exit code is DISTINCT from every code `apr parity` already emits:
    /// 3 (FileNotFound), 5 (ValidationFailed), 8 (InferenceFailed), 9
    /// (FeatureDisabled). A classifier that cannot tell a refusal from a crash
    /// would launder a real crash as "unmeasured".
    #[test]
    fn parity_refusal_exit_code_is_distinct_from_the_codes_parity_already_emits() {
        use crate::error::CliError;
        let refused = parity_refusal_for("qwen3moe", Vec::<&str>::new())
            .expect("refused")
            .into_error();
        assert_eq!(refused.exit_code_value(), PARITY_REFUSED_EXIT);
        for other in [
            CliError::FileNotFound(std::path::PathBuf::from("x")).exit_code_value(),
            CliError::ValidationFailed("x".to_string()).exit_code_value(),
            CliError::InferenceFailed("x".to_string()).exit_code_value(),
            CliError::FeatureDisabled("x".to_string()).exit_code_value(),
            CliError::ParityFailed("x".to_string()).exit_code_value(),
        ] {
            assert_ne!(
                PARITY_REFUSED_EXIT, other,
                "the refusal code must not collide with a code apr parity already uses"
            );
        }
    }

    /// The one-line stderr shape `scripts/check_model_parity.sh` greps for.
    /// If this format changes, C14 silently reclassifies every refusal as FAIL,
    /// so the literal prefix is asserted here and in the script's case table.
    #[test]
    fn parity_refusal_line_has_the_shape_c14_greps_for() {
        let r = parity_refusal_for("qwen3moe", Vec::<&str>::new()).expect("refused");
        let line = r.line();
        assert!(
            line.starts_with("parity: REFUSED architecture=qwen3_moe — "),
            "C14 greps this prefix: {line}"
        );
        assert!(
            line.ends_with("(#3367)"),
            "the issue is the last token: {line}"
        );
        assert_eq!(line.lines().count(), 1, "exactly ONE line");
    }

    /// `--json` mode emits the refusal as data, with the same three fields.
    #[test]
    fn parity_refusal_json_carries_architecture_reason_and_issue() {
        let r = parity_refusal_for("qwen3_5", Vec::<&str>::new()).expect("refused");
        let v = r.json();
        let refused = v.get("refused").expect("top-level `refused` key");
        assert_eq!(
            refused.get("architecture").and_then(|a| a.as_str()),
            Some("qwen3_5")
        );
        assert_eq!(refused.get("issue").and_then(|a| a.as_str()), Some("#3090"));
        assert!(refused
            .get("reason")
            .and_then(|a| a.as_str())
            .is_some_and(|s| s.contains("GPU forward")));
        // A refusal is NOT a parity result: it must not carry the keys the
        // judge reads, or `check_model_parity.sh` would judge an empty run.
        assert!(v.get("metrics").is_none(), "a refusal has no metrics");
        assert!(v.get("parity").is_none(), "a refusal makes no parity claim");
    }

    /// #3685: what the PMAT-785 capability gate returns for an F16 model, verbatim in shape.
    fn f16_capability_mismatch() -> realizar::error::RealizarError {
        realizar::error::RealizarError::CapabilityMismatch {
            architecture: "qwen2".to_string(),
            missing_ops: "GPU GEMV kernel for the model's quantization type".to_string(),
            suggestion: "Model carries a quant type without a verified GPU kernel (e.g. \
                         Q5_1/Q8_1/Q2_K/Q3_K/Q8_K/F16/IQ*). It will use CPU inference to avoid \
                         silent Q4_K-decode garbage (PMAT-785)."
                .to_string(),
        }
    }

    /// #3685: a CUDA init the capability gate refused is a REFUSAL (exit 12) carrying the
    /// one line check_model_parity.sh reads as UNMEASURED-TOOL, never "parity disproven"
    /// (exit 5): nothing was compared.
    #[test]
    fn parity_refusal_3685_capability_refused_quant_is_exit_12_with_the_refusal_line() {
        let err = f16_capability_mismatch();
        let r = capability_refusal("qwen2", Some(1), &err).expect("a capability refusal");
        assert_eq!(
            r.line(),
            format!("parity: REFUSED architecture=qwen2 quant=F16 reason={err}"),
            "the one stderr line the issue specifies"
        );
        assert!(r.line().starts_with("parity: REFUSED architecture="), "the prefix C14 greps");
        let code = r.into_error().exit_code_value();
        assert_eq!(code, PARITY_REFUSED_EXIT, "a refusal exits 12");
        assert_ne!(code, 5, "5 means a comparison ran and failed; none ran");
    }

    /// The gate can also refuse for a missing op rather than a quant type: still a refusal,
    /// and the line then names no quant.
    #[test]
    fn parity_refusal_3685_capability_refusal_without_a_named_quant_is_still_exit_12() {
        let err = f16_capability_mismatch();
        let r = capability_refusal("phi3", None, &err).expect("a capability refusal");
        assert!(r.line().contains("architecture=phi3") && !r.line().contains("quant="));
        assert_eq!(r.into_error().exit_code_value(), PARITY_REFUSED_EXIT);
    }

    /// Any OTHER CUDA init failure (an OOM, a driver error) is not a refusal: the caller
    /// keeps mapping it as before, so a real crash is never laundered as UNMEASURED-TOOL.
    #[test]
    fn parity_refusal_3685_other_cuda_init_errors_are_not_refusals() {
        let oom = realizar::error::RealizarError::InferenceError("cuMemAlloc: out of memory".into());
        assert!(capability_refusal("qwen2", Some(1), &oom).is_none());
    }

    /// `--json` carries the quant too, and still no parity result.
    #[test]
    fn parity_refusal_3685_capability_refusal_json_names_the_quant() {
        let r = capability_refusal("qwen2", Some(1), &f16_capability_mismatch()).expect("refused");
        let v = r.json();
        let refused = v.get("refused").expect("top-level `refused` key");
        assert_eq!(refused.get("quant").and_then(|q| q.as_str()), Some("F16"));
        assert!(v.get("metrics").is_none() && v.get("parity").is_none());
    }
}
