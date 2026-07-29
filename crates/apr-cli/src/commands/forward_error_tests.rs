// PMAT-540 Phase 5: Tests for forward_error/QA helper functions

#[cfg(test)]
mod forward_error_tests {
    use super::*;

    // ========================================================================
    // strip_quant_suffix
    // ========================================================================

    #[test]
    fn strip_quant_suffix_q4k() {
        assert_eq!(strip_quant_suffix("model-q4k"), "model");
    }

    #[test]
    fn strip_quant_suffix_q4_k_m() {
        assert_eq!(strip_quant_suffix("model-q4_k_m"), "model");
    }

    #[test]
    fn strip_quant_suffix_q6k() {
        assert_eq!(strip_quant_suffix("model-q6k"), "model");
    }

    #[test]
    fn strip_quant_suffix_f16() {
        assert_eq!(strip_quant_suffix("model-f16"), "model");
    }

    #[test]
    fn strip_quant_suffix_f32() {
        assert_eq!(strip_quant_suffix("model-f32"), "model");
    }

    #[test]
    fn strip_quant_suffix_no_suffix() {
        assert_eq!(strip_quant_suffix("model"), "model");
    }

    #[test]
    fn strip_quant_suffix_preserves_non_quant() {
        assert_eq!(strip_quant_suffix("qwen2.5-coder-7b"), "qwen2.5-coder-7b");
    }

    // ========================================================================
    // compute_argmax
    // ========================================================================

    #[test]
    fn compute_argmax_basic() {
        assert_eq!(compute_argmax(&[1.0, 3.0, 2.0]), Some(1));
    }

    #[test]
    fn compute_argmax_first() {
        assert_eq!(compute_argmax(&[9.0, 1.0, 2.0]), Some(0));
    }

    #[test]
    fn compute_argmax_last() {
        assert_eq!(compute_argmax(&[1.0, 2.0, 99.0]), Some(2));
    }

    #[test]
    fn compute_argmax_empty() {
        assert_eq!(compute_argmax(&[]), None);
    }

    #[test]
    fn compute_argmax_single() {
        assert_eq!(compute_argmax(&[42.0]), Some(0));
    }

    #[test]
    fn compute_argmax_negative() {
        assert_eq!(compute_argmax(&[-3.0, -1.0, -2.0]), Some(1));
    }

    // ========================================================================
    // hf_cache_dir_matches
    // ========================================================================

    #[test]
    fn hf_cache_dir_matches_standard() {
        assert!(hf_cache_dir_matches("models--Qwen--Qwen2.5-Coder-7B", "qwen2.5-coder"));
    }

    #[test]
    fn hf_cache_dir_matches_case_insensitive() {
        assert!(hf_cache_dir_matches("models--Meta--Llama-3-8B", "llama-3"));
    }

    #[test]
    fn hf_cache_dir_matches_no_prefix() {
        assert!(!hf_cache_dir_matches("Qwen--Qwen2.5", "qwen2.5"));
    }

    #[test]
    fn hf_cache_dir_matches_no_match() {
        assert!(!hf_cache_dir_matches("models--OpenAI--Whisper", "llama"));
    }

    // ========================================================================
    // detect_size_from_filename (word boundary)
    // ========================================================================

    #[test]
    fn detect_size_from_filename_3b_with_boundary() {
        assert_eq!(detect_size_from_filename("model-3b-chat"), Some("3b"));
    }

    #[test]
    fn detect_size_from_filename_7b_dot_gguf() {
        assert_eq!(detect_size_from_filename("llama-7b.gguf"), Some("7b"));
    }

    #[test]
    fn detect_size_from_filename_0_5b() {
        assert_eq!(detect_size_from_filename("model-0.5b-instruct"), Some("0.5b"));
    }

    #[test]
    fn detect_size_from_filename_no_match_hex() {
        // Random hex like "3bF2" should NOT match "3b"
        assert_eq!(detect_size_from_filename("tmp3bF2a1"), None);
    }

    #[test]
    fn detect_size_from_filename_no_match_hex_prefixed_digit() {
        // Reproducer for CI flake (workspace-test #25276156029): random hex like
        // ".tmp97b...gguf" has "7b" at a position where the BEFORE-char is a
        // digit ('9'). Without rejecting digit-prefixed matches we mis-detected
        // the empty file as "7b". Must return None.
        assert_eq!(detect_size_from_filename(".tmp97b1234.gguf"), None);
        // 132b pattern — '3' before, '2b' is the candidate
        assert_eq!(detect_size_from_filename(".tmp132b.gguf"), None);
        // 'model3b' (letter prefix) is still valid — see _3b_standalone test
        assert_eq!(detect_size_from_filename("model3b"), Some("3b"));
    }

    #[test]
    fn detect_size_from_filename_14b() {
        assert_eq!(detect_size_from_filename("model-14b-chat.gguf"), Some("14b"));
    }

    #[test]
    fn detect_size_from_filename_underscore_variant() {
        assert_eq!(detect_size_from_filename("model_0_5b"), Some("0.5b"));
    }

    // ========================================================================
    // estimate_size_from_file
    // ========================================================================

    #[test]
    fn estimate_size_from_file_zero_bytes() {
        let file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
        assert_eq!(estimate_size_from_file(file.path()), "0.5b");
    }

    #[test]
    fn estimate_size_from_file_nonexistent() {
        // Non-existent file → metadata fails → unwrap_or(0) → "0.5b"
        assert_eq!(
            estimate_size_from_file(std::path::Path::new("/nonexistent/model.gguf")),
            "0.5b"
        );
    }

    // ========================================================================
    // run_format_parity_gate: non-GGUF SKIP (regression — previously FAILed,
    // making `apr qa <safetensors>` fail overall even though the gate can't
    // meaningfully run without a GGUF primary.)
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn format_parity_skips_safetensors_primary() {
        use super::QaConfig;
        use std::io::Write;

        // Minimal SafeTensors file: 8-byte LE length of empty JSON metadata
        // `{}` = 2 bytes → header reads `0200 0000 0000 0000` then `{}`.
        let mut tmp = tempfile::NamedTempFile::with_suffix(".safetensors").expect("temp");
        tmp.write_all(&2u64.to_le_bytes()).expect("write len");
        tmp.write_all(b"{}").expect("write meta");
        tmp.flush().expect("flush");

        let config = QaConfig::default();
        let result = super::run_format_parity_gate(tmp.path(), &config).expect("gate ran");

        assert!(result.skipped, "non-GGUF primary must SKIP, not FAIL/PASS");
        assert!(result.passed, "skipped gates count as passed in summary");
        assert!(
            result.message.contains("Non-GGUF"),
            "skip reason should say Non-GGUF, got: {}",
            result.message
        );
    }

    #[cfg(feature = "inference")]
    #[test]
    fn format_parity_skips_apr_primary() {
        use super::QaConfig;
        use std::io::Write;

        // APR v2 magic: b"APR\0" followed by u32 LE version
        let mut tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("temp");
        tmp.write_all(b"APR\0").expect("write magic");
        tmp.write_all(&2u32.to_le_bytes()).expect("write version");
        tmp.flush().expect("flush");

        let config = QaConfig::default();
        let result = super::run_format_parity_gate(tmp.path(), &config).expect("gate ran");

        assert!(result.skipped, "APR primary must SKIP, not FAIL");
        assert!(result.passed, "skipped gates count as passed");
        assert!(result.message.contains("Non-GGUF"), "got: {}", result.message);
    }

    // ========================================================================
    // PMAT-743: format-parity discovery must ignore apr's own conversion
    // artifacts (`*.converted*.safetensors`) — they are circular references,
    // not independent SafeTensors, and are frequently stale / double-converted.
    // ========================================================================

    #[test]
    fn synthetic_artifact_detection() {
        assert!(is_synthetic_conversion_artifact("m-q4_k_m.converted.safetensors"));
        assert!(is_synthetic_conversion_artifact(
            "m-q4_k_m.converted.converted.safetensors"
        ));
        // Genuine HF SafeTensors never contain a `.converted.` segment.
        assert!(!is_synthetic_conversion_artifact("model.safetensors"));
        assert!(!is_synthetic_conversion_artifact(
            "model-00001-of-00002.safetensors"
        ));
        assert!(!is_synthetic_conversion_artifact("qwen2.5-coder.safetensors"));
    }

    #[test]
    fn snapshot_discovery_prefers_genuine_over_converted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path();
        // A genuine shard alongside two synthetic conversion artifacts.
        for name in [
            "model-00001-of-00002.safetensors",
            "m-q4_k_m.converted.safetensors",
            "m-q4_k_m.converted.converted.safetensors",
        ] {
            std::fs::write(p.join(name), b"x").expect("write");
        }
        let found = super::find_safetensors_in_snapshot(p).expect("a genuine shard exists");
        assert_eq!(
            found.file_name().unwrap().to_str().unwrap(),
            "model-00001-of-00002.safetensors",
            "must pick the genuine shard, never a .converted artifact"
        );
    }

    #[test]
    fn snapshot_discovery_skips_when_only_converted_artifacts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path();
        // Only synthetic artifacts exist (GGUF-only model whose cache was polluted).
        for name in [
            "m-q4_k_m.converted.safetensors",
            "m-q4_k_m.converted.converted.safetensors",
        ] {
            std::fs::write(p.join(name), b"x").expect("write");
        }
        assert!(
            super::find_safetensors_in_snapshot(p).is_none(),
            "must find NO independent reference (artifacts ignored), not pick a circular one"
        );
    }

    // ========================================================================
    // PMAT-815: format_parity must SKIP (not FAIL) when the SafeTensors
    // reference is genuinely ABSENT — a missing OPTIONAL input is not a format
    // divergence. Mirrors run_ollama_parity_gate, which SKIPs when Ollama is
    // unavailable. The SKIP must NOT mask a real divergence: an EXPLICIT
    // --safetensors-path is still honored (returned Ok), so a present-but-
    // diverging reference still reaches the FAIL path.
    //
    // resolve_safetensors_path is the locus of the SKIP-vs-FAIL decision:
    //   Ok(path)  -> gate proceeds to actually run the parity comparison
    //   Err(gate) -> gate short-circuits with that GateResult (SKIP or FAIL)
    // ========================================================================

    /// Falsifier case (a): GGUF with NO reference on disk → SKIP, not FAIL.
    /// RED-on-bug: pre-fix this returned `GateResult::failed`, so `.skipped`
    /// was false and `.passed` was false → `apr qa <gguf>` failed overall.
    #[test]
    fn resolve_safetensors_absent_reference_skips_not_fails() {
        use super::QaConfig;

        // Isolated tempdir + a UNIQUE base name so none of the four discovery
        // strategies (sibling file/subdir, HF cache, APR cache — all substring-
        // keyed on the base name) can match anything on the dev/CI machine.
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf = dir
            .path()
            .join("zzqaparitynoref8f3a1c-q4_k_m.gguf");
        std::fs::write(&gguf, b"GGUF\0\0\0\0").expect("write gguf stub");

        let config = QaConfig::default();
        assert!(
            config.safetensors_path.is_none(),
            "precondition: no explicit reference path"
        );

        let result = super::resolve_safetensors_path(&gguf, &config, Duration::from_millis(0));

        match result {
            Ok(p) => panic!(
                "absent reference must NOT resolve to a path; got Ok({})",
                p.display()
            ),
            Err(gate) => {
                assert!(
                    gate.skipped,
                    "absent reference must SKIP, not FAIL (got skipped={}, passed={}, msg={})",
                    gate.skipped, gate.passed, gate.message
                );
                assert!(
                    gate.passed,
                    "a SKIPPED gate counts as passed in the summary (got passed=false, msg={})",
                    gate.message
                );
                assert!(
                    gate.message.to_lowercase().contains("no safetensors reference"),
                    "skip reason must state the reference is unavailable, got: {}",
                    gate.message
                );
            }
        }
    }

    /// Falsifier case (b)/(c) guard: an EXPLICIT --safetensors-path is returned
    /// verbatim (Ok), so the gate proceeds to the real comparison and a present-
    /// but-diverging reference still reaches FAIL — the SKIP does not swallow it.
    #[test]
    fn resolve_safetensors_explicit_path_is_honored_not_skipped() {
        use super::QaConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let gguf = dir.path().join("model-q4_k_m.gguf");
        let explicit = dir.path().join("reference.safetensors");

        let config = QaConfig {
            safetensors_path: Some(explicit.clone()),
            ..QaConfig::default()
        };

        let result = super::resolve_safetensors_path(&gguf, &config, Duration::from_millis(0));

        match result {
            Ok(p) => assert_eq!(
                p, explicit,
                "an explicit --safetensors-path must be returned verbatim so the \
                 parity comparison actually runs (a real divergence must still FAIL, \
                 not be masked by the absent-reference SKIP)"
            ),
            Err(gate) => panic!(
                "an explicit reference must NOT short-circuit to SKIP/FAIL here; \
                 got skipped={}, passed={}, msg={}",
                gate.skipped, gate.passed, gate.message
            ),
        }
    }

    // ========================================================================
    // PMAT-QA-FMTPARITY-DECODE-001 — the gate must compare DECODE, not one
    // prefill argmax.
    // ========================================================================

    /// The anti-theater assertion. The gate this replaced ran one forward per
    /// format and compared a single final-position argmax; a cached-decode
    /// divergence at step 32 was structurally invisible to it. If this floor is
    /// ever lowered, the gate silently reverts to that shape while continuing
    /// to report itself as cross-format parity.
    #[test]
    #[cfg(feature = "inference")]
    fn format_parity_decode_floor_is_at_least_64_steps() {
        assert!(
            super::FORMAT_PARITY_MIN_DECODE_STEPS >= 64,
            "the cross-format gate must compare at least 64 decode steps; \
             found {}. Below this the gate cannot observe the cache-path \
             divergence class it exists to catch.",
            super::FORMAT_PARITY_MIN_DECODE_STEPS
        );
    }

    /// `--max-tokens` must not be able to shrink the gate back below the floor.
    #[test]
    #[cfg(feature = "inference")]
    fn max_tokens_cannot_shrink_decode_below_the_floor() {
        for max_tokens in [0usize, 1, 8, 32, 63] {
            let steps = max_tokens.max(super::FORMAT_PARITY_MIN_DECODE_STEPS);
            assert!(
                steps >= 64,
                "--max-tokens {max_tokens} shrank the parity decode to {steps} steps"
            );
        }
        // …but a LARGER request is still honored.
        assert_eq!(128usize.max(super::FORMAT_PARITY_MIN_DECODE_STEPS), 128);
    }

    #[test]
    #[cfg(feature = "inference")]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![0.5f32, -1.25, 3.0, 0.0];
        let c = super::cosine_similarity(&v, &v);
        assert!((c - 1.0).abs() < 1e-6, "expected 1.0, got {c}");
    }

    #[test]
    #[cfg(feature = "inference")]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(super::cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "inference")]
    fn cosine_similarity_mismatched_lengths_is_zero_not_panic() {
        assert_eq!(super::cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(super::cosine_similarity(&[], &[]), 0.0);
    }

    /// A clean run over the full step budget passes, and reports the budget as
    /// the threshold so the summary table shows N/N rather than a token id.
    #[test]
    #[cfg(feature = "inference")]
    fn decode_parity_all_steps_agreed_passes() {
        let report = super::DecodeParityReport {
            steps: 64,
            agreed: 64,
            near_ties: 0,
            first_divergence: None,
        };
        let gate = super::compare_decode_parity(&report, Duration::from_millis(1));
        assert!(gate.passed, "message was: {}", gate.message);
        assert!(!gate.skipped);
        assert_eq!(gate.value, Some(64.0));
        assert_eq!(gate.threshold, Some(64.0));
    }

    /// A near-tie is an exemption, not a failure: the top-1 differed but the
    /// distributions were the same vector to within FP noise.
    #[test]
    #[cfg(feature = "inference")]
    fn decode_parity_near_ties_do_not_fail_the_gate() {
        let report = super::DecodeParityReport {
            steps: 64,
            agreed: 61,
            near_ties: 3,
            first_divergence: None,
        };
        let gate = super::compare_decode_parity(&report, Duration::from_millis(1));
        assert!(gate.passed, "message was: {}", gate.message);
        assert_eq!(gate.value, Some(64.0));
    }

    /// The RED-turning case, and the whole point of the change: a divergence
    /// that appears only after many decode steps must FAIL. The single-argmax
    /// gate this replaced would have passed this scenario, because step 0
    /// agreed.
    #[test]
    #[cfg(feature = "inference")]
    fn decode_parity_late_divergence_fails_the_gate() {
        let report = super::DecodeParityReport {
            steps: 64,
            agreed: 32,
            near_ties: 0,
            first_divergence: Some((32, 100, 200, 0.41)),
        };
        let gate = super::compare_decode_parity(&report, Duration::from_millis(1));
        assert!(
            !gate.passed,
            "a divergence at decode step 32 MUST fail the gate; got PASS with: {}",
            gate.message
        );
        assert!(!gate.skipped, "a divergence is a FAIL, never a SKIP");
        assert!(
            gate.message.contains("step 32"),
            "the message must name the diverging step so the failure is actionable; got: {}",
            gate.message
        );
    }
}
