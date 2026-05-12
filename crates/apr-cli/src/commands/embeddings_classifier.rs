//! CRUX-C-13 — /v1/embeddings endpoint classifier (PARTIAL_ALGORITHM_LEVEL)
//!
//! Pure algorithm-level discharge of the FALSIFY gates in
//! `contracts/crux-C-13-v1.yaml`:
//!   * FALSIFY-001 — response shape: len(data)==len(input), every vector
//!     length == hidden_size, indices are 0..N-1 in order.
//!   * FALSIFY-002 — determinism: cosine(v, v) ≈ 1.0 for non-zero v.
//!   * FALSIFY-003 — usage accounting: total == prompt, prompt >= 1.
//!   * FALSIFY-004 — CLI surface parses `--embeddings-enabled`.

/// OpenAI-style embeddings determinism bound.
pub const EMBEDDINGS_COSINE_TOLERANCE: f64 = 1e-6;

/// ─── FALSIFY-C-13-001 response shape ────────────────────────────────────

/// One row of `.data[]` as extracted from the JSON response.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRow<'a> {
    pub index: u64,
    pub embedding: &'a [f32],
}

/// Outcome of the response-shape classifier.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingsShapeOutcome {
    Ok {
        n_rows: usize,
    },
    RowCountMismatch {
        input_len: usize,
        data_len: usize,
    },
    VectorDimensionMismatch {
        row: usize,
        expected: usize,
        got: usize,
    },
    IndexOutOfOrder {
        row: usize,
        expected_index: u64,
        got_index: u64,
    },
}

/// Validate the response against `input_len` and `hidden_size`.
pub fn classify_embeddings_response_shape(
    input_len: usize,
    data: &[EmbeddingRow<'_>],
    hidden_size: usize,
) -> EmbeddingsShapeOutcome {
    if data.len() != input_len {
        return EmbeddingsShapeOutcome::RowCountMismatch {
            input_len,
            data_len: data.len(),
        };
    }
    for (i, row) in data.iter().enumerate() {
        if row.embedding.len() != hidden_size {
            return EmbeddingsShapeOutcome::VectorDimensionMismatch {
                row: i,
                expected: hidden_size,
                got: row.embedding.len(),
            };
        }
        if row.index != i as u64 {
            return EmbeddingsShapeOutcome::IndexOutOfOrder {
                row: i,
                expected_index: i as u64,
                got_index: row.index,
            };
        }
    }
    EmbeddingsShapeOutcome::Ok { n_rows: data.len() }
}

/// ─── FALSIFY-C-13-002 determinism / cosine ──────────────────────────────

/// Cosine similarity of two vectors; `None` on length mismatch, empty
/// input, or zero-norm on either side (ill-defined ratio).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

/// Outcome of the determinism classifier.
#[derive(Debug, Clone, PartialEq)]
pub enum DeterminismOutcome {
    Deterministic { cosine: f64 },
    NonDeterministic { cosine: f64, tolerance: f64 },
    InvalidInput,
}

/// Compare two embedding vectors produced by repeated calls with the
/// same input. Deterministic iff cosine ≥ 1.0 − tolerance (floating
/// point wobble is tolerated within the spec's 1e-6 window).
pub fn classify_determinism(v1: &[f32], v2: &[f32], tolerance: f64) -> DeterminismOutcome {
    match cosine_similarity(v1, v2) {
        None => DeterminismOutcome::InvalidInput,
        Some(c) if c >= 1.0 - tolerance => DeterminismOutcome::Deterministic { cosine: c },
        Some(c) => DeterminismOutcome::NonDeterministic {
            cosine: c,
            tolerance,
        },
    }
}

/// ─── FALSIFY-C-13-003 usage accounting ──────────────────────────────────

/// Outcome of the usage-token classifier.
#[derive(Debug, Clone, PartialEq)]
pub enum UsageOutcome {
    Ok { prompt: u64, total: u64 },
    TotalMismatchesPrompt { prompt: u64, total: u64 },
    PromptTokensZero,
}

/// Validate an OpenAI-style `usage` block: `total == prompt` and
/// `prompt >= 1`. Embeddings never produce completion tokens, so any
/// drift between total and prompt indicates mis-accounting that
/// downstream metering clients will incorrectly bill.
pub fn classify_usage_tokens(prompt: u64, total: u64) -> UsageOutcome {
    if prompt == 0 {
        return UsageOutcome::PromptTokensZero;
    }
    if total != prompt {
        return UsageOutcome::TotalMismatchesPrompt { prompt, total };
    }
    UsageOutcome::Ok { prompt, total }
}

/// ─── FALSIFY-C-13-004 CLI surface ───────────────────────────────────────

/// Outcome of the `--embeddings-enabled` argv parser.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingsFlagOutcome {
    Enabled,
    Disabled,
    MalformedFlag { raw: String },
}

/// Parse `apr serve` argv looking for the `--embeddings-enabled` flag.
/// Accepts bare (`--embeddings-enabled`) and `=` forms
/// (`--embeddings-enabled=true|false`). Absent ⇒ `Disabled` (default off,
/// as in llama.cpp/vllm).
pub fn parse_embeddings_flag(argv: &[&str]) -> EmbeddingsFlagOutcome {
    for arg in argv {
        if *arg == "--embeddings-enabled" {
            return EmbeddingsFlagOutcome::Enabled;
        }
        if let Some(val) = arg.strip_prefix("--embeddings-enabled=") {
            return match val.to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => EmbeddingsFlagOutcome::Enabled,
                "false" | "0" | "no" | "off" => EmbeddingsFlagOutcome::Disabled,
                _ => EmbeddingsFlagOutcome::MalformedFlag {
                    raw: (*arg).to_string(),
                },
            };
        }
    }
    EmbeddingsFlagOutcome::Disabled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(index: u64, v: &[f32]) -> EmbeddingRow<'_> {
        EmbeddingRow {
            index,
            embedding: v,
        }
    }

    // ─── shape classifier ─────────────────────────────────────────────────

    #[test]
    fn shape_ok_for_three_rows_of_hidden_size_4() {
        let a = [0.1f32; 4];
        let b = [0.2f32; 4];
        let c = [0.3f32; 4];
        let data = vec![row(0, &a), row(1, &b), row(2, &c)];
        assert_eq!(
            classify_embeddings_response_shape(3, &data, 4),
            EmbeddingsShapeOutcome::Ok { n_rows: 3 }
        );
    }

    #[test]
    fn shape_row_count_mismatch_is_reported() {
        let a = [0.0f32; 4];
        let data = vec![row(0, &a)];
        assert_eq!(
            classify_embeddings_response_shape(2, &data, 4),
            EmbeddingsShapeOutcome::RowCountMismatch {
                input_len: 2,
                data_len: 1,
            }
        );
    }

    #[test]
    fn shape_vector_dim_mismatch_is_reported_with_row() {
        let a = [0.0f32; 4];
        let b = [0.0f32; 3]; // wrong
        let data = vec![row(0, &a), row(1, &b)];
        assert_eq!(
            classify_embeddings_response_shape(2, &data, 4),
            EmbeddingsShapeOutcome::VectorDimensionMismatch {
                row: 1,
                expected: 4,
                got: 3,
            }
        );
    }

    #[test]
    fn shape_index_out_of_order_is_reported() {
        let a = [0.0f32; 4];
        let b = [0.0f32; 4];
        let data = vec![row(0, &a), row(5, &b)];
        assert_eq!(
            classify_embeddings_response_shape(2, &data, 4),
            EmbeddingsShapeOutcome::IndexOutOfOrder {
                row: 1,
                expected_index: 1,
                got_index: 5,
            }
        );
    }

    #[test]
    fn shape_empty_input_empty_data_is_ok() {
        let data: Vec<EmbeddingRow<'_>> = vec![];
        assert_eq!(
            classify_embeddings_response_shape(0, &data, 4),
            EmbeddingsShapeOutcome::Ok { n_rows: 0 }
        );
    }

    // ─── cosine_similarity ────────────────────────────────────────────────

    #[test]
    fn cosine_identical_nonzero_is_one() {
        let v = vec![1.0f32, 2.0, 3.0];
        let c = cosine_similarity(&v, &v).unwrap();
        assert!((c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &b).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn cosine_opposite_is_negative_one() {
        let a = vec![1.0f32, 2.0];
        let b = vec![-1.0f32, -2.0];
        assert!((cosine_similarity(&a, &b).unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_length_mismatch_is_none() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), None);
    }

    #[test]
    fn cosine_zero_norm_is_none() {
        let a = vec![0.0f32; 3];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), None);
        assert_eq!(cosine_similarity(&b, &a), None);
    }

    #[test]
    fn cosine_empty_vectors_is_none() {
        let a: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &a), None);
    }

    // ─── determinism classifier ───────────────────────────────────────────

    #[test]
    fn determinism_identical_vector_is_deterministic() {
        let v = vec![0.1f32, 0.2, 0.3, 0.4];
        match classify_determinism(&v, &v, EMBEDDINGS_COSINE_TOLERANCE) {
            DeterminismOutcome::Deterministic { cosine } => {
                assert!((cosine - 1.0).abs() < EMBEDDINGS_COSINE_TOLERANCE);
            }
            other => panic!("expected Deterministic, got {:?}", other),
        }
    }

    #[test]
    fn determinism_tiny_noise_within_tolerance_is_deterministic() {
        // 1e-7 perturbation ⇒ cosine drop < 1e-6 for a unit-norm vector.
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![1.0f32 - 1e-7, 0.0, 0.0];
        match classify_determinism(&v1, &v2, EMBEDDINGS_COSINE_TOLERANCE) {
            DeterminismOutcome::Deterministic { .. } => {}
            other => panic!("expected Deterministic, got {:?}", other),
        }
    }

    #[test]
    fn determinism_large_drift_is_non_deterministic() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        match classify_determinism(&v1, &v2, EMBEDDINGS_COSINE_TOLERANCE) {
            DeterminismOutcome::NonDeterministic { cosine, tolerance } => {
                assert!(cosine < 1.0 - tolerance);
                assert_eq!(tolerance, EMBEDDINGS_COSINE_TOLERANCE);
            }
            other => panic!("expected NonDeterministic, got {:?}", other),
        }
    }

    #[test]
    fn determinism_zero_vector_is_invalid() {
        let z = vec![0.0f32; 3];
        assert_eq!(
            classify_determinism(&z, &z, EMBEDDINGS_COSINE_TOLERANCE),
            DeterminismOutcome::InvalidInput
        );
    }

    #[test]
    fn determinism_length_mismatch_is_invalid() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(
            classify_determinism(&a, &b, EMBEDDINGS_COSINE_TOLERANCE),
            DeterminismOutcome::InvalidInput
        );
    }

    // ─── usage classifier ─────────────────────────────────────────────────

    #[test]
    fn usage_equal_prompt_and_total_is_ok() {
        assert_eq!(
            classify_usage_tokens(8, 8),
            UsageOutcome::Ok {
                prompt: 8,
                total: 8
            }
        );
    }

    #[test]
    fn usage_zero_prompt_is_rejected_even_if_total_matches() {
        assert_eq!(classify_usage_tokens(0, 0), UsageOutcome::PromptTokensZero);
    }

    #[test]
    fn usage_total_exceeds_prompt_is_mismatch() {
        assert_eq!(
            classify_usage_tokens(5, 6),
            UsageOutcome::TotalMismatchesPrompt {
                prompt: 5,
                total: 6
            }
        );
    }

    #[test]
    fn usage_total_below_prompt_is_mismatch() {
        assert_eq!(
            classify_usage_tokens(5, 3),
            UsageOutcome::TotalMismatchesPrompt {
                prompt: 5,
                total: 3
            }
        );
    }

    #[test]
    fn usage_classifier_is_deterministic() {
        assert_eq!(classify_usage_tokens(7, 7), classify_usage_tokens(7, 7));
    }

    // ─── flag parser ──────────────────────────────────────────────────────

    #[test]
    fn flag_bare_form_enables() {
        assert_eq!(
            parse_embeddings_flag(&["apr", "serve", "--embeddings-enabled"]),
            EmbeddingsFlagOutcome::Enabled
        );
    }

    #[test]
    fn flag_equals_true_enables() {
        assert_eq!(
            parse_embeddings_flag(&["apr", "serve", "--embeddings-enabled=true"]),
            EmbeddingsFlagOutcome::Enabled
        );
    }

    #[test]
    fn flag_equals_1_yes_on_enable() {
        for v in ["1", "yes", "on", "True", "YES"] {
            let arg = format!("--embeddings-enabled={}", v);
            assert_eq!(
                parse_embeddings_flag(&["apr", "serve", &arg]),
                EmbeddingsFlagOutcome::Enabled,
                "form {:?} should enable",
                arg
            );
        }
    }

    #[test]
    fn flag_equals_false_disables() {
        assert_eq!(
            parse_embeddings_flag(&["apr", "serve", "--embeddings-enabled=false"]),
            EmbeddingsFlagOutcome::Disabled
        );
    }

    #[test]
    fn flag_equals_zero_no_off_disable() {
        for v in ["0", "no", "off", "False", "NO"] {
            let arg = format!("--embeddings-enabled={}", v);
            assert_eq!(
                parse_embeddings_flag(&["apr", "serve", &arg]),
                EmbeddingsFlagOutcome::Disabled,
                "form {:?} should disable",
                arg
            );
        }
    }

    #[test]
    fn flag_equals_garbage_is_malformed() {
        match parse_embeddings_flag(&["apr", "serve", "--embeddings-enabled=maybe"]) {
            EmbeddingsFlagOutcome::MalformedFlag { raw } => {
                assert_eq!(raw, "--embeddings-enabled=maybe");
            }
            other => panic!("expected MalformedFlag, got {:?}", other),
        }
    }

    #[test]
    fn flag_absent_defaults_to_disabled() {
        assert_eq!(
            parse_embeddings_flag(&["apr", "serve", "--port", "8080"]),
            EmbeddingsFlagOutcome::Disabled
        );
    }

    #[test]
    fn flag_parser_is_deterministic() {
        let argv = &["apr", "serve", "--embeddings-enabled"];
        assert_eq!(parse_embeddings_flag(argv), parse_embeddings_flag(argv));
    }

    #[test]
    fn tolerance_constant_matches_spec() {
        assert_eq!(EMBEDDINGS_COSINE_TOLERANCE, 1e-6);
    }
}
