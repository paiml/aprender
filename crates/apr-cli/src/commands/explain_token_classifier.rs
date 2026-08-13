//! Token-selection explanation classifier (CRUX-F-19).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-19-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! an already-captured `apr explain --format jsonl` body (one JSON object per
//! line, each describing a sampled token plus its candidate list):
//!
//!   * `classify_schema` — every line parses as a JSON object with the
//!     5 required keys (`step`, `sampled_id`, `candidates`, optional
//!     `token_id`, optional `token_str`); each candidate has
//!     `token_id`, `pre_prob`, `post_prob`, `rank`.
//!   * `classify_probs_normalize` — Σ post_prob across each step's
//!     candidates ≈ 1.0 within 1e-5.
//!   * `classify_sampled_in_candidates` — `sampled_id` appears in the
//!     candidates list for every step (the sampler cannot return a
//!     token that wasn't in the considered set).
//!   * `classify_greedy_picks_argmax` — when the caller asserts the
//!     trace was produced under `--temp 0 --top-k 1`, every step's
//!     `sampled_id` equals the argmax of `pre_prob` across candidates.
//!
//! Full discharge blocks on a live `apr explain` emitter wired to the
//! sampler — tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Outcome of `classify_schema`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExplainSchemaOutcome {
    Ok {
        line_count: usize,
    },
    Empty,
    LineNotJson {
        line_no: usize,
        message: String,
    },
    LineNotAnObject {
        line_no: usize,
    },
    LineMissingField {
        line_no: usize,
        field: &'static str,
    },
    CandidatesNotArray {
        line_no: usize,
    },
    CandidatesEmpty {
        line_no: usize,
    },
    CandidateNotAnObject {
        line_no: usize,
        index: usize,
    },
    CandidateMissingField {
        line_no: usize,
        index: usize,
        field: &'static str,
    },
}

/// Outcome of `classify_probs_normalize`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExplainProbsOutcome {
    Ok,
    NotNormalized {
        line_no: usize,
        step: i64,
        sum: f64,
        tolerance: f64,
    },
}

/// Outcome of `classify_sampled_in_candidates`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExplainSampledOutcome {
    Ok,
    SampledNotInCandidates {
        line_no: usize,
        step: i64,
        sampled_id: i64,
    },
}

/// Outcome of `classify_greedy_picks_argmax`.
#[derive(Debug, Clone, PartialEq)]
pub enum ExplainGreedyOutcome {
    Ok,
    NotArgmax {
        line_no: usize,
        step: i64,
        sampled_id: i64,
        sampled_pre_prob: f64,
        argmax_id: i64,
        argmax_pre_prob: f64,
    },
}

/// Parse a JSONL body into a list of (line_no, line, parsed) tuples.
/// Returns the schema outcome on the first error encountered.
fn parse_jsonl(body: &str) -> Result<Vec<(usize, Value)>, ExplainSchemaOutcome> {
    let mut out = Vec::new();
    let mut nonblank = 0usize;
    for (idx, raw) in body.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        nonblank += 1;
        let v: Value =
            serde_json::from_str(trimmed).map_err(|e| ExplainSchemaOutcome::LineNotJson {
                line_no,
                message: e.to_string(),
            })?;
        if !v.is_object() {
            return Err(ExplainSchemaOutcome::LineNotAnObject { line_no });
        }
        out.push((line_no, v));
    }
    if nonblank == 0 {
        return Err(ExplainSchemaOutcome::Empty);
    }
    Ok(out)
}

/// Schema gate: every JSONL line is an object with `step`, `sampled_id`, and
/// a non-empty `candidates` array of `{token_id, pre_prob, post_prob, rank}`.
pub fn classify_schema(body: &str) -> ExplainSchemaOutcome {
    let parsed = match parse_jsonl(body) {
        Ok(p) => p,
        Err(e) => return e,
    };
    for (line_no, v) in &parsed {
        for f in ["step", "sampled_id", "candidates"] {
            if v.get(f).is_none() {
                return ExplainSchemaOutcome::LineMissingField {
                    line_no: *line_no,
                    field: match f {
                        "step" => "step",
                        "sampled_id" => "sampled_id",
                        _ => "candidates",
                    },
                };
            }
        }
        let Some(cands) = v.get("candidates").and_then(Value::as_array) else {
            return ExplainSchemaOutcome::CandidatesNotArray { line_no: *line_no };
        };
        if cands.is_empty() {
            return ExplainSchemaOutcome::CandidatesEmpty { line_no: *line_no };
        }
        for (i, c) in cands.iter().enumerate() {
            if !c.is_object() {
                return ExplainSchemaOutcome::CandidateNotAnObject {
                    line_no: *line_no,
                    index: i,
                };
            }
            for f in ["token_id", "pre_prob", "post_prob", "rank"] {
                if c.get(f).is_none() {
                    return ExplainSchemaOutcome::CandidateMissingField {
                        line_no: *line_no,
                        index: i,
                        field: match f {
                            "token_id" => "token_id",
                            "pre_prob" => "pre_prob",
                            "post_prob" => "post_prob",
                            _ => "rank",
                        },
                    };
                }
            }
        }
    }
    ExplainSchemaOutcome::Ok {
        line_count: parsed.len(),
    }
}

/// Verify Σ post_prob ≈ 1.0 (within `tolerance`) per step.
pub fn classify_probs_normalize(body: &str, tolerance: f64) -> ExplainProbsOutcome {
    let parsed = match parse_jsonl(body) {
        Ok(p) => p,
        Err(_) => return ExplainProbsOutcome::Ok, // schema gate handles malformed input
    };
    for (line_no, v) in parsed {
        let step = v.get("step").and_then(Value::as_i64).unwrap_or(0);
        let Some(cands) = v.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        let sum: f64 = cands
            .iter()
            .filter_map(|c| c.get("post_prob").and_then(Value::as_f64))
            .sum();
        if (sum - 1.0).abs() > tolerance {
            return ExplainProbsOutcome::NotNormalized {
                line_no,
                step,
                sum,
                tolerance,
            };
        }
    }
    ExplainProbsOutcome::Ok
}

/// Verify `sampled_id` is in `candidates[*].token_id` for every step.
pub fn classify_sampled_in_candidates(body: &str) -> ExplainSampledOutcome {
    let parsed = match parse_jsonl(body) {
        Ok(p) => p,
        Err(_) => return ExplainSampledOutcome::Ok,
    };
    for (line_no, v) in parsed {
        let step = v.get("step").and_then(Value::as_i64).unwrap_or(0);
        let sampled = v.get("sampled_id").and_then(Value::as_i64).unwrap_or(-1);
        let Some(cands) = v.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        let found = cands
            .iter()
            .filter_map(|c| c.get("token_id").and_then(Value::as_i64))
            .any(|id| id == sampled);
        if !found {
            return ExplainSampledOutcome::SampledNotInCandidates {
                line_no,
                step,
                sampled_id: sampled,
            };
        }
    }
    ExplainSampledOutcome::Ok
}

/// Verify that under greedy decoding (`temp == 0`, `top_k == 1`) the
/// sampled token is the argmax of `pre_prob`.
pub fn classify_greedy_picks_argmax(body: &str) -> ExplainGreedyOutcome {
    let parsed = match parse_jsonl(body) {
        Ok(p) => p,
        Err(_) => return ExplainGreedyOutcome::Ok,
    };
    for (line_no, v) in parsed {
        let step = v.get("step").and_then(Value::as_i64).unwrap_or(0);
        let sampled = v.get("sampled_id").and_then(Value::as_i64).unwrap_or(-1);
        let Some(cands) = v.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        let mut best: Option<(i64, f64)> = None;
        let mut sampled_pre: f64 = f64::NAN;
        for c in cands {
            let id = c.get("token_id").and_then(Value::as_i64).unwrap_or(-1);
            let pre = c.get("pre_prob").and_then(Value::as_f64).unwrap_or(0.0);
            if id == sampled {
                sampled_pre = pre;
            }
            match best {
                None => best = Some((id, pre)),
                Some((_, b)) if pre > b => best = Some((id, pre)),
                _ => {}
            }
        }
        let Some((argmax_id, argmax_pre)) = best else {
            continue;
        };
        if argmax_id != sampled {
            return ExplainGreedyOutcome::NotArgmax {
                line_no,
                step,
                sampled_id: sampled,
                sampled_pre_prob: sampled_pre,
                argmax_id,
                argmax_pre_prob: argmax_pre,
            };
        }
    }
    ExplainGreedyOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_body() -> String {
        // Two steps with normalized probs and the sampled id present.
        let l0 = r#"{"step":0,"sampled_id":7,"candidates":[
            {"token_id":7,"pre_prob":0.6,"post_prob":0.7,"rank":0},
            {"token_id":3,"pre_prob":0.3,"post_prob":0.2,"rank":1},
            {"token_id":5,"pre_prob":0.1,"post_prob":0.1,"rank":2}
        ]}"#;
        let l1 = r#"{"step":1,"sampled_id":3,"candidates":[
            {"token_id":3,"pre_prob":0.5,"post_prob":0.5,"rank":0},
            {"token_id":7,"pre_prob":0.4,"post_prob":0.5,"rank":1}
        ]}"#;
        // Strip the embedded newlines/indentation so each line is one record.
        let mut out = String::new();
        out.push_str(&l0.split_whitespace().collect::<String>());
        out.push('\n');
        out.push_str(&l1.split_whitespace().collect::<String>());
        out.push('\n');
        out
    }

    #[test]
    fn schema_ok_on_good_body() {
        let out = classify_schema(&good_body());
        assert_eq!(out, ExplainSchemaOutcome::Ok { line_count: 2 });
    }

    #[test]
    fn schema_rejects_empty_body() {
        assert_eq!(classify_schema(""), ExplainSchemaOutcome::Empty);
    }

    #[test]
    fn schema_rejects_non_json_line() {
        let body = "not json\n";
        assert!(matches!(
            classify_schema(body),
            ExplainSchemaOutcome::LineNotJson { line_no: 1, .. }
        ));
    }

    #[test]
    fn schema_rejects_non_object_line() {
        let body = "[1,2,3]\n";
        assert_eq!(
            classify_schema(body),
            ExplainSchemaOutcome::LineNotAnObject { line_no: 1 }
        );
    }

    #[test]
    fn schema_rejects_missing_field() {
        let body =
            r#"{"step":0,"candidates":[{"token_id":1,"pre_prob":1.0,"post_prob":1.0,"rank":0}]}"#;
        assert!(matches!(
            classify_schema(body),
            ExplainSchemaOutcome::LineMissingField {
                line_no: 1,
                field: "sampled_id"
            }
        ));
    }

    #[test]
    fn schema_rejects_candidate_missing_field() {
        let body =
            r#"{"step":0,"sampled_id":1,"candidates":[{"token_id":1,"pre_prob":1.0,"rank":0}]}"#;
        assert!(matches!(
            classify_schema(body),
            ExplainSchemaOutcome::CandidateMissingField {
                line_no: 1,
                index: 0,
                field: "post_prob"
            }
        ));
    }

    #[test]
    fn schema_rejects_empty_candidates() {
        let body = r#"{"step":0,"sampled_id":1,"candidates":[]}"#;
        assert!(matches!(
            classify_schema(body),
            ExplainSchemaOutcome::CandidatesEmpty { line_no: 1 }
        ));
    }

    #[test]
    fn probs_normalize_ok_on_good_body() {
        assert_eq!(
            classify_probs_normalize(&good_body(), 1e-5),
            ExplainProbsOutcome::Ok
        );
    }

    #[test]
    fn probs_normalize_reports_violation() {
        let body = r#"{"step":0,"sampled_id":1,"candidates":[
            {"token_id":1,"pre_prob":0.5,"post_prob":0.6,"rank":0},
            {"token_id":2,"pre_prob":0.5,"post_prob":0.6,"rank":1}
        ]}"#;
        // Strip whitespace so it's one JSONL line.
        let body = body.split_whitespace().collect::<String>();
        assert!(matches!(
            classify_probs_normalize(&body, 1e-5),
            ExplainProbsOutcome::NotNormalized { .. }
        ));
    }

    #[test]
    fn sampled_in_candidates_ok_on_good_body() {
        assert_eq!(
            classify_sampled_in_candidates(&good_body()),
            ExplainSampledOutcome::Ok
        );
    }

    #[test]
    fn sampled_in_candidates_reports_missing() {
        let body = r#"{"step":0,"sampled_id":99,"candidates":[
            {"token_id":1,"pre_prob":1.0,"post_prob":1.0,"rank":0}
        ]}"#;
        let body = body.split_whitespace().collect::<String>();
        assert!(matches!(
            classify_sampled_in_candidates(&body),
            ExplainSampledOutcome::SampledNotInCandidates { sampled_id: 99, .. }
        ));
    }

    #[test]
    fn greedy_picks_argmax_ok_on_good_body() {
        assert_eq!(
            classify_greedy_picks_argmax(&good_body()),
            ExplainGreedyOutcome::Ok
        );
    }

    #[test]
    fn greedy_picks_argmax_reports_non_argmax() {
        // sampled_id=3 but argmax pre_prob is token 7
        let body = r#"{"step":0,"sampled_id":3,"candidates":[
            {"token_id":7,"pre_prob":0.9,"post_prob":1.0,"rank":0},
            {"token_id":3,"pre_prob":0.1,"post_prob":0.0,"rank":1}
        ]}"#;
        let body = body.split_whitespace().collect::<String>();
        match classify_greedy_picks_argmax(&body) {
            ExplainGreedyOutcome::NotArgmax {
                sampled_id,
                argmax_id,
                ..
            } => {
                assert_eq!(sampled_id, 3);
                assert_eq!(argmax_id, 7);
            }
            other => panic!("expected NotArgmax, got {other:?}"),
        }
    }
}
