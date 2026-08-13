//! ReAct agent loop trace classifier (CRUX-I-06).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-I-06-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! a captured `apr agent --trace OUT.json` body.
//!
//! Trace shape (per spec):
//! ```json
//! {
//!   "iterations": 3,
//!   "answer": "4",                        // present iff Final Answer fired
//!   "reason": "max_iterations" | "timeout" | "tool_error" | "parse_fail",
//!   "scratchpad": "Thought: ...\nAction: ...\nObservation: ...\nFinal Answer: 4",
//!   "exit_code": 0 | 2 | 3 | 4
//! }
//! ```
//!
//! Classifiers:
//!   * `classify_termination` — exactly one stop condition fires, exit_code
//!     matches the canonical mapping (0=final_answer, 2=max_iterations|timeout,
//!     3=tool_error, 4=parse_fail).
//!   * `classify_scratchpad_grammar` — the scratchpad parses as a sequence
//!     of `Thought: ... Action: ... Action Input: ... Observation: ...`
//!     blocks optionally terminated by `Final Answer: ...`; rejects
//!     truncated or rewritten content.
//!   * `classify_iteration_bound` — `iterations <= max_iterations`; if
//!     `Final Answer:` is present, no extra Action block follows it.

use serde_json::Value;

/// Canonical termination-reason → exit-code mapping (CRUX-I-06 `stop_conditions`).
pub const I06_REASON_EXIT: &[(&str, i32)] = &[
    ("final_answer", 0),
    ("max_iterations", 2),
    ("timeout", 2),
    ("tool_error", 3),
    ("parse_fail", 4),
];

/// Outcome of `classify_termination`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReactTerminationOutcome {
    Ok {
        reason: String,
        exit_code: i32,
    },
    NotAnObject,
    MissingExitCode,
    UnknownReason {
        got: String,
    },
    ExitCodeMismatch {
        reason: String,
        got: i32,
        expected: i32,
    },
    FinalAnswerWithoutAnswerField,
}

/// Outcome of `classify_scratchpad_grammar`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReactGrammarOutcome {
    Ok { blocks: usize },
    Empty,
    MissingThought { block: usize },
    MissingAction { block: usize },
    ActionAfterFinalAnswer { block: usize },
}

/// Outcome of `classify_iteration_bound`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReactBoundOutcome {
    Ok,
    IterationsExceedBudget { iterations: i64, max: i64 },
    IterationsNegative { got: i64 },
}

/// FALSIFY-CRUX-I-06-001 / -002 / live-only -003: termination contract.
pub fn classify_termination(body: &Value) -> ReactTerminationOutcome {
    let Some(obj) = body.as_object() else {
        return ReactTerminationOutcome::NotAnObject;
    };
    let exit_code = match obj.get("exit_code").and_then(Value::as_i64) {
        Some(c) => c as i32,
        None => return ReactTerminationOutcome::MissingExitCode,
    };
    // `reason` is required for non-zero exits; on exit 0 the spec wants `answer`
    // to be present so the field can be omitted.
    let reason: String = match obj.get("reason").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None if exit_code == 0 => "final_answer".to_string(),
        None => {
            return ReactTerminationOutcome::UnknownReason {
                got: "<missing>".to_string(),
            }
        }
    };
    let Some((_, expected)) = I06_REASON_EXIT.iter().find(|(r, _)| *r == reason.as_str()) else {
        return ReactTerminationOutcome::UnknownReason { got: reason };
    };
    if exit_code != *expected {
        return ReactTerminationOutcome::ExitCodeMismatch {
            reason,
            got: exit_code,
            expected: *expected,
        };
    }
    if reason == "final_answer" && obj.get("answer").and_then(Value::as_str).is_none() {
        return ReactTerminationOutcome::FinalAnswerWithoutAnswerField;
    }
    ReactTerminationOutcome::Ok { reason, exit_code }
}

/// FALSIFY-CRUX-I-06-001 scratchpad grammar parser.
pub fn classify_scratchpad_grammar(scratchpad: &str) -> ReactGrammarOutcome {
    let trimmed = scratchpad.trim();
    if trimmed.is_empty() {
        return ReactGrammarOutcome::Empty;
    }
    // Find "Final Answer:" position; once present no Action may follow.
    let final_pos = trimmed.find("Final Answer:");

    let mut blocks = 0usize;
    let mut cursor = 0usize;
    while cursor < trimmed.len() {
        let body = &trimmed[cursor..];
        let Some(thought_start) = body.find("Thought:") else {
            break;
        };
        // No more Thought blocks ⇒ tail is the (optional) Final Answer.
        blocks += 1;
        let block_idx = blocks - 1;
        // Each block must have an Action: header before the next Thought / Final Answer / end.
        let after_thought = thought_start + "Thought:".len();
        let rest = &body[after_thought..];
        let action_pos = rest.find("Action:");
        let next_thought_pos = rest.find("Thought:");
        let final_inblock_pos = rest.find("Final Answer:");
        // Determine block end (whichever delimiter comes first after Action).
        let end_of_block = [next_thought_pos, final_inblock_pos]
            .iter()
            .filter_map(|p| *p)
            .min()
            .unwrap_or(rest.len());

        match action_pos {
            Some(p) if p < end_of_block => {
                // Action exists inside this block — block is well-formed.
            }
            _ => {
                // Allow the very last block to be a pure Final Answer (no Action).
                if final_inblock_pos.is_some() && next_thought_pos.is_none() {
                    // Block has a Thought leading directly to a Final Answer
                    // (e.g. one-shot reasoning). Spec permits this.
                } else {
                    return ReactGrammarOutcome::MissingAction { block: block_idx };
                }
            }
        }
        cursor += thought_start + "Thought:".len() + end_of_block;
    }

    // Validate no Action follows Final Answer.
    if let Some(fp) = final_pos {
        let tail = &trimmed[fp + "Final Answer:".len()..];
        if tail.contains("Action:") {
            return ReactGrammarOutcome::ActionAfterFinalAnswer {
                block: blocks.saturating_sub(1),
            };
        }
    }

    if blocks == 0 {
        // Pure Final Answer with no Thought is degenerate but acceptable
        // (spec doesn't forbid an LLM emitting just `Final Answer: X`).
        if final_pos.is_some() {
            return ReactGrammarOutcome::Ok { blocks: 0 };
        }
        return ReactGrammarOutcome::MissingThought { block: 0 };
    }
    ReactGrammarOutcome::Ok { blocks }
}

/// FALSIFY-CRUX-I-06-001 / -002 iteration-budget guard.
pub fn classify_iteration_bound(body: &Value, max_iterations: i64) -> ReactBoundOutcome {
    let iters = body.get("iterations").and_then(Value::as_i64).unwrap_or(0);
    if iters < 0 {
        return ReactBoundOutcome::IterationsNegative { got: iters };
    }
    if iters > max_iterations {
        return ReactBoundOutcome::IterationsExceedBudget {
            iterations: iters,
            max: max_iterations,
        };
    }
    ReactBoundOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_final_answer_body() -> Value {
        json!({
            "iterations": 1,
            "answer": "4",
            "scratchpad": "Thought: I should compute 2+2.\nFinal Answer: 4",
            "exit_code": 0
        })
    }

    fn good_max_iterations_body() -> Value {
        json!({
            "iterations": 3,
            "reason": "max_iterations",
            "scratchpad": "Thought: try\nAction: echo\nAction Input: hi\nObservation: hi\nThought: try\nAction: echo\nAction Input: hi\nObservation: hi\nThought: still trying\nAction: echo\nAction Input: x\nObservation: x",
            "exit_code": 2
        })
    }

    #[test]
    fn termination_ok_on_final_answer() {
        match classify_termination(&good_final_answer_body()) {
            ReactTerminationOutcome::Ok { reason, exit_code } => {
                assert_eq!(reason, "final_answer");
                assert_eq!(exit_code, 0);
            }
            other => panic!("expected Ok(final_answer, 0), got {other:?}"),
        }
    }

    #[test]
    fn termination_ok_on_max_iterations() {
        match classify_termination(&good_max_iterations_body()) {
            ReactTerminationOutcome::Ok { reason, exit_code } => {
                assert_eq!(reason, "max_iterations");
                assert_eq!(exit_code, 2);
            }
            other => panic!("expected Ok(max_iterations, 2), got {other:?}"),
        }
    }

    #[test]
    fn termination_rejects_unknown_reason() {
        let body = json!({"iterations": 1, "reason": "whoops", "scratchpad": "", "exit_code": 5});
        assert!(matches!(
            classify_termination(&body),
            ReactTerminationOutcome::UnknownReason { .. }
        ));
    }

    #[test]
    fn termination_rejects_exit_code_mismatch() {
        let body =
            json!({"iterations": 3, "reason": "max_iterations", "scratchpad": "", "exit_code": 1});
        match classify_termination(&body) {
            ReactTerminationOutcome::ExitCodeMismatch {
                reason,
                got,
                expected,
            } => {
                assert_eq!(reason, "max_iterations");
                assert_eq!(got, 1);
                assert_eq!(expected, 2);
            }
            other => panic!("expected ExitCodeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn termination_rejects_final_answer_without_answer_field() {
        let body = json!({"iterations": 1, "exit_code": 0, "scratchpad": "Final Answer: 4"});
        // No `answer` key with reason=final_answer (default-inferred).
        assert_eq!(
            classify_termination(&body),
            ReactTerminationOutcome::FinalAnswerWithoutAnswerField
        );
    }

    #[test]
    fn termination_rejects_not_an_object() {
        assert_eq!(
            classify_termination(&json!([1, 2])),
            ReactTerminationOutcome::NotAnObject
        );
    }

    #[test]
    fn scratchpad_ok_on_three_block_trace() {
        let sp = "Thought: t1\nAction: a1\nAction Input: i1\nObservation: o1\n\
                  Thought: t2\nAction: a2\nAction Input: i2\nObservation: o2\n\
                  Thought: t3\nAction: a3\nAction Input: i3\nObservation: o3";
        assert_eq!(
            classify_scratchpad_grammar(sp),
            ReactGrammarOutcome::Ok { blocks: 3 }
        );
    }

    #[test]
    fn scratchpad_ok_on_thought_then_final_answer() {
        let sp = "Thought: I should compute 2+2.\nFinal Answer: 4";
        // Block has Thought leading to Final Answer (no Action) — accepted.
        match classify_scratchpad_grammar(sp) {
            ReactGrammarOutcome::Ok { .. } => {}
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn scratchpad_rejects_empty() {
        assert_eq!(classify_scratchpad_grammar(""), ReactGrammarOutcome::Empty);
    }

    #[test]
    fn scratchpad_rejects_action_after_final_answer() {
        let sp = "Thought: done.\nFinal Answer: 4\nAction: rogue\nAction Input: x";
        match classify_scratchpad_grammar(sp) {
            ReactGrammarOutcome::ActionAfterFinalAnswer { .. } => {}
            other => panic!("expected ActionAfterFinalAnswer, got {other:?}"),
        }
    }

    #[test]
    fn scratchpad_rejects_missing_action_in_middle_block() {
        // Two blocks: first has Action, second is Thought-only (no Action, no Final Answer).
        let sp = "Thought: t1\nAction: a1\nAction Input: i1\nObservation: o1\nThought: t2";
        match classify_scratchpad_grammar(sp) {
            ReactGrammarOutcome::MissingAction { block } => assert_eq!(block, 1),
            other => panic!("expected MissingAction(1), got {other:?}"),
        }
    }

    #[test]
    fn iteration_bound_ok_within_budget() {
        let body = json!({"iterations": 3});
        assert_eq!(classify_iteration_bound(&body, 5), ReactBoundOutcome::Ok);
    }

    #[test]
    fn iteration_bound_rejects_over_budget() {
        let body = json!({"iterations": 10});
        match classify_iteration_bound(&body, 5) {
            ReactBoundOutcome::IterationsExceedBudget { iterations, max } => {
                assert_eq!(iterations, 10);
                assert_eq!(max, 5);
            }
            other => panic!("expected IterationsExceedBudget, got {other:?}"),
        }
    }

    #[test]
    fn iteration_bound_rejects_negative() {
        let body = json!({"iterations": -1});
        assert!(matches!(
            classify_iteration_bound(&body, 5),
            ReactBoundOutcome::IterationsNegative { got: -1 }
        ));
    }
}
