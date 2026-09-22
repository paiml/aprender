// `apr qa --json` must never exit having written zero bytes (#3842).
//
// THE DEFECT, measured 2026-09-22 on gx10: `apr qa --json` on qwen35-27b-q4km ran
// 78 seconds of GPU work and its own stderr recorded `F2 guard: passed in 78442 ms
// on 20 positions` — then wrote NOTHING, because `run_qa(...)?` propagated past the
// `if json` block. `model_ladder.sh` appended the empty result as an empty row, the
// receipt assembler dropped the empty line, and a RED REQUIRED RUNG vanished from
// its own receipt while the red counter kept counting: `red: 3`, two reds in rows.
//
// A gate that produces no document is worse than one that fails. A failure is
// evidence; an absence is not, and no per-row judging can find a row that was never
// written.
//
// WHY THE EXISTING COVERAGE DID NOT CATCH IT: `test_run_with_json_output` drives
// `run(..., json = true, ...)` on an invalid GGUF — exactly this path — and asserts
// only `result.is_err()`. The test named for the output never looked at the output.
// These tests assert on the DOCUMENT.
#[cfg(test)]
mod qa_json_never_empty_tests {
    use super::*;

    fn a_report() -> QaReport {
        QaReport {
            model: "m.gguf".to_string(),
            passed: true,
            gates: vec![],
            gates_executed: 1,
            gates_skipped: 0,
            total_duration_ms: 5,
            timestamp: "2026-09-22T00:00:00Z".to_string(),
            summary: "ok".to_string(),
            system_info: None,
        }
    }

    /// The invariant, stated directly: no report yields an empty document.
    #[test]
    fn the_json_document_is_never_empty() {
        for report in [
            a_report(),
            qa_report_for_error(
                std::path::Path::new("/models/qwen35-27b.gguf"),
                &CliError::ValidationFailed("boom".to_string()),
            ),
            QaReport {
                model: String::new(),
                summary: String::new(),
                timestamp: String::new(),
                ..a_report()
            },
        ] {
            let doc = qa_json_document(&report);
            assert!(
                !doc.trim().is_empty(),
                "qa_json_document produced an empty document"
            );
            serde_json::from_str::<serde_json::Value>(&doc)
                .expect("the emitted document must parse as JSON");
        }
    }

    /// The error path emits a document, and it says the run did not complete.
    /// MUTANT: drop the `emit_qa_json` call from the `Err` arm in `run` and the
    /// production symptom returns — 0 bytes on stdout with a non-zero exit.
    #[test]
    fn the_error_report_names_the_failure_and_is_not_silently_passing() {
        let e = CliError::ValidationFailed("F2 guard passed then no report".to_string());
        let report = qa_report_for_error(std::path::Path::new("/models/x.gguf"), &e);

        assert!(!report.passed, "an incomplete run must not read as passed");
        assert_eq!(report.model, "/models/x.gguf");

        // A FAILED gate, not an empty list: an empty `gates` reads as "nothing
        // failed" to anything counting failures, which is the same ambiguity that
        // let a missing row look like an absent model rather than a red one.
        assert_eq!(report.gates.len(), 1, "expected exactly one failure gate");
        assert_eq!(report.gates[0].name, "qa_run");
        assert!(!report.gates[0].passed);
        assert!(
            report.gates[0].message.contains("did not complete"),
            "gate message must say the run did not complete, got: {}",
            report.gates[0].message
        );
        assert!(
            report.summary.contains("F2 guard passed then no report"),
            "the summary must carry the underlying error, got: {}",
            report.summary
        );

        let doc = qa_json_document(&report);
        let v: serde_json::Value = serde_json::from_str(&doc).expect("parses");
        assert_eq!(v["passed"], serde_json::json!(false));
        assert_eq!(v["gates"][0]["name"], serde_json::json!("qa_run"));
    }

    /// The last-resort document is built without serde, because it exists for the
    /// case where serde did not work. Reaching for the thing that just failed is
    /// how a fallback becomes decoration.
    #[test]
    fn the_escaper_produces_parseable_json_for_hostile_input() {
        for raw in [
            "plain",
            "has \"quotes\"",
            "back\\slash",
            "new\nline\ttab\r",
            "control\u{0}\u{1f}chars",
            "unicode ✓ é",
        ] {
            let escaped = json_escaped(raw);
            let parsed: String = serde_json::from_str(&escaped)
                .unwrap_or_else(|e| panic!("{escaped} did not parse: {e}"));
            assert_eq!(parsed, raw, "escaping must round-trip");
        }
    }

    /// A report that serializes normally is passed through unchanged, so the
    /// fallback never shadows a good document.
    #[test]
    fn a_serializable_report_is_emitted_verbatim_not_via_the_fallback() {
        let report = a_report();
        let doc = qa_json_document(&report);
        let v: serde_json::Value = serde_json::from_str(&doc).expect("parses");
        assert_eq!(v["passed"], serde_json::json!(true));
        assert_eq!(v["gates_executed"], serde_json::json!(1));
        assert!(
            !doc.contains("could not serialize"),
            "a serializable report must not take the fallback path"
        );
    }
}
