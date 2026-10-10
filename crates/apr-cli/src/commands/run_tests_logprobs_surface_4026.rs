// =============================================================================
// #4026: `--logprobs K` is refused by name on every surface that would drop it
// =============================================================================

/// Over every flag combination, K > 0 passes exactly where the machine document
/// is printed, and K = 0 passes everywhere. Pinned to `emits_machine_output`, so
/// the refusal and the arms of `print_run_output` cannot drift apart.
#[test]
fn logprobs_refusal_matches_the_machine_output_predicate() {
    for &stream in &[false, true] {
        for &benchmark in &[false, true] {
            for fmt in ["json", "text", "table"] {
                assert!(
                    refuse_logprobs_without_machine_output(0, stream, fmt, benchmark).is_ok(),
                    "K=0 refused at stream={stream} fmt={fmt} benchmark={benchmark}"
                );
                assert_eq!(
                    refuse_logprobs_without_machine_output(5, stream, fmt, benchmark).is_ok(),
                    emits_machine_output(stream, fmt, benchmark),
                    "K=5 at stream={stream} fmt={fmt} benchmark={benchmark}"
                );
            }
        }
    }
}

/// The text surface: the run used to record the logprobs, print none and exit 0.
#[test]
fn logprobs_on_the_text_surface_is_refused_by_name() {
    let msg = refuse_logprobs_without_machine_output(3, false, "text", false)
        .expect_err("refused")
        .to_string();
    assert!(msg.contains("--logprobs 3"), "{msg}");
    assert!(msg.contains("text output"), "{msg}");
}

/// `--json --benchmark` prints the human benchmark blob, not the JSON document,
/// so it is a `--benchmark` refusal even with `--json`.
#[test]
fn logprobs_with_json_benchmark_is_refused_by_name() {
    let msg = refuse_logprobs_without_machine_output(2, false, "json", true)
        .expect_err("refused")
        .to_string();
    assert!(msg.contains("--logprobs 2"), "{msg}");
    assert!(msg.contains("--benchmark output"), "{msg}");
}

/// `--logprobs` is capped at 20 at the flag, as `top_logprobs` is on HTTP: 20
/// parses, 21 does not. Parsed on a 16 MB stack, as in the #3754 tests.
#[test]
fn logprobs_flag_is_capped_at_twenty() {
    let parses = |k: &'static str| {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                crate::Cli::try_parse_from(["apr", "run", "m.gguf", "--json", "--logprobs", k])
                    .is_ok()
            })
            .expect("spawn parse thread")
            .join()
            .expect("join parse thread")
    };
    assert!(parses("20"), "--logprobs 20 must parse");
    assert!(!parses("21"), "--logprobs 21 must be refused at the flag");
}
