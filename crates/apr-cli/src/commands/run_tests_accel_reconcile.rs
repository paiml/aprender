// #3602 — `apr run --gpu` must not report a CPU fallback as success.
//
// **The defect, measured on lambda (RTX 4090 sm_89, `apr 0.68.2 (e6f77c98c)`,
// `qwen2.5-coder-0.5b-instruct-q4_k_m`, 2026-09-21):**
//
// ```text
// $ apr run <model> --prompt Hi --max-tokens 1 --gpu --json
// exit=0
// stderr: warning: GPU output diverges from CPU at position 1 (cosine 0.4153) — falling back to CPU
// stdout: { ..., "used_gpu": false, "inference_time_ms": 33646.13 }
// ```
//
// Exit 0. Zero `selected:` lines, zero `parity:` lines. A run that was asked for
// the GPU, was refused the GPU at runtime, took 33.6 s on the CPU, and reported
// success — with `used_gpu: false` as the only signal, which is the same value a
// deliberate CPU run reports.
//
// **The mechanism was already built and had no caller.** `registry::after_generation`
// carries the decision (R-0b, #3002/#3042) — forced ⇒ refuse, default ⇒ corrective
// line — with its own unit tests at `registry.rs`, and a `git grep` found its only
// callers were those tests. `registry::announce` and `registry::parity_line` are in
// the same state. `accel.rs` states the rule the whole layer exists to enforce:
// *"a silent CPU fallback is exactly that override wearing a performance number."*
//
// So this file does NOT test a new decision. It tests that the recorded one is
// reachable from `apr run`, which is what was missing.
//
// **Both directions, because "refuse whenever `used_gpu` is false" would pass the
// first test and break every honest CPU run.**
//
// | case | `accel_forced` | `used_gpu` | required |
// |---|---|---|---|
// | asked for GPU, ran on CPU | true | `Some(false)` | **refusal** — the defect |
// | asked for GPU, ran on GPU | true | `Some(true)` | silent pass |
// | asked for nothing, ran on CPU | false | `Some(false)` | silent pass |
// | asked for nothing, ran on GPU | false | `Some(true)` | silent pass |
// | backend did not report | true | `None` | silent pass — absent is not false |
//
// **Round 1 of the AD-04 quorum turned this PR FAIL on two counts, both correct,
// and both are fixed here.** (1) `reconcile_accelerator(...)?` ran BEFORE
// `print_run_output`, so a rejected `--gpu` run early-returned and `--json`
// emitted nothing at all — the surface the ticket names. (2) the caller's
// `if let Some(note)` arm was UNREACHABLE: `announced` was `Some("gpu")` exactly
// when `forced` was true, and `after_generation`'s corrective-line branch needs
// `forced == false`. A branch with no reachable caller — the very defect this PR
// fixes, reproduced one layer down while fixing it. Lane 1 (gemini-3.1-pro-high)
// found both; lane 3 passed the PR without seeing either.
//
// The last row is deliberate. `used_gpu: None` means the engine did not report,
// which is not evidence that it fell back; refusing on it would turn every
// non-reporting path into a hard error. Absence is Unknown, never Fail.

/// THE CASE THAT SHIPPED. `--gpu`, ran on CPU ⇒ refusal, not a success blob.
#[test]
fn a_forced_accelerator_that_ran_on_cpu_is_refused() {
    let result = RunResult {
        text: "Hello".to_string(),
        duration_secs: 33.646,
        cached: true,
        tokens_generated: Some(1),
        tok_per_sec: Some(0.0),
        used_gpu: Some(false),
        generated_tokens: Some(vec![9707]),
        token_texts: None,
        usage: Default::default(),
    };

    let err = reconcile_accelerator(true, &result)
        .expect_err("--gpu that ran on CPU must refuse, not return a note");

    // The message has to name the escape hatch, or the refusal is a dead end for
    // someone who genuinely wants the CPU run.
    let msg = err.to_string();
    assert!(
        msg.contains("--no-gpu"),
        "the refusal must tell the user how to run on CPU deliberately: {msg}"
    );
}

/// The falsifier for the fix: a healthy GPU run must stay silent. A build that
/// "always refuses" passes the test above and fails this one.
#[test]
fn a_forced_accelerator_that_actually_ran_on_gpu_says_nothing() {
    let result = gpu_result(Some(true));
    reconcile_accelerator(true, &result).expect(
        "nothing needs saying when the GPU was asked for and the GPU ran",
    );
}

/// The other falsifier: an ordinary CPU run was never a fallback. A build that
/// keys on `used_gpu == Some(false)` alone passes the first test and breaks
/// every `apr run` without `--gpu`.
#[test]
fn an_unforced_cpu_run_is_not_a_fallback() {
    let result = gpu_result(Some(false));
    reconcile_accelerator(false, &result)
        .expect("no accelerator was requested, so there is nothing to reconcile");
}

/// Absent is Unknown, never Fail: a backend that did not report `used_gpu` has
/// not reported a fallback.
#[test]
fn a_backend_that_did_not_report_is_not_treated_as_a_fallback() {
    let result = gpu_result(None);
    reconcile_accelerator(true, &result)
        .expect("used_gpu: None is Unknown, not false — not a CPU fallback");
}

/// `used_gpu: false` alone collapses "ran on CPU deliberately" and "was refused
/// the GPU". The JSON must separate them, since that is what a consumer reads.
#[test]
fn the_json_distinguishes_a_deliberate_cpu_run_from_a_rejected_gpu_run() {
    let cpu = build_final_json(&gpu_result(Some(false)), "m.gguf", 1, false);
    let rejected = build_final_json(&gpu_result(Some(false)), "m.gguf", 1, true);

    // Same `used_gpu` — this is exactly the ambiguity that let a 33.6 s CPU
    // fallback be read as a GPU timing.
    assert_eq!(cpu["used_gpu"], rejected["used_gpu"]);

    assert_eq!(cpu["backend"]["requested"], "default");
    assert_eq!(cpu["backend"]["fell_back"], false);

    assert_eq!(rejected["backend"]["requested"], "gpu");
    assert_eq!(rejected["backend"]["ran"], "cpu");
    assert_eq!(
        rejected["backend"]["fell_back"], true,
        "a requested GPU that did not run is the whole finding of #3602"
    );
}

/// A GPU run that succeeded must not be labelled a fallback — the `fell_back`
/// key has to be able to be false while `requested` is `gpu`, or it is
/// decoration rather than a measurement.
#[test]
fn a_successful_gpu_run_is_not_labelled_a_fallback() {
    let json = build_final_json(&gpu_result(Some(true)), "m.gguf", 1, true);
    assert_eq!(json["backend"]["requested"], "gpu");
    assert_eq!(json["backend"]["ran"], "gpu");
    assert_eq!(json["backend"]["fell_back"], false);
}

fn gpu_result(used_gpu: Option<bool>) -> RunResult {
    RunResult {
        text: "Hello".to_string(),
        duration_secs: 1.0,
        cached: true,
        tokens_generated: Some(1),
        tok_per_sec: Some(1.0),
        used_gpu,
        generated_tokens: Some(vec![9707]),
        token_texts: None,
        usage: Default::default(),
    }
}

/// The drift guard for `emits_machine_output`. Round 2 of the quorum found
/// `--json --benchmark` leaking a human success blob on a refused run, because
/// the refusal path carried its own copy of the condition and dropped
/// `!benchmark`. This pins the predicate to the arms it describes over EVERY
/// flag combination, so the two cannot diverge again silently.
#[test]
fn the_machine_output_predicate_matches_print_run_output() {
    for &stream in &[false, true] {
        for &benchmark in &[false, true] {
            for fmt in ["json", "text", "table"] {
                // Transcribed from `print_run_output`'s own two early returns.
                let stream_arm = stream && !benchmark;
                let json_arm = fmt == "json" && !benchmark;
                assert_eq!(
                    emits_machine_output(stream, fmt, benchmark),
                    stream_arm || json_arm,
                    "predicate disagrees with print_run_output at \
                     stream={stream} fmt={fmt} benchmark={benchmark}"
                );
            }
        }
    }
}

/// The case that leaked, called out by name so a future reader sees the bug and
/// not just the invariant: `--json --benchmark` is NOT a machine surface.
#[test]
fn json_plus_benchmark_is_not_a_machine_surface() {
    assert!(
        !emits_machine_output(false, "json", true),
        "--json --benchmark prints the HUMAN benchmark blob; treating it as a \
         machine surface leaks a success rendering for a refused run"
    );
    assert!(emits_machine_output(false, "json", false));
    assert!(emits_machine_output(true, "text", false));
    assert!(!emits_machine_output(true, "json", true));
}

// =============================================================================
// #3817: --gpu on an architecture with no CUDA forward refuses BEFORE the load
// =============================================================================

#[cfg(test)]
mod forced_accelerator_refusal_tests {
    use crate::commands::run::forced_accelerator_refusal;

    /// done_when 1: refused by name, naming the architecture, the ticket and the
    /// release — and saying it is a refusal rather than a fallback.
    #[test]
    fn gpu_forced_on_qwen3moe_is_refused_by_name() {
        let reason = forced_accelerator_refusal(true, Some("qwen3moe"))
            .expect("--gpu on qwen3moe must refuse");
        assert!(reason.contains("qwen3moe"), "{reason}");
        assert!(reason.contains("#3714"), "{reason}");
        assert!(reason.contains("0.70.0"), "{reason}");
        assert!(reason.contains("refusal, not a fallback"), "{reason}");
        assert!(reason.contains("--gpu"), "names the flag to drop: {reason}");
    }

    /// done_when 2: the working CPU path must not be touched. Without the forced
    /// accelerator there is nothing to refuse, whatever the architecture.
    #[test]
    fn without_a_forced_accelerator_nothing_is_refused() {
        for arch in ["qwen3moe", "qwen3_moe", "qwen2", "llama"] {
            assert!(
                forced_accelerator_refusal(false, Some(arch)).is_none(),
                "{arch}: a run that did not force the GPU is the CPU path, which works"
            );
        }
    }

    /// done_when 4: every other architecture's `--gpu` behaviour is unchanged.
    /// A case table, not an inspection — one dense Q4_K family, the hybrid, and
    /// a non-qwen model, as the ticket asks.
    #[test]
    fn every_other_architecture_is_unaffected_by_the_refusal() {
        for arch in ["qwen2", "qwen2.5", "qwen3", "qwen35", "llama", "mistral", "gpt2", "gemma"] {
            assert!(
                forced_accelerator_refusal(true, Some(arch)).is_none(),
                "{arch} has a CUDA forward and must still reach it under --gpu"
            );
        }
    }

    /// A source whose architecture we cannot read (a hub id, a non-GGUF file) is
    /// not refused: the refusal must never be a guess.
    #[test]
    fn an_unreadable_architecture_is_not_refused() {
        assert!(forced_accelerator_refusal(true, None).is_none());
    }

    /// THE MUTANT THIS ROW OWNS. Restoring the CPU fallback under `--gpu` means
    /// `forced_accelerator_refusal` answering `None` for qwen3moe; this test is
    /// what goes RED, and it says what it caught.
    #[test]
    fn removing_the_refusal_would_restore_the_silent_cpu_fallback() {
        let refused = forced_accelerator_refusal(true, Some("qwen3moe"));
        assert!(
            refused.is_some(),
            "MUTANT CAUGHT: --gpu on qwen3moe no longer refuses. Without this refusal the run \
             loads 18 GB, generates on the CPU, and exits 14 AFTER the fact — the user asked for \
             the GPU and was told the wrong thing about their own hardware (#3817)."
        );
    }
}
