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
        logprobs: None,
        text: "Hello".to_string(),
        duration_secs: 33.646,
        cached: true,
        tokens_generated: Some(1),
        tok_per_sec: Some(0.0),
        used_gpu: Some(false),
        gpu_attempted: None,
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
        logprobs: None,
        text: "Hello".to_string(),
        duration_secs: 1.0,
        cached: true,
        tokens_generated: Some(1),
        tok_per_sec: Some(1.0),
        used_gpu,
        gpu_attempted: None,
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

    /// #3714 R2 folded a CUDA forward for qwen3moe into 0.69.1, so `--gpu` on it
    /// now REACHES that forward instead of being refused. This test used to be
    /// `gpu_forced_on_qwen3moe_is_refused_by_name`; the fold that landed the forward
    /// inverted it, as the refusal's own doc said it would ("deleting the arm here
    /// is what turns the refusal off"). Whether the GPU then actually serves is not a
    /// unit-test question: it is measured per host, and a GPU that fails shows as
    /// `fell_back: true` (#3826), never as a quiet CPU answer.
    #[test]
    fn gpu_forced_on_qwen3moe_reaches_its_cuda_forward() {
        for arch in ["qwen3moe", "qwen3_moe", "Qwen3MoeForCausalLM", "Qwen3CoderForCausalLM"] {
            assert!(
                forced_accelerator_refusal(true, Some(arch)).is_none(),
                "{arch} has a CUDA forward (#3714) and --gpu must reach it"
            );
        }
    }

    /// Qwen3.5 MoE normalises to the same `qwen3_moe` but is a hybrid the qwen3moe
    /// forward cannot run, so `--gpu` on it is still refused BY NAME, before a load.
    #[test]
    fn gpu_forced_on_qwen35_moe_is_still_refused_by_name() {
        for arch in ["qwen35moe", "qwen3_5_moe", "Qwen3_5MoeForCausalLM"] {
            let reason = forced_accelerator_refusal(true, Some(arch))
                .unwrap_or_else(|| panic!("--gpu on {arch} must refuse"));
            assert!(reason.contains(arch), "{reason}");
            assert!(reason.contains("SSM"), "says why: {reason}");
            assert!(reason.contains("refusal, not a fallback"), "{reason}");
            assert!(reason.contains("--gpu"), "names the flag to drop: {reason}");
        }
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

    /// THE MUTANT THIS ROW OWNS, after the #3714 fold. Removing the refusal outright
    /// (instead of narrowing it) would route a Qwen3.5-MoE hybrid to the qwen3moe
    /// CUDA forward, which does not run its SSM layers: wrong output, where there used
    /// to be a refusal. This is what goes RED if the narrowing is undone.
    #[test]
    fn removing_the_qwen35_moe_refusal_would_route_a_hybrid_to_the_wrong_forward() {
        let refused = forced_accelerator_refusal(true, Some("qwen35moe"));
        assert!(
            refused.is_some(),
            "MUTANT CAUGHT: --gpu on qwen35moe (Qwen3.5-35B-A3B) no longer refuses, and the \
             normalizer folds it into qwen3_moe — so it would reach a forward that cannot run \
             its Gated-DeltaNet/SSM layers (#3714, #3817)."
        );
    }
}

// ── #3826: `fell_back` must name a fallback that HAPPENED ──────────────────
//
// `apr run --format json` reported `"fell_back": false` on a run whose own
// stderr said `attempting fallback`, because the field required the user to
// have ASKED: `accel_forced && used_gpu == Some(false)`. A bare `apr run` on a
// cuda build attempts CUDA unasked, and when the F2 gate refuses that result
// (cosine 0.4153 on qwen2.5-coder-0.5b, #3804/#3602) the run falls back to CPU
// — with `accel_forced == false`, so the field said no fallback occurred.
//
// A consumer counting fallbacks therefore saw NONE, on every host, forever:
// #3720 makes this JSON an explicit consumer contract, so the number it reads
// was not merely imprecise, it was unreachable.
#[cfg(test)]
mod pmat3826_fell_back_names_a_real_fallback {
    use super::*;

    fn result(gpu_attempted: Option<bool>, used_gpu: Option<bool>) -> RunResult {
        let mut r = gpu_result(used_gpu);
        r.gpu_attempted = gpu_attempted;
        r
    }

    fn fell_back(accel_forced: bool, attempted: Option<bool>, used: Option<bool>) -> bool {
        let json = build_final_json(&result(attempted, used), "m.gguf", 8, accel_forced);
        json["backend"]["fell_back"].as_bool().expect("fell_back is a bool")
    }

    /// Every `(attempted, used_gpu)` state, with its expectation under BOTH
    /// `accel_forced` values — because the two terms are a disjunction and the
    /// interesting rows are the ones where they disagree.
    ///
    /// `(attempted, used_gpu, want_when_forced, want_when_not, what this run is)`
    const CASES: &[(Option<bool>, Option<bool>, bool, bool, &str)] = &[
        (
            Some(true), Some(false), true, true,
            "THE #3826 CASE: an accelerator was entered and its result refused, \
             so the CPU redid the work. A fallback WHETHER OR NOT it was asked \
             for — this is the row where the old predicate reported none",
        ),
        (
            Some(true), Some(true), false, false,
            "the accelerator was entered and produced the answer — not a fallback",
        ),
        (
            Some(false), Some(false), true, false,
            "nothing was attempted. Asked for: the user did not get the backend \
             they named, so it is a fallback (#3602). Not asked for: a plain CPU \
             run, and calling that a fallback is the over-correction of dropping \
             `accel_forced` outright",
        ),
        (
            None, Some(false), true, false,
            "the path did not say whether it attempted. #3602's own fixture — \
             under --gpu the run still did not reach the GPU, which is the \
             finding; unasked, there is nothing to report",
        ),
        (
            Some(true), None, false, false,
            "entered an accelerator, never reported whether it produced tokens. \
             Absent is Unknown, never Fail: `reconcile_accelerator` refuses to \
             call this a fallback and the JSON beside it must agree",
        ),
    ];

    /// Asking is SUFFICIENT, never NECESSARY. #3826 was that `fell_back`
    /// treated it as necessary, so a refused accelerator nobody asked for
    /// reported nothing. The fix must not swing the other way and retire
    /// #3602's row, which is exactly what the narrower predicate did.
    #[test]
    fn asking_for_the_gpu_is_sufficient_for_a_fallback_never_necessary() {
        let wrong: Vec<String> = CASES
            .iter()
            .flat_map(|(a, u, want_forced, want_unforced, why)| {
                [(true, *want_forced), (false, *want_unforced)]
                    .into_iter()
                    .filter_map(move |(forced, want)| {
                        let got = fell_back(forced, *a, *u);
                        (got != want).then(|| {
                            format!(
                                "\n  - attempted={a:?} used_gpu={u:?} accel_forced={forced}: \
                                 expected fell_back={want}, got {got}. {why}"
                            )
                        })
                    })
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "#3826 REGRESSION: {} of {} (state x accel_forced) combinations report \
             the wrong fallback. A harness counting fallbacks reads this field and \
             nothing else:{}",
            wrong.len(),
            CASES.len() * 2,
            wrong.join("")
        );
    }

    /// The consumer symptom, asserted as a symptom rather than as a field value:
    /// an unasked-for, refused accelerator must be COUNTABLE.
    #[test]
    fn a_harness_counting_fallbacks_sees_the_unasked_one() {
        let runs = [
            fell_back(false, Some(true), Some(false)), // bare run, CUDA refused
            fell_back(false, Some(false), Some(false)), // plain CPU run
            fell_back(true, Some(true), Some(false)),  // --gpu, refused
        ];
        let counted = runs.iter().filter(|b| **b).count();
        assert_eq!(
            counted, 2,
            "#3826: a harness counting fallbacks across these three runs must see 2 \
             (the bare run whose CUDA result was refused, and the explicit --gpu one). \
             Seeing 1 means the unasked-for fallback is invisible again — which is the \
             defect, not a rounding difference."
        );
    }
}
