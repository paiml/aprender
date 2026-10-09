//! What a run owes its caller beyond the text (#3718).
//!
//! `apr run --json` reported the generated tokens and nothing about the prompt, so a
//! consumer that needed "how many tokens did the model actually read" rebuilt the
//! model's tokenizer from the GGUF to count them (RAH, #3716: 13,495 on one lane). And
//! a reply cut at `max_tokens` looked exactly like a reply that finished, so a
//! consumer could not tell a truncated verdict from a malformed one.
//!
//! [`RunReport`] carries the two facts only the decode path knows: why generation
//! ended, and the context window the prompt was measured against. The prompt count
//! itself is `InferenceResult::input_token_count`, which is the post-chat-template
//! count `prepare_tokens` fed to the model.

/// Why generation ended.
///
/// Shared by `apr run --json` and the OpenAI-compatible server (`api::types`
/// re-exports it), so both surfaces spell the same outcome the same way.
///
/// Dogfood 0.63.0 (#2375 finding 6): the STREAMING chat path emitted the string
/// literal `"stop"` in its terminal chunk no matter what happened, while the
/// non-streaming path on the identical request correctly reported `"length"`
/// when the generation hit `max_tokens`. A client that streams therefore cannot
/// tell a truncated answer from a finished one, and every "continue from where
/// you stopped" flow silently breaks.
///
/// The type exists so that literal cannot come back: the terminal-chunk
/// constructor takes a `FinishReason`, which is only obtainable from
/// [`FinishReason::from_generation`] (or an explicit, named variant). There is
/// no `&str` parameter left to hardcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted a stop token or a stop string matched.
    Stop,
    /// Generation was cut off at its token budget: `max_tokens`, or the room left
    /// in the context window when a path shrinks the budget to fit it.
    Length,
}

impl FinishReason {
    /// Decide the reason from what the generation actually did.
    ///
    /// Mirrors `finalize_chat_text` / `completion_finish_reason`: a matched stop
    /// string wins over the budget, and a model that terminated early is
    /// `Stop`. Only "ran to the budget with no stop match" is `Length`.
    ///
    /// `max_tokens` must be the budget the loop RAN with, not the one requested:
    /// a server that clamps the request to the context window and then compares
    /// against the unclamped number reports every clamped cut as `Stop` (#3718).
    #[must_use]
    pub fn from_generation(stopped: bool, completion_tokens: usize, max_tokens: usize) -> Self {
        if !stopped && completion_tokens >= max_tokens {
            Self::Length
        } else {
            Self::Stop
        }
    }

    /// Decide the reason from the tokens a decode loop returned.
    ///
    /// Every generate loop `apr run` reaches ends, short of an error, in one of two
    /// ways: it sampled a stop token, or it spent its budget. The loops disagree
    /// about the stop token itself. The Qwen3.5 hybrid, MoE and wgpu loops push it
    /// and then break; the dense CPU and CUDA loops break before pushing. So the
    /// last token alone cannot separate "stopped" from "cut":
    ///
    /// - a last token in `stop_tokens` is `Stop` (a loop that pushes it);
    /// - otherwise a run that filled `budget` is `Length`;
    /// - a shorter run can only have ended on a stop token its loop consumed
    ///   without pushing, which is `Stop`.
    #[must_use]
    pub fn from_decode(generated: &[u32], stop_tokens: &[u32], budget: usize) -> Self {
        let ended_on_stop = generated.last().is_some_and(|t| stop_tokens.contains(t));
        Self::from_generation(ended_on_stop, generated.len(), budget)
    }

    /// The wire string OpenAI clients match on.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The token budget a decode loop actually ran with.
///
/// Only the dense CPU loop shrinks `max_tokens` to the room left in the context
/// window (`effective_max_tokens`, aprender#2376), so `clamps_to_context` is true
/// for that loop alone. Every other loop runs the request's `max_tokens` as is.
/// A prompt that does not fit at all is refused before this is asked
/// (`ContextLimitExceeded`), so the subtraction saturates only on paths that do
/// not check.
#[must_use]
pub fn decode_budget(
    max_tokens: usize,
    prompt_len: usize,
    context_length: usize,
    clamps_to_context: bool,
) -> usize {
    if clamps_to_context {
        max_tokens.min(context_length.saturating_sub(prompt_len))
    } else {
        max_tokens
    }
}

/// What a run reports beyond [`super::InferenceResult`].
///
/// Each field is `None` on a path that does not know it, never a default that
/// reads as a measurement. The APR and SafeTensors paths report none of them today.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Why generation ended.
    pub finish_reason: Option<FinishReason>,
    /// The model's context window, from its metadata.
    pub context_length: Option<usize>,
    /// #3602: whole ms inside the GPU-vs-CPU parity guards ([`GuardCost::validate_ms`]).
    pub validate_ms: Option<u64>,
    /// #3602: whole ms of a GPU attempt that was refused and redone on the CPU
    /// ([`GuardCost::rejected_ms`]).
    pub rejected_ms: Option<u64>,
}

// #3602: `apr run --gpu` on qwen2.5-coder-0.5b took 12.7 s against 11.5 s with
// `--no-gpu`, and nothing a run reported said where the GPU run's time went. The
// load-time parity gate and the F2 first-token check each run a CPU reference
// forward, and a GPU attempt they refuse is paid for and then redone on the CPU.
// Thread-local for the reason `GENERATION_START` is: the guards run on the
// dispatch thread, below any signature that could carry a duration back up.
std::thread_local! {
    static GUARD_COST: std::cell::Cell<GuardCost> =
        const { std::cell::Cell::new(GuardCost { validate_ms: None, rejected_ms: None }) };
}

/// #3602: the wall time a run spent in the GPU-vs-CPU guards, in ms.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GuardCost {
    /// Inside the parity guards: the load-time gate and the F2 first-token check,
    /// summed. Part of `setup_ms`, never of generation. `None` when neither ran.
    pub validate_ms: Option<f64>,
    /// GPU attempts that ended on the CPU, summed (CUDA, then wgpu): each from
    /// entering its backend until it handed the model back, whether a guard refused
    /// it or the device would not start. Includes their `validate_ms`. `None` when
    /// no attempt was refused. A
    /// forward failure mid-turn, which the dense session replays on the CPU
    /// (#4268), is not counted here.
    pub rejected_ms: Option<f64>,
}

impl GuardCost {
    /// The report's whole-ms fields: `(validate_ms, rejected_ms)`.
    #[must_use]
    pub fn whole_ms(self) -> (Option<u64>, Option<u64>) {
        let whole = |ms: f64| ms.max(0.0).round() as u64;
        (self.validate_ms.map(whole), self.rejected_ms.map(whole))
    }
}

fn add_ms(slot: Option<f64>, start: std::time::Instant) -> Option<f64> {
    Some(slot.unwrap_or(0.0) + start.elapsed().as_secs_f64() * 1000.0)
}

/// Run a parity guard, adding its wall time to this thread's `validate_ms`.
pub(crate) fn time_guard<T>(guard: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let verdict = guard();
    GUARD_COST.with(|c| {
        let mut cost = c.get();
        cost.validate_ms = add_ms(cost.validate_ms, start);
        c.set(cost);
    });
    verdict
}

/// Run a GPU attempt. When it hands the model back for the CPU (`Err`), its wall
/// time is added to this thread's `rejected_ms`; a GPU that produced the answer
/// adds nothing.
pub(crate) fn time_gpu_attempt<T, E>(
    attempt: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    let start = std::time::Instant::now();
    let outcome = attempt();
    if outcome.is_err() {
        GUARD_COST.with(|c| {
            let mut cost = c.get();
            cost.rejected_ms = add_ms(cost.rejected_ms, start);
            c.set(cost);
        });
    }
    outcome
}

/// Take this thread's guard cost, leaving it empty. Called before a dispatch, so
/// no run inherits an earlier one's, and after it, to report.
pub(crate) fn take_guard_cost() -> GuardCost {
    GUARD_COST.with(std::cell::Cell::take)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 7;

    /// The case table. Each row names the loop shape it stands for, because the
    /// rule is only correct if it is correct for every shape.
    #[test]
    fn from_decode_case_table() {
        let rows: &[(&str, &[u32], usize, FinishReason)] = &[
            // Loops that PUSH the stop token (qwen35, MoE, wgpu).
            (
                "pushed stop, under budget",
                &[5, 6, EOS],
                8,
                FinishReason::Stop,
            ),
            (
                "pushed stop as the last budgeted token",
                &[5, 6, EOS],
                3,
                FinishReason::Stop,
            ),
            // Loops that CONSUME it (dense CPU, dense CUDA).
            (
                "consumed stop, under budget",
                &[5, 6],
                8,
                FinishReason::Stop,
            ),
            (
                "consumed stop, nothing generated",
                &[],
                8,
                FinishReason::Stop,
            ),
            // The cut, on either shape.
            ("budget spent, no stop", &[5, 6, 9], 3, FinishReason::Length),
            ("zero budget", &[], 0, FinishReason::Length),
            // A clamped budget (dense CPU at the context edge): the cut is below
            // max_tokens, and it is still a cut.
            ("clamped budget spent", &[5, 6], 2, FinishReason::Length),
        ];
        for (name, generated, budget, want) in rows {
            assert_eq!(
                FinishReason::from_decode(generated, &[EOS], *budget),
                *want,
                "{name}"
            );
        }
    }

    /// The defect this rule exists for: a clamped cut compared against the
    /// REQUESTED budget reads as a stop. Pinned so the call sites cannot regress
    /// to passing `max_tokens` where the loop's budget belongs.
    #[test]
    fn clamped_cut_is_length_only_against_the_budget_it_ran_with() {
        let generated = [5, 6, 9, 10];
        let requested = 256;
        let ran_with = decode_budget(requested, 60, 64, true);
        assert_eq!(ran_with, 4);
        assert_eq!(
            FinishReason::from_decode(&generated, &[EOS], requested),
            FinishReason::Stop
        );
        assert_eq!(
            FinishReason::from_decode(&generated, &[EOS], ran_with),
            FinishReason::Length
        );
    }

    #[test]
    fn decode_budget_clamps_only_when_the_loop_does() {
        assert_eq!(decode_budget(100, 10, 64, true), 54);
        assert_eq!(decode_budget(100, 10, 64, false), 100);
        assert_eq!(decode_budget(20, 10, 64, true), 20);
        assert_eq!(decode_budget(100, 70, 64, true), 0);
    }

    #[test]
    fn wire_strings_are_the_openai_ones() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
    }

    #[test]
    fn an_unreported_run_is_none_not_a_default_reason() {
        let report = RunReport::default();
        assert_eq!(report.finish_reason, None);
        assert_eq!(report.context_length, None);
    }

    // -----------------------------------------------------------------------
    // Through `run_inference_report` on a real GGUF file: the dense CPU loop,
    // which clamps its budget to the context and consumes its stop token.
    // -----------------------------------------------------------------------

    /// The pygmy GGUF's context window (`build_executable_pygmy_gguf`).
    const PYGMY_CONTEXT: usize = 32;

    fn pygmy_file() -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp gguf");
        file.write_all(&crate::gguf::test_factory::build_executable_pygmy_gguf())
            .expect("write gguf");
        file.flush().expect("flush gguf");
        file
    }

    fn run(
        file: &tempfile::NamedTempFile,
        prompt: Vec<u32>,
        max_tokens: usize,
        stop: Vec<u32>,
    ) -> crate::error::Result<(super::super::InferenceResult, RunReport)> {
        let config = super::super::InferenceConfig::new(file.path())
            .with_input_tokens(prompt)
            .with_max_tokens(max_tokens)
            .with_stop_tokens(stop)
            .without_gpu();
        super::super::run_inference_report(&config)
    }

    /// The case #3718 exists for, at the `apr run` layer: a request whose budget
    /// is larger than the room left in the context. The dense CPU loop stops at
    /// the context edge, short of `max_tokens`, and that is a cut.
    #[test]
    fn context_clamped_cut_reports_length_with_the_counts() {
        let file = pygmy_file();
        let prompt = vec![1, 2, 3, 4];
        let (result, report) = run(&file, prompt.clone(), 1000, vec![]).expect("run");
        assert_eq!(result.input_token_count, prompt.len());
        assert_eq!(report.context_length, Some(PYGMY_CONTEXT));
        assert_eq!(
            result.generated_token_count,
            PYGMY_CONTEXT - prompt.len(),
            "control: the loop must run to the context edge, or this is not measuring a clamp"
        );
        assert_eq!(report.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn max_tokens_cut_reports_length() {
        let file = pygmy_file();
        let (result, report) = run(&file, vec![1, 2, 3, 4], 3, vec![]).expect("run");
        assert_eq!(result.generated_token_count, 3);
        assert_eq!(report.finish_reason, Some(FinishReason::Length));
    }

    /// The converse: a stop token under budget is `Stop`. The dense CPU loop
    /// consumes it without pushing, so this is the "shorter run" branch.
    #[test]
    fn a_stop_token_under_budget_reports_stop() {
        let file = pygmy_file();
        let prompt = vec![1, 2, 3, 4];
        let (first, _) = run(&file, prompt.clone(), 1, vec![]).expect("probe run");
        let first_token = first.tokens[prompt.len()];
        let (result, report) = run(&file, prompt, 10, vec![first_token]).expect("run");
        assert_eq!(
            result.generated_token_count, 0,
            "control: greedy is deterministic, so the probed token must stop the run at once"
        );
        assert_eq!(report.finish_reason, Some(FinishReason::Stop));
    }

    /// done_when 1's case row: a fixed TEXT prompt through the GGUF's own
    /// tokenizer, the path `apr run --prompt` takes. The expected ids are derived
    /// by hand from the vocabulary, not from apr's encoder: under a SentencePiece
    /// vocabulary that holds `▁Hello`, `▁world` and `!` whole, "Hello world!" is
    /// `<s> ▁Hello ▁world !`, so a conforming tokenizer feeds 4 tokens.
    #[test]
    fn a_fixed_text_prompt_counts_the_tokens_the_tokenizer_feeds() {
        use std::io::Write;
        let mut vocab: Vec<String> = ["<unk>", "<s>", "</s>", "▁Hello", "▁world", "!"]
            .map(String::from)
            .to_vec();
        vocab.extend((vocab.len()..32).map(|i| format!("<filler_{i}>")));
        let pieces: Vec<&str> = vocab.iter().map(String::as_str).collect();
        let bytes = crate::gguf::test_factory::build_executable_pygmy_gguf_with(|b| {
            b.add_string("tokenizer.ggml.model", "llama")
                .add_string_array("tokenizer.ggml.tokens", &pieces)
                .add_u32("tokenizer.ggml.bos_token_id", 1)
                .add_u32("tokenizer.ggml.eos_token_id", 2)
        });
        let mut file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp gguf");
        file.write_all(&bytes).expect("write gguf");
        file.flush().expect("flush gguf");

        let config = super::super::InferenceConfig::new(file.path())
            .with_prompt("Hello world!")
            .with_max_tokens(2)
            .without_gpu();
        let (result, report) = super::super::run_inference_report(&config).expect("run");
        assert_eq!(
            result.tokens.get(..4),
            Some(&[1, 3, 4, 5][..]),
            "the prompt must be fed as <s> ▁Hello ▁world !"
        );
        assert_eq!(result.input_token_count, 4, "prompt_tokens is what was fed");
        assert_eq!(report.context_length, Some(PYGMY_CONTEXT));
    }

    /// A prompt that does not fit is refused, never cut to fit: done_when 3's
    /// "or an error".
    #[test]
    fn a_prompt_longer_than_the_context_is_refused() {
        let file = pygmy_file();
        let prompt: Vec<u32> = (0..(PYGMY_CONTEXT as u32 + 4)).map(|t| t % 32).collect();
        let err = run(&file, prompt, 4, vec![]).expect_err("an over-long prompt must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains(&(PYGMY_CONTEXT + 4).to_string())
                && msg.contains(&PYGMY_CONTEXT.to_string()),
            "the refusal must name the prompt length and the context: {msg}"
        );
    }

    fn pause() {
        std::thread::sleep(std::time::Duration::from_millis(30));
    }

    /// #3602 falsifier, planted: an attempt that hands the model back for the CPU
    /// is charged to `rejected_ms`, for at least as long as it ran.
    #[test]
    fn a_refused_gpu_attempt_is_charged_to_rejected_ms() {
        let _ = take_guard_cost();
        let out: std::result::Result<(), &str> = time_gpu_attempt(|| {
            pause();
            Err("planted refusal")
        });
        assert_eq!(
            out,
            Err("planted refusal"),
            "the attempt's own outcome passes through"
        );
        let cost = take_guard_cost();
        let rejected = cost.rejected_ms.expect("a refused attempt must be charged");
        assert!(
            rejected >= 30.0,
            "charged {rejected} ms for a 30 ms attempt"
        );
        assert_eq!(cost.validate_ms, None, "control: no guard ran");
    }

    /// The healthy twin: a GPU that produced the answer is charged nothing.
    #[test]
    fn an_accepted_gpu_attempt_is_not_charged() {
        let _ = take_guard_cost();
        let out: std::result::Result<u32, ()> = time_gpu_attempt(|| {
            pause();
            Ok(7)
        });
        assert_eq!(out, Ok(7));
        assert_eq!(take_guard_cost(), GuardCost::default());
    }

    /// The load-time gate and the F2 check of one attempt are summed, each
    /// verdict passes through, and taking the cost leaves it empty.
    #[test]
    fn guard_time_is_summed_over_both_guards() {
        let _ = take_guard_cost();
        let gate = time_guard(|| {
            pause();
            "gate"
        });
        let f2 = time_guard(|| {
            pause();
            "f2"
        });
        assert_eq!((gate, f2), ("gate", "f2"));
        let validate = take_guard_cost().validate_ms.expect("two guards ran");
        assert!(validate >= 60.0, "two 30 ms guards summed to {validate} ms");
        assert_eq!(
            take_guard_cost(),
            GuardCost::default(),
            "take must leave it empty"
        );
    }

    #[test]
    fn whole_ms_rounds_and_keeps_none() {
        let cost = GuardCost {
            validate_ms: Some(1234.5),
            rejected_ms: None,
        };
        assert_eq!(cost.whole_ms(), (Some(1235), None));
        let cost = GuardCost {
            validate_ms: Some(0.4),
            rejected_ms: Some(2.6),
        };
        assert_eq!(
            cost.whole_ms(),
            (Some(0), Some(3)),
            "a guard that ran is Some(0), not None"
        );
    }

    /// A run never reports an earlier run's guard time. The plant is what a
    /// refused GPU attempt leaves on this thread; the CPU run after it ran no
    /// guard, so it must report neither field.
    #[test]
    fn a_cpu_run_does_not_inherit_an_earlier_runs_guard_cost() {
        let file = pygmy_file();
        let _: std::result::Result<(), ()> = time_gpu_attempt(|| time_guard(|| Err(())));
        assert_ne!(
            GUARD_COST.with(std::cell::Cell::get),
            GuardCost::default(),
            "control: the plant must take, or this test measures nothing"
        );
        let (_, report) = run(&file, vec![1, 2, 3, 4], 2, vec![]).expect("run");
        assert_eq!((report.validate_ms, report.rejected_ms), (None, None));
    }
}
