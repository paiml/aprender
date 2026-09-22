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
/// reads as a measurement. The APR and SafeTensors paths report neither today.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Why generation ended.
    pub finish_reason: Option<FinishReason>,
    /// The model's context window, from its metadata.
    pub context_length: Option<usize>,
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
}
