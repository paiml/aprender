// #3801: the think-budget guard.
//
// With thinking ON, greedy decoding of a small Qwen3.5 often never closes its think
// block. Measured: Qwen3.5-0.8B "What is 2+2?" still reasoning at 16384 tokens. Pinned
// llama.cpp d1d3c3396 does the same on 1-2 of the 3 golden prompts, so the model does
// this, not apr. The guard is the bounded-thinking method Qwen documents: when the
// reasoning reaches its budget unclosed, close it with the template's own tag and
// generate the answer. It is reported (`reasoning_truncated`), never silent.

/// What the guard appends to reasoning that reached its budget unclosed: the close tag
/// the split reads, then the blank line Qwen's templates put before an answer.
pub const THINK_BUDGET_CLOSE: &str = "\n</think>\n\n";

/// How much of a `max_tokens` completion the reasoning may use before the guard closes
/// it: three quarters, so the answer always keeps a quarter of the budget.
#[must_use]
pub fn think_budget(max_tokens: usize) -> usize {
    (max_tokens.saturating_mul(3) / 4).max(1)
}

/// One generation pass: its text and how many tokens it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// The generated text (the completion only, never the context).
    pub text: String,
    /// Tokens generated.
    pub tokens: usize,
}

/// A completion generated under the think-budget guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardedCompletion {
    /// The whole completion, including the guard's close when it fired.
    pub text: String,
    /// Tokens generated across both passes (the injected close is not counted).
    pub tokens: usize,
    /// Whether the guard closed the think block because the model had not.
    pub reasoning_truncated: bool,
}

impl ChatPrompt {
    /// Whether `completion` of this prompt ends inside a think block.
    #[must_use]
    pub fn still_thinking(&self, completion: &str) -> bool {
        self.split(completion, 0).is_err()
    }

    /// Generate a completion of this prompt under the think-budget guard (#3801).
    ///
    /// `generate(context, max_tokens)` continues `context` (raw text, never
    /// re-templated) for at most `max_tokens` tokens and returns only what it
    /// generated.
    /// - A prompt that does not think gets one call with the whole budget.
    /// - A thinking prompt gets [`think_budget`] tokens first. A block still open after
    ///   them is closed with [`THINK_BUDGET_CLOSE`], and the answer is generated from
    ///   prompt + reasoning + close within what remains (`reasoning_truncated`).
    /// - A first pass that closed its block but stopped at the budget is continued the
    ///   same way, without a close, so the answer is never cut at three quarters.
    ///
    /// # Errors
    ///
    /// Whatever `generate` returns.
    pub fn generate_with_think_budget<E>(
        &self,
        max_tokens: usize,
        mut generate: impl FnMut(&str, usize) -> Result<Generation, E>,
    ) -> Result<GuardedCompletion, E> {
        if !self.thinking {
            let only = generate(&self.text, max_tokens)?;
            return Ok(GuardedCompletion {
                text: only.text,
                tokens: only.tokens,
                reasoning_truncated: false,
            });
        }
        let budget = think_budget(max_tokens);
        let first = generate(&self.text, budget)?;
        let open = self.still_thinking(&first.text);
        let remaining = max_tokens.saturating_sub(first.tokens);
        if (!open && first.tokens < budget) || remaining == 0 {
            return Ok(GuardedCompletion {
                text: first.text,
                tokens: first.tokens,
                reasoning_truncated: false,
            });
        }
        let mut text = first.text;
        if open {
            text.push_str(THINK_BUDGET_CLOSE);
        }
        let rest = generate(&format!("{}{text}", self.text), remaining)?;
        text.push_str(&rest.text);
        Ok(GuardedCompletion {
            text,
            tokens: first.tokens + rest.tokens,
            reasoning_truncated: open,
        })
    }
}

#[cfg(test)]
mod chat_template_think_budget_tests {
    use super::{think_budget, ChatPrompt, Generation, GuardedCompletion, THINK_BUDGET_CLOSE};

    const OPENS: &str = "<|im_start|>assistant\n<think>\n";

    fn prompt(text: &str, thinking: bool) -> ChatPrompt {
        ChatPrompt {
            text: text.to_string(),
            thinking,
        }
    }

    /// A scripted model: each call gets the next reply, and the (context, budget) it saw
    /// is recorded.
    fn scripted(
        replies: Vec<(&'static str, usize)>,
    ) -> (
        std::rc::Rc<std::cell::RefCell<Vec<(String, usize)>>>,
        impl FnMut(&str, usize) -> Result<Generation, String>,
    ) {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let log = seen.clone();
        let mut replies = replies.into_iter();
        let generate = move |context: &str, budget: usize| {
            log.borrow_mut().push((context.to_string(), budget));
            let (text, tokens) = replies.next().ok_or("no more replies")?;
            Ok(Generation {
                text: text.to_string(),
                tokens: tokens.min(budget),
            })
        };
        (seen, generate)
    }

    #[test]
    fn the_budget_leaves_the_answer_a_quarter() {
        assert_eq!(think_budget(4096), 3072);
        assert_eq!(think_budget(512), 384);
        assert_eq!(think_budget(1), 1);
        assert_eq!(think_budget(0), 1);
    }

    #[test]
    fn a_prompt_that_does_not_think_is_one_call_with_the_whole_budget() {
        let (seen, generate) = scripted(vec![("4", 1)]);
        let out = prompt("P", false).generate_with_think_budget(4096, generate).unwrap();
        assert_eq!(out, GuardedCompletion { text: "4".into(), tokens: 1, reasoning_truncated: false });
        assert_eq!(*seen.borrow(), vec![("P".to_string(), 4096)]);
    }

    #[test]
    fn reasoning_that_closes_within_the_budget_is_one_call() {
        let (seen, generate) = scripted(vec![("add them</think>\n\n4", 40)]);
        let p = prompt(OPENS, true);
        let out = p.generate_with_think_budget(4096, generate).unwrap();
        assert!(!out.reasoning_truncated);
        assert_eq!(seen.borrow().len(), 1);
        assert_eq!(seen.borrow()[0].1, 3072, "the first pass gets the think budget");
        assert_eq!(p.split(&out.text, 4096).unwrap().answer, "4");
    }

    #[test]
    fn reasoning_still_open_at_the_budget_is_closed_by_the_guard_and_answered() {
        let (seen, generate) = scripted(vec![("Wait, let me check again", 3072), ("4", 1)]);
        let p = prompt(OPENS, true);
        let out = p.generate_with_think_budget(4096, generate).unwrap();
        assert!(out.reasoning_truncated, "the guard fired and must say so");
        assert_eq!(out.tokens, 3073);
        let seen = seen.borrow();
        assert_eq!(
            seen[1],
            (format!("{OPENS}Wait, let me check again{THINK_BUDGET_CLOSE}"), 1024),
            "the answer continues prompt + reasoning + close, within what remains"
        );
        let split = p.split(&out.text, 4096).unwrap();
        assert_eq!(split.reasoning.as_deref(), Some("Wait, let me check again"));
        assert_eq!(split.answer, "4");
    }

    #[test]
    fn a_block_the_model_opens_itself_is_guarded_the_same_way() {
        // Qwen3: the prompt does not open the block, the completion does.
        let (_, generate) = scripted(vec![("<think>\nhmm", 3072), ("Paris.", 2)]);
        let p = prompt("<|im_start|>assistant\n", true);
        let out = p.generate_with_think_budget(4096, generate).unwrap();
        assert!(out.reasoning_truncated);
        assert_eq!(p.split(&out.text, 4096).unwrap().answer, "Paris.");
    }

    #[test]
    fn an_answer_cut_by_the_think_budget_is_continued_without_a_close() {
        let (seen, generate) = scripted(vec![("ok</think>\n\nThe capital", 3072), (" is Paris.", 3)]);
        let p = prompt(OPENS, true);
        let out = p.generate_with_think_budget(4096, generate).unwrap();
        assert!(!out.reasoning_truncated, "the model closed its own block");
        assert!(!seen.borrow()[1].0.contains(THINK_BUDGET_CLOSE));
        assert_eq!(p.split(&out.text, 4096).unwrap().answer, "The capital is Paris.");
    }

    #[test]
    fn a_budget_spent_entirely_on_reasoning_leaves_the_block_to_the_split() {
        let (_, generate) = scripted(vec![("still going", 1)]);
        let p = prompt(OPENS, true);
        let out = p.generate_with_think_budget(1, generate).unwrap();
        assert!(!out.reasoning_truncated);
        assert!(p.split(&out.text, 1).is_err(), "nothing remains to answer in: the split refuses");
    }
}
