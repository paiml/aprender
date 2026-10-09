//! #3793 (#3568 PR 2/4): the one engine under a constraint.
//!
//! [`Session::generate_constrained`] is [`Session::generate`] with `constraint` masking the
//! logits before every choice. There is no second loop: the entry, the witness, the prompt
//! admission, the budget, the prefill and the token choice are the session's own, so a
//! constrained `apr run` enters the same engine every other verb does (#4263).

use super::{choose_token, turn_budget, witness, ArchForward, Entry, EntryKind, Session, Turn};
use crate::constrain::{ConstraintError, TokenConstraint};
use crate::error::{RealizarError, Result};
use crate::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

/// Why a constrained generation ended (#3793).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrainedStop {
    /// The output is a complete document: the model ended it (end-of-sequence, which the
    /// constraint admits only there), or the budget ended exactly on a complete document.
    Complete,
    /// The token budget ran out before the document was complete.
    Length,
}

/// A stop token on a constrained output: `Complete` when the document is whole (the mask
/// admits end-of-sequence only then), else the engine's refusal. Never a silent stop
/// mid-document.
fn constrained_end(
    constraint: &mut dyn TokenConstraint,
    token: u32,
    position: usize,
) -> Result<ConstrainedStop> {
    if constraint.is_complete() {
        Ok(ConstrainedStop::Complete)
    } else {
        Err(RealizarError::Constraint(ConstraintError::Rejected {
            token,
            position,
            reason: "a stop token before the output was a complete document".to_string(),
        }))
    }
}

impl<F: ArchForward> Session<F> {
    /// [`Self::generate`] with every step constrained (#3568 PR 2, #3793): the same entry,
    /// prompt admission, budget, prefill and token choice, with `constraint` setting what it
    /// forbids to `-inf` before the repetition penalty and the choice.
    ///
    /// Every choice runs on host logits ([`ArchForward::forward`]), never the
    /// `forward_greedy` fast path, which picks on the device before a mask could apply.
    ///
    /// It ends at a stop token, which the constraint admits only once the output is a
    /// complete document (`Complete`), or at the budget (`Complete` if the document happens
    /// to be whole there, else `Length`). The returned tokens are the prompt plus the reply,
    /// never the stop token. Cancellation through `config.cancel` ends the turn with what was
    /// chosen so far.
    ///
    /// # Errors
    /// As [`Self::generate`], plus the constraint's refusal
    /// ([`RealizarError::Constraint`]): no token allowed, or a stop before the document was
    /// complete.
    pub fn generate_constrained(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        constraint: &mut dyn TokenConstraint,
    ) -> Result<(Turn, ConstrainedStop)> {
        use rand::SeedableRng;
        witness(Entry {
            arch: self.arch(),
            kind: EntryKind::Generate,
            session: self.id,
            digest: super::prompt_digest(prompt),
            on_gpu: self.on_gpu(),
        });
        let context_length = self.context_length();
        self.admit_prompt(prompt, context_length)?;
        let (budget, context_limited) =
            turn_budget(prompt.len(), config.max_tokens, context_length);
        self.reserve(prompt.len() + budget)?;

        let reused = self.prepare_prompt(prompt)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let mut tokens = prompt.to_vec();
        let mut stop = None;
        for generated in 0..budget {
            if config.cancel.is_cancelled() {
                break;
            }
            let next = self.constrained_choice(&tokens, config, constraint, &mut rng)?;
            if config.stop_tokens.contains(&next) {
                stop = Some(constrained_end(constraint, next, generated)?);
                break;
            }
            constraint.accept(next).map_err(RealizarError::Constraint)?;
            tokens.push(next);
        }
        let stop = match stop {
            Some(stop) => stop,
            None if constraint.is_complete() => ConstrainedStop::Complete,
            None => ConstrainedStop::Length,
        };
        let context_capped = context_limited && tokens.len() == prompt.len() + budget;
        let turn = Turn {
            tokens,
            reused,
            used_gpu: self.on_gpu(),
            context_capped,
        };
        Ok((turn, stop))
    }

    /// Bring the state to `tokens` and choose the next token under `constraint`: the mask,
    /// then the repetition penalty, then the engine's choice.
    fn constrained_choice(
        &mut self,
        tokens: &[u32],
        config: &QuantizedGenerateConfig,
        constraint: &mut dyn TokenConstraint,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<u32> {
        let (mut logits, _) = self.advance_to(tokens)?;
        constraint
            .mask(&mut logits)
            .map_err(RealizarError::Constraint)?;
        OwnedQuantizedModel::apply_repeat_penalty(
            &mut logits,
            tokens,
            config.repeat_penalty,
            config.repeat_last_n,
        );
        Ok(choose_token(&logits, config, rng))
    }
}
