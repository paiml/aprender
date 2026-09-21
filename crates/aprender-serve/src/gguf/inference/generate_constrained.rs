/// Why a constrained generation ended (#3793).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrainedStop {
    /// The output is a complete document: the model ended it (end-of-sequence, which the
    /// constraint admits only there), or the budget ended exactly on a complete document.
    Complete,
    /// The token budget ran out before the document was complete.
    Length,
}

/// Map a constraint's refusal into the crate error, TYPED (see `RealizarError::Constraint`).
pub(crate) fn constraint_refusal(e: crate::constrain::ConstraintError) -> RealizarError {
    RealizarError::Constraint(e)
}

/// The sampler both CPU loops share, unchanged (#3793): greedy at temperature 0 or `top_k` 1,
/// else seeded top-k/top-p. A constrained step calls it AFTER the mask (and, in the dense loop,
/// after the repetition penalty), exactly where the unconstrained loop calls its own.
pub(crate) fn sample_step(logits: &[f32], config: &QuantizedGenerateConfig, rng: &mut StdRng) -> u32 {
    if config.temperature == 0.0 || config.top_k == 1 {
        ops::argmax(logits)
    } else {
        OwnedQuantizedModel::sample_topk_seeded(
            logits,
            config.temperature,
            config.top_k,
            config.top_p,
            rng,
        )
    }
}

/// End-of-sequence on a constrained output: `Complete` when the document is whole (the mask
/// admits it only then), else the engine's refusal. Never a silent stop mid-document.
pub(crate) fn constrained_end(
    constraint: &mut dyn crate::constrain::TokenConstraint,
    token: u32,
    position: usize,
) -> Result<ConstrainedStop> {
    if constraint.is_complete() {
        Ok(ConstrainedStop::Complete)
    } else {
        Err(constraint_refusal(crate::constrain::ConstraintError::Rejected {
            token,
            position,
            reason: "a stop token before the output was a complete document".to_string(),
        }))
    }
}

/// Where a constrained loop stands when its budget is spent.
pub(crate) fn constrained_budget_end(
    constraint: &mut dyn crate::constrain::TokenConstraint,
) -> ConstrainedStop {
    if constraint.is_complete() {
        ConstrainedStop::Complete
    } else {
        ConstrainedStop::Length
    }
}

impl OwnedQuantizedModel {
    /// [`Self::generate_with_cache`] with every step constrained (#3568 PR 2, #3793): the same
    /// prefill, KV cache, budget and sampler, with `constraint` masking each step.
    ///
    /// It ends at end-of-sequence, which the constraint admits only once the output is a
    /// complete document (`Complete`), or at the budget (`Complete` if the document happens to
    /// be whole there, else `Length`). The returned tokens include the prompt and never the
    /// end-of-sequence token, as `generate_with_cache`'s do.
    ///
    /// # Errors
    /// A forward pass failure, or the constraint's refusal (`RealizarError::Constraint`).
    pub fn generate_with_cache_constrained(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        constraint: &mut dyn crate::constrain::TokenConstraint,
    ) -> Result<(Vec<u32>, ConstrainedStop)> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }
        let max_tokens = self.effective_max_tokens(prompt.len(), config.max_tokens)?;
        let max_seq_len = prompt.len() + max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();
        let mut rng = StdRng::seed_from_u64(config.seed);

        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        for gen_idx in 0..max_tokens {
            if config.cancel.is_cancelled() {
                break;
            }
            // mask, then this loop's own sampler: the repetition penalty, then sample_step
            constraint.mask(&mut logits).map_err(constraint_refusal)?;
            Self::apply_repeat_penalty(
                &mut logits,
                &tokens,
                config.repeat_penalty,
                config.repeat_last_n,
            );
            let next = sample_step(&logits, config, &mut rng);
            if config.stop_tokens.contains(&next) {
                return Ok((tokens, constrained_end(constraint, next, gen_idx)?));
            }
            constraint.accept(next).map_err(constraint_refusal)?;
            tokens.push(next);
            if tokens.len() >= max_seq_len {
                break;
            }
            logits = self.forward_single_with_cache(next, &mut cache, prompt.len() + gen_idx)?;
        }
        Ok((tokens, constrained_budget_end(constraint)))
    }
}
