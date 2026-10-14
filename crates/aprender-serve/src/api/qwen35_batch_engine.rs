//! #5279 P2 (#5293): the Qwen3.5 hybrid behind the CUDA batch scheduler.
//!
//! [`Qwen35BatchEngine`] is a [`BatchEngine`] over the server's resident
//! [`Qwen35Session`](crate::gguf::qwen35_session::Qwen35Session). A lone
//! request is one turn of the session, exactly as the unbatched server served
//! it. A batch of m > 1 gives each request a device state of its own beside
//! the session's, prefilled from its prompt and stepped together through
//! [`Qwen35CudaModel::forward_batch`]. A plain greedy slot gets its token
//! from the device; a slot with logprobs (#4971), a repetition penalty or
//! sampling gets its logits from the same step and picks its token from
//! them, in the session's order. The session's state, and the prefix it
//! keeps for its next turn, are never touched by a batch.
//!
//! A finished slot's state is kept, with a copy of its recurrent half taken
//! where the session would take one (a chat's history, else the prompt but
//! its last token). A later prompt that begins with a kept copy's tokens is
//! resumed from it and prefills only the rest (#5279 P7).
//!
//! A sampled slot draws from its own sampler, seeded from its request's
//! `seed` as its session turn's is, so it samples what it would alone. A
//! batch the session serves from the CPU records no logprobs. The chat
//! handler keeps a request that asks for a trace on the session.

use std::sync::Arc;

use super::cuda_batch_scheduler::{
    BatchEngine, CudaBatchConfig, CudaBatchRequest, RecordSink, TokenCallback,
};
use crate::api::{AppState, Qwen35Served};
use crate::error::{RealizarError, Result};
use crate::gguf::logprobs::StepLogprobs;
use crate::gguf::qwen35_session::Qwen35Session;
use crate::gguf::Qwen35CudaCheckpoint;
use crate::gguf::{BatchedDecodeState, QuantizedGenerateConfig};
use crate::gguf::{OwnedQuantizedModel, Qwen35CudaModel, Qwen35CudaState, Qwen35StepOut};
use crate::session::{choose_token, penalty_active};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// #5279 P7: finished slots' states kept for later admissions, at most.
const POOL_MAX: usize = 16;

/// #5279 P7: a slot's state is allocated for a whole number of these
/// positions, so a later, slightly longer prompt still fits it.
const STATE_GRANULE: usize = 256;

/// A device state and the prefix it can resume from: `copy` holds its
/// recurrent half after `tokens` (empty: no usable copy), and its KV rows
/// below `tokens.len()` still hold those tokens.
struct Held {
    state: Qwen35CudaState,
    tokens: Vec<u32>,
    copy: Option<Qwen35CudaCheckpoint>,
}

/// The batch engine over a served Qwen3.5 session.
pub struct Qwen35BatchEngine {
    served: Arc<Qwen35Served>,
    /// Slot `i`'s device state. `None` only while a decode step has it out.
    states: Vec<Option<Qwen35CudaState>>,
    /// Slot `i`'s prefix copy, `(tokens, copy)` as in [`Held`].
    copies: Vec<(Vec<u32>, Option<Qwen35CudaCheckpoint>)>,
    /// #5279 P7: finished slots' states, for a later prompt that repeats the
    /// prefix one of them holds.
    pool: Vec<Held>,
    /// Slot `i`'s sampler, seeded from its request's `seed` as its session
    /// turn's is, so a sampled slot draws what it would draw alone.
    rngs: Vec<StdRng>,
    /// A decode step failed, so the batch's states are unknown: none is pooled.
    poisoned: bool,
    /// Prompt rows that admissions resumed from a kept copy instead of
    /// prefilling.
    reused_rows: u64,
}

impl Qwen35BatchEngine {
    /// An engine over `served`, holding no batch.
    #[must_use]
    pub fn new(served: Arc<Qwen35Served>) -> Self {
        Self {
            served,
            states: Vec::new(),
            copies: Vec::new(),
            pool: Vec::new(),
            rngs: Vec::new(),
            poisoned: false,
            reused_rows: 0,
        }
    }

    /// #5279 P7: prompt rows that admissions resumed from a kept prefix
    /// instead of prefilling.
    #[must_use]
    pub fn reused_rows(&self) -> u64 {
        self.reused_rows
    }

    /// Take slot `i`'s state out, with its copy; `None` when it has none.
    fn release(&mut self, i: usize) -> Option<Held> {
        let state = self.states.get_mut(i)?.take()?;
        let (tokens, copy) = std::mem::take(&mut self.copies[i]);
        Some(Held {
            state,
            tokens,
            copy,
        })
    }

    /// Keep `held` for a later admission, dropping the oldest past
    /// [`POOL_MAX`].
    fn keep(&mut self, held: Held) {
        self.pool.push(held);
        if self.pool.len() > POOL_MAX {
            self.pool.remove(0);
        }
    }

    /// Pool every slot's state, unless a failed step left them unknown.
    fn pool_all(&mut self) {
        let poisoned = std::mem::take(&mut self.poisoned);
        for i in 0..self.states.len() {
            if let (Some(held), false) = (self.release(i), poisoned) {
                self.keep(held);
            }
        }
        self.states.clear();
        self.copies.clear();
    }

    /// A state holding every token of `prompt` but the last, resumed from
    /// the pooled state that holds the longest prefix of it, or allocated.
    /// With `split`, the state's recurrent half is copied after
    /// `prompt[..split]` on the way.
    fn admit(
        &mut self,
        model: &mut Qwen35CudaModel<'static>,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        split: Option<usize>,
    ) -> Result<Held> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "qwen35_batch: an empty prompt has no token to decode from".to_string(),
            });
        }
        let head = &prompt[..prompt.len() - 1];
        let need = prompt.len() + config.max_tokens;
        let held = match take_best(&mut self.pool, head, need) {
            Some(held) => held,
            None => {
                let len = need
                    .next_multiple_of(STATE_GRANULE)
                    .min(self.served.context_length)
                    .max(need);
                Held {
                    state: model.new_state_with_len(len)?,
                    tokens: Vec::new(),
                    copy: None,
                }
            },
        };
        let (held, reused) = prefilled_state(model, head, held, split)?;
        self.reused_rows += reused as u64;
        Ok(held)
    }

    /// Seat `held` in slot `i`, or in a new last slot.
    fn seat(&mut self, i: Option<usize>, held: Held) {
        let Held {
            state,
            tokens,
            copy,
        } = held;
        if let Some(i) = i {
            self.states[i] = Some(state);
            self.copies[i] = (tokens, copy);
        } else {
            self.states.push(Some(state));
            self.copies.push((tokens, copy));
        }
    }
}

/// Take from `pool` the state that resumes `head` furthest and has room for
/// `need` positions; else the one with room whose copy is worth least. `None`
/// when none has room.
fn take_best(pool: &mut Vec<Held>, head: &[u32], need: usize) -> Option<Held> {
    let rank = |h: &Held| {
        if !h.tokens.is_empty() && head.starts_with(&h.tokens) {
            (1, h.tokens.len())
        } else {
            (0, usize::MAX - h.tokens.len())
        }
    };
    let best = (0..pool.len())
        .filter(|&i| pool[i].state.max_seq_len() >= need)
        .max_by_key(|&i| rank(&pool[i]))?;
    Some(pool.remove(best))
}

impl AppState {
    /// #5293: run the Qwen3.5 session's chat requests through a CUDA batch
    /// scheduler, so c concurrent requests decode as one batch. A state with no
    /// Qwen3.5 session is returned unchanged.
    #[must_use]
    pub fn with_qwen35_batch_scheduler(self, config: CudaBatchConfig) -> Self {
        let Some(served) = self.qwen35_session() else {
            return self;
        };
        let report = config.report(crate::api::admission_ceiling_reason(None));
        let in_flight = crate::api::InFlightCounter::new();
        let engine = Arc::new(std::sync::RwLock::new(Qwen35BatchEngine::new(served)));
        let tx = super::cuda_batch_scheduler::spawn_cuda_batch_scheduler(
            engine,
            config,
            in_flight.clone(),
        );
        self.with_cuda_batch_tx(tx)
            .with_scheduler_report(report, Some(in_flight))
    }
}

/// The session behind `served`, or why it cannot be had.
fn lock(served: &Qwen35Served) -> Result<std::sync::MutexGuard<'_, Qwen35Session>> {
    served
        .session
        .lock()
        .map_err(|_| RealizarError::UnsupportedOperation {
            operation: "qwen35_batch".to_string(),
            reason: "an earlier request panicked while holding the session".to_string(),
        })
}

/// A slot that picks its token by argmax, the choice [`choose_token`] makes
/// for it too.
fn is_greedy(config: &QuantizedGenerateConfig) -> bool {
    config.temperature == 0.0 || config.top_k == 1
}

/// A slot's sampler, seeded as its session turn's is.
fn seeded(config: &QuantizedGenerateConfig) -> StdRng {
    StdRng::seed_from_u64(config.seed)
}

/// `held` advanced to hold `head`, a prompt's every token but the last: the
/// first decode step runs the last one, at position `head.len()`, as the
/// dense batched path does. A copy whose tokens begin `head` is restored and
/// only the rest is prefilled (#5279 P7); otherwise the state starts from
/// zero. With `split`, the recurrent half is copied after `head[..split]`.
/// Returns the state and the rows it resumed instead of prefilling.
fn prefilled_state(
    model: &mut Qwen35CudaModel<'static>,
    head: &[u32],
    mut held: Held,
    split: Option<usize>,
) -> Result<(Held, usize)> {
    let resumes = !held.tokens.is_empty() && head.starts_with(&held.tokens);
    let restored = match (&held.copy, resumes) {
        (Some(copy), true) => model.restore_checkpoint(copy, &mut held.state)?,
        _ => false,
    };
    let start = if restored {
        held.tokens.len()
    } else {
        model.reset_state(&mut held.state)?;
        held.tokens.clear();
        0
    };
    let mut pos = start;
    if let Some(k) = split.filter(|&k| k > start && k <= head.len()) {
        model.prefill(&head[start..k], &mut held.state, start)?;
        model.save_checkpoint(&held.state, &mut held.copy)?;
        held.tokens = head[..k].to_vec();
        pos = k;
    }
    if pos < head.len() {
        model.prefill(&head[pos..], &mut held.state, pos)?;
    }
    Ok((held, start))
}

/// The device model, or `None` when the session serves from the CPU.
fn device_model<'s>(
    session: &'s mut Qwen35Session,
    served: &Qwen35Served,
    probe: &[u32],
) -> Result<Option<&'s mut Qwen35CudaModel<'static>>> {
    let model = session.batch_device_model(probe)?;
    served
        .on_gpu
        .store(model.is_some(), std::sync::atomic::Ordering::Relaxed);
    Ok(model)
}

/// A batch that cannot be had: `operation` refused because the session left
/// the GPU after the batch formed.
fn left_the_gpu(operation: &str) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: "the Qwen3.5 session is serving from the CPU".to_string(),
    }
}

/// One session turn for `prompt`, its tokens to `on_token` without the stop
/// token that ends it. `false` from `on_token` ends the turn. With `logprobs`
/// (#4971) each token's record goes to its sink ahead of the token.
fn session_turn(
    session: &mut Qwen35Session,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
    logprobs: Option<&RecordSink>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<crate::session::Turn> {
    let top_n = logprobs.map(|sink| sink.top_n);
    session.generate_with_logprobs(prompt, config, top_n, &mut |token, record| {
        if config.stop_tokens.contains(&token) {
            return true;
        }
        if let (Some(sink), Some(record)) = (logprobs, record) {
            let _ = sink.records.send(record);
        }
        on_token(token)
    })
}

impl BatchEngine for Qwen35BatchEngine {
    fn generate_single(&mut self, req: CudaBatchRequest) {
        if req.prompt_ids.is_empty() {
            return;
        }
        let served = Arc::clone(&self.served);
        let mut session = match lock(&served) {
            Ok(s) => s,
            Err(e) => {
                let _ = req.token_tx.try_send(Err(e.to_string()));
                return;
            },
        };
        let mut tokens = Vec::new();
        let logprobs = req.logprobs.as_ref();
        let result = session_turn(
            &mut session,
            &req.prompt_ids,
            &req.config,
            logprobs,
            &mut |t| {
                if req.non_streaming {
                    tokens.push(t);
                    true
                } else {
                    req.token_tx.try_send(Ok(t)).is_ok()
                }
            },
        );
        let decode_ended = std::time::Instant::now();
        served
            .on_gpu
            .store(session.on_gpu(), std::sync::atomic::Ordering::Relaxed);
        drop(session);
        match result {
            Ok(turn) => {
                for t in tokens {
                    if req.token_tx.try_send(Ok(t)).is_err() {
                        break;
                    }
                }
                if let Some(tx) = req.timing_tx {
                    let _ = tx.send(crate::api::PhaseTimings::from_marks(
                        turn.prefill_started,
                        turn.prefill_ended,
                        decode_ended,
                    ));
                }
            },
            Err(e) => {
                let _ = req.token_tx.try_send(Err(e.to_string()));
            },
        }
    }

    fn batched_setup_and_prefill(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        mut on_tokens: Vec<TokenCallback>,
        max_kv_slots: usize,
    ) -> Result<BatchedDecodeState> {
        let m = prompts.len();
        if m == 0 || configs.len() != m || on_tokens.len() != m || prompts.iter().any(Vec::is_empty)
        {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_batch: {m} prompts (none may be empty), {} configs and {} callbacks",
                    configs.len(),
                    on_tokens.len()
                ),
            });
        }
        let served = Arc::clone(&self.served);
        let mut session = lock(&served)?;
        self.pool_all();
        self.rngs = configs.iter().map(seeded).collect();
        let splits: Vec<_> = prompts
            .iter()
            .map(|p| session.batch_checkpoint_at(p))
            .collect();
        let done = if let Some(model) = device_model(&mut session, &served, &prompts[0])? {
            for ((prompt, config), &split) in prompts.iter().zip(configs).zip(&splits) {
                let held = self.admit(model, prompt, config, split)?;
                self.seat(None, held);
            }
            vec![false; m]
        } else {
            // The GPU was rejected (or never held): the batch is served one
            // request after another by the session, and is over before Phase 2.
            for ((prompt, config), on_token) in prompts.iter().zip(configs).zip(&mut on_tokens) {
                session_turn(&mut session, prompt, config, None, &mut **on_token)?;
            }
            vec![true; m]
        };
        eprintln!("[#5293] Qwen3.5 batch: m={m} prefilled");
        Ok(BatchedDecodeState {
            m,
            max_kv_slots,
            max_tokens_max: configs.iter().map(|c| c.max_tokens).max().unwrap_or(0),
            gen_idx: 0,
            hidden_dim: 0,
            intermediate_dim: 0,
            num_layers: 0,
            vocab_size: 0,
            eps: 0.0,
            prompts: prompts.to_vec(),
            configs: configs.to_vec(),
            on_tokens,
            sequences: prompts.to_vec(),
            positions: prompts.iter().map(|p| p.len() - 1).collect(),
            last_tokens: prompts.iter().map(|p| p[p.len() - 1]).collect(),
            done,
            embed_buf: Vec::new(),
            pos_buf: Vec::new(),
            recorders: (0..m).map(|_| None).collect(),
            records: (0..m).map(|_| None).collect(),
        })
    }

    fn batched_decode_step(&mut self, state: &mut BatchedDecodeState) -> Result<Vec<u32>> {
        let served = Arc::clone(&self.served);
        let mut session = lock(&served)?;
        let model = device_model(&mut session, &served, &[])?
            .ok_or_else(|| left_the_gpu("qwen35_batch_decode"))?;
        // Every live slot steps in the batch. A slot that needs its logits on
        // the host (logprobs, a repetition penalty or sampling) gets them from
        // the batched step, and picks its token as its session turn does.
        let live: Vec<usize> = (0..state.m).filter(|&i| !state.done[i]).collect();
        let host: Vec<bool> = live
            .iter()
            .map(|&i| {
                let c = &state.configs[i];
                !(is_greedy(c) && !penalty_active(c) && state.recorders[i].is_none())
            })
            .collect();
        let mut next = state.last_tokens.clone();

        // The states are lent to the batch and returned before any error is
        // passed on.
        let mut lent: Vec<Qwen35CudaState> = live
            .iter()
            .map(|&i| self.states[i].take().expect("a live slot holds its state"))
            .collect();
        let tokens: Vec<u32> = live.iter().map(|&i| state.last_tokens[i]).collect();
        let positions: Vec<usize> = live.iter().map(|&i| state.positions[i]).collect();
        let stepped = if live.is_empty() {
            Ok(Vec::new())
        } else {
            model.forward_batch(&tokens, &mut lent, &positions, &host)
        };
        for (&i, s) in live.iter().zip(lent) {
            self.states[i] = Some(s);
        }
        self.poisoned |= stepped.is_err();

        for (&i, out) in live.iter().zip(stepped?) {
            let mut logits = match out {
                Qwen35StepOut::Token(t) => {
                    next[i] = t;
                    continue;
                },
                Qwen35StepOut::Logits(logits) => logits,
            };
            let c = &state.configs[i];
            // The session's order: the penalty over everything the slot holds,
            // then argmax (greedy) or a sample; a record reads the logits after
            // the penalty, before temperature.
            OwnedQuantizedModel::apply_repeat_penalty(
                &mut logits,
                &state.sequences[i],
                c.repeat_penalty,
                c.repeat_last_n,
            );
            next[i] = choose_token(&logits, c, &mut self.rngs[i]);
            if let Some(top_n) = state.recorders[i].as_ref().map(|r| r.top_n) {
                let step = state.sequences[i].len() - state.prompts[i].len();
                state.records[i] = Some(StepLogprobs::of(step, next[i], &logits, top_n));
            }
        }
        Ok(next)
    }

    fn add_slot_to_batch(
        &mut self,
        state: &mut BatchedDecodeState,
        prompt: Vec<u32>,
        config: QuantizedGenerateConfig,
        on_token: TokenCallback,
    ) -> Result<()> {
        if state.m >= state.max_kv_slots {
            return Err(RealizarError::UnsupportedOperation {
                operation: "add_slot_to_batch".to_string(),
                reason: format!("batch full (m={}, max={})", state.m, state.max_kv_slots),
            });
        }
        let served = Arc::clone(&self.served);
        let mut session = lock(&served)?;
        let split = session.batch_checkpoint_at(&prompt);
        let model = device_model(&mut session, &served, &[])?
            .ok_or_else(|| left_the_gpu("add_slot_to_batch"))?;
        let held = self.admit(model, &prompt, &config, split)?;
        self.seat(None, held);
        self.rngs.push(seeded(&config));
        state.max_tokens_max = state.max_tokens_max.max(state.gen_idx + config.max_tokens);
        state.positions.push(prompt.len() - 1);
        state.last_tokens.push(prompt[prompt.len() - 1]);
        state.sequences.push(prompt.clone());
        state.prompts.push(prompt);
        state.configs.push(config);
        state.on_tokens.push(on_token);
        state.done.push(false);
        state.recorders.push(None);
        state.records.push(None);
        state.m += 1;
        Ok(())
    }

    fn recycle_slot(
        &mut self,
        state: &mut BatchedDecodeState,
        slot_idx: usize,
        prompt: Vec<u32>,
        config: QuantizedGenerateConfig,
        on_token: TokenCallback,
    ) -> Result<()> {
        if !state.done.get(slot_idx).copied().unwrap_or(false) {
            return Err(RealizarError::UnsupportedOperation {
                operation: "recycle_slot".to_string(),
                reason: format!("slot {slot_idx} is not done"),
            });
        }
        let served = Arc::clone(&self.served);
        let mut session = lock(&served)?;
        let split = session.batch_checkpoint_at(&prompt);
        let model = device_model(&mut session, &served, &[])?
            .ok_or_else(|| left_the_gpu("recycle_slot"))?;
        // The finished slot's state goes to the pool first, so the new prompt
        // can resume from it or reuse its buffers.
        if let Some(held) = self.release(slot_idx) {
            self.keep(held);
        }
        let held = self.admit(model, &prompt, &config, split)?;
        self.seat(Some(slot_idx), held);
        self.rngs[slot_idx] = seeded(&config);
        state.max_tokens_max = state.max_tokens_max.max(state.gen_idx + config.max_tokens);
        state.positions[slot_idx] = prompt.len() - 1;
        state.last_tokens[slot_idx] = prompt[prompt.len() - 1];
        state.sequences[slot_idx].clone_from(&prompt);
        state.prompts[slot_idx] = prompt;
        state.configs[slot_idx] = config;
        state.on_tokens[slot_idx] = on_token;
        state.done[slot_idx] = false;
        state.recorders[slot_idx] = None;
        state.records[slot_idx] = None;
        Ok(())
    }

    fn batched_cleanup(&mut self, _state: &BatchedDecodeState) {
        self.pool_all();
    }

    /// A failed setup may have left the device in any state: nothing is kept.
    fn reset_batched_state(&mut self) {
        self.states.clear();
        self.copies.clear();
        self.pool.clear();
        self.rngs.clear();
        self.poisoned = false;
    }
}

#[cfg(test)]
#[path = "qwen35_batch_engine_tests.rs"]
mod tests;
