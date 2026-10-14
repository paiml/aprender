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
//! What a batched slot does not do, as on the dense batched path: `seed`. A
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
use crate::gguf::{
    select_batched_token, OwnedQuantizedModel, Qwen35CudaModel, Qwen35CudaState, Qwen35StepOut,
};
use crate::gguf::{BatchedDecodeState, QuantizedGenerateConfig};
use crate::session::penalty_active;

/// The batch engine over a served Qwen3.5 session.
pub struct Qwen35BatchEngine {
    served: Arc<Qwen35Served>,
    /// Slot `i`'s device state. `None` only while a decode step has it out.
    states: Vec<Option<Qwen35CudaState>>,
}

impl Qwen35BatchEngine {
    /// An engine over `served`, holding no batch.
    #[must_use]
    pub fn new(served: Arc<Qwen35Served>) -> Self {
        Self {
            served,
            states: Vec::new(),
        }
    }
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

/// A slot that picks its token by argmax, the choice `select_batched_token`
/// makes for it too.
fn is_greedy(config: &QuantizedGenerateConfig) -> bool {
    config.temperature == 0.0 || config.top_k == 1
}

/// A device state for `prompt` and its whole budget, holding every prompt
/// token but the last: the first decode step runs the last one, at position
/// `prompt.len() - 1`, as the dense batched path does.
fn prefilled_state(
    model: &mut Qwen35CudaModel<'static>,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
) -> Result<Qwen35CudaState> {
    if prompt.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "qwen35_batch: an empty prompt has no token to decode from".to_string(),
        });
    }
    let mut state = model.new_state_with_len(prompt.len() + config.max_tokens)?;
    let head = &prompt[..prompt.len() - 1];
    if !head.is_empty() {
        model.prefill(head, &mut state, 0)?;
    }
    Ok(state)
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
        self.states.clear();
        let done = if let Some(model) = device_model(&mut session, &served, &prompts[0])? {
            for (prompt, config) in prompts.iter().zip(configs) {
                self.states
                    .push(Some(prefilled_state(model, prompt, config)?));
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
            next[i] = if is_greedy(c) {
                crate::gguf::ops::argmax(&logits)
            } else {
                select_batched_token(&logits, c.temperature, c.top_k, c.top_p)
            };
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
        let model = device_model(&mut session, &served, &[])?
            .ok_or_else(|| left_the_gpu("add_slot_to_batch"))?;
        self.states
            .push(Some(prefilled_state(model, &prompt, &config)?));
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
        let model = device_model(&mut session, &served, &[])?
            .ok_or_else(|| left_the_gpu("recycle_slot"))?;
        // The finished slot's state is dropped before the new one is allocated.
        self.states[slot_idx] = None;
        self.states[slot_idx] = Some(prefilled_state(model, &prompt, &config)?);
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
        self.states.clear();
    }

    fn reset_batched_state(&mut self) {
        self.states.clear();
    }
}

#[cfg(test)]
#[path = "qwen35_batch_engine_tests.rs"]
mod tests;
