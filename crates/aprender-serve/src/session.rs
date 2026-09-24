//! `realizar::session`: the one generation engine every verb drives (#4263).
//!
//! 0.69.2 had about 23 families of generation loop: `apr run`, `apr chat`, `apr
//! serve`, `bench`, `qa` and `eval` each held their own copy of prefill,
//! decode, token choice and stop handling, and the copies drifted apart. One
//! Qwen3.5 prompt served by `apr serve` and by `apr run` could reach different
//! prefill paths (#4250).
//!
//! # The seam
//!
//! [`Session`] owns the loop: the context checks, the turn budget, prefix
//! reuse, token choice, stop tokens, cancellation, the teacher-forced scoring
//! walk, and the witness. An architecture supplies only [`ArchForward`]:
//! "advance the state over these tokens from this position and give me the
//! logits", plus sizing and route reporting. It never supplies a generate
//! loop. That is where the shared attention/FFN blocks of the 0.74 "Any
//! Model" epic (#4001, #3422/#3423) plug in: a model built from shared blocks
//! is one more `ArchForward`, and every verb serves it with no new loop.
//!
//! # The contract an [`ArchForward`] owes
//!
//! - **Batched prefill.** A span of more than one token is prefilled in one
//!   batched call where the backend has one, and
//!   [`ArchForward::batched_prefills`] counts it.
//! - **Loud fallback.** A GPU failure moves the backend to the CPU, prints the
//!   reason, records it in [`ArchForward::notices`], and replays the whole
//!   span. It never re-routes silently, and it never returns logits from a
//!   state it did not finish.
//!
//! # The witness
//!
//! [`Session::generate`] and [`Session::score`] each record an [`Entry`].
//! `tests_engine_identity` checks that each verb × arch produced its entry. A
//! verb that decodes through its own loop, or calls an [`ArchForward`]
//! directly, leaves no entry and turns the guard RED.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::error::{RealizarError, Result};
use crate::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

/// One architecture's forward on one backend: the only per-arch code a verb
/// reaches, and only through a [`Session`].
pub trait ArchForward {
    /// The GGUF architecture served (`qwen35`, `qwen2`, ...). The guard keys
    /// its verb × arch matrix on it.
    fn arch(&self) -> &'static str;

    /// Whether the forward currently runs on the GPU.
    fn on_gpu(&self) -> bool;

    /// The model's declared context length, in tokens.
    fn context_length(&self) -> usize;

    /// Spans prefilled in one batched call.
    fn batched_prefills(&self) -> usize;

    /// Every line printed about the route (route notice, `Backend:` line,
    /// each fallback), in order.
    fn notices(&self) -> &[String];

    /// Make the state hold at least `positions` positions (and bind any
    /// per-thread device context) before a turn. Returns `true` when the
    /// positions the state held were dropped (a new state was allocated), so
    /// the next [`ArchForward::forward`] starts from 0.
    ///
    /// # Errors
    /// The state cannot be sized on any backend, or the caller asked for a
    /// backend that cannot hold it and refuses a fallback.
    fn reserve(&mut self, positions: usize) -> Result<bool>;

    /// Judge the backend on `probe` before it serves (the F2 guard), falling
    /// back loudly if it is rejected. The default judges nothing.
    ///
    /// # Errors
    /// The fallback itself failed.
    fn validate(&mut self, _probe: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Where to keep a copy of the state while prefilling `prompt` (#4214):
    /// the state after `prompt[..k]` is what a later prompt most likely
    /// repeats — a chat's history before its generation header. `None` (the
    /// default): this forward keeps no copies, and a prompt that does not
    /// extend what the state holds is prefilled from 0.
    fn checkpoint_at(&self, _prompt: &[u32]) -> Option<usize> {
        None
    }

    /// Keep a copy of the state as it holds now, replacing any earlier copy.
    ///
    /// # Errors
    /// The copy could not be made; the session then keeps none.
    fn save_checkpoint(&mut self) -> Result<()> {
        Ok(())
    }

    /// Return the state to the copy [`ArchForward::save_checkpoint`] kept.
    /// `false` when there is none to return to (never saved, or dropped by a
    /// reallocation or a fallback): the state is then unchanged.
    ///
    /// # Errors
    /// The copy could not be put back; what the state holds is then unknown.
    fn restore_checkpoint(&mut self) -> Result<bool> {
        Ok(false)
    }

    /// The state holds `tokens[..start]` (`start == 0`: reset it). Advance it
    /// to hold all of `tokens` and return the logits after the last one.
    ///
    /// # Errors
    /// A forward failure no fallback can recover from.
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>>;

    /// [`ArchForward::forward`] for a greedy turn, where only the argmax is
    /// wanted: advance the state the same way and return the argmax token,
    /// chosen on the device, so the logits never cross to the host (#4268: the
    /// dense CUDA decode reads back one id per token, not a vocabulary of
    /// logits). `None` means the backend has no such path and did NOTHING —
    /// the state is as it was, and the session calls
    /// [`ArchForward::forward`] instead. The default has no such path.
    ///
    /// # Errors
    /// As [`ArchForward::forward`].
    fn forward_greedy(&mut self, _tokens: &[u32], _start: usize) -> Result<Option<u32>> {
        Ok(None)
    }
}

/// What one [`Session::generate`] call did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    /// The prompt followed by the generated tokens.
    pub tokens: Vec<u32>,
    /// Leading prompt tokens served from the state an earlier call left, not
    /// prefilled again. 0 when the prompt did not extend it.
    pub reused: usize,
    /// Whether the GPU served the turn to its end.
    pub used_gpu: bool,
    /// The generation stopped at the model's declared context length, before
    /// `max_tokens` and before a stop token. It is the one reason a reply is
    /// shorter than asked for that the caller did not choose.
    pub context_capped: bool,
}

/// A loaded model plus its decode state: the one engine (#4263).
///
/// `Send` whenever its forward is, because `apr serve` drives a session from
/// blocking-pool threads.
pub struct Session<F: ArchForward> {
    forward: F,
    /// The tokens whose forward the state holds, in order.
    processed: Vec<u32>,
    /// Unique per session in this process: the witness names the session a
    /// call entered, so a guard can ask "did THIS verb's session serve it?"
    /// without re-deriving the verb's prompt tokens.
    id: u64,
    /// The tokens whose forward the forward's saved copy holds (#4214), or
    /// `None` when it keeps no copy the session trusts.
    checkpoint: Option<Vec<u32>>,
}

static NEXT_SESSION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl<F: ArchForward> Session<F> {
    /// Wrap a loaded forward. The state starts empty.
    pub fn new(forward: F) -> Self {
        Self {
            forward,
            processed: Vec::new(),
            id: NEXT_SESSION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            checkpoint: None,
        }
    }

    /// This session's witness id (see [`entries_of_session`]).
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The architecture's forward, for route reporting and tests.
    pub fn engine(&self) -> &F {
        &self.forward
    }

    /// The architecture's forward, mutably: for tests that set a forward's
    /// knobs. A verb never drives the forward directly.
    pub(crate) fn engine_mut(&mut self) -> &mut F {
        &mut self.forward
    }

    /// See [`ArchForward::arch`].
    #[must_use]
    pub fn arch(&self) -> &'static str {
        self.forward.arch()
    }

    /// See [`ArchForward::on_gpu`].
    #[must_use]
    pub fn on_gpu(&self) -> bool {
        self.forward.on_gpu()
    }

    /// See [`ArchForward::context_length`].
    #[must_use]
    pub fn context_length(&self) -> usize {
        self.forward.context_length()
    }

    /// See [`ArchForward::batched_prefills`].
    #[must_use]
    pub fn batched_prefills(&self) -> usize {
        self.forward.batched_prefills()
    }

    /// See [`ArchForward::notices`].
    #[must_use]
    pub fn notices(&self) -> &[String] {
        self.forward.notices()
    }

    /// Positions the decode state currently holds.
    #[must_use]
    pub fn processed_len(&self) -> usize {
        self.processed.len()
    }

    /// `tokens` strictly extends what the state holds.
    fn extends(&self, tokens: &[u32]) -> bool {
        !self.processed.is_empty()
            && tokens.len() > self.processed.len()
            && tokens.starts_with(&self.processed)
    }

    /// Make the state hold exactly `tokens`; return the logits after the last
    /// one and how many leading tokens were already held.
    fn advance_to(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, usize)> {
        let start = if self.extends(tokens) {
            self.processed.len()
        } else {
            0
        };
        Ok((self.step(tokens, start)?, start))
    }

    /// Bring the state up to a strict prefix of `prompt` that
    /// [`Self::advance_and_choose`] then extends with the rest: resume from the
    /// checkpoint an earlier turn left when `prompt` does not extend the state
    /// but repeats what the checkpoint holds (#4214: the same prompt again;
    /// #4274: a chat history re-rendered with the last reply changed), and
    /// prefill up to a new checkpoint where [`ArchForward::checkpoint_at`] says.
    /// Returns how many leading tokens of `prompt` were already held.
    fn prepare_prompt(&mut self, prompt: &[u32]) -> Result<usize> {
        let mut start = if self.extends(prompt) {
            self.processed.len()
        } else {
            0
        };
        if start == 0 {
            if let Some(held) = self.checkpoint.take() {
                if prompt.len() > held.len() && prompt.starts_with(&held) {
                    match self.forward.restore_checkpoint() {
                        Ok(true) => {
                            start = held.len();
                            self.processed.clone_from(&held);
                            self.checkpoint = Some(held);
                        },
                        Ok(false) => {},
                        Err(e) => {
                            self.processed.clear();
                            return Err(e);
                        },
                    }
                } else {
                    self.checkpoint = Some(held);
                }
            }
        }
        let reused = start;
        if let Some(k) = self
            .forward
            .checkpoint_at(prompt)
            .filter(|&k| k > start && k < prompt.len())
        {
            self.step(&prompt[..k], start)?;
            self.checkpoint = match self.forward.save_checkpoint() {
                Ok(()) => Some(prompt[..k].to_vec()),
                Err(_) => None,
            };
        }
        Ok(reused)
    }

    /// Forward `tokens` from `start` (the state holds `tokens[..start]`) and
    /// record what the state then holds.
    fn step(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        // Positions under the checkpoint are about to be rewritten: the copy
        // no longer matches the attention rows it would resume on.
        if self.checkpoint.as_ref().is_some_and(|c| start < c.len()) {
            self.checkpoint = None;
        }
        match self.forward.forward(tokens, start) {
            Ok(logits) => {
                self.processed.truncate(start);
                self.processed.extend_from_slice(&tokens[start..]);
                Ok(logits)
            },
            Err(e) => {
                // What the state holds after a failed forward is unknown.
                self.processed.clear();
                self.checkpoint = None;
                Err(e)
            },
        }
    }

    /// Make the state hold exactly `tokens` and choose the token after them;
    /// return it and how many leading tokens were already held. A greedy
    /// choice with no repetition penalty goes through
    /// [`ArchForward::forward_greedy`] when the backend has it.
    fn advance_and_choose(
        &mut self,
        tokens: &[u32],
        config: &QuantizedGenerateConfig,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(u32, usize)> {
        if is_greedy(config) && !penalty_active(config) {
            let start = if self.extends(tokens) {
                self.processed.len()
            } else {
                0
            };
            if self.checkpoint.as_ref().is_some_and(|c| start < c.len()) {
                self.checkpoint = None;
            }
            match self.forward.forward_greedy(tokens, start) {
                Ok(Some(next)) => {
                    self.processed.truncate(start);
                    self.processed.extend_from_slice(&tokens[start..]);
                    return Ok((next, start));
                },
                Ok(None) => {},
                Err(e) => {
                    self.processed.clear();
                    self.checkpoint = None;
                    return Err(e);
                },
            }
        }
        let (mut logits, reused) = self.advance_to(tokens)?;
        OwnedQuantizedModel::apply_repeat_penalty(
            &mut logits,
            tokens,
            config.repeat_penalty,
            config.repeat_last_n,
        );
        Ok((choose_token(&logits, config, rng), reused))
    }

    fn reserve(&mut self, positions: usize) -> Result<()> {
        if self.forward.reserve(positions)? {
            self.processed.clear();
            self.checkpoint = None;
        }
        Ok(())
    }

    /// Generate from `prompt` with `config`'s token choice and stop tokens.
    /// `on_token` is called with each new token as it is chosen; returning
    /// `false` ends the turn after that token. A prompt that strictly extends
    /// what the state holds prefills only the new suffix.
    ///
    /// The token choice is argmax at temperature 0 or `top_k` 1, else seeded
    /// top-k/top-p. Cancellation through `config.cancel` ends the turn early
    /// with the tokens chosen so far.
    ///
    /// # Errors
    /// An empty prompt; a prompt the declared context cannot hold with room to
    /// answer (refused whole, never truncated); a forward failure no fallback
    /// can recover from.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Turn> {
        use rand::SeedableRng;
        witness(Entry {
            arch: self.arch(),
            kind: EntryKind::Generate,
            session: self.id,
            digest: prompt_digest(prompt),
            on_gpu: self.on_gpu(),
        });
        let arch = self.arch();
        let context_length = self.context_length();
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: format!("{arch} session: the prompt is empty"),
            });
        }
        if prompt.len() >= context_length {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "{arch} session: the prompt is {} tokens and this model declares a context of \
                     {context_length} (context_length in the GGUF) — it cannot fit with room to \
                     answer, so it was refused whole rather than truncated",
                    prompt.len(),
                ),
            });
        }
        // Every position the turn can reach, reserved up front so the state
        // never grows (and never re-prefills) mid-generation.
        let (budget, context_limited) =
            turn_budget(prompt.len(), config.max_tokens, context_length);
        self.reserve(prompt.len() + budget)?;

        let reused = self.prepare_prompt(prompt)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let (mut next, _) = self.advance_and_choose(prompt, config, &mut rng)?;
        let mut tokens = prompt.to_vec();
        let mut context_capped = false;
        for generated in 1..=budget {
            if config.cancel.is_cancelled() {
                break;
            }
            tokens.push(next);
            let keep_going = on_token(next);
            if !keep_going || config.stop_tokens.contains(&next) {
                break;
            }
            if generated == budget {
                context_capped = context_limited;
                break;
            }
            next = self.advance_and_choose(&tokens, config, &mut rng)?.0;
        }
        Ok(Turn {
            tokens,
            reused,
            used_gpu: self.on_gpu(),
            context_capped,
        })
    }

    /// Teacher-forced scoring, for perplexity: forward `tokens` one position
    /// at a time from 0 and call `on_logits(pos, logits)` with the logits after
    /// each position `pos` (the prediction of `tokens[pos + 1]`). Returning
    /// `false` stops early. The backend is judged on the whole sequence first,
    /// as it is judged on a prompt.
    ///
    /// # Errors
    /// An empty sequence; one longer than the declared context; a forward
    /// failure no fallback can recover from.
    pub fn score(
        &mut self,
        tokens: &[u32],
        on_logits: &mut dyn FnMut(usize, &[f32]) -> bool,
    ) -> Result<()> {
        witness(Entry {
            arch: self.arch(),
            kind: EntryKind::Score,
            session: self.id,
            digest: prompt_digest(tokens),
            on_gpu: self.on_gpu(),
        });
        if tokens.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: format!("{} session: the sequence to score is empty", self.arch()),
            });
        }
        if tokens.len() > self.context_length() {
            return Err(RealizarError::ContextLimitExceeded {
                provided: tokens.len(),
                maximum: self.context_length(),
            });
        }
        self.reserve(tokens.len())?;
        self.forward.validate(tokens)?;
        for pos in 0..tokens.len() {
            let (logits, _) = self.advance_to(&tokens[..=pos])?;
            if !on_logits(pos, &logits) {
                break;
            }
        }
        Ok(())
    }
}

/// How many tokens a turn may generate: `max_tokens`, or fewer when the
/// declared context ends first, and whether it did. The caller has already
/// refused a prompt of `context_length` tokens or more.
pub(crate) fn turn_budget(
    prompt_len: usize,
    max_tokens: usize,
    context_length: usize,
) -> (usize, bool) {
    let room = context_length.saturating_sub(prompt_len);
    (max_tokens.min(room), room < max_tokens)
}

/// Whether `config` asks for the argmax: temperature 0 or `top_k` 1.
pub(crate) fn is_greedy(config: &QuantizedGenerateConfig) -> bool {
    config.temperature == 0.0 || config.top_k == 1
}

/// Whether `config`'s repetition penalty changes any logit (#4268: the dense
/// loops applied it before every choice, so the engine does too).
pub(crate) fn penalty_active(config: &QuantizedGenerateConfig) -> bool {
    config.repeat_penalty != 1.0 && config.repeat_last_n > 0
}

/// The engine's token choice: argmax at temperature 0 or `top_k` 1, else
/// seeded top-k/top-p.
pub(crate) fn choose_token(
    logits: &[f32],
    config: &QuantizedGenerateConfig,
    rng: &mut rand::rngs::StdRng,
) -> u32 {
    if is_greedy(config) {
        crate::gguf::ops::argmax(logits)
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

/// Which entry point a verb came through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// [`Session::generate`].
    Generate,
    /// [`Session::score`].
    Score,
}

/// One call into the engine, as the witness recorded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// [`ArchForward::arch`] of the session that served the call.
    pub arch: &'static str,
    /// Which entry point.
    pub kind: EntryKind,
    /// [`prompt_digest`] of the prompt (or scored sequence).
    pub digest: u64,
    /// Whether the session was on the GPU when the call entered.
    pub on_gpu: bool,
    /// [`Session::id`] of the session that served the call.
    pub session: u64,
}

/// How many entries the witness keeps. The guard picks prompts no other test
/// uses, so it only needs its own entries to survive the calls made in
/// parallel with it.
const WITNESS_CAPACITY: usize = 1024;

static WITNESS: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());

/// A stable digest of a token sequence (FNV-1a over the little-endian ids),
/// so the witness holds no prompt text.
#[must_use]
pub fn prompt_digest(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn witness(entry: Entry) {
    // A poisoned witness still records: the guard must not go vacuous because
    // an unrelated test panicked while holding the lock.
    let mut w = WITNESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if w.len() == WITNESS_CAPACITY {
        w.pop_front();
    }
    w.push_back(entry);
}

/// Every retained entry whose digest is `prompt_digest(tokens)`, oldest first.
#[must_use]
pub fn entries_for(tokens: &[u32]) -> Vec<Entry> {
    let digest = prompt_digest(tokens);
    WITNESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|e| e.digest == digest)
        .cloned()
        .collect()
}

/// Every retained entry the session `id` served, oldest first.
#[must_use]
pub fn entries_of_session(id: u64) -> Vec<Entry> {
    WITNESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|e| e.session == id)
        .cloned()
        .collect()
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
