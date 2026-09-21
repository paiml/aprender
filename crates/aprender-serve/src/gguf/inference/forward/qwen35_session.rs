//! #3595 (`apr chat`) / #3571 (`apr serve`): a Qwen3.5 hybrid held resident for
//! a whole session.
//!
//! [`run_qwen35_generate_dispatch`](crate::gguf::forward_qwen35::run_qwen35_generate_dispatch)
//! serves ONE call: it builds the hybrid, uploads it, runs the F2 guard (which
//! hashes the whole file), prefills the prompt token by token, decodes, and
//! drops all of it on return. That is the right shape for `apr run`. It was
//! also what `apr chat` did on every turn, measured at 0.69.0 on both GPU hosts
//! (#3595): the weights re-uploaded per message (9B on the RTX 4090:
//! 1187 → 6936 → 1576 MiB, twice), the file re-hashed (27B: 8 s a turn), the
//! whole conversation re-prefilled, the PTX re-JIT'd on sm_121 — and a turn-2
//! re-allocation that lost the device to a neighbour fell back to the CPU
//! mid-conversation.
//!
//! [`Qwen35Session`] does each of those once. Its decode state outlives the
//! call: a prompt that EXTENDS the tokens the state holds prefills only the new
//! suffix; any other prompt resets the state in place and prefills whole — on
//! the same resident model, never a rebuilt one. The state grows (doubling) up
//! to the model's declared context length; a prompt that cannot fit is refused
//! by name, and a generation cut short by the context says so in
//! [`Qwen35Turn::context_capped`]. Nothing is truncated silently.

use crate::error::{RealizarError, Result};
use crate::gguf::forward_qwen35::{
    qwen35_route, qwen35_route_notice, Qwen35Model, Qwen35Route, Qwen35State,
    QWEN35_GPU_FALLBACK_PREFIX,
};
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

/// The smallest decode state a session allocates. Growing by doubling from
/// here keeps a short conversation from paying for the model's whole declared
/// context (262 144 positions for every Qwen3.5 size).
const MIN_CAPACITY: usize = 4096;

/// What one [`Qwen35Session::generate`] call did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Turn {
    /// The prompt followed by the generated tokens — the shape
    /// `run_qwen35_generate` returns.
    pub tokens: Vec<u32>,
    /// Leading prompt tokens served from the state an earlier call left, not
    /// prefilled again. 0 when the prompt did not extend it.
    pub reused: usize,
    /// Whether the GPU served the turn to its end.
    pub used_gpu: bool,
    /// The generation stopped at the model's declared context length, before
    /// `max_tokens` and before a stop token — the one reason a reply is shorter
    /// than asked for that the caller did not choose.
    pub context_capped: bool,
}

/// Why a step could not complete: a GPU failure the session recovers from by
/// moving to the CPU, or anything else, which the caller gets.
enum Step {
    Gpu(String),
    Fatal(RealizarError),
}

impl From<RealizarError> for Step {
    fn from(e: RealizarError) -> Self {
        Self::Fatal(e)
    }
}

/// The device side of a GPU session.
#[cfg(feature = "cuda")]
struct GpuBackend {
    model: crate::gguf::cuda::Qwen35CudaModel<'static>,
    /// The session's decode state; `None` until the first turn sizes it.
    state: Option<crate::gguf::cuda::Qwen35CudaState>,
    device_name: String,
    /// The receipt key's model half, hashed once at load.
    hash: crate::gguf::forward_qwen35::Qwen35ModelHash,
    /// The F2 guard has accepted this device model (it runs once, on the first
    /// turn's real prompt).
    validated: bool,
}

#[cfg(feature = "cuda")]
impl GpuBackend {
    /// Build the hybrid on the device, once.
    fn build(
        qwen: &'static Qwen35Model<'static>,
        mapped: &MappedGGUFModel,
        notices: &mut Vec<String>,
    ) -> std::result::Result<Self, String> {
        use crate::gguf::forward_qwen35::{Qwen35ModelHash, QWEN35_F2_PROBE_MAX};
        let executor = crate::cuda::CudaExecutor::new(0)
            .map_err(|e| format!("CUDA initialization failed: {e}"))?;
        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let vram_mb = executor.memory_info().unwrap_or((0, 0)).1 / (1024 * 1024);
        // The model's OWN state serves only the F2 guard's probe (at most
        // QWEN35_F2_PROBE_MAX positions plus the one decode step); the session
        // allocates the decode state the conversation needs.
        let model = crate::gguf::cuda::Qwen35CudaModel::with_max_seq_len(
            qwen,
            executor,
            QWEN35_F2_PROBE_MAX + 2,
        )
        .map_err(|e| format!("the CUDA model would not build: {e}"))?;
        // The same line, byte for byte, the one-shot path prints — once here,
        // not once per turn.
        say(
            notices,
            format!(
                "Backend: GPU (CUDA, {device_name}, {vram_mb} MB VRAM) [qwen35 hybrid forward, #3090]"
            ),
        );
        Ok(Self {
            model,
            state: None,
            device_name,
            hash: Qwen35ModelHash::of(mapped.data()),
            validated: false,
        })
    }
}

/// Where a session runs its forward.
enum Backend {
    #[cfg(feature = "cuda")]
    Gpu(Box<GpuBackend>),
    /// The CPU forward; the state is `None` until the first turn sizes it.
    Cpu(Option<Qwen35State>),
}

/// A Qwen3.5 hybrid, loaded once and kept for every call after (#3595/#3571).
///
/// The host model lives as long as the process. A session is what `apr chat`
/// and `apr serve` hold for their whole run — one per process — and the device
/// model borrows the host model it was built from, so the base and the hybrid
/// are leaked on purpose, to give that borrow the lifetime it already has.
/// Dropping a session releases the device model and its state, not the host
/// copy.
pub struct Qwen35Session {
    qwen: &'static Qwen35Model<'static>,
    backend: Backend,
    /// The tokens whose forward the decode state holds, in order.
    processed: Vec<u32>,
    /// Positions the decode state was allocated for (0: not yet allocated).
    capacity: usize,
    /// Positions the current turn can reach — what a mid-turn move to the CPU
    /// must allocate, not just what the turn has reached so far.
    turn_positions: usize,
    /// `{arch}.context_length` from the GGUF.
    context_length: usize,
    /// Every line the session has told the user about its route, in order.
    notices: Vec<String>,
}

impl Qwen35Session {
    /// Load the hybrid from `mapped` and, unless `no_gpu`, put it on the GPU.
    ///
    /// The route notice is [`qwen35_route_notice`]'s — the one place that
    /// knows what each route owes the user. A GPU that cannot take the model is
    /// the loud fallback every hybrid path prints
    /// ([`QWEN35_GPU_FALLBACK_PREFIX`]), and the session serves from the CPU.
    ///
    /// # Errors
    /// The base or a hybrid layer would not load. A GPU failure is never an
    /// error — it is the fallback.
    pub fn load(mapped: &MappedGGUFModel, no_gpu: bool) -> Result<Self> {
        let base = Qwen35Model::create_base_model(&mapped.model, mapped.data())?;
        let context_length = base.config.context_length.max(1);
        let base: &'static OwnedQuantizedModel = Box::leak(Box::new(base));
        let qwen: &'static Qwen35Model<'static> = Box::leak(Box::new(
            Qwen35Model::from_model_and_layers(base, &mapped.model, mapped.data())?,
        ));

        let mut notices = Vec::new();
        let route = qwen35_route(no_gpu, cfg!(feature = "cuda"));
        if let Some(notice) = qwen35_route_notice(route) {
            say(&mut notices, notice.to_string());
        }
        let backend = match route {
            #[cfg(feature = "cuda")]
            Qwen35Route::Gpu => match GpuBackend::build(qwen, mapped, &mut notices) {
                Ok(gpu) => Backend::Gpu(Box::new(gpu)),
                Err(reason) => {
                    say(&mut notices, fallback_line(&reason));
                    Backend::Cpu(None)
                },
            },
            _ => Backend::Cpu(None),
        };
        Ok(Self {
            qwen,
            backend,
            processed: Vec::new(),
            capacity: 0,
            turn_positions: 0,
            context_length,
            notices,
        })
    }

    /// Whether the session is currently serving from the GPU.
    #[must_use]
    pub fn on_gpu(&self) -> bool {
        match self.backend {
            #[cfg(feature = "cuda")]
            Backend::Gpu(_) => true,
            Backend::Cpu(_) => false,
        }
    }

    /// The model's declared context length, in tokens.
    #[must_use]
    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    /// Every line the session has printed about its route — the route notice,
    /// the `Backend:` line, each fallback — in order. The printed banner and
    /// the route actually taken can then be checked against each other, which
    /// is what #3595's two contradictory lines needed and nothing asserted.
    #[must_use]
    pub fn notices(&self) -> &[String] {
        &self.notices
    }

    /// Positions the decode state currently holds.
    #[must_use]
    pub fn processed_len(&self) -> usize {
        self.processed.len()
    }

    /// Generate from `prompt` with `config`'s token choice and stop tokens,
    /// calling `on_token` with each new token as it is chosen; `on_token`
    /// returning `false` ends the turn after that token.
    ///
    /// The token choice is `run_qwen35_generate`'s exactly: argmax at
    /// temperature 0 or `top_k` 1, else seeded top-k/top-p. A turn that fails on
    /// the GPU moves the whole session to the CPU, loudly, and finishes there —
    /// the tokens already chosen are kept, never re-emitted.
    ///
    /// Cancellation through `config.cancel` ends the turn early with the tokens
    /// chosen so far, as every other decode loop does.
    ///
    /// # Errors
    /// An empty prompt; a prompt the model's declared context cannot hold
    /// (refused whole — never truncated); a CPU forward failure.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Qwen35Turn> {
        use rand::SeedableRng;
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "qwen35 session: the prompt is empty".to_string(),
            });
        }
        if prompt.len() >= self.context_length {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35 session: the prompt is {} tokens and this model declares a context of \
                     {} (context_length in the GGUF) — it cannot fit with room to answer, so it \
                     was refused whole rather than truncated",
                    prompt.len(),
                    self.context_length
                ),
            });
        }
        // Every position the turn can reach, allocated up front so the state
        // never grows (and never re-prefills) mid-generation.
        let (budget, context_limited) =
            turn_budget(prompt.len(), config.max_tokens, self.context_length);
        self.turn_positions = prompt.len() + budget;
        // A session may be driven from any thread (it is `Send`; `apr serve` runs
        // every request on a blocking-pool worker), and a CUDA context is current
        // per thread: bind it here, before the first allocation or launch.
        #[cfg(feature = "cuda")]
        if let Backend::Gpu(gpu) = &self.backend {
            if let Err(e) = gpu.model.make_current() {
                self.fall_back_to_cpu(&format!(
                    "the CUDA context would not bind to this thread: {e}"
                ))?;
            }
        }
        self.ensure_capacity_or_fall_back()?;

        let (mut logits, reused) = self.advance_to(prompt)?;
        let mut tokens = prompt.to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let mut context_capped = false;
        for generated in 1..=budget {
            if config.cancel.is_cancelled() {
                break;
            }
            let next = choose_token(&logits, config, &mut rng);
            tokens.push(next);
            let keep_going = on_token(next);
            if !keep_going || config.stop_tokens.contains(&next) {
                break;
            }
            if generated == budget {
                context_capped = context_limited;
                break;
            }
            logits = self.advance_to(&tokens)?.0;
        }
        Ok(Qwen35Turn {
            tokens,
            reused,
            used_gpu: self.on_gpu(),
            context_capped,
        })
    }

    /// `tokens` strictly extends what the state holds.
    fn extends(&self, tokens: &[u32]) -> bool {
        !self.processed.is_empty()
            && tokens.len() > self.processed.len()
            && tokens.starts_with(&self.processed)
    }

    /// Make the decode state hold exactly `tokens` and return the logits after
    /// the last one, with how many leading tokens were already held. A GPU
    /// failure moves the session to the CPU and replays `tokens` there.
    fn advance_to(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, usize)> {
        loop {
            match self.try_advance_to(tokens) {
                Ok(done) => return Ok(done),
                Err(Step::Gpu(reason)) => self.fall_back_to_cpu(&reason)?,
                Err(Step::Fatal(e)) => return Err(e),
            }
        }
    }

    fn try_advance_to(&mut self, tokens: &[u32]) -> std::result::Result<(Vec<f32>, usize), Step> {
        #[cfg(feature = "cuda")]
        self.validate_gpu_once(tokens)?;
        let start = if self.extends(tokens) {
            self.processed.len()
        } else {
            self.reset_state()?;
            0
        };
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate().skip(start) {
            logits = self.forward(token, pos)?;
            self.processed.push(token);
        }
        Ok((logits, start))
    }

    /// The F2 guard, once per session, on the first prompt the GPU sees — the
    /// probe the one-shot path takes, so the verdict means the same thing.
    #[cfg(feature = "cuda")]
    fn validate_gpu_once(&mut self, probe: &[u32]) -> std::result::Result<(), Step> {
        let qwen = self.qwen;
        let Backend::Gpu(gpu) = &mut self.backend else {
            return Ok(());
        };
        if gpu.validated {
            return Ok(());
        }
        let outcome = crate::gguf::forward_qwen35::f2_validate_qwen35_receipted_hashed(
            &mut gpu.model,
            qwen,
            probe,
            &gpu.hash,
            &gpu.device_name,
        );
        if !outcome.accepted {
            return Err(Step::Gpu(
                "the F2 CPU-parity guard rejected the GPU path".to_string(),
            ));
        }
        gpu.validated = true;
        Ok(())
    }

    /// One token's forward at `pos` on the current backend.
    fn forward(&mut self, token: u32, pos: usize) -> std::result::Result<Vec<f32>, Step> {
        let qwen = self.qwen;
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Gpu(gpu) => {
                let state = gpu
                    .state
                    .as_mut()
                    .ok_or_else(|| Step::Gpu("the device state was never allocated".to_string()))?;
                gpu.model.forward_single(token, state, pos).map_err(|e| {
                    Step::Gpu(format!("the GPU forward failed at position {pos}: {e}"))
                })
            },
            Backend::Cpu(state) => {
                let state = state.as_mut().ok_or_else(|| RealizarError::InvalidShape {
                    reason: "qwen35 session: the CPU state was never allocated".to_string(),
                })?;
                Ok(qwen.forward_single_qwen35(token, state, pos)?)
            },
        }
    }

    /// Return the decode state to position 0 without reallocating it.
    fn reset_state(&mut self) -> std::result::Result<(), Step> {
        self.processed.clear();
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Gpu(gpu) => {
                if let Some(state) = gpu.state.as_mut() {
                    gpu.model
                        .reset_state(state)
                        .map_err(|e| Step::Gpu(format!("the device state would not reset: {e}")))?;
                }
            },
            Backend::Cpu(state) => {
                if let Some(state) = state.as_mut() {
                    state.reset();
                }
            },
        }
        Ok(())
    }

    /// Make room for the current turn, moving to the CPU if the device cannot.
    fn ensure_capacity_or_fall_back(&mut self) -> Result<()> {
        match self.ensure_capacity() {
            Ok(()) => Ok(()),
            Err(Step::Gpu(reason)) => self.fall_back_to_cpu(&reason),
            Err(Step::Fatal(e)) => Err(e),
        }
    }

    /// Grow the decode state to hold the current turn: at least doubling, at
    /// least [`MIN_CAPACITY`], never past the declared context. A new state
    /// starts empty, so what the old one held is prefilled again.
    fn ensure_capacity(&mut self) -> std::result::Result<(), Step> {
        let positions = self.turn_positions;
        let have_state = match &self.backend {
            #[cfg(feature = "cuda")]
            Backend::Gpu(gpu) => gpu.state.is_some(),
            Backend::Cpu(state) => state.is_some(),
        };
        if have_state && positions <= self.capacity {
            return Ok(());
        }
        let capacity = positions
            .max(self.capacity.saturating_mul(2))
            .max(MIN_CAPACITY)
            .min(self.context_length)
            .max(positions);
        self.processed.clear();
        let qwen = self.qwen;
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Gpu(gpu) => {
                // Free the old state before asking for the larger one.
                gpu.state = None;
                gpu.state = Some(gpu.model.new_state_with_capacity(capacity).map_err(|e| {
                    Step::Gpu(format!(
                        "a decode state for {capacity} positions would not allocate: {e}"
                    ))
                })?);
            },
            Backend::Cpu(state) => *state = Some(qwen.new_state(capacity)),
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Move the session to the CPU forward, printing why, and size its state
    /// for the current turn.
    fn fall_back_to_cpu(&mut self, reason: &str) -> Result<()> {
        say(&mut self.notices, fallback_line(reason));
        self.backend = Backend::Cpu(None);
        self.capacity = 0;
        self.processed.clear();
        match self.ensure_capacity() {
            Ok(()) => Ok(()),
            Err(Step::Fatal(e)) => Err(e),
            Err(Step::Gpu(reason)) => Err(RealizarError::UnsupportedOperation {
                operation: "qwen35_session".to_string(),
                reason: format!("the CPU backend reported a GPU failure: {reason}"),
            }),
        }
    }
}

/// Print `line` to stderr and keep it in the session's notices.
fn say(notices: &mut Vec<String>, line: String) {
    eprintln!("{line}");
    notices.push(line);
}

/// The loud, never-silent GPU fallback, in the one shape every hybrid path uses.
fn fallback_line(reason: &str) -> String {
    format!("{QWEN35_GPU_FALLBACK_PREFIX}, falling back to CPU: {reason}")
}

/// How many tokens a turn may generate: `max_tokens`, or fewer when the
/// declared context ends first — and whether it did. The caller has already
/// refused a prompt of `context_length` tokens or more.
fn turn_budget(prompt_len: usize, max_tokens: usize, context_length: usize) -> (usize, bool) {
    let room = context_length.saturating_sub(prompt_len);
    (max_tokens.min(room), room < max_tokens)
}

/// `run_qwen35_generate`'s token choice, verbatim.
fn choose_token(
    logits: &[f32],
    config: &QuantizedGenerateConfig,
    rng: &mut rand::rngs::StdRng,
) -> u32 {
    if config.temperature == 0.0 || config.top_k == 1 {
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

#[cfg(test)]
#[path = "qwen35_session_tests.rs"]
mod tests;
