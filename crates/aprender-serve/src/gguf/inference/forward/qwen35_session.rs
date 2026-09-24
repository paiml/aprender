//! #3595 (`apr chat`) / #3571 (`apr serve`): a Qwen3.5 hybrid held resident for
//! a whole session.
//!
//! `apr run`'s old `run_qwen35_generate_dispatch` (deleted in #4263; `apr run`
//! now loads a one-call session, [`Qwen35Session::load_for_run`]) served ONE call: it builds the hybrid, uploads it, runs the F2 guard (which
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
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel};

/// The smallest decode state a session allocates. Growing by doubling from
/// here keeps a short conversation from paying for the model's whole declared
/// context (262 144 positions for every Qwen3.5 size).
const MIN_CAPACITY: usize = 4096;

/// Set to `per-token` to make a session prefill one token at a time, the path
/// 0.69.1 served, instead of the batched prefill (0.69.3). It exists so the two
/// can be compared token for token on ONE binary; the session says loudly that
/// it is set.
pub const QWEN35_SESSION_PREFILL_ENV: &str = "APR_QWEN35_SESSION_PREFILL";

/// What one Qwen3.5 turn did: the engine's [`Turn`](crate::session::Turn).
pub type Qwen35Turn = crate::session::Turn;

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
    /// Build the hybrid on the device, once. With `plan_positions`, the device
    /// memory is planned for that many positions BEFORE a byte is uploaded, and
    /// a context that cannot fit is [`GpuBuild::Refused`] (#3596).
    fn build(
        qwen: &'static Qwen35Model<'static>,
        mapped: &MappedGGUFModel,
        notices: &mut Vec<String>,
        plan_positions: Option<usize>,
    ) -> std::result::Result<Self, GpuBuild> {
        use crate::gguf::forward_qwen35::{Qwen35ModelHash, QWEN35_F2_PROBE_MAX};
        let executor = crate::cuda::CudaExecutor::new(0)
            .map_err(|e| format!("CUDA initialization failed: {e}"))?;
        let planned = match plan_positions {
            Some(positions) => Some(plan_capacity(qwen, &executor, positions)?),
            None => None,
        };
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
        .map_err(|e| GpuBuild::Fallback(format!("the CUDA model would not build: {e}")))?;
        let mut model = model;
        if let Some((attention, rows)) = planned {
            model.set_prefill_chunk_rows(rows);
            model.set_prefill_attention(attention);
        }
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

/// Why the device model was not built.
#[cfg(feature = "cuda")]
enum GpuBuild {
    /// A reason to serve from the CPU instead — printed, never silent.
    Fallback(String),
    /// The context does not fit the device (#3596): refused before loading,
    /// with the arithmetic. NOT a fallback — at the lengths that trip this the
    /// CPU forward takes hours, which is a stall, not a fallback.
    Refused(Box<crate::capacity::CapacityRefusal>),
}

#[cfg(feature = "cuda")]
impl From<String> for GpuBuild {
    fn from(reason: String) -> Self {
        Self::Fallback(reason)
    }
}

/// Will `positions` fit? Decided from the host model and the MEASURED free
/// memory, before a byte is uploaded — never discovered as an OOM mid-prefill
/// (#3596). cuBLAS f32 attention while its plan fits, flash only when flash
/// alone fits; bigger chunks on a unified-memory host. A path passed over is
/// printed.
#[cfg(feature = "cuda")]
fn plan_capacity(
    qwen: &Qwen35Model<'_>,
    executor: &crate::cuda::CudaExecutor,
    positions: usize,
) -> std::result::Result<(crate::gguf::cuda::PrefillAttention, usize), GpuBuild> {
    let device_memory = crate::capacity::measure_device_memory(executor)?;
    let (gpu_free, gpu_total) = device_memory.plan_free_total();
    let attention_paths =
        crate::gguf::cuda::Qwen35CudaModel::prefill_attention_candidates_for(qwen, executor);
    let chunk_rows_to_try: &[usize] = match device_memory {
        crate::capacity::DeviceMemory::Unified { .. } => &[
            crate::gguf::cuda::UNIFIED_PREFILL_CHUNK_ROWS,
            crate::gguf::cuda::PREFILL_MAX_CHUNK_ROWS,
        ],
        crate::capacity::DeviceMemory::Discrete { .. } => {
            &[crate::gguf::cuda::PREFILL_MAX_CHUNK_ROWS]
        },
    };
    let mut passed_over = Vec::new();
    let planned =
        crate::capacity::plan_first_fit(&attention_paths, chunk_rows_to_try, |attention, rows| {
            let verdict = crate::capacity::plan(&crate::capacity::CapacityInputs {
                memory: Some(device_memory),
                ..crate::gguf::cuda::Qwen35CudaModel::capacity_inputs(
                    qwen, positions, gpu_free, gpu_total, attention, rows,
                )
            });
            if let crate::capacity::CapacityVerdict::Refused(r) = &verdict {
                passed_over.push(crate::capacity::passed_over_line(
                    attention.as_str(),
                    rows,
                    r,
                ));
            }
            verdict
        });
    match planned {
        Ok(fit) => {
            if !passed_over.is_empty() {
                eprintln!(
                    "[qwen35] prefill plan: {} did not fit; using {} at {} rows ({:.0} MiB)",
                    passed_over.join("; "),
                    fit.0.as_str(),
                    fit.1,
                    fit.2.total_mb
                );
            }
            if fit.2.kv_dtype != crate::capacity::KvDtype::F32 {
                // `capacity_inputs` reports no f16 decode, so a plan cannot
                // choose it; if that ever changes without the f16 cache
                // existing, refuse loudly.
                return Err(GpuBuild::Fallback(format!(
                    "the capacity plan chose a {:?} KV cache, which this build cannot allocate",
                    fit.2.kv_dtype
                )));
            }
            Ok((fit.0, fit.1))
        },
        Err(Some(refusal)) => Err(GpuBuild::Refused(refusal)),
        Err(None) => Err(GpuBuild::Fallback(
            "no prefill attention path to plan".to_string(),
        )),
    }
}

/// Where a session runs its forward.
enum Backend {
    #[cfg(feature = "cuda")]
    Gpu(Box<GpuBackend>),
    /// The CPU forward; the state is `None` until the first turn sizes it.
    Cpu(Option<Qwen35State>),
}

/// A Qwen3.5 hybrid held resident for the whole run (#3595/#3571): the
/// engine's [`Session`](crate::session::Session) over [`Qwen35Forward`].
pub type Qwen35Session = crate::session::Session<Qwen35Forward>;

impl crate::session::Session<Qwen35Forward> {
    /// Load the hybrid from `mapped` and, unless `no_gpu`, put it on the GPU.
    /// See [`Qwen35Forward::load`].
    ///
    /// # Errors
    /// The base or a hybrid layer would not load.
    pub fn load(mapped: &MappedGGUFModel, no_gpu: bool) -> Result<Self> {
        Ok(Self::new(Qwen35Forward::load(mapped, no_gpu)?))
    }

    /// `apr run`'s load: one call of at most `positions` positions, on the
    /// host model `qwen` (see [`Qwen35Forward::leak_host`]). The device memory
    /// is planned for `positions` before the upload and a context that cannot
    /// fit is refused (#3596); the decode state is sized to exactly the call.
    ///
    /// # Errors
    /// [`RealizarError::CapacityRefused`] when the GPU was asked for and the
    /// context does not fit it.
    pub fn load_for_run(
        qwen: &'static Qwen35Model<'static>,
        mapped: &MappedGGUFModel,
        no_gpu: bool,
        positions: usize,
    ) -> Result<Self> {
        Ok(Self::new(Qwen35Forward::from_host(
            qwen,
            mapped,
            no_gpu,
            Some(positions),
        )?))
    }

    /// The hybrid's layers — Gated `DeltaNet` and full attention together, all
    /// resident on the one backend the session serves from.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.engine().qwen.layers.len()
    }
}

/// The Qwen3.5 hybrid's forward: the only Qwen3.5 code a verb reaches, and
/// only through a [`Qwen35Session`].
///
/// The host model lives as long as the process. A session is what `apr chat`
/// and `apr serve` hold for their whole run — one per process — and the device
/// model borrows the host model it was built from, so the base and the hybrid
/// are leaked on purpose, to give that borrow the lifetime it already has.
/// Dropping a session releases the device model and its state, not the host
/// copy.
pub struct Qwen35Forward {
    qwen: &'static Qwen35Model<'static>,
    backend: Backend,
    /// Positions the decode state was allocated for (0: not yet allocated).
    capacity: usize,
    /// Positions the current turn can reach — what a mid-turn move to the CPU
    /// must allocate, not just what the turn has reached so far.
    turn_positions: usize,
    /// Bumped each time the decode state is (re)allocated, so
    /// [`ArchForward::reserve`] can say whether what it held was dropped.
    allocations: u64,
    /// The smallest state allocated: [`MIN_CAPACITY`] for a resident session,
    /// 0 for `apr run`'s one call.
    min_capacity: usize,
    /// `{arch}.context_length` from the GGUF.
    context_length: usize,
    /// Every line the session has told the user about its route, in order.
    notices: Vec<String>,
    /// [`QWEN35_SESSION_PREFILL_ENV`] asked for the one-token prefill.
    per_token_prefill: bool,
    /// Prompts the GPU prefilled in one batched call — the evidence that `apr
    /// serve` took the batched path, not a speed that merely looks like it.
    batched_prefills: usize,
}

impl Qwen35Forward {
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
        Self::from_host(Self::leak_host(mapped)?, mapped, no_gpu, None)
    }

    /// Build the host model (base and hybrid layers) and leak it, to give the
    /// device model's borrow the process lifetime it has. A caller that loads
    /// the same file more than once (`apr run` under `qa`/`eval`) keeps the one
    /// it got, rather than leaking a copy per call.
    ///
    /// # Errors
    /// The base or a hybrid layer would not load.
    pub fn leak_host(mapped: &MappedGGUFModel) -> Result<&'static Qwen35Model<'static>> {
        let base = Qwen35Model::create_base_model(&mapped.model, mapped.data())?;
        let base: &'static OwnedQuantizedModel = Box::leak(Box::new(base));
        Ok(Box::leak(Box::new(Qwen35Model::from_model_and_layers(
            base,
            &mapped.model,
            mapped.data(),
        )?)))
    }

    /// [`Self::leak_host`] once per file per process: `apr qa`/`eval` call
    /// `apr run`'s path many times on one model, and each call would otherwise
    /// leak a fresh host copy. Keyed by the canonical path, length and mtime,
    /// so a file rewritten in place is loaded again.
    ///
    /// # Errors
    /// As [`Self::leak_host`].
    pub fn cached_host(
        path: &std::path::Path,
        mapped: &MappedGGUFModel,
    ) -> Result<&'static Qwen35Model<'static>> {
        type Key = (std::path::PathBuf, u64, Option<std::time::SystemTime>);
        static HOSTS: std::sync::Mutex<Vec<(Key, &'static Qwen35Model<'static>)>> =
            std::sync::Mutex::new(Vec::new());
        let meta = std::fs::metadata(path).ok();
        let key: Key = (
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
            meta.as_ref().map_or(0, std::fs::Metadata::len),
            meta.and_then(|m| m.modified().ok()),
        );
        let mut hosts = HOSTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, host)) = hosts.iter().find(|(k, _)| *k == key) {
            return Ok(host);
        }
        let host = Self::leak_host(mapped)?;
        hosts.push((key, host));
        Ok(host)
    }

    /// Put the host model `qwen` on its backend. `plan_positions`: see
    /// [`Qwen35Session::load_for_run`]; `None` is a resident session whose state
    /// grows by doubling from [`MIN_CAPACITY`].
    fn from_host(
        qwen: &'static Qwen35Model<'static>,
        mapped: &MappedGGUFModel,
        no_gpu: bool,
        plan_positions: Option<usize>,
    ) -> Result<Self> {
        let context_length = qwen.base.config.context_length.max(1);
        let mut notices = Vec::new();
        let per_token_prefill =
            std::env::var(QWEN35_SESSION_PREFILL_ENV).as_deref() == Ok("per-token");
        if per_token_prefill {
            say(
                &mut notices,
                format!("[qwen35] {QWEN35_SESSION_PREFILL_ENV}=per-token: prompts prefill one token at a time, not batched"),
            );
        }
        let route = qwen35_route(no_gpu, cfg!(feature = "cuda"));
        if let Some(notice) = qwen35_route_notice(route) {
            say(&mut notices, notice.to_string());
        }
        let backend = match route {
            #[cfg(feature = "cuda")]
            Qwen35Route::Gpu => match GpuBackend::build(qwen, mapped, &mut notices, plan_positions)
            {
                Ok(gpu) => Backend::Gpu(Box::new(gpu)),
                Err(GpuBuild::Refused(refusal)) => {
                    return Err(RealizarError::CapacityRefused(refusal));
                },
                Err(GpuBuild::Fallback(reason)) => {
                    say(&mut notices, fallback_line(&reason));
                    Backend::Cpu(None)
                },
            },
            _ => Backend::Cpu(None),
        };
        // A one-call state is sized to the call: the plan above was made for
        // exactly `plan_positions`, not for MIN_CAPACITY.
        let min_capacity = if plan_positions.is_some() {
            0
        } else {
            MIN_CAPACITY
        };
        Ok(Self {
            qwen,
            backend,
            capacity: 0,
            turn_positions: 0,
            allocations: 0,
            min_capacity,
            context_length,
            notices,
            per_token_prefill,
            batched_prefills: 0,
        })
    }

    /// The host model the forward serves from (the base's config, tokenizer
    /// metadata and weights).
    #[must_use]
    pub fn base(&self) -> &'static OwnedQuantizedModel {
        self.qwen.base
    }

    /// Advance the state from holding `tokens[..start]` to holding `tokens`.
    fn try_forward(&mut self, tokens: &[u32], start: usize) -> std::result::Result<Vec<f32>, Step> {
        #[cfg(feature = "cuda")]
        self.validate_gpu_once(tokens)?;
        if start == 0 {
            self.reset_state()?;
        }
        // A prompt, or a turn's new suffix, goes through the batched prefill (#3596)
        // — the path `apr run` takes. Decode steps (one new token) stay one-token.
        #[cfg(feature = "cuda")]
        if tokens.len().saturating_sub(start) > 1 {
            if let Some(logits) = self.try_batched_prefill(&tokens[start..], start)? {
                return Ok(logits);
            }
        }
        if tokens.len().saturating_sub(start) > 1 {
            if let Some(logits) = self.try_cpu_prefill(&tokens[start..], start)? {
                return Ok(logits);
            }
        }
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate().skip(start) {
            logits = self.forward_one(token, pos)?;
        }
        Ok(logits)
    }

    /// #4228: prefill `new` at positions `pos0..` layer by layer on the CPU and
    /// return the last position's logits — bitwise the per-token path's. `None`
    /// when the session is not on the CPU, was told to prefill per token, or the
    /// state cannot take all of `new` at once (the caller then goes per token).
    fn try_cpu_prefill(
        &mut self,
        new: &[u32],
        pos0: usize,
    ) -> std::result::Result<Option<Vec<f32>>, Step> {
        let qwen = self.qwen;
        let Backend::Cpu(Some(state)) = &mut self.backend else {
            return Ok(None);
        };
        if self.per_token_prefill || !qwen.prefill_fits(state, pos0, new.len()) {
            return Ok(None);
        }
        let t0 = std::time::Instant::now();
        let logits = qwen.forward_prefill_qwen35(new, state, pos0)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[qwen35] cpu batched prefill: {} tokens in {ms:.0} ms ({:.1} tok/s, from position {pos0})",
            new.len(),
            new.len() as f64 * 1000.0 / ms.max(1e-9),
        );
        self.batched_prefills += 1;
        Ok(Some(logits))
    }

    /// Prefill `new` at positions `pos0..` in one batched call on the GPU and return
    /// the last position's logits; `None` when the session is not on the GPU, was
    /// told to prefill per token, or the prefill's workspace does not fit (said
    /// loudly — the caller then prefills one token at a time, on the GPU still).
    ///
    /// A failed prefill is a GPU failure: the session moves to the CPU, which is
    /// also what `prefill`'s "on Err the state must be discarded" requires.
    #[cfg(feature = "cuda")]
    fn try_batched_prefill(
        &mut self,
        new: &[u32],
        pos0: usize,
    ) -> std::result::Result<Option<Vec<f32>>, Step> {
        let qwen = self.qwen;
        let Backend::Gpu(gpu) = &mut self.backend else {
            return Ok(None);
        };
        if self.per_token_prefill {
            return Ok(None);
        }
        let end = pos0 + new.len();
        if let Err(why) = fit_prefill_plan(qwen, &mut gpu.model, end) {
            eprintln!(
                "[qwen35] batched prefill: {} tokens at position {pos0} would not fit ({why}); \
                 prefilling one token at a time on the GPU",
                new.len()
            );
            return Ok(None);
        }
        let state = gpu
            .state
            .as_mut()
            .ok_or_else(|| Step::Gpu("the device state was never allocated".to_string()))?;
        // `prefill` ends in the logits download, so this clock stops on finished work.
        let t0 = std::time::Instant::now();
        let logits = batched_prefill_outcome(gpu.model.prefill(new, state, pos0), new.len(), pos0)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        // The line `apr run` prints, plus the start position: a serve log must show
        // WHICH prefill it took, or a per-token regression reads as a slow GPU.
        eprintln!(
            "[qwen35] batched prefill: {} tokens in {ms:.0} ms ({:.0} tok/s, chunk {} rows, attention {}, from position {pos0})",
            new.len(),
            new.len() as f64 * 1000.0 / ms.max(1e-9),
            gpu.model.prefill_chunk_rows(end),
            gpu.model.prefill_attention_mode().as_str(),
        );
        self.batched_prefills += 1;
        Ok(Some(logits))
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
    fn forward_one(&mut self, token: u32, pos: usize) -> std::result::Result<Vec<f32>, Step> {
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

    /// Bind the CUDA context to THIS thread before the first allocation or launch.
    ///
    /// A CUDA context is per-thread, and a session runs under `spawn_blocking` in
    /// `apr serve`, so the thread that binds is not the thread that built the
    /// model. A bind failure falls back to the CPU rather than failing the turn.
    fn bind_cuda_context_or_fall_back(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Backend::Gpu(gpu) = &self.backend {
            if let Err(e) = gpu.model.make_current() {
                self.fall_back_to_cpu(&format!(
                    "the CUDA context would not bind to this thread: {e}"
                ))?;
            }
        }
        Ok(())
    }

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
            .max(self.min_capacity)
            .min(self.context_length)
            .max(positions);
        self.allocations += 1;
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

impl crate::session::ArchForward for Qwen35Forward {
    fn arch(&self) -> &'static str {
        "qwen35"
    }

    fn on_gpu(&self) -> bool {
        match self.backend {
            #[cfg(feature = "cuda")]
            Backend::Gpu(_) => true,
            Backend::Cpu(_) => false,
        }
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn batched_prefills(&self) -> usize {
        self.batched_prefills
    }

    fn notices(&self) -> &[String] {
        &self.notices
    }

    fn reserve(&mut self, positions: usize) -> Result<bool> {
        let before = self.allocations;
        self.turn_positions = positions;
        self.bind_cuda_context_or_fall_back()?;
        self.ensure_capacity_or_fall_back()?;
        Ok(self.allocations != before)
    }

    fn validate(&mut self, probe: &[u32]) -> Result<()> {
        #[cfg(feature = "cuda")]
        match self.validate_gpu_once(probe) {
            Ok(()) => {},
            Err(Step::Gpu(reason)) => self.fall_back_to_cpu(&reason)?,
            Err(Step::Fatal(e)) => return Err(e),
        }
        #[cfg(not(feature = "cuda"))]
        let _ = probe;
        Ok(())
    }

    /// A GPU failure moves the forward to the CPU, loudly, and replays all of
    /// `tokens` there: the tokens a turn already chose are kept, never
    /// re-emitted.
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let mut start = start;
        loop {
            match self.try_forward(tokens, start) {
                Ok(logits) => return Ok(logits),
                Err(Step::Gpu(reason)) => {
                    self.fall_back_to_cpu(&reason)?;
                    start = 0;
                },
                Err(Step::Fatal(e)) => return Err(e),
            }
        }
    }
}

/// Choose the prefill attention and chunk rows for a prefill ending at `end`, the
/// way `apr run` plans them (cuBLAS f32 attention while it fits, flash when only
/// flash fits; bigger chunks on a unified-memory host), against the device memory
/// free NOW — the resident weights and the session's decode state are already
/// allocated, so only the prefill's own workspace and the overhead are asked for.
#[cfg(feature = "cuda")]
fn fit_prefill_plan(
    qwen: &Qwen35Model<'_>,
    model: &mut crate::gguf::cuda::Qwen35CudaModel<'static>,
    end: usize,
) -> std::result::Result<(), String> {
    let memory = crate::capacity::measure_device_memory(model.executor_mut())?;
    let (free, _) = memory.plan_free_total();
    let rows_to_try: &[usize] = match memory {
        crate::capacity::DeviceMemory::Unified { .. } => &[
            crate::gguf::cuda::UNIFIED_PREFILL_CHUNK_ROWS,
            crate::gguf::cuda::PREFILL_MAX_CHUNK_ROWS,
        ],
        crate::capacity::DeviceMemory::Discrete { .. } => {
            &[crate::gguf::cuda::PREFILL_MAX_CHUNK_ROWS]
        },
    };
    let attentions = crate::gguf::cuda::Qwen35CudaModel::prefill_attention_candidates_for(
        qwen,
        model.executor_mut(),
    );
    let (attention, rows) = choose_prefill_plan(
        free,
        &attentions,
        rows_to_try,
        |attention, rows| {
            model.set_prefill_attention(attention);
            model.set_prefill_chunk_rows(rows);
            model.prefill_workspace_bytes(end) as u64 + crate::capacity::OVERHEAD_BYTES
        },
        |attention| attention.as_str(),
    )?;
    model.set_prefill_attention(attention);
    model.set_prefill_chunk_rows(rows);
    Ok(())
}

/// The first `(attention, rows)`, attention-major in the order given, whose
/// `need` fits in `free` bytes. When none fits, the Err names every candidate
/// refused and the free MiB; the session then prefills one token at a time on
/// the GPU and says so. Pure, so the no-fit branch is tested without a device
/// (#4255).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn choose_prefill_plan<A: Copy>(
    free: u64,
    attentions: &[A],
    rows_to_try: &[usize],
    mut need: impl FnMut(A, usize) -> u64,
    name: impl Fn(A) -> &'static str,
) -> std::result::Result<(A, usize), String> {
    let mut refused = Vec::new();
    for &attention in attentions {
        for &rows in rows_to_try {
            let need = need(attention, rows);
            if need <= free {
                return Ok((attention, rows));
            }
            refused.push(format!(
                "{} at {rows} rows needs {} MiB",
                name(attention),
                need >> 20
            ));
        }
    }
    Err(format!("{}; {} MiB free", refused.join(", "), free >> 20))
}

/// What a batched prefill's result means for the session. A failed prefill is a
/// GPU failure ([`Step::Gpu`]): the session moves to the CPU, which also discards
/// the device state as `prefill`'s contract requires. It is never a per-token
/// retry on that state, and never a [`Step::Fatal`] that ends the turn (#4255).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn batched_prefill_outcome<E: std::fmt::Display>(
    result: std::result::Result<Vec<f32>, E>,
    tokens: usize,
    pos0: usize,
) -> std::result::Result<Vec<f32>, Step> {
    result.map_err(|e| {
        Step::Gpu(format!(
            "the GPU batched prefill of {tokens} tokens at position {pos0} failed: {e}"
        ))
    })
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

#[cfg(test)]
#[path = "qwen35_session_tests.rs"]
mod tests;
