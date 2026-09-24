//! #4234: continuous batching for Qwen3.5 decode — one device model shared by
//! several sessions, their decode steps run together.
//!
//! `apr serve` gives each in-flight request its own [`Qwen35Session`] (its own
//! decode state), and every session of one server holds the SAME
//! [`SharedGpu`]: the resident weights are uploaded once. A session's prefill
//! locks the model and runs alone. A session's decode step (one new token) goes
//! to [`SharedGpu::decode`], which is where the batching happens:
//!
//! - the first caller to find no step running becomes the step's **leader**; it
//!   waits for as many callers as the previous step carried (at most
//!   [`DEFAULT_BATCH_WINDOW`]), takes every pending token, and runs them as ONE
//!   [`Qwen35CudaModel::forward_batch`];
//! - every other caller waits for its own result, which the leader posts.
//!
//! Requests therefore join the batch at the first step after their prefill and
//! leave it when their turn ends — nothing is padded, and a lone stream never
//! waits (the previous step carried one, and one is pending).
//!
//! `forward_batch` is bitwise `forward_single` per sequence (its test), so a
//! stream's tokens do not depend on which other streams shared its steps.
//!
//! [`Qwen35Session`]: super::Qwen35Session
//! [`Qwen35CudaModel::forward_batch`]: crate::gguf::cuda::Qwen35CudaModel::forward_batch

use crate::gguf::cuda::{Qwen35CudaModel, Qwen35CudaState};
use crate::gguf::forward_qwen35::Qwen35ModelHash;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// How long a step's leader waits for the rest of the previous step's streams.
/// A stream's gap between two steps is its sampling and callback time, well
/// under a millisecond; this bounds the cost of a stream that has left.
pub const DEFAULT_BATCH_WINDOW: Duration = Duration::from_millis(2);

/// The device model and what is known about it once.
pub(super) struct GpuModel {
    pub(super) model: Qwen35CudaModel<'static>,
    /// The F2 guard has accepted this device model (once per model, on the
    /// first prompt any session sends it).
    pub(super) validated: bool,
}

/// One device model, shared by every session of a server (see the module docs).
pub(super) struct SharedGpu {
    model: Mutex<GpuModel>,
    queue: Mutex<Queue>,
    arrived: Condvar,
    window: Duration,
    pub(super) device_name: String,
    /// The receipt key's model half, hashed once at load.
    pub(super) hash: Qwen35ModelHash,
}

/// What a decode step gives back: the logits or why the GPU failed, and the
/// state — `None` only if the step's leader panicked with it.
pub(super) type Decoded = (
    std::result::Result<Vec<f32>, String>,
    Option<Qwen35CudaState>,
);

struct Ticket {
    id: u64,
    token: u32,
    position: usize,
    state: Qwen35CudaState,
}

#[derive(Default)]
struct Queue {
    pending: Vec<Ticket>,
    done: Vec<(u64, Decoded)>,
    /// A leader is gathering or running a step.
    leader: bool,
    next_id: u64,
    /// Sequences the previous step carried — what the next leader waits for.
    last_batch: usize,
    stats: BatchStats,
}

/// Counters over the model's lifetime: the evidence that batching engaged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// Decode steps run through [`SharedGpu::decode`].
    pub steps: u64,
    /// Sequences those steps carried; `sequences / steps` is the mean batch.
    pub sequences: u64,
    /// The largest batch one step carried.
    pub widest: usize,
}

fn relock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl SharedGpu {
    pub(super) fn new(
        model: Qwen35CudaModel<'static>,
        device_name: String,
        hash: Qwen35ModelHash,
    ) -> Self {
        Self {
            model: Mutex::new(GpuModel {
                model,
                validated: false,
            }),
            queue: Mutex::new(Queue::default()),
            arrived: Condvar::new(),
            window: DEFAULT_BATCH_WINDOW,
            device_name,
            hash,
        }
    }

    /// The device model, exclusively — for a prefill, a reset, an allocation or
    /// the F2 guard. A decode step in flight finishes first.
    pub(super) fn lock(&self) -> MutexGuard<'_, GpuModel> {
        relock(&self.model)
    }

    /// See [`BatchStats`].
    pub(super) fn stats(&self) -> BatchStats {
        relock(&self.queue).stats
    }

    /// Advance `state` by `token` at `position` in the next batched step, and
    /// hand the state back with the logits.
    pub(super) fn decode(&self, token: u32, position: usize, state: Qwen35CudaState) -> Decoded {
        let mut q = relock(&self.queue);
        let id = q.next_id;
        q.next_id += 1;
        q.pending.push(Ticket {
            id,
            token,
            position,
            state,
        });
        self.arrived.notify_all();
        loop {
            if let Some(i) = q.done.iter().position(|(d, _)| *d == id) {
                return q.done.swap_remove(i).1;
            }
            if q.leader {
                q = self.arrived.wait(q).unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            q.leader = true;
            let expect = q.last_batch.max(1);
            let deadline = Instant::now() + self.window;
            while q.pending.len() < expect {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                q = self
                    .arrived
                    .wait_timeout(q, deadline - now)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            let tickets = std::mem::take(&mut q.pending);
            drop(q);
            let mut step = Step {
                shared: self,
                ids: tickets.iter().map(|t| t.id).collect(),
                posted: false,
            };
            let results = self.run(tickets);
            step.post(results);
            q = relock(&self.queue);
        }
    }

    /// One `forward_batch` over every ticket, under the model lock.
    fn run(&self, tickets: Vec<Ticket>) -> Vec<(u64, Decoded)> {
        let mut m = self.lock();
        let tokens: Vec<u32> = tickets.iter().map(|t| t.token).collect();
        let positions: Vec<usize> = tickets.iter().map(|t| t.position).collect();
        let (ids, mut states): (Vec<u64>, Vec<Qwen35CudaState>) =
            tickets.into_iter().map(|t| (t.id, t.state)).unzip();
        // The leader is whichever session thread got here first; bind the one
        // context to it, as `reserve` does for a session's own thread.
        let outcome = m
            .model
            .make_current()
            .and_then(|()| match states.as_mut_slice() {
                // A lone stream takes `forward_single` — the same logits bit for bit
                // (the batch's test), without the batch's per-step buffers.
                [state] => m
                    .model
                    .forward_single(tokens[0], state, positions[0])
                    .map(|logits| vec![logits]),
                _ => {
                    let mut refs: Vec<&mut Qwen35CudaState> = states.iter_mut().collect();
                    m.model.forward_batch(&tokens, &mut refs, &positions)
                },
            });
        drop(m);
        match outcome {
            Ok(rows) => ids
                .into_iter()
                .zip(rows)
                .zip(states)
                .map(|((id, logits), state)| (id, (Ok(logits), Some(state))))
                .collect(),
            Err(e) => {
                let why = format!("the batched GPU decode step failed: {e}");
                ids.into_iter()
                    .zip(states)
                    .map(|(id, state)| (id, (Err(why.clone()), Some(state))))
                    .collect()
            },
        }
    }
}

/// A step's leadership. Posting hands every result to its waiter; dropping it
/// unposted (the leader panicked) still frees the leadership and answers every
/// waiter with an error, so no follower waits forever.
struct Step<'a> {
    shared: &'a SharedGpu,
    ids: Vec<u64>,
    posted: bool,
}

impl Step<'_> {
    fn post(&mut self, results: Vec<(u64, Decoded)>) {
        let mut q = relock(&self.shared.queue);
        q.stats.steps += 1;
        q.stats.sequences += results.len() as u64;
        q.stats.widest = q.stats.widest.max(results.len());
        q.last_batch = results.len();
        q.done.extend(results);
        q.leader = false;
        self.posted = true;
        drop(q);
        self.shared.arrived.notify_all();
    }
}

impl Drop for Step<'_> {
    fn drop(&mut self) {
        if self.posted {
            return;
        }
        let mut q = relock(&self.shared.queue);
        for &id in &self.ids {
            q.done.push((
                id,
                (Err("the batched decode step panicked".to_string()), None),
            ));
        }
        q.leader = false;
        drop(q);
        self.shared.arrived.notify_all();
    }
}
