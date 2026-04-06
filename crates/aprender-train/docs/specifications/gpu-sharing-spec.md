# GPU Sharing Specification

**Status:** Implemented (all 3 phases complete, 143 GPU tests pass)
**Scope:** Allow 2–3 QLoRA fine-tuning jobs to run concurrently on a single GPU, with optional multi-node heterogeneous support.

### Implementation Summary

| Phase | Status | Tests | PRs | Examples |
|-------|--------|-------|-----|----------|
| 1: VRAM guard + ledger + wait + profiler + MPS | Complete | 45 | entrenar#217 | `gpu_ledger` |
| 2: Multi-adapter pipeline + scheduling + adapters-config | Complete | 55 | — | `multi_adapter_training` |
| 3: Cluster config + placement + coordinator + SSH + health check | Complete | 43 | entrenar#220, #224 | `cluster_training` |

**Key modules:** `entrenar::gpu::{guard, ledger, wait, profiler, mps, cluster, placement, coordinator}`
**CLI flags:** `--wait-gpu`, `--vram`, `--experimental-mps`, `--gpu-share`, `--adapters`, `--adapters-config`, `--gpus`, `--role`, `--bind`, `--coordinator`, `--expect-workers`

## Problem

Today, `apr finetune --method qlora` assumes exclusive GPU access. A 7B QLoRA job uses ~7.3 GB VRAM on a 24 GB GPU — 70% of VRAM sits idle. Launching a second job causes silent crashes because neither job checks available resources before allocating.

## Goals

1. **G-SHARE-001:** 2–3 QLoRA jobs share one GPU automatically — no user flags, no external daemons.
2. **G-SHARE-002:** Fair compute sharing — each job gets proportional SM access.
3. **G-SHARE-003:** VRAM guard prevents OOM crashes — jobs queue when VRAM is insufficient.
4. **G-SHARE-004:** Optional multi-node scheduling across heterogeneous GPUs (4090 + Jetson + SSH Intel).

## Key Decision: MPS vs Multi-Adapter (mLoRA) Approach

### Option A: CUDA MPS (Multi-Process, OS-level sharing)

Each QLoRA job runs as a separate `apr` process. MPS daemon partitions SMs between processes.

**Pros:** Simple to implement (~50 lines), no architectural change, processes are independent.

**Fatal flaws identified by literature review:**

1. **No fault isolation.** A CUDA kernel crash in one MPS client kills ALL clients on that GPU (NVIDIA docs, Guardian paper arXiv:2401.09290). With custom PTX kernels (trueno GEMM, NF4 dequant), this is a real risk. Hours of training across all jobs lost.

2. **Base model duplicated per process.** Each `apr` process loads its own copy of the NF4 base model. 3 concurrent 7B jobs = 3 × 7.3 GB = 21.9 GB just for base weights, leaving ~2.5 GB for everything else. **Practically limits to 2 concurrent 7B jobs max.**

3. **Static SM partitioning hurts training.** `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is set once at context creation, cannot be changed dynamically (NVIDIA docs). Training workloads have variable compute demands (forward vs backward vs optimizer) — static partitioning wastes cycles. (Xing et al. 2025, arXiv:2508.08448; LithOS SOSP '25)

4. **Thread percentage is per-client at init, not rebalancing.** Job 1 starts at 100%. Job 2 starts, sets 50%. Job 1 still runs at 100%. No rebalancing without restart.

5. **1,500x latency bug.** On H100, enabling MPS without explicit thread percentage caused kernel latency to jump from 65μs to 100ms (NVIDIA Forum). Auto-starting MPS without careful configuration is dangerous.

6. **MPS on Jetson is experimental.** Thread percentage partitioning shows unreliable behavior below 30% (NVIDIA Forum). Not production-ready.

7. **No one does this for training.** Zero documented cases of concurrent QLoRA training with MPS across NVIDIA forums, PyTorch forums, GitHub, or blog posts (web search, March 2026).

### Option B: Multi-Adapter Single-Process (mLoRA approach) — RECOMMENDED

A single `apr` process loads the frozen base model once, then concurrently trains multiple LoRA adapters using batch fusion.

**Pros:**
- Base model loaded once — saves (N-1) × model_size VRAM for N concurrent adapters
- No fault isolation problem — single process, single CUDA context
- Application-level scheduling with priority support
- Validated in production (AntGroup deploys mLoRA; Ye et al. arXiv:2312.02515)
- 20-96% throughput improvement over sequential execution (LoRAFusion, arXiv:2510.00206)

**Cons:**
- Requires architectural change to `InstructPipeline` (support multiple adapter sets)
- More complex implementation (~400-600 lines vs ~50 for MPS)
- Single point of failure (process crash loses all adapters)

### Decision

**Phase 1: VRAM guard + sequential queuing (no MPS).** Ship the safety net.
**Phase 2: Multi-adapter single-process.** The literature strongly favors this approach.
**MPS: Not recommended.** Too many foot-guns for the marginal benefit. Can be offered as an opt-in `--experimental-mps` flag for power users.

## Architecture

```
Phase 1: VRAM Guard + Sequential Queue
┌───────────────────────────────────────────┐
│  apr finetune (job 1) ──► GPU (exclusive) │
│  apr finetune (job 2) ──► WAIT (--wait-gpu)│
│  apr finetune (job 3) ──► WAIT            │
│       │           │          │             │
│       ▼           ▼          ▼             │
│  ┌─────────────────────────────┐          │
│  │  VRAM Ledger (flock + JSON) │          │
│  │  - Budget reservations      │          │
│  │  - Dead PID cleanup         │          │
│  │  - Lease-based expiry       │          │
│  └─────────────────────────────┘          │
└───────────────────────────────────────────┘

Phase 2: Multi-Adapter Training
┌───────────────────────────────────────────┐
│  apr finetune --multi-adapter             │
│  ┌─────────────────────────────────┐      │
│  │  Frozen Base Model (loaded once)│      │
│  │  ┌────────┐ ┌────────┐ ┌─────┐ │      │
│  │  │Adapter1│ │Adapter2│ │Ad..N│ │      │
│  │  │LoRA Q/V│ │LoRA Q/V│ │Lo.. │ │      │
│  │  └────────┘ └────────┘ └─────┘ │      │
│  │  BatchLoRA fusion → single GPU  │      │
│  └─────────────────────────────────┘      │
└───────────────────────────────────────────┘

Phase 3: Multi-Node (via forjar)
┌──────────┐  ┌──────────┐  ┌────────────┐
│ 4090 box │  │ Jetson   │  │ Intel(SSH) │
│ 24 GB    │  │ 8 GB     │  │ CPU only   │
│ 2-3 adapt│  │ 1 adapter│  │ ≤350M only │
└────┬─────┘  └────┬─────┘  └──────┬─────┘
     └──────────────┼───────────────┘
                    ▼
           ┌────────────────┐
           │  JobScheduler  │
           └────────────────┘
```

## Phase 1: VRAM Guard + Sequential Queue (G-SHARE-003)

### 1.1 VRAM Guard + Ledger (shipped together — inseparable for correctness)

The VRAM guard and ledger MUST ship together. A guard without a ledger has a TOCTOU race: two jobs check `cuMemGetInfo()` simultaneously, both see enough free VRAM, both allocate, one OOMs.

```rust
/// VRAM reservation ledger. Uses flock for mutual exclusion and
/// atomic write (write-to-temp, rename) for crash safety.
///
/// Contract: C-VRAM-001 — CudaTrainer::new() MUST NOT allocate
/// if ledger + budget exceeds total × reserve_factor.
pub struct VramLedger {
    path: PathBuf,          // ~/.cache/entrenar/gpu-ledger.json
    gpu_uuid: String,       // nvidia-smi -L UUID (not index — survives hotplug)
    reserve_factor: f32,    // 0.85 for discrete, 0.60 for unified memory
}

#[derive(Serialize, Deserialize)]
struct Reservation {
    pid: u32,
    budget_mb: usize,
    task: String,
    started: DateTime<Utc>,
    lease_expires: DateTime<Utc>,  // auto-expire after 24h
}
```

**Protocol:**
1. Acquire `flock` on ledger file
2. Read reservations, prune entries where:
   - PID is dead (`/proc/{pid}/stat` does not exist)
   - Lease expired (started + 24h < now)
3. Check: `sum(active.budget_mb) + my_budget <= total_mb × reserve_factor`
4. Write reservation via atomic rename (`write tmp → rename`)
5. Release lock
6. On exit: best-effort cleanup via `Drop` + `atexit`. Accept that `kill -9` leaves stale entries — lease expiry handles this.

**Reserve factors:**
- Discrete GPU (RTX 4090): 0.85 (15% headroom for driver, JIT, scratch)
- Unified memory (Jetson): 0.60 (40% headroom for OS + system processes)

**GPU identification:** Use UUID from `nvidia-smi -L` (e.g., `GPU-abcd-1234`), NOT index. GPU indices shift if a GPU goes offline.

### 1.2 Wait-and-Retry Mode

```rust
/// Poll ledger until VRAM budget is available.
/// Timeout prevents infinite wait if GPU is permanently occupied.
pub fn wait_for_vram(
    ledger: &VramLedger,
    budget_mb: usize,
    timeout: Duration,
) -> Result<(), GpuError> {
    let start = Instant::now();
    loop {
        // Acquire lock, check ledger + cuMemGetInfo, release lock
        match ledger.try_reserve(budget_mb) {
            Ok(reservation) => return Ok(()),
            Err(GpuError::InsufficientMemory { available, .. }) => {
                if start.elapsed() > timeout {
                    return Err(GpuError::Timeout { budget_mb });
                }
                eprintln!(
                    "[GPU] Waiting for {} MB VRAM ({} MB available, {} reserved)...",
                    budget_mb, available, ledger.total_reserved()
                );
                std::thread::sleep(Duration::from_secs(30));
            }
            Err(e) => return Err(e),
        }
    }
}
```

### 1.3 Actual VRAM Tracking

The `--vram` flag is aspirational, not actual. Kernels allocate scratch buffers, cuBLAS workspaces, and JIT memory outside the budget. The ledger must track **real** peak usage.

```rust
/// After init completes, measure actual VRAM consumption and update ledger.
fn record_actual_vram(ledger: &VramLedger, reservation_id: u64) {
    let (free_after, total) = cuMemGetInfo().unwrap();
    let actual_mb = (total - free_after) / (1024 * 1024);
    ledger.update_actual(reservation_id, actual_mb);
    if actual_mb > budget_mb * 120 / 100 {
        eprintln!(
            "[GPU] WARNING: actual VRAM ({} MB) exceeds budget ({} MB) by {:.0}%",
            actual_mb, budget_mb, (actual_mb as f32 / budget_mb as f32 - 1.0) * 100.0
        );
    }
}
```

### 1.4 CLI Surface

```bash
# Default: VRAM guard + ledger (auto)
apr finetune model.apr --task instruct --method qlora --vram 8

# Wait for VRAM if GPU is busy (default timeout: 1 hour)
apr finetune model.apr --task instruct --method qlora --vram 8 --wait-gpu

# Custom timeout
apr finetune model.apr --task instruct --method qlora --vram 8 --wait-gpu --timeout 3600

# Check GPU status
apr gpu status
# GPU-abcd-1234: RTX 4090 (24.5 GB, discrete)
#   Reserve factor: 85%
#   Processes: 1
#   VRAM: 7.3 / 24.5 GB (30%)
#   Reservations:
#     PID 12345: 8.0 GB budget / 7.3 GB actual (instruct-qlora-7b) — 2h 15m
#   Available for new jobs: 13.5 GB (ledger) / 17.2 GB (cuMemGetInfo)
```

### 1.5 Optional MPS (Experimental, Opt-In)

MPS is **not** auto-started. Available as `--experimental-mps` for users who understand the risks.

```bash
# Opt-in MPS — user accepts fault propagation risk
apr finetune model.apr --vram 8 --experimental-mps --gpu-share 50
```

Requirements when `--experimental-mps` is used:
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set **before** CUDA context creation
- `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` set per client to prevent OOM cascades
- `EXCLUSIVE_PROCESS` compute mode enforced
- Warning printed: "MPS enabled — a GPU fault in any job will crash all jobs"
- Checkpoint frequency increased (every 100 steps) to limit blast radius

### 1.6 Implementation Estimates (revised)

| Change | Location | Lines | Notes |
|--------|----------|-------|-------|
| VRAM guard + error handling | `CudaTrainer::new()` | ~40 | Check cuMemGetInfo + ledger |
| VRAM ledger (flock + JSON) | `entrenar::gpu::ledger` | ~250 | Atomic write, lease expiry, PID check |
| Actual VRAM tracking | `CudaTrainer` post-init | ~30 | cuMemGetInfo after allocation |
| Wait-for-VRAM | `entrenar::gpu::ledger` | ~60 | Poll loop + timeout |
| `apr gpu status` | `apr-cli` | ~120 | NVML query + ledger display |
| `--wait-gpu` / `--vram` flags | `apr-cli` arg parsing | ~20 | Wire through to CudaTrainer |
| Optional MPS (experimental) | `entrenar::gpu::mps` | ~80 | Daemon check, env var, warnings |

**Total: ~600 lines** across entrenar + apr-cli.

### 1.7 Design-by-Contract Requirements (Mandatory)

All GPU sharing components MUST be developed using provable contracts, brick profiling, and layer tracing. No implementation may be merged without its corresponding contract, profiler instrumentation, and trace spans.

#### 1.7.1 Provable Contracts (YAML)

Each component requires a YAML contract in `provable-contracts/contracts/entrenar/`:

| Component | Contract | Key Obligations |
|-----------|----------|-----------------|
| VRAM Ledger | `vram-ledger-v1.yaml` | TOCTOU prevention (flock), atomic write crash safety, lease expiry correctness, dead PID cleanup |
| VRAM Guard | `vram-guard-v1.yaml` | C-VRAM-001 (no alloc if over budget), actual vs budget tracking, OOM prevention |
| Wait Queue | `gpu-wait-queue-v1.yaml` | Timeout guarantee, poll interval bounded, FIFO fairness via lease expiry |

**Contract workflow:**
```bash
# 1. Validate contract
pv validate contracts/entrenar/vram-ledger-v1.yaml

# 2. Generate scaffold + harnesses
pv generate contracts/entrenar/vram-ledger-v1.yaml -o generated/

# 3. Implement against generated trait
# 4. Run property tests (probar)
pv probar contracts/entrenar/vram-ledger-v1.yaml

# 5. Run Kani bounded model checking (where applicable)
cargo kani --harness verify_ledger_capacity_invariant
```

**Mandatory contract elements:**
- `equations:` — Mathematical invariants (capacity arithmetic, timing bounds)
- `proof_obligations:` — Formal properties to verify (invariant, bound, equivalence)
- `falsification_tests:` — Popperian tests that attempt to break each obligation
- `affected_files:` — Exact module paths and function names
- `qa_gate:` — Gate ID for CI integration

#### 1.7.2 Brick Profiling (StepProfiler Pattern)

GPU sharing operations MUST be instrumented with the `StepProfiler` brick-phase pattern (KAIZEN-047). New phases added to the GPU module's profiler:

```rust
// New phases for gpu::ledger profiling
const LEDGER_ACQUIRE: usize = 0;   // flock acquisition time
const LEDGER_READ: usize = 1;      // JSON parse + PID prune
const VRAM_QUERY: usize = 2;       // cuMemGetInfo / NVML call
const LEDGER_WRITE: usize = 3;     // Atomic write (temp + rename)
const LEDGER_RELEASE: usize = 4;   // flock release
const WAIT_POLL: usize = 5;        // Single poll iteration
const NUM_GPU_PHASES: usize = 6;

const GPU_PHASE_NAMES: [&str; NUM_GPU_PHASES] = [
    "lock_acq", "ledger_rd", "vram_qry", "ledger_wr", "lock_rel", "wait_poll",
];
```

**Policy (from KAIZEN-047):** All future GPU sharing optimization tickets MUST cite profiler data, not code-reading estimates. The profiler output is the single source of truth for optimization priority.

**Zero-overhead contract:** When `GpuProfiler` is disabled, all `begin`/`end` calls MUST be no-ops with zero `Instant::now()` calls. Verified by contract C-GPUPROF-001.

#### 1.7.3 Layer Tracing (TraceStep Pattern)

GPU sharing operations MUST emit trace spans via the global `TRACER` (ITP-SPEC-001). New `TraceStep` variants:

```rust
pub enum TraceStep {
    // ... existing variants ...
    /// VRAM ledger lock acquire + reservation
    LedgerReserve,
    /// VRAM ledger cleanup (dead PID + lease expiry)
    LedgerCleanup,
    /// cuMemGetInfo VRAM query
    VramQuery,
    /// Wait-for-VRAM poll iteration
    WaitPoll,
    /// VRAM ledger release on Drop
    LedgerRelease,
}
```

**Integration pattern:**
```rust
TRACER.span(TraceStep::LedgerReserve, format!("budget={budget_mb}MB"), || {
    ledger.try_reserve(budget_mb)
})
```

**Dr. Popper analysis extension:** The trace report MUST classify GPU sharing overhead (ledger I/O, flock contention, NVML calls) vs productive compute. If sharing overhead > 5% of step time, report flags it as falsification: "GPU sharing overhead exceeds budget."

#### 1.7.4 Implementation Gate

No GPU sharing code may be merged without:

1. **Contract exists:** `pv validate` passes on the component's YAML contract
2. **Profiler instrumented:** Every I/O and syscall path has `begin`/`end` phase markers
3. **Tracer spans:** All public functions emit `TRACER.span()` when tracing is enabled
4. **Falsification tests pass:** `pv probar` property tests + at least one Popperian falsification per proof obligation
5. **Profiler data reviewed:** First PR must include profiler output showing overhead is < 5% of training step time

## Phase 2: Multi-Adapter Single-Process (G-SHARE-001, G-SHARE-002)

### 2.1 Design: Shared Base, Multiple Adapters

Inspired by mLoRA (Ye et al. VLDB 2025, arXiv:2312.02515) and LoRAFusion (Zhu et al. arXiv:2510.00206).

```rust
/// A single InstructPipeline that trains N LoRA adapter sets concurrently.
/// The frozen NF4 base model is loaded once. Each adapter set has its own:
/// - LoRA A/B matrices (Q and V projections)
/// - AdamW optimizer state
/// - Training data iterator
/// - Checkpoint directory
pub struct MultiAdapterPipeline {
    base_model: Transformer,              // loaded once, shared
    cuda_blocks: Vec<CudaBlock>,          // NF4 blocks, shared
    adapters: Vec<AdapterSlot>,           // N independent adapter sets
}

struct AdapterSlot {
    lora_layers: Vec<LoRALayer>,
    optimizer_states: Vec<GpuLoraOptimizerState>,
    data_iter: Box<dyn Iterator<Item = PreparedSample>>,
    checkpoint_dir: PathBuf,
    metrics: Vec<InstructEpochMetrics>,
    config: InstructConfig,               // per-adapter hyperparameters
}
```

### 2.2 BatchLoRA Forward/Backward

Instead of running N sequential forward+backward passes through the base model, batch the adapter computations:

```
For each training step:
  1. Forward through shared NF4 blocks (once, for longest sequence)
  2. For each adapter:
     a. Apply LoRA delta: h' = h + B_i @ A_i @ x (fused GEMM)
     b. Compute loss against adapter's target
  3. Backward through shared blocks (once)
  4. For each adapter:
     a. Compute LoRA gradients (adapter-specific)
     b. AdamW update (adapter-specific)
```

**VRAM savings:** N adapters on a 7B model:
- MPS approach: N × 7.3 GB base = 7.3N GB
- Multi-adapter: 7.3 GB base + N × 0.02 GB adapters = 7.3 + 0.02N GB
- 3 adapters: MPS = 21.9 GB vs multi-adapter = 7.36 GB (**3x savings**)

### 2.3 Scheduling Within Process

```rust
enum AdapterSchedule {
    /// All adapters process the same batch (data parallelism over adapters)
    Synchronized,
    /// Round-robin: each step trains one adapter
    RoundRobin,
    /// Priority: adapter with highest val_loss gets the next step
    PriorityValLoss,
}
```

### 2.4 CLI Surface

```bash
# Train 3 adapters concurrently on the same base model
apr finetune model.apr --task instruct --method qlora \
    --adapter data/corpus-a.jsonl:checkpoints/adapter-a \
    --adapter data/corpus-b.jsonl:checkpoints/adapter-b \
    --adapter data/corpus-c.jsonl:checkpoints/adapter-c \
    --rank 16 --epochs 3 --max-seq-len 512

# Or via config file
apr finetune model.apr --task instruct --method qlora \
    --adapters-config adapters.toml
```

### 2.5 Implementation Estimates

| Change | Location | Lines |
|--------|----------|-------|
| `MultiAdapterPipeline` struct | `entrenar::finetune` | ~200 |
| BatchLoRA forward/backward | `instruct_pipeline.rs` | ~150 |
| Per-adapter optimizer state | `cuda_optim.rs` | ~60 |
| Per-adapter checkpointing | `instruct_trainer.rs` | ~80 |
| Adapter scheduling (round-robin) | `instruct_trainer.rs` | ~40 |
| `--adapter` CLI flag | `apr-cli` | ~50 |
| Adapters config TOML parsing | `apr-cli` | ~40 |

**Total: ~620 lines.**

## Phase 3: Multi-Node via Forjar (G-SHARE-004)

### 3.1 Scope Change from v1

The `--replicas` concept is replaced. Running N identical jobs is wasted compute. Instead:

```bash
# Distribute DIFFERENT adapter jobs across nodes
apr train submit --cluster cluster.yaml \
    --adapters-config adapters.toml \
    --model checkpoints/qwen2.5-coder-7b.apr
```

Each node trains different adapters (different data/hyperparameters). NOT replicas.

### 3.2 Cluster Config

```yaml
# ~/.config/entrenar/cluster.yaml
nodes:
  - name: desktop
    host: localhost
    gpus:
      - uuid: GPU-abcd-1234
        type: rtx-4090
        vram_mb: 24564
        memory_type: discrete
    max_adapters: 3

  - name: jetson
    host: jetson.local
    transport: ssh
    gpus:
      - uuid: GPU-efgh-5678
        type: jetson-orin
        vram_mb: 8192
        memory_type: unified    # 60% reserve factor
    max_adapters: 1

  - name: intel-box
    host: 10.0.0.5
    transport: ssh
    user: noah
    gpus: []
    cpu_cores: 16
    ram_mb: 65536
    max_adapters: 1             # CPU-only, ≤350M models
```

### 3.3 Job Placement

Score each node for each adapter job:

```
score = (free_vram / adapter_budget) × gpu_flops_factor × (1 / current_load)
```

Where `gpu_flops_factor`:
- RTX 4090: 1.0 (reference)
- Jetson Orin: 0.06 (8 SMs vs 128)
- CPU (Intel): 0.01

### 3.4 Checkpoint Coordination

Each node runs its adapter independently. All nodes use the same evaluation seed (deterministic val split) so val_loss is comparable across nodes.

Coordinator polls nodes via forjar SSH transport:

```
Every 5 minutes:
  For each node:
    ssh node "cat checkpoint_dir/best/metadata.json"
    Compare val_loss across all adapters
    Report leaderboard to user
```

Best adapter checkpoint is pulled to coordinator at end of training.

### 3.5 Known Limitations

- **Jetson MPS is experimental.** Thread percentage unreliable below 30%. Phase 3 uses multi-adapter (single process), not MPS on Jetson.
- **Cross-compilation required.** `apr` must be built per arch (x86_64, aarch64). forjar can trigger remote `cargo build` or deploy pre-built binaries.
- **Data transfer.** 7B .apr file = 7.5 GB. SCP to Jetson over gigabit takes ~60s. To remote SSH node depends on bandwidth.
- **No shared gradient aggregation.** Each adapter trains independently — this is NOT distributed data parallelism. Adapters may explore different quality/speed tradeoffs.

### 3.6 Implementation Estimates

| Component | Owner | Lines |
|-----------|-------|-------|
| `cluster.yaml` schema + validation | entrenar | ~150 |
| Node health check (SSH + GPU query via NVML) | forjar transport | ~250 |
| Job placement algorithm | entrenar | ~200 |
| Remote job launch via forjar | entrenar + forjar | ~400 |
| Checkpoint polling + aggregation | entrenar | ~150 |
| `apr train submit` CLI | apr-cli | ~100 |
| `apr cluster status` CLI | apr-cli | ~80 |

**Total: ~1,330 lines** across entrenar, apr-cli, and forjar.

## Phasing

| Phase | Scope | Status | Lines | Unlocks |
|-------|-------|--------|-------|---------|
| **1** | VRAM guard + ledger + `--wait-gpu` + `apr gpu status` + MPS | Complete | ~800 | No more silent crashes; sequential queuing |
| **2** | Multi-adapter single-process training + adapters-config TOML | Complete | ~700 | 2-3 concurrent adapters on one GPU, 3x VRAM savings |
| **3** | Multi-node cluster + SSH transport + health check + checkpoint pull | Complete | ~1,200 | Heterogeneous cluster training |

All three phases implemented. 143 GPU tests pass. Zero SATD across all modules.

## Falsification Record

### Self-identified (pre-review)

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| F-001 | MPS thread % is per-context at init, not dynamic | Critical | Dropped MPS as default; opt-in only |
| F-002 | MPS requires root or matching UID | High | Dropped MPS as default |
| F-003 | MPS incompatible with cuda-gdb; crashes cascade | Critical | Dropped MPS as default |
| F-004 | VRAM guard has TOCTOU race without ledger | Critical | Ledger ships with guard (inseparable) |
| F-005 | `--vram` budget doesn't track actual usage | High | Added post-init actual VRAM measurement |
| F-006 | nvidia-smi is slow (200-500ms) and counts non-training procs | Medium | Use NVML (trueno already links libnvidia-ml) |
| F-007 | Multi-node replicas = wasted compute | High | Replaced replicas with distinct adapters |
| F-008 | Jetson unified memory makes VRAM accounting wrong | High | Separate reserve_factor (0.60 vs 0.85) |
| F-009 | Checkpoint aggregation underspecified | High | Deterministic val split + polling protocol |
| F-010 | VRAM fragmentation not handled | Medium | Documented as limitation; pre-allocate at init |

### Batuta Oracle Review

| Issue | Severity | Resolution |
|-------|----------|------------|
| MPS env var must be set before cuCtxCreate | Critical | Dropped MPS as default |
| MPS daemon check (`-l`) hangs if daemon crashed | Medium | Added timeout; use socket check instead |
| MPS auto-start not idempotent across driver versions | Medium | Dropped MPS as default |
| SM partition race between concurrent launches | High | Serialized via ledger lock |
| Flock not crash-safe (kill -9) | High | Added lease-based expiry (24h) |
| 90% reserve factor is arbitrary | Medium | Made configurable; different for unified memory |
| Line count estimates 2.5-3x too low | High | Revised all estimates upward |
| Phase 2 (multi-node) is premature | High | Moved to Phase 3; Phase 2 is now multi-adapter |
| GPU hotplug changes indices | Medium | Use UUID, not index |
| Signal handler + flock = potential deadlock | Medium | Use atexit + lease expiry, not signal handlers |

### ArXiv Literature Review

| Paper | Key Finding | Impact on Spec |
|-------|-------------|----------------|
| Xing et al. 2025 (arXiv:2508.08448) | MPS static SM allocation "fails to achieve high utilization" for dynamic workloads | Validates dropping MPS as default |
| Guardian (arXiv:2401.09290) | Fatal GPU fault under MPS reported to ALL clients | Confirms F-003; MPS too dangerous for multi-job training |
| mLoRA (arXiv:2312.02515, VLDB '25) | Single-process multi-adapter saves (N-1)×model_size VRAM; deployed at AntGroup | Adopted as Phase 2 architecture |
| LoRAFusion (arXiv:2510.00206) | Batch fusion across adapters: 1.47x avg speedup | Validates BatchLoRA approach |
| StellaTrain (SIGCOMM '24) | RTX 4090 = 73% of A100 speed at 1/5 price | Validates consumer GPU as training target |
| SIRIUS (USENIX ATC '25) | Dynamic SM allocation via SM mask outperforms MPS static allocation | Future work: explore SM mask API |
| LithOS (SOSP '25) | Fine-grained TPC-level scheduling: 4.7x latency reduction vs MPS | Validates that MPS is the wrong abstraction |
| Metis (USENIX ATC '24) | Heterogeneous GPU scheduling: 1.05-8.43x speedup | Informs Phase 3 placement algorithm |
| Majeed & Meribout 2025 | Jetson scheduling is a separate research area with different constraints | Validates treating Jetson as a special case |

### Web Search Findings

| Finding | Source | Impact |
|---------|--------|--------|
| Zero documented cases of concurrent QLoRA training with MPS | NVIDIA/PyTorch forums, GitHub | Confirms this is uncharted territory for MPS |
| 1,500x latency regression without explicit thread % | NVIDIA Forum | MPS without explicit config is dangerous |
| MPS on Jetson: thread % unreliable below 30% | NVIDIA Forum | Jetson MPS is experimental-grade |
| `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` prevents OOM cascades | NVIDIA docs | Added to experimental MPS section |
| Databricks: MPS benefits drop for models >3B, hurt for >7B | Databricks blog | Confirms MPS is wrong for 7B QLoRA |
| MPS designed for "cooperative MPI processes", not multi-tenant | NVIDIA docs | MPS is architecturally mismatched |

## Non-Goals

- **CUDA MPS as default** — too many foot-guns; opt-in experimental only.
- **Multi-GPU within one job** (data parallelism / FSDP) — out of scope.
- **Kubernetes / container orchestration** — too heavy for target audience.
- **Dynamic model parallelism** — splitting layers across GPUs is a different problem.
- **Cloud spot instance management** — use SkyPilot for that.
- **Distributed gradient aggregation** — each adapter trains independently.

## References

### Academic Papers
- Ye et al. "mLoRA: Fine-Tuning LoRA Adapters via Pipeline Parallelism" (VLDB '25, arXiv:2312.02515)
- Zhu et al. "LoRAFusion: Efficient LoRA Fine-Tuning for LLMs" (arXiv:2510.00206)
- Xing et al. "Towards Efficient GPU Multitasking in the Era of LLM" (arXiv:2508.08448)
- Pavlidakis et al. "Guardian: Safe GPU Sharing" (arXiv:2401.09290)
- Wang et al. "SIRIUS: Colocating ML Inference and Training" (USENIX ATC '25)
- Coppock et al. "LithOS: GPU Operating System" (SOSP '25, arXiv:2504.15465)
- Um et al. "Metis: Heterogeneous GPU Training" (USENIX ATC '24)
- Lee et al. "ParvaGPU: Efficient Spatial GPU Sharing" (SC '24, arXiv:2409.14447)
- Zheng et al. "Online Scheduling for Multi-LoRA Fine-Tuning" (ICPP '24)
- Gilman et al. "Characterizing Concurrency Mechanisms for NVIDIA GPUs" (arXiv:2110.00459)
- Majeed & Meribout "Scheduling on Heterogeneous Edge GPUs" (arXiv:2506.01377)

### Industry Sources
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/)
- [NVIDIA MPS: When to Use](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
- [Databricks: Scaling Small LLMs with NVIDIA MPS](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps)
- [cuMemGetInfo API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)

### Internal
- forjar SSH transport: `../forjar/src/transport/`
- entrenar CudaTrainer: `src/autograd/cuda_training.rs`
- trueno NVML bindings: `../trueno/trueno-gpu/src/driver/`
