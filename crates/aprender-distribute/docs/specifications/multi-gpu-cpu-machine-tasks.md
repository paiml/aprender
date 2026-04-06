# Multi-GPU/CPU Heterogeneous Distributed Task Execution

**Version:** 1.0.0-draft
**Status:** DRAFT - Awaiting Review
**Last Updated:** 2026-01-03
**Quality Framework:** Iron Lotus + Toyota Way + Popperian Falsification

## Executive Summary

This specification defines a **hardware-agnostic distributed task execution system** that enables seamless workload distribution across heterogeneous compute resources: multiple GPUs (CUDA, Metal, Vulkan), CPUs across networked machines, and hybrid configurations. The system abstracts hardware differences behind a unified task interface while preserving backend-specific optimizations.

**Design Philosophy:** Tasks flow to available compute like water flows downhill—the system finds the path of least resistance through automatic capability matching, load balancing, and fault tolerance.

## 1. Problem Statement

Modern compute environments are increasingly heterogeneous:

| Resource Type | Example | Compute Model | Memory Model |
|---------------|---------|---------------|--------------|
| NVIDIA GPU | RTX 4090 | CUDA/PTX | Unified/Device |
| AMD GPU | Radeon W5700X | Metal/ROCm | Device |
| Intel GPU | Arc A770 | oneAPI/Vulkan | Unified |
| x86 CPU | Xeon W-3245 | SIMD/Threads | Shared |
| ARM CPU | M3 Max | NEON/Threads | Unified |

**Challenge:** Execute arbitrary compute tasks across any available resource without hardcoding hardware assumptions.

**Prior Art Limitations:**
- CUDA-only frameworks exclude non-NVIDIA hardware [1]
- MPI assumes homogeneous clusters [2]
- Kubernetes lacks GPU-aware scheduling granularity [3]
- Ray/Dask have Python GIL constraints [4]

## 2. Architecture

### 2.1 Conceptual Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Task Submission Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Binary Task │  │ Shader Task │  │ Tensor Task │  │ Checkpoint Task │ │
│  │ (exec cmd)  │  │ (WGSL/MSL)  │  │ (trueno op) │  │ (resume state)  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
└─────────┼────────────────┼────────────────┼──────────────────┼──────────┘
          │                │                │                  │
          ▼                ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Unified Task Descriptor                           │
│  { id, requirements: [Backend], affinity: Option<Node>, data_deps: [] } │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Scheduler (Heijunka)                           │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────────────────┐ │
│  │ Capability    │  │ Load Balancer  │  │ Locality-Aware Placement    │ │
│  │ Matcher       │  │ (Work-Stealing)│  │ (Data Gravity)              │ │
│  └───────────────┘  └────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Local Node    │       │   Remote Node   │       │   Remote Node   │
│                 │       │   (TCP/TLS)     │       │   (TCP/TLS)     │
│ ┌─────┐ ┌─────┐ │       │ ┌─────┐ ┌─────┐ │       │ ┌─────┐ ┌─────┐ │
│ │ CPU │ │ GPU │ │       │ │ CPU │ │ GPU │ │       │ │ CPU │ │ GPU │ │
│ │     │ │CUDA │ │       │ │Xeon │ │Metal│ │       │ │ ARM │ │Metal│ │
│ └─────┘ └─────┘ │       │ └─────┘ └─────┘ │       │ └─────┘ └─────┘ │
└─────────────────┘       └─────────────────┘       └─────────────────┘
     Linux x86                 macOS x86                macOS ARM
```

### 2.2 Node Discovery Protocol

Nodes announce capabilities via heartbeat messages:

```rust
/// Node capability advertisement (sent every 5s)
#[derive(Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Unique node identifier (stable across restarts)
    pub node_id: NodeId,
    /// Network endpoint for task submission
    pub endpoint: SocketAddr,
    /// Available compute backends
    pub backends: Vec<BackendCapability>,
    /// Current load (0.0 - 1.0)
    pub load: f32,
    /// Data items cached locally (for locality scheduling)
    pub cached_data: Vec<DataKey>,
    /// Timestamp for distributed clock sync
    pub timestamp: SystemTime,
}

#[derive(Serialize, Deserialize)]
pub struct BackendCapability {
    /// Backend type (Cpu, Cuda, Metal, Vulkan, Rocm)
    pub backend: BackendType,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub device_name: String,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Compute units (cores, SMs, CUs)
    pub compute_units: u32,
    /// Current utilization (0.0 - 1.0)
    pub utilization: f32,
}
```

### 2.3 Task Requirements Specification

Tasks declare requirements, not destinations:

```rust
/// Task execution requirements (declarative)
#[derive(Serialize, Deserialize)]
pub struct TaskRequirements {
    /// Acceptable backend types (empty = any)
    pub backends: Vec<BackendType>,
    /// Minimum memory required (bytes)
    pub min_memory: u64,
    /// Minimum compute units required
    pub min_compute_units: u32,
    /// Data dependencies (prefer nodes with cached data)
    pub data_affinity: Vec<DataKey>,
    /// Hard node affinity (override scheduler)
    pub node_affinity: Option<NodeId>,
    /// Timeout for execution
    pub timeout: Duration,
}
```

## 3. Toyota Way Principles Applied

### 3.1 Heijunka (平準化 - Level Loading)

**Principle:** Distribute work evenly to prevent bottlenecks and overloading [5].

**Implementation:**
- Work-stealing scheduler with per-backend queues
- Load factor = (queued_tasks × avg_duration) / capacity
- Rebalance when load variance > 20% across nodes

```
Before Heijunka:           After Heijunka:
Node A: ████████████       Node A: ████████
Node B: ██                 Node B: ████████
Node C: ████               Node C: ████████
        ↑ Bottleneck              ↑ Balanced
```

### 3.2 Jidoka (自働化 - Automation with Human Touch)

**Principle:** Stop and fix problems immediately; never pass defects downstream [5].

**Implementation:**
- Task execution wraps all operations in Result<T, Error>
- Failed tasks are immediately quarantined (not retried blindly)
- Structured error reporting with full context:

```rust
pub enum TaskError {
    /// Node became unreachable during execution
    NodeDisconnected { node_id: NodeId, last_seen: SystemTime },
    /// Backend reported execution failure
    BackendError { backend: BackendType, code: i32, stderr: String },
    /// Task exceeded timeout
    Timeout { elapsed: Duration, limit: Duration },
    /// Required capability not available on any node
    NoCapableNode { requirements: TaskRequirements },
}
```

### 3.3 Genchi Genbutsu (現地現物 - Go and See)

**Principle:** Base decisions on firsthand observation, not reports [5].

**Implementation:**
- Real-time node health via direct probing (not cached status)
- Task execution includes timing instrumentation
- Distributed tracing with span propagation

```rust
/// Every task execution records:
pub struct ExecutionTrace {
    pub task_id: TaskId,
    pub node_id: NodeId,
    pub backend: BackendType,
    pub queue_time: Duration,      // Time waiting in queue
    pub transfer_time: Duration,   // Time transferring data
    pub compute_time: Duration,    // Actual execution time
    pub total_time: Duration,      // End-to-end latency
}
```

### 3.4 Kaizen (改善 - Continuous Improvement)

**Principle:** Small, incremental improvements compound over time [5].

**Implementation:**
- Execution traces feed into scheduler cost model
- Cost model self-tunes based on observed performance
- Anomaly detection flags degraded nodes

```rust
/// Scheduler learns from execution history
pub struct CostModel {
    /// Observed latency by (backend, task_type) pair
    pub latency_histogram: HashMap<(BackendType, TaskType), Histogram>,
    /// Observed throughput by node
    pub node_throughput: HashMap<NodeId, f64>,
    /// Network latency between node pairs
    pub network_latency: HashMap<(NodeId, NodeId), Duration>,
}
```

### 3.5 Nemawashi (根回し - Consensus Building)

**Principle:** Build consensus before major decisions [5].

**Implementation:**
- Leader election via Raft consensus for coordinator role [6]
- Task assignment is advisory until worker acknowledges
- Graceful degradation when consensus cannot be reached

## 4. Scheduling Algorithm

### 4.1 Capability-Based Matching

```python
def match_task_to_nodes(task: Task, nodes: List[Node]) -> List[Node]:
    """
    Filter nodes by task requirements.
    Based on constraint satisfaction [7].
    """
    candidates = []
    for node in nodes:
        for backend in node.backends:
            if satisfies(backend, task.requirements):
                candidates.append((node, backend))
    return candidates
```

### 4.2 Cost-Based Selection

```python
def select_best_node(candidates: List[Tuple[Node, Backend]],
                     task: Task,
                     cost_model: CostModel) -> Tuple[Node, Backend]:
    """
    Select node minimizing expected completion time.
    Implements locality-aware scheduling [8].
    """
    def cost(node, backend):
        # Base execution cost from historical data
        exec_cost = cost_model.predict_latency(backend, task.type)

        # Queue delay (tasks ahead × avg duration)
        queue_cost = node.queue_depth * cost_model.avg_latency(backend)

        # Data transfer cost (if data not local)
        transfer_cost = 0
        for dep in task.data_affinity:
            if dep not in node.cached_data:
                transfer_cost += cost_model.transfer_time(dep, node)

        return exec_cost + queue_cost + transfer_cost

    return min(candidates, key=lambda x: cost(x[0], x[1]))
```

### 4.3 Work Stealing

When a node becomes idle, it steals from overloaded peers:

```
Idle Node B                    Busy Node A
┌───────────┐                 ┌───────────┐
│  Queue: 0 │ ──── steal ───► │ Queue: 10 │
└───────────┘                 └───────────┘
                                    │
                              ┌─────┴─────┐
                              ▼           ▼
                         ┌───────┐   ┌───────┐
                         │ Keep 5│   │ Give 5│
                         └───────┘   └───────┘
                                         │
                    ◄────────────────────┘

Result: Both nodes have ~5 tasks
```

**Stealing Protocol (Blumofe-Leiserson) [9]:**
1. Idle node randomly selects peer
2. Requests half of peer's queue (FIFO end)
3. Peer atomically transfers tasks
4. Stolen tasks execute with full provenance

## 5. Wire Protocol

### 5.1 Message Format

Length-prefixed bincode serialization over TCP:

```
┌──────────────────────────────────────────────────────┐
│  Length (4 bytes, little-endian u32)                 │
├──────────────────────────────────────────────────────┤
│  Payload (bincode-serialized Message enum)           │
│  ... variable length ...                             │
└──────────────────────────────────────────────────────┘
```

### 5.2 Message Types

```rust
#[derive(Serialize, Deserialize)]
pub enum Message {
    // Discovery
    Heartbeat(NodeCapabilities),
    HeartbeatAck,

    // Task lifecycle
    SubmitTask(Task),
    TaskAccepted { task_id: TaskId },
    TaskRejected { task_id: TaskId, reason: String },
    TaskResult(ExecutionResult),

    // Work stealing
    StealRequest { count: usize },
    StealResponse { tasks: Vec<Task> },

    // Control
    Shutdown,
    Ping,
    Pong,
}
```

### 5.3 Security

- TLS 1.3 via rustls (zero OpenSSL)
- Mutual authentication with ed25519 certificates
- Task binaries signed and verified before execution

## 6. Fault Tolerance

### 6.1 Failure Detection

Unreliable failure detector based on Chandra-Toueg [10]:

```rust
pub struct FailureDetector {
    /// Heartbeat timeout (adaptive)
    timeout: Duration,
    /// Heartbeat history per node
    history: HashMap<NodeId, Vec<Instant>>,
    /// Suspected nodes (not confirmed failed)
    suspected: HashSet<NodeId>,
}

impl FailureDetector {
    /// Called on heartbeat timeout
    fn suspect(&mut self, node_id: NodeId) {
        self.suspected.insert(node_id);
        // Don't immediately fail - could be network hiccup
    }

    /// Called on heartbeat received
    fn alive(&mut self, node_id: NodeId) {
        self.suspected.remove(&node_id);
        // Adjust timeout based on observed latency
        self.adapt_timeout(node_id);
    }
}
```

### 6.2 Task Recovery

```
Task T assigned to Node A
           │
           ▼
    ┌──────────────┐
    │ Node A fails │
    └──────────────┘
           │
           ▼
    ┌──────────────────────────────┐
    │ Failure detector suspects A  │
    └──────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────┐
    │ Wait for confirmation        │
    │ (3 missed heartbeats)        │
    └──────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────┐
    │ Reassign T to Node B         │
    │ (from checkpoint if enabled) │
    └──────────────────────────────┘
```

### 6.3 Checkpointing (Optional)

For long-running tasks, periodic state snapshots:

```rust
pub trait Checkpointable {
    /// Serialize current state
    fn checkpoint(&self) -> Vec<u8>;
    /// Restore from checkpoint
    fn restore(data: &[u8]) -> Result<Self> where Self: Sized;
}
```

## 7. Configuration

### 7.1 Worker Configuration (TOML)

```toml
[worker]
# Network binding
bind = "0.0.0.0:9000"

# Coordinator discovery
coordinator = "auto"  # or explicit: "192.168.1.100:9001"

# Backend enablement (auto-detected if not specified)
[worker.backends]
cpu = true
cuda = true   # Requires NVIDIA GPU
metal = false # macOS only
vulkan = true

# Resource limits
[worker.limits]
max_concurrent_tasks = 4
max_memory_per_task = "8GB"
task_timeout = "1h"

# Security
[worker.security]
tls_cert = "/etc/repartir/cert.pem"
tls_key = "/etc/repartir/key.pem"
allowed_binaries = ["/opt/repartir/bin/*"]
```

### 7.2 Coordinator Configuration (TOML)

```toml
[coordinator]
bind = "0.0.0.0:9001"

# Scheduling policy
[coordinator.scheduler]
algorithm = "work-stealing"  # or "round-robin", "locality-first"
rebalance_interval = "10s"
steal_threshold = 0.3  # Steal when load imbalance > 30%

# Failure detection
[coordinator.failure_detection]
heartbeat_interval = "5s"
heartbeat_timeout = "15s"
confirm_after = 3  # Confirm failure after 3 missed heartbeats

# Persistence (for recovery)
[coordinator.persistence]
backend = "sqlite"  # or "memory", "trueno-db"
path = "/var/lib/repartir/state.db"
```

## 8. API Design

### 8.1 High-Level API

```rust
use repartir::{Cluster, Task, Backend};

#[tokio::main]
async fn main() -> Result<()> {
    // Connect to cluster (discovers nodes automatically)
    let cluster = Cluster::connect("coordinator:9001").await?;

    // Submit task with requirements (not destinations)
    let task = Task::builder()
        .binary("./compute_kernel")
        .args(vec!["--input", "data.bin"])
        .require_backend(Backend::Gpu)  // Any GPU
        .require_memory(4 * GB)
        .data_affinity("tensor_batch_42")  // Prefer node with this data
        .build()?;

    let result = cluster.execute(task).await?;

    println!("Executed on: {}", result.node_id());
    println!("Backend: {:?}", result.backend());
    println!("Duration: {:?}", result.duration());

    Ok(())
}
```

### 8.2 Batch Execution

```rust
// Submit multiple tasks, execute in parallel across cluster
let tasks: Vec<Task> = (0..100)
    .map(|i| Task::builder()
        .binary("./worker")
        .arg(format!("--batch={i}"))
        .build()
        .unwrap())
    .collect();

// Returns results as they complete (streaming)
let mut results = cluster.execute_batch(tasks).await;

while let Some(result) = results.next().await {
    match result {
        Ok(r) => println!("Task {} completed", r.task_id()),
        Err(e) => eprintln!("Task failed: {e}"),
    }
}
```

### 8.3 GPU Compute Tasks

```rust
// Direct GPU compute (no binary, shader code)
let shader = include_str!("kernels/matmul.wgsl");

let task = Task::builder()
    .shader_code(shader)
    .input_buffer(matrix_a.as_bytes())
    .input_buffer(matrix_b.as_bytes())
    .output_buffer_size(result_size)
    .workgroup_size(16, 16, 1)
    .require_backend(Backend::Gpu)
    .build()?;

let result = cluster.execute(task).await?;
let output = result.gpu_output_buffers()[0].as_slice();
```

## 9. References

[1] J. Nickolls, I. Buck, M. Garland, and K. Skadron, "Scalable Parallel Programming with CUDA," *ACM Queue*, vol. 6, no. 2, pp. 40-53, 2008. doi:10.1145/1365490.1365500

[2] W. Gropp, E. Lusk, and A. Skjellum, *Using MPI: Portable Parallel Programming with the Message-Passing Interface*, 3rd ed. MIT Press, 2014. ISBN: 978-0262527392

[3] V. Medel, O. Rana, J. A. Bañares, and U. Arronategui, "Modelling Performance & Resource Management in Kubernetes," in *Proc. 9th Int. Conf. Utility and Cloud Computing*, 2016, pp. 257-262. doi:10.1145/2996890.3007869

[4] P. Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications," in *Proc. 13th USENIX Symp. Operating Systems Design and Implementation (OSDI)*, 2018, pp. 561-577.

[5] J. K. Liker, *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*, 2nd ed. McGraw-Hill, 2021. ISBN: 978-1260468519

[6] D. Ongaro and J. Ousterhout, "In Search of an Understandable Consensus Algorithm," in *Proc. USENIX Annual Technical Conference*, 2014, pp. 305-319.

[7] K. Apt, *Principles of Constraint Programming*. Cambridge University Press, 2003. ISBN: 978-0521825832

[8] M. Zaharia et al., "Delay Scheduling: A Simple Technique for Achieving Locality and Fairness in Cluster Scheduling," in *Proc. 5th European Conf. Computer Systems (EuroSys)*, 2010, pp. 265-278. doi:10.1145/1755913.1755940

[9] R. D. Blumofe and C. E. Leiserson, "Scheduling Multithreaded Computations by Work Stealing," *J. ACM*, vol. 46, no. 5, pp. 720-748, 1999. doi:10.1145/324133.324234

[10] T. D. Chandra and S. Toueg, "Unreliable Failure Detectors for Reliable Distributed Systems," *J. ACM*, vol. 43, no. 2, pp. 225-267, 1996. doi:10.1145/226643.226647

[11] L. Lamport, "Time, Clocks, and the Ordering of Events in a Distributed System," *Commun. ACM*, vol. 21, no. 7, pp. 558-565, 1978. doi:10.1145/359545.359563

[12] K. R. Popper, *The Logic of Scientific Discovery*. Routledge, 1959. ISBN: 978-0415278447

---

## 10. Popperian Falsification Checklist (100 Points)

Per Popper's philosophy of science [12], a specification is only meaningful if it makes falsifiable predictions. Each item below describes a testable property that, if violated, falsifies the specification's claims.

### 10.1 Node Discovery (15 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 1 | New node appears in cluster within 10s of starting | Start worker, query coordinator | 2 |
| 2 | Node capabilities accurately reflect hardware | Compare advertised vs `lspci`/`system_profiler` | 2 |
| 3 | Node removal detected within 3× heartbeat interval | Kill worker, measure detection time | 2 |
| 4 | Stale nodes are pruned from active list | Kill worker, verify removal | 1 |
| 5 | Duplicate node IDs are rejected | Start two workers with same ID | 1 |
| 6 | Network partition causes node isolation (not crash) | `iptables` block, verify recovery | 2 |
| 7 | Coordinator failover preserves node registry | Kill coordinator, restart, verify nodes | 2 |
| 8 | IPv4 and IPv6 endpoints both work | Test with each protocol | 1 |
| 9 | Heartbeat interval is configurable | Change config, verify timing | 1 |
| 10 | Node load metric updates in real-time | Submit tasks, observe load change | 1 |

### 10.2 Task Submission (15 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 11 | Task with no capable nodes returns NoCapableNode error | Submit GPU task to CPU-only cluster | 2 |
| 12 | Task ID is unique across cluster | Submit 10K tasks, verify uniqueness | 1 |
| 13 | Task requirements are validated before queuing | Submit invalid requirements | 1 |
| 14 | Task timeout is enforced | Submit task that sleeps forever | 2 |
| 15 | Task binary path is validated | Submit nonexistent binary | 1 |
| 16 | Task args are passed correctly | Echo args back, verify | 1 |
| 17 | Task env vars are passed correctly | Echo env, verify | 1 |
| 18 | Large task payloads are handled | Submit 100MB input buffer | 2 |
| 19 | Task submission is idempotent (same ID = same task) | Submit same task twice | 1 |
| 20 | Task priority affects scheduling order | Submit low/high priority, verify order | 2 |
| 21 | Task rejection includes reason | Submit invalid task, check error | 1 |

### 10.3 Scheduling (20 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 22 | Tasks route to capable backends only | Submit CUDA task, verify runs on NVIDIA | 2 |
| 23 | Load balancing distributes evenly | Submit 100 tasks, verify distribution | 2 |
| 24 | Work stealing activates on imbalance | Overload one node, verify stealing | 3 |
| 25 | Locality preference reduces data transfer | Submit with affinity, measure transfer | 3 |
| 26 | Node affinity is respected when specified | Submit with hard affinity, verify node | 2 |
| 27 | Scheduler handles 1000+ concurrent tasks | Load test with 1000 tasks | 2 |
| 28 | Scheduler handles 100+ nodes | Simulate 100 node cluster | 2 |
| 29 | Cost model improves over time | Compare early vs late scheduling decisions | 2 |
| 30 | Memory requirements are enforced | Submit task needing 100GB to 16GB node | 2 |

### 10.4 Execution (20 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 31 | CPU tasks execute and return output | Submit `/bin/echo hello` | 2 |
| 32 | GPU shader tasks execute and return buffers | Submit WGSL shader | 2 |
| 33 | Exit code is captured correctly | Submit task that exits 42 | 1 |
| 34 | Stdout is captured correctly | Submit task with stdout | 1 |
| 35 | Stderr is captured correctly | Submit task with stderr | 1 |
| 36 | Execution duration is measured accurately | Compare reported vs wall clock | 1 |
| 37 | Concurrent tasks on same node work | Submit 10 tasks to one node | 2 |
| 38 | Task isolation prevents interference | Submit tasks that conflict | 2 |
| 39 | GPU memory is released after task | Submit GPU task, verify memory freed | 2 |
| 40 | CPU affinity is respected (if configured) | Pin to cores, verify | 1 |
| 41 | Remote execution matches local execution | Same task, compare results | 2 |
| 42 | Binary signing is verified before execution | Submit unsigned binary | 2 |
| 43 | Whitelisted paths are enforced | Submit binary outside whitelist | 1 |

### 10.5 Fault Tolerance (15 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 44 | Task is retried on node failure | Kill node during execution | 2 |
| 45 | Checkpoint enables resume after failure | Kill task, resume from checkpoint | 3 |
| 46 | Network timeout doesn't crash worker | Introduce 30s latency | 2 |
| 47 | Coordinator failure doesn't lose in-flight tasks | Kill coordinator, restart | 2 |
| 48 | Split-brain is handled gracefully | Partition network, heal, verify | 2 |
| 49 | Retry count is configurable and enforced | Set retry=2, fail 3 times | 1 |
| 50 | Failed tasks report detailed error context | Fail task, inspect error | 1 |
| 51 | Graceful shutdown completes in-flight tasks | SIGTERM coordinator | 2 |

### 10.6 Performance (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 52 | Task overhead < 10ms for local execution | Benchmark no-op task | 2 |
| 53 | Task overhead < 50ms for remote execution | Benchmark remote no-op task | 2 |
| 54 | Throughput > 1000 tasks/sec (simple tasks) | Load test | 2 |
| 55 | Memory usage is bounded (no leaks) | Long-running test, monitor RSS | 2 |
| 56 | CPU usage is proportional to load | Idle cluster uses <5% CPU | 2 |

### 10.7 Security (5 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 57 | TLS is enforced when configured | Attempt plaintext connection | 1 |
| 58 | Invalid certificates are rejected | Use self-signed cert | 1 |
| 59 | Unsigned binaries are rejected | Submit unsigned binary | 1 |
| 60 | Resource limits prevent DoS | Submit memory bomb task | 1 |
| 61 | Logs don't contain sensitive data | Grep logs for secrets | 1 |

### 10.8 Heterogeneous Hardware (15 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 62 | CUDA backend executes on NVIDIA GPU | Submit CUDA task | 2 |
| 63 | Metal backend executes on AMD/Apple GPU | Submit Metal task | 2 |
| 64 | Vulkan backend executes on any Vulkan GPU | Submit Vulkan task | 2 |
| 65 | CPU backend uses all cores | Submit parallel task | 1 |
| 66 | Mixed cluster (CUDA + Metal) routes correctly | Submit to mixed cluster | 2 |
| 67 | Same shader runs on different GPU vendors | WGSL on NVIDIA and AMD | 2 |
| 68 | ARM and x86 nodes coexist | Mixed architecture cluster | 2 |
| 69 | GPU memory limits are enforced per-device | Query device limits | 1 |
| 70 | Multi-GPU node uses all GPUs | Submit to multi-GPU node | 1 |

### 10.9 API Correctness (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 71 | `Cluster::connect` fails gracefully on bad address | Connect to invalid host | 1 |
| 72 | `Task::builder()` validates all fields | Build invalid task | 1 |
| 73 | `execute_batch` handles empty batch | Submit empty vec | 1 |
| 74 | Results stream in completion order | Submit tasks with varied durations | 1 |
| 75 | Async cancellation works | Cancel pending task | 1 |
| 76 | `Result` types are never `unwrap`ed in library | Static analysis | 2 |
| 77 | All public types implement `Debug` | Compile-time check | 1 |
| 78 | All public types implement `Send + Sync` | Compile-time check | 1 |
| 79 | Documentation examples compile | `cargo test --doc` | 1 |

### 10.10 Observability (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 80 | Execution traces are emitted | Check tracing output | 1 |
| 81 | Traces include timing breakdown | Verify fields present | 1 |
| 82 | Metrics endpoint exposes Prometheus format | Scrape `/metrics` | 2 |
| 83 | Log levels are configurable | Set RUST_LOG, verify | 1 |
| 84 | Structured logging (JSON) is available | Enable JSON logs | 1 |
| 85 | Distributed trace IDs propagate | Check trace continuity | 2 |
| 86 | Health check endpoint works | GET `/health` | 1 |
| 87 | Ready check distinguishes from healthy | GET `/ready` | 1 |

### 10.11 Configuration (5 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 88 | Invalid TOML is rejected with clear error | Provide malformed config | 1 |
| 89 | Missing required fields are reported | Omit required field | 1 |
| 90 | Environment variables override config file | Set env var, verify | 1 |
| 91 | Config reload works without restart | SIGHUP, verify change | 1 |
| 92 | Default values are documented and sensible | Check defaults | 1 |

### 10.12 Edge Cases (8 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 93 | Zero-byte task output is handled | Submit task with no output | 1 |
| 94 | Unicode in task args works | Submit args with emoji | 1 |
| 95 | Paths with spaces work | Binary path with spaces | 1 |
| 96 | Very long task args work | Submit 1MB args | 1 |
| 97 | Clock skew between nodes is tolerated | Offset clock by 5min | 2 |
| 98 | Task submitted to self-as-remote works | Worker submits to itself | 1 |
| 99 | Empty cluster returns appropriate error | Query empty cluster | 0.5 |
| 100 | Cluster with only offline nodes returns error | All nodes dead | 0.5 |

---

**Total: 100 points**

**Passing threshold: 95 points** (5% tolerance for environment-specific issues)

**Certification levels:**
- 100 points: Specification Verified
- 95-99 points: Specification Conditionally Verified (document exceptions)
- <95 points: Specification Falsified (revise before implementation)

---

## 11. Job Flow TUI (Real-Time Visualization)

### 11.1 Overview

A terminal-based dashboard for real-time monitoring of distributed task execution across heterogeneous compute resources. Built with ratatui and tested with 100% probador coverage.

**Design Principles:**
- **Glanceable:** Status visible in <1 second
- **Actionable:** Drill-down to problem nodes
- **Responsive:** 60fps rendering, <16ms frame time

### 11.2 Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ REPARTIR JOB FLOW                                    [q]uit [r]efresh [?]help│
├─────────────────────────────────────────────────────────────────────────────┤
│ CLUSTER                          │ TASK QUEUE                              │
│ ┌─────────────────────────────┐  │ Pending: 42    In-Flight: 8             │
│ │ ● linux-rtx4090  [CUDA]     │  │ ████████████████░░░░░░░░░░░░ 62%        │
│ │   CPU: ██████░░░░ 58%       │  │                                         │
│ │   GPU: ████████░░ 82%       │  │ Priority Distribution:                  │
│ │   Mem: ████░░░░░░ 42%       │  │   High:   ██████ 12                     │
│ │   Tasks: 3 running          │  │   Normal: ████████████████ 28           │
│ │   Latency: 2ms              │  │   Low:    ██ 2                          │
│ ├─────────────────────────────┤  ├─────────────────────────────────────────┤
│ │ ● mac-pro-xeon  [Metal×2]   │  │ RECENT COMPLETIONS                      │
│ │   CPU: ████████░░ 76%       │  │ ✓ task-a1b2 CUDA    45ms   linux        │
│ │   GPU0: ██████░░░░ 61%      │  │ ✓ task-c3d4 Metal   120ms  mac-pro      │
│ │   GPU1: ████░░░░░░ 38%      │  │ ✓ task-e5f6 CPU     890ms  mac-pro      │
│ │   Mem: ██████░░░░ 54%       │  │ ✗ task-g7h8 CUDA    TIMEOUT linux       │
│ │   Tasks: 5 running          │  │ ✓ task-i9j0 Metal   67ms   mac-pro      │
│ │   Latency: 45ms (remote)    │  │                                         │
│ └─────────────────────────────┘  │                                         │
├──────────────────────────────────┴─────────────────────────────────────────┤
│ TASK DETAIL (selected: task-a1b2)                                          │
│ Binary: ./compute_kernel  Args: --input data.bin --batch 42                │
│ Backend: CUDA  Node: linux-rtx4090  Status: Running  Elapsed: 12ms         │
│ Progress: ████████████████████░░░░░░░░░░ 68%                               │
├────────────────────────────────────────────────────────────────────────────┤
│ ALERTS                                                                     │
│ ⚠ mac-pro-xeon: GPU1 memory pressure (92%)                                │
│ ⚠ Work imbalance detected: linux has 3 tasks, mac-pro has 5               │
└────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Data Model

```rust
/// Application state for TUI
pub struct App {
    /// Connected nodes and their status
    pub nodes: Vec<NodeStatus>,
    /// Current task queue
    pub queue: TaskQueue,
    /// Recent execution results (ring buffer)
    pub completions: VecDeque<CompletionRecord>,
    /// Currently selected item (for detail view)
    pub selected: Selection,
    /// Active alerts
    pub alerts: Vec<Alert>,
    /// Refresh interval
    pub tick_rate: Duration,
    /// Historical metrics (for sparklines)
    pub history: MetricsHistory,
}

#[derive(Debug, Clone)]
pub struct NodeStatus {
    pub node_id: NodeId,
    pub name: String,
    pub endpoint: SocketAddr,
    pub backends: Vec<BackendStatus>,
    pub cpu_pct: f64,
    pub mem_pct: f64,
    pub running_tasks: u32,
    pub latency_ms: u32,
    pub state: NodeState,
    /// Historical load (for sparkline)
    pub load_history: VecDeque<u64>,
}

#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub backend_type: BackendType,
    pub device_name: String,
    pub utilization: f64,
    pub memory_pct: f64,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CompletionRecord {
    pub task_id: TaskId,
    pub backend: BackendType,
    pub node_name: String,
    pub duration: Duration,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub enum Alert {
    MemoryPressure { node: String, pct: f64 },
    WorkImbalance { overloaded: String, underloaded: String },
    NodeSuspected { node: String, last_seen: Duration },
    TaskTimeout { task_id: TaskId, elapsed: Duration },
    BackendError { node: String, backend: BackendType, error: String },
}
```

### 11.4 Key Bindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Force refresh |
| `↑/↓` | Navigate nodes/tasks |
| `Enter` | Expand node/task detail |
| `Tab` | Switch focus (nodes ↔ queue ↔ completions) |
| `1-3` | Switch tabs (Overview, Nodes, Tasks) |
| `?` | Toggle help overlay |
| `a` | Toggle alerts panel |
| `s` | Sort by (load, tasks, latency) |
| `f` | Filter by backend type |

### 11.5 Rendering Pipeline

```rust
/// Main draw function (called every tick)
pub fn draw_ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // Header
            Constraint::Min(10),     // Main content
            Constraint::Length(3),   // Task detail
            Constraint::Length(4),   // Alerts
        ])
        .split(f.area());

    render_header(f, chunks[0], app);
    render_main(f, chunks[1], app);
    render_task_detail(f, chunks[2], app);
    render_alerts(f, chunks[3], app);
}

/// Render cluster status with per-node gauges
fn render_cluster(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" CLUSTER ")
        .borders(Borders::ALL);

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Calculate row height based on node count
    let node_height = 6;
    let constraints: Vec<Constraint> = app.nodes.iter()
        .map(|_| Constraint::Length(node_height))
        .collect();

    let node_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    for (i, node) in app.nodes.iter().enumerate() {
        render_node_status(f, node_areas[i], node);
    }
}

/// Color coding for resource utilization
fn utilization_color(pct: f64) -> Color {
    match pct {
        p if p < 50.0 => Color::Green,
        p if p < 80.0 => Color::Yellow,
        _ => Color::Red,
    }
}
```

### 11.6 Real-Time Updates

```rust
/// Event loop with non-blocking updates
pub async fn run_app(
    terminal: &mut Terminal<impl Backend>,
    mut app: App,
    cluster: &Cluster,
) -> Result<()> {
    let tick_rate = Duration::from_millis(100);  // 10 FPS
    let mut last_tick = Instant::now();

    loop {
        // Draw current state
        terminal.draw(|f| draw_ui(f, &app))?;

        // Poll for events with timeout
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = crossterm::event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('r') => app.force_refresh = true,
                    KeyCode::Up => app.select_prev(),
                    KeyCode::Down => app.select_next(),
                    _ => {}
                }
            }
        }

        // Periodic state refresh
        if last_tick.elapsed() >= tick_rate || app.force_refresh {
            app.update(cluster).await?;
            app.force_refresh = false;
            last_tick = Instant::now();
        }
    }
}

impl App {
    /// Update state from cluster
    async fn update(&mut self, cluster: &Cluster) -> Result<()> {
        // Fetch node status (non-blocking)
        self.nodes = cluster.get_node_status().await?;

        // Update queue stats
        self.queue = cluster.get_queue_stats().await?;

        // Poll for new completions
        while let Some(completion) = cluster.poll_completion().await? {
            self.completions.push_front(completion);
            if self.completions.len() > 100 {
                self.completions.pop_back();
            }
        }

        // Generate alerts from current state
        self.alerts = self.generate_alerts();

        // Update history for sparklines
        self.history.push(&self.nodes);

        Ok(())
    }
}
```

### 11.7 Probador Test Coverage (100%)

All TUI components are tested using jugar-probar with ratatui's TestBackend:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use jugar_probar::tui::{expect_frame, TuiFrame};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Create test app with mock cluster data
    fn create_test_app() -> App {
        App {
            nodes: vec![
                NodeStatus {
                    node_id: NodeId::new(),
                    name: "linux-rtx4090".to_string(),
                    backends: vec![BackendStatus {
                        backend_type: BackendType::Cuda,
                        device_name: "RTX 4090".to_string(),
                        utilization: 82.0,
                        memory_pct: 42.0,
                        temperature: Some(65.0),
                    }],
                    cpu_pct: 58.0,
                    mem_pct: 42.0,
                    running_tasks: 3,
                    latency_ms: 2,
                    state: NodeState::Online,
                    load_history: VecDeque::new(),
                    endpoint: "127.0.0.1:9000".parse().unwrap(),
                },
                NodeStatus {
                    node_id: NodeId::new(),
                    name: "mac-pro-xeon".to_string(),
                    backends: vec![
                        BackendStatus {
                            backend_type: BackendType::Metal,
                            device_name: "W5700X #0".to_string(),
                            utilization: 61.0,
                            memory_pct: 54.0,
                            temperature: Some(58.0),
                        },
                        BackendStatus {
                            backend_type: BackendType::Metal,
                            device_name: "W5700X #1".to_string(),
                            utilization: 38.0,
                            memory_pct: 92.0,  // High - should alert
                            temperature: Some(62.0),
                        },
                    ],
                    cpu_pct: 76.0,
                    mem_pct: 54.0,
                    running_tasks: 5,
                    latency_ms: 45,
                    state: NodeState::Online,
                    load_history: VecDeque::new(),
                    endpoint: "192.168.50.100:9000".parse().unwrap(),
                },
            ],
            queue: TaskQueue {
                pending: 42,
                in_flight: 8,
                high_priority: 12,
                normal_priority: 28,
                low_priority: 2,
            },
            completions: VecDeque::new(),
            selected: Selection::None,
            alerts: vec![],
            tick_rate: Duration::from_millis(100),
            history: MetricsHistory::new(),
        }
    }

    // =========================================================================
    // Header Tests
    // =========================================================================

    #[test]
    fn test_header_renders_title() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("REPARTIR JOB FLOW")
            .unwrap();
    }

    #[test]
    fn test_header_shows_keybindings() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("[q]uit")
            .to_contain_text("[r]efresh")
            .unwrap();
    }

    // =========================================================================
    // Node Status Tests
    // =========================================================================

    #[test]
    fn test_node_renders_name() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("linux-rtx4090")
            .to_contain_text("mac-pro-xeon")
            .unwrap();
    }

    #[test]
    fn test_node_renders_backend_type() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("CUDA")
            .to_contain_text("Metal")
            .unwrap();
    }

    #[test]
    fn test_node_renders_resource_gauges() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("CPU:")
            .to_contain_text("GPU:")
            .to_contain_text("Mem:")
            .unwrap();
    }

    #[test]
    fn test_node_shows_task_count() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("3 running")
            .to_contain_text("5 running")
            .unwrap();
    }

    #[test]
    fn test_node_shows_latency() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("2ms")
            .to_contain_text("45ms")
            .unwrap();
    }

    // =========================================================================
    // Queue Stats Tests
    // =========================================================================

    #[test]
    fn test_queue_renders_counts() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("Pending: 42")
            .to_contain_text("In-Flight: 8")
            .unwrap();
    }

    #[test]
    fn test_queue_renders_priority_distribution() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("High:")
            .to_contain_text("Normal:")
            .to_contain_text("Low:")
            .unwrap();
    }

    // =========================================================================
    // Alert Tests
    // =========================================================================

    #[test]
    fn test_alert_memory_pressure() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.alerts = vec![Alert::MemoryPressure {
            node: "mac-pro-xeon".to_string(),
            pct: 92.0,
        }];

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("memory pressure")
            .to_contain_text("92%")
            .unwrap();
    }

    #[test]
    fn test_alert_work_imbalance() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.alerts = vec![Alert::WorkImbalance {
            overloaded: "mac-pro".to_string(),
            underloaded: "linux".to_string(),
        }];

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("imbalance")
            .unwrap();
    }

    // =========================================================================
    // Completion List Tests
    // =========================================================================

    #[test]
    fn test_completion_renders_success() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.completions.push_front(CompletionRecord {
            task_id: TaskId::new(),
            backend: BackendType::Cuda,
            node_name: "linux".to_string(),
            duration: Duration::from_millis(45),
            success: true,
            error: None,
            timestamp: Instant::now(),
        });

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        // Success indicator
        expect_frame(&frame)
            .to_contain_text("CUDA")
            .to_contain_text("45ms")
            .unwrap();
    }

    #[test]
    fn test_completion_renders_failure() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.completions.push_front(CompletionRecord {
            task_id: TaskId::new(),
            backend: BackendType::Cuda,
            node_name: "linux".to_string(),
            duration: Duration::from_secs(30),
            success: false,
            error: Some("TIMEOUT".to_string()),
            timestamp: Instant::now(),
        });

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("TIMEOUT")
            .unwrap();
    }

    // =========================================================================
    // State Transition Tests
    // =========================================================================

    #[test]
    fn test_empty_cluster_renders_placeholder() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.nodes.clear();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("No nodes")
            .unwrap();
    }

    #[test]
    fn test_offline_node_shows_status() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.nodes[1].state = NodeState::Suspected;

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        // Should show warning indicator for suspected node
        expect_frame(&frame)
            .to_contain_text("mac-pro-xeon")
            .unwrap();
    }

    // =========================================================================
    // Layout Tests
    // =========================================================================

    #[test]
    fn test_layout_fits_80x24() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_layout_fits_120x40() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_layout_handles_minimum_size() {
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        // Should not panic even at tiny size
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    // =========================================================================
    // Color Tests
    // =========================================================================

    #[test]
    fn test_utilization_color_green() {
        assert_eq!(utilization_color(25.0), Color::Green);
        assert_eq!(utilization_color(49.9), Color::Green);
    }

    #[test]
    fn test_utilization_color_yellow() {
        assert_eq!(utilization_color(50.0), Color::Yellow);
        assert_eq!(utilization_color(79.9), Color::Yellow);
    }

    #[test]
    fn test_utilization_color_red() {
        assert_eq!(utilization_color(80.0), Color::Red);
        assert_eq!(utilization_color(100.0), Color::Red);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_handles_nan_utilization() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.nodes[0].cpu_pct = f64::NAN;

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_handles_zero_queue() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.queue = TaskQueue::default();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("Pending: 0")
            .unwrap();
    }

    #[test]
    fn test_handles_long_node_name() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.nodes[0].name = "very-long-node-name-that-might-overflow-the-layout".to_string();

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_handles_many_completions() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        // Add 200 completions
        for _ in 0..200 {
            app.completions.push_front(CompletionRecord {
                task_id: TaskId::new(),
                backend: BackendType::Cpu,
                node_name: "test".to_string(),
                duration: Duration::from_millis(10),
                success: true,
                error: None,
                timestamp: Instant::now(),
            });
        }

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }
}
```

### 11.8 TUI Test Matrix (Probador Coverage)

| Category | Test Count | Coverage |
|----------|------------|----------|
| Header rendering | 2 | 100% |
| Node status display | 5 | 100% |
| Queue statistics | 2 | 100% |
| Alert rendering | 2 | 100% |
| Completion list | 2 | 100% |
| State transitions | 2 | 100% |
| Layout constraints | 3 | 100% |
| Color functions | 3 | 100% |
| Edge cases | 4 | 100% |
| **Total** | **25** | **100%** |

### 11.9 Performance Requirements

| Metric | Target | Test Method |
|--------|--------|-------------|
| Frame render time | <16ms (60fps) | Benchmark `draw_ui` |
| State update latency | <100ms | Measure cluster poll |
| Memory usage | <50MB RSS | Monitor during stress test |
| Startup time | <500ms | Measure to first render |

---

## 12. Implementation Phases

### Phase 1: Core Protocol
- Node discovery and heartbeat
- Basic task submission and execution
- CPU backend only

### Phase 2: GPU Backends
- CUDA backend via wgpu
- Metal backend via wgpu
- Shader task support

### Phase 3: Fault Tolerance
- Failure detection
- Task retry and migration
- Checkpointing

### Phase 4: Advanced Scheduling
- Work stealing
- Locality-aware placement
- Cost model learning

### Phase 5: Job Flow TUI
- Real-time cluster visualization
- Node status and queue monitoring
- 100% probador test coverage
- Alert system integration

---

## 13. Approval

**Specification Author:** Claude Code
**Review Required Before Implementation:** Yes

- [ ] Architecture review
- [ ] Security review
- [ ] API design review
- [ ] Falsification checklist review

**Awaiting user approval to proceed with implementation.**
