# Repartir v2.0: Data Integration Specification

**Version**: 2.0.0-draft
**Date**: 2025-11-22
**Status**: Design Phase

## Executive Summary

Repartir v2.0 extends the distributed computing framework with persistent state management, data-locality aware scheduling, and advanced ML patterns. This version integrates with trueno-db for high-performance checkpointing and trueno for SIMD-accelerated tensor operations.

## 1. Architecture Overview

### 1.1 New Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Repartir v2.0                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  High-Level API (Pool + State Management)             │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Scheduler (Work-Stealing + Data Locality)            │ │
│  │   - Task queue with priority                           │ │
│  │   - Data locality hints                                │ │
│  │   - Checkpoint coordination                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  State Layer (NEW)                                     │ │
│  │   ┌──────────────┐  ┌──────────────┐                  │ │
│  │   │ Checkpoint   │  │  Data Cache  │                  │ │
│  │   │  Manager     │  │  (trueno-db) │                  │ │
│  │   └──────────────┘  └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Executor Backends                                     │ │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐      │ │
│  │  │  CPU   │  │  GPU   │  │ Remote │  │ Tensor │      │ │
│  │  │(Native)│  │(wgpu)  │  │ (TCP)  │  │(trueno)│      │ │
│  │  └────────┘  └────────┘  └────────┘  └────────┘      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
    ┌─────────┐      ┌───────────┐      ┌─────────┐
    │ trueno  │      │ trueno-db │      │  tokio  │
    │ (SIMD)  │      │ (Storage) │      │ (async) │
    └─────────┘      └───────────┘      └─────────┘
```

### 1.2 Integration Points

**trueno-db Integration (v0.3.0):**
- Checkpoint storage (Arrow/Parquet format)
- Task result caching
- Data-locality metadata
- SQL query interface for state inspection

**trueno Integration (v0.6.0):**
- SIMD tensor operations
- GPU tensor operations (when available)
- Zero-copy data sharing

## 2. Feature Specifications

### 2.1 Checkpointing to Persistent Storage

**Goal**: Enable task restart from last checkpoint on failure.

**Design**:
```rust
use trueno_db::storage::StorageEngine;
use trueno_db::query::QueryEngine;

pub struct CheckpointManager {
    storage: StorageEngine,
    checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    /// Create a new checkpoint for a task
    pub async fn checkpoint(
        &self,
        task_id: Uuid,
        state: &TaskState,
    ) -> Result<CheckpointId>;

    /// Restore task from last checkpoint
    pub async fn restore(
        &self,
        task_id: Uuid,
    ) -> Result<Option<TaskState>>;

    /// List all checkpoints for a task
    pub async fn list_checkpoints(
        &self,
        task_id: Uuid,
    ) -> Result<Vec<CheckpointMetadata>>;

    /// Delete old checkpoints (retention policy)
    pub async fn cleanup(
        &self,
        retention_days: u32,
    ) -> Result<usize>;
}

/// Task state that can be checkpointed
#[derive(Serialize, Deserialize)]
pub struct TaskState {
    pub task_id: Uuid,
    pub iteration: u64,
    pub data: Vec<u8>,  // Serialized application state
    pub timestamp: SystemTime,
}
```

**Storage Format**:
- Use trueno-db's Parquet backend for efficient columnar storage
- Schema: `[task_id: Uuid, checkpoint_id: Uuid, iteration: u64, timestamp: Timestamp, data: Binary]`
- Indexed by task_id for fast lookups

**Fault Tolerance Flow**:
1. Task opts-in to checkpointing via builder: `.checkpoint_interval(Duration::from_secs(60))`
2. Executor periodically calls checkpoint callback
3. On task failure, scheduler queries for last checkpoint
4. Task resumes from checkpoint state

### 2.2 Distributed State Management

**Goal**: Coordinate state across distributed workers.

**Design**:
```rust
pub struct StateStore {
    local_db: trueno_db::Database,
    remote_workers: Vec<WorkerInfo>,
}

impl StateStore {
    /// Store key-value configuration
    pub async fn put(
        &self,
        key: &str,
        value: &[u8],
    ) -> Result<()>;

    /// Retrieve key-value configuration
    pub async fn get(
        &self,
        key: &str,
    ) -> Result<Option<Vec<u8>>>;

    /// Store task result for data-locality
    pub async fn cache_result(
        &self,
        task_id: Uuid,
        result: &ExecutionResult,
    ) -> Result<()>;

    /// Query which worker has cached data
    pub async fn locate_data(
        &self,
        data_key: &str,
    ) -> Result<Vec<WorkerId>>;
}
```

**Implementation**:
- Phase 1 (v2.0): Local trueno-db for single-node state
- Phase 2 (v2.1): Add gRPC replication when trueno-db Phase 3 ships
- Phase 3 (v2.2): Full distributed KV store with consensus

### 2.3 Data-Locality Aware Scheduling

**Goal**: Prefer scheduling tasks on workers that already have required data.

**Design**:
```rust
pub struct LocalityScheduler {
    scheduler: Scheduler,
    state_store: StateStore,
}

impl LocalityScheduler {
    /// Submit task with data locality hints
    pub async fn submit_with_locality(
        &self,
        task: Task,
        data_keys: Vec<String>,
    ) -> Result<Uuid> {
        // Query which workers have data
        let worker_locations = self.state_store
            .locate_data_batch(&data_keys)
            .await?;

        // Assign affinity to workers with data
        let affinity = self.calculate_affinity(worker_locations);

        self.scheduler.submit_with_affinity(task, affinity).await
    }

    /// Calculate worker affinity score
    fn calculate_affinity(
        &self,
        locations: HashMap<WorkerId, usize>,
    ) -> HashMap<WorkerId, f64> {
        // Score = data_items_present / total_data_items
        // Workers with more data get higher priority
        locations.into_iter()
            .map(|(worker, count)| (worker, count as f64))
            .collect()
    }
}
```

**Scheduling Algorithm**:
1. Task specifies required data keys (e.g., tensor IDs, checkpoint IDs)
2. Scheduler queries StateStore for data locations
3. Calculate affinity score: `score = local_data_items / total_data_items`
4. Prefer workers with higher affinity scores
5. Fallback to standard work-stealing if no locality match

**Metrics**:
- Track locality hit rate: `tasks_with_local_data / total_tasks`
- Measure network transfer savings

### 2.4 Advanced ML Patterns

**Goal**: Support pipeline and tensor parallelism for distributed training.

#### 2.4.1 Pipeline Parallelism

```rust
pub struct PipelineStage {
    stage_id: usize,
    model_shard: Vec<u8>,  // Serialized model layers
    input_device: DeviceId,
    output_device: DeviceId,
}

pub struct PipelineExecutor {
    stages: Vec<PipelineStage>,
    micro_batch_size: usize,
}

impl PipelineExecutor {
    /// Execute pipeline across multiple workers
    pub async fn execute_pipeline(
        &self,
        input: Tensor,
    ) -> Result<Tensor> {
        let micro_batches = self.split_micro_batches(input);

        // Schedule stages across workers
        for (i, stage) in self.stages.iter().enumerate() {
            let task = Task::builder()
                .binary("model_stage")
                .arg(format!("--stage={}", i))
                .data(stage.model_shard.clone())
                .backend(Backend::Gpu)
                .build()?;

            // Overlap computation and communication
            // Stage N processes micro-batch K while Stage N+1 processes K-1
        }

        Ok(output)
    }
}
```

**Key Features**:
- Split model across N workers (layers → devices)
- Micro-batching for pipeline bubbles reduction
- Overlap computation and communication
- Integration with trueno for tensor ops

#### 2.4.2 Tensor Parallelism

```rust
use trueno::Tensor;

pub struct TensorParallelExecutor {
    devices: Vec<DeviceId>,
    shard_strategy: ShardStrategy,
}

#[derive(Clone, Copy)]
pub enum ShardStrategy {
    RowShard,     // Shard along rows
    ColumnShard,  // Shard along columns
    Custom,       // User-defined sharding
}

impl TensorParallelExecutor {
    /// Distribute tensor operation across devices
    pub async fn parallel_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor> {
        // Shard tensors across devices
        let a_shards = self.shard_tensor(a, self.devices.len())?;
        let b_shards = self.shard_tensor(b, self.devices.len())?;

        // Execute on each device
        let mut tasks = Vec::new();
        for (device_id, (a_shard, b_shard)) in
            self.devices.iter().zip(a_shards.iter().zip(&b_shards))
        {
            let task = self.create_matmul_task(device_id, a_shard, b_shard);
            tasks.push(task);
        }

        // All-reduce to combine results
        let results = self.execute_all(tasks).await?;
        self.all_reduce(results)
    }

    /// All-reduce operation using ring topology
    async fn all_reduce(
        &self,
        partial_results: Vec<Tensor>,
    ) -> Result<Tensor> {
        // Ring all-reduce algorithm
        // Minimizes communication: O(N * D / P) where D = data size, P = devices
        todo!("Implement ring all-reduce")
    }
}
```

**Communication Patterns**:
- All-reduce: Sum gradients across all workers
- Scatter-gather: Distribute data, collect results
- Ring topology: Minimize communication overhead

## 3. API Design

### 3.1 Opt-In Checkpointing

```rust
use repartir::{Pool, Task, Backend};
use repartir::checkpoint::CheckpointConfig;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create pool with checkpointing enabled
    let pool = Pool::builder()
        .cpu_workers(4)
        .checkpoint_dir("./checkpoints")
        .checkpoint_interval(Duration::from_secs(60))
        .build()?;

    // Task with checkpointing
    let task = Task::builder()
        .binary("/bin/my_long_task")
        .arg("--data=large_dataset")
        .backend(Backend::Cpu)
        .enable_checkpointing(true)  // Opt-in
        .checkpoint_callback(Box::new(|state: &mut TaskState| {
            // Custom checkpoint logic
            state.data = serialize_state()?;
            Ok(())
        }))
        .build()?;

    let result = pool.submit(task).await?;
    pool.shutdown().await;
    Ok(())
}
```

### 3.2 Data-Locality Scheduling

```rust
use repartir::{Pool, Task, Backend};
use repartir::scheduling::DataLocality;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let pool = Pool::builder()
        .cpu_workers(4)
        .remote_workers(vec!["worker1:9000", "worker2:9000"])
        .enable_data_locality(true)
        .build()?;

    // Task with data locality hints
    let task = Task::builder()
        .binary("/bin/process_data")
        .backend(Backend::Cpu)
        .data_dependencies(vec![
            "tensor_batch_42",
            "checkpoint_123",
        ])
        .build()?;

    // Scheduler automatically prefers workers with cached data
    let result = pool.submit(task).await?;

    pool.shutdown().await;
    Ok(())
}
```

### 3.3 Tensor Operations

```rust
use repartir::tensor::{TensorExecutor, Tensor};
use repartir::Backend;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let executor = TensorExecutor::new()
        .backend(Backend::Gpu)
        .build()?;

    // SIMD tensor operations via trueno
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);

    let result = executor.add(&a, &b).await?;

    println!("Result: {:?}", result.to_vec());
    Ok(())
}
```

## 4. Implementation Phases

### Phase 1: Foundation (v2.0.0) - 2 weeks
- [ ] Add trueno-db dependency
- [ ] Implement CheckpointManager
- [ ] Add checkpoint storage to Parquet
- [ ] Update Task builder with checkpoint options
- [ ] Add checkpoint tests
- [ ] Basic data-locality metadata tracking

### Phase 2: Locality Scheduling (v2.1.0) - 2 weeks
- [ ] Implement LocalityScheduler
- [ ] Add affinity-based task assignment
- [ ] Track data location metadata
- [ ] Add locality hit rate metrics
- [ ] Performance benchmarks

### Phase 3: Tensor Operations (v2.2.0) - 2 weeks
- [ ] Add trueno dependency
- [ ] Implement TensorExecutor wrapper
- [ ] SIMD tensor operations
- [ ] GPU tensor operations (optional)
- [ ] Tensor parallelism primitives

### Phase 4: ML Patterns (v2.3.0) - 3 weeks
- [ ] Pipeline parallelism executor
- [ ] Tensor parallelism executor
- [ ] All-reduce implementation
- [ ] Ring topology communication
- [ ] ML benchmarks (vs PyTorch DDP)

## 5. Performance Targets

**Checkpointing**:
- Checkpoint latency: <100ms for 10MB state
- Storage overhead: <10% of task execution time
- Parquet compression: 5-10x compared to raw binary

**Data Locality**:
- Locality hit rate: >80% for iterative workloads
- Network transfer reduction: 3-5x compared to random assignment
- Scheduling overhead: <1ms per task

**Tensor Operations**:
- SIMD speedup: 2-5x vs scalar (trueno benchmarks)
- GPU speedup: 10-50x for large tensors
- Zero-copy sharing: Avoid serialization overhead

## 6. Quality Gates

All v2.0 code must meet:
- **Test Coverage**: ≥90% (maintain v1.0 standards)
- **Integration Tests**: End-to-end checkpoint/restore flows
- **Property Tests**: Checkpoint consistency, data locality correctness
- **Benchmarks**: Compare checkpointing vs no-checkpoint overhead
- **Documentation**: API docs, examples, architecture diagrams

## 7. Dependencies

**New Dependencies**:
```toml
[dependencies]
# Data integration (v2.0)
trueno-db = { version = "0.3", features = ["distributed"] }
trueno = { version = "0.6", features = ["gpu", "parallel"] }

# Already have
tokio = "1.35"
serde = "1.0"
bincode = "1.3"
```

**Feature Flags**:
```toml
[features]
default = ["cpu"]
cpu = ["tokio", "num_cpus"]
gpu = ["wgpu", "pollster"]
remote = ["tokio", "bincode"]
remote-tls = ["remote", "rustls", "tokio-rustls"]
checkpoint = ["trueno-db"]  # NEW
tensor = ["trueno"]          # NEW
full = ["cpu", "gpu", "remote", "remote-tls", "checkpoint", "tensor"]
```

## 8. Backwards Compatibility

**v1.0 API Preserved**:
- All v1.0 APIs remain unchanged
- New features are opt-in via builders
- Checkpointing disabled by default
- Data locality disabled by default

**Migration Path**:
```rust
// v1.0 code still works
let pool = Pool::builder().cpu_workers(4).build()?;

// v2.0 features opt-in
let pool = Pool::builder()
    .cpu_workers(4)
    .checkpoint_dir("./checkpoints")  // NEW
    .enable_data_locality(true)       // NEW
    .build()?;
```

## 9. Open Questions

1. **trueno-db Distribution**: Phase 3 not released yet. How to handle distributed checkpoints?
   - **Answer**: Start with local-only, prepare architecture for distributed extension

2. **Checkpoint Serialization**: Binary vs structured format?
   - **Answer**: Use Parquet (structured) for queryability and compression

3. **Data Locality Granularity**: Track at file level, tensor level, or block level?
   - **Answer**: Start with task-level (coarse), refine to tensor-level in v2.1

4. **All-Reduce Algorithm**: Ring, tree, or butterfly topology?
   - **Answer**: Ring all-reduce (proven in Horovod, PyTorch DDP)

## 10. Success Criteria

v2.0 is successful if:
- ✅ Checkpointing reduces wasted work on failures by >90%
- ✅ Data locality improves network efficiency by >3x
- ✅ Tensor operations match trueno benchmarks (2-5x SIMD speedup)
- ✅ Test coverage ≥90%
- ✅ Zero regressions in v1.0 functionality

## 11. References

1. **trueno-db**: https://crates.io/crates/trueno-db
2. **trueno**: https://crates.io/crates/trueno
3. **Horovod**: Ring All-Reduce algorithm
4. **PyTorch DDP**: Distributed Data Parallel patterns
5. **GPipe**: Pipeline parallelism (Huang et al., 2019)
6. **Megatron-LM**: Tensor parallelism (Shoeybi et al., 2019)

---

**Next Steps**: Begin Phase 1 implementation with CheckpointManager.
