# Repartir: Distributed Computing

**repartir** is the Sovereign AI Stack's distributed computing library, providing CPU, GPU, and remote task execution with work-stealing scheduling.

## Overview

| Attribute | Value |
|-----------|-------|
| Version | 1.1.x |
| crates.io | [repartir](https://crates.io/crates/repartir) |
| docs.rs | [repartir](https://docs.rs/repartir) |
| License | MIT |

## Key Features

- **100% Rust, Zero C/C++**: Complete auditability for sovereign AI
- **Work-Stealing Scheduler**: Based on Blumofe & Leiserson (1999)
- **Multi-Backend Execution**: CPU, GPU, and Remote executors
- **Iron Lotus Quality**: 95% coverage, 80% mutation score

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    repartir Pool                            │
├─────────────────────────────────────────────────────────────┤
│                      Scheduler                              │
│              (Work-Stealing, Task Queue)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CpuExecutor │  │ GpuExecutor │  │   RemoteExecutor    │  │
│  │             │  │             │  │                     │  │
│  │  Rayon-like │  │    wgpu     │  │   TCP/TLS           │  │
│  │  AVX2/512   │  │ Vulkan/Metal│  │  Multi-Node         │  │
│  │    NEON     │  │ DX12/WebGPU │  │  Distributed        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | Local multi-core execution with work-stealing |
| `gpu` | wgpu GPU compute (Vulkan/Metal/DX12/WebGPU) |
| `remote` | TCP-based distributed execution |
| `remote-tls` | TLS-secured remote execution |
| `tensor` | trueno SIMD tensor integration |
| `checkpoint` | trueno-db + Parquet state persistence |
| `tui` | Job flow TUI visualization |
| `full` | All features enabled |

## Quick Start

### Installation

```toml
[dependencies]
repartir = { version = "1.1", features = ["cpu"] }

# With GPU support
repartir = { version = "1.1", features = ["cpu", "gpu"] }

# Full distributed with all features
repartir = { version = "1.1", features = ["full"] }
```

### Basic CPU Pool

```rust
use repartir::{Pool, task::{Task, Backend}};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create pool with 8 CPU workers
    let pool = Pool::builder()
        .cpu_workers(8)
        .build()?;

    // Submit a task
    let task = Task::builder()
        .binary("./worker")
        .arg("--input").arg("data.csv")
        .backend(Backend::Cpu)
        .build()?;

    let result = pool.submit(task).await?;

    if result.is_success() {
        println!("Output: {}", result.stdout_str()?);
    }

    pool.shutdown().await;
    Ok(())
}
```

### GPU Execution

```rust
use repartir::executor::gpu::GpuExecutor;
use repartir::executor::Executor;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Initialize GPU executor (auto-selects best GPU)
    let gpu = GpuExecutor::new().await?;

    println!("GPU: {}", gpu.device_name());
    println!("Compute units: {}", gpu.capacity());

    // GPU selection priority:
    // 1. Discrete GPU (dedicated graphics)
    // 2. Integrated GPU (CPU-integrated)
    // 3. Software rasterizer (fallback)

    Ok(())
}
```

### Multi-Machine Distribution

**Step 1: Start workers on each node**

```bash
# On node1 (192.168.1.10)
repartir-worker --bind 0.0.0.0:9000

# On node2 (192.168.1.11)
repartir-worker --bind 0.0.0.0:9000

# On node3 (192.168.1.12)
repartir-worker --bind 0.0.0.0:9000
```

**Step 2: Connect from coordinator**

```rust
use repartir::executor::remote::RemoteExecutor;
use repartir::task::{Task, Backend};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Connect to remote workers
    let executor = RemoteExecutor::builder()
        .add_worker("192.168.1.10:9000")
        .add_worker("192.168.1.11:9000")
        .add_worker("192.168.1.12:9000")
        .build()
        .await?;

    // Task distributed to available worker
    let task = Task::builder()
        .binary("./gpu-workload")
        .arg("--shard=0")
        .backend(Backend::Gpu)
        .build()?;

    let result = executor.execute(task).await?;
    println!("Result: {:?}", result.stdout_str()?);

    Ok(())
}
```

### TLS-Secured Remote Execution

```rust
use repartir::executor::tls::TlsRemoteExecutor;

let executor = TlsRemoteExecutor::builder()
    .add_worker("node1.internal:9443")
    .cert_path("./certs/client.pem")
    .key_path("./certs/client.key")
    .ca_path("./certs/ca.pem")
    .build()
    .await?;
```

## SIMD Tensor Operations

With the `tensor` feature, repartir integrates with trueno for SIMD-accelerated operations:

```rust
use repartir::tensor::{TensorExecutor, Tensor};
use repartir::task::Backend;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let executor = TensorExecutor::builder()
        .backend(Backend::Cpu)  // Uses AVX2/AVX-512/NEON
        .build()?;

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);

    // SIMD-accelerated operations
    let sum = executor.add(&a, &b).await?;
    let product = executor.mul(&a, &b).await?;
    let dot = executor.dot(&a, &b).await?;

    println!("Sum: {:?}", sum.as_slice());
    println!("Product: {:?}", product.as_slice());
    println!("Dot product: {}", dot);

    Ok(())
}
```

## Checkpointing

With the `checkpoint` feature, repartir can persist state using trueno-db and Parquet:

```rust
use repartir::checkpoint::CheckpointManager;

let checkpoint = CheckpointManager::new("./checkpoints")?;

// Save state
checkpoint.save("training_epoch_10", &model_state).await?;

// Restore on failure
let state = checkpoint.load("training_epoch_10").await?;
```

## Job Flow TUI

Monitor distributed jobs with the TUI dashboard:

```bash
cargo run --bin job-flow --features tui,remote
```

```text
┌─ Job Flow Monitor ─────────────────────────────────────────┐
│ Workers: 3 active   │  Tasks: 45 pending / 120 completed   │
├─────────────────────┴──────────────────────────────────────┤
│ Node                 │ Status  │ Load │ Tasks │ Uptime     │
├──────────────────────┼─────────┼──────┼───────┼────────────┤
│ 192.168.1.10:9000    │ Active  │ 78%  │ 15    │ 2h 34m     │
│ 192.168.1.11:9000    │ Active  │ 65%  │ 18    │ 2h 34m     │
│ 192.168.1.12:9000    │ Active  │ 82%  │ 12    │ 2h 30m     │
└──────────────────────┴─────────┴──────┴───────┴────────────┘
```

## Integration with Batuta

Batuta uses repartir for distributed orchestration:

```rust
use batuta::backend::{select_backend, to_repartir_backend};
use batuta::oracle::types::HardwareSpec;

// MoE router selects optimal backend
let backend = select_backend(
    OpComplexity::High,
    Some(DataSize::samples(1_000_000)),
    &HardwareSpec {
        has_gpu: true,
        is_distributed: true,
        node_count: Some(4),
        ..Default::default()
    },
);

// Map to repartir backend
let repartir_backend = to_repartir_backend(backend);
```

## Backend Selection Criteria

Batuta's MoE router uses the 5x PCIe rule (Gregg & Hazelwood, 2011):

| Complexity | Scalar | SIMD | GPU |
|------------|--------|------|-----|
| Low (O(n)) | <1M | >1M | Never |
| Medium (O(n log n)) | <10K | 10K-100K | >100K |
| High (O(n³)) | <1K | 1K-10K | >10K |

**GPU is beneficial when:** `compute_time > 5 × transfer_time`

## Performance Considerations

### Work-Stealing Efficiency

The Blumofe & Leiserson work-stealing algorithm provides:
- **O(T₁/P + T∞)** expected time with P processors
- **Near-linear speedup** for embarrassingly parallel workloads
- **Low contention** through randomized stealing

### GPU vs CPU Decision

```rust
// Automatic backend selection
let backend = if data_size > 100_000 && complexity == High {
    Backend::Gpu
} else if data_size > 1_000 {
    Backend::Cpu  // SIMD-accelerated
} else {
    Backend::Cpu  // Scalar
};
```

### Remote Execution Overhead

- **Serialization**: bincode (fast, compact)
- **Network**: Length-prefixed TCP messages
- **Latency**: ~1ms per task submission (local network)

## Comparison with Alternatives

| Feature | repartir | Rayon | tokio | Ray |
|---------|----------|-------|-------|-----|
| Language | Rust | Rust | Rust | Python |
| GPU Support | Yes (wgpu) | No | No | Yes |
| Distributed | Yes | No | No | Yes |
| Work-Stealing | Yes | Yes | No | Yes |
| TLS | Yes | N/A | Yes | Yes |
| Pure Rust | Yes | Yes | Yes | No |

## Example: Distributed ML Training

```rust
use repartir::executor::remote::RemoteExecutor;
use repartir::task::{Task, Backend};

async fn distributed_training(
    nodes: &[&str],
    epochs: usize,
) -> repartir::error::Result<()> {
    let executor = RemoteExecutor::builder()
        .add_workers(nodes)
        .build()
        .await?;

    for epoch in 0..epochs {
        // Distribute training shards
        let tasks: Vec<_> = (0..nodes.len())
            .map(|shard| {
                Task::builder()
                    .binary("./train")
                    .arg("--epoch").arg(epoch.to_string())
                    .arg("--shard").arg(shard.to_string())
                    .arg("--total-shards").arg(nodes.len().to_string())
                    .backend(Backend::Gpu)
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Execute in parallel
        for task in tasks {
            let result = executor.execute(task).await?;
            println!("Shard completed: {:?}", result.exit_code());
        }

        println!("Epoch {} complete", epoch);
    }

    Ok(())
}
```

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Trueno](./trueno.md) | [Aprender](./aprender.md)
