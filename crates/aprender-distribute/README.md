<div align="center">
[![CI](https://github.com/paiml/repartir/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/repartir/actions/workflows/ci.yml)

<h1 align="center">Repartir: Sovereign AI-Grade Distributed Computing</h1>

[![CI](https://img.shields.io/badge/CI-Jidoka%20Gates-green)](.github/workflows/jidoka-gates.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-100%25-orange.svg)](https://www.rust-lang.org/)
[![Zero C/C++](https://img.shields.io/badge/C%2FC%2B%2B-0%25-blue.svg)](#)

</div>

**Repartir** is a pure Rust library for distributed execution across CPUs, GPUs, and remote machines. Built on the **Iron Lotus Framework** (Toyota Way principles for systems programming) and validated by the **certeza** testing methodology.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Feature Flags](#feature-flags)
- [Architecture](#architecture)
- [Iron Lotus Framework](#iron-lotus-framework)
- [Testing (Certeza Methodology)](#testing-certeza-methodology)
- [Sovereign AI Principles](#sovereign-ai-principles)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [Academic Foundations](#academic-foundations)
- [Comparison with Existing Systems](#comparison-with-existing-systems)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- ✅ **100% Rust, Zero C/C++**: True digital sovereignty through complete auditability
- ✅ **Memory Safety Guaranteed**: Provably safe via RustBelt formal verification
- ✅ **Work-Stealing Scheduler**: Based on Blumofe & Leiserson (1999)
- ✅ **Priority-Based Execution**: High, Normal, and Low priority queues
- ✅ **Fault Tolerance**: Task retry, timeout handling, graceful failure
- ✅ **Supply Chain Security**: Dependency pinning, binary signing, license enforcement
- ✅ **Iron Lotus Quality**: ≥95% coverage target, ≥80% mutation score, formal verification
- ✅ **Certeza Testing**: Three-tiered testing (sub-second → minutes → hours)

## Installation

```bash
# From crates.io
cargo add repartir

# From source
git clone https://github.com/paiml/repartir
cd repartir
cargo install --path .
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
repartir = "2.0.3"
tokio = { version = "1.35", features = ["rt-multi-thread", "macros"] }
```

### Basic Example

```rust
use repartir::{Pool, task::{Task, Backend}};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create a pool with 4 CPU workers
    let pool = Pool::builder()
        .cpu_workers(4)
        .build()?;

    // Submit a task
    let task = Task::builder()
        .binary("/bin/echo")
        .arg("Hello from Repartir!")
        .backend(Backend::Cpu)
        .build()?;

    let result = pool.submit(task).await?;

    if result.is_success() {
        println!("Output: {}", result.stdout_str()?.trim());
    }

    pool.shutdown().await;
    Ok(())
}
```

### Run the Example

```bash
cargo run --example hello_repartir
```

### Comprehensive v1.1 Showcase

See all v1.1 features in action:

```bash
# Generate TLS certificates first
./scripts/generate-test-certs.sh ./certs

# Run comprehensive showcase
cargo run --example v1_1_showcase --features full
```

**Demonstrates:**
- ✅ CPU executor with work-stealing (48 workers, Blumofe & Leiserson algorithm)
- ✅ GPU detection (NVIDIA RTX 4090, 2048 compute units)
- ✅ TLS encryption (certificate-based auth, TLS 1.3)
- ✅ Priority scheduling (High/Normal/Low queues)
- ✅ Parallel speedup (3.82x with 4 workers)
- ✅ Fault tolerance (graceful error handling)

### v2.0 Data Integration Features

Repartir v2.0 introduces **Parquet checkpoint storage** and **data-locality aware scheduling** for enterprise-grade distributed computing.

#### Checkpoint with Parquet Storage (v2.0)

Enable persistent checkpointing with Apache Parquet format (5-10x compression vs JSON):

```rust
use repartir::checkpoint::{CheckpointManager, CheckpointId};
use repartir::task::TaskState;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create checkpoint manager with Parquet backend
    let manager = CheckpointManager::new("./checkpoints")?;

    // Save checkpoint (automatically uses Parquet format)
    let checkpoint_id = CheckpointId::new();
    let state = TaskState::new(vec![1, 2, 3, 4]);
    manager.save(checkpoint_id, &state).await?;

    // Restore checkpoint (supports both Parquet and legacy JSON)
    let restored = manager.restore(checkpoint_id).await?;

    println!("Restored iteration: {}", restored.iteration);
    Ok(())
}
```

**Storage efficiency:**
- **SNAPPY compression**: 5-10x smaller checkpoint files
- **Columnar format**: Optimized for analytical queries
- **Backward compatible**: Reads both `.parquet` and `.json` formats

**Run the example:**
```bash
cargo run --example checkpoint_example --features checkpoint
```

#### Locality-Aware Scheduling (v2.0)

Minimize network transfers by scheduling tasks on workers that already have required data:

```rust
use repartir::{Pool, task::Task, scheduler::Scheduler};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let scheduler = Scheduler::with_capacity(100);

    // Track data locations
    let worker_id = uuid::Uuid::new_v4();
    scheduler.data_tracker()
        .track_data("dataset_A", worker_id).await;
    scheduler.data_tracker()
        .track_data("dataset_B", worker_id).await;

    // Submit task with affinity (automatically prefers worker_id)
    let task = Task::builder()
        .binary("/usr/bin/process")
        .build()?;

    scheduler.submit_with_data_locality(
        task,
        &["dataset_A".to_string(), "dataset_B".to_string()]
    ).await?;

    // Check locality metrics
    let metrics = scheduler.locality_metrics().await;
    println!("Locality hit rate: {:.1}%", metrics.hit_rate() * 100.0);

    Ok(())
}
```

**Scheduling intelligence:**
- **Affinity scoring**: `(data_items_present / total_data_items)`
- **Automatic optimization**: Prefers workers with matching data
- **Real-time metrics**: Track locality hit rate (0.0 to 1.0)

**Performance benefits:**
- Reduces network I/O for data-intensive workloads
- Improves cache utilization
- Enables efficient batch processing

#### Tensor Operations with SIMD (v2.0)

High-performance tensor operations with automatic SIMD optimization:

```rust
use repartir::tensor::{TensorExecutor, Tensor};
use repartir::task::Backend;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create tensor executor with CPU backend
    let executor = TensorExecutor::builder()
        .backend(Backend::Cpu)
        .build()?;

    // SIMD-accelerated operations
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);

    // Element-wise operations
    let sum = executor.add(&a, &b).await?;
    let product = executor.mul(&a, &b).await?;

    // Dot product
    let dot = executor.dot(&a, &b).await?;
    println!("Dot product: {}", dot);

    // Scalar operations
    let scaled = executor.scalar_mul(&a, 2.5).await?;

    Ok(())
}
```

**SIMD acceleration:**
- Leverages trueno library's AVX2/AVX-512 optimizations
- 2-8x speedup vs scalar operations on modern CPUs
- Automatic backend selection based on CPU features
- f32 precision for optimal SIMD register utilization

**Operations:**
- Element-wise: `add()`, `sub()`, `mul()`, `div()`
- Dot product: `dot()`
- Scalar: `scalar_mul()`

**Run the example:**
```bash
cargo run --example tensor_example --features tensor
```

## Feature Flags

Repartir supports multiple execution backends via feature flags:

```toml
[dependencies]
# CPU only (default)
repartir = "2.0.3"

# With GPU support (v1.1+)
repartir = { version = "2.0.3", features = ["gpu"] }

# With remote execution (v1.1+)
repartir = { version = "2.0.3", features = ["remote"] }

# With TLS encryption (v1.1+)
repartir = { version = "2.0.3", features = ["remote-tls"] }

# With Parquet checkpointing (v2.0+)
repartir = { version = "2.0.3", features = ["checkpoint"] }

# With SIMD tensor operations (v2.0+)
repartir = { version = "2.0.3", features = ["tensor"] }

# All features
repartir = { version = "2.0.3", features = ["full"] }
```

### GPU Executor (v1.1+)

The GPU executor uses [wgpu](https://wgpu.rs/) for cross-platform GPU compute:

```rust
use repartir::executor::gpu::GpuExecutor;
use repartir::executor::Executor;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let executor = GpuExecutor::new().await?;
    println!("GPU: {}", executor.device_name());
    println!("Compute units: {}", executor.capacity());
    Ok(())
}
```

**Supported backends:**
- Vulkan (Linux/Windows/Android)
- Metal (macOS/iOS)
- DirectX 12 (Windows)
- WebGPU (browsers)

**Note (v1.1):** GPU detection and initialization only. Binary task execution on GPU requires compute shader compilation (v1.2+ with rust-gpu).

```bash
cargo run --example gpu_detect --features gpu
```

### TLS Encryption (v1.1+)

Secure remote execution with TLS/SSL encryption using [rustls](https://github.com/rustls/rustls):

```rust
use repartir::executor::tls::TlsConfig;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Generate test certificates:
    // ./scripts/generate-test-certs.sh ./certs

    let tls_config = TlsConfig::builder()
        .client_cert("./certs/client.pem")
        .client_key("./certs/client.key")
        .server_cert("./certs/server.pem")
        .server_key("./certs/server.key")
        .ca_cert("./certs/ca.pem")
        .build()?;

    println!("TLS enabled!");
    Ok(())
}
```

**Security features:**
- TLS 1.3 end-to-end encryption
- Certificate-based authentication
- Perfect forward secrecy
- MITM attack protection

**Generate test certificates:**
```bash
./scripts/generate-test-certs.sh ./certs
cargo run --example tls_example --features remote-tls
```

⚠️ **WARNING**: The included certificate generator creates self-signed certificates for **TESTING ONLY**. For production, use certificates from a trusted CA (Let's Encrypt, DigiCert, etc.).

### Messaging Patterns (v1.1+)

Advanced messaging for distributed coordination with PUB/SUB and PUSH/PULL patterns:

#### Publish-Subscribe (PUB/SUB)

One publisher broadcasts to multiple subscribers:

```rust
use repartir::messaging::{PubSubChannel, Message};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let channel = PubSubChannel::new();

    // Subscribe to topics
    let mut events = channel.subscribe("events").await;
    let mut alerts = channel.subscribe("alerts").await;

    // Publish messages
    channel.publish("events", Message::text("Task completed")).await?;

    // All subscribers receive broadcast
    if let Some(msg) = events.recv().await {
        println!("Event: {}", msg.as_text()?);
    }

    Ok(())
}
```

**Use cases**: Event notifications, logging, monitoring, real-time updates

#### Push-Pull (PUSH/PULL)

Work distribution with automatic load balancing:

```rust
use repartir::messaging::{PushPullChannel, Message};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let channel = PushPullChannel::new(100);

    // Producers push work
    channel.push(Message::text("Work item 1")).await?;
    channel.push(Message::text("Work item 2")).await?;

    // Consumers pull work (load balanced)
    let work = channel.pull().await;

    Ok(())
}
```

**Use cases**: Work queues, job scheduling, pipeline processing, task distribution

**Run examples:**
```bash
cargo run --example pubsub_example
cargo run --example pushpull_example
```

## Architecture

Repartir follows a clean, layered architecture with pepita providing low-level primitives:

```
┌─────────────────────────────────────────────────────────────────┐
│                         repartir                                 │
│                  (High-level Distributed API)                    │
├─────────────────────────────────────────────────────────────────┤
│  Pool │ Scheduler │ Serverless │ Checkpoint │ Messaging         │
├─────────────────────────────────────────────────────────────────┤
│                      Executor Backends                           │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │
│  │  CPU   │  │  GPU   │  │MicroVM │  │  SIMD  │  │ Remote │    │
│  │(v1.0)  │  │(v1.1+) │  │(v2.0)  │  │(v2.0)  │  │(v1.1+) │    │
│  └────────┘  └────────┘  └───┬────┘  └───┬────┘  └────────┘    │
└──────────────────────────────┼───────────┼──────────────────────┘
                               │           │
┌──────────────────────────────▼───────────▼──────────────────────┐
│                          pepita                                  │
│              (Sovereign AI Kernel Interfaces)                    │
├─────────────┬─────────────┬─────────────┬───────────────────────┤
│   vmm.rs    │  virtio.rs  │  simd.rs    │      zram.rs          │
│  (MicroVMs) │  (devices)  │  (vectors)  │   (compression)       │
├─────────────┼─────────────┼─────────────┼───────────────────────┤
│   gpu.rs    │ scheduler   │  executor   │     pool.rs           │
│  (compute)  │(work-steal) │ (backends)  │   (high-level)        │
├─────────────┴─────────────┴─────────────┴───────────────────────┤
│          io_uring │ ublk │ blk_mq │ memory (Kernel ABI)         │
└─────────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Purpose | Backend |
|--------|---------|---------|
| **`executor::cpu`** | Execute binaries on CPU threads with work-stealing | Native threads |
| **`executor::gpu`** | GPU compute detection and shader execution | wgpu (Vulkan/Metal/DX12) |
| **`executor::microvm`** | Hardware-isolated execution in KVM MicroVMs | pepita::vmm |
| **`executor::simd`** | SIMD-accelerated vector operations | pepita::simd (AVX-512/NEON) |
| **`executor::remote`** | Distributed execution over TCP | rustls TLS 1.3 |
| **`serverless`** | Function-as-a-Service with warm pools | pepita::vmm + pepita::virtio |
| **`scheduler`** | Priority-based work-stealing scheduler | Blumofe-Leiserson algorithm |
| **`checkpoint`** | Parquet-based checkpoint storage | Apache Parquet |
| **`messaging`** | PUB/SUB and PUSH/PULL patterns | tokio channels |

### Pepita Integration (v2.0+)

Repartir v2.0 integrates with [pepita](../pepita) for hardware-accelerated execution:

#### SIMD Executor

Execute vectorized operations using AVX-512/AVX2/SSE/NEON:

```rust
use repartir::executor::simd::{SimdExecutor, SimdTask};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let executor = SimdExecutor::new();

    // Check capabilities
    println!("SIMD: {}-bit vectors", executor.vector_width());

    // Vector addition (SIMD accelerated)
    let a = vec![1.0f32; 10000];
    let b = vec![2.0f32; 10000];
    let task = SimdTask::vadd_f32(a, b);
    let result = executor.execute_simd(task).await?;

    println!("Throughput: {:.2}M elem/s", result.throughput() / 1_000_000.0);
    Ok(())
}
```

**Supported operations**: `vadd_f32`, `vadd_f64`, `vmul_f32`, `dot_f32`, `matmul_f32`

#### MicroVM Executor

Hardware-isolated execution with sub-100ms cold start:

```rust
use repartir::executor::microvm::{MicroVmExecutor, MicroVmExecutorConfig};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let config = MicroVmExecutorConfig::builder()
        .memory_mib(256)
        .vcpus(2)
        .warm_pool(true, 3)  // Keep 3 VMs warm
        .build()?;

    let executor = MicroVmExecutor::new(config)?;

    // VMs from warm pool start in <5ms
    // Cold start is <100ms
    Ok(())
}
```

**Features**: KVM isolation, warm pools, jailer security, virtio devices

#### Serverless Functions

Function-as-a-Service with automatic warm pool management:

```rust
use repartir::serverless::{Function, FunctionService, Runtime, Trigger, HttpMethod};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let mut service = FunctionService::new();

    let function = Function::builder()
        .name("process-data")
        .runtime(Runtime::RustNative { binary: PathBuf::from("./target/release/worker") })
        .memory_mib(256)
        .trigger(Trigger::Http {
            path: "/api/process".to_string(),
            methods: vec![HttpMethod::Post],
        })
        .build()?;

    service.register(function)?;

    // Invoke function
    let request = InvocationRequest::new("process-data", b"input data");
    let response = service.invoke(request).await?;

    Ok(())
}
```

**Run examples:**
```bash
cargo run --example simd_example --features simd
cargo run --example microvm_example --features microvm
cargo run --example serverless_example --features serverless
```

## Iron Lotus Framework

Repartir embodies Toyota Production System principles:

### Genchi Genbutsu (現地現物 - "Go and See")
- **Radical Transparency**: Every operation traceable from API → scheduler → executor
- **No Black Boxes**: 100% pure Rust, zero opaque C/C++ libraries
- **AST-Level Inspection**: Code structure visible via pmat

### Jidoka (自働化 - "Automation with Human Touch")
- **Automated Quality Gates**: CI enforces clippy, rustfmt, tests, coverage
- **Andon Cord**: Build fails immediately on any defect
- **No Manual Checks**: Machines verify before humans review

### Kaizen (改善 - "Continuous Improvement")
- **Technical Debt Grading**: TDG score must never decrease
- **Ratchet Effect**: Each PR improves or maintains quality
- **Five Whys**: Root cause analysis for all incidents

### Muda (無駄 - "Waste Elimination")
- **No Overproduction**: Zero YAGNI features
- **No Waiting**: Fast compilation with sccache
- **No Transportation**: Zero-copy data flow, single language
- **No Defects**: EXTREME TDD with mutation testing

## Testing (Certeza Methodology)

Repartir uses a three-tiered testing approach:

### Tier 1: ON-SAVE (Sub-Second)
Fast feedback for flow state:
```bash
make tier1
```
- Unit tests (84 tests)
- `cargo check`
- `cargo clippy`
- `cargo fmt`

**Target**: < 3 seconds

### Tier 2: ON-COMMIT (1-5 Minutes)
Comprehensive pre-commit gate:
```bash
make tier2
```
- All tests (84 tests)
- Property-based tests (proptest)
- Coverage analysis (target ≥95%)
- Documentation tests
- Security audit (cargo-audit, cargo-deny)

**Target**: 1-5 minutes

### Tier 3: ON-MERGE (Hours)
Exhaustive validation:
```bash
make tier3
```
- Mutation testing (cargo-mutants, target ≥80%)
- Formal verification (Kani, for critical paths)
- Extended fuzzing
- Performance benchmarks

**Target**: 1-6 hours (run overnight or in CI)

## Test Results (v2.0.3)

```
✓ 84 tests PASSED (0 warnings)
```

### Recent Fixes (v2.0.3)
- macOS test compatibility: `/bin/true` replaced with `/bin/sh -c true` for cross-platform test execution

## Sovereign AI Principles

### Digital Sovereignty Requirements
1. **Auditability**: Trace execution from user API → hardware instruction
2. **Supply Chain Independence**: Deterministic rebuild from source
3. **No Foreign Dependencies**: Zero opaque binary blobs
4. **Memory Safety Guarantees**: Provable absence of vulnerabilities

### Supply Chain Security
- **Dependency Pinning**: `Cargo.lock` committed and reviewed
- **License Enforcement**: Only MIT/Apache-2.0/BSD allowed (via cargo-deny)
- **Binary Signing**: ed25519 signatures for distributed binaries
- **Audit Trail**: `cargo tree` logged in CI

### Memory Safety (NSA/CISA Mandate)
Per NSA/CISA joint guidance on memory-safe languages:
- ✅ Rust provides compile-time memory safety guarantees
- ✅ RustBelt formal verification proves soundness
- ✅ `#![deny(unsafe_code)]` in v1.0 (no unsafe code)
- ✅ Eliminates buffer overflows, use-after-free, data races

## Roadmap

### v1.0: Sovereign Foundation (Current)
- ✅ CPU executor with work-stealing scheduler
- ✅ Priority-based task scheduling
- ✅ High-level Pool API
- ✅ Comprehensive testing (84 tests)
- ✅ Iron Lotus quality gates
- ✅ Supply chain security

### v1.1: Production Hardening (Complete)
- ✅ GPU executor skeleton (wgpu detection, v1.2 for rust-gpu compute)
- ✅ Remote executor (TCP transport, length-prefixed bincode protocol)
- ✅ TLS encryption (rustls, certificate-based auth, TLS 1.3)
- ✅ Performance benchmarks vs Ray/Dask (5 benchmark suites)
- ✅ Mutation testing ≥85% (framework + documentation)
- ✅ Comprehensive Makefile (tier1/tier2/tier3, coverage enforcement)
- ✅ bashrs purification (POSIX-compliant shell code)
- ✅ Advanced messaging patterns (PUB/SUB, PUSH/PULL)

### v2.0: Data Integration (Complete)
- ✅ **Parquet Checkpoint Storage** (Phase 1): SNAPPY compression, 5-10x size reduction
- ✅ **Data-Locality Tracking** (Phase 1): DataLocationTracker with batch affinity queries
- ✅ **Affinity-Based Scheduling** (Phase 2): Automatic locality-aware task assignment
- ✅ **Locality Metrics** (Phase 2): Real-time hit rate tracking (0.0 to 1.0)
- ✅ **Tensor Operations** (Phase 3): SIMD-accelerated operations via trueno (2-8x speedup)
- ✅ **Performance Benchmarks** (Phase 2): Locality scheduling validated (<1ms overhead)
- [ ] Advanced ML patterns (pipeline/tensor parallelism) - Phase 4

### v3.0: Enterprise & Cloud
- [ ] RDMA support (low-latency networking)
- [ ] Multi-tenant isolation
- [ ] Kubernetes operator
- [ ] FIPS 140-2 compliance mode

## Contributing

Contributions welcome! Please ensure:

1. All tests pass: `make tier2`
2. Coverage ≥95%: `make coverage` (when configured)
3. Clippy passes: `cargo clippy -- -D warnings`
4. Code formatted: `cargo fmt`
5. No SATD comments (TODO without ticket number)

See [Iron Lotus Code Review Framework](docs/specifications/repartir-distributed-cpu-gpu-data-hpc-spec.md#12-the-iron-lotus-code-review-framework) for detailed guidelines.

## Documentation

- [Full Specification](docs/specifications/repartir-distributed-cpu-gpu-data-hpc-spec.md) (1,500 lines, 10+ academic citations)
- [API Documentation](https://docs.rs/repartir) (run `cargo doc --open`)
- [Examples](examples/) - Hands-on demonstrations

## Academic Foundations

Repartir is grounded in peer-reviewed research:

1. **Jung et al. (2017)** - RustBelt: Formal verification of Rust's safety
2. **Blumofe & Leiserson (1999)** - Provably optimal work-stealing
3. **Chandra & Toueg (1996)** - Unreliable failure detectors
4. **NSA/CISA (2023)** - Memory-safe languages guidance
5. **Pereira et al. (2017)** - Energy efficiency of Rust vs C++

See [specification](docs/specifications/repartir-distributed-cpu-gpu-data-hpc-spec.md#12-academic-foundations) for complete citations.

## Comparison with Existing Systems

| Feature                | Repartir (v2.0) | Ray       | Dask      |
|------------------------|-----------------|-----------|-----------|
| Language               | Rust            | Python    | Python    |
| C Dependencies         | Zero*           | Many      | Some      |
| GPU Support            | Yes (wgpu)      | Limited   | No        |
| Work Stealing          | Yes             | No        | Yes       |
| Fault Tolerance        | Yes             | Yes       | Limited   |
| Memory Safety          | Guaranteed      | Runtime   | Runtime   |
| Binary Execution       | Yes             | No        | No        |
| Remote Execution       | Yes (TCP)       | Yes       | Yes       |

*Note: rustls (used for TLS) currently depends on aws-lc-rs (C). Pure Rust alternatives under evaluation for v1.2+.

## See Also

- [Cookbook](examples/) — 9 runnable examples

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Iron Lotus Framework**: Toyota Production System for systems programming
- **Certeza Project**: Asymptotic test effectiveness methodology
- **PAIML Stack**: trueno, aprender, paiml-mcp-agent-toolkit, bashrs
- **Rust Community**: rust-gpu, wgpu, tokio, and the broader ecosystem

---

**Built with the Iron Lotus Framework**
*Quality is not inspected in; it is built in.*

