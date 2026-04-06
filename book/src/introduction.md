# Introduction

Welcome to **Repartir**, a sovereign AI-grade distributed computing library for Rust that enables seamless execution of tasks across heterogeneous compute resources including CPUs, GPUs, and remote machines.

## What is Repartir?

Repartir is a **pure Rust** distributed computing framework that provides:

- **CPU Execution**: Async process execution with work-stealing scheduler
- **GPU Compute**: WGSL shader execution via wgpu (cross-platform)
- **Remote Execution**: TCP-based task distribution with optional TLS
- **Message-Oriented**: PubSub and PushPull patterns for coordination
- **Fault Tolerant**: Retry logic, heartbeat monitoring, task migration

## Why Repartir?

### 🦀 100% Pure Rust

Unlike Ray (Python + C++) or Dask (Python), Repartir is built entirely in Rust with **zero C/C++ dependencies**. This provides:

- **Memory Safety**: Provable absence of buffer overflows and use-after-free bugs
- **Digital Sovereignty**: Complete auditability from source code to execution
- **Supply Chain Security**: No opaque binary blobs or foreign dependencies
- **Performance**: Native code performance without FFI overhead

### 🌐 Heterogeneous Execution

Repartir treats CPUs, GPUs, and remote workers as first-class execution backends:

```rust
use repartir::{Pool, Task, Backend};

let pool = Pool::builder()
    .cpu_workers(8)
    .gpu_workers(2)
    .remote_workers(vec!["worker1:9000", "worker2:9000"])
    .build()
    .await?;

// CPU task
let cpu_task = Task::builder()
    .binary("./compute")
    .args(vec!["--input", "data.csv"])
    .backend(Backend::Cpu)
    .build()?;

// GPU shader task
let shader = r#"
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        output[global_id.x] = input[global_id.x] * 2.0;
    }
"#;

let gpu_task = Task::builder()
    .shader_code(shader.as_bytes().to_vec())
    .input_buffer(input_data)
    .output_buffer_size(output_size)
    .backend(Backend::Gpu)
    .build()?;
```

### 🏭 Iron Lotus Quality Framework

Repartir embodies the **Toyota Production System** principles for software:

- **Jidoka (自働化)**: Automated quality gates that stop the line on defects
- **Genchi Genbutsu (現地現物)**: Radical transparency—every operation is auditable
- **Kaizen (改善)**: Continuous improvement with 95%+ test coverage

### 📊 Certeza Testing Methodology

Three-tiered testing approach:

- **Tier 1**: Sub-second unit tests (on every save)
- **Tier 2**: 1-5 minute comprehensive tests (on commit)
- **Tier 3**: Mutation testing & formal verification (on merge)

Current metrics:
- **97.20% test coverage** (exceeds 95% target)
- **117 tests passing** (92 unit, 9 integration, 4 property, 12 doc)
- **87.3% quality score** (PMAT Grade A+)

## Design Principles

### Message-Oriented Architecture

Inspired by ZeroMQ patterns:

- **PubSub**: Broadcast control messages (shutdown, configuration updates)
- **PushPull**: Asynchronous work distribution pipelines
- **Pure Rust Transport**: tokio + bincode (no zmq C library)

### Work-Stealing Scheduler

Based on Blumofe & Leiserson's provably optimal algorithm:

- **Scalable**: O(1) task submission and local dequeue
- **Adaptive**: Automatically balances load across heterogeneous resources
- **Cache-Friendly**: Local LIFO reduces cache misses

### Binary Execution Model

Tasks execute **compiled Rust binaries** for:

- **Security**: No arbitrary code execution (vetted binaries only)
- **Performance**: Native code performance
- **Isolation**: Process-level sandboxing via renacer

## Who Should Use Repartir?

### Scientific Computing

- Embarrassingly parallel simulations (Monte Carlo, parameter sweeps)
- GPU-accelerated numerical computations
- HPC cluster workflows

### Machine Learning

- Distributed training across GPUs (via aprender integration)
- Hyperparameter search
- Model inference at scale

### Sovereign Infrastructure

- **National AI Capability**: Build AI infrastructure without foreign dependencies
- **Regulatory Compliance**: GDPR, data sovereignty requirements
- **Critical Infrastructure**: Energy grids, defense systems
- **Academic Research**: Reproducible, auditable infrastructure

## What's Next?

- **[Getting Started](./getting-started/installation.md)**: Install Repartir and run your first task
- **[Architecture](./architecture/overview.md)**: Understand the system design
- **[Examples](./examples/cpu-task.md)**: Explore real-world use cases
- **[Development](./development/contributing.md)**: Contribute to the project

## Project Status

- **Version**: 1.0.0
- **Status**: Production-ready for CPU and GPU execution
- **License**: MIT
- **Repository**: [github.com/paiml/repartir](https://github.com/paiml/repartir)
- **Crate**: [crates.io/crates/repartir](https://crates.io/crates/repartir)

---

> **Note**: Repartir is part of the **PAIML Stack** (Pragmatic AI Machine Learning), which includes:
> - **trueno**: SIMD-accelerated tensor operations
> - **aprender**: ML framework for distributed training
> - **repartir**: Distributed computing primitives (this project)
> - **renacer**: Process lifecycle management

Let's build the future of distributed computing, one pure Rust line at a time. 🦀
