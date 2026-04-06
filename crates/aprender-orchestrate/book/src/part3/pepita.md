# Pepita: Sovereign AI Kernel Interfaces

**pepita** is the Sovereign AI Stack's kernel interface library, providing minimal Linux kernel interfaces (`io_uring`, `ublk`, `blk-mq`) and distributed computing primitives for sovereign AI workloads.

## Overview

| Attribute | Value |
|-----------|-------|
| Version | 0.1.x |
| crates.io | [pepita](https://crates.io/crates/pepita) |
| docs.rs | [pepita](https://docs.rs/pepita) |
| License | MIT OR Apache-2.0 |

## Key Features

- **First-Principles Rust**: Zero external dependencies in kernel mode
- **100% Rust, Zero C/C++**: Complete auditability for sovereign AI
- **no_std Compatible**: Core kernel interfaces work without standard library
- **Work-Stealing Scheduler**: Blumofe-Leiserson algorithm implementation
- **Iron Lotus Quality**: 417 tests, 95% coverage

## Design Principles

Pepita follows the **Iron Lotus Framework**:

1. **First-Principles Rust**: Zero external dependencies in kernel mode
2. **Pure Rust Sovereignty**: 100% auditable, zero C/C++ dependencies
3. **Toyota Way Quality**: Jidoka, Poka-yoke, Genchi Genbutsu
4. **EXTREME TDD**: Comprehensive test coverage

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                           User Code                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                          pool.rs                                 │
│                    (High-level Pool API)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       scheduler.rs                               │
│              (Work-Stealing, Blumofe-Leiserson)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       executor.rs                                │
│                    (Backend Dispatch)                            │
├─────────────┬─────────────┬─────────────┬───────────────────────┤
│   CPU       │    GPU      │   MicroVM   │        SIMD           │
│ (threads)   │  (wgpu)     │   (KVM)     │    (AVX/NEON)         │
└─────────────┴──────┬──────┴──────┬──────┴───────────┬───────────┘
                     │             │                  │
              ┌──────▼──────┐ ┌────▼─────┐    ┌───────▼───────┐
              │   gpu.rs    │ │  vmm.rs  │    │   simd.rs     │
              │   (wgpu)    │ │  (KVM)   │    │ (AVX-512/NEON)│
              └─────────────┘ └────┬─────┘    └───────────────┘
                                   │
                            ┌──────▼──────┐
                            │  virtio.rs  │
                            │(vsock,block)│
                            └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Interfaces (no_std)                    │
├─────────────┬─────────────┬─────────────┬───────────────────────┤
│  io_uring   │    ublk     │   blk_mq    │       memory          │
│ (async I/O) │(block dev)  │ (multiqueue)│   (DMA/pages)         │
└─────────────┴─────────────┴─────────────┴───────────────────────┘
```

## Module Overview

### Core Kernel Interfaces (`no_std` compatible)

| Module | Purpose | Key Types |
|--------|---------|-----------|
| **`io_uring`** | Linux async I/O interface | `IoUringSqe`, `IoUringCqe` |
| **`ublk`** | Userspace block device driver | `UblkCtrlCmd`, `UblkIoDesc`, `UblkIoCmd` |
| **`blk_mq`** | Multi-queue block layer | `TagSetConfig`, `Request`, `RequestOp` |
| **`memory`** | Physical/virtual memory management | `DmaBuffer`, `PageAllocator`, `Pfn` |
| **`error`** | Unified error types | `KernelError`, `Result` |

### Distributed Computing (`std` required)

| Module | Purpose | Key Types |
|--------|---------|-----------|
| **`scheduler`** | Work-stealing scheduler | `Scheduler`, `WorkerDeque` |
| **`executor`** | Execution backends | `CpuExecutor`, `Backend` |
| **`task`** | Task definitions | `Task`, `TaskId`, `ExecutionResult` |
| **`pool`** | High-level API | `Pool`, `PoolBuilder` |
| **`transport`** | Wire protocol | `Message`, `Transport` |
| **`fault`** | Fault tolerance | `RetryPolicy`, `CircuitBreaker` |

### Sovereign Infrastructure (`std` required)

| Module | Purpose | Key Types |
|--------|---------|-----------|
| **`zram`** | Compressed RAM block device | `ZramDevice`, `ZramConfig`, `ZramStats` |
| **`vmm`** | KVM-based MicroVM runtime | `MicroVm`, `VmConfig`, `VmState` |
| **`virtio`** | Virtio device implementations | `VirtQueue`, `VirtioVsock`, `VirtioBlock` |
| **`simd`** | SIMD-accelerated operations | `SimdCapabilities`, `SimdOps`, `MatrixOps` |
| **`gpu`** | GPU compute via wgpu | `GpuDevice`, `ComputeKernel`, `GpuBuffer` |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support |
| `kernel` | True no_std without alloc |
| `proptest` | Property-based testing support |

## Quick Start

### Installation

```toml
[dependencies]
pepita = "0.1"

# Kernel mode (no_std)
pepita = { version = "0.1", default-features = false, features = ["kernel"] }
```

### io_uring - Async I/O

```rust
use pepita::io_uring::{IoUringSqe, IoUringCqe, IORING_OP_URING_CMD};

// Submission queue entry - describes an I/O operation
let sqe = IoUringSqe::new(IORING_OP_URING_CMD, fd, addr, len);

// Completion queue entry - result of the operation
let cqe: IoUringCqe = /* from kernel */;
assert_eq!(cqe.res, 0); // Success
```

**Why it matters**: io_uring eliminates syscall overhead by batching I/O operations. One syscall can submit hundreds of operations.

### ublk - Userspace Block Devices

```rust
use pepita::ublk::{UblkCtrlCmd, UblkIoDesc, UBLK_U_CMD_ADD_DEV};

// Control command - add a new block device
let cmd = UblkCtrlCmd::new(UBLK_U_CMD_ADD_DEV, dev_id);

// I/O descriptor - describes a read/write request
let io_desc: UblkIoDesc = /* from kernel */;
let sector = io_desc.start_sector();
```

**Why it matters**: ublk allows implementing block devices entirely in userspace with near-native performance.

### zram - Compressed Memory

```rust
use pepita::zram::{ZramDevice, ZramConfig, ZramCompressor};

// Create a 1GB compressed RAM device
let config = ZramConfig::with_size(1024 * 1024 * 1024)
    .compressor(ZramCompressor::Lz4);
let device = ZramDevice::new(config)?;

// Write a page (4KB)
let data = [0u8; 4096];
device.write_page(0, &data)?;

// Check compression stats
let stats = device.stats();
println!("Compression ratio: {:.2}x", stats.compression_ratio());
```

**Why it matters**: zram provides swap/storage that lives in compressed RAM. A 4GB system can effectively have 12-16GB of memory.

### MicroVM Runtime

```rust
use pepita::vmm::{MicroVm, VmConfig, VmState};

let config = VmConfig::builder()
    .vcpus(2)
    .memory_mb(256)
    .kernel_path("/boot/vmlinuz")
    .build()?;

let vm = MicroVm::create(config)?;
vm.start()?;
let exit_reason = vm.run()?;
```

**Why it matters**: MicroVMs provide hardware-level isolation with sub-100ms cold start. Each function runs in its own VM.

### Work-Stealing Scheduler

```rust
use pepita::scheduler::Scheduler;
use pepita::task::{Task, Priority};

let scheduler = Scheduler::with_workers(4);

let task = Task::builder()
    .binary("./compute")
    .priority(Priority::High)
    .build()?;

scheduler.submit(task).await?;
```

**Why it matters**: Work stealing provides automatic load balancing. Idle workers steal from busy workers' queues.

## Integration with Repartir

Pepita provides the low-level primitives that [repartir](./repartir.md) uses for its high-level distributed computing API:

```rust
// repartir uses pepita's SIMD executor
use repartir::executor::simd::{SimdExecutor, SimdTask};

let executor = SimdExecutor::new(); // Uses pepita::simd internally
let task = SimdTask::vadd_f32(a, b);
let result = executor.execute_simd(task).await?;

// repartir uses pepita's MicroVM for serverless
use repartir::executor::microvm::MicroVmExecutor;

let executor = MicroVmExecutor::new(config)?; // Uses pepita::vmm internally
```

## Use Cases

### Sovereign Infrastructure

Pepita provides building blocks for a complete Docker/Lambda/Kubernetes replacement in pure Rust:

| Use Case | Pepita Module |
|----------|---------------|
| Container replacement | `vmm` (MicroVMs) |
| Storage backend | `ublk`, `blk_mq` |
| Swap/memory extension | `zram` |
| High-throughput I/O | `io_uring` |
| Serverless isolation | `vmm` + `virtio` |

### High-Performance Computing

- **SIMD acceleration**: Auto-detects AVX-512/AVX2/SSE4.1/NEON
- **GPU compute**: Cross-platform via wgpu (Vulkan/Metal/DX12)
- **Work stealing**: Near-linear speedup for parallel workloads

## Comparison with Alternatives

| Feature | pepita | QEMU | Firecracker | Docker |
|---------|--------|------|-------------|--------|
| Language | Rust | C | Rust | Go/C |
| Isolation | VM | VM | VM | Container |
| Boot time | <100ms | seconds | ~100ms | ~500ms |
| Dependencies | 0 | many | few | many |
| Pure Rust | Yes | No | Partial | No |
| no_std | Yes | No | No | No |

## Performance

```text
running 417 tests
test result: ok. 417 passed; 0 failed; 0 ignored
```

### Benchmarks

| Operation | pepita | Baseline |
|-----------|--------|----------|
| io_uring submit | 50ns | N/A |
| zram write (4KB) | 2us | 10us (disk) |
| MicroVM boot | 80ms | 500ms (Docker) |
| SIMD matmul (1Kx1K) | 5ms | 50ms (scalar) |

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Repartir](./repartir.md) | [Trueno](./trueno.md)
