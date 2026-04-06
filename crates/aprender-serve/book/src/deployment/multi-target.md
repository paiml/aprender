# Multi-Target Deployment

Realizar supports deployment to multiple targets with automatic capability detection.

## Supported Targets

| Target | Features | Use Case |
|--------|----------|----------|
| **Native** | Full (SIMD, GPU, threads) | Bare metal, EC2 |
| **Lambda** | SIMD, no GPU | AWS Lambda ARM64 |
| **Docker** | Full | Container deployments |
| **WASM** | CPU-only, no threads | Cloudflare Workers |

## Target Detection

```rust
use realizar::target::DeployTarget;

let target = DeployTarget::detect();

match target {
    DeployTarget::Native => println!("Running on bare metal"),
    DeployTarget::Lambda => println!("Running in AWS Lambda"),
    DeployTarget::Docker => println!("Running in Docker container"),
    DeployTarget::Wasm => println!("Running in WebAssembly"),
}
```

### Detection Heuristics

```
                    ┌─────────────────────┐
                    │  DeployTarget::detect()  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ cfg!(wasm32)?       │
                    └──────────┬──────────┘
                         yes   │   no
                    ┌──────────┴──────────┐
                    ▼                     ▼
               DeployTarget::Wasm    ┌────────────────────┐
                                     │ AWS_LAMBDA_FUNCTION │
                                     │ _NAME env var?     │
                                     └─────────┬──────────┘
                                          yes  │   no
                                     ┌─────────┴──────────┐
                                     ▼                    ▼
                                DeployTarget::Lambda  ┌──────────────┐
                                                      │ /.dockerenv  │
                                                      │ exists?      │
                                                      └──────┬───────┘
                                                        yes  │   no
                                                      ┌──────┴───────┐
                                                      ▼              ▼
                                                 DeployTarget::Docker  DeployTarget::Native
```

## Target Capabilities

Each target has specific capabilities:

```rust
pub struct TargetCapabilities {
    pub supports_simd: bool,
    pub supports_gpu: bool,
    pub supports_threads: bool,
    pub supports_filesystem: bool,
    pub supports_async_io: bool,
    pub max_memory_mb: u32,  // 0 = unlimited
}
```

### Capability Matrix

| Capability | Native | Lambda | Docker | WASM |
|------------|--------|--------|--------|------|
| SIMD | ✅ | ✅ | ✅ | ❌ |
| GPU | ✅ | ❌ | ✅ | ❌ |
| Threads | ✅ | ✅ | ✅ | ❌ |
| Filesystem | ✅ | ⚠️ /tmp | ✅ | ❌ |
| Async I/O | ✅ | ✅ | ✅ | ✅ |
| Max Memory | ∞ | 10GB | ∞ | 128MB |

## Adaptive Inference

Select inference strategy based on capabilities:

```rust
use realizar::target::{DeployTarget, TargetCapabilities};

let target = DeployTarget::detect();
let caps = target.capabilities();

if caps.supports_gpu {
    // Use GPU-accelerated inference
    model.infer_gpu(&input)
} else if caps.supports_simd {
    // Use SIMD-accelerated inference
    model.infer_simd(&input)
} else {
    // Fallback to scalar inference
    model.infer_scalar(&input)
}
```

## Build Targets

### Native (Default)

```bash
cargo build --release
```

### Lambda ARM64

```bash
# Cross-compile for ARM64 Lambda
cargo build --release --target aarch64-unknown-linux-gnu --features lambda

# Package for Lambda
zip -j function.zip target/aarch64-unknown-linux-gnu/release/bootstrap
```

### Docker

```bash
docker build -t realizar:latest .
```

### WASM

```bash
cargo build --release --target wasm32-unknown-unknown
wasm-opt -O3 target/wasm32-unknown-unknown/release/realizar.wasm -o realizar.wasm
```
