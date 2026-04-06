# Installation

## Prerequisites

Repartir requires:

- **Rust**: 1.75+ (for stable async traits)
- **Operating System**: Linux, macOS, Windows
- **Optional**: GPU drivers for GPU execution

## Install Rust

If you don't have Rust installed, use rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Add Repartir to Your Project

Add repartir to your `Cargo.toml`:

```toml
[dependencies]
repartir = "1.0"

# Enable specific backends (features)
repartir = { version = "1.0", features = ["cpu", "gpu", "remote", "remote-tls"] }
```

## Feature Flags

Repartir uses cargo features to enable different backends:

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `cpu` | CPU executor (default) | `tokio`, `num_cpus` |
| `gpu` | GPU executor via wgpu | `wgpu`, `futures` |
| `remote` | Remote TCP execution | `tokio`, `bincode` |
| `remote-tls` | TLS for remote executor | `rustls`, `tokio-rustls` |
| `full` | All features enabled | All of the above |

### Minimal Installation (CPU only)

```toml
[dependencies]
repartir = "1.0"  # Default: cpu feature only
```

### Full Installation (All Features)

```toml
[dependencies]
repartir = { version = "1.0", features = ["full"] }
```

### GPU-Only Installation

```toml
[dependencies]
repartir = { version = "1.0", features = ["gpu"] }
```

## GPU Requirements

For GPU execution, ensure you have appropriate drivers:

### Linux (Vulkan)

```bash
# Ubuntu/Debian
sudo apt install vulkan-utils libvulkan-dev

# Fedora/RHEL
sudo dnf install vulkan-tools vulkan-loader-devel

# Verify
vulkaninfo
```

### macOS (Metal)

Metal drivers are included with macOS. No additional installation needed.

### Windows (DirectX 12)

DirectX 12 is included with Windows 10/11. No additional installation needed.

## Verify Installation

Create a new project:

```bash
cargo new my-distributed-app
cd my-distributed-app
```

Add repartir to `Cargo.toml`:

```toml
[dependencies]
repartir = "1.0"
tokio = { version = "1.35", features = ["rt-multi-thread", "macros"] }
```

Test with a simple example in `src/main.rs`:

```rust
use repartir::executor::{CpuExecutor, Executor};
use repartir::task::Task;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let executor = CpuExecutor::new();

    let task = Task::builder()
        .binary("/bin/echo")
        .arg("Hello from Repartir!")
        .build()?;

    let result = executor.execute(task).await?;
    println!("{}", result.stdout_str()?);

    Ok(())
}
```

Run it:

```bash
cargo run
# Output: Hello from Repartir!
```

## Development Installation

For contributors, clone the repository:

```bash
git clone https://github.com/paiml/repartir.git
cd repartir

# Build with all features
cargo build --all-features

# Run tests
cargo test --all-features

# Run examples
cargo run --example cpu_executor --features cpu
cargo run --example gpu_compute --features gpu
```

## Next Steps

- **[Quick Start](./quick-start.md)**: Run your first distributed task
- **[First Task](./first-task.md)**: Understand the task model
- **[Core Concepts](./core-concepts.md)**: Learn the architecture
