# Bashrs: Rust to Shell Transpiler

> **"Write Rust, deploy shell. Deterministic bootstrap scripts for any environment."**

Bashrs transpiles Rust code to portable POSIX shell scripts. It enables writing complex installation and bootstrap logic in Rust while deploying as zero-dependency shell scripts.

## Overview

| Attribute | Value |
|-----------|-------|
| **Version** | 6.41.0 |
| **Layer** | L3: Transpilers |
| **Direction** | Rust → Shell |
| **Repository** | [github.com/paiml/bashrs](https://github.com/paiml/bashrs) |

## Why Bashrs?

### The Bootstrap Problem

When deploying software, you face a chicken-and-egg problem:

1. Your installer needs dependencies (Rust, Python, Node...)
2. But you're trying to install those dependencies
3. The only universal runtime is `/bin/sh`

### Traditional Solutions

| Approach | Problem |
|----------|---------|
| Shell scripts | Hard to test, platform bugs, no type safety |
| Python installers | Requires Python pre-installed |
| Go binaries | Large binaries, need per-platform builds |
| curl \| bash | Security concerns, no verification |

### Bashrs Solution

Write your installer in Rust with full type safety and testing, then transpile to a portable shell script:

```
Rust (tested, typed) → bashrs → Shell (universal, portable)
```

## Capabilities

### rust_to_shell

Transpile Rust functions to shell:

```rust
// install.rs
use bashrs::prelude::*;

#[bashrs::main]
fn main() {
    // Check if Rust is installed
    if !command_exists("rustc") {
        println("Installing Rust...");
        curl("https://sh.rustup.rs", "-sSf") | sh();
    }

    // Install the application
    cargo(&["install", "batuta"]);

    println("Installation complete!");
}
```

Generates:

```bash
#!/bin/sh
set -e

main() {
    # Check if Rust is installed
    if ! command -v rustc >/dev/null 2>&1; then
        echo "Installing Rust..."
        curl -sSf https://sh.rustup.rs | sh
    fi

    # Install the application
    cargo install batuta

    echo "Installation complete!"
}

main "$@"
```

### bootstrap_scripts

Generate deterministic bootstrap scripts for reproducible environments:

```rust
use bashrs::prelude::*;

#[bashrs::bootstrap]
fn setup_dev_environment() {
    // Deterministic package installation
    apt_install(&["build-essential", "pkg-config", "libssl-dev"]);

    // Rust toolchain
    rustup_install("stable");
    rustup_component_add(&["clippy", "rustfmt", "llvm-tools-preview"]);

    // Cargo tools
    cargo_install(&["cargo-nextest", "cargo-llvm-cov", "cargo-mutants"]);

    // Verify installation
    assert_command("cargo --version");
    assert_command("cargo nextest --version");
}
```

### cross_platform_shell

Generate POSIX-compliant shell code that works everywhere:

```rust
use bashrs::prelude::*;

#[bashrs::portable]
fn detect_os() -> String {
    // Bashrs generates portable OS detection
    match os() {
        Os::Linux => "linux",
        Os::MacOS => "darwin",
        Os::Windows => "windows",  // WSL/Git Bash
        Os::FreeBSD => "freebsd",
    }
}

#[bashrs::portable]
fn install_package(name: &str) {
    // Generates package manager detection
    match package_manager() {
        Apt => apt_install(&[name]),
        Brew => brew_install(&[name]),
        Dnf => dnf_install(&[name]),
        Pacman => pacman_install(&[name]),
    }
}
```

Generates:

```bash
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux";;
        Darwin*) echo "darwin";;
        MINGW*|MSYS*|CYGWIN*) echo "windows";;
        FreeBSD*) echo "freebsd";;
        *) echo "unknown";;
    esac
}

install_package() {
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y "$1"
    elif command -v brew >/dev/null 2>&1; then
        brew install "$1"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "$1"
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm "$1"
    else
        echo "No supported package manager found" >&2
        exit 1
    fi
}
```

## Integration with Batuta

Generate installation scripts for batuta deployments:

```rust
use bashrs::prelude::*;

#[bashrs::main]
fn install_batuta() {
    println("=== Batuta Installation ===");

    // Step 1: System dependencies
    println("Installing system dependencies...");
    install_build_essentials();

    // Step 2: Rust toolchain
    println("Setting up Rust...");
    ensure_rust_installed();
    rustup_update();

    // Step 3: Install batuta
    println("Installing batuta...");
    cargo_install(&["batuta"]);

    // Step 4: Verify
    println("Verifying installation...");
    let version = capture("batuta --version");
    println(format!("Installed: {}", version));

    println("=== Installation Complete ===");
}
```

## Integration with Repartir

Generate cluster node bootstrap scripts:

```rust
use bashrs::prelude::*;

#[bashrs::main]
fn bootstrap_worker_node() {
    let coordinator = env_required("COORDINATOR_HOST");
    let node_id = env_or("NODE_ID", &generate_node_id());

    println(format!("Bootstrapping worker node: {}", node_id));

    // Install repartir
    cargo_install(&["repartir"]);

    // Configure node
    write_file("/etc/repartir/config.toml", &format!(r#"
[node]
id = "{}"
coordinator = "{}"

[resources]
cpus = {}
memory_gb = {}
"#, node_id, coordinator, num_cpus(), memory_gb()));

    // Start worker service
    systemctl_enable("repartir-worker");
    systemctl_start("repartir-worker");

    println("Worker node ready!");
}
```

## CLI Usage

```bash
# Transpile Rust to shell
bashrs transpile install.rs -o install.sh

# Build and run directly
bashrs run install.rs

# Generate with specific shell target
bashrs transpile --target bash install.rs    # Bash-specific features
bashrs transpile --target posix install.rs   # POSIX-only (most portable)
bashrs transpile --target zsh install.rs     # Zsh-specific features

# Verify generated script
bashrs verify install.sh  # Check for common issues

# Test on multiple shells
bashrs test install.rs --shells bash,dash,zsh
```

## Example: Multi-Stage Installer

```rust
use bashrs::prelude::*;

#[bashrs::main]
fn main() {
    let args = parse_args();

    match args.command.as_str() {
        "install" => install(),
        "uninstall" => uninstall(),
        "upgrade" => upgrade(),
        "doctor" => doctor(),
        _ => print_help(),
    }
}

fn install() {
    println("Installing Sovereign AI Stack...");

    // Phase 1: Base dependencies
    section("Phase 1: System Dependencies");
    install_system_deps();

    // Phase 2: Rust ecosystem
    section("Phase 2: Rust Toolchain");
    install_rust_ecosystem();

    // Phase 3: Stack components
    section("Phase 3: Stack Components");
    cargo_install(&[
        "trueno",
        "aprender",
        "batuta",
        "repartir",
        "renacer",
    ]);

    // Phase 4: Verification
    section("Phase 4: Verification");
    verify_installation();

    success("Installation complete!");
}

fn doctor() {
    println("Checking installation health...");

    check("Rust compiler", "rustc --version");
    check("Cargo", "cargo --version");
    check("Trueno", "cargo install --list | grep trueno");
    check("Batuta", "batuta --version");

    println("All checks passed!");
}
```

## Comparison with Alternatives

| Feature | Raw Shell | Bashrs | Ansible | Docker |
|---------|-----------|--------|---------|--------|
| Zero dependencies | Yes | Yes | No | No |
| Type safety | No | Yes | No | N/A |
| Testable | Hard | Yes | Hard | Yes |
| Cross-platform | Maybe | Yes | Yes | Yes |
| Reproducible | No | Yes | Yes | Yes |
| Size | Tiny | Tiny | Large | Large |

## Key Takeaways

- **Write Rust, deploy shell:** Full Rust safety, universal deployment
- **Zero dependencies:** Generated scripts need only `/bin/sh`
- **Deterministic:** Same input always generates same output
- **Testable:** Test your Rust code, deploy the shell
- **Cross-platform:** POSIX-compliant output works everywhere

---

**Previous:** [Decy: C/C++ to Rust](./decy.md)
**Next:** [Ruchy: Systems Scripting](./ruchy.md)
