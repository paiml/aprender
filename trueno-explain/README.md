# trueno-explain

PTX/SIMD/wgpu visualization and tracing CLI for Trueno.

[![CI](https://github.com/paiml/trueno/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/trueno/actions)

## Features

- **PTX Analysis**: Register pressure, memory patterns, instruction counts
- **Bug Detection**: Static analysis for common GPU bugs (PARITY-114, etc.)
- **TUI Mode**: Interactive kernel exploration
- **SIMD/wgpu**: Vectorization and shader analysis (coming soon)

## Installation

```bash
cargo install trueno-explain
```

## Usage

```bash
# Analyze a GEMM kernel
trueno-explain ptx -K gemm_tiled -m 1024 -n 1024

# Interactive TUI mode
trueno-explain tui -K gemm_tiled

# Hunt for PTX bugs
trueno-explain bugs -K q4k_gemm --strict --fail-on-bugs

# Compare kernels
trueno-explain compare -a gemm_naive -b gemm_tiled --json
```

## Available Kernels

| Name | Description |
|------|-------------|
| `vector_add` | Simple vector addition |
| `gemm_naive` | Naive matrix multiply |
| `gemm_tiled` | Tiled matrix multiply |
| `softmax` | Numerically stable softmax |
| `q4k_gemm` | Q4_K quantized matmul |
| `q5k_gemm` | Q5_K quantized matmul |
| `q6k_gemm` | Q6_K quantized matmul |

## Commands

| Command | Description |
|---------|-------------|
| `ptx` | Analyze PTX code generation |
| `tui` | Interactive TUI explorer |
| `bugs` | Hunt for PTX bugs |
| `compare` | Compare two kernels |
| `diff` | Compare against baseline (CI integration) |
| `simd` | Analyze SIMD vectorization |
| `wgpu` | Analyze WGSL shaders |

## License

MIT License - see [LICENSE](../LICENSE)

## Part of Trueno

This crate is part of [Trueno](https://github.com/paiml/trueno).
