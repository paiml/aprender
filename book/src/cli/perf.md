<!-- PCU: cli-perf | contract: contracts/apr-cli-commands-v1.yaml -->

# apr perf

Unified performance analysis across every backend: profile kernels, shaders and
SIMD functions, build roofline models, save and diff baselines, and verify
performance contracts.

**Category**: Analysis

This was the standalone `cgp` binary ("Compute-GPU-Profile"). That binary is
gone; the capability is reached as `apr perf`.

## Not to be confused with `apr profile` / `apr bench`

`apr profile <MODEL>` and `apr bench <MODEL>` operate on a **model file**.
`apr perf` operates on **kernels, shaders, functions and backends** — it never
takes a model. They are complementary, not alternatives.

## Synopsis

```text
apr perf profile <TARGET> ...     # kernel|cublas|wgpu|metal|simd|wasm|quant|
                                  # scalar|parallel|compare|scaling|binary|
                                  # python|library
apr perf bench --bench NAME [--counters LIST] [--check-regression]
               [--threshold PCT] [--roofline]
apr perf roofline --target BACKEND [--kernels LIST] [--export FILE] [--empirical]
apr perf diff [--baseline REF] [--current REF] [--before SHA] [--after SHA]
apr perf contract verify [--contracts-dir DIR] [--contract FILE]
                         [--fail-on-regression] [--self-verify]
apr perf contract generate --kernel NAME --size N [--tolerance PCT]
apr perf trace <BINARY> [--duration D]
apr perf explain <TARGET> [--kernel NAME]
apr perf baseline [--save FILE] [--load FILE]
apr perf doctor
apr perf compete <WORKLOAD> --ours CMD [--theirs CMD]... [--label LIST]
apr perf tui
```

`--json` is `apr`'s global flag and propagates into every subcommand above,
which is how `cgp`'s own top-level `--json` behaved.

## Profile targets

| Target | Required arguments | Profiles |
|--------|--------------------|----------|
| `kernel` | `--name`, `--size` | A CUDA PTX kernel via ncu + CUPTI (`--roofline`, `--metrics`) |
| `cublas` | `--op`, `--size` | cuBLAS / cuBLASLt operations |
| `wgpu` | `--shader` | wgpu compute shaders (`--dispatch`, `--target`) |
| `metal` | `--shader` | Apple Metal kernels (`--dispatch`) |
| `simd` | `--function`, `--size`, `--arch` | CPU SIMD functions |
| `wasm` | `--function`, `--size` | WASM SIMD128 via wasmtime |
| `quant` | `--kernel` + `--size`, **or** `--all` | Quantized CPU kernels (Q4K/Q5K/Q6K/Q8/NF4) |
| `scalar` | `--function`, `--size` | Scalar baseline |
| `parallel` | `--function`, `--size` | Rayon workloads (`--threads`) |
| `compare` | `--kernel`, `--size`, `--backends` | Cross-backend comparison |
| `scaling` | `--size` | Thread-count sweep (`--max-threads`, `--runs`) |
| `binary` | `<PATH>` | An arbitrary binary (`--kernel-filter`, `--trace`, `--duration`) |
| `python` | `-- <ARGS>` | A Python script |
| `library` | `--so`, `--symbol` | A shared-library symbol |

`apr perf profile quant` requires either `--kernel` **and** `--size`, or
`--all`; a bare invocation is refused rather than silently profiling a default.

## `contract verify --self-verify`

Note the flag is spelled `--self-verify`, not `--self`. The field carries
`#[arg(long, name = "self")]`, and in clap 4 `name` sets the argument *id*, not
the long spelling — which is still derived from the field name. `--self` never
worked; this is documented rather than changed, so existing invocations of
`--self-verify` keep working.

## Examples

<!-- example-cost: trivial -->
```bash
apr perf doctor
```

<!-- example-cost: trivial -->
```bash
apr perf profile scalar --function dot --size 4096
```

<!-- example-cost: moderate -->
```bash
apr perf roofline --target avx2 --export /tmp/roofline.json
apr perf profile compare --kernel gemm --size 1024 --backends scalar,simd
```

## Commands that are not implemented

`apr perf tui` and `apr perf profile metal` / `profile library` have no
implementation. They **fail** with a message saying so. Previously they printed
a note (or echoed their arguments back) and exited `0`, so a caller — including
a CI gate — read "did nothing" as "succeeded". That is the #2407 defect class;
an advertised command with no implementation must fail.

## Full help

Run `apr perf --help`, or `apr perf <SUBCOMMAND> --help`, for the complete
option list.

## See also

- Model-level profiling: [`apr profile`](./profile.md), [`apr bench`](./bench.md)
- Backend load-testing: [`apr compute`](./compute.md)
- Source: [`crates/aprender-cgp/src/cli.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-cgp/src/cli.rs)
