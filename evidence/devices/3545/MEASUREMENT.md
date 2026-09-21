# #3545 — `apr devices` describes the binary it is part of

Host `noah-Lambda-Vector`, **RTX 4090 physically present for all three rows**, so every difference
below is about the BUILD and not about the hardware.

| build | cuda `status` | cuda `source` | |
|---|---|---|---|
| `--features cuda`, **before** | `unavailable` | **`not-compiled`** | the shipped defect |
| `--features cuda`, **after** | `ready` | **`dlopen`** | fixed |
| `--no-default-features --features inference` | `unavailable` | `not-compiled` | honest, unchanged |

The third row is the falsifier: **the fix is not "always say yes."** A build without the feature
still reports `not-compiled` on a host that has a GPU, because the question `devices` answers is
about the binary.

## Mechanism

`crates/apr-cli/Cargo.toml`:

```
cuda = ["inference", "realizar/cuda", "entrenar/cuda", "aprender-train-distill?/cuda"]
                                    ^ no trueno/cuda
```

`realizar/cuda` gave the inference stack CUDA, so `apr run --gpu` really did run on the GPU. But
`apr devices` reads **`trueno`'s** backend registry, which was built without **its** `cuda` feature,
so `default_factories()` omitted `CudaFactory`, no factory produced a cuda entry, and the registry
emitted `missing_entry(Cuda)` → `Source::NotCompiled`.

**Two flags decided one fact.** The fix adds `trueno/cuda` so a single flag decides both what runs
and what is reported.

## A correction to #3545's acceptance query

Criterion 1 asks for:

```
apr devices --json | jq -e '.entries[]|select(.kind=="cuda")|.source.kind=="compiled-in"'
```

**That would be false even after a correct fix, and satisfying it would require the registry to lie.**
`Source::CompiledIn` is documented as *"Linked into the binary"*; `Source::Dlopen(path)` as *"Loaded
at run time from this library path"*. The CUDA **factory** is linked in; the CUDA **driver** is
dlopened at run time, which is what the fixed build reports. The truthful assertion is the negative
one:

```
.source.kind != "not-compiled"     # and status.state == "ready" where a device exists
```

Criterion 2 (a `-cpu` build answers `not-compiled`) is correct as written and is the arm that keeps
the fix honest.

## The gate

`crates/apr-cli/tests/devices_reports_its_own_build.rs` asserts the two flags are one flag, in both
directions, and **RED-turns on the defect**: removing `trueno/cuda` from the feature fails it with
`built with apr-cli/cuda, but the registry `devices` reads says not-compiled`. Restoring it passes.
No GPU and no model are required for either arm — the cuda arm needs only the feature.
