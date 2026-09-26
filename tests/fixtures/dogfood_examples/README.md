# `dogfood_examples` selftest fixture

The scratch cargo workspace `scripts/dogfood_examples.sh --selftest` copies into a
`mktemp -d` directory. One package, no dependencies. The first seven examples are one per
classification the script infers from output alone:

| example | what it does | expected class |
|---|---|---|
| `ok` | exits 0 | `pass` |
| `bad` | exits 3 | `fail` (rc 3) |
| `hang` | sleeps forever | `timeout` (only the `timeout(1)` wrapper can decide this) |
| `needs_arg` | prints a `Usage:` line to stderr, exits 2 | `needs-args`, citing that line |
| `nohw` | prints a CUDA driver error to stderr, exits 1 | `needs-hardware`, citing that line |
| `nodata` | prints `Model not found at …` and `Download with: apr pull hf://…` to stderr, exits 1 | `needs-data`, citing that line |
| `nofeature` | prints `This example requires the 'compression' feature.`, exits 1 | `needs-feature`, citing that line |

The manifest is `Cargo.toml.in`, not `Cargo.toml`, on purpose: a nested real
manifest inside the repo tree is a second package that `cargo metadata`,
`cargo package` and every workspace-wide guard would have to be told about.
The selftest renames it on copy.

No dependencies and no `build.rs`, so the whole scratch build is one rustc
invocation per target. (On lambda-labs every build also waits for a heavy-cargo slot,
so wall time there is set by the queue, not the fixture.)

The `hang` example is the mutation witness: with the `timeout` wrapper defeated
(`DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT=1`, honoured only under `--selftest`) it can
never be classified `timeout`, and the selftest's mutation row proves that.

## Declared examples

The second half of the fixture exercises `[package.metadata.dogfood-examples]`,
the per-example declaration at the end of `Cargo.toml.in` (it must stay the LAST
table, so the selftest can append refusal variants to it). A declaration is a
claim the run checks, never a skip: the example must print its `expect` line, and
an example that exits 0 is `pass` whatever it declares.

| example | declared | what it does | expected class |
|---|---|---|---|
| `server` | `long-running` | loops forever | `long-running` (alive at `--long-running-secs`) |
| `server_crash` | `long-running` | prints AddrInUse, exits 1 | `fail` (a server that dies is not long-running) |
| `decl_args` | `needs-args` | prints its usage line, exits 2 | `needs-args`, citing that line |
| `decl_drift` | `needs-hardware` | panics on an assertion | `fail`: no line contains its `expect` |
| `decl_ok` | `needs-data` | exits 0 | `pass` |
| `decl_net` | `needs-net` | prints a refused worker connection, exits 1 | `needs-net` |
| `decl_tty` | `needs-tty` | prints ENXIO, exits 1 | `needs-tty` |
| `decl_hang` | `needs-data` | loops forever | `timeout` (only `long-running` excuses a hang) |
| `decl_build` | `needs-data` | does not compile | `fail` (a declaration covers the run, never the build) |

The refusal rows append one invalid declaration each (a target that does not
exist, an unknown class, no `expect`, no `reason`, a value that is not a table)
and assert exit 2 with `FAIL (declaration)` before anything is built.
