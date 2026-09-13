# `dogfood_examples` selftest fixture

The scratch cargo workspace `scripts/dogfood_examples.sh --selftest` copies into a
`mktemp -d` directory. One package, no dependencies, five examples — one per
classification the script is allowed to emit:

| example | what it does | expected class |
|---|---|---|
| `ok` | exits 0 | `pass` |
| `bad` | exits 3 | `fail` (rc 3) |
| `hang` | sleeps forever | `timeout` (only the `timeout(1)` wrapper can decide this) |
| `needs_arg` | prints a `Usage:` line to stderr, exits 2 | `needs-args`, citing that line |
| `nohw` | prints a CUDA driver error to stderr, exits 1 | `needs-hardware`, citing that line |

The manifest is `Cargo.toml.in`, not `Cargo.toml`, on purpose: a nested real
manifest inside the repo tree is a second package that `cargo metadata`,
`cargo package` and every workspace-wide guard would have to be told about.
The selftest renames it on copy.

No dependencies and no `build.rs`, so the whole scratch build is one rustc
invocation per target and the selftest stays under two minutes.

The `hang` example is the mutation witness: with the `timeout` wrapper defeated
(`DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT=1`, honoured only under `--selftest`) it can
never be classified `timeout`, and the selftest's mutation row proves that.
