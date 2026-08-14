<!-- PCU: cli-probar | contract: contracts/apr-page-cli-probar-v1.yaml -->

# apr probar

The Probar testing framework for WASM apps and games: run tests, record them,
report on them, check WASM compliance, verify audio/video/animation output, and
stress the runtime.

**Category**: Tools & Integration

`apr probar tensor` is apr's own (PMAT-481 tensor visual regression). Every
other subcommand was the standalone `probador` binary's. That binary is gone
(APR-MONO: one installed binary, `apr`), and its subcommands are flattened in
here, so `apr probar <cmd>` is spelled exactly as `probador <cmd>` was.

## Synopsis

```text
apr probar [--color auto|always|never] <COMMAND>
```

| Command | What it does |
|---------|--------------|
| `tensor` | Export tensor activations for visual regression testing (apr's own) |
| `test` | Run tests |
| `record` | Record a test execution (`<TEST>` required) |
| `report` | Generate reports |
| `coverage` | Generate coverage heatmaps |
| `init` | Initialize a new Probar project |
| `config` | Show configuration |
| `serve` | Start the WASM development server |
| `build` | Build a WASM package |
| `watch` | Watch for changes and rebuild |
| `playbook` | Run state-machine playbooks |
| `comply` | WASM compliance checks C001–C010 |
| `av-sync` | Verify audio/visual synchronization against EDL ground truth |
| `audio` | Verify audio quality (levels, clipping, silence) |
| `video` | Verify video quality (codec, resolution, FPS, duration) |
| `animation` | Verify animation timing and easing curves |
| `stress` | Browser/WASM stress tests |
| `llm` | LLM inference testing (requires `--features llm`) |

## Two flags that changed spelling, and why

Both are collisions with `apr`'s own root flags, which are **global** and so
propagate onto every subcommand. Neither capability was lost.

### `--json-out`, not `--json`, on `apr probar coverage`

`probador coverage --json <FILE>` wrote a JSON coverage file. `apr` declares a
global `--json: bool` meaning "output as JSON". Both would claim the argument
id `json` with different types, which clap does **not** catch in
`Command::debug_assert` — it panics at parse time with *"Mismatch between
definition and access of `json`"*, so the subcommand would abort on every
invocation. The file output is therefore spelled `--json-out <FILE>`, and
`--json` keeps its apr-wide meaning.

```bash
apr probar coverage --json-out cov.json --png cov.png
```

### `apr probar comply migrate --version <V>` needed the auto flag disabled

`migrate` takes a real `--version` argument (the version to migrate *to*), and
the root command sets `propagate_version = true`, which pushes clap's
auto-generated `--version` onto every subcommand. Two arguments then claimed the
id `version`. This conflict shipped undetected in the standalone `probador`
binary because nothing there ever called `Command::debug_assert()`; apr's
`test_cli_parsing_valid` does, which is how it surfaced. The auto flag is now
disabled on that one leaf, so the documented argument works and
`apr probar --version` still answers.

## Verbosity

`apr`'s global `-v/--verbose` and `-q/--quiet` drive probador's verbosity:
`--quiet` selects Quiet, `--verbose` selects Verbose. probador's own `-vv` /
`-vvv` (its Debug level) has no spelling here, because apr's `--verbose` is a
boolean rather than a repeatable count.

## `--color`

`probador` carried `--color` as a global on its root command. Flattening only
its subcommands would have dropped it, so `apr probar --color <auto|always|never>`
is declared on the container and forwarded to the probador config.

## Examples

<!-- example-cost: trivial -->
```bash
apr probar test --help
apr probar comply --help
```

<!-- example-cost: trivial -->
```bash
apr probar tensor model.apr --format json --golden ./refs
```

## Full help

Run `apr probar --help`, or `apr probar <SUBCOMMAND> --help`, for the complete
option list.

## See also

- Source: [`crates/apr-cli/src/commands/probar.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/probar.rs)
- Handlers: [`crates/aprender-test-cli/src/run.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-test-cli/src/run.rs)
- Contract: [`contracts/apr-page-cli-probar-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-probar-v1.yaml)
