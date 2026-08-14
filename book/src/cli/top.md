# apr top

System monitor: CPU, memory, disk, network, GPU, sensors, processes, PSI,
open files, battery, containers — rendered with the presentar-terminal
zero-allocation cell buffer.

**Category**: Observability & Pipeline

## Synopsis

```text
apr top [OPTIONS]
```

## Where this came from

`apr top` is the whole of the former `ptop` binary
(`aprender-present-terminal`, `--features ptop`). Nothing about it changed
except its address.

`ptop` was published, and `cargo install aprender-present-terminal` dropped a
binary called `ptop` into `~/.cargo/bin` — a name several unrelated system
monitors already ship, where the last install silently wins. `top` is the name
every UNIX user already reaches for, and it is unambiguous inside `apr`.

`apr monitor` was not available: that is the **training-run** monitor
(`apr monitor <experiment-dir>`), which watches loss curves, not the machine.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --refresh <MS>` | `1000` | Metrics refresh interval in milliseconds |
| `--deterministic` | off | Deterministic mode for testing: no timestamps, no dynamic data |
| `--no-color` | off | Plain text, no ANSI colour |
| `--render-once` | off | Render one frame to stdout and exit (for comparison/testing) |
| `--width <N>` | `120` | Terminal width used by `--render-once` |
| `--height <N>` | `40` | Terminal height used by `--render-once` |
| `-c, --config <PATH>` | auto | Custom YAML config file; unreadable paths warn and fall back to defaults |
| `--dump-config` | off | Print the default configuration to stdout and exit |
| `--qa-timing` | off | Emit input/render/collect timing diagnostics to stderr every 2s |
| `--explode <PANEL>` | none | Expand one panel to the full screen |

`--explode` accepts `cpu`, `memory` (`mem`), `disk`, `network` (`net`),
`process` (`proc`, `processes`), `gpu`, `sensors` (`sensor`), `connections`
(`conn`), `psi` (`pressure`), `files` (`file`), `battery` (`bat`) and
`containers` (`container`, `docker`).

> **Changed on rehome.** The `ptop` binary took `--explode` as a free-form
> string: an unrecognised panel printed a warning to stderr, rendered the
> ordinary dashboard, and exited **0**. `apr top` refuses an unknown panel at
> parse time. Every name the binary accepted still works.

## Examples

<!-- example-cost: interactive -->
```bash
apr top
```

```bash
# One deterministic frame at a fixed size — this is what the visual
# regression tests capture.
apr top --render-once --deterministic --width 120 --height 40
```

```bash
# Just the memory panel, full screen, refreshed four times a second.
apr top --explode memory --refresh 250
```

```bash
apr top --dump-config > ~/.config/ptop/config.yaml
```

## Full help

Run `apr top --help` for the complete option list.

## See also

- [`apr cbtop`](./cbtop.md) — ComputeBrick *pipeline* monitor (per-brick inference timings), not a system monitor
- [`apr monitor`](./monitor.md) — training-run monitor
- Source: [`crates/apr-cli/src/commands/present.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/present.rs)
- Implementation: [`crates/aprender-present-terminal/src/ptop/run.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-present-terminal/src/ptop/run.rs)
