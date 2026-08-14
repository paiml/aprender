<!-- PCU: cli-simulate | contract: contracts/apr-page-cli-simulate-v1.yaml -->

# apr simulate

Reproducible simulation engine: run and verify EDD (Experiment-Driven Development)
experiments, check EMC (Experiment Method Card) compliance, and render simulations
to SVG.

**Category**: Data

## Was `simular`

This command surface was published as a standalone binary named `simular`
("to simulate" in Spanish). The engine is unchanged — `apr simulate` calls
`simular::cli::run_cli`, the same entry point that binary's `main` called — but
the binary is gone, because a name in a second language tells nobody which
project ships it.

| Before | Now |
|--------|-----|
| `simular run x.yaml --seed 7 -v` | `apr simulate run x.yaml --seed 7 -v` |
| `simular render --domain orbit --fps 60` | `apr simulate render --domain orbit --fps 60` |
| `simular validate x.yaml` | `apr simulate validate x.yaml` |
| `simular verify x.yaml --runs 5` | `apr simulate verify x.yaml --runs 5` |
| `simular emc-check x.yaml` | `apr simulate emc-check x.yaml` |
| `simular emc-validate card.yaml` | `apr simulate emc-validate card.yaml` |
| `simular list-emc` | `apr simulate list-emc` |

Two deliberate differences, both of which turn a silent misread into a refusal:

- **`--format` is now a closed set.** `simular` parsed it as
  `Some("svg-frames") => SvgFrames, _ => SvgKeyframes`, so `--format svg-fames`
  rendered keyframes and printed `Format: svg-keyframes` as though it had
  understood. A misspelling is now rejected with exit 2 and a suggestion.
- **`-v/--verbose` on `run` is apr's global flag**, not a subcommand-local
  duplicate. Same spelling, same effect, and `apr --verbose simulate run …`
  now works too.

Exit codes are unchanged: a failed experiment, a schema violation or an
unsupported render domain all exit **1**.

## Synopsis

```text
apr simulate <COMMAND>

  run <EXPERIMENT>           Run an EDD experiment from a YAML file
  render                     Render a simulation to SVG
  validate <EXPERIMENT>      Validate against the EDD v2 schema
  verify <EXPERIMENT>        Re-run N times and check reproducibility
  emc-check <EXPERIMENT>     Check EMC compliance
  emc-validate <EMC>         Validate an EMC file against the EDD v2 EMC schema
  list-emc                   List EMCs available in the library
```

### `run`

| Argument | Default | Meaning |
|----------|---------|---------|
| `<EXPERIMENT>` | required | Path to the experiment YAML file |
| `--seed <N>` | the experiment's own seed | Override the declared seed |
| `-v`, `--verbose` | off | EMC library load count and per-criterion detail |

### `render`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--domain <DOMAIN>` | `orbit` | `orbit` or `bouncing_balls`; anything else is refused |
| `--format <FORMAT>` | `svg-keyframes` | `svg-keyframes` (template + `keyframes.json`) or `svg-frames` (one SVG per frame) |
| `--output <DIR>` | `.` | Output directory, created if absent |
| `--fps <N>` | `60` | Frames per second |
| `--duration <SECONDS>` | `10.0` | Simulated duration; frames = fps × duration |
| `--seed <N>` | `42` | Seed pinning the draw |

### `verify`

| Argument | Default | Meaning |
|----------|---------|---------|
| `<EXPERIMENT>` | required | Path to the experiment YAML file |
| `--runs <N>` | `3` | Number of runs whose results must agree |

`validate`, `emc-check` and `emc-validate` each take a single required path.
`list-emc` takes no arguments.

## Example

<!-- example-cost: trivial -->
```bash
apr simulate --help
apr simulate list-emc
```

Render two frames of the orbit demo:

<!-- example-cost: trivial -->
```bash
apr simulate render --domain orbit --format svg-keyframes \
    --output /tmp/apr-orbit --fps 2 --duration 1 --seed 7
```

## Full help

Run `apr simulate --help`, or `apr simulate <SUBCOMMAND> --help`, for the
complete option list.

## See also

- Dispatch: [`crates/apr-cli/src/simulate_commands.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/simulate_commands.rs)
- Engine: [`crates/aprender-simulate/src/cli/`](https://github.com/paiml/aprender/blob/main/crates/aprender-simulate/src/cli/mod.rs)
- Contract: [`contracts/apr-page-cli-simulate-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-simulate-v1.yaml)
