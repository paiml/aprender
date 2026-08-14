<!-- PCU: cli-data | contract: contracts/apr-page-cli-data-v1.yaml -->

# apr data

Dataset pipeline: ML-dataset hygiene (audit, split, balance, dedup,
decontaminate) plus format conversion, inspection, mixing, filtering, a
registry, drift detection, quality scoring and federated splits — all powered
by alimentar.

**Category**: Data

## Absorbed the `alimentar` binary

`aprender-data` used to publish a binary called `alimentar`, and that name
**collides on crates.io**: an unrelated published `alimentar` crate ships a bin
of the same name, so `cargo install aprender-data` and `cargo install alimentar`
overwrote each other in `~/.cargo/bin` — whichever ran last won, silently, and
neither tool told you.

The binary is gone. Every one of its commands is now an `apr data` subcommand,
dispatched through `alimentar::cli::dispatch` — the same entry point its `main`
reached, so there is one implementation, not two.

| Before | Now |
|--------|-----|
| `alimentar convert a.csv b.parquet` | `apr data convert a.csv b.parquet` |
| `alimentar info \| head \| schema <path>` | `apr data info \| head \| schema <path>` |
| `alimentar mix a.parquet:0.8 b.parquet:0.2 -o m.parquet` | `apr data mix …` |
| `alimentar fim <in> -o <out>` | `apr data fim <in> -o <out>` |
| `alimentar filter-text <in> -o <out>` | `apr data filter-text <in> -o <out>` |
| `alimentar view <path>` | `apr data view <path>` |
| `alimentar import local\|hf …` | `apr data import local\|hf …` |
| `alimentar registry init\|list\|push\|pull\|search\|show-info\|delete` | `apr data registry …` |
| `alimentar drift detect\|report\|sketch\|merge\|compare` | `apr data drift …` |
| `alimentar quality check\|report\|score\|profiles` | `apr data quality …` |
| `alimentar fed manifest\|plan\|split\|verify` | `apr data fed …` |
| `alimentar repl` | `apr data repl` |
| `alimentar dedup <in> -o <out>` | **`apr data dedup-text <in> -o <out>`** |

### Why `dedup` became `dedup-text`

`apr data dedup` already existed and does something **else**: it removes exact
duplicate *rows* from a JSONL file, matching whole objects independently of key
order. alimentar's `dedup` loads a Parquet/CSV/JSON dataset through Arrow and
collapses rows whose *text column* repeats, auto-detecting that column when
`--column` is omitted.

Two different operations on two different inputs cannot share one name, and
merging them would have silently changed one of the two behaviours. Both are
kept, under names that say which is which:

```text
apr data dedup      <file.jsonl> -o <out.jsonl>          # exact whole-row, JSONL
apr data dedup-text <file.parquet> -o <out.parquet>      # text-column, Arrow
                    [--column <NAME>]
```

### Compile-time gating (unchanged)

`hub` (HuggingFace push) and `doctest` (Python doctest extraction) sit behind
alimentar's non-default `hf-hub` and `doctest` features. They were absent from
the default `alimentar` binary too, so `apr data` has exactly the surface a
default `cargo install aprender-data` gave you.

### Two flag changes forced by apr's global options

`apr` has global `-v/--verbose` and `-q/--quiet`, and clap refuses a whole
subcommand when two arguments claim one short. Under `apr data registry`:

- `push`, `pull` and `delete` take `--version <SEMVER>` — a **dataset** version
  — long-form only. It also sets `disable_version_flag`, so the argument is not
  shadowed by apr's auto-generated `--version`. `apr --version` still reports
  the binary version at the top level.

## Synopsis

```text
apr data <COMMAND>

  # ML-dataset hygiene (JSONL classification datasets)
  audit           Audit a JSONL dataset for quality issues
  split           Stratified train/val/test split
  decontaminate   Check for benchmark contamination via n-gram overlap
  dedup           Remove exact duplicate rows
  balance         Resample to address class imbalance

  # Dataset tooling (Arrow: Parquet/CSV/JSON)
  convert         Convert between data formats
  info            Display dataset information
  head            Display first N rows
  schema          Display dataset schema
  mix             Mix multiple datasets with weighted sampling
  fim             Apply Fill-in-the-Middle transform for code models
  dedup-text      Deduplicate by text content
  filter-text     Filter by text quality signals
  view            Interactive TUI viewer
  import          Import from local files or HuggingFace Hub
  registry        Dataset sharing and discovery
  drift           Data drift detection
  quality         Data quality checking
  fed             Federated split coordination
  repl            Interactive REPL for data exploration
```

## Example

<!-- example-cost: trivial -->
```bash
apr data --help
apr data quality profiles
```

Inspect and convert a dataset:

<!-- example-cost: trivial -->
```bash
printf '{"text":"alpha","label":1}\n{"text":"beta","label":2}\n' > /tmp/apr-ds.jsonl
apr data schema /tmp/apr-ds.jsonl
apr data head /tmp/apr-ds.jsonl -n 2
apr data convert /tmp/apr-ds.jsonl /tmp/apr-ds.parquet
```

## Full help

Run `apr data --help`, or `apr data <SUBCOMMAND> --help`, for the complete
option list.

## See also

- Hygiene commands: [`crates/apr-cli/src/commands/data.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/data.rs)
- Dataset tooling: [`crates/aprender-data/src/cli/`](https://github.com/paiml/aprender/blob/main/crates/aprender-data/src/cli/mod.rs)
- Contract: [`contracts/apr-page-cli-data-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-data-v1.yaml)
