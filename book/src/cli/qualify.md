<!-- PCU: cli-qualify | contract: contracts/apr-page-cli-qualify-v1.yaml -->

# apr qualify

Cross-subcommand smoke test (does every tool handle this model?)

**Category**: Quality & Evaluation

## Synopsis

```text
apr qualify [OPTIONS]
```

## Example

```bash
apr qualify qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr qualify` runs the entire `apr` subcommand suite — inspect, validate, lint,
tensors, hex, tree, flow, run, bench, eval — against a single model and reports
which tools succeeded. It exists because models often work with one tool and
break another (a converter might emit valid metadata but a malformed tensor
table). Qualify catches "format not supported" surprises before users do.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--tier T` | `smoke` (Phase 1), `standard` (+contracts), `full` (+playbook) | `--tier standard` |
| `--timeout SEC` | Per-gate timeout (default 120s) | `--timeout 60` |
| `--skip LIST` | Comma-separated gates to skip | `--skip bench,eval` |
| `--json` | One JSON envelope per gate | `--json` |
| `-v, --verbose` | Show each subcommand's stdout | `--verbose` |

## Common workflows

**Smoke a model fresh from the converter.**

```bash
apr qualify newly-converted.apr --tier smoke --json | \
    jq '.gates[] | select(.status == "FAIL")'
```

**Pre-publish full sweep including 10-stage check + canary + eval.**

```bash
apr qualify qwen2.5-coder-1.5b.apr --tier full --timeout 300
```

## Troubleshooting

- **"timeout exceeded"** — bump `--timeout`; 7B models on CPU can need 300s+ for
  bench/eval gates.
- **One gate hangs the whole run** — `apr qualify` enforces per-gate timeouts,
  but a deadlocked subcommand may still need a manual SIGTERM. Use
  `--skip <gate>` to bypass and file a bug.
- **Different tiers contradict each other** — `smoke` only checks command exit
  codes; `standard` adds contract validation; `full` adds the playbook. If
  smoke passes but standard fails, the bug is in tensor-layout contracts.

## See also

- Source: [`crates/apr-cli/src/commands/qualify.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/qualify.rs)
- Contract: [`contracts/apr-page-cli-qualify-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-qualify-v1.yaml)

