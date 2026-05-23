<!-- PCU: cli-code | contract: contracts/apr-page-cli-code-v1.yaml -->

# apr code

Sovereign AI coding assistant — all inference local via realizar (PMAT-182)

**Category**: Inference

## Synopsis

```text
apr code [OPTIONS]
```

## Example

```bash
apr code -p "review this Python function" --max-turns 1
```

## What this does

`apr code` is the local-first coding agent — a Claude Code / Cursor analogue with
all inference performed by `realizar` against a local model (`qwen2.5-coder-*` is
the default sweet spot). It reads `APR.md` / `CLAUDE.md` for project context,
supports tool use, multi-turn sessions, and emits Claude-Code-compatible traces
for the `ccpa measure` parity harness. Used non-interactively (`-p`) it behaves
like `claude -p`; interactively it opens a TUI loop.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-p, --print PROMPT` | Non-interactive: print response and exit | `-p "explain this diff"` |
| `--model M` | Override default model path | `--model qwen2.5-coder-7b.apr` |
| `--project DIR` | Project root (loads APR.md / CLAUDE.md) | `--project ../mylib` |
| `--max-turns N` | Stop after N agent turns | `--max-turns 5` |
| `--resume [ID]` | Resume a previous session | `--resume sess-abc123` |
| `--output-format FMT` | `text` or `json` (Claude Code-style envelope) | `--output-format json` |
| `--emit-trace PATH` | Write a CCPA trace JSONL | `--emit-trace run.jsonl` |

## Common workflows

**One-shot code review from a CI job.**

```bash
git diff HEAD~1 | apr code -p "Review this diff. Flag bugs only." \
    --model qwen2.5-coder-7b.apr --max-turns 1 --output-format json
```

**Iterative refactor with session persistence.**

```bash
apr code --project . --model qwen2.5-coder-7b.apr
# > rename `Foo` to `Bar` across the crate
# (exit, then continue later)
apr code --resume                       # picks the most recent session
```

## Troubleshooting

- **`No APR.md or CLAUDE.md found`** — agent context is empty. Either add a
  `CLAUDE.md` at the project root or pass `--project /path/with/context`.
- **Slow first turn (30s+ to TTFT)** — that's model load + KV warm-up, not a hang.
  Subsequent turns reuse the loaded weights. Use a smaller model
  (`qwen2.5-coder-0.5b`) for snappier iteration.
- **JSON output is missing `result` field** — confirm `--output-format json` is
  passed AFTER `-p`. The non-interactive envelope is only emitted in print mode.

## See also

- Source: [`crates/apr-cli/src/commands/code.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/code.rs)
- Contract: [`contracts/apr-page-cli-code-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-code-v1.yaml)

