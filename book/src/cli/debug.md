<!-- PCU: cli-debug | contract: contracts/apr-page-cli-debug-v1.yaml -->

# apr debug

Simple debugging output ("drama" mode available)

**Category**: Inspection

## Synopsis

```text
apr debug [OPTIONS]
```

## Example

```bash
apr debug qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr debug` is the catch-all "tell me what's in this file" command — magic bytes,
metadata, ASCII string extraction, optional hex dump. It is intentionally less
strict than `apr validate` (won't fail on warnings) and less specialized than
`apr tensors` / `apr inspect`. The `--drama` flag is the verbose, narrated
variant used for runbooks and screencasts.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--drama` | Theatrical narrated output | `--drama` |
| `--hex` | Include hex dump section | `--hex` |
| `--strings` | Extract ASCII strings | `--strings` |
| `--limit N` | Cap output lines (default 256) | `--limit 100` |

## Common workflows

**Quick triage when `apr run` fails.**

```bash
apr debug suspect.gguf --strings | head -50
# Look for unexpected vocab tokens, broken metadata strings, etc.
```

**Capture a model snapshot for a bug report.**

```bash
apr debug qwen2.5-coder-1.5b.apr --drama --hex --limit 512 > bug-1234-debug.txt
gh issue create --title "..." --body-file bug-1234-debug.txt
```

## Troubleshooting

- **Output empty for a known-good model** — most likely a permission issue;
  `apr debug` will exit silently if it can't read the file. Check with
  `ls -l <model>`.
- **`--strings` returns gibberish** — the file is encrypted or compressed.
  `apr debug` doesn't decrypt; use `apr decrypt` first if applicable.
- **Drama mode mangles JSON pipelines** — drama mode prints to stderr. Use
  `--json` instead for any scripting use.

## See also

- Source: [`crates/apr-cli/src/commands/debug.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/debug.rs)
- Contract: [`contracts/apr-page-cli-debug-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-debug-v1.yaml)

