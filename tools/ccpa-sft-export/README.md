# ccpa-sft-export

Convert [claude-code-parity-apr](../../../claude-code-parity-apr) **teacher.stream.ndjson**
captures (real Claude Code trajectories) into **apr-code `<tool_call>` SFT JSONL** suitable
for `apr finetune` (entrenar `InstructSample`).

This is the SFT export pipeline for apr-code distillation — the previously-missing piece
that lets us fine-tune a small student to emit apr-code tool_calls instead of Markdown/prose.

## What it does

Walks every `teacher.stream.ndjson` under the CCPA `evidence/` tree (138 captures, 40 unique
fixtures, 6,569 raw Claude Code tool_use calls) and:

1. **Remaps** the Anthropic-native tool schema to the apr-code schema:
   | Anthropic | apr-code  | field remap |
   |-----------|-----------|-------------|
   | `Read`    | `file_read`  | `file_path`→`path` |
   | `Write`   | `file_write` | `file_path`→`path`, `content` |
   | `Edit`    | `file_edit`  | `file_path`→`path`, `old_string`→`old`, `new_string`→`new` |
   | `Bash`    | `shell`      | `command` |
   | `Grep`    | `grep`       | `pattern`, `path` |
   | `Glob`    | `glob`       | `pattern` |
   (Task*/ToolSearch/Agent/AskUserQuestion have no apr-code equivalent → dropped: 127 calls.)

2. Emits each assistant `tool_use` turn as an `entrenar` `InstructSample`:
   - `system`      = apr-code `CODE_SYSTEM_PROMPT`
   - `instruction` = running observation transcript (prior tool_calls + tool_results)
   - `response`    = the literal `<tool_call>{"name":..,"input":..}</tool_call>` envelope

   (`InstructSample` has no tool_calls field — the response *string* IS the tool_call JSON.)

## Usage

```bash
cargo build --release
TARGET=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')
BIN=$TARGET/release/ccpa-sft-export

# Stratified spike set (recommended): 40 samples per apr-code tool, with context
$BIN --balanced --per-tool 40 --out ../../datasets/apr_code_sft_balanced.jsonl

# First-action-only curated set (124 samples)
$BIN --curated --out ../../datasets/apr_code_sft_curated.jsonl

# Full corpus (5,773 samples)
$BIN --full --out apr_code_sft_full.jsonl
```

## Corpus stats (measured)

| Mode      | Samples | Notes |
|-----------|---------|-------|
| `--balanced --per-tool 40` | 200 | 40 each: file_read/file_edit/file_write/shell/grep; 184 with context |
| `--curated` | 124 | one first-action tool_call per trajectory, deduped |
| `--full`    | 5,773 | every remappable turn (6,442 remapped − 669 dedup) |

Raw corpus: 138 streams, 6,569 tool_use calls (Bash 4190, Write 939, Edit 599, Read 579,
Grep 135, + 132 unmappable). 100% of emitted responses are valid parseable apr-code tool_calls.
