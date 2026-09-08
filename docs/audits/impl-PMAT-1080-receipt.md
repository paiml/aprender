---
status: complete
ticket: PMAT-1080
row: F-1
issue: 3022
also_closes: 3024
epic: 2873
priority: P0
branch: agent/F-1
kind: code
model: claude-opus-5[1m]
host: noah-Lambda-Vector
---
# impl receipt — PMAT-1080 (PP-066 row F-1, #3022 / #3024): the model you named is the model that answers

`apr chat` silently substituted its built-in toy demo model for a sharded SafeTensors
model — no error, exit 0, zero tokens, the real model's path in the banner. The nightly
gate that exists to catch exactly this was green the same night, because it never calls
`apr chat` and never names a sharded model.

## The defect, reproduced on `origin/main` before anything was changed

Binary `apr 0.65.2 (c04eda87d)` — built in this worktree from the tip of `main`. Model: a
**2-shard fixture derived offline** from `~/models/qwen2.5-coder-0.5b-instruct-safetensors`
(no download; see "the fixture" below).

| cell | `origin/main` | this branch |
|---|---|---|
| single-file × `apr run` | ok | ok |
| single-file × `apr chat` | ok — `Loaded SafeTensors format in 0.50s (988.1 MB)` | ok |
| sharded × `apr run` | **ok** — `Output: 2 + 2 equals 4.` | ok |
| sharded × `apr chat` | **`Chat Demo (Tiny Model)` · `Loaded Demo format in 0.00s (0.0 MB)` · `[0 tokens in 0.0s]` · exit 0** | **`Model Chat (Sharded SafeTensors)` · `Loaded Sharded SafeTensors format in 0.00s (988.1 MB)` · `[12 tokens in 3.7s = 3.2 tok/s]` · `Assistant: 2 + 2 equals 4.`** |

Records: `evidence/format-honesty/{before,after}/`, verdicts in `before/VERDICT.txt`.

**One correction to the report.** #3022's headline and the session note both read as though
the sharded path were broken generally. Measured here, **`apr run` on a sharded index
works** — row 3 above, on `main`, unmodified. The defect is `apr chat` alone. Whatever
failed for the reporter on `apr run` with a real 7B pull is not reproduced by this
fixture and stays `[U]`; the matrix now covers that cell on every run, so a regression
there is caught rather than inferred.

## Root cause, and why one arm would not have been the fix

`Path::extension()` returns the LAST dot-segment: on `model.safetensors.index.json` it is
`Some("json")`, which matched no arm of `detect_format` and fell through to
`ModelFormat::Demo`. That is the instance. The class is that **two independent decisions**
existed: `print_welcome_banner` asked the PATH, `ChatSession::new` asked the first eight
BYTES, nothing compared them, and so "banner says SafeTensors, session loads Demo" was a
reachable state with no falsifier anywhere in the tree.

The fix is therefore structural, not an added arm:

- `resolve_chat_format(path)` is the ONE decision — suffix first (before `extension()`),
  then the magic bytes of an odd-named file, then a **refusal**. `Demo` is not an outcome
  for a path that exists, and `resolve_chat_model` has already proven existence.
- The refusal carries `CliError::ModelLoadFailed` — **exit 6, read from `error.rs`, never
  typed** — and names the file, the recognised formats and #3022.
- Every consumer (banner, tokenizer, architecture detection, generator) is handed that one
  value, so the banner can no longer name a format the loader did not use.
- `ModelFormat::ShardedSafeTensors` reaches the transformer through the same three calls
  `apr run` already makes (`infer/mod_log_transformer_eos.rs:116`):
  `load_from_index` → `SafetensorsConfig::load_from_sibling` → `convert_sharded`.
- The load line reports `index.metadata.total_size` for a sharded model. Printing the
  manifest's own ~20 KB would have been the same lie under a different name: `0.0 MB`.

```
$ apr chat /tmp/notes.txt
error: Model load failed: /tmp/notes.txt is not a model format apr chat can load.
  Recognised: .apr, .gguf, .safetensors, and a sharded SafeTensors index (*.safetensors.index.json).
  apr chat will not substitute its built-in demo model for a file you named (#3022).
$ echo $?
6
```

## The fixture: a sharded model with no download (#3024's "deliberately kept small")

`scripts/make_sharded_safetensors.py` splits one `model.safetensors` into N shards plus a
HuggingFace index, offline, deterministically, with no dependencies. A shard boundary is a
property of the index and the file split, not of the parameter count: two shards of a 0.5B
exercise `weight_map`, the per-shard header rewrite and the cross-shard lookup exactly as
four shards of a 7B do — at 988 MB and a few seconds instead of a multi-gigabyte nightly
fetch. Its own case table (7 rows) proves every tensor's bytes survive the split, that the
index names exactly N shards, that a second run is byte-identical, and that `--shards 1`
and a non-safetensors input are refused — because a fixture that is not actually sharded
would make a green matrix prove nothing.

## Falsifiers (I3) — mutation RED → GREEN, run locally, both mutants

| mutant | rows that turned RED | restored |
|---|---|---|
| `detect_format`'s `.safetensors.index.json` arm never matches | `sharded_index_resolves_to_the_sharded_format_not_demo` (3 passed / 1 failed) | 4/4 GREEN |
| `resolve_chat_format` returns `Ok(Demo)` instead of refusing | `a_truncated_file_is_refused_rather_than_read_as_a_format`, `an_existing_unrecognised_file_is_refused_not_demoted`, `resolve_never_answers_demo_for_a_file_the_user_named` (1 passed / 3 failed) | 4/4 GREEN |

And the gate's own RED leg, which is the point of #3024: the **same** matrix, the **same**
fixture, two binaries —

```
origin/main            : FAIL  sharded x apr chat   the DEMO model answered for a file the user named (#3022)   rc=1
this branch            : ok    4/4 cells                                                                        rc=0
```

The three unaffected cells stay GREEN on the defective build, so the gate discriminates
rather than merely failing.

## What runs where

| falsifier | where |
|---|---|
| FALSIFY-FMT-HONESTY-001/002 (6 unit rows) | `workspace-test` (`cargo test -p apr-cli --lib`) |
| FALSIFY-FMT-HONESTY-003 (7-row judge case table) | `ci / gate` → `guard-tree` (`check_format_command_matrix.sh --self-test`) |
| FALSIFY-FMT-HONESTY-004 (7-row splitter case table) | `ci / gate` → `guard-tree` (`make_sharded_safetensors.py --self-test`) |
| the live 4-cell matrix | a host with a model: `check_format_command_matrix.sh --matrix --apr <bin>` |

`guard_tree.sh --dry-run --no-cargo` classifies the new guard
`skipped: ... wired-with-args in ci.yml`, which is the intended state: the dispatcher must
not run it bare (it needs `--self-test` or `--matrix`), and the explicit step runs it.
`check_guards_are_wired.sh` PASS (ratcheted), `check_shell_lint_ratchet.sh` PASS
(174 scripts, 8 error lines, baseline 8).

## Not in this row

- **The nightly wiring (#3024 ask 3)** — a `qwen-story` beat or a sibling workflow running
  the live matrix against real models. That is a workflow change and belongs to the CI
  lane, filed as row F-2; this row ships the runner and its receipt so that row is a
  wiring change and nothing else.
- **Sharded GGUF** (#3024's fourth matrix cell, `merge_gguf_shards`) — no fixture builder
  exists for it and none is derived here; it stays an uncovered axis and is named in F-2.
- **`apr serve`** on a sharded index — the matrix covers `run` and `chat`; `serve` needs a
  port and a client and is F-2's to add.
- The reporter's `apr run` failure on a real 7B pull — not reproduced (row 3 above passes
  on `main`); `[U]` pending their exact command and output.
