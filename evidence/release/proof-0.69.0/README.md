# `release-readiness-v1` on v0.69.0's own evidence (aprender#3715 done_when 5)

The question: had this shape been the gate at 0.69.0's T-1, would it have stopped the train, and would it have
named the leaks that actually shipped to the publish step? **Yes: RED, 795 findings.** Every leak the issue lists
is named. This is `gate.json` in this directory.

## What went in: 0.69.0's measurements, translated, none invented

| file | from | measured |
|---|---|---|
| `models/{lambda,gx10}.json` | `rel-069-state/ladder/0.69.0/<host>.json` (the v1 ladder receipt, apr 0.69.0 (225b2a9ab)), `q4k-sweep-<host>.jsonl` (the Q4_K sweep at the same binary), and the listing of the ladder contract's inventory dirs on each host (2026-09-21T16:20:36Z, lambda locally, gx10 over the operator-authorized SSH) | yes, by the 0.69.0 train and the sweep |
| `dogfood-receipt.json` | `rel-069-state/preflight-wt/.dogfood/receipt-20260921T121638Z.json`, byte for byte: the receipt 0.69.0's T-1 "inherited" its GO from (#3708) | yes |
| `translate.py` | the translation itself. Its docstring is the rule set | — |

The rules `translate.py` follows, so that the translation cannot manufacture a pass:
- The **inventory** is what each host holds. A `sha256` is carried **only** where the 0.69.0 ladder hashed that exact file on that host. Everything else has no hash, and pv names it `unmeasuredModel`: the release never identified the file, so it never proved it.
- The ladder row becomes the `(run, think-off, 4k)` row, with `verdict` = its green bit and `fallback` from `backends.cuda`. `prompt_tokens` and `answer_chars` are **absent** because 0.69.0 ran golden prompts and recorded neither.
- A sweep row for a file with no measured hash is carried **unkeyed**: pv names it `unkeyedRow`, with its verdict.
- **No row** exists for chat, serve, code, think-on, or any other rung, because 0.69.0 never ran them. There is **no** kernel-diff or tokenizer-parity receipt, because neither existed.

## What the shape says

```
pv lint contracts --gate shapes --shape release-readiness-v1 \
   --release-version 0.69.0 --release-commit 225b2a9abbd171b3249794722f8da5fc8027e565 \
   --receipts evidence/release/proof-0.69.0/models \
   --kernel-receipts <empty dir> --tokenizer-receipts <empty dir> \
   --dogfood-receipt evidence/release/proof-0.69.0/dogfood-receipt.json
→ exit 1, verdict Fail, 795 findings; cells 768 (16 with a row), 16 models, 8 tokenizer cells
```

| leak (#3715 done_when 5) | how the shape names it |
|---|---|
| **qwen3-8b on lambda** | `release-cell/0.69.0/lambda/Qwen3-8B-Q4_K_M.gguf/run/think-off/4k`: `verdict: "fail"`. It is the only ladder model with a failing verdict; the same file on gx10 does not fail |
| **qwen2.5-coder on lambda** (0.5b, 7b) | `release-host/0.69.0/lambda`: `unmeasuredModel` for both files, and `unkeyedRow … verdict=fail` for the sweep rows (GPU gibberish, and a fallback to CPU with rc 14) |
| **qwen3moe on both hosts** | `unmeasuredModel` on lambda (`Qwen3-30B-A3B-Instruct-2507`, `Qwen3-Coder-30B-A3B-Instruct`, the first with `unkeyedRow … verdict=fail`: no gates emitted) and on gx10 (`Qwen3-Coder-30B-A3B-Instruct`, which the gx10 sweep never reached) |
| **chat / serve / code absent** | 752 of 768 cells have no row (`minCount`): for each of the 16 (host, model) pairs, 47 of 48 cells are missing, meaning everything except the one golden `run` |
| **the inherited dogfood** (#3708) | `release-subject/0.69.0`: the receipt says `verdict: "NO-GO"` for commit `5f9bd9344` at version `0.68.2`. It is not GO and not fresh. The autopilot's inherited "GO" was derived from it by excluding its version row |
| no kernel differential, no tokenizer parity | both hosts `kernelReceipt minCount 1`, every model's `release-tokenizer/…` cell missing |

Every ladder model's one golden row also fails `contextMet` and `answered`, because 0.69.0 never sized a
prompt or measured an answer. That is the "overdo it" bar (#3710) applied to a release that predates it, and it
is honest: nothing measured at 4k tokens was ever recorded.

**What is NOT shown here:** that 0.69.1 passes. That is the other half of done_when 5 ("on 0.69.1 it passes with
N/N cells named"). It needs the producer (#3712's cells[], inventory arithmetic, kernel and tokenizer receipts)
and the fixes to land. The gate will report it at T-1 through `release.cell_names`.
