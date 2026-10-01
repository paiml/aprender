# 05-14 — the fail-closed epsilon verdict, proven in its real scope

Verification Discipline rule 4: extending a guard's scope requires re-proving it IN the new
scope. Task 1's proof is over synthetic aggregates. This file records the same guard proven over
the real committed measurements, by RUNNING it — never by inspection.

Host: macOS/arm64, worktree `agent-a6b92c30c35610e1b`, branch `worktree-agent-a6b92c30c35610e1b`,
base `b3386b063`. Guard commit `e890cccde`.

---

## Which check is PERMANENT and which is a moment in time

This distinction is the point of the section, not a caveat on it.

**DURABLE — re-runnable forever, no contract dependency, cannot flip on what 05-03 lands:**

1. The nine default-suite tests filtered by `epsilon_basis`. No `--ignored`, no 86.7 MB checkout,
   no training; the aggregates are ordinary `f64` inputs. Two-sided in both directions that
   matter (a declaration moves the verdict; the identical window on a gated class refuses).
2. Of those, `evidence_epsilon_basis_real_four_cell_aggregates_refuse_under_no_resolved_table` is
   the permanent real-data guard. It carries 05-01's MEASURED four-cell aggregates as literals
   and derives them under
   `no-frozen-table-by-construction-05-14@0000000|seeds=13|cells=s64e1b16` — an architecture
   component that is not a real encoder fingerprint and never will be, so `table_for` resolves
   `None` for it whatever 05-03 freezes. The test asserts that `None` explicitly, so if the id
   ever DID resolve the guard fails loudly rather than silently ceasing to be red.

**STATE-DEPENDENT — a control on today's state, deliberately not durable:**

3. The `--ignored` four-cell combine below. It asserts RED only while the production regime is
   uncalibrated; once 05-03 commits a production table the same command must go GREEN. Its
   verify block therefore DERIVES the expected status from the derivation's own
   `REGIME TABLE FOR THE DERIVED REGIME:` line — the lookup that actually decides the verdict —
   rather than hardcoding RED or proxying the precondition from contract or source text. A
   contract-text proxy would false-flip on the option-F halt branch, where 05-03 still lands
   MEASURED/COVERED prose naming all ten contracted seeds while no production table exists; the
   block would then fail forever, which is the very defect this restructuring avoids. The block
   also fails loudly if the line is absent, so a naming drift cannot silently pick a default.

---

## RED CONTROL — a behaviour delta on IDENTICAL input

Byte for byte the invocation plan 05-01 recorded exiting `rc=0` (FINDING 3 / FINDING 6):

```bash
APRENDER_CALIBRATION_STORE=".planning/phases/05-benchmark-and-claims-gate/calibration-store" \
APRENDER_CALIBRATION_COMBINE="s8:13,s8:31,s8:53,s64:13" \
CARGO_INCREMENTAL=0 cargo test --release -p aprender-train --lib --features setfit \
  production_calibration -- --ignored --nocapture > /tmp/combine.log 2>&1; rc=$?
```

| | 05-01 (recorded) | 05-14 (measured here) |
|---|---|---|
| exit status | **`rc=0`** | **`rc=101`** |
| `PASSES RUN` | 12 | 12 |
| cells loaded | 4 (`s8:13,s8:31,s8:53,s64:13`) | 4, same |
| classes printing `EMPTY` | 5 | 5, same numbers |
| verdict | none — `supports_margin` only REPORTED | typed refusal, panics after the report is written |

Status captured on its own line (`cmd > log 2>&1; rc=$?`), never through a pipe.
**No SKIP occurred in either direction** — the pinned checkout is present at
`~/.cache/aprender/minilm-l6-v2-1110a243` (`full_manifest.json` + `model.safetensors`, 86.7 MB),
and the log shows twelve timing rows and a full basis table, so the refusal is over the real
basis rather than an early abort.

The mechanism line the verify block keys on:

```text
REGIME TABLE FOR THE DERIVED REGIME: absent — no frozen table covers
`minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13,31,53|cells=s64e1b16,s8e1b16`,
so EVERY class is required to carry a legal window
```

The derived basis, verbatim (note `gating` is the column this plan added):

```text
CROSS-CELL EPSILON BASIS — PROVISIONAL, NOT THE FROZEN EPSILON.
Derived from 4 of 6 boundary cells. MISSING: s64:31, s64:53.
class                worst_ctrl    worst_nnull   best_real     10x_lower     10x_upper     noise_floor   eps/noise     nnull_moved   window        median/min    gating
embedding            0.000e0       1.553e-4      1.813e-3      1.553e-3      1.813e-4      1.779e-6      n/a           true          EMPTY 8.56x   3.4e0         required
layer_norm_weight    0.000e0       4.802e-7      1.891e-4      4.802e-6      1.891e-5      5.960e-8      3.17e2        false         EXISTS        3.2e0         required
layer_norm_bias      0.000e0       9.844e-5      7.112e-4      9.844e-4      7.112e-5      5.960e-8      n/a           true          EMPTY 13.84x  8.5e0         required
projection_weight    0.000e0       1.051e-4      1.231e-3      1.051e-3      1.231e-4      5.960e-8      n/a           true          EMPTY 8.54x   4.5e0         required
projection_bias      0.000e0       1.101e-4      3.447e-4      1.101e-3      3.447e-5      5.960e-8      n/a           true          EMPTY 31.94x  1.5e1         required
attention_key_bias   0.000e0       9.278e-9      1.714e-7      9.278e-8      1.714e-8      1.133e-9      n/a           true          EMPTY 5.41x   2.4e0         required
```

And the refusal that now accompanies it, quoted in full:

```text
EPSILON BASIS COLLAPSED: 5 of 6 classes have NO legal epsilon window, and this regime requires every one of them
  regime: minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13,31,53|cells=s64e1b16,s8e1b16
  frozen table: absent — every class is required, because a class the contract has not recorded cannot be exempted by omission
  embedding            lower 1.553e-3 EXCEEDS upper 1.813e-4 by 8.56x
  layer_norm_bias      lower 9.844e-4 EXCEEDS upper 7.112e-5 by 13.84x
  projection_weight    lower 1.051e-3 EXCEEDS upper 1.231e-4 by 8.54x
  projection_bias      lower 1.101e-3 EXCEEDS upper 3.447e-5 by 31.94x
  attention_key_bias   lower 9.278e-8 EXCEEDS upper 1.714e-8 by 5.41x
  A run whose epsilon basis is empty for a gated class is not evidence that an epsilon basis exists (D-18). Do not freeze an epsilon for a class listed above.
```

Every exceed factor reproduces 05-01's FINDING 1 table to the digit (8.56 / 13.84 / 8.54 / 31.94
/ 5.41), and `layer_norm_weight` is again the sole survivor. The report is written to its
destination BEFORE the panic — a derivation whose numbers cannot be read is worse than one that
fails loudly.

---

## GREEN CONTROL on real data — the fixture matrix

```bash
CARGO_INCREMENTAL=0 cargo test --release -p aprender-train --lib --features setfit \
  calibration_matrix_epsilon_basis -- --ignored --nocapture   # rc=0
```

```text
REGIME TABLE FOR THE DERIVED REGIME: resolved for
`minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4`
— gated classes: [embedding, layer_norm_weight, layer_norm_bias, projection_weight, projection_bias]

class                ... window        median/min    gating
embedding            ... EXISTS        3.7e0         required
layer_norm_weight    ... EXISTS        1.9e0         required
layer_norm_bias      ... EXISTS        2.6e0         required
projection_weight    ... EXISTS        2.7e0         required
projection_bias      ... EXISTS        3.4e0         required
attention_key_bias   ... EXISTS        1.8e0         declared-ungated

Every class above has a non-empty window (10x_lower < 10x_upper).
```

Verdict GREEN, `rc=0`. All five gated classes legal. `attention_key_bias` is present and carries
the `declared-ungated` annotation, and the exclusion is traceable to a mechanism — the resolved
table's own gated set is printed on the line above it, not asserted in prose.

**Neither documented refusal branch fired.** The fixture regime id RESOLVED through `table_for`,
so the strict branch never applied, and no class the fixture's frozen table GATES was refused.
The predicate was not adjusted; nothing was loosened to obtain this green.

**Stated so it is not over-read:** in the fixture regime `attention_key_bias`'s window happens to
EXIST, so this run does not exercise the declaration CHANGING a verdict. That two-sided
demonstration is `evidence_epsilon_basis_declared_ungated_moves_the_verdict_without_removing_the_row`
in the default suite, which places one identical empty window on the declared-ungated class
(success, row retained and annotated) and on a gated class (refusal), and additionally derives
the ungated case under a no-table regime (refusal — an unrecorded class is never ungated by
default).

---

## Isolation and integrity at the finished state

| check | result |
|---|---|
| `setfit::` lib suite | `322 passed; 0 failed; 3 ignored` — 05-02 baseline **313** + this plan's **9** new tests, exactly |
| committed store digests | `committed calibration store: 12 of 12 pairs verified` (pair count asserted BEFORE any agreement check) |
| `git status --short` on the store | empty |
| `git status --short` at the finished state | empty (evidence.rs committed) |
| serialization surface | `git diff -U0 … \| grep '^-' \| grep -c 'serde\|to_canonical_bytes'` = **0** |
| strict-window comparisons in `evidence.rs` | **1** (was 2) |
| `fn epsilon_basis` definitions | **1** |
| `cargo clippy -p aprender-train --lib --features setfit --all-targets` | zero findings in `evidence.rs` |
| `cargo check -p aprender-train --lib --features setfit` | zero new warnings from `aprender-train` |
| `rustfmt --check` on `evidence.rs` | clean |

---

## MEASURED ENVIRONMENT HAZARD — `cargo test` output is rewritten on this host

Recorded because both verify blocks in the plan grep for a line this host does not emit, and
because it is the same artifact family as 05-01's `git log --oneline` rewrite that reported seven
existing commits as MISSING.

The `rtk` hook rewrites `cargo test` output into a one-line summary:

```text
cargo test: 313 passed, 3 ignored, 7661 filtered out (1 suite, 37.59s)
```

There is **no `test result: ok. N passed` line anywhere in the captured log** — measured, not
assumed: `grep -c "test result" /tmp/baseline-setfit.log` returned 0 on a run that exited 0. The
plan's `grep -E 'test result: ok\. [1-9][0-9]* passed' … || exit 1` would therefore fail on a
GREEN run, and the count floor it feeds would read 0.

`rtk proxy cargo test …` yields the raw libtest stream, with the `test result:` line intact.
**Every `cargo test` in both verify blocks was run through `rtk proxy`.** Same commands, same
assertions, unrewritten stream. This matches the standing Phase 2 ruling that `git status
--porcelain` assertions must go through `rtk proxy` for the same reason.

The plan's own counting rule was obeyed throughout: `grep -c` in COMMAND SUBSTITUTION, never via
a redirected file. Confirmed on this host — the redirected form is decorated (`2 matches in 1F:
…`), the substituted form yields a bare integer.
