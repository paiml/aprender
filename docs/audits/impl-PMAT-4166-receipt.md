# PMAT-4166 — PVL-001 EV-11 `pv lint` ratchets (theorem pairing + depends_on): implementation receipt

## Identity

| | |
|---|---|
| Ticket | PMAT-4166, `kind:code` — "PVL-11: pv lint ratchets — theorem pairing and depends_on" |
| Row | paiml/infra PVL-001 EV-11 @00553b0b (probe and accept quoted in `crates/aprender-contracts-cli/tests/ev11_lint_ratchets.rs`) |
| Branch | `PMAT-4166-pvl-11-lint-ratchets`, stacked on `origin/PMAT-4076-ont7-valid-under` @`2b4eba725` (ONT-7, not yet on main) |
| Reviewed head | `708d2ae81` |
| Assigned by | cop aprender-cf |
| Author | aprender-98, claude-opus-5-5 |
| Verdict | **DONE at the receipt.** Batching is the default, so it is not armed and has no PR; the cop folds it. The receipt commit adds only this file and `docs/audits/quorum-PMAT-4166.json` on top of the judged head |

## What landed

- **`pv lint --gate theorem-pairing` (PV-RAT-001).** A Lean theorem module is a `.lean` file under `<lean base>/ProvableContracts/Theorems/`. It is paired when its FULL dotted module name appears, on identifier boundaries, in a `.md` file under `book/` or `crates/aprender-contracts-staging/book/`. The debt `unpaired_theorem_modules` may not rise.
- **`pv lint --gate depends-on-present` (PV-RAT-002).** It counts contracts of kernel kind under `kind()`, which already reads a `registry: true` kernel as Registry, whose `metadata.depends_on` is empty. The debt `contracts_without_depends_on` may not rise.
- **Gates 14 and 15 of every `pv lint` run** (R-8: computed everywhere, armed per repo). They are not armed here: `armed_gates` in `contracts/lint-baseline.json` is unchanged.
- **`--gate` is repeatable.** The verdict is the meet of the named gates (refusal > reject > decline > pass), and an unknown name is refused before any gate runs.
- **The gates never write.** A missing baseline gives `Unknown(Report)` with the count printed and exit 2, never a pass. Having no Lean base, no book page or no kernel contract is a decline, since zero measured is a decline (ONT R-2).
- **`scripts/lint_ratchet.sh` / `make lint-ratchet` is the only writer.** It lowers a number, records an absent one, and refuses a rise (exit 1). It edits line by line, so `armed_gates` stays on one line for `check_ont_ratchet.sh`. `check_ont_ratchet.sh --write` carries the new keys.
- **Baseline** `contracts/lint-baseline.json`: `command: "make lint-ratchet"`, `unpaired_theorem_modules: 130`, `contracts_without_depends_on: 278`.
- **Gate contract** `contracts/pvl-lint-ratchets-v1.yaml`: RAT-INV-001..004, FALSIFY-RAT-001..006, `formal:` written in Σ glyphs.
- **CI wiring:** `ci/explicit-test-commands.d/456-aprender-contracts-cli-ev11-lint-ratchets.cmd`, plus `scripts/tree_reader_tests.txt`.

## The probe, on the real corpus at the reviewed head

The probe passes: both gates `Pass` with exit 0, and `git diff --exit-code contracts/lint-baseline.json` is clean afterwards (test `the_probe_passes_on_the_real_corpus_and_writes_nothing`).

**The measured counts differ from the spec row's figures.** The spec's 5/165 and 110/1818 are not what this corpus measures:

| Measure | Count |
|---|---|
| Theorem modules | 131 |
| Paired | 1 |
| Unpaired | 130 |
| Contracts checked | 1830 |
| Kernel-kind contracts | 387 |
| Kernel-kind without `depends_on` | 278 |

The book mentions the file STEM of 80 of the 165 `.lean` files, mostly as ordinary words, so a stem rule would pair by accident. The strict rule is disclosed in the module doc. The sonnet seat re-derived 131/1 independently with `find`.

## Verification (author-run at 708d2ae81; mutants at 1c28b4c38)

| Check | Result |
|---|---|
| `make gate` | exit 0 |
| aprender-contracts `--lib` | 1732 passed, 0 failed |
| aprender-contracts-cli, all 23 test binaries | 385 passed, 0 failed (private TMPDIR, see Open) |
| aprender-core `--test readme_contract` | 15/0 |
| fmt; clippy `-D warnings` (lib + tests, both crates); bashrs | clean / 0 errors |
| `pv validate contracts/pvl-lint-ratchets-v1.yaml`; `pv extract --check` | clean |
| guards | check_tree_reader_tests, check_explicit_test_commands, check_complexity_ratchet, check_readme_claims, check_ont_ratchet (30/30), check_dogfood_coverage |
| `scripts/lint_ratchet.sh --self-test` | 12/12 |

**MUST-RED, measured.** Each mutant was planted on the committed tree and restored from HEAD. Each fails its witness, with 0 compile errors:

| Mutant | What it breaks | Witness that fails |
|---|---|---|
| M1 | ratchet off | rise tests, lib + CLI |
| M2 | loose boundary | boundary tests |
| M4 | gates not pushed into the full run | real-corpus / every-gate-verdict tests |
| M5 | no baseline passes | no-baseline tests, lib + CLI |
| M6 | pairing by stem | many |
| M9 | kernel filter off | depends tests + probe |
| M7 | `FOREIGN_TOP_KEYS` reduced | `check_ont_ratchet.sh` self-test rc=1 |
| M8 | a rise is accepted | `lint_ratchet.sh` self-test rc=1 |

M3 (the registry clause removed) was an **equivalent mutant**. It looked red only because of the Σ failure below, since `kind()` already excludes registries. The clause was removed as redundant in `1c28b4c38`, with a comment saying why.

## Quorum — `docs/audits/quorum-PMAT-4166.json`

The rule is the operator's, verbatim, 2026-09-24: "we are STILL draining agy too quick.  we need to switch default to sonnet, agy, haiku". No seat is the author id, and fable was not used. The 429 fallback ("if gemini returns 429 I want fallback sonnet + haiku + opus5.5") was not needed, because gemini answered. **Not degraded.**

| Seat | Model (measured) | Verdict | Findings |
|---|---|---|---|
| agy, width 1 | gemini-3.1-pro-high | PASS | 0 |
| Claude Code | claude-sonnet-5 (`modelUsage`) | PASS | 0 |
| Claude Code | claude-haiku-4-5 (`modelUsage`) | PASS | 0 |

- **Agreed: yes, 3/3, on head `708d2ae81`, base `2b4eba725`.** `diff_sha256` = `41943551…af873e` for all seats.
- **One brief.** `quorum-review.sh` deletes its brief on exit, and this run recorded `prompt_sha256: null`. The brief was regenerated with `--brief-only` at the same head, base and ticket, through a scratch copy patched with one line that copies the brief before exit. It matches on two counts:
  - It is the same 76556 bytes the agy run recorded.
  - Its embedded diff hashes to the agy artifact's `diff_sha256` under the judged-diff rule.

  The Claude seats read that file: sha256 `a6436565…e8cd0c67`.
- **Sonnet seat denials.** It had 10 permission denials (`cargo test`, shell loops). The lane allow-list is read-only, so it verified by reading and with `find`/`grep`. Its verdict does not rest on a test it ran.

## Open, non-blocking

- `/tmp/scripts/contract_duplicate_stem_baseline.txt` is a stray file from another session. It breaks the aprender-contracts-cli tempdir tests when `TMPDIR=/tmp`, which is a hermeticity defect to file. The CLI suite was run with a private TMPDIR.
- The first commit `02567b5be` failed the Σ gate: `⇔` is undeclared, and `=` pushed `formal_prose` from 1464 to 1465. It was fixed in `1c28b4c38` (`⟺`, and `¬(… ≠ …)`), and the reviewed head is clean.
- **Stacking.** This branch rides on ONT-7 (PMAT-4076). ONT-8 (#4077, rmedia-82) may also stack on ONT-7. The gate numbering, the count in `mod_tests.rs` and the `not_armed` list in `ont6_lint_verdict.rs` are the shared conflict points, and rmedia-82 has been told.
