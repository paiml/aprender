# PMAT-3537 — receipt (ONT-001 row ONT-6b)

**Ticket:** PMAT-3537 (aprender#3537) — ONT-6b: a kind-less contract that fails a kernel-only rule says `no metadata.kind, judged kernel by default` on the first error the default caused. **Kind:** code.
**Row:** ONT-001 v4.9 §5 **ONT-6b**, `docs/specifications/paiml-ontology.md` on paiml/infra main; depends_on ONT-6 (bound); K̂ 30 [U]; source infra#751 (apex-da's measurement, 2026-09-19).
**Branch:** `PMAT-3537-ont6b-kind-default` from `origin/main` at `338f1d49b` (release 0.68.2), in a worktree of the persistent aprender clone; `~/src/aprender` untouched.

## The defect is an absent distinction, not a missing message

`ContractKind` derives `Default = Kernel` and `Metadata::kind` carries `#[serde(default)]`, so by the time anything can read `Contract::kind()` the difference between *declared kernel* and *defaulted to kernel* is gone. 666 of this corpus's 1,298 contracts declare no kind at all and are judged by the kernel rules because of that default.

Measured on this branch's base, pv 0.68.1 — `tests/fixtures/ont/kind-default/kindless-failing.yaml` and `declared-kernel-failing.yaml`, which differ by one line:

```
[ERROR] SCHEMA-003: equations must contain at least one equation
[ERROR] PROVABILITY-001: Kernel contract has no proof_obligations
```

Byte-identical, both files. Real witness: rmedia `crates/rmedia-core/contracts/dialogue-v1.yaml`, kind-less, fails `PROVABILITY-001: Kernel contract has no kani_harnesses` with nothing naming the default.

## Change, all of the diff

- `schema/types.rs`: `Contract` gains `#[serde(skip)] pub kind_declared: bool`, a parse artifact beside `unknown_top_level_keys`. `false` is the safe default for a `Contract` built in code: it only ever adds an explanation to an error that already fired.
- `schema/parser.rs`: `parse_contract_str` sets it from a new `kind_is_declared(yaml)`, which reads the RAW mapping — serde has already substituted `Kernel` by then, and that substitution is what has to be observed. Absent or non-mapping `metadata` → `false`. It deserializes a `KindProbe` whose only field is `metadata: BTreeMap<String, IgnoredAny>`, so every OTHER top-level value is drained rather than built: deserializing the whole document into a `serde_yaml::Value` map would FAIL on `contracts/apr-cli-commands-v1.yaml` (duplicate `subcommands:` under `commands:`) and report "no kind declared" about a file that declares one. A unit test carries that shape, with an anti-vacuity assertion that a whole-file `Value` parse really does refuse it.
- `schema/validator.rs`: `validate_contract` takes `let before = violations.len();` before the kernel-only branch and, when `!contract.kind_declared`, calls the new `explain_kind_default(&mut violations[before..])`. That slice is by construction the findings the default is responsible for, so no rule id is matched and the printer is untouched. It decorates the **first `Severity::Error`**, once — a warning is not a verdict, and repeating the sentence on all six kernel rules would bury the file's actual problem under its own explanation. New `pub const KIND_DEFAULT_EXPLANATION`.
- Four fixtures under `tests/fixtures/ont/kind-default/`, one per arm; `crates/aprender-contracts-cli/tests/ont6b_kind_default.rs` (six legs); `ci/explicit-test-commands.d/344-aprender-contracts-cli-ont6b-kind-default.cmd`. No `ci.yml` change; 344 is free (339 #3516, 340 main, 341 #3530, 342 ONT-4c1, 343 ONT-4b2).

## The four arms, measured

| fixture | rc | occurrences of the explanation |
|---|---|---|
| `kindless-failing.yaml` | 1 | 1, on the first `[ERROR]` (`SCHEMA-003`) |
| `declared-kernel-failing.yaml` | 1 | 0 |
| `kindless-passing.yaml` | 0 | 0 |
| `pattern.yaml` | 0 | 0 |

## Mutation — and why each mutant's red is distinguishable

Every mutation below **asserts its own edit applied** before the tests are run: a `perl` substitution that matches nothing exits 0, so an unapplied mutation is indistinguishable from one the tests fail to catch. The first attempt at this table produced four "all six legs pass" results that were entirely unapplied edits — recorded here because that is the exact vacuity shape this row's own review is about.

| mutation | legs that go RED |
|---|---|
| drop the text (`if false && !contract.kind_declared`) | `kindless_failing_names_the_default_on_its_first_error`, `the_explanation_is_printed_once_not_once_per_kernel_rule` |
| decorate even when the kind was declared (`if true`) | `declared_kernel_is_judged_by_the_kind_it_declares…`, `the_two_failing_fixtures_differ_only_by_the_explanation` |
| decorate the LAST error instead of the first (`.filter(…).last()`) | `kindless_failing_names_the_default_on_its_first_error` |
| decorate warnings too (`.find(\|_\| true)`) | `kindless_but_complete_passes_and_never_mentions_kind` |

Each mutant names a **different** set of legs and the unmutated tree is green in the same sequence, so a mutant's red is distinguishable from a broken fixture's red (the corollary infra-dd stated on 2026-09-20: a control that only demands red passes whenever the harness breaks).

## R-23 dogfood

- **aprender**, this branch's `pv` vs `pv` built from the persistent main worktree at `daad9f7c5`, over all **1300** tracked `contracts/*.yaml`: **0 files differ**, and **0** carry the explanation — no tracked aprender contract is a kind-less REJECT, so the change is inert on the corpus exactly as the row predicts ("the 666 passing kind-less contracts print exactly what main's pv prints"). Comparand named because a differential over an unnamed base is not a measurement.
- **rmedia**, read-only `git archive` export of `~/src/rmedia` at `3ff2f4d`, `crates/rmedia-core/contracts/dialogue-v1.yaml` (kind-less, `grep -c '^\s*kind:'` = 0):
  - main's pv: `[ERROR] PROVABILITY-001: Kernel contract has no kani_harnesses`, exit 1
  - this pv: `[ERROR] PROVABILITY-001: Kernel contract has no kani_harnesses (no metadata.kind, judged kernel by default)`, exit 1
  A verdict, not a decline — `validate` has no arming, so R-2 does not apply.

verification:
  cmd=cargo test -p aprender-contracts-cli --test ont6b_kind_default  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3537-receipt.md  sha256=0   # 6 passed; 0 failed
  cmd=cargo test -p aprender-contracts --lib  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3537-receipt.md  sha256=0   # 1666 passed; 0 failed; 5 ignored (adds the kind_is_declared regression test)
  cmd=pv validate over the four fixtures (both binaries)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3537-receipt.md  sha256=0   # rc 1/1/0/0, explanation count 1/0/0/0; on main's pv the first two are byte-identical
  cmd=four planted mutations, each asserting its own match count, then `cargo test --test ont6b_kind_default`  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3537-receipt.md  sha256=0   # every mutant RED on a distinct pair of legs; tree restored and green
  cmd=differential over 1300 contracts, this pv vs main@daad9f7c5's pv  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3537-receipt.md  sha256=0   # 0 differing, 0 carrying the explanation
  cmd=pv validate <rmedia export>/crates/rmedia-core/contracts/dialogue-v1.yaml  claimed_exit=1  rerun_exit=1  log_path=docs/audits/impl-PMAT-3537-receipt.md  sha256=0   # the explanation on the first error; main's pv prints the same line without it

## Deviations, named

- Executed direct (one behaviour, three files, one test table); review is the quorum. `goal.sh set` not run.
- The infra-side carrier (ledger row + `scripts/ont/done_when/ONT-6b.sh` probe, already on infra main since v4.9) is a SEPARATE PR against paiml/infra, opened once this merges and its squash sha is known. This PR binds nothing on its own.
- The ticket is `--github-issue 3537` (a new aprender issue), not infra#751: `--github-issue 751` derives PMAT-751, which aprender's roadmap already uses for an unrelated SATD entry. The row's infra source is named in the issue body instead.
