# PMAT-3233 — ONT-1 (`pv census`) implementation receipt

row: **ONT-1** of `infra:docs/specifications/paiml-ontology.md` (ONT-001 v4.3), which inherits
PVL-001 v3 §0/§2/§3/§5. PR: paiml/aprender#3281. branch: `PMAT-3233-pv-census`.

## Why this PR is larger than "add a census subcommand"

#3281 began as the counting half of ONT-1 and was measured PARTIAL on 2026-09-15
(`infra:docs/audits/ONT-001/receipts/2026-09-15T1645Z-ONT-1-escalate.md`): six of the row's
clauses were red, one of them contradicting census.rs's own doc comment. The operator ruled
**"Extend #3281"** rather than split the row, so this PR now carries the whole ONT-1 clause set.
**All 23 changed files appear below**, each against the clause that requires it; nothing here is incidental. Round 3 lane 1 refuted an earlier version of this table that named only 15 of them, and refuted its "1841" as well: main carries 1842.

| file | ONT-1 clause that requires it |
|---|---|
| `crates/aprender-contracts-cli/src/commands/census.rs` | the subcommand itself; `--format json` keys `n_files/n_parsed/n_parse_errors`; `by_anchoring`; `timing.n_runs==5` |
| `crates/aprender-contracts-cli/src/cli.rs` | the probe is `pv census contracts/ --format json` — the flag was `--json`, which made the probe exit 2 |
| `crates/aprender-contracts-cli/src/contract_walk.rs`, `src/lib.rs` | exit vocabulary: empty dir ⇒ 2 `decline:`, 1 malformed ⇒ 1 `reject:` (was 1/`error:` and 0/counted-as-valid) |
| `crates/aprender-contracts/src/schema/parser.rs`, `src/lint/gates.rs` | the corpus definition: `binding.yaml`/`external-corpora.yaml` and `quarantine/` are not contracts |
| `contracts/census.json` | "tracked; `make contracts` asserts `git diff --exit-code` on it" |
| `contracts/external-corpora.yaml` | "the archived corpus is declared, not counted" |
| **`scripts/lint-provenance.sh`** | **R-10: every mark carries provenance.** Interim mark linter; `--self-test` is wired into `make contracts` |
| `scripts/readme_sync.sh`, `scripts/check_readme_claims.sh` | "README's count reads the census, not `find`" |
| `scripts/tests/ratchet_semantics_test.sh` | the fixtures need a census or the readme class cannot be judged |
| `README.md`, `Makefile` | the count (1842 → **1791**) and the regenerate+diff target |
| `docs/audits/surface_audit.csv` | 76 `pv <sub>` citations recomputed against the merged `cli.rs` |
| `contracts/apr-cli-commands-v1.yaml` | the CLI SURFACE contract: `census` must be a declared subcommand, or the probe `pv census` cites a surface the repo never declared |
| `crates/aprender-contracts-cli/Cargo.toml`, `Cargo.lock` | `serde` renders the JSON the probe parses; `sha2` computes `id_set_sha256`. The lock is their two-line consequence |
| `crates/aprender-contracts-cli/src/commands/mod.rs` | `pub mod census;` — the registration without which the subcommand does not exist |
| `docs/roadmaps/roadmap.yaml` | PMAT-3233's own row; `pmat work status` reads the ticket back through it, and the quorum brief reads it from there |
| `docs/audits/impl-PMAT-3233-receipt.md` | this receipt |
| `docs/roadmaps/entries/PMAT-3233.yaml` | main 's #3352 made roadmap.yaml a GENERATED aggregate mid-review; an edited entry is refused without its fragment, and the fragment SUPERSEDES the base entry, so it carries the corrected title and spec |
| `docs/audits/quorum-PMAT-3233.json` | the quorum artifact `pmat-merge` requires (`agreed=true` bound to the judged diff) |

`lint-provenance.sh` was reported "out of scope" by quorum round 2 lane 1. That lane receives only
Title/Status/Priority, and this file did not exist for it to read — the finding is a briefing gap,
and this receipt is its remedy.

## Clause table — measured at the head this receipt is committed on

| clause | before | after |
|---|---|---|
| `pv census contracts/ --format json` parses | rc 2 `unexpected argument '--format'` | rc 0 |
| `.n_parse_errors==0 and .n_files>0 and .n_files==(.n_parsed+.n_parse_errors)` | keys absent | true, `n_files=1791` |
| empty dir | rc 1 `error:` | rc 2 `decline: 0 contracts under <path>` |
| 3 valid + 1 malformed | rc 0, malformed **counted** | rc 1 `reject: 1 parse error under <path>` |
| `by_anchoring` / `{type}` vs `{type,ref}` | PASS | PASS (unchanged) |
| deterministic output | PASS | PASS (unchanged) |
| `timing.n_runs == 5` | key absent | present; `census_cpu_ms_p50`/`lint_cpu_ms_p50` **null until PVL EV-9** (operator ruling) |
| `census.json` tracked + `make contracts` diffs it | absent | tracked; `make contracts` rc 0 |
| `git_sha` | — | **null + `id_set_sha256`** (operator ruling: a sha that is wrong in a worktree is worse than absent) |
| "triage the live unparsed file" | — | **stale premise**: 0 of 1791 fail `pv validate` |

green at this head: census 1791 unchanged on regeneration · `readme_sync.sh --check` ok ·
guard self-test 7/7 · live guard `base=1791 merge=1791 delta=+0` · ratchet suite 12/12 ·
dogfood gate PASS · all nine CLI test targets ok · `aprender-contracts` 1526 passed ·
`make contracts` rc 0 · clippy 0 errors · `cargo fmt --all --check` clean.

## The second round-2 finding, refuted by measurement

Lane 1 also asked for a `census` row in `docs/audits/surface_audit.csv`. The absence is real; the
fix is not. Adding the row was measured to break the dogfood gate:

    G2.3 floors FAIL: low-confidence AND ungated is 206, must be <= 204

A new surface row is born low-confidence and ungated, so it spends the G2.3 budget. Landing it
requires gating or raising confidence on the row in the same commit — which is ONT-6's subject,
not ONT-1's. Recorded here rather than silently dropped.

## Two local guard failures that are NOT this branch

`guard_tree.sh --no-cargo` is red on this box with `check_baseline_ratchets.sh` and
`check_complexity_ratchet.sh`. Both are `tool_version` failures: `cb200_baseline.txt` and
`complexity_baseline.txt` are recorded under **pmat 3.40.1**, this box runs **3.40.2**. Measured:
neither file is in this diff, both headers are byte-identical at the merge-base `dfd3d5f96` and at
this head, and **both guards fail the same way at `dfd3d5f96` with none of this branch's code
present**. The complexity ratchet's own D2 table reads `base 679 / merge 679` — flat. CI passes
both, which is only possible if its runner resolves the 3.40.1 the headers name.

They are therefore **not repaired here, deliberately**: re-recording under 3.40.2 would make the
committed baseline disagree with the instrument CI runs, converting a local-only red into a
repo-wide one. The fix belongs to the commit that moves the toolchain pin, per the guards' own
message ("Re-measure and restamp this baseline under the instrument the FLEET runs").

## Round 7: the R-10 linter was vacuous on the only file it ships against

Two independent lanes refuted `scripts/lint-provenance.sh`, and re-measuring here confirmed it:

    lines the old filter fed the loop, on contracts/external-corpora.yaml:  1
    of those, lines containing a digit:                                     0
    => numeric claims examined: 0, exit 0
    appending `unmarked_total: 4242` to the file:                    still exit 0

The filter was `rg -N '^\s*(-|\|)\s*\S'` — Markdown list and table rows. Its one production
target is YAML, whose claims are key-values, so it read nothing and passed. That is
"0 violations over 0 files", the defect shape this repo names as its signature, inside the
guard written to satisfy R-10. The `--self-test` did not catch it because both fixtures were
`.md`: a fixture per FORM, where the rule is a fixture per FORM VARIANT.

Fixed: Markdown rows AND YAML key-values are scanned; identifier/command keys (`schema`,
`ref`, `repo`, `name`, `mark`, `counted_by`, `item_type`, `id`) are exempt so a schema URI is
not mistaken for a measurement; **the count examined is printed**, so a silent zero cannot
recur; and the self-test carries a yaml red/green pair, an exempt-key case, and an assertion
that the count examined is non-zero.

Proven to discriminate, not asserted: restoring the markdown-only filter makes the self-test
go RED (`rc=1`) on three assertions, including `the YAML fixture examined 0 claims`.
`contracts/external-corpora.yaml`'s two real claims (`head:`, `n_files:`) now carry inline
marks; `pv census` re-derived `contracts/census.json` byte-identically, so the marks are
comments and change no published figure.
