# PMAT-4077 — ONT-8 evidence: implementation receipt

## Identity

| | |
|---|---|
| Ticket | PMAT-4077 (derived from paiml/aprender#4077 with `pmat work add --github-issue`), `kind:code` |
| Row | paiml/infra `docs/specifications/paiml-ontology.md` v4.14, ONT-8, probe (:667) |
| Branch | `PMAT-4077-ont8-evidence`, stacked on `origin/PMAT-4076-ont7-valid-under` 2b4eba725 (ONT-7, not yet folded) |
| Implementation head | `55f97614f` |
| Assigned by | cop aprender-cf (release-gate item: the ontology spec must be complete in 0.70) |
| Author | rmedia-82, claude-opus-5-5 |
| Gate number | 16. EV-11 (PMAT-4166) takes 14 and 15, as agreed with aprender-98 |
| Verdict | **DONE at the receipt.** Batching is the default: not armed, no PR opened, the cop folds |

## What landed

- `pv lint --gate evidence`, and **gate 16** of every `pv lint` run (R-8: computed everywhere, armed per repo; not armed
  here). Module: `crates/aprender-contracts/src/lint/evidence_gate.rs`.
- The rules:
  - **PV-ONT-017**: closed keys at `evidence{level, mark, provenance}`, `provenance{wasGeneratedBy, wasAttributedTo,
    generatedAtTime}` and `wasGeneratedBy{command, git_sha}`. Each level must be a mapping. F-16's `author:` is refused
    at all three levels.
  - **PV-ONT-018**: the level is deserialised into `proof_status::ProofLevel`, the one enum. So `levels_source` is
    `"enum"` and there is no second list. L0, `l2`, L6 and a missing level are refused.
  - **PV-ONT-019**: the mark must be one of V/C/A/U, and C or V must name `wasGeneratedBy.command`.
  - **PV-ONT-020**: a V `git_sha` must be 40 lowercase hex characters. It must equal `census.git_sha` when the census
    records one. Otherwise it must resolve with `git -C <contract_dir> cat-file -e <sha>^{commit}`.
  - **PV-ONT-021**: `wasAttributedTo` must be one of Σ `agents`. `generatedAtTime` must be a non-empty string.
- R-17: `entity.type` is counted into `entity_types_checked` and never branched on.
- Declines: no Σ, or no evidence block anywhere, exits 2 (R-2). A malformed Σ exits 3. When git cannot look (no
  binary, a shallow clone, not a repository), the result is `Unknown(ToolAbsent)`, counted in
  `unresolved_git_shas`. It is never a finding.
- Corpus witnesses: 4 evidence blocks, on model-capability-ladder (gguf, L2), refusal-receipt (json, L1), ont-shapes
  (pv-contract, L1) and release-readiness (release-evidence, L1). Each level was measured with `pv proof-status`
  before and after the edit and did not change. Each block is mark C and cites that command.
- Σ: the reader for `agents` is `lint/evidence_gate.rs`.
- Gate contract: `contracts/ont-evidence-v1.yaml`, with 6 falsification tests.

## The probe, on the real corpus at the implementation head

```
pv lint contracts/ --gate evidence --format json | jq -e '.verdict=="Pass" and .levels_source=="enum" and .entity_types_checked>=3'
→ true, exit 0 (1831 contracts checked, 4 with evidence, entity_types_checked 4)
```

The full `pv lint contracts/` gives verdict Pass, with `not_armed` = [reverse-coverage, valid-under, evidence].

## Spec defects, reported to infra-83 and not worked around silently

1. `[V] git_sha == census.git_sha` cannot be satisfied: the operator ruled on 2026-09-16 that `census.json` `git_sha`
   is always null. The gate binds to the census when it is non-null, and otherwise to the repository.
2. The spec's examples use `wasAttributedTo: orchestrator` and `"lane:aprender"`, and neither is a Σ agent. Σ
   declares `[pv, pmat, human]`. The gate follows Σ.
3. The spec says levels are L0–L5, but EV-3 removed L0 from the one enum. The gate follows the enum.

## Verification (author-run)

| Check | Result |
|---|---|
| `make gate` at 55f97614f | 92 guard PASS, 0 FAIL; touched-crate selection exceeded its cap, so it ran `cargo check --workspace --tests`, which was clean |
| aprender-contracts `--lib` | 1738 passed, 0 failed, 5 ignored |
| `lint::evidence_gate` unit tests | 20 passed |
| aprender-contracts-cli `ont8_evidence_gate` / `ont7_valid_under_gate` / `ont6_lint_verdict` | 8/0, 13/0, 7/0 |
| clippy `-D warnings` (contracts + contracts-cli, all targets), fmt | clean |
| `cargo deny check advisories` | ok |
| guards | check_ont_ratchet, check_complexity_ratchet, check_tree_reader_tests, check_explicit_test_commands, check_readme_claims, readme_sync, `pv extract --check`, `make roadmap-aggregate-check` |

**MUST-RED, measured.** Each mutant was planted on the tree, then the file was restored from a copy:
- M1: the level check accepts anything.
- M2: a skip on `entity.type == readme`.

Together they turn 3 tests red.

**An environment hazard found here, and not caused by this change.** `ont6_lint_verdict`'s corpus is a tempdir
whose parent is `$TMPDIR`. The duplicate-stems gate reads `<contract_dir>/../scripts/contract_duplicate_stem_baseline.txt`,
and a `/tmp/scripts/` created on this host at 08:41 made two ONT-6 tests fail with "48 stale". With an isolated
`TMPDIR`, all 7 tests pass. aprender-98 owns that test and has recorded this for the fold.

## Not done here

- Arming `evidence` in `contracts/lint-baseline.json`. R-8 says a gate is armed per repo, and that is the cop's call
  at the fold.
- Adding evidence blocks across the rest of the corpus. Four witnesses cover four entity types. More blocks are
  corpus work, not gate work.
