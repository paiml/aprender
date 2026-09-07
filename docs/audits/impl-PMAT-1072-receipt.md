---
status: complete
ticket: PMAT-1072
issue: 3028
kind: code
branch: agent/pp-066-roadmap-ids
model: claude-fable-5-1
tokens_used: ~120k (this andon, inside session 42763d92)
wall_clock_s: ~2400
turns: 14
---
# impl receipt — PMAT-1072: main RED on `roadmap-valid` (pmat 3.39.0 duplicate ids)

**Andon.** main @ 3792afa3d push run 34049248452: `ci / security` failed at the upstream
`roadmap-valid` step (paiml/.github sovereign-ci.yml:1207, bare `pmat work validate` from
PATH) with 12 duplicate ids; `ci / gate` RED; merge queue evicted #3021 (auto-merge
disarmed). Ticket #3028; fleet drift paiml/infra#468.

## Root cause (measured)

| claim | measurement |
|---|---|
| roadmap unchanged between green and red heads | `git diff --stat b0a0a51b2 3792afa3d -- docs/roadmaps/roadmap.yaml` empty |
| the pin passes, the PATH pmat fails | `$PMAT work validate` (3.37.0) → passed; `pmat work validate` (3.39.0) → 12 duplicates, identical to the CI log |
| PATH pmat moved today | intel `~/.cargo/bin/pmat` 3.39.0 mtime 17:30:02Z (auditd `cargo install pmat` 17:26–17:30Z); lambda 17:33:38Z; green PR run 34041439234 started 15:10Z, red main run 17:38Z |
| the rule is deliberate | pmat 3.39.0 CHANGELOG, PMAT-674 / pmat PR #1196: every `id:` line, subtask ids included |
| the duplicates are legacy nested records | 12 `subtasks:` child copies (`{id, github_issue, title: <id>, status, completion}`) under APR-ANTIGRAVITY-PARITY-001 (2), PMAT-710 (5), PMAT-716 (4), PMAT-719 (1); pmat 3.39.0 `work add` has no parent/subtask flag and never writes this shape |

## Fix

1. `docs/roadmaps/roadmap.yaml`: the 12 nested records dropped byte-exactly (four parents now `subtasks: []`); 64 deletions, 0 insertions; both pmat versions print `Validation passed`.
2. `scripts/check_roadmap_ids_unique.sh` (+ `contracts/apr-roadmap-ids-unique-v1.yaml`, kind: pattern, `pv validate` via the pin: valid): every `id:` in the PARSED tree unique, nested included, block-scalar text excluded; `--self-test` 6 rows both polarities; wired in ci.yml `guard-runner-labels` (case table, then live).
3. `scripts/lib/roadmap_diff.py`: `subtasks` joins `LIFECYCLE_KEYS` (structure the tooling maintains, not content); G-6 self-test row 3b added (18/18); live G-6 on this branch: `added=0 lifecycle=4 reserialised=0 deleted=0 PASS`.
4. The PMAT-1072 mint rides the session docs commit (A is the only minter there); the hotfix roadmap diff is the dedup alone.

## Verification (every command re-run by the orchestrator)

| check | result |
|---|---|
| `bash scripts/check_roadmap_ids_unique.sh --self-test` | 6/6 rows |
| `bash scripts/check_roadmap_ids_unique.sh` | PASS 809 ids, all unique |
| live twin on main @ 3792afa3d's roadmap | FAIL 12 duplicate ids (the same 12 as run 34049248452) |
| mutation RMID-F-001 (walk skips lists) | rows 1–4 RED, restored |
| mutation RMID-F-003 (unwire both ci.yml lines) | `check_guards_are_wired.sh` RED: `NEW: check_roadmap_ids_unique.sh`, restored |
| `check_roadmap_diff_additive.sh --self-test` / live | 18/18 · PASS (lifecycle=4) |
| `check_guards_are_wired.sh`, `check_baseline_ratchets.sh`, `check_complexity_ratchet.sh`, `check_no_claim_literals.sh`, `check_perf_claims_cite_receipts.sh`, `check_roadmap_completion_is_cited.sh`, `check_row_pr_write_set.sh` (orchestrator branch), `check_shell_lint_ratchet.sh` | all PASS |
| bashrs on the new script | 0 warnings, 0 errors (info-level SC1012/SC2316 remain: printf `\n` and `[ ]`, the repo's house style) |
| `pv validate contracts/apr-roadmap-ids-unique-v1.yaml` (pin) | valid |

CI mutation pair: main push run 34049248452 (RED, `roadmap-valid`) → this PR's run (GREEN, same step) — recorded in the PR body.

## Gaps

- The upstream step still runs bare PATH `pmat`; the pin it should use is infra#468's decision (3.37.0 declared vs 3.39.0 on the hosts). aprender's `scripts/pmat_bin.sh` PMAT_PIN=3.37.0 is now behind the fleet; moving it is a G-10a re-stamp, a separate ticket.
- `docs/roadmaps/roadmap.yaml.lock` / `.bak` are tracked on main (pmat 3.39.0's high-water mark and a backup); untouched here.
