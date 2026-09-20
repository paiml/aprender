# APR-RELEASE-001 — Release train + build kaizen

Spec for the `paiml-implement` harness (repo `paiml/paiml-implement` — not `pmat-implement`).
Drop at `~/src/aprender/docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md`.
One spec, one command, one merged PR or one shipped tag per session.
**Revision 2026-09-17** — the 0.68 train proved the A0–B2 chain on a real tag for the first time
(five gate defects found, all fixed gate-side, tag never moved). Adds: hard rules 9–16 (9–13 from
the 09-15/16 rulings this copy had not carried; 14 fan-out by construction — no single-host release
step, T-3 refuses without the fan-out ledger, idle-during-train is a §8 stop, release-commit→tag
wall-clock ratchet; 15 gates prove first-green on a real target before they block; 16 PR work fanned out by pool
label across all four hosts, mini included), clean-room-on-
tag inheritance from the release commit, B2 split by capability (cpu shards + gpu), publish order
derived from `cargo metadata` (TIERS deleted), unattended cascade under standing authorization with
receipts, v0.68.0 = GitHub-only / 0.68.1 = crates.io, P0·Fan-out as the 0.69 train's first PR.
Earlier revisions kept where still true. *(Merged 2026-09-18 onto the `main` lineage — #3268 §4.1–4.3, §5.1, §6.1–6.4, §10–§12 and #3455 `check_milestone_cut.sh` — which the 09-15/16/17 working copy had been cut before; where the two disagreed the later operator ruling wins and the older text is kept as history.)*
**Revision 2026-09-16** — adds §1.6 critical path for 0.68, milestone hygiene (§6), the
CI-wedge mechanism and its two sanctioned rules (P0·Unwedge), clean-room-on-tag (T-3), reaper
budget as a manifest fact, and the capacity-source rule (no runners API, ever).
**Revision 2026-09-15** — adds §1.5 release-critical scope (0.68 = Qwen 3.5), the roadmap-fragment
and runner/pin rows (P0·Frag, P0·Runners, P0·Pins), merge-queue batching, the WIP hook, and the
measurement-hygiene rules learned on 2026-09-15. Earlier text kept where still true.

## §0 Run it

From `~/src/aprender`:

```
Implement docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md autonomously.
```

**Selector — every session does exactly one of these, first match wins:**

| # | If | Do |
|---|---|---|
| **0** | `yoga`, `gx10` or `mini` is under-utilised (§1 packing rule) while intel has queue pressure — and, every wakeup regardless, anything arrived since the last sample is untriaged (§6.1) | **P0, minutes not a session:** arm every green PR, route what can leave intel, reap disk (§5 P0·Pack, P0·Reap); triage what arrived (§6.1); record the `pack:` and `triage:` lines; then continue to the first matching row below |
| **1a** | a §1.5 release-critical ticket for the next tag is open and not merged | **one feature PR on it** (TDD, contract, `apr-dogfood` slice) — this row outranks every §5 row and never yields a slot to them; a second session may run §5 in parallel |
| 1 | ≥ 48 h since the last tag on `main` **and** no SKIPPED record for the current HEAD **and** §1.5 scope for that tag is merged | run the train (§4) |
| 2 | else a §5 row whose *Done* test fails at HEAD | do the first such row, one PR |
| 3 | else the last train (shipped or skipped) has no once-per-train triage record | do the once-per-train pass: §6.3 capacity check and the T-5 reconcile receipt |
| 4 | else | emit the §7 report, exit 0 |

Nothing in this spec asks a question. Running it ten times a day is safe. Why the loop never
terminates and the four things it moves — repo, released binaries, CRUX competitors, fleet — on the
`pv` ontology substrate: §12.

## §1 Goal

Ship a tag every 48–72 h (`0.67 → 0.68 → …`) **on a clock, not on scope**, and keep
shrinking the wall-clock from *PR opened* to *tag published* so the clock stays cheap.
The objective is elapsed time and green trains. **Packing rule (operator, 2026-09-12, verbatim):** "these two boxes: yoga and gx10 should be always 80% full of PRs from aprender if ANY queue pressure on intel … not acceptable to have slow releases when boxes are idel". Utilisation is not the goal for its own sake; an idle GPU box next to an intel queue is lost release time and is a **P0 defect**, not a state to tolerate. `mini` (Apple M4) was declared a full-time aprender build host on 2026-09-13 (#3205) and is under the same rule; its ceiling is the macOS-capable job classes, not capacity. Measure it every wakeup, **from the ledger, never from a runner-list snapshot**:

```
record          = docs/build-ledger/<date>/<sha>-<host>-fleet-pack-*.json   # written by §5 P0·Pack
intel pressure  = aprender_runs_queued > 0, or a workspace-test running, on intel
under-utilised  = intel pressure AND (busy/online < 0.8 on yoga, on gx10, OR on mini)
```

Two measurement traps, both already paid for: a `busy` snapshot from the runners API cannot see
ephemeral runners, and an hourly **average** hides saturation — `occ_1h` read 9.5 % on 2026-09-14
while 15 of 16 intel workers were busy at load 162. The `pack:` line carries the instantaneous
`busy/online` at sample time; an average is a trend, not a verdict.

Coupling, one line:

```
max PRs per train  ≈  3 × 72 h  /  p95 `ci / gate` wall-clock      (upper bound; 3 = merge-queue parallelism, §3.4)
```

The merge queue builds 3 entries in parallel (§3.4); gate latency still bounds release throughput.
Compute this on every train. p95 is `[U]` until §5 P0 lands — `make build-report` does not exist on
`main` as of 2026-09-14.

**Capacity source rule:** `GET /orgs/{org}/actions/runners` is never used — not for pack,
not for unwedge, not for any gate (operator, 2026-09-15). Capacity = `verify-fleet-bin` on each
host. A rule that needs the runners API is dark by construction; rewrite it to read the oracle.
`fleet-bin.sh` is the only runner oracle. Do not reimplement it (2026-09-15: a MainPID-keyed
restart target read the wrapper, not the listener; its busy-skip could never fire).

## §1.5 Release-critical scope (operator 2026-09-15)

Scope is normally assigned after the fact (§4). The exception is a **named release-critical
item**, declared here, which the tag must carry:

| Tag | Must carry | Ticket(s) | Evidence to cut |
|---|---|---|---|
| **0.68** | **Qwen 3.5 support** — model load, tokenizer, inference parity against `aprender-canonical-benchmark-rfc.md`, `apr` CLI surface | 0.68 epic in `docs/roadmaps/roadmap.yaml` (`pmat work add` it if absent) | contract `contracts/apr-qwen35-v1.yaml` `N obligations, 0 failed`; `apr-dogfood` go receipt on the cut sha; provenance marks on every perf figure, `[X]` for third-party |

Rules: a train for a tag with unmerged §1.5 scope is **not cut** (T-0 waits) — this is the one
exception to §3.1, and if it holds > 72 h it is a §8 andon with the ticket named. Work on §1.5
scope is never paused for §5 rows: with 3 session slots, ≥ 1 is always on §1.5 until merged.
§1.5 work never touches hosts; GPU dev runs on lambda-labs (4090, not a runner) or yoga's GPU
via existing make targets.

## §1.6 Train state and critical path (updated 2026-09-17)

- **v0.68.0** — GitHub-only: tag, 16 assets, `install.sh` verified on intel/gx10. Not on crates.io (B2 found lib tests naming path-only sibling dev-deps, #3425). Release note points at 0.68.1. A published tag is never moved or deleted.
- **0.68.1** — **SHIPPED 2026-09-17**: 74/74 on crates.io, cascade 22:00:38Z→22:41:36Z (2458 s, unattended, 429-only retries); clean-room on the tag = B2-cpu paiml/infra run 35255889257 + B2-gpu paiml/aprender run 35268563585; records in `docs/audits/release/v0.68.1/`. Measured corrections to the plan below: B2-gpu runs from paiml/aprender (`b2-gpu.yml`) because the org GPU runner groups admit that repo only; #3466 needed a 48g container + 8 threads (one test holds 32.6 GB — a thread cap alone cannot fit 32g); publish preflight R6 judges the cycle, not the shape (#3469). Chain as planned: clean-room on tag (B2-cpu + B2-gpu, `--no-fail-fast`) → dry-run receipt committed → `publish_strict.sh` (derived acyclic order, one crate per call, stop on first non-zero, 429/5xx ×3) → post-publish `cargo install apr-cli --version 0.68.1` on intel + gx10 → receipts on the release → SHIPPED. Gate-side fixes for it: infra#654 (inheritable overlay), #3466 thread cap, #3464 A3 install, #3465 B2-gpu split.
- **0.69** — first PR is P0·Fan-out (rule 14), second is P0·PR-Fan-out (rule 16), before any scope work. §1.5 scope: #3090 GPU DeltaNet, #3208, #3341, #3369, #3464/#3465/#3466 test hygiene, mini neutral shard, unwedge rule 3, TIERS deletion, pmat 3.41.0 pin.

### 0.68 critical path (2026-09-16, operator) — DONE 2026-09-16T22:58Z

One line, in order; every session slot not on queue-unblocking work is on it:

    #3331 llama.cpp pin (MERGED) → parity evidence series PR1 #3354 → PR2 → PR3 #3356
    → #3114 out of draft (contracts discharged, Closes #3091 #3303) → #3091 merged → T-0 cut

Andon: **2026-09-18T12:19:39Z** if #3091 is unmerged (§8). The train waits for §1.5 scope; it
does not cut a 0.68 that is not Qwen 3.5. Report per-PR ETA against this clock every interval.
Push the next PR in the series the moment the previous one enqueues; disk is no longer the
constraint (gx10 ≥ 1 TB free after P0·Reap).

## §2 Ground truth — verify at HEAD before writing anything

| Fact | Source of truth | Mark |
|---|---|---|
| `intel` — clean-room runner, 8 concurrent, memory-bound, 3.6 TB NVMe | `infra/machines/intel/forjar.yaml` | `[V]` snapshot |
| `yoga` — CI runner, RTX 4060 8 GB, 32 GB RAM, 10G NIC needs `bolt.service` | `infra/machines/yoga/forjar.yaml` | `[V]` snapshot |
| `gx10` — aarch64 GB10, sm_121, 120 GB unified; **not** a documented general runner | `infra/machines/gx10/forjar.yaml` | `[V]` snapshot |
| `main` protected; required checks are exactly `gate` and `workspace-test` — `present` is NOT required, it is the review-receipt backlog | `gh api repos/paiml/aprender/rules/branches/main` | `[V]` 2026-09-14 |
| Last tag = `git describe --tags --abbrev=0` on `main`; next minor = that + 1 | git | `[V]` live |
| p95 `ci / gate`, tag→publish, cascade wall time (automated) | — | `[U]` unmeasured |
| `workspace-test` is pinned `runs-on: [self-hosted, X64, Linux, clean-room]` (#3104) — the long pole never lands on gx10; #3139 lifts the pin (795 s on gx10-pool3, 34693750990) | `.github/workflows/ci.yml` | `[V]` 2026-09-12 |
| gx10 and yoga: `/mnt/nvme-raid0 -> /home/noah/eph-work/intel-mirror`; per-PR target dirs under `targets/{aprender-ci,sovereign-ci-aprender}/<pr>`; pool runners are docker containers (`sovereign-gpu-runner:2.337.0`); no reaper existed until 2026-09-12 (gx10 hit 100 %, 146 GB reclaimed by hand, then declared in forjar) | `ssh gx10`, `ssh yoga`, `machines/{gx10,yoga}/forjar.yaml` | `[V]` 2026-09-12 |
| Roadmap path is `docs/roadmaps/roadmap.yaml` — **no `infra/` prefix** post-APR-MONO. Every PR editing it serialises the merge queue (Amdahl s=1). Fragments: `docs/roadmaps/entries/PMAT-NNNN.yaml`; `roadmap.yaml` generated on `main` by `make roadmap-aggregate` (idempotent, deterministic, supersede-on-duplicate; contract `contracts/apr-roadmap-fragments-v1.yaml`) | aprender #3297; `paiml/.github` #73 | `[V]` 2026-09-15 |
| Fragment gate lives upstream in `sovereign-ci.yml` (#73), opt-in on presence of `entries/` on base, `NOT-RUN` otherwise; consumers SHA-pin the workflow, so the **pin bump is what arms it** | `.github/workflows/ci.yml` `uses: paiml/.github/...@<sha>` | `[V]` 2026-09-15 |
| `guard_tree.sh` runs **every** script in `scripts/` bare unless wired-with-args, release-time, or unwired baseline — a new guard is armed the moment it lands | `scripts/guard_tree.sh` `skip_reason()` | `[V]` 2026-09-15 |
| `guard-cargo` / `guard-tree` are bare-host jobs (`runs-on: [self-hosted, Linux, clean-room]`, pool spans intel/yoga/gx10), not container jobs — the deciding tool version is the **runner user's PATH** | `.github/workflows/ci.yml` | `[V]` 2026-09-15 |
| Tool-pin order of truth: container image ≥ hosts ≥ infra declarations ≥ repo `tools.toml`; assert the repo pin last, never converge a pool downward. `machines/fleet-hosts/pmat-pins.txt` maps machine→manifest; the version lives in `stack-tool-pmat.version` in each host's `forjar.yaml` | `tool-pin-check.sh` (prints its tree on line 1, fails closed off-main) | `[V]` 2026-09-15 |
| yoga carried **undeclared** runners `yoga-build2/3` with `clean-room`, ephemeral, no unit, no declaration; they ejected #3295 three times with 6 s cancels. Ruling: undeclared runners are residue → deregister; reaper drains any live runner not in the host's forjar declarations | infra#601, audit log 2026-09-15 | `[V]` 2026-09-15 |
| pmat repo is `paiml/paiml-mcp-agent-toolkit`, default branch `master`; `RoadmapServiceIo::save()` is the sole locked roadmap writer (one bypass in `ticket_validate_migrate.rs`, ticketed) | pmat #1364, #1368 | `[V]` 2026-09-15 |
| forjar `plan` vs `show` resolve templates differently — defect, ticketed in forjar; prejob pin hook is measurement-only (`gate=NOT-ARMED`) until the materialized pin file exists | forjar ticket; infra #603 | `[V]` 2026-09-15 |

Live `forjar.yaml` beats this table. Record the diff in the receipt and continue.

**Measurement hygiene (2026-09-15, three stale-checkout hits in one day):** every measurement
prints its tree (`HEAD` vs `origin/main`, `behind=0`) on line 1 and refuses to run off-main
without `--allow-branch`; sessions work in per-session `git worktree`s, never the primary
checkout; never `git stash` on a shared tree. "N commits ahead" ≠ unlanded — check
`gh pr list --state merged --search head:<branch>` and diff against `main`. Scope code writers
with `pmat query` by serialisation site, not grep. Verify `runs-on` before naming the layer
that decides a gate.

## §3 Hard rules

1. **The train is never delayed.** Cut can't go green in one attempt → train **SKIPPED**,
   reason recorded against that HEAD, no retry until `main` moves. Next eligible in 48 h.
2. **Two consecutive skips, or ≥ 72 h since the last tag with no runnable train → andon.**
   Stop, five-whys terminating in a mechanism, escalate. Do not cut a third.
3. **No invented numbers.** Every threshold cites its measurement command or carries `[U]`
   and is not a gate. No build target is set before P0 has ≥ 20 ledger records.
4. **Heijunka is bounded by the fleet, not by one.** WIP = what the fleet can build without
   starving a box: the merge queue builds entries in parallel (ruleset 17836320; group size **8** once P0·Frag lands, 3 before — verify in the live ruleset).
   **Never dequeue a PR because another is in CI**; dequeue only a group that is *known* RED
   (roadmap-additive guard, a red required check) and fix or trim it rather than park it. Under
   intel pressure, arm every green PR and prefer the ones whose jobs can land on `yoga`/`gx10`.
   Splitting *one* gate across hosts (§5 P2) is the other half of the same rule. "Parallel"
   inside a session means ONE Claude subagent at a time and fan-out through agy (§10) — never two
   sessions merging.
   *(Amended 2026-09-12 by operator ruling; the previous text said "one aprender PR in CI at a
   time" and this session dequeued four PRs on it — #3175's group was running its workspace-test
   on yoga at the time.)*
5. **No ad-hoc host CONFIG.** Durable host changes are `machines/<host>/forjar.yaml` →
   `forjar apply` → `make -C machines/<host> verify-systemd-units`. Repo edits are inert until
   deployed (2026-04-26 ENOSPC was exactly this). **SSH for measurement and for unclogging is
   expected** (`ssh gx10`, `ssh yoga`: `df`, `du`, `systemctl status`, `docker system df`,
   reclaiming a full disk that is blocking the queue right now). The reclaim and its forjar
   encoding are one piece of work: what was done by hand once is declared so the next time is
   automatic. *(Amended 2026-09-12: "you DO HAVE SSH (ssh gx10, ssh yoga)".)*
6. **Ledger is append-only, one file per run:** `docs/build-ledger/<YYYY-MM-DD>/<sha>-<host>-<job>.json`.
   Never a shared file (G-11 rebuild-storm class). The ledger *is* the project memory.
7. **Publishing: clean-room is the hard gate, named first.** Then `publish_strict.sh` from a
   detached checkout of the promoted tag — committed dry-run receipt, derived acyclic order, one
   crate per call, stop on first non-zero, 429/5xx ×3, never `--allow-dirty`. No workflow runs
   `cargo publish`. **Unattended under standing operator authorization (2026-09-17)**; the receipt
   records `attended_min` and cascade start/end. Anything else needing Noah is a §8 stop.
   *(History: until 0.68 this rule ran `scripts/cascade-drain.sh --target 0.N.0 --passes 30`; its TIERS order was never
   topological — 47 violations at v0.68.1, #3462 — and only worked through the drain's retries. Operator ruling
   2026-09-13 stands: "T4 is never mine … releases are automated by release train … for ALL releases".)*
8. `pmat work add` from the driver session only. Stop the line on RED — no reruns-as-passes,
   no `--skip`, no waivers.
9. **No hand-squash of batches.** Batch verification is the merge queue's job (merge_group,
   group size 8); each PR lands as its own commit. A red batch fans out one job per PR, never
   hand-bisected.
10. **WIP is enforced at the producer.** paiml-implement's pre-push hook refuses to mint a PR when
   open aprender PRs ≥ 13 (basis: measured drain 8.8/day × 1.5 day, 2026-09-15; ratchet after 20
   records) and emits an issue instead. Sweeps emit issues, not PRs.
11. **Undeclared is residue.** A live runner with no declaration in `machines/<host>/forjar.yaml`
   is deregistered, not adopted; a declaration that says less than the fleet has is the same defect
   as one that says more. Label edits are never used to dodge a runner.
12. **Reuse the verified instrument.** Before writing a check, `pmat query` for the existing one.
   A busy-skip or preflight that structurally reads zero is a hatch.
13. **Never cancel a run with started jobs to "unwedge".** Cancel only by the P0·Unwedge rules,
   never by hand, never a `merge_group` run (2026-09-16: a 16/18-complete run was cancelled on a
   misdiagnosis; cost, a full rebuild on the critical path).
14. **Release-train work is fanned out by construction** (operator 2026-09-17: "we have had these
   machines mostly idle and we are slipping releases" — permanent, structural, not audit-class). No
   T-step job may carry a single-host `runs-on`; every step declares its host set: T-1 deep → intel ·
   T-2 dogfood → lambda-labs (dev box, GPU, not a runner) · clean-room A0–B1 → yoga (X64 container)
   · B2-cpu → sharded by crate across intel + yoga + gx10 (aarch64 shard = crates that build there),
   partitioned by measured binary duration from the ledger, `--no-fail-fast` per shard, one
   aggregated verdict · B2-gpu → yoga/gx10 bare host, CUDA · mini → macOS neutral shard once the
   probe lands. Pre-tag verification runs on the release COMMIT; the tag inherits when
   `tag sha == commit sha` (the assert checks equality). **The T-3 tag step refuses unless the
   fan-out ledger shows every declared shard ran on its declared host set.** `pack:
   P0-UNDERUTILIZED` during T-1/T-2/clean-room is a §8 stop. Release-commit→tag wall clock is a
   shrink-only ratchet after 3 records (baseline 0.68.1 `[A]`, target ≤ 30 min; today 3–4 h across
   serial iterations). **0.69 does not cut on a serial chain.**
15. **Gates are proved on their real target before they can block.** A0–B2 had never completed on
   a tag before 0.68; five stops were gate defects (A1 crates.io deadlock; CLI-only overlay not
   inherited by child cargo; B2 fail-fast hiding exposure; `apr` not on PATH from a skipped A3;
   32 GB container OOM). A new gate ships with a first-green receipt on a real tag or it runs
   measurement-only.
16. **PR work is fanned out by construction** (operator 2026-09-17; same doctrine as rule 14,
   applied to `ci / gate`). Every `ci.yml` job declares a pool, never a host. Pools: OS-neutral
   jobs (fmt, clippy, doc, mutants, contract tests, CPU-only crate tests) → `[self-hosted,
   rust-neutral]` on intel + yoga + gx10 + mini · `workspace-test` → `clean-room` arch-neutral on
   intel + yoga + gx10, sharded by crate once the ledger shows p95 > 20 min · CUDA jobs → yoga /
   gx10 · container clean-room jobs → intel + yoga. The neutral list is **measured**: one probe run
   per job per host on a sha green on intel, verdict + duration parity, recorded in the ledger;
   a job leaves the neutral pool the first time its verdict differs across hosts for the same sha.
   mini joins the pools through forjar (GNU coreutils/bash for the runner user, tool pins per
   `pmat-pins.txt`, prejob hook, reaper, `verify-fleet-bin`, three-way pin check) — "cowork first"
   is retired. `pack:` during queue pressure with any pooled box < 0.8 busy/online is a §8 stop.
   p95 `ci / gate` wall clock is a shrink-only ratchet after 3 records. Ticket: 0.69, immediately
   after P0·Fan-out lands; no §1.5 scope PR merges before it.

## §4 The train — each step has its own already-done test

| Step | Action | Skip if |
|---|---|---|
| **T-0 Cut** | **precondition, before the bump PR opens:** `bash scripts/check_milestone_cut.sh 0.N.0` exits 0, meaning the milestone holds no open issue or PR except this train's own release epic. That epic is closed at the end of the train, and the script names it as admitted. An item not riding the train is carried by moving it to the next milestone with a `slipped_from: 0.N.0` comment (§6.3 spill), never by leaving it open. Then cut sha = `main` HEAD; bump minor; `CHANGELOG` from merged PR titles since last tag | tag `v0.N.0` exists — **T-2 preflight before the cut (operator 2026-09-17, from 0.69):** `scripts/dogfood.sh --phase pre-publish` runs on `main` HEAD *before* the bump PR opens; the bump PR is **refused at arm time** without a GO receipt for its parent sha; the release commit moves **zero** times. Measured need: 0.67 stopped 3×, 0.68 stopped 4× at T-2 after the cut, every red a release-phase-only gate (perf041 marker age, `pv validate` over all contracts, CB-200, C14) |
| **T-1 Deep** | `deep` green on cut sha (**the check is `deep`, not `ci / deep`** — §4.3): full tests, doctests, `--no-default-features`, feature matrix (§4.1), GPU, every `cargo run --example` (§4.2) | green run recorded for this sha |
| **T-2 Dogfood** | `apr-dogfood` skill go/no-go receipt; `apr-cookbook` current; release notes generated | receipt exists for this sha — **plus one row (2026-09-16, #3366):** `install.sh --version v0.N.0` on a clean intel (x86_64) and a clean gx10 (aarch64) resolves the asset, the sha256 verifies, and `apr --version` prints `v0.N.0`; both hosts, both rows green, or T-2 is red. The row fetches from the URL the script itself advertises, so a 404 there is a T-2 failure, not a docs nit — **Nightly producers (operator 2026-09-17):** any nightly that feeds a T-2 row (today: `cuda-nightly.yml` → PP-26 marker) and fails **≥ 2 consecutive** runs auto-opens an issue on the next milestone and shows as `NIGHTLY-RED` in §7; the T-2 receipt records the witness host's provenance when it differs from the producer's (0.68: lambda sm_89 witness; gx10 producer red since 09-12, #3096 stays 0.70) — **Inherited T-2 (operator 2026-09-17):** T-2 on the release commit inherits the parent-sha preflight GO when the bump diff touches only `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, `crates/*/Cargo.toml`; the receipt records `inherited_from: <sha>`; any other path → full T-2. First-chain rule: the clean-room on the tag runs through B2 and past it to the end; if a gate after B2 stops, the gate id is reported before any fix |
| **T-2 Models** | **(2026-09-18, EPIC #3477)** the model capability ladder — `contracts/model-capability-ladder-v1.yaml` — has a receipt from **every required host** (`lambda` sm_89, `gx10` sm_121) for this cut: `bash scripts/model_ladder.sh` on each host writes `evidence/dogfood/models/<version>/<host>.json` (`apr qa` capability_match + golden_output per rung, plus one `apr run` per claimed backend that must not print `falling back to CPU`); `bash scripts/check_model_ladder.sh` (declared in `[package.metadata.dogfood]`, so `dogfood.sh` runs it in every phase) is green. Why a row of its own: 0.68.1 shipped dense Qwen3 producing garbage on CUDA on both fleet GPUs; the per-inference parity guard fell back to CPU, so `apr run` looked fine and the throughput gates PASSED on the garbage path — the only surface that said so was `apr qa`, which no train step ran on a Qwen3. The unit of release evidence is now the triple (architecture, backend, silicon); "a model ran" is not one. A missing receipt is **FAIL, not DEFER** (a dev build measures it). The ladder at `origin/main` is a floor — rungs, hosts and backends may be added in a PR and never removed. At 4a538ddef this row is RED on both hosts (#3413 qwen3/cuda, #3432 qwen35/capability_match): the 0.68.2 train cannot cut until both land, which is the point | receipts for this version from every `required: true` host, gate exit 0 |
| **T-3 Promote** | **immediately before the tag, after the bump PR has merged:** `bash scripts/check_milestone_cut.sh 0.N.0` exits 0 again. Exit ≠ 0 is RED and there is no tag. The T-0 read is not enough, because an item reopened between the reads is invisible to it (v0.68.0 shipped with #3091 reopened 85 min before the tag, #3445). Then tag; clean-room on the tag; GitHub release with T-2 receipt attached **Then (rule 14, 2026-09-17):** fan-out per rule 14 on the release commit (T-1, A0–B1, B2-cpu shards, B2-gpu in parallel); tag only when every shard is green; clean-room-on-tag **inherits** when `tag sha == commit sha`, else **dispatch `clean-room.yml` on the tag immediately**; the run's first step asserts `HEAD == tag` and fails otherwise; record the run id; GitHub release with T-2 receipt attached. Interim until infra#621 (ref input) lands — the nightly clones `main` at 23:00Z and has never tested a tag (0 of 10 runs after v0.66/v0.67) | release `v0.N.0` exists — release `v0.N.0` exists with a green clean-room run id on its sha — **ordering change (2026-09-16, #3366):** T-3 builds the `apr-*` binary assets on the tag **before** T-2 runs, because T-2's installer row consumes them; a T-2 receipt that predates the tag's assets is [U]. Release notes: the installer is the headline after Qwen 3.5 |
| **T-4 Publish** | `publish_strict.sh` from a detached checkout of the tag, `HEAD == tag` asserted, clean tree; order derived from `cargo metadata` at the tag (acyclic, re-checked before start; TIERS is deleted — #3462); one crate per call; stop on first non-zero and report crate + what is already published (never hand-rolled back); retry only 429/5xx ×3 with backoff; never `--allow-dirty`; refuses without the clean-room run id on the tag sha (#3335) and a committed dry-run receipt (`--dry-run --no-verify --locked`, rc=0, tree clean). **Unattended under standing operator authorization** (2026-09-17); receipt records `attended_min` and cascade start/end (the next kaizen number). Post-publish: `cargo install apr-cli --version 0.N.M` on intel + gx10 via a `fleet-hosts` make target, `apr --version` matches; all crates verified on crates.io; run id, receipt and per-crate timestamps attached to the release | all crates at `0.N.M` on crates.io |
| **T-5 Reconcile** | **hard gate (operator 2026-09-13: kaizen)** — the §6 reconcile predicates hold, receipt `docs/build-ledger/<date>/<sha>-reconcile.json` committed; the train has no `DONE` line without it | receipt exists for this sha and every predicate reads 0 |

Any step RED → SKIPPED, no partial promotion. Scope is assigned to trains after the fact:
0.67 contains whatever merged before the 0.67 cut, by definition.

### §4.1 Feature matrix — what T-1 means by it

One `cargo check -p <crate> --no-default-features --features <one feature>` per **(crate, feature)
pair**, not a powerset: a powerset is 2^n per crate (aprender-orchestrate alone declares 78 features)
and infeasible. The universe comes from `cargo metadata`, never a written list.

This clause was in T-1 from the start and was never built. Its first run, 2026-09-14 (#3262),
measured **100 of 430 pairs RED — every one unreachable from any crate's default set**. That is the
whole reason it can go dark: `cargo check --workspace` stays green over all of it, because feature
unification hands each crate whatever its siblings enabled.

Five shapes account for all 100; check them in this order:

1. **Definition gated, caller not.** Often the gate is over-broad — **un-gate it, do not spread it**.
2. **Declared in `Cargo.toml`, never applied in code.** `grep -rn 'feature = "X"' --include='*.rs'`
   returning nothing is the test.
3. **The implicit dep-feature used where the composite was meant.** An optional dep `foo` creates
   feature `foo`, which links foo and nothing else.
4. **A feature that supplies nothing its code needs** — usually a dependency removed for lock bloat
   with the gated code left behind. **Measure `Cargo.lock` before re-adding**: a version already
   locked by a sibling costs zero new packages.
5. **Struct drift** — an upstream type grows a field and the one initializer in the tree is never
   updated. Nothing else in the train watches for this; the matrix is what catches it.

**Known-red list.** A pair that cannot be made green in the same train is listed with a date and an
issue reference, and is reported, not fatal — a lane born red is a lane taught to be ignored. A
listed pair that **PASSES is fatal**: the list may only shrink, and it shrinks by deleting a line.
Re-derive the list against the tree immediately before the lane is undrafted; a dated list is
evidence of when it was true, not that it still is.

**A feature that cannot be built at all** — its dependency is deliberately absent, or circular —
emits ONE `compile_error!` naming the dependency and the exact edit that would enable it, with the
real gate moved to a private `__x-linked` feature so the refusal is the only diagnostic. Precedent
in tree: `aprender-simulate/z3-proofs`, `aprender-core/showcase-profile`, `aprender-zram-core/cuda`.

Fixing one cause exposes the next — the nextest-fail-fast class. Re-sweep after every fix; a
single-cause sweep is not a measurement.

### §4.2 Examples — run, not just build

T-1 says *every* `cargo run --example`. Building them is a different clause and much cheaper (83 s
for 980 targets vs ~2 h to run them). Measured 2026-09-14 on a random five with `-- --help` and a
60 s cap: **three ran past it**. They are not CLIs, they are compute demos that ignore argv, so
**timeout is a PASS**. The assertion T-1 actually owes is *every example starts and does not crash*.
Asserting a duration here would be a wall-clock assertion in a required check, which
`check_no_timing_in_required.sh` exists to forbid.

Both clauses carry a **vacuity floor** (measured 980 examples / 430 pairs; floors 800 / 400): a
discovery that finds almost nothing reports zero failures, which reads exactly like a pass.

### §4.3 The check is `deep`, and it cannot be proven before it is merged

This spec said `ci / deep` in five places. No such check can exist here, and the T-1
already-done test was reading for a string that would never appear.

GitHub names a check after the job that emits it. A job inside a reusable workflow is
prefixed with the name of the job that CALLS it — `ci.yml` calls `paiml/.github`'s
`sovereign-ci.yml` as a job named `ci`, which is why this repo reports `ci / lint`,
`ci / test`, `ci / gate`. A job in a workflow file of its own carries no prefix, which is
why `workspace-test`, `guard-tree`, `guard-cargo` and `gate` appear bare.

So `ci / deep` would require adding a `deep` job to the ORG-WIDE reusable workflow, with
blast radius across every repo that consumes it. The deep lane lives in its own
`.github/workflows/deep.yml` with a job named `deep`, and emits **`deep`**. Amending this
spec is the cheap half of that trade; amending an org-wide workflow to match a string this
document happened to write is the expensive half.

**The sequencing hazard, measured 2026-09-14.** `deep.yml` fires only on `workflow_dispatch`
and a cron — deliberately, since a deep lane must not run per-PR. But:

```
$ gh workflow run deep.yml --ref PMAT-1098-ci-deep-lane
HTTP 404: workflow deep.yml not found on the default branch
```

`workflow_dispatch` is honoured only for workflows already on the default branch. A lane that
has no `pull_request` trigger and is not yet on `main` therefore **cannot be exercised at all
before it merges** — its first execution would be the cut it gates. §4's rule for a red step is
SKIPPED, so a lane born red costs the train silently rather than failing loudly.

T-1 is therefore not satisfied by `deep` EXISTING. The train's already-done test is a green
`deep` run **recorded against a sha on `main`**, and the first such run must be a deliberate
`gh workflow run deep.yml --ref main` after the lane lands and before a cut is attempted. A
gate whose first run is the thing it certifies is the defect class this document exists to
remove; it must not be reintroduced by the gate that closes it.

## §5 Build rows — in order, one PR each, only when no train is due

**P0 · Frag** *(2026-09-15; the queue is serial until this lands)*. Roadmap fragments per §2:
#3297 (split + aggregator + contract) → PR3 pin bump of `sovereign-ci.yml` to the #73 SHA, whose
RED→GREEN evidence is a fragment-less PR failing under the new SHA. Non-conforming legacy ids
(99 of 879: 47 non-`PREFIX-N`, 44 prose, 8 prose-tailed) are frozen in base or re-id'd in a
migration PR that lands **before** PR3 — decided in the contract, never a hatch.
`impl-estimates.jsonl` is `merge=union` (append-only, one record per line).
*Done:* zero `roadmap.yaml` diffs in any open PR; merge-queue group size 8; two PRs merge in the
same group with zero conflicts; three consecutive `make roadmap-aggregate` on `main` byte-identical.

**P0 · Runners** *(2026-09-15)*. Fleet tooling parity in `machines/fleet-hosts/` (host-
parameterized `verify-fleet-bin`, oracle restart, `deploy-runner-prejob-hook`, `runner-journal`,
`.runner` inventory) with intel/yoga/gx10 as thin callers — no per-host Makefile copies. Reaper
gains two gates: (a) any live runner not in the host's forjar declarations → drained + reported;
(b) N consecutive sub-10 s cancellations from one runner name → drained + reported. Ephemeral
spawners run preflight **before** `config.sh`. *Done:* every live runner on every host is
declared; `verify-fleet-bin` converged on intel, yoga, gx10; a merge_group run completes with
zero ejections across 10 trains.

**P0 · Pins** *(2026-09-15)*. `tool-pin-check.sh` three-way (manifest == materialized pin file ==
resolved as the runner user) on every unit; prejob hook refuses a job on mismatch (`gate=ARMED`
only once the forjar materializer lands); `pmat-pin-check` promoted to a nightly infra gate
failing on `drifted>0 || unchecked>0`. Pin moves: `tools.toml` 3.40.0 → 3.40.1 (#3301, measured)
→ 3.41.0 after pmat publish, hosts + image in the same convergence. *Done:* `matched=N drifted=0
unchecked=0` as runner principal on every host; `tools.toml` == every unit; one convergence per
release.

**P0 · Fan-out** *(2026-09-17; the 0.69 train's first PR; precedes every §1.5 scope row)*. Implement
rule 14: shard B2 by crate across hosts with a partition derived from the ledger; B2-gpu bare-host
target; per-shard ledger records (`host, crates, duration, peak_rss_mb, verdict`); aggregated
verdict; T-3 refusal without the fan-out ledger; `pack:` P0 as §8 stop; wall-clock ratchet. Also:
B2 thread cap for heavy crates declared in gate config with its measurement as basis (#3466); A3
install runs for aprender so `apr` is on PATH (#3464); aprender-gpu excluded from B2-cpu only by a
printed `requires: cuda` reason and only because B2-gpu is required (#3465); publish overlay in the
container `$CARGO_HOME/config.toml` so child cargo inherits it (infra#654); 0.69 crate fix: hardware
tests report a typed N/A with reason when the driver is absent, never a bare failure or silent pass.
*Done:* one release commit verified across ≥ 4 hosts in parallel; release-commit→tag ≤ 30 min;
`pack:` shows every host busy during the train; three consecutive trains with zero serial fallbacks.

**P0 · PR-Fan-out** *(2026-09-17; the 0.69 train's second PR, directly after P0·Fan-out)*. Implement
rule 16: pool labels via forjar on all four hosts; mini onboarding (coreutils/bash, pins, hook,
reaper, oracle); per-job probe run on each host with verdict + duration parity in the ledger; the
measured neutral list moved to `[self-hosted, rust-neutral]`; `workspace-test` to the arch-neutral
clean-room pool, sharded when p95 > 20 min; `pack:` gains mini and an eligible-jobs count.
*Done:* every `ci.yml` job carries a pool label; a PR's gate runs on ≥ 3 hosts; p95 `ci / gate`
below the 0.68 baseline `[A]` for 3 consecutive trains; `pack:` never `P0-UNDERUTILIZED` under
queue pressure.

**P0 · Unwedge** *(2026-09-16, #3358)*. Mechanism, measured: a slow-dispatching `pull_request`
run holds `ci-<pr>`; GitHub supersedes only `in_progress`, so a run whose jobs trickle out over
hours on congested pools is never superseded; later pushes pend with 0 jobs, both required
contexts go absent, the PR reads BLOCKED and cannot enqueue. Two rules in `ci-unwedge`, reaper
cadence, run metadata + `verify-fleet-bin` only:
1. **Superseded head** — any `pull_request` run whose `head_sha ≠` the PR's current head is
   cancelled. Never `merge_group`. This is the rule that would have freed `ci-3354`.
2. **Aged with idle capacity** — `queued` > 30 min, regardless of job count, while
   `verify-fleet-bin` shows idle listeners on the job's label pool → cancel + report. Falsifier
   recorded: during the 2026-09-16 incident pools were busy, so this rule correctly does not fire.
Plus: `concurrency.cancel-in-progress` is `true` only for `pull_request`, `false` for
`merge_group`. *Done:* contract — after one reaper period no `pull_request` run exists whose
head is not its PR's head; mutation: a stale-head run on a test PR is cancelled within one
period; zero BLOCKED-with-pending=0 PRs across 10 trains.

**P0 · Pack** *(operator 2026-09-12; precedes everything while it fails)*. Every wakeup: sample
busy/online per host (§1), decide `intel pressure`, and act in minutes — arm every green PR, re-run
withdrawn CI, trim non-additive roadmap diffs, route arch-neutral work off intel, dispatch the
nightlies that produce T-1 evidence on the idle GPU boxes. Record the sample as a ledger record
`docs/build-ledger/<date>/<sha>-fleet-pack.json` `{at, intel_busy, intel_online, gx10_busy,
gx10_online, yoga_busy, yoga_online, mini_busy, mini_online, intel_pressure, aprender_runs_queued,
aprender_runs_live, idle_gpu_runners, verdict}` (5 records on `main` as of 2026-09-14, none yet with the `mini` fields).
*Done:* over the trailing 10 trains, in every sample with `intel_pressure=true`, median busy/online
≥ 0.8 on `yoga`, `gx10` and `mini`; the §7 `pack:` line is never `P0-UNDERUTILIZED` two wakeups in a row.
The structural lever is `workspace-test` leaving its `X64` pin (#3139: 82,085 tests in 795 s on gx10
against 1,000–6,000 s on intel); until it lands, `pack` is bounded by the short jobs.

**P0 · Reap** *(operator 2026-09-12: "automated disk clearing/reaping … managed by forjar and p0 if
it gets blocked")*. `ci-reaper` + `ci-disk-watch` (one script, per-host thresholds citing a
measurement) declared in `machines/{intel,gx10,yoga}/forjar.yaml`, applied, timers verified active;
`/mnt/nvme-raid0 → ~/eph-work/intel-mirror` declared on gx10/yoga so a reimage keeps the layout.
Admission at job start: `ci_self_hosted_preflight.sh` prints `disk_free_gb` (measurement first; the
gate threshold waits for 20 records, §3.3). *Done:* zero ENOSPC across 10 trains; a queue blocked on
disk is P0 and stops the current build row.

*(2026-09-16, P0·Reap:)* `REAPER_RUN_BUDGET_SEC` and every threshold are declared in the manifest, not script defaults
(2026-09-16: the budget worked via a `${VAR:-420}` default — declare it so it is a fact). The
sweep must never skip permanently: a "skip while any job live" guard on a CI host is
`skip == always` (428 skips, 0 sweeps, 319 GB); per-PR trees are attributable and TTL'd, the
shared registry keeps the live guard; N consecutive skips exits non-zero; one ledger line per run
(`free_gb_before/after, swept, skipped, reason`), resumable partial runs allowed.

**P0 · Instrument.** One ledger record per gate job: `sha host job queue_wait_s exec_s
total_s peak_rss_mb free_disk_gb exit`. Add `make build-report`: p50/p95 `total_s` and
`queue_wait_s` per host, 10 slowest test targets, and the §1 PRs-per-train number.
*Done:* ≥ 20 records; `make build-report` runs on a clean checkout.

**P1 · Two lanes.** `ci / gate` (every PR) = compile + **fast set**. `deep` (tags +
nightly) = everything else. Fast set is derived from the ledger, not opinion: rank tests by
failures-caught-per-second, keep the shortest prefix that caught every failure the full
suite caught; record the cut and the escape count. Runner: `cargo nextest` for the fast set
only if measured faster than `cargo test` on the same sha three times; record the delta
either way.
*Done:* one sha runs both lanes; a tag cannot publish with `deep` red; escape count
over the next 10 merges is in the ledger.

**P2 · Shard across hosts.** Infra PR first (`paiml/infra`): runner labels for `yoga` and
`gx10` plus intel's disk preflight (`PREFLIGHT_PRESSURE_GB`, `PREFLIGHT_CRITICAL_GB`) via
forjar. Route by capability, then idleness:

| Work | Host |
|---|---|
| x86 CUDA build + CUDA unit tests | `yoga` |
| aarch64 / sm_121 (`-DGGML_CUDA_ARCHITECTURES=121`) | `gx10` |
| clean-room, CPU differential — reserved, never shared with a perf run | `intel` |

Admission control replaces "never idle": start a job only when free disk ≥ 1 × p95 run size
for that job; andon below 2 ×. Queue depth per host is a ledger field.
*Done:* one PR's gate runs on ≥ 2 hosts; `build-report` prints p95 queue-wait per host;
zero ENOSPC in 10 trains.

**P3 · Ratchet.** After 10 trains of post-P2 data, land as fail-closed checks: p95
`ci / gate` ≤ 0.70 × baseline `[A]`; tag→publish ≤ 0.70 × baseline `[A]`. Nightly coverage
job, low priority, report-only: target = measured baseline + 2 points per ratchet toward
95 %. Re-evaluate each train; ratchet downward only, ≤ 10 % per step.

### §5.1 The debt tax — paid on every row, scheduled by none

*(measured 2026-09-14; the largest single time sink of that day)* The pre-commit
gate refuses ANY commit touching a file that carries a function over cyclomatic 30 / cognitive 25,
and `--no-verify` is banned. So a one-line fix in a debt-carrying file costs the decomposition of
every offender in that file. Measured on one day's work: **11 pre-existing violations** paid down to
land small fixes — worst cases cognitive 91 (`execute_llm_score`), 73 (`help_producer_truth::resolve`),
61 (`execute_llm_load`). None were in code that day's changes wrote.

This is a real cost and it is not optional, so it is scheduled rather than absorbed silently:

- The decomposition lands in the SAME commit as the fix that triggered it, and the commit message
  names each offender with its before-number. A stack green only at the tip hides which change
  broke what.
- Extraction, not rewriting. Prefer moving a whole block to a named function over clever
  restructuring; a behaviour-preserving refactor is the only kind admissible here.
- When a match arm is extracted, re-match behind a `let … else` that refuses by name. Never panic,
  never `unwrap()`.
- Repeated identical blocks are the cheapest win — one helper taking the two things that differ
  collapsed nine sites in one file.
- **Record per train:** `debt: files_touched <n> | violations_paid <n> | worst_before <n>`. After 10
  trains, if `violations_paid` is not falling, the gate's thresholds or the tree's debt is the build
  row, not the fixes.

*Done:* no train spends more turns on debt paydown than on the change that required it, measured
over a trailing 5 trains `[U]`.

## §6 Triage — continuous (§6.1), three surfaces (§6.2), capacity-bounded (§6.3) — and the T-5 reconcile, a HARD gate

**Measured 2026-09-13 (0.67.0 train):** 308 open issues, 424 opened vs 124 closed in 30 d, 95 open
issues already cited by a commit on `main`, 48 merged PRs of which 9 carried a closing reference,
880 remote branches of which 847 have no PR and 832 are unmerged, 13 DIRTY PRs listed and never
decided. Filing was the work product and closing was nobody's; the milestone pass assigned
attributes, not decisions. **Operator ruling: reconciliation is part of every train (kaizen), not
a separate chore.** A train that ships code and leaves its own tickets, branches and dead PRs
behind is not done.

- **Milestone = release scope, not a label.** The next-tag milestone holds only §1.5 scope,
  its children, and the gates a cut requires; everything else is retargeted to N+1/N+2
  mechanically at every pass (2026-09-16: ms 0.68 carried 277 open items; it carries 6).

### §6.1 Cadence — every wakeup, not once per train

"Once per train" is what let 20 of 34 open PRs carry no milestone at all on 2026-09-14, every one
of them opened in the preceding two days. A train is 48–72 h; a PR opened an hour after the pass is
invisible for the rest of it. Triage runs on the **P0 · Pack** wakeup, beside the fleet sample —
same cadence, same receipt, and it is equally P0 (operator: "ticket, pull requests, branches that
are not triaged are P0").

The per-wakeup pass is mechanical and bounded: anything opened since the last sample gets a
milestone and a label; `BEHIND` PRs get `gh pr update-branch`; `DIRTY` PRs get their conflicting
files named in a comment. Nothing here is a judgement call. The once-per-train pass keeps only
what needs the whole window: the §6.3 capacity check and the T-5 reconcile.

### §6.2 Three surfaces — issues, PRs, and BRANCHES

Branches were the unwatched one. Measured 2026-09-14: **107 remote branches, 35 with an open PR,
72 without**. Of those 72 — 32 younger than 7 d, **27 in a 7–14 d band that no rule looks at**, 13
already R-3-eligible. R-3 archives a branch with no PR and a tip older than 14 d, so work that
stalls on day 8 is invisible for six more days and then deleted without ever having been seen.

| Surface | Per-wakeup test | Fix |
|---|---|---|
| Issue | opened since last sample → milestone + label | assign; `untriaged_issues = 0` |
| PR | open → milestone; `BEHIND` → update-branch; `DIRTY` → conflicting files named | `untriaged_prs = 0` |
| Branch | no open PR and tip older than **7 d** | open a PR (draft is fine) or archive it now — do not wait for R-3 to delete it at 14 d |

A branch with no PR is not work in progress; it is work nobody can see. Seven days is the point at
which it must become visible or become history.

### §6.3 Prioritisation — capacity, not preference

§4 says scope is assigned after the fact: "0.67 contains whatever merged before the 0.67 cut, by
definition." That is right for what a train *contains* and wrong as a plan for what it *promises*.
With no capacity rule, a milestone is a dumping ground with a date on it.

**Measured 2026-09-14**, closure = **6.1 issues/day** over the trailing 7 days:

| Milestone | Open | Due in | Needs | Verdict |
|---|---:|---:|---:|---|
| 0.68.0 | 280 | 1 d | ~46 d | **over by 45 d** |
| 0.69.0 | 53 | 4 d | ~9 d | over by 5 d |
| 0.70.0 | 15 | 7 d | ~2 d | fits |

A date that is 45 days of arithmetic away from its content is not a commitment; it is a label, and
every number computed from it is fiction.

**The rule, and it is arithmetic so it stays inside §6's no-judgement-calls design:**

```
capacity(M) = days_remaining(M) × closure_rate_p50(trailing 7 d)
```

- `open(M) > capacity(M)` → the milestone is **OVERCOMMITTED**. Report it on the `triage:` line
  every wakeup. It is not an error; it is a number that must be visible.
- At T-0 of each train, an overcommitted next-milestone **spills**, lowest priority first, until
  `open(M) ≤ capacity(M)`. Spill order is by label — `P0` never spills, then `P1`, then unlabelled,
  then by age, oldest kept. The operator sets priority by labelling; **the arithmetic sets the cut
  line**, so no train needs a judgement call about scope.
- A `P0` set that alone exceeds capacity is a **stop** (§8): the release is over-promised at the
  priority level the operator controls, and only the operator can resolve that.
- `closure_rate` is MEASURED, never assumed. First train with fewer than 7 days of data reports
  `[U]` and spills nothing.

**Arrival is the other half.** Closure of 6.1/day against a ledger that grew net +149 in ten days
means the cut line moves further out every train no matter how it is drawn. §6's arrival/closure
ratio (R-5) is the control on that; capacity only decides what a date is allowed to claim.

### T-5 reconcile predicates — every one must read 0 in the receipt

| Predicate | Derive with | Fix |
|---|---|---|
| R-1 fixed-but-open | open issues cited by a `main` commit since the previous tag whose subject/body uses `Closes/Fixes/Resolves #N` or is `fix(...)`: must be ∅ | close with the commit sha as receipt |
| R-2 closing-reference | merged PRs since the previous tag whose body cites an issue without a closing keyword: must be ∅ going forward (guard `scripts/check_pr_closes_issue.sh` on the PR; `Refs #N` is allowed only with `no-close:` and a reason) | the guard refuses the PR |
| R-3 dead branches | remote branches with no open PR and a tip older than 14 d: must be ∅ | archive to `refs/archive/<branch>` (objects kept, reversible), then delete the head |
| R-4 dirty PRs | PRs `DIRTY` for more than one train: must be ∅ | a verdict per PR in the train: rebase (union tool, or by hand) or close with the reason |
| R-5 ratio | `closure / arrival` over the train window, recorded; ratchet as below | — |

The receipt is `{sha, window:[prev_tag, tag], R1..R5: {count, list}, actions:{closed, archived, rebased, pr_closed}}`
and is committed to the ledger; the autopilot writes `DONE` only after `check_reconcile.sh <tag>` reads
every predicate 0. Falsifier: seed one fixed-but-open issue (a closed test issue reopened) — the check
must go RED.

### §6.4 Lifecycle — stale, the closure ratchet, and what is recorded

Queues stabilise when closure ≥ arrival; age falls when WIP is capped. That is the whole
mechanism.

- Every issue opened since the last train: labelled and sized. **`untriaged = 0`** at the end
  of every pass — hard target today, needs no baseline.
- **Counted per surface, never as one number.** Measured 2026-09-14: issues were 319/320 triaged
  while PRs were 20 of 34 with no milestone at all. A single `untriaged` figure reported the clean
  surface and hid the breached one. The pass ends only when `untriaged_issues = 0` AND
  `untriaged_prs = 0`.
- **Triage is not disposal.** Over the trailing 10 days the issue ledger grew net **+149** while
  being ~100 % triaged, and **312 of 320 open issues were opened by the agent itself**. Classifying
  a finding does not close it. If `arrival > closure` for 3 consecutive trains, filing new findings
  as issues stops being free: the next train's build row is closure, not features.
- **Mechanical PR states are the pass's job, not a reviewer's**: `BEHIND` → `gh pr update-branch`
  (measured: 14 → 4 in one pass), `DIRTY` → named on the PR with the conflicting files.
- PR lifecycle, deterministic: no activity for 2 trains → label `stale` + comment; `stale`
  and no activity for 1 more train → closed. (~9 days, no discretion.)
- Record per train: `arrival closure open_prs age_p95 untriaged`.
- Ratchet: if `closure ≥ arrival` holds over any trailing 5 trains, cut the stale label from
  2 trains to 1. Total open count gets a target only after 10 trains `[U]`.

## §7 Session report — exactly this, in the PR body or the release

```
APR-RELEASE-001 | did=<TRAIN|BUILD|TRIAGE|NOOP> | train=v0.<N>.0 | verdict=<SHIPPED|SKIPPED|MERGED|NOOP>
fanout:  shards <n> hosts <intel,yoga,gx10,mini> | commit→tag <min> | ratchet p95 <min> | serial fallbacks <n>
nightly: NIGHTLY-RED <workflow> <consecutive fails> <issue|auto-opened #N> | none   (any nightly feeding a T-2 row; operator 2026-09-17)
train:   step reached <T-0..T-5> | skip reason <none|…> | cascade wall min <n|[U]> | attended min 0
build:   row <P0..P3|none> | PR <url|none> | records added <n>
gate:    p95 ci/gate <min|[U]> | max PRs/train <n|[U]> | queue p95 intel <s> yoga <s> gx10 <s>
pack:    intel <busy>/<online> | gx10 <busy>/<online> | yoga <busy>/<online> | mini <busy>/<online> | intel-pressure <yes|no> | verdict <OK|P0-UNDERUTILIZED>
triage:  arrival <n> | closure <n> | open PRs <n> (age p95 <d>) | untriaged issues <n> prs <n>
branches: total <n> | no-PR <n> | no-PR >7d <n> | archived this pass <n>
capacity: milestone <M> open <n> | closure/day <x|[U]> | capacity <n> | verdict <FITS|OVERCOMMITTED by <n>d> | spilled <n>
matrix:  pairs <n> | red <n> | known-red <n> | stale-known-red <n> | examples started <n>/<n>
debt:    files touched <n> | violations paid <n> | worst before <n>
quorum:  rounds <n> | width <n> | verdicts <a/b/c> | overridden <yes|no> | lattice <ok|prose>
beats:   won <n> | parity <n> | loss <n> | measured-on <published|dev> | crux ✅ <n> 🔨 <n> ❌ <n|[U]> | drift <n>
ontology: types <n>/16 | extractors <n>/14 | anchored <n>/<total> | shaped <n> | bindable-unanchored <n|[U]> | deltas <n>/<sweep PRs> | row merged <ONT-x|none: reason> | upstream sha <12hex|DRIFT>
reconcile: R1 fixed-open <n> | R2 no-close <n> | R3 dead-branches <n> | R4 dirty>1train <n> | R5 closure/arrival <x> | receipt <path|MISSING>
scope:   §1.5 <ticket> | obligations <passed>/<total> | merged <yes|no> | blocker <none|…> | series <PR1 eta> <PR2 eta> <PR3 eta> vs andon <ts>
queue:   open PRs <n> | WIP cap <13> | group size <n> | ejections since last report <n> | roadmap.yaml diffs in open PRs <n>
pins:    tool-pin-check tree <sha> | matched <n> drifted <n> unchecked <n> | hook <ARMED|NOT-ARMED>
tree:    HEAD <sha> origin/main <sha> behind <n>   # every measurement in this report was taken at this tree
stops:   <none|list>
next:    train eligible at <timestamp>
```

## §8 Stop conditions — stop and report, do not work around

- Two consecutive SKIPPED trains, or ≥ 72 h with no runnable train → andon.
- `deep` red on a tag → no publish.
- Cascade dry-run non-zero → stop before any real publish.
- Measured max PRs/train < 10 `[A]` → stop cutting trains; finish P1/P2 first.
- `yoga` or `gx10` has no runner unit in live `forjar.yaml` and the infra PR is unmerged → stop at P2.
- Fewer than 20 ledger records → stop at P0.
- `yoga`, `gx10` or `mini` under-utilised (§1) while intel has queue pressure, two wakeups in a row → P0:
  stop the current build row, pack first (arm, route, reap), report the `pack:` line.
- A runner box at or below `REAPER_CRITICAL_GB` free, or any job dead on ENOSPC → P0: reclaim
  over SSH now, then land the forjar change that makes it automatic, before anything else.
- A host DECLARED a full-time build host sits at 0 % occupancy for a whole session while any
  queue has pressure → P0: that is a routing or job-class defect, not spare capacity. (`mini`
  measured at 0.0 % all of 2026-09-14 while declared full-time in #3205 — its ceiling is the
  macOS-capable job classes, not capacity.)
- A `deep` known-red list names a pair that now PASSES → stop: the list is stale and the lane
  is asserting something that is no longer true (§4.1).
- A design fork appears that §6 cannot decide without judgement → §10, not a coin flip and not a
  question to the operator.
- The `P0`-labelled set of the next milestone alone exceeds its capacity (§6.3) → stop: the release
  is over-promised at the one priority level the operator controls, and only the operator can cut it.
- Any T-step job with a single-host `runs-on`, or T-3 without the fan-out ledger → stop; the train
  does not proceed on a serial chain (rule 14).
- Any `ci.yml` job with a single-host `runs-on`, or a pooled box idle under queue pressure → stop (rule 16).
- A published tag is never moved or deleted; a fix after tagging is the next patch version.
- A gate that has never been green on a real target may not block; it runs measurement-only until
  its first-green receipt exists (rule 15).
- Any queue entry ejected twice on the same cause → stop, five-whys, no third re-queue.
- Any measurement whose tree line shows `behind≠0`, or taken outside a per-session worktree →
  discard and re-measure from `origin/main`.
- Any step would need an invented threshold, a hand-squashed batch, a hand-cancelled run with started
  jobs, a label edit to dodge a runner, an env bypass of a prejob gate, `--allow-dirty`, or a host CONFIG change made over SSH
  instead of through forjar (§3.5 — SSH for measurement and for unclogging is expected) → stop.
- The upstream ontology spec, at the version this spec cites, is not reachable at a committed sha
  (2026-09-14: infra `main` has v3.1; v4.3, sha256 `512a16d5e09c…`, is an untracked draft) → stop: no lane on another host can read the premise, and every ontology
  verdict is unverifiable by construction. Fixed in `infra` (ONT-P), not here (§11).
- A sweep PR (§11.1) merged with no `ont-delta:` line → P0: a finding went into a prose sink and
  nothing mechanical can read it back.
- An `ont.*` counter in `contracts/lint-baseline.json` moved outside `make ont-ratchet` → RED
  (ONT R-6).
- A gate special-cases `entity.type` beyond `proof` applicability → RED (ONT R-17, ONT F-25).
- A beat contract RED on the published binary, or a `beats:` line saying `measured-on published`
  from a dev build → stop the claim: the scoreboard says LOSS or `[U]`, the beat is the next build
  row, and nothing is published ahead of its measurement (`docs/BEATS.md` withdrawal precedent).

## §9 Budget and the one risk to measure first

K̂ = 14 sessions (10 trains to `0.76` + P0, P1, infra, P2) · K = 18 · andon at 16 sessions
or 2 consecutive skips `[A]`.

The 82-crate cascade is automated (ruling 2026-09-13). **Measured on the `0.67` train**
(`docs/build-ledger/2026-09-13/e45eaab47-lambda-vector-train.json`): `t4_wall_minutes: 70`,
`attended_minutes: 0`. That is 3.5× the ~20 min `[A]` line this paragraph drew before any
measurement existed, so by its own rule the cascade — not the build — is the next kaizen target.
Where the 70 minutes go is `[U]`: record per-crate wall on the `0.68` train before naming a fix.
The ratchet takes §5 P3's shape (≤ 0.70 × baseline, `[A]`) once three trains have measured it.

**Revised 2026-09-15/17:** +3 sessions for P0·Frag/Runners/Pins, +N for §1.5 scope (N from the epic's
`impl-estimates.jsonl` rows once they carry `unit`; `[U]` until then) → K̂ = 17 + N, K = 22 + N. **Measured on the
`0.68.1` train** (`docs/build-ledger/2026-09-17/1661c7138-lambda-release-train-v0.68.1.json`): cascade 2458 s
(22:00:38Z→22:41:36Z, 71 uploads, 11 crates waited on HTTP 429 — the registry's rate limit, not the build, is now
the cascade's floor), `attended_min: 0`; release-commit→tag 41 min; release-commit→published 8 h 16 min across
four serial clean-room iterations. The kaizen numbers from 0.69 on are cascade wall-clock and release-commit→tag
(rule 14 ratchet), both from the ledger.

## §10 Decision procedure — for the forks §6 refuses to make

§6 is deliberately "no judgement calls". Design forks still occur — four ways to fix one broken
feature declaration, restore-vs-delete a dead GPU path — and they are not triage. They go here.

**Fan out through agy, not through more Claude subagents.** Lanes cost agy credits; orchestrator
turns are the scarce budget. `lane-group.sh run --out-dir D --width 3 --scratch -- agy-lane.sh
--mode plan --repo-root <toplevel> --prompt …`. Width 3 is the default; 10 is the cap.

**The brief carries the measurements, not the question.** Name the files, the line numbers, the
acceptance command, and every option including the ones you dislike. A lane that has to go and
measure the premise will measure it differently from the next lane, and the vote becomes noise.

**Plant the trap.** Include one question whose obvious answer is wrong — a rename that looks
mechanical but is not. A quorum that misses it has told you how much its other answers are worth.
(2026-09-14: `Lz4WarpShuffleKernel` → `Lz4WarpCompressKernel` reads as a rename and is a
literal-only encoder swapped for a real match-finder. All three lanes caught it.)

**A verdict is a claim until re-run.** The orchestrator re-executes the acceptance command itself
and records both columns. A lane reporting "all 9 selections green, 84 tests pass" is evidence of
what the lane believes.

**A premise error voids the vote, and the fix is another round — not the orchestrator's judgement.**
Measured on #3179: round 1 voted 2/3 to make a dead path compile. Three facts were then verified by
reading the tree (a stub decompress kernel, a launched entry point that does not exist, a host-memory
API that never existed here). Round 2 with those facts overturned round 1 unanimously. **Re-run the
quorum with the corrected premise; do not silently substitute your own conclusion.**

**Overriding the majority is allowed exactly once and must be recorded.** Only on a fact no lane in
that round had, stated in the commit and the PR, with the losing option's own argument quoted. On
#3179 round 2 split 2/1 to delete; the minority option shipped because the crate's default build is
green and the module in question is live public API — both deleting lanes had argued from "a dark,
unused feature". When the fact is not decisive, the majority wins and your discomfort is not
evidence.

**Prefer the reversible option when the vote is close.** A `compile_error!` can be deleted later; a
deleted module and its public types cannot be recovered from a reviewer's memory.

**Record per train:** `quorum: rounds <n> | width <n> | verdicts <a/b/c> | overridden <yes|no>`.


## §11 Ontology kaizen — a surface sweep ends in `pv`, or it did not end

Upstream: `ONT-001 v4.3` (`infra/docs/specifications/paiml-ontology.md`, sha256 `512a16d5e09c…`).
Its §6 assigns **aprender** every row but three. This section is how those rows get worked
continuously instead of in one heroic push, and what the train owes the ontology every time it
sweeps a surface. Upstream ids are written `ONT R-n` / `ONT F-n` / `ONT §n` here; a bare `§n` is
this spec; a bare `R-1`–`R-5` is a T-5 predicate (§6.3) and an unhyphenated `R1–R6` is the publish
preflight (§3.7).

### §11.0 Measured baseline — `main` @ `fa6e35f23`, 2026-09-14

| ONT-001 requires | aprender has | derive with |
|---|---|---|
| contracts may carry `entity:` / `shape:` / `evidence:` | **0 / 0 / 0** of 1818 | `grep -rlE '^entity:' contracts --include='*.yaml' \| wc -l` |
| `pv census` — one cardinality, `by_entity_type`, `by_anchoring` (ONT-1) | absent | `pv --help` |
| `pv extract` → `contracts.nt`, the only triple producer (ONT R-18) | absent | `pv --help` |
| `crates/aprender-contracts/src/ontology/` (ONT decision 4) | directory does not exist | `ls` |
| 16 entity types named (ONT §0.0); 14 extractors named, 7 in v1alpha1 scope (ONT §3.7) | **0** | — |
| `contracts/lint-baseline.json` with `armed_gates` (ONT §3.9) · `make ont-ratchet` (ONT R-6) | neither exists | `git ls-files contracts/lint-baseline.json` · `grep -n '^ont-ratchet' Makefile` |
| aprender implements every ONT row (ONT §6) | **1 of 17** merged — ONT-2a, #3224 | `git log --grep='ONT-' origin/main` |
| an aprender ticket per row | 1 of 17 (#3222, closed, no milestone) → epic #3269 | `gh issue list --search 'ONT in:title' --state all` |
| `pv kaizen` **is** the kaizen loop | code-only: bindings, call sites, E0/E1/E2 assertions | `crates/aprender-contracts-cli/src/commands/kaizen.rs` |
| the upstream spec itself | infra `main` tracks **v3.1** (414 lines, `f1269d0`); **v4.3** — the version this section is measured against, sha256 `512a16d5e09c…` — is an untracked draft on one box | `git -C ../infra show origin/main:docs/specifications/paiml-ontology.md \| head -1` vs `head -1` of the local file |

Two of these are the whole point of this section. **`pv kaizen` is blind to every surface that is
not Rust**: the train sweeps features, examples, README, `CLAUDE.md`, workflows, model files and
CSVs, and the loop that is supposed to improve on each sweep cannot see any of them. And **the
upstream version this section cites is untracked** — `main` has v3.1, the v4.3 draft lives on one
box — so a quorum lane on gx10, yoga or mini cannot read the premise at all, and every ontology
verdict it returns is unverifiable by construction.

### §11.1 The rule — a sweep closes with an ontology delta, not a paragraph

A **surface** is anything the train already sweeps: the `(crate, feature)` matrix (§4.1), the
examples (§4.2), the `apr` command registry, `contracts/`, the published docs, `.github/workflows/`,
model files, and the three triage surfaces (§6.2). A **sweep PR** is one that writes a finding into
a prose sink — operationally, one touching any of `.github/workflows/night.yml`,
`docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` (this spec — the sink the train writes its findings into; a
*design* spec such as a `*-MASTER.md` is the finding's home, not a sink, and touching one is not a
sweep — aprender#3535, three maintainer PRs red for a delta that did not exist),
`contracts/apr-cli-commands-v1.yaml`, `README.md`, `CLAUDE.md`, or a
known-red list anywhere.

A **delta** is one of exactly four things, any one of which closes the sweep:

1. an **entity type + extractor** registered in Σ, with its `pc_extract` planted defect
   (ONT §3.7, ONT R-3);
2. a **shape** whose violation *is* the defect class just found, with the found instance as its RED
   fixture;
3. a **verdict reason** added to the ONT-6 `Unknown{…}` set when the sweep could not decide — never
   a new way to `Pass`;
4. a **`resolves:` target** (`path` · `symbol` · `ci-step` · `contract` · `comply-rule`) that turns a
   prose claim into a checkable one.

A sweep that ends in a known-red list, an issue body or a spec paragraph and nothing else **is not
finished**: it produced a fact only a human re-reading can use. The live example is this spec's own
§4.1 — 100 red pairs encoded as an `awk` matcher inside `night.yml`, no contract, no shape, and a
list that must be re-derived by hand before every undraft. That is the shape of debt §11 exists to
stop creating.

Every sweep PR body carries one line, in the form `scripts/check_pr_closes_issue.sh` already
enforces for `Closes #N`: `ont-delta: <type|shape|reason|resolves> <id>` or
`ont-delta: none <reason>`. Absent is a PR-body lint failure, not a review comment. (#3268, the PR
that adds this section, is a spec paragraph and carries `ont-delta: none`.)

### §11.2 The ratchet — four counters up, one down, `make ont-ratchet` only

In `contracts/lint-baseline.json` beside `armed_gates` (ONT §3.9) — the file and the `make` target
land with ONT-1/ONT-6; neither exists today:

| Counter | Direction | Today |
|---|---|---|
| `ont.entity_types_registered` | ↑ | 0 |
| `ont.extractors_implemented` | ↑ | 0 |
| `ont.contracts_anchored` (`entity:` present) | ↑ | 0 |
| `ont.contracts_shaped` (`shape:` present) | ↑ | 0 |
| `ont.unanchored_but_bindable` (ONT R-5: kernel-kind with a binding, or naming a file) | ↓ | `[U]` |

Moves only through `make ont-ratchet` (ONT R-6). A PR that lowers an ↑ counter or raises the ↓ one
is RED. No bulk rewrite: ≤5 corpus files per PR except a named ratchet touch (ONT R-5, ONT F-8).

### §11.3 Cadence — one row per train, computed everywhere, armed per repo

- **Nightly**, in the same job as §4.1/§4.2 and under the same vacuity floors: recompute the five
  counters. Until `pv census` exists they come from the greps in §11.0 and are marked `[U]`, never
  silently reported as 0.
- **One ONT row per train.** 16 rows outstanding; trains run every 2–3 days ⇒ ≈40 days to ONT-10
  `[A]`. That is a derived horizon, not a promise — the rows spill by §6.3 like any other work.
- **An ONT row PR obeys ONT §0.2**: never pushed while a release-titled run is in progress, so a
  moving branch never races T-0..T-3 of a train.
- **New gates arrive unarmed** (ONT R-8, ONT §3.9). An ONT gate lands computing everywhere; arming
  is a later, separate PR whose body shows the counter it moved.
- **T-5 carries the `ontology:` line but does not gate on it** until
  `ont.extractors_implemented ≥ 1`. A gate over an empty extractor set is the ONT R-2 vacuity class
  — `Unknown`, never `Pass`. The ratchet check (§11.2) is separate and arms from its first commit:
  it is about the JSON file, not the extractors.

### §11.4 Why this makes the quorum more effective — the premise stops being prose

§10 already requires the brief to carry the measurements. #3179 round 1 is what it costs when it
cannot: three facts had to be read out of the tree by hand before the vote meant anything, and the
round was void. An ontology is the mechanical form of that rule.

- **Premises cite ids.** A brief names `contract:<id>`, `symbol:<crate>::<path>`, or a
  `(crate, feature)` pair — something every lane resolves identically — not a sentence every lane
  re-measures differently. #3179's round-1 premise named a launched kernel entry point that does
  not exist in the tree; as `resolves: symbol` it returns `Unknown{…}` at extraction, before any
  lane votes, and an `Unknown` premise cannot reduce to `Pass`.
- **Verdicts are ONT-6 lattice elements**: `Pass · Unknown{<reason>} · Fail`, reason from the closed
  set (ONT §3.4). A lane answering outside the lattice has answered `Unknown{Prose}`: it does not
  arm, and it does not count toward a majority.
- **Reduce is `meet = min`, not a vote count.** One `Fail` is `Fail`; `Pass` ∧ `Unknown` is
  `Unknown`. A 2/3 majority with one lattice-invalid verdict reduces to `Unknown`, not to a majority.
- **The plant is a lattice element too.** `pc_shape` / `pc_extract` are the mechanical form of §10's
  planted trap: a round whose planted defect was not caught is `Unknown{PositiveControlFailed}` and
  the round is void, by rule rather than by the orchestrator noticing.
- Every quorum receipt records `premise=<ids>` and `verdict=<element>`.

### §11.5 Report lines

Two lines change in §7, and only there: `ontology:` is added, and `quorum:` gains
`| lattice <ok|prose>`.

### §11.6 Stop conditions

Four added to §8, and only there: an unreachable upstream sha; a merged sweep PR with no
`ont-delta:`; an `ont.*` counter moved outside `make ont-ratchet`; a gate special-casing
`entity.type` beyond `proof`.

### §11.7 Falsifiers

| # | Rule | Assertion | Mutation |
|---|---|---|---|
| FR-1 | §11.1 | every sweep PR body carries `ont-delta:` — a `scripts/check_pr_closes_issue.sh`-class predicate, run where that one runs in `ci.yml` | delete the line from a sweep PR → RED |
| FR-2 | §11.2 | the five counters are recomputed from the tree and compared to `lint-baseline.json`; a hand-moved counter is RED regardless of arming | hand-raise `contracts_anchored` → RED |
| FR-3 | §11.3 | an ONT gate's landing PR arms nothing | arm it in the landing PR → RED |
| FR-4 | §11.4 | a quorum receipt whose verdict is not an ONT-6 element never reduces to `Pass` | record "looks fine" as a verdict → reduce RED |
| FR-5 | §11.0 | the sha256 pinned above matches the upstream file at read time | edit upstream → `upstream sha DRIFT` |

## §12 The chain of reasoning — why this loop never terminates, and the four things it moves

This section explains the spec instead of adding to it. Read it as an argument: each step says
what the loop does, why, which section is the mechanism, and what would falsify the step. Numbers
are dated or marked `[U]`/`[C]`/`[A]`, as everywhere else.

### §12.1 Why it runs forever

1. **The selector is total.** §0 is "first match wins" and row 4 always matches: emit the §7
   report, exit 0. There is no state in which a session has nothing legal to do, so there is no
   state in which the loop ends. Idle is a report, not a stop. *Falsified by:* a session that
   exits without a §7 block.
2. **A stop stops the session, never the loop.** Every §8 line names the mechanism whose absence
   caused it — a forjar unit, a ratchet target, a tracked spec, a comply rule. The next session's
   first match is the row that lands that mechanism, so the same stop cannot recur. That is the
   difference between kaizen and a retry loop. *Falsified by:* one §8 line stopping two consecutive
   sessions with no PR between them.
3. **Every counter is a ratchet.** The known-red list only shrinks (§4.1); `armed_gates` never
   shrinks (ONT §3.9); `ont.*` moves only through `make ont-ratchet` (§11.2); the stale label
   tightens only after closure ≥ arrival held five trains (§6.4); P3 tightens ≤ 0.70× and only
   downward (§5). Session N cannot undo session N−1, so the direction of a thousand sessions is
   the direction of one. *Falsified by:* any counter moving the wrong way outside its named target.
4. **No number is a guess, and a guess that becomes measurable is replaced.** §3.3: every threshold
   cites its command or is `[U]` and does not gate. The worked example is §9 — "~20 min `[A]`" was
   drawn before a cascade had ever been timed; the 0.67 train measured 70, and §9 now says so and
   names the cascade as the next target. That is the loop reading its own ledger and moving.
   *Falsified by:* a threshold that gates while marked `[U]` or `[A]`.
5. **The ledger is the memory, and it is append-only.** §3.6: one file per run, never a shared
   file. A session starts by reading it (§2, "verify at HEAD before writing anything") and ends by
   writing to it (§7). Nothing the loop learns depends on a session remembering it. *Falsified
   by:* a rule in this spec whose measurement cannot be derived from `docs/build-ledger/`.
6. **The clock cuts the train, not the scope.** §1: a tag every 48–72 h; §4: scope is whatever
   merged before the cut. There is no "not ready" — a cut that cannot go green is SKIPPED against
   that HEAD and the next train is 48 h away, so the loop cannot stall waiting for a feature.
   *Falsified by:* two consecutive SKIPPED trains — §8's andon, the one place the loop escalates
   instead of continuing.

### §12.2 What it moves — four axes on one substrate

Every wakeup is: pack + triage in minutes (§0 row 0) → the train if one is due (row 1) → else one
build row (row 2) → else the once-per-train pass (row 3) → else the report (row 4). Each axis
below has its measured position, its mechanism, its ratchet and its §7 line. The ledger carries
all of them, and the ratchets read the ledger.

**A. The repo.** Position, 2026-09-14: 100 of 430 feature pairs RED and never built (§4.1); 727
integration binaries against 73 the CI ever runs (the dark-targets triage) `[C]`; 11 complexity
violations paid in one day (§5.1); issues +149 net in 10 d at 6.1 closures/day (§6); 0 of 1818
contracts anchored (§11.0). Mechanism: every sweep surfaces one class of dark defect, fixes it in
the same PR, and pays the debt tax on the file it touched. Ratchets: known-red shrinks,
`violations_paid` falls, closure ≥ arrival, `ont.*` up. Lines: `matrix:` `debt:` `triage:`
`branches:` `ontology:`.

**B. The released binaries.** Position: v0.67.0 shipped 2026-09-13 by the automated train,
`attended_minutes: 0`, cascade 70 min; CUDA `apr` assets for arm64 and x86_64 are required on every
tag (operator, 2026-09-10). Mechanism: T-0..T-5 (§4) on the clock. The binary improves at exactly
the rate the repo does, because scope is what merged. The improvement that reaches a user is
measured on what the user installs — `cargo install aprender`, then the dogfood on *that* binary
(`apr-dogfood` G13 with `DOGFOOD_ALLOW_UNPINNED=1`, its deliberate crates.io mode) — in flight as
#3202, the post-publish phase. Until it lands, a train's binary claims are claims about the dev
build. Ratchet: §5 P3, tag→publish ≤ 0.70× baseline. Line: `train:`.

**C. The competitors.** Before this section the spec had no competitor line: the train shipped
binaries and nothing in it said where they stand. Two instruments exist and both are ratchets.

- **CRUX** (`docs/specifications/crux-competitive-research-ux-workflows.md`): 275 user stories
  across 9 monitored competitors — Ollama, llama.cpp, PyTorch, Hugging Face, vLLM, OpenCLAW,
  ecosystem interop, HF kernels-community, the APR-QA playbook. Coverage at v2.2 intake `[C]`:
  ✅ 39 · 🔨 80 · ❌ 156 (56.7 % missing); registry drift is open as #3172 (0.69.0).
  **Approaching** is ❌ → 🔨 → ✅ one story at a time, demand tier 5 first. Those are tickets, so
  §6.3's capacity arithmetic already schedules them and no judgement call is needed. CRUX's
  declared falsifier `FALSIFY-CRUX-010` is not found under `crates/` or `scripts/` on `main`
  (2026-09-14): the coverage count is `[U]` until it is, and landing it is the first CRUX row.
- **BEATS** (`docs/BEATS.md`, 16 `contracts/beat-*.yaml`): Pillar 4 has Ollama GPU decode at
  **parity** — 1.015–1.109× on sm_89, the 1.371× headline withdrawn, `beat_threshold: 0.9000` a
  no-collapse floor and not a win; llama.cpp c=1 decode a narrow loss (1.55× faster); fail-closed
  correctness WON 10/10. **Surpassing** is a beat contract per competitor verb whose threshold is a
  floor first and moves above 1.0 only on three agreeing medians, measured on the published binary
  on the host class the user gets. A CPU-only published binary made the llama.cpp ratio
  uncomputable (APR-PARITY-001) — which is why axis B's post-publish dogfood precedes any ratio.

Rules already in force and kept: never blanket-concede speed; never claim ahead of the
measurement; a withdrawn claim stays withdrawn until re-measured; a beat RED on the published
binary is the next build row and the scoreboard says LOSS. Line: `beats:`.

**D. The fleet.** Position: intel is 16 workers on one 32-core box shared across repos; gx10 is
GB10 sm_121 aarch64; yoga is RTX 4060 sm_89; mini is Apple M4 on the macOS job classes, declared
full-time in #3205 and measured at 0.0 % all of 2026-09-14. Mechanism: §1's packing rule (≥ 80 %
of yoga, gx10 and mini under any intel pressure — a P0), §5 P0·Pack every wakeup, P2 routing by
capability then idleness (x86 CUDA → yoga; sm_121 → gx10; clean-room → intel; macOS classes →
mini), the merge queue 3-parallel (§3.4). "As quickly as possible" is one equation,
`max PRs/train ≈ 3 × 72 h / p95 gate`: gate latency *is* release throughput, so an idle box beside a
queue is lost release time, never spare capacity. Ratchet: P3, p95 ≤ 0.70× baseline. Line: `pack:`.

**E. `pv` and the ontology — the substrate under A–D.** §11: `pv kaizen` is the loop's own
improvement instrument and today sees only Rust. Every surface in A–D becomes an entity type with
an extractor and a shape, so a sweep's finding is a `Fail` the next sweep computes instead of a
paragraph a human re-reads, and a quorum's premise is an id every lane resolves the same way.
Sixteen rows, one per train, every gate landing unarmed. This is what lets the loop's *judgement*
improve and not only its counters: each round's premise is checkable by the next.

### §12.3 How the axes compose

A improves B by construction — scope is what merged. B is claimable only through C's instruments,
on the published artifact. C's ❌ stories are A's tickets, scheduled by §6.3's arithmetic. D sets
the rate of all three, `3 × 72 h / p95`. E makes each finding of A–D machine-readable, so the next
session — or the next quorum lane — starts from a fact and not a memory. Remove any one and the
loop still runs (§12.1); it learns slower. That is why none of them is a stop condition and all of
them are report lines.

### §12.4 Falsifiers

| # | Step | Assertion | Mutation |
|---|---|---|---|
| FX-1 | §12.1.1 | every session ends with a §7 block, `NOOP` included | end one without → receipt lint RED |
| FX-2 | §12.1.2 | no §8 line stops two consecutive sessions with no PR between them | repeat a stop → andon |
| FX-3 | §12.1.3 | every counter on a §7 line has a named ratchet target and moves one way | move one backward by hand → RED |
| FX-4 | §12.2.C | `beats:` says `measured-on published` only when the dogfood ran on a `cargo install aprender` binary (G13: no embedded SHA, `DOGFOOD_ALLOW_UNPINNED=1`) | report `published` from a dev build → RED |
| FX-5 | §12.2.C | a beat's status in `docs/BEATS.md` matches its contract | change the status without the contract → `readme_contract` RED |
