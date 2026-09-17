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
Earlier revisions kept where still true.
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
| **0** | `yoga` or `gx10` is under-utilised (§1 packing rule) while intel has queue pressure | **P0, minutes not a session:** arm every green PR, route what can leave intel, reap disk (§5 P0·Pack, P0·Reap); record the `pack:` line; then continue to the first matching row below |
| **1a** | a §1.5 release-critical ticket for the next tag is open and not merged | **one feature PR on it** (TDD, contract, `apr-dogfood` slice) — this row outranks every §5 row and never yields a slot to them; a second session may run §5 in parallel |
| 1 | ≥ 48 h since the last tag on `main` **and** no SKIPPED record for the current HEAD **and** §1.5 scope for that tag is merged | run the train (§4) |
| 2 | else a §5 row whose *Done* test fails at HEAD | do the first such row, one PR |
| 3 | else the last train (shipped or skipped) has no triage record | do the triage pass (§6) |
| 4 | else | emit the §7 report, exit 0 |

Nothing in this spec asks a question. Running it ten times a day is safe.

## §1 Goal

Ship a tag every 48–72 h (`0.67 → 0.68 → …`) **on a clock, not on scope**, and keep
shrinking the wall-clock from *PR opened* to *tag published* so the clock stays cheap.
The objective is elapsed time and green trains. **Packing rule (operator, 2026-09-12, verbatim):** "these two boxes: yoga and gx10 should be always 80% full of PRs from aprender if ANY queue pressure on intel … not acceptable to have slow releases when boxes are idel". Utilisation is not the goal for its own sake; an idle GPU box next to an intel queue is lost release time and is a **P0 defect**, not a state to tolerate. Measure it every wakeup:

```
make -C machines/fleet-hosts verify-fleet-bin HOST=<h>   # live listeners + busy, per host, via fleet-bin.sh (/proc scan, comm==Runner.Listener)
gh run list --repo paiml/aprender --status queued        # intel pressure from the queue side (no runners API — operator-rejected 2026-09-15)
intel pressure  = any aprender job queued, or a workspace-test running, on intel
under-utilised  = intel pressure AND (busy/online < 0.8 on yoga OR on gx10)
```
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
- **0.68.1** — **SHIPPED 2026-09-17**: 74/74 on crates.io, cascade 22:00:38Z→22:41:36Z (2458 s, unattended, 429-only retries); clean-room on the tag = B2-cpu paiml/infra run 35255889257 + B2-gpu paiml/aprender run 35268563585; records in `docs/audits/release/v0.68.1/`. B2-gpu runs from paiml/aprender (`b2-gpu.yml`) because the org GPU runner groups admit that repo only; #3466 needed 48g + 8 threads (one test holds 32.6 GB — a thread cap alone cannot fit 32g); publish preflight R6 now judges the cycle, not the shape (#3469). Chain as run: clean-room on tag (B2-cpu + B2-gpu, `--no-fail-fast`) → dry-run receipt committed → `publish_strict.sh` (derived acyclic order, one crate per call, stop on first non-zero, 429/5xx ×3) → post-publish `cargo install apr-cli --version 0.68.1` on intel + gx10 → receipts on the release → SHIPPED. Gate-side fixes for it: infra#654 (inheritable overlay), #3466 thread cap, #3464 A3 install, #3465 B2-gpu split.
- **0.69** — first PR is P0·Fan-out (rule 14), second is P0·PR-Fan-out (rule 16), before any scope work. §1.5 scope: #3090 GPU DeltaNet, #3208, #3341, #3369, #3464/#3465/#3466 test hygiene, mini neutral shard, unwedge rule 3, TIERS deletion, pmat 3.41.0 pin.

### 0.68 critical path (2026-09-16, operator) — DONE 2026-09-16T22:58Z

One line, in order; every session slot not on queue-unblocking work is on it:

    #3331 llama.cpp pin (MERGED) → parity evidence series PR1 #3354 → PR2 → PR3 #3356
    → #3114 out of draft (contracts discharged, Closes #3091 #3303) → #3091 merged → T-0 cut

Andon: **2026-09-18T12:19:39Z** if #3091 is unmerged (§8). The train waits for §1.5 scope; it
does not cut a 0.68 that is not Qwen 3.5. Report per-PR ETA against this clock every interval.
Push the next PR in the series the moment the previous one enqueues; disk is no longer the
constraint (gx10 ≥ 1 TB free after P0·Reap).

Coupling, one line:

```
max PRs per train  ≈  72 h  /  p95 `ci / gate` wall-clock      (upper bound)
```

One PR in CI at a time means gate latency *is* release throughput. Compute this on every
train. p95 is `[U]` until §5 P0 lands.

## §2 Ground truth — verify at HEAD before writing anything

| Fact | Source of truth | Mark |
|---|---|---|
| `intel` — clean-room runner, 8 concurrent, memory-bound, 3.6 TB NVMe | `infra/machines/intel/forjar.yaml` | `[V]` snapshot |
| `yoga` — CI runner, RTX 4060 8 GB, 32 GB RAM, 10G NIC needs `bolt.service` | `infra/machines/yoga/forjar.yaml` | `[V]` snapshot |
| `gx10` — aarch64 GB10, sm_121, 120 GB unified; **not** a documented general runner | `infra/machines/gx10/forjar.yaml` | `[V]` snapshot |
| `main` protected; required check literally named `ci / gate` | org ruleset | `[V]` |
| Last tag = `git describe --tags --abbrev=0` on `main`; next minor = that + 1 | git | `[V]` live |
| p95 `ci / gate`, tag→publish, attended cascade time | — | `[U]` unmeasured |
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
   inside a session still means ≤ 3 read-only subagents, never two sessions merging.
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
| **T-0 Cut** | cut sha = `main` HEAD; bump minor; `CHANGELOG` from merged PR titles since last tag | tag `v0.N.0` exists — **T-2 preflight before the cut (operator 2026-09-17, from 0.69):** `scripts/dogfood.sh --phase pre-publish` runs on `main` HEAD *before* the bump PR opens; the bump PR is **refused at arm time** without a GO receipt for its parent sha; the release commit moves **zero** times. Measured need: 0.67 stopped 3×, 0.68 stopped 4× at T-2 after the cut, every red a release-phase-only gate (perf041 marker age, `pv validate` over all contracts, CB-200, C14) |
| **T-1 Deep** | `ci / deep` green on cut sha: full tests, doctests, `--no-default-features`, feature matrix, GPU, every `cargo run --example` | green run recorded for this sha |
| **T-2 Dogfood** | `apr-dogfood` skill go/no-go receipt; `apr-cookbook` current; release notes generated | receipt exists for this sha — **plus one row (2026-09-16, #3366):** `install.sh --version v0.N.0` on a clean intel (x86_64) and a clean gx10 (aarch64) resolves the asset, the sha256 verifies, and `apr --version` prints `v0.N.0`; both hosts, both rows green, or T-2 is red. The row fetches from the URL the script itself advertises, so a 404 there is a T-2 failure, not a docs nit — **Nightly producers (operator 2026-09-17):** any nightly that feeds a T-2 row (today: `cuda-nightly.yml` → PP-26 marker) and fails **≥ 2 consecutive** runs auto-opens an issue on the next milestone and shows as `NIGHTLY-RED` in §7; the T-2 receipt records the witness host's provenance when it differs from the producer's (0.68: lambda sm_89 witness; gx10 producer red since 09-12, #3096 stays 0.70) — **Inherited T-2 (operator 2026-09-17):** T-2 on the release commit inherits the parent-sha preflight GO when the bump diff touches only `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, `crates/*/Cargo.toml`; the receipt records `inherited_from: <sha>`; any other path → full T-2. First-chain rule: the clean-room on the tag runs through B2 and past it to the end; if a gate after B2 stops, the gate id is reported before any fix |
| **T-3 Promote** | fan-out per rule 14 on the release commit (T-1, A0–B1, B2-cpu shards, B2-gpu in parallel); tag only when every shard is green; clean-room-on-tag **inherits** when `tag sha == commit sha`, else **dispatch `clean-room.yml` on the tag immediately**; the run's first step asserts `HEAD == tag` and fails otherwise; record the run id; GitHub release with T-2 receipt attached. Interim until infra#621 (ref input) lands — the nightly clones `main` at 23:00Z and has never tested a tag (0 of 10 runs after v0.66/v0.67) | release `v0.N.0` exists with a green clean-room run id on its sha — **ordering change (2026-09-16, #3366):** T-3 builds the `apr-*` binary assets on the tag **before** T-2 runs, because T-2's installer row consumes them; a T-2 receipt that predates the tag's assets is [U]. Release notes: the installer is the headline after Qwen 3.5 |
| **T-4 Publish** | `publish_strict.sh` from a detached checkout of the tag, `HEAD == tag` asserted, clean tree; order derived from `cargo metadata` at the tag (acyclic, re-checked before start; TIERS is deleted — #3462); one crate per call; stop on first non-zero and report crate + what is already published (never hand-rolled back); retry only 429/5xx ×3 with backoff; never `--allow-dirty`; refuses without the clean-room run id on the tag sha (#3335) and a committed dry-run receipt (`--dry-run --no-verify --locked`, rc=0, tree clean). **Unattended under standing operator authorization** (2026-09-17); receipt records `attended_min` and cascade start/end (the next kaizen number). Post-publish: `cargo install apr-cli --version 0.N.M` on intel + gx10 via a `fleet-hosts` make target, `apr --version` matches; all crates verified on crates.io; run id, receipt and per-crate timestamps attached to the release | all crates at `0.N.M` on crates.io |

Any step RED → SKIPPED, no partial promotion. Scope is assigned to trains after the fact:
0.67 contains whatever merged before the 0.67 cut, by definition.

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
gx10_online, yoga_busy, yoga_online, intel_pressure, verdict}`.
*Done:* over the trailing 10 trains, in every sample with `intel_pressure=true`, median busy/online
≥ 0.8 on both `yoga` and `gx10`; the §7 `pack:` line is never `P0-UNDERUTILIZED` two wakeups in a row.
The structural lever is `workspace-test` leaving its `X64` pin (#3139: 82,085 tests in 795 s on gx10
against 1,000–6,000 s on intel); until it lands, `pack` is bounded by the short jobs.

**P0 · Reap** *(operator 2026-09-12: "automated disk clearing/reaping … managed by forjar and p0 if
it gets blocked")*. `ci-reaper` + `ci-disk-watch` (one script, per-host thresholds citing a
measurement) declared in `machines/{intel,gx10,yoga}/forjar.yaml`, applied, timers verified active;
`/mnt/nvme-raid0 → ~/eph-work/intel-mirror` declared on gx10/yoga so a reimage keeps the layout.
`REAPER_RUN_BUDGET_SEC` and every threshold are declared in the manifest, not script defaults
(2026-09-16: the budget worked via a `${VAR:-420}` default — declare it so it is a fact). The
sweep must never skip permanently: a "skip while any job live" guard on a CI host is
`skip == always` (428 skips, 0 sweeps, 319 GB); per-PR trees are attributable and TTL'd, the
shared registry keeps the live guard; N consecutive skips exits non-zero; one ledger line per run
(`free_gb_before/after, swept, skipped, reason`), resumable partial runs allowed.
Admission at job start: `ci_self_hosted_preflight.sh` prints `disk_free_gb` (measurement first; the
gate threshold waits for 20 records, §3.3). *Done:* zero ENOSPC across 10 trains; a queue blocked on
disk is P0 and stops the current build row.

**P0 · Instrument.** One ledger record per gate job: `sha host job queue_wait_s exec_s
total_s peak_rss_mb free_disk_gb exit`. Add `make build-report`: p50/p95 `total_s` and
`queue_wait_s` per host, 10 slowest test targets, and the §1 PRs-per-train number.
*Done:* ≥ 20 records; `make build-report` runs on a clean checkout.

**P1 · Two lanes.** `ci / gate` (every PR) = compile + **fast set**. `ci / deep` (tags +
nightly) = everything else. Fast set is derived from the ledger, not opinion: rank tests by
failures-caught-per-second, keep the shortest prefix that caught every failure the full
suite caught; record the cut and the escape count. Runner: `cargo nextest` for the fast set
only if measured faster than `cargo test` on the same sha three times; record the delta
either way.
*Done:* one sha runs both lanes; a tag cannot publish with `ci / deep` red; escape count
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

## §6 Triage pass — once per train, no judgement calls

Queues stabilise when closure ≥ arrival; age falls when WIP is capped. That is the whole
mechanism.

- **Milestone = release scope, not a label.** The next-tag milestone holds only §1.5 scope,
  its children, and the gates a cut requires; everything else is retargeted to N+1/N+2
  mechanically at every pass (2026-09-16: ms 0.68 carried 277 open items; it carries 6).
- Every issue opened since the last train: labelled and sized. **`untriaged = 0`** at the end
  of every pass — hard target today, needs no baseline.
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
train:   step reached <T-0..T-4> | skip reason <none|…> | attended min <n|[U]>
build:   row <P0..P3|none> | PR <url|none> | records added <n>
gate:    p95 ci/gate <min|[U]> | max PRs/train <n|[U]> | queue p95 intel <s> yoga <s> gx10 <s>
pack:    intel <busy>/<online> | gx10 <busy>/<online> | yoga <busy>/<online> | intel-pressure <yes|no> | verdict <OK|P0-UNDERUTILIZED>
scope:   §1.5 <ticket> | obligations <passed>/<total> | merged <yes|no> | blocker <none|…> | series <PR1 eta> <PR2 eta> <PR3 eta> vs andon <ts>
queue:   open PRs <n> | WIP cap <13> | group size <n> | ejections since last report <n> | roadmap.yaml diffs in open PRs <n>
pins:    tool-pin-check tree <sha> | matched <n> drifted <n> unchecked <n> | hook <ARMED|NOT-ARMED>
tree:    HEAD <sha> origin/main <sha> behind <n>   # every measurement in this report was taken at this tree
triage:  arrival <n> | closure <n> | open PRs <n> (age p95 <d>) | untriaged <n>
stops:   <none|list>
next:    train eligible at <timestamp>
```

## §8 Stop conditions — stop and report, do not work around

- Two consecutive SKIPPED trains, or ≥ 72 h with no runnable train → andon.
- `ci / deep` red on a tag → no publish.
- Cascade dry-run non-zero → stop before any real publish.
- Measured max PRs/train < 10 `[A]` → stop cutting trains; finish P1/P2 first.
- `yoga` or `gx10` has no runner unit in live `forjar.yaml` and the infra PR is unmerged → stop at P2.
- Fewer than 20 ledger records → stop at P0.
- `yoga` or `gx10` under-utilised (§1) while intel has queue pressure, two wakeups in a row → P0:
  stop the current build row, pack first (arm, route, reap), report the `pack:` line.
- A runner box at or below `REAPER_CRITICAL_GB` free, or any job dead on ENOSPC → P0: reclaim
  over SSH now, then land the forjar change that makes it automatic, before anything else.
- Any T-step job with a single-host `runs-on`, or T-3 without the fan-out ledger → stop; the train
  does not proceed on a serial chain (rule 14).
- Any `ci.yml` job with a single-host `runs-on`, or a pooled box idle under queue pressure → stop (rule 16).
- A published tag is never moved or deleted; a fix after tagging is the next patch version.
- A gate that has never been green on a real target may not block; it runs measurement-only until
  its first-green receipt exists (rule 15).
- Any queue entry ejected twice on the same cause → stop, five-whys, no third re-queue.
- Any measurement whose tree line shows `behind≠0`, or taken outside a per-session worktree →
  discard and re-measure from `origin/main`.
- Any step would need an invented threshold, a hand-squashed batch, a hand-cancelled run with
  started jobs, a label edit to dodge a runner, `--allow-dirty`, an env bypass of a prejob gate, or
  SSH beyond measurement/unclog → stop.

## §9 Budget and the one risk to measure first

K̂ = 14 sessions (10 trains to `0.76` + P0, P1, infra, P2) · K = 18 · andon at 16 sessions
or 2 consecutive skips `[A]`. **Revised 2026-09-15:** +3 for P0·Frag/Runners/Pins, +N for §1.5
Qwen 3.5 (N from the epic's `impl-estimates.jsonl` rows once they carry `unit`; `[U]` until then)
→ K̂ = 17 + N, K = 22 + N. Session slots: 3; ≥ 1 on §1.5 until 0.68 scope is merged.

The cascade is unattended from 0.68.1 under standing authorization; 0.67's attended cascade
measured 08:01→08:43Z for 74 crates (~42 min). The kaizen numbers from 0.69 on are cascade
wall-clock and release-commit→tag wall-clock (rule 14 ratchet), both from the ledger.
