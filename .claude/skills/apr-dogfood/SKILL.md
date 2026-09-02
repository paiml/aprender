---
# EXPLICIT name (#2332). Without this, the skill takes its name from the
# directory, and `dogfood` collides with a personal user-scope skill at
# ~/.claude/skills/dogfood/. On any machine where both exist the USER one wins
# and this file NEVER APPEARS in the session's skill listing — it cannot be
# invoked and nothing warns. Edits look effective and change nothing that runs.
# That is what happened when #2357 hardened Gates 1 and 13.
name: apr-dogfood
allowed-tools: Bash(cargo:*), Bash(apr:*), Bash(pmat:*), Bash(gh:*), Bash(git:*), Bash(find:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(grep:*), Bash(diff:*), Bash(timeout:*), Bash(jq:*), Bash(python3:*), Bash(echo:*), Bash(cat:*), Bash(rm:*), Bash(ssh:*), Read, Glob, Grep, Agent
description: Dogfood the aprender release surface — derive every interface from the built binaries, measure gate coverage against the surface ledger, exercise the covered set, and emit a go/no-go receipt
---

# aprender Dogfood — Coverage-First Pre-Release Protocol

**Version**: 3.0 (supersedes the 19-gate v2.0 skill — Gates 0–18, 924 lines at
`origin/main`; every gate body below is spliced from it **verbatim**)
**Preserves**: every falsifier ID from v2.0 (`FALSIFY-QA-*`, `F-SILENT-*`, `F-META-*`,
`F-COV-*`, `F-CHAOS-*`, `F-DIFF-*`, `F-WORKTREE-HEAD-001`, `F-EXPORT-ROUNDTRIP-001`,
`F-VALIDATE-QUALITY-001`, `F-RUN-EXIT-SANITY-001`, `F-7B-INFERENCE-001`,
`F-APR-INFERENCE-PARITY-001`)
**Defers to** (v3.1, aprender#2640 — this line used to read "**Absorbs**: `dogfood.sh`
release-gate protocol", and "absorbs" is the wrong verb: it invited a second copy
and the runner promptly grew one):
`scripts/dogfood.sh` is the ONE fleet release-gate protocol and
`.claude/skills/dogfood/SKILL.md` is its ONE prose. This skill layers aprender's
surface-coverage ledger ON TOP of it and must not restate a gate the runner owns.
The `invariance.py` transport gate likewise belongs to the runner
(`scripts/invariance.py`), not to this file.

Three documents describe overlapping release work; each owns exactly one scope:

| skill | scope | source of truth for |
|---|---|---|
| `dogfood` | ANY Rust crate in the fleet | the generic pre-release protocol, `scripts/dogfood.sh` |
| `apr-dogfood` (this file) | this repo's shipped surface | gate coverage against the surface ledger |
| `pre-release` | `apr-cli` only | the crates.io publish gates |
**New**: Phase 2 (coverage ledger), Phase 5 (fleet hardware matrix), the mutation registry

**Contracts**:
- `contracts/apr-cli-qa-v1.yaml` — baseline (10 equations, 10 falsification tests)
- `contracts/apr-qa-silent-fallback-v1.yaml` — bad-input injection (5)
- `contracts/apr-qa-metamorphic-v1.yaml` — quant equivalence, multi-arch, roundtrip (5)
- `contracts/apr-qa-coverage-v1.yaml` — category coverage, SATD, complexity (5)
- `contracts/apr-qa-chaos-v1.yaml` — memory, OOM, signals, overwrite (5)
- `contracts/apr-qa-differential-v1.yaml` — ollama parity, tokenizer, concurrency (5)
- `contracts/apr-dogfood-coverage-v1.yaml` — **NEW, Phase 2; must be authored**

**Spec**: `docs/specifications/apr-cli-qa-spec.md`
**Ledgers**: `docs/audits/dogfood-<version>-ledger.md`, `docs/audits/surface_audit.csv`

---

## Why this rewrite exists

The v2.0 skill has 19 gates and they work — where they look. The surface audit
(832 features across 28 binaries) measured where they look:

| | Features | Covered by a gate | Coverage |
|---|---:|---:|---:|
| **Total** | 832 | 142 | **17.1%** |
| `apr` | 369 | 142 | 38.5% |
| **The other 27 binaries** | **463** | **0** | **0.0%** |

And coverage by quality band:

| Band | n | Covered | | |
|---|---:|---:|---|---|
| 1–2 (broken) | 80 | 58 | **72.5%** | where gates look, they find defects |
| 3–4 | 49 | 27 | 55.1% | |
| 5–6 | 673 | 29 | **4.3%** | |
| 7–8 | 8 | 8 | 100% | |
| 9–10 | 20 | 20 | 100% | |

**Read the 5–6 band correctly.** 672 features scored 6, and **644 of those are
uncovered** — scored 6 because no ledger finding exists *and* no gate covers them. That is not "fine" — it is *unlooked-at*. The
1–2 band's 72.5% coverage is the control: gates find defects at a high rate when
pointed at something. 644 unmeasured features is the prediction of where the next
201-finding ledger comes from.

The v2.0 skill cannot see this, because it has no denominator. Every gate answers
"did this check pass?" and none answers "what fraction of the shipped surface did
we check at all?" A gate suite with no denominator reports a clean sweep over
whatever subset it happens to cover — the vacuity failure `dogfood_surfaces.sh`
already guards against per-enumeration, applied one level up.

**So the organizing gate of v3.0 is coverage itself.**

---

## Doctrine (non-negotiable; do not relitigate)

1. **Gates or theater.** A check that cannot fail CI is documentation. Every gate
   in this file carries a named mutation that must turn it RED (§ Mutation
   registry). A gate with no registered mutation is inadmissible.
2. **Never enumerate from a written list.** Binaries from `cargo metadata`,
   commands from `--help` on the **built** binary, routes from the router, tools
   from a live `tools/list`. The command count has been claimed as 36, 77, 103,
   and 111 — every one from a stale hardcoded list, every one failing silently
   because a shrunken universe reports "all passed."
3. **Vacuity-guard every enumeration.** An implausibly small surface FAILS. It
   does not sweep clean.
4. **Exit 0 is not a pass.** Every probe must EXCLUDE an outcome. The 0.63.0 audit
   found tests asserting `is_ok()` on invalid input, which locks the defect in.
5. **Silent-pass is the top severity class regardless of surface.** Fabricating a
   green result, ignoring a flag, exiting 0 on failure. The ledger clusters that
   name it: `lint-passes-on-bad-obs`, `json-flag-emits-text`, `offline-fails-open`,
   `nan-threshold-disarms`.
6. **Read-only.** This skill audits. It never modifies files. Defects become
   GitHub issues and `pmat work add` tickets.
7. **Deterministic receipts.** Sorted enumerations, fixed seeds, fixed prompts, no
   wall-clock assertions, no timestamps in the receipt body. The same tree yields
   a byte-identical receipt. Verify with `--twice`.
8. **Stop the line.** Any red gate stops the release. Fix the root cause via
   five-whys to the owning module. Never bypass, never `--skip`.

---

## Context

- Workspace version: !`cargo metadata --no-deps --format-version 1 | jq -r '.packages[]|select(.name=="apr-cli")|.version'`
- HEAD: !`git rev-parse --short HEAD`
- Worktree clean: !`git status --porcelain | wc -l` modified paths
- apr pinned to HEAD: !`. scripts/apr_bin.sh >/dev/null 2>&1 && "$APR" --version || echo "NOT built from HEAD — every verdict below would describe a binary you are not running"`
- Models available: !`find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" -o -name "*.safetensors" \) -type f 2>/dev/null | wc -l`
- Surface ledger: !`test -f docs/audits/surface_audit.csv && wc -l < docs/audits/surface_audit.csv || echo "ABSENT — Phase 2 will FAIL"`
- Clusters: !`test -f docs/audits/surface_audit.csv && python3 -c "import csv,collections;r=list(csv.DictReader(open('docs/audits/surface_audit.csv')));c=collections.Counter(x['cluster_label'] for x in r);g=collections.Counter(x['cluster_label'] for x in r if x['in_dogfood_skill'].strip().lower()=='yes');print(f'{sum(1 for k in c if g[k])}/{len(c)} clusters gated, {sum(g.values())}/{len(r)} features gated')" || echo "ABSENT"`

## Arguments

$ARGUMENTS

| Flag | Effect |
|---|---|
| *(none)* | Tiers 0–2. The daily loop. |
| `--release` | All tiers, all phases. Required before any publish. |
| `--fleet` | Adds Phase 5 (four-host matrix). Implied by `--release`. |
| `--twice` | Runs the whole sweep twice and diffs receipts. Determinism check. |
| `--binary <name>` | Restrict to one binary. Coverage denominator narrows with it. |
| `<model path>` | Use that model. Otherwise auto-discover from `~/models`. |

---

# Phase 0 — Identity guards

Everything downstream is void if the thing under test is not the thing you think
it is. These run first, serially, and a failure here **aborts** rather than
degrading to a warning.

## G0.1 — Binary provenance (`FALSIFY-QA-005`)

```bash
cargo install --path crates/apr-cli --force 2>&1 | tail -5
# Resolve through the guard, NEVER a bare `apr`. Sourcing exports $APR and
# hard-fails when it was not built from HEAD.
. scripts/apr_bin.sh || exit 1
APR_BIN_STRICT=1 "$APR" --version
git rev-parse --short HEAD
```

PASS iff the version string contains the HEAD hash.

**This gate exists because of ledger #2384**, a P0 **closed COMPLETED 2026-08-11**
(fixed by #2424; the ledger row still shows no `Fixed by`, so ledger and GitHub
disagree — treat the gate as a regression ratchet, not an open-defect probe): the MCP
server and `apr code` both execute a bare `apr` resolved from `PATH`. Every tool
ran a *different* 0.60.0 binary while `apr.version` reported 0.63.0. Never
resolve a bare `apr` anywhere in this protocol.

## G0.2 — Crate identity from `cargo metadata`, never from `sed`

Resolve name and version via `cargo metadata --no-deps`, filtered by
`manifest_path`. Do not grep `Cargo.toml`.

Measured 2026-08-20 on `pforge-cli`: `sed` yielded an empty version for a crate
using `version.workspace = true`, and an empty version made *version-unpublished*
PASS, because the crates.io lookup for `""` finds nothing. A false pass in the
gate whose entire job is to stop an immutable double-publish.

Never proceed with an empty version. Abort with the candidate member list.

## G0.3 — Worktree HEAD sanity (`F-WORKTREE-HEAD-001`)

Contract: `contracts/apr-version-traceability-v1.yaml` § FALSIFY-VERSION-004

Body carried **verbatim** from v2.0 Gate 13 (hardened by #2357 — do not re-derive).


Contract: `contracts/apr-version-traceability-v1.yaml` § FALSIFY-VERSION-004

Catches [#1862](https://github.com/paiml/aprender/issues/1862) — `apr --version`
reporting a stale commit hash in git worktrees because `build.rs` watches a
hardcoded `../../.git/HEAD` path that doesn't exist in a worktree layout.

```bash
# After cargo install, the SHA MUST match git rev-parse --short HEAD.
# Run this from inside the source checkout (or worktree) you just built from.
. scripts/apr_bin.sh || exit 1
APR_SHA=$("$APR" --version 2>&1 | grep -oE '\([a-f0-9]{7,}\)' | tr -d '()')
HEAD_SHA=$(git rev-parse --short HEAD)
if [ -n "$APR_SHA" ] && [ "$APR_SHA" = "$HEAD_SHA" ]; then
  echo "G13 PASS: $APR SHA ($APR_SHA) matches HEAD"
elif [ -z "$APR_SHA" ]; then
  # NOT a SKIP. A binary with no embedded SHA cannot be shown to be the code
  # under test, and "cannot prove" must never read as "fine". This branch used
  # to SKIP for "likely crates.io install" — and the binary that wins PATH
  # resolution on this machine is `~/.local/bin/apr`, which reports exactly
  # `apr 0.60.0 (v0.60.0+no-git)`. So the one stale artifact most likely to be
  # executed was also the one that made the protocol's ONLY freshness gate
  # excuse itself. If you are deliberately dogfooding a published crates.io
  # build, say so explicitly with DOGFOOD_ALLOW_UNPINNED=1.
  if [ "${DOGFOOD_ALLOW_UNPINNED:-0}" = "1" ]; then
    echo "G13 SKIP (explicitly allowed): $APR has no embedded SHA"
  else
    echo "G13 FAIL: $APR has NO embedded SHA — cannot prove it is HEAD. Set DOGFOOD_ALLOW_UNPINNED=1 only for a deliberate crates.io dogfood."
  fi
else
  echo "G13 FAIL: $APR SHA=$APR_SHA but HEAD=$HEAD_SHA (#1862)"
fi
```

Build.rs static check (no install required):
```bash
# build.rs MUST use git rev-parse --git-dir / --git-common-dir for worktree-safe
# rerun-if-changed triggers — not a hardcoded ../../.git/HEAD path.
if grep -qE 'rev-parse.*--git-(dir|common-dir)' crates/apr-cli/build.rs \
   && ! grep -qE '\.\./\.\./\.git/HEAD' crates/apr-cli/build.rs; then
  echo "G13 PASS (static): build.rs uses worktree-safe git resolution"
else
  echo "G13 FAIL (static): build.rs still uses hardcoded .git/HEAD path"
fi
```

PASS if both checks pass (or SHA check SKIPs cleanly on crates.io builds).

## G0.4 — Feature-scope resolution

Determine the feature set from `[features]` only. Two traps, both real:

- `cli = { e2e = "…" }` under `[package.metadata.transports]` is **not** a cargo
  feature. Matching it makes every gate run `--features cli` against a crate that
  has no such feature.
- `--all-features` is **not** a safe fallback. `pmat` declares `broken-tests` (49
  quarantined sites) and `red-phase-tests` ("expected failures … NEVER add to any
  feature bundle"). Enabling them measures the quarantine, yielding a permanent
  RED that says nothing about release readiness — and a gate that is always red is
  one everybody learns to walk past.

Fallback is every declared feature **minus** known-broken quarantines, and the
exclusion is **reported**. A silent exclusion is how a gate quietly stops covering
what it claims to.

## G0.5 — Duplicate-binary divergence check *(new)*

Two `apr` bin targets exist and are the same program **by hand-duplication, not by
sharing**: `src/bin/apr.rs` delegates to `apr_cli::cli_main()`, while
`crates/apr-cli/src/main.rs` re-implements the same prologue (SIGPIPE reset,
aarch64 FPCR.FZ16 clear, `NO_COLOR`, the `--version --json` intercept). **They have
already diverged** — the `dhat-heap` `#[global_allocator]` exists only in the
latter.

Assert both entry points produce byte-identical output for `--version`,
`--version --json`, and `--help`. Any divergence is RED.

---

# Phase 1 — Surface derivation

Delegate to `scripts/dogfood_surfaces.sh`. Do not reimplement it here; do not
substitute a list.

Enumerate, sorted `LC_ALL=C`:

| Kind | Source of truth |
|---|---|
| Binaries | `cargo metadata` — every `[[bin]]` and every `src/bin/*.rs` |
| CLI commands | `--help` on the **built** binary, recursively to leaf subcommands |
| Flag-selected code paths | `value_parser` domains and dispatch arms (see below) |
| HTTP routes | the mounted router |
| MCP tools | a live `tools/list` |

**Grepping the source is not an acceptable substitute.** A regex over clap
`Subcommand` enums reports 0 subcommands for `simular`, a clap-derive CLI. The
binary is the only thing that knows what the binary accepts.

## G1.1 — Vacuity guard

Every enumeration yielding implausibly few items **FAILS the run**. It does not
report a clean sweep over an empty set. Record the threshold per surface and the
observed count.

## G1.2 — Feature-vs-flag classification

A **feature** is a distinct user-visible capability with its own code path and its
own success/failure semantics. A flag that selects a materially different code
path is a feature (`--backend cuda`, `--task pretrain`, streaming vs not). A flag
that only formats or labels output is not (`--json`, `--verbose`, `-o`).

## G1.3 — `value_parser` domain consistency *(new)*

For every flag whose value selects a code path, assert a `value_parser` pins the
domain at **every** declaration site.

Live defect this gate exists to catch: `apr serve run --backend` is declared with
**no** `value_parser` — at HEAD, `crates/apr-cli/src/serve_commands.rs:73` is exactly
`#[arg(long, value_name = "BACKEND")]`, zero `BACKEND_VALUES` — while `apr chat`
(`extended_commands.rs:85`, arg line `:84`) and `apr run` (`commands_enum.rs:154`) both
pin `BACKEND_VALUES` (the constant itself is `commands_enum.rs:10`). The value is consumed (`serve/types.rs:57`). That constant's own
doc-comment says it exists to prevent "a run whose throughput number was taken
through a backend the caller did not ask for" — reproduced on the serve path. The
doc also claims it covers "apr run / apr chat"; there are **three** declaration
sites, not two.

Also RED: `apr bench` has no `--backend` at all, despite being the throughput
command.

## G1.4 — Help-vs-dispatch completeness *(new)*

For every command with a `--task`-style selector, assert the set of documented
values equals the set of reachable dispatch arms.

Live defect: `apr eval --task` dispatches 10 named arms plus a default perplexity
path (`dispatch_analysis.rs:1389-1425`), but `--help` documents exactly two
(`extended_commands.rs:138`). Nine paths — `humaneval`, `mbpp`, `code`,
`contamination`, `compare`, `verify`, `correlation`, `human`, `plan` — are
reachable and undocumented. The line range was `1383-1419` when it was
measured at `4bbfeb07f` and the arms have moved twice since; the count read
"Eight" against a list of nine, and both were re-derived from the arms
themselves rather than carried forward (PERF-046).

Undocumented-but-reachable is RED. Documented-but-unreachable is RED. This is the
phantom-subcommand protocol (v2.0 P9) generalized to flag domains.

---

# Phase 2 — Coverage ledger ⭐ THE GATE THAT ORGANIZES THE REST

Phase 1 produced the **denominator**. This phase computes the numerator and gates
on the ratio.

## G2.1 — Ledger freshness

`docs/audits/surface_audit.csv` must exist and must have been regenerated against
the current HEAD. A ledger older than the commit under test **FAILS**.

This is the staleness arm. Without it the coverage number measures *declared*
state rather than *resolved* state — the exact enforcement-theater shape this
whole protocol exists to prevent.

Schema (10 columns, RFC 4180):

```
binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,cluster_id,cluster_label,evidence_path,confidence
```

Every row requires `evidence_path` **and** a non-empty `cluster_label`. A row with
no evidence path is an unevidenced claim; a row with no cluster label sits outside
every per-cluster floor. Both fail schema validation.

**There is ONE ledger.** The clustered file *replaced* the 8-column one; it did not
land beside it as `surface_audit_clustered.csv`. Two ledgers over one surface is
the drift hazard this repo keeps re-finding — the second copy goes stale in
silence and every consumer then has to be told which is authoritative. The
clustered file was a superset in shape (same 830 rows, same order, two extra
columns), so replacing cost nothing and keeping both would have bought a
permanent divergence. Three cells *did* disagree — `apr run --backend
{cpu,cuda,wgpu}` cited `commands_enum.rs:110` in the clustered snapshot and `:154`
in the landed ledger; `:154` is the `backend:` arg and `:110` a chat-template arg,
so the landed value won. That disagreement, found on the day the two files
existed side by side, is the argument.

## G2.2 — Denominator reconciliation

The Phase 1 runtime enumeration and the CSV row set must agree.

- A feature in the binary but **not** in the CSV → the ledger is stale. RED.
- A feature in the CSV but **not** in the binary → the surface shrank without the
  ledger noticing. RED.

Neither may be silently reconciled. This is what makes the CSV a *ledger* rather
than a *list* — the list is regenerated, the ledger is checked against it.

## G2.3 — Coverage floors

Compute `in_dogfood_skill == yes` over the reconciled denominator, overall and per
binary and per quality band.

Baselines below are **measured**, not chosen — computed from
`docs/audits/surface_audit.csv` (830 rows, 28 binaries) on 2026-08-22 at
`4bbfeb07f`. They are the committed starting point of the ratchet.

| Floor | Measured baseline | Threshold | Verdict rule |
|---|---:|---:|---|
| Overall coverage | **142/830 = 17.1%** (142/832 today) | `>= 142` covered rows | **may never decrease** |
| `apr` coverage | 142/367 = 38.7% (142/369 today) | `>= 142` | may never decrease |
| Per-binary coverage | **27 of 28 binaries at 0%** | covered may never fall | ratchet only — superseded at `--release` by the per-cluster arm below |
| **Per-cluster coverage** | **9 of 14 clusters at 0 earned gates**, under 142 of 830 features gated (142 of 832 today) | ≥ 1 **earned** gate per `cluster_label` | **RED at `--release`**; ratchet always |
| **Cluster membership** | 0 declared reassignments | a feature may not change `cluster_label` undeclared | RED — a silent move retires an obligation |
| Quality ≤ 4, uncovered | **44** | `0` | RED — a known-broken feature with no gate |
| `verified_hardware` UNKNOWN | **427** | `<= 427` | may never increase |
| `confidence == low` and uncovered | **204** | `<= 204` | may never increase |

Band baselines, same measurement: 1–2 → 58/80 (72.5%); 3–4 → 27/49 (55.1%);
5–6 → 29/673 (4.3%); 7–8 → 8/8; 9–10 → 20/20.

**Set the initial thresholds from measurement, not from a round number.** Commit
today's values as the baseline and ratchet. A threshold chosen before measurement
either never fires or fires constantly.

The ratchet is the point: coverage is allowed to be 17.1%. It is not allowed to
become 17.0%.

## G2.5 — The per-cluster floor ⭐ THE THIRD FLOOR

Three floors now, not two: **overall**, **per-binary**, **per-cluster**.

### Why a binary is the wrong unit

`aprender-orchestrate` ships 184 features that are three unrelated subsystems —
95 Banco HTTP routes, a 56-feature agent stack, 17 Pacha secrets commands. A
per-binary floor of "≥ 1 gate" lets **one gate on Pacha make all 184 look
touched**. The cluster is the unit whose members share a module, a dispatch path
and a failure mode, which is the property that makes a gate on one member
evidence about the rest.

### What is enforced

```
per-cluster ratchet   (always)   no cluster_label's gate count may fall
                                 the zero-gate cluster count may not rise
                                 a cluster_label on the comparand may not vanish
per-cluster release arm          every cluster_label carries >= 1 gate   # RED at --release
T2 reporting pairing  (always)   cluster coverage is reported WITH the feature %,
                                 never one alone — ENFORCED, not documented
```

Every floor is derived from `git show origin/main:docs/audits/surface_audit.csv`,
the same comparand the other floors use. **No cluster count is a literal in any
gate file.** While `main` still carries the 8-column ledger the ratchet prints a
`SCHEMA UPGRADE` banner instead of passing silently; that branch is self-closing
and a half-migrated comparand is a hard failure, not an upgrade.

### The four uses of the clustering

1. **Stratified sampling frame.** 830 gates were never the target. Features in a
   cluster share a failure mode, so a gate on one member is evidence about the
   cluster. The goal is *n* gates per cluster allocated by expected defect yield,
   with *n* scaling **sub-linearly** in cluster size. Nine clusters at zero is
   nine clusters with **no evidence at all** — that is the gap, not the 690
   uncovered rows.
2. **Sibling sweep.** A defect in cluster X makes X's remaining members a
   mandatory sweep list in the same ticket. See Phase 3.
3. **Harness amortization.** 95 Banco routes is ONE HTTP harness, not 95 gates.
   76 `pv` subcommands is ONE contract-CLI harness. Cluster size estimates
   gates-per-unit-effort and should drive ticket **order**:
   `http-orchestrate-banco` and `contracts-pv` are the two cheapest large wins in
   the repo and both sit at zero.
4. **Per-cluster floor replacing per-binary** as the release arm — the reason
   above.

### The three traps

| | Trap | Enforcement |
|---|---|---|
| **T1** | `cluster_id` is a k-means label and **permutes on re-run**. An id in a contract, gate or waiver silently re-points at a different cluster next time the surface moves — the stale-hardcoded-list class in new clothes. | `cluster_label` is the durable key and is **human-owned** after first assignment, never regenerated. `scripts/check_no_cluster_id_keys.sh` is an **allowlist**: a standalone `cluster_id` token is a key unless the line is backticked, follows a comment marker, declares the schema, or carries `cluster-id-guard allow (<reason>)`. Its first version listed four banned syntaxes and passed twelve ordinary ones. Case table run in CI beside the scan. |
| **T1b** | Cluster **membership** was unratcheted, so the release arm was satisfiable by relabelling: move an already-gated feature into a zero-gate cluster and the zero stops existing with no gate written. | A move must be declared in `docs/audits/cluster_reassignments.yaml` (from, to, reason), and the zero-gate set counts **earned** gates only — a gate that walked in from another cluster is evidence about where it came from. Declaring buys legibility; only writing a gate buys evidence. |
| **T2** | **Cluster coverage ≠ feature coverage.** One gate in a 95-member cluster is 1%, not "covered". Reporting the proxy alone builds the vacuity failure one level up: a clean sweep over a proxy, looking *stricter* than what it replaced. **This is the most important trap.** | `enforce_pairing()` in `scripts/lib/dogfood_coverage_gate.py` reads the report back before printing it and fails the gate if any line states a cluster fraction without a feature fraction beside it. It also fails on an empty report. |
| **T3** | Clustering is a **prior, never evidence.** It says where to look; it cannot assert a feature works. | Severity comes from the 0.63.0 ledger. `quality_1_10` is **never** derived from cluster membership. |

`scripts/dogfood_cluster.py` produces the **k-selection evidence only** — the
inertia/silhouette sweep and `docs/audits/surface_audit_elbow.png`. It does not
write `cluster_id`/`cluster_label` and re-running it will not regenerate them.
That follows from T1: the labels are human-owned, so the columns are deliberately
not reproducible from the script.

## G2.4 — The 44

44 features carry quality ≤ 4 (a live ledger defect) **and** have no gate. All 44
are in `apr`. Every one is a defect that shipped and would ship again unnoticed.

Enumerate them in the receipt by name. At `--release`, this list must be empty or
every entry must carry an open issue number and an explicit written waiver.

---

# Phase 3 — Tiered gate execution

Tier 0 and 1 are the daily loop. Tier 2 adds cost. Tier 3 is release-only.

## G3.0 — The sibling sweep rule (process, not a script)

**When a defect lands in cluster X, X's remaining uncovered members become a
MANDATORY sweep list in the same ticket.**

The prior is measured, not assumed: the 0.63.0 ledger collapsed **201 findings
into 37 root causes — ~5.4 findings per cause**. Defects arrive in sibling
groups. Clustering supplies that prior *mechanically and in advance* instead of
retrospectively, after someone notices the fourth instance.

The rule, exactly:

1. A defect is found in a feature whose `cluster_label` is X.
2. The ticket enumerates **every uncovered member of X**, by name, from
   `docs/audits/surface_audit.csv`.
3. Each swept sibling ends in one of three states, recorded in the ticket:
   **also broken** (its own finding), **checked and sound** (evidence cited), or
   **not reachable** (why, and what would make it reachable).
4. A sweep that finds nothing is still a result. A sweep that was never
   enumerated is an open obligation, and the ticket does not close.

This is read-only, like the rest of the skill: the sweep produces findings and
tickets, never fixes.

**Do not skip the sweep because the cluster is large.** A 95-member cluster is
exactly where the prior pays: one HTTP harness covers the sweep, which is
point 3 of the four uses above.

## Tier 0 — Cheap, always (parallel)

| ID | Gate | Falsifier |
|---|---|---|
| T0.1 | Build & install | `FALSIFY-QA-005` |
| T0.2 | `fmt` / `clippy` / `deny` | — |
| T0.3 | Contract validation (`pv`) | Gate 4 v2.0 |
| T0.4 | Code quality — SATD, complexity | Gate 5 v2.0 |
| T0.5 | Open-issue sweep | Gate 7 v2.0 |
| T0.6 | Changelog mentions the version | — |
| T0.7 | `git` worktree clean | — |

Bodies carried **verbatim** from v2.0 Gates 4–7.

### Gate 4: Contract Validation

```bash
# Verify contract is valid
python3 -c "import yaml; yaml.safe_load(open('contracts/apr-cli-qa-v1.yaml')); print('VALID')"

# Run integration tests that enforce the contract
cargo test -p apr-cli --test cli_commands 2>&1 | tail -3
cargo test -p aprender-core --test readme_contract 2>&1 | tail -3
cargo test -p aprender-core --test monorepo_invariants 2>&1 | tail -3
```

PASS if all 3 test suites pass. FAIL if any test fails.

### Gate 5: Code Quality

```bash
cargo test -p apr-cli --lib 2>&1 | grep "test result:"
cargo clippy -p apr-cli --lib -- -D warnings 2>&1 | grep "^error:" | wc -l
```

PASS if 0 test failures and 0 clippy errors. WARN if clippy warnings.

### Gate 6: Coverage Check

```bash
cargo llvm-cov -p aprender-core --lib --no-report 2>&1 | tail -3
cargo llvm-cov report 2>&1 | tail -1
```

PASS if coverage >= 95%.

### Gate 7: Open Issues

```bash
gh issue list --repo paiml/aprender --state open --limit 20
```

Always PASS (informational).

## Tier 1 — The protocol grid

The gate bodies below are carried **verbatim** from v2.0 Gate 2 (full command
grid) and Gate 3 (P1–P12). The table indexes them; P13 and P14 are new and
still to be authored.

| | Protocol | Falsifier |
|---|---|---|
| P1 | Silent-flag | `FALSIFY-QA-007` |
| P2 | Exit-code contradiction | `FALSIFY-QA-006` |
| P3 | Flag-echo | — |
| P4 | Cross-subcommand consistency | `FALSIFY-QA-010` |
| P5 | Cache integrity | — |
| P6 | GPU/CPU parity | — |
| P7 | NaN/Inf sentinel | `FALSIFY-QA-004` |
| P8 | Version sanity | `FALSIFY-QA-005` |
| P9 | Phantom subcommand | `FALSIFY-QA-008` |
| P10 | JSON schema stability | `FALSIFY-QA-003` |
| P11 | Default-defamation | — |
| P12 | Hardware cascade | — |
| **P13** | **`value_parser` domain** *(new, G1.3)* | **to author** |
| **P14** | **Help-vs-dispatch completeness** *(new, G1.4)* | **to author** |

**P6 carries the CF-4 constraint.** The canonical failure (#1864) is a CPU-vs-GPU
parity gate that sampled only decode step 0 and stayed green while GPU output
collapsed to gibberish across later steps. **Any gate touching an autoregressive,
temporal, cached, streaming, or multi-turn path must validate over a horizon, never
a single sample.** Minimum 64 cached positions for attention parity. A single-step
probe on a compounding system is inadmissible.
### Gate 2: Full Command Grid (FALSIFY-QA-001, FALSIFY-QA-009)

Auto-discover models:
```bash
find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" -o -name "*.safetensors" \) -type f 2>/dev/null
```

Pick one per format. For EACH format, exercise ALL command categories:

### 2a. Inspection (11 commands)
```bash
. scripts/apr_bin.sh || exit 1
for cmd in inspect debug validate lint tensors trace diff hex tree flow explain; do
  echo -n "$cmd: " && timeout 30 "$APR" $cmd $MODEL 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2b. QA & Evaluation (8 commands)
```bash
. scripts/apr_bin.sh || exit 1
for cmd in check qa qualify bench eval canary compare-hf parity; do
  echo -n "$cmd: " && timeout 60 "$APR" $cmd $MODEL 2>&1 | head -1 && echo "OK" || echo "SKIP/FAIL"
done
```

### 2c. Transform (9 commands)
```bash
. scripts/apr_bin.sh || exit 1
for cmd in convert export import quantize merge prune compile encrypt decrypt; do
  echo -n "$cmd: " && timeout 30 "$APR" $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2d. Inference (4 commands) — timeout 60
```bash
. scripts/apr_bin.sh || exit 1
"$APR" run $MODEL "What is 2+2?" --max-tokens 16 2>&1 | tail -3
"$APR" serve plan $MODEL 2>&1 | head -5
```

### 2e. Registry (4 commands)
```bash
. scripts/apr_bin.sh || exit 1
"$APR" list 2>&1 | head -5
"$APR" list --json 2>&1 | jq length
"$APR" gpu 2>&1 | head -5
```

### 2f. Training & Data (6 commands)
```bash
. scripts/apr_bin.sh || exit 1
for cmd in finetune distill train tokenize tune data; do
  echo -n "$cmd: " && timeout 10 "$APR" $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2g. UI & Pipeline (7 commands)
```bash
. scripts/apr_bin.sh || exit 1
for cmd in tui monitor runs experiment pipeline diagnose showcase; do
  echo -n "$cmd: " && "$APR" $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2h. Remaining (8 commands)
```bash
. scripts/apr_bin.sh || exit 1
for cmd in rosetta publish oracle probar ptx ptx-map code cbtop; do
  echo -n "$cmd: " && "$APR" $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

SKIP (not FAIL) if no models found. FAIL if any command panics or crashes.

### Gate 3: Protocol Checks (12 protocols from apr-cookbook)

### P1. Silent-Flag Protocol (FALSIFY-QA-007)
```bash
. scripts/apr_bin.sh || exit 1
diff <("$APR" inspect $M 2>&1) <("$APR" inspect --json $M 2>&1)
diff <("$APR" inspect $M 2>&1) <("$APR" inspect --vocab $M 2>&1)
diff <("$APR" list 2>&1) <("$APR" list --json 2>&1)
```
FAIL if any flag produces identical output (no-op flag).

### P2. Exit-Code Contradiction (FALSIFY-QA-006)
```bash
for cmd in "apr lint $M" "apr validate /nonexistent" "apr rm nonexistent-id"; do
  out=$(eval "$cmd" 2>&1); ec=$?
  echo "$out" | grep -qiE 'error|fail' && [ "$ec" -eq 0 ] && echo "P1 EXIT-CODE LIE: $cmd"
done
```

### P3. Flag-Echo Protocol
```bash
. scripts/apr_bin.sh || exit 1
out=$("$APR" run $M "test" --max-tokens 8 --temperature 0.5 2>&1)
# Verify temperature is actually 0.5, not default
```

### P4. Cross-Subcommand Consistency (FALSIFY-QA-010)
```bash
. scripts/apr_bin.sh || exit 1
F_INSPECT=$("$APR" inspect --json $M 2>/dev/null | jq -r '.architecture // empty')
F_CHECK=$("$APR" check $M 2>&1 | grep -i arch | head -1)
echo "inspect=$F_INSPECT check=$F_CHECK"
```

### P5. Cache Integrity
```bash
. scripts/apr_bin.sh || exit 1
BEFORE=$("$APR" list 2>&1 | wc -l)
# pull, list, rm cycle should be consistent
```

### P6. GPU/CPU Parity (if GPU available)
```bash
. scripts/apr_bin.sh || exit 1
"$APR" gpu 2>&1 | head -3
# If GPU present: compare apr run --device cpu vs --device gpu
```

### P7. NaN/Inf Sentinel (FALSIFY-QA-004)
```bash
for cmd in "apr run $M 'test' --max-tokens 8" "apr bench $M --iterations 1"; do
  timeout 30 eval "$cmd" 2>&1 | grep -qE '\bNaN\b|\bInf\b' && echo "P0 NaN: $cmd"
done
```

### P8. Version Sanity (FALSIFY-QA-005)
```bash
. scripts/apr_bin.sh || exit 1
"$APR" --version | grep -qE '\(unknown\)|0000000' && echo "P3 VERSION SENTINEL"
```

### P9. Phantom Subcommand (FALSIFY-QA-008)
```bash
. scripts/apr_bin.sh || exit 1
"$APR" --help | awk '/^  [a-z]/{print $1}' | while read cmd; do
  "$APR" "$cmd" --help 2>&1 | grep -qi "not.*implemented" && echo "PHANTOM: $cmd"
done
```

### P10. JSON Schema Stability (FALSIFY-QA-003)
```bash
for cmd in "apr inspect --json $M" "apr list --json" "apr gpu --json"; do
  eval "$cmd" 2>&1 | jq . > /dev/null 2>&1 || echo "P2 INVALID JSON: $cmd"
done
```

### P11. Default-Defamation Protocol
```bash
. scripts/apr_bin.sh || exit 1
"$APR" eval $M 2>&1 | grep -qi 'garbage\|broken\|corrupt' && echo "P3 DEFAMATION"
```

### P12. Hardware Cascade Protocol
```bash
. scripts/apr_bin.sh || exit 1
# If GPU fails, does CPU fallback work?
"$APR" gpu 2>&1 | head -3
```



## Tier 2 — Injection and metamorphic

| ID | Gate | Falsifiers |
|---|---|---|
| T2.1 | Silent-fallback injection | `F-SILENT-001..005` |
| T2.2 | Metamorphic — quant equivalence, multi-arch, roundtrip | `F-META-001..005` |
| T2.3 | Coverage completeness | `F-COV-001..005` |
| T2.4 | Chaos — memory, OOM, signals, overwrite | `F-CHAOS-001..005` |
| T2.5 | Differential vs ollama — tokenizer, concurrency | `F-DIFF-001..005` |
| T2.6 | APR → GGUF export round-trip | `F-EXPORT-ROUNDTRIP-001` |
| T2.7 | `validate --quality` sanity | `F-VALIDATE-QUALITY-001` |
| T2.8 | `apr run` exit reflects output validity | `F-RUN-EXIT-SANITY-001` |
| T2.9 | Fresh-convert `.apr` inference parity, CPU+GPU | `F-APR-INFERENCE-PARITY-001` |
Bodies carried **verbatim** from v2.0 Gates 8–12, 14–16 and 18.

### Gate 8: Silent-Fallback Injection (F-SILENT-001 through F-SILENT-005)

Contract: `contracts/apr-qa-silent-fallback-v1.yaml`

Bad inputs MUST fail LOUD (non-zero exit + stderr message), never silently degrade.

### S1. Truncated file detection (GH-707)
```bash
. scripts/apr_bin.sh || exit 1
M_GGUF=$(find ~/models -maxdepth 2 -name "*.gguf" -type f | head -1)
if [ -n "$M_GGUF" ]; then
  SIZE=$(stat -c%s "$M_GGUF")
  head -c $((SIZE / 2)) "$M_GGUF" > /tmp/apr-qa-truncated.gguf
  # IMPORTANT: capture exit code without piping (pipe loses $?)
  OUTPUT=$("$APR" validate /tmp/apr-qa-truncated.gguf 2>&1); EC=$?
  echo "$OUTPUT" | tail -3
  [ "$EC" -ne 0 ] && echo "S1 PASS: truncated file rejected (exit $EC)" || echo "S1 FAIL: truncated file accepted (GH-707)"
fi
```

### S2. Bad file rejection
```bash
. scripts/apr_bin.sh || exit 1
OUTPUT=$("$APR" bench /dev/null --iterations 1 2>&1); EC=$?
echo "$OUTPUT" | tail -1
[ "$EC" -ne 0 ] && echo "S2 PASS: /dev/null rejected (exit $EC)" || echo "S2 FAIL: /dev/null accepted"
```

### S3. Unknown architecture handling (GH-704 pattern)
```bash
. scripts/apr_bin.sh || exit 1
# Check that Qwen3.5 SSM model gets a clear error, not silent llama fallback
M_SSM=$(find ~/models -maxdepth 2 -name "*Qwen3.5*" -o -name "*qwen35*" 2>/dev/null | head -1)
if [ -n "$M_SSM" ]; then
  OUTPUT=$("$APR" run "$M_SSM" "test" --max-tokens 1 2>&1); EC=$?
  echo "$OUTPUT" | grep -qi "not.*supported\|unsupported\|SSM" && \
    echo "S3 PASS: unsupported arch gives clear error" || echo "S3 FAIL: no clear error for unsupported arch"
else
  echo "S3 SKIP: no SSM model available"
fi
```

### S4. Corrupted metadata detection
```bash
. scripts/apr_bin.sh || exit 1
if [ -n "$M_GGUF" ]; then
  cp "$M_GGUF" /tmp/apr-qa-corrupt.gguf
  dd if=/dev/zero of=/tmp/apr-qa-corrupt.gguf bs=1 count=64 seek=8 conv=notrunc 2>/dev/null
  OUTPUT=$("$APR" validate /tmp/apr-qa-corrupt.gguf 2>&1); EC=$?
  echo "$OUTPUT" | tail -1
  [ "$EC" -ne 0 ] && echo "S4 PASS: corrupt metadata rejected (exit $EC)" || echo "S4 FAIL: corrupt metadata accepted"
fi
```

### S5. Missing model graceful (FALSIFY-QA-002)
```bash
. scripts/apr_bin.sh || exit 1
OUTPUT=$("$APR" inspect /nonexistent/model.gguf 2>&1); EC=$?
echo "$OUTPUT" | tail -1
[ "$EC" -ne 0 ] && echo "S5 PASS: missing model exits non-zero (exit $EC)" || echo "S5 FAIL: exit 0 for missing model"
```

PASS if all 5 checks reject bad input. FAIL if any bad input is silently accepted.

### Gate 9: Metamorphic Testing (F-META-001 through F-META-005)

Contract: `contracts/apr-qa-metamorphic-v1.yaml`

### M1. Format roundtrip (GGUF→APR→GGUF tensor fidelity)
```bash
. scripts/apr_bin.sh || exit 1
if [ -n "$M_GGUF" ]; then
  "$APR" convert "$M_GGUF" --quantize q4k -o /tmp/apr-qa-rt.apr 2>&1 | tail -1  # --quantize is REQUIRED
  # If convert succeeds, check tensor count matches
  if [ -f /tmp/apr-qa-rt.apr ]; then
    ORIG_TENSORS=$("$APR" tensors "$M_GGUF" --json 2>/dev/null | jq length 2>/dev/null || echo 0)
    RT_TENSORS=$("$APR" tensors /tmp/apr-qa-rt.apr --json 2>/dev/null | jq length 2>/dev/null || echo 0)
    echo "M1 orig=$ORIG_TENSORS rt=$RT_TENSORS"
    [ "$ORIG_TENSORS" -gt 0 ] && [ "$RT_TENSORS" -gt 0 ] && echo "M1 PASS" || echo "M1 WARN: tensor count mismatch"
  else
    echo "M1 SKIP: convert not available for this model"
  fi
else
  echo "M1 SKIP: no GGUF model"
fi
```

### M2. Multi-architecture smoke
```bash
. scripts/apr_bin.sh || exit 1
# Check that inspect works across all available model architectures
ARCH_COUNT=0
for m in $(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" -o -name "*.safetensors" \) -type f 2>/dev/null); do
  ARCH=$(timeout 10 "$APR" inspect --json "$m" 2>/dev/null | jq -r '.architecture // empty' 2>/dev/null)
  [ -n "$ARCH" ] && ARCH_COUNT=$((ARCH_COUNT + 1)) && echo "  M2 arch=$ARCH ($m)"
done
[ "$ARCH_COUNT" -ge 2 ] && echo "M2 PASS: $ARCH_COUNT architectures inspected" || echo "M2 WARN: only $ARCH_COUNT architectures available"
```

### M3. Temperature determinism (temp=0 → identical output across 3 runs)
```bash
. scripts/apr_bin.sh || exit 1
M_APR=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -n "$M_APR" ]; then
  OUT1=$(timeout 60 "$APR" run "$M_APR" "Say hello" --max-tokens 4 --temperature 0.0 2>&1 | grep "^Output:" | head -1)
  OUT2=$(timeout 60 "$APR" run "$M_APR" "Say hello" --max-tokens 4 --temperature 0.0 2>&1 | grep "^Output:" | head -1)
  if [ "$OUT1" = "$OUT2" ] && [ -n "$OUT1" ]; then
    echo "M3 PASS: temp=0 deterministic"
  else
    echo "M3 WARN: temp=0 outputs differ (may be non-deterministic sampling)"
  fi
else
  echo "M3 SKIP: no APR model"
fi
```

PASS if M1+M2+M3 all pass. WARN if any are skipped due to missing models.

### Gate 10: Coverage Completeness (F-COV-001 through F-COV-005)

Contract: `contracts/apr-qa-coverage-v1.yaml`

### V1. Contract YAML validity (all 6 QA contracts parse)
```bash
VALID=0; TOTAL=0
for c in contracts/apr-cli-qa-v1.yaml contracts/apr-qa-metamorphic-v1.yaml \
  contracts/apr-qa-silent-fallback-v1.yaml contracts/apr-qa-differential-v1.yaml \
  contracts/apr-qa-chaos-v1.yaml contracts/apr-qa-coverage-v1.yaml; do
  TOTAL=$((TOTAL+1))
  python3 -c "import yaml; yaml.safe_load(open('$c')); print('  VALID: $c')" 2>&1 && VALID=$((VALID+1))
done
echo "V1: $VALID/$TOTAL contracts valid"
[ "$VALID" -eq "$TOTAL" ] && echo "V1 PASS" || echo "V1 FAIL"
```

### V2. Zero High-severity SATD in apr-cli
```bash
SATD_HIGH=$(pmat analyze satd -p crates/apr-cli/ 2>&1 | grep -c "High" 2>/dev/null || echo "0")
echo "V2: $SATD_HIGH High-severity SATD items"
[ "$SATD_HIGH" -eq 0 ] && echo "V2 PASS" || echo "V2 WARN: $SATD_HIGH High SATD items"
```

### V3. Critical modules exercised (no panic on real model)
```bash
. scripts/apr_bin.sh || exit 1
M=$(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" \) -type f | head -1)
if [ -n "$M" ]; then
  V3_PASS=0; V3_TOTAL=0
  for cmd in "hex" "profile --iterations 1"; do
    V3_TOTAL=$((V3_TOTAL+1))
    timeout 30 "$APR" $cmd "$M" 2>&1 | head -3 >/dev/null && V3_PASS=$((V3_PASS+1)) && echo "  V3 $cmd: OK" || echo "  V3 $cmd: FAIL/SKIP"
  done
  for cmd in "serve plan" "train plan"; do
    V3_TOTAL=$((V3_TOTAL+1))
    timeout 10 "$APR" $cmd "$M" 2>&1 | head -3 >/dev/null && V3_PASS=$((V3_PASS+1)) && echo "  V3 $cmd: OK" || echo "  V3 $cmd: FAIL/SKIP"
  done
  echo "V3: $V3_PASS/$V3_TOTAL modules exercised"
  [ "$V3_PASS" -ge 2 ] && echo "V3 PASS" || echo "V3 WARN"
else
  echo "V3 SKIP: no model"
fi
```

### V4. Complexity hotspots tracked
```bash
# Count true CC>15 functions via JSON. The ANSI-coloured text output also
# contains section headers whose last numeric field exceeds 15 (e.g. the
# refactoring-time estimate or the per-file Cyclomatic totals), so awk over
# stdout was over-counting; use the structured format instead.
HIGH_CC=$(pmat analyze complexity -p crates/apr-cli/ --format json 2>/dev/null \
  | jq '[.files[].functions[] | select(.metrics.cyclomatic > 15)] | length' 2>/dev/null \
  || echo "0")
echo "V4: $HIGH_CC functions with CC > 15"
[ "$HIGH_CC" -le 3 ] && echo "V4 PASS" || echo "V4 WARN: $HIGH_CC high-complexity functions"
```

### Gate 10 Verdict (GH-716)
```bash
# Re-compute V1-V4 for a single aggregate verdict. V1 (contracts parse) and
# V3 (critical modules run) are required; V2 (SATD) and V4 (complexity) are
# quality signals that demote PASS → WARN but never cause FAIL on their own.
V1_OK=0
for c in contracts/apr-cli-qa-v1.yaml contracts/apr-qa-metamorphic-v1.yaml \
  contracts/apr-qa-silent-fallback-v1.yaml contracts/apr-qa-differential-v1.yaml \
  contracts/apr-qa-chaos-v1.yaml contracts/apr-qa-coverage-v1.yaml; do
  python3 -c "import yaml; yaml.safe_load(open('$c'))" 2>/dev/null && V1_OK=$((V1_OK+1))
done
V2_SATD=$(pmat analyze satd -p crates/apr-cli/ 2>&1 | grep -c "High" 2>/dev/null || echo "0")
V4_CC=$(pmat analyze complexity -p crates/apr-cli/ --format json 2>/dev/null \
  | jq '[.files[].functions[] | select(.metrics.cyclomatic > 15)] | length' 2>/dev/null \
  || echo "0")
M=$(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" \) -type f | head -1)
V3_OK=$([ -n "$M" ] && echo 1 || echo 0)

if [ "$V1_OK" -eq 6 ] && [ "$V3_OK" -eq 1 ]; then
  if [ "$V2_SATD" -eq 0 ] && [ "$V4_CC" -le 3 ]; then
    echo "Gate 10: PASS (V1=6/6 V2=0 SATD V3=model V4=$V4_CC CC)"
  else
    echo "Gate 10: WARN (V1+V3 pass; V2=$V2_SATD SATD V4=$V4_CC CC)"
  fi
else
  echo "Gate 10: FAIL (V1=$V1_OK/6 V3=$([ "$V3_OK" = "1" ] && echo ok || echo no-model))"
fi
```

PASS requires V1+V3. V2 (SATD) and V4 (complexity) demote PASS → WARN.

### Gate 11: Chaos Engineering (F-CHAOS-001 through F-CHAOS-005)

Contract: `contracts/apr-qa-chaos-v1.yaml`

### C1. Memory budget (RSS sanity check)
```bash
M=$(find ~/models -maxdepth 2 -name "*.gguf" -type f -size -1G | head -1)
if [ -n "$M" ]; then
  MODEL_KB=$(du -k "$M" | cut -f1)
  RSS_KB=$(/usr/bin/time -v timeout 30 apr inspect "$M" 2>&1 | grep "Maximum resident" | awk '{print $NF}' 2>/dev/null || echo 0)
  if [ "$RSS_KB" -gt 0 ]; then
    BUDGET_KB=$(( MODEL_KB * 3 + 524288 ))
    echo "C1: model=${MODEL_KB}KB RSS=${RSS_KB}KB budget=${BUDGET_KB}KB"
    [ "$RSS_KB" -lt "$BUDGET_KB" ] && echo "C1 PASS" || echo "C1 WARN: RSS exceeds 3x model + 512MB"
  else
    echo "C1 SKIP: /usr/bin/time not available"
  fi
else
  echo "C1 SKIP: no small GGUF model"
fi
```

### C2. Overwrite protection
```bash
. scripts/apr_bin.sh || exit 1
touch /tmp/apr-qa-existing.apr
"$APR" convert /dev/null -o /tmp/apr-qa-existing.apr 2>&1; EC=$?
# Should either fail (non-zero) or prompt — never silently overwrite
[ "$EC" -ne 0 ] && echo "C2 PASS: existing file not silently overwritten" || echo "C2 WARN: may have overwritten"
```

### C3. SIGINT handling
```bash
. scripts/apr_bin.sh || exit 1
M_APR=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -n "$M_APR" ]; then
  timeout 5 "$APR" run "$M_APR" "Tell me a very long story about everything" --max-tokens 500 &
  PID=$!
  sleep 2
  kill -INT $PID 2>/dev/null
  wait $PID 2>/dev/null; EC=$?
  # SIGINT should produce exit 130 or similar non-zero, NOT leave zombie
  [ "$EC" -ne 0 ] && echo "C3 PASS: SIGINT handled (exit $EC)" || echo "C3 WARN: SIGINT exit 0"
else
  echo "C3 SKIP: no APR model"
fi
```

PASS if C1+C2+C3 all pass. WARN on skips.

### Gate 12: Differential Testing (F-DIFF-001 through F-DIFF-005)

Contract: `contracts/apr-qa-differential-v1.yaml`

### D1. Cross-format tensor agreement
```bash
. scripts/apr_bin.sh || exit 1
M_GGUF=$(find ~/models -maxdepth 2 -name "*.gguf" -type f | head -1)
M_APR=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -n "$M_GGUF" ] && [ -n "$M_APR" ]; then
  GGUF_COUNT=$("$APR" tensors "$M_GGUF" --json 2>/dev/null | jq length 2>/dev/null || echo 0)
  APR_COUNT=$("$APR" tensors "$M_APR" --json 2>/dev/null | jq length 2>/dev/null || echo 0)
  echo "D1: GGUF tensors=$GGUF_COUNT APR tensors=$APR_COUNT"
  [ "$GGUF_COUNT" -gt 0 ] && [ "$APR_COUNT" -gt 0 ] && echo "D1 PASS: both formats report tensors" || echo "D1 WARN"
else
  echo "D1 SKIP: need both GGUF and APR models"
fi
```

### D2. Ollama parity (if ollama installed)
```bash
if command -v ollama &>/dev/null; then
  OLLAMA_MODELS=$(ollama list 2>/dev/null | tail -n +2 | head -3)
  if [ -n "$OLLAMA_MODELS" ]; then
    echo "D2: ollama available with models — manual parity check recommended"
    echo "D2 SKIP: automated parity not yet wired"
  else
    echo "D2 SKIP: ollama installed but no models"
  fi
else
  echo "D2 SKIP: ollama not installed"
fi
```

### D3. JSON schema consistency across commands
```bash
. scripts/apr_bin.sh || exit 1
M=$(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" \) -type f | head -1)
if [ -n "$M" ]; then
  D3_PASS=0
  for cmd in "inspect --json" "check --json" "list --json" "gpu --json"; do
    timeout 15 "$APR" $cmd $M 2>/dev/null | jq . >/dev/null 2>&1 && D3_PASS=$((D3_PASS+1))
  done
  echo "D3: $D3_PASS/4 JSON outputs valid"
  [ "$D3_PASS" -ge 3 ] && echo "D3 PASS" || echo "D3 WARN"
else
  echo "D3 SKIP: no model"
fi
```

PASS if D1+D3 pass. SKIP for D2 (requires ollama setup).
### Gate 14: APR → GGUF Export Round-trip (F-EXPORT-ROUNDTRIP-001)

Contract: `contracts/apr-export-num-layers-v1.yaml`

Catches [#1865](https://github.com/paiml/aprender/issues/1865) — `apr export
<model>.apr --format gguf` panicking with exit 101 on APR files missing
`num_layers` metadata. Every APR file in the registry must export without
panic; exit 5 (clean ValidationFailed) is acceptable, exit 101 is a FAIL.

```bash
. scripts/apr_bin.sh || exit 1
G14_PASS=0
G14_TOTAL=0
for apr in $(find ~/models -maxdepth 2 -name "*.apr" -type f 2>/dev/null); do
  G14_TOTAL=$((G14_TOTAL+1))
  OUT=$(timeout 60 "$APR" export "$apr" --format gguf -o /tmp/g14-rt.gguf 2>&1); EC=$?
  # IMPORTANT: capture exit code via OUT=$(...); EC=$? — never via pipe (see Pre-Gate note).
  if [ "$EC" -eq 101 ] || echo "$OUT" | grep -qE "thread .* panicked"; then
    echo "G14 FAIL ($apr): panic exit=$EC"
  elif [ "$EC" -eq 0 ] || [ "$EC" -eq 5 ]; then
    G14_PASS=$((G14_PASS+1))
    echo "G14 OK ($apr): exit=$EC (0=success, 5=clean validation error)"
  else
    echo "G14 WARN ($apr): unexpected exit=$EC"
  fi
  rm -f /tmp/g14-rt.gguf
done
[ "$G14_TOTAL" -eq 0 ] && echo "G14 SKIP: no APR models found" \
  || { [ "$G14_PASS" -eq "$G14_TOTAL" ] && echo "G14 PASS: $G14_PASS/$G14_TOTAL exported without panic" \
       || echo "G14 FAIL: only $G14_PASS/$G14_TOTAL clean"; }
```

PASS if every APR file either exports successfully or exits 5. FAIL on any
panic (exit 101 or stderr panic message). SKIP if no APR models in registry.

### Gate 15: validate --quality Sanity (F-VALIDATE-QUALITY-001)

Contract: `contracts/apr-validate-quality-threshold-v1.yaml`

Catches [#1866](https://github.com/paiml/aprender/issues/1866) — `apr validate
--quality` returning Grade F exit 5 on every working model because 22/25
checks are stubbed `Skip(Not implemented)` and the threshold gate compared
against the full 100-point ceiling.

```bash
. scripts/apr_bin.sh || exit 1
# Find a known-good model — one that apr qa says is fine.
M=$(find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" \) -type f | head -1)
if [ -z "$M" ]; then
  echo "G15 SKIP: no model available"
else
  OUT=$(timeout 90 "$APR" validate "$M" --quality 2>&1); EC=$?
  # apr qa is the canonical pass/fail (CLAUDE.md). If qa passes, validate --quality
  # MUST NOT exit non-zero solely because checks are unimplemented.
  QA_OUT=$(timeout 120 "$APR" qa "$M" 2>&1 | grep -E "ALL GATES PASSED|FAIL"); QA_PASSES=$?
  if echo "$QA_OUT" | grep -q "ALL GATES PASSED" && [ "$EC" -ne 0 ]; then
    echo "G15 FAIL: apr qa says ✓ ALL GATES PASSED but apr validate --quality exit=$EC (#1866)"
    echo "         likely score threshold counting Skip(Not implemented) against runnable denom"
  else
    echo "G15 PASS: validate --quality consistent with apr qa verdict (exit=$EC)"
  fi
fi
```

PASS if `apr validate --quality` exits 0 on any model that `apr qa` passes.
FAIL on the inconsistency that #1866 captured.

### Gate 16: `apr run` Exit Code Reflects Output Validity (F-RUN-EXIT-SANITY-001)

Contract: `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml`

Catches the secondary defect from [#1864](https://github.com/paiml/aprender/issues/1864)
— `apr run` exiting 0 even when GPU dispatch produced obvious gibberish.

```bash
. scripts/apr_bin.sh || exit 1
M=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -z "$M" ]; then
  echo "G16 SKIP: no APR model"
else
  OUT=$(timeout 90 "$APR" run "$M" "What is 2+2?" --max-tokens 16 2>&1); EC=$?
  # Heuristic gibberish detectors. Real models answering 2+2 should produce
  # digits or short English. If the output contains chat-template control tokens
  # (e.g. <|im_start|>, <|endoftext|>) repeated, OR is dominated by a single
  # non-numeric word repeating, treat that as a parity-gate-missed regression.
  if echo "$OUT" | grep -qE '<\|im_start\|>.*<\|im_start\|>' \
     || echo "$OUT" | grep -qE '<\|endoftext\|>.*<\|endoftext\|>'; then
    if [ "$EC" -eq 0 ]; then
      echo "G16 FAIL: chat-template gibberish + exit 0 (#1864 secondary)"
    else
      echo "G16 PASS: gibberish detected AND exit=$EC (gate fired)"
    fi
  else
    OUTPUT_LINE=$(echo "$OUT" | sed -n '/^Output:/,$p' | tail -n +2 | tr -d '[:space:]')
    if [ -n "$OUTPUT_LINE" ] && [ "$EC" -eq 0 ]; then
      echo "G16 PASS: clean output, exit=0"
    elif [ "$EC" -ne 0 ]; then
      echo "G16 PASS: non-zero exit=$EC (clean failure path)"
    else
      echo "G16 WARN: output unparseable but exit=0 — inspect manually"
    fi
  fi
fi
```

PASS if `apr run` either emits clean output with exit 0, or non-clean output
with non-zero exit. FAIL when chat-template gibberish leaks through with exit 0.
### Gate 18: Fresh-Convert `.apr` Inference Parity (F-APR-INFERENCE-PARITY-001)

Contract: `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` (the `.apr`↔GGUF inference invariant)

Catches the PMAT-888 class: a `.apr` **converted by the CURRENT binary** produces garbage on
inference (mojibake / cosine ~0.7) while the source GGUF is coherent — a converter/loader
regression. Gate 16 only runs a PRE-EXISTING `~/models/*.apr` (converted by an OLD binary, so it
still works), and `inspect`/`validate`/`tensors` (Gate 2a) all pass on a broken-for-inference `.apr`
— so none of them catch it. The native `.apr` format is the whole project; its inference path MUST
be gated on a FRESH convert against the GGUF, on BOTH CPU and GPU. This is the gate the 0.50.0
post-publish QA was missing (it tested `.apr` inspect/validate but never `.apr` *run*).

```bash
. scripts/apr_bin.sh || exit 1
M_GGUF=$(find ~/models -maxdepth 2 -name "*.gguf" -type f -size -3G | head -1)
if [ -z "$M_GGUF" ]; then echo "G18 SKIP: no GGUF model"; else
  # NB: `apr convert` REQUIRES --quantize (or --compress). Omitting it fails with
  # "At least one of --quantize or --compress must be specified" and writes NO .apr —
  # which a naive test misreads as empty inference output. ALWAYS pass --quantize.
  "$APR" convert "$M_GGUF" --quantize q4k -o /tmp/g18-fresh.apr 2>&1 | tail -1
  [ -f /tmp/g18-fresh.apr ] || echo "G18 FAIL: apr convert produced no .apr (forgot --quantize?)"
  norm(){ sed -n '/^Output:/,$p' | tail -n +2 | tr -d '[:space:]'; }
  # The P0 (PMAT-888) was GARBAGE, not a verbosity diff. GGUF runs may be response-CACHED
  # (terser, e.g. "4" vs ".apr"'s fresh "2+2 equals 4."), so the gate is the .apr's
  # COHERENCE (the real P0 signal), NOT byte-equality with GGUF.
  coherent(){ [ -n "$1" ] && echo "$1" | grep -qE '[0-9A-Za-z]' \
              && ! echo "$1" | grep -qE 'ä|ã|�|<\|im_start|<\|endoftext'; }
  GGUF_OUT=$(timeout 120 "$APR" run "$M_GGUF"        --no-gpu --prompt "What is 2+2?" --max-tokens 12 2>&1 | norm)
  APR_OUT=$( timeout 120 "$APR" run /tmp/g18-fresh.apr --no-gpu --prompt "What is 2+2?" --max-tokens 12 2>&1 | norm)
  echo "G18 gguf=[$GGUF_OUT] apr=[$APR_OUT]"
  if coherent "$APR_OUT"; then
    [ "$GGUF_OUT" = "$APR_OUT" ] && echo "G18 PASS: fresh .apr CPU inference coherent AND == GGUF" \
                                 || echo "G18 PASS: fresh .apr CPU inference coherent (differs from possibly-cached GGUF; both valid answers)"
  elif [ -z "$APR_OUT" ]; then
    echo "G18 FAIL: fresh .apr produced NO output (broken inference, or convert wrote no model)"
  else
    echo "G18 FAIL: fresh .apr produces garbage while GGUF coherent (PMAT-888 converter/loader regression)"
  fi
  # GPU leg (if a GPU is present): the .apr GPU path must also be coherent
  if apr gpu 2>&1 | grep -qiE 'cuda|gpu.*(yes|available|RTX|GB10)'; then
    APR_GPU=$(timeout 120 "$APR" run /tmp/g18-fresh.apr --prompt "What is 2+2?" --max-tokens 12 2>&1 | norm)
    coherent "$APR_GPU" && echo "G18-GPU PASS: fresh .apr GPU coherent" \
      || echo "G18-GPU FAIL: fresh .apr GPU=[$APR_GPU] not coherent"
  fi
  rm -f /tmp/g18-fresh.apr
fi
```

PASS if a freshly-converted `.apr`'s CPU (and GPU, when present) inference matches the source GGUF
token-for-token. FAIL on garbage (the PMAT-888 regression). SKIP if no GGUF model is available.


## Tier 3 — Release only

| ID | Gate | Note |
|---|---|---|
| T3.1 | 7B inference smoke | `F-7B-INFERENCE-001` |
| T3.2 | `cargo publish --dry-run` | authoritative: cargo's own registry resolution |
| T3.3 | version-unpublished | depends on G0.2 |
| T3.4 | security, second source | cargo-deny's GREEN is only as wide as RustSec |
| T3.5 | **Clean-room publishability** | **the hard gate — every crate builds from crates.io alone, no sibling-path tricks. Runs on `intel`. Name it first in any release plan.** |
T3.1's body is carried **verbatim** from v2.0 Gate 17.

### Gate 17: 7B Inference Smoke (F-7B-INFERENCE-001)

Catches [#1864](https://github.com/paiml/aprender/issues/1864) directly. The
README claims `Qwen2.5-Coder 7B Q4_K 225+ tok/s RTX 4090` as the headline
configuration; if 7B GPU inference produces gibberish, the canonical demo
is broken.

```bash
. scripts/apr_bin.sh || exit 1
M_7B=$(find ~/models -maxdepth 2 -name "*7b*q4*" -type f 2>/dev/null | head -1)
if [ -z "$M_7B" ]; then
  echo "G17 SKIP: no 7B Q4_K model in registry"
else
  # apr qa Golden Output gate already encodes correctness; reuse it.
  OUT=$(timeout 300 "$APR" qa "$M_7B" 2>&1 | grep -E "Golden Output")
  if echo "$OUT" | grep -q "FAIL"; then
    echo "G17 FAIL: 7B Golden Output gate FAILS — $OUT (#1864)"
  elif echo "$OUT" | grep -q "PASS"; then
    echo "G17 PASS: 7B Golden Output gate passes"
  else
    echo "G17 SKIP: Golden Output gate didn't run (no GPU? --assert-gpu missing?)"
  fi
fi
```

PASS when `apr qa` Golden Output gate passes on the 7B Q4_K model. FAIL on
the regression that #1864 captured. SKIP when the 7B model isn't available
or the gate didn't run.
## Pre-Gate Note: Exit-Code Capture Methodology (lesson from 2026-05-22 dogfood)

When a falsifier needs to assert "command X exits Y", **never** chain through
a pipe and read `$?` — `$?` after a pipe reports the LAST command's status,
not the original command's. Two real bugs were filed in a 2026-05-22 dogfood
session and immediately retracted as false positives because of this:

```bash
. scripts/apr_bin.sh || exit 1
# WRONG — $? is head's exit, not apr's
"$APR" publish /nonexistent paiml/test 2>&1 | head -8; echo "exit=$?"   # always 0

# RIGHT — captures the command's real exit code
OUT=$("$APR" publish /nonexistent paiml/test 2>&1); EC=$?
echo "$OUT" | tail -1; echo "exit=$EC"
```

All new gates (G13-G17) follow the `OUT=$(...); EC=$?` pattern. Existing
gates that still pipe-then-`$?` should be migrated when next touched.

See [memory/feedback_test_methodology_can_fake_bugs.md] for the broader lesson.



---

# Phase 4 — Transport gates

Three transports ship (CLI, HTTP, MCP). Reachability is necessary and not
sufficient.

## G4.1 — Transport declaration

`[package.metadata.transports]` in `Cargo.toml`, versioned with the code it
describes:

```toml
[package.metadata.transports]
cli  = { e2e = "e2e_cli_t" }
mcp  = { e2e = "e2e_mcp_stdio_t", features = ["mcp"] }
http = { e2e = "e2e_http_serve_t", features = ["http", "lua"] }
```

No declaration → RED. An undeclared transport is an unverified one. A transport
declared as a bare bool → RED; a declaration with no `e2e` target verifies nothing.

**At HEAD (`4bbfeb07f`) this gate is RED**: neither `crates/apr-cli/Cargo.toml` nor
the root `Cargo.toml` contains a `[package.metadata.transports]` block, while the
binary ships all three transports. That is a real finding, not a hypothetical — and
it means G4.2 and G4.3 must appear in the first receipt as SKIP naming this blocker.

When this gate is RED, the two downstream gates must still **appear in the receipt
as SKIP, each naming what blocked it**. A gate that vanishes from the receipt reads
as one that passed.

## G4.2 — Interface parity

Each declared target must (1) exist, (2) reference `CARGO_BIN_EXE_` — spawn the
**shipped binary**, not the library, which is the reachability property a
library-level suite structurally cannot see — and (3) run at least one passing
test. `0 tests, ok` is a vacuous pass and FAILS.

## G4.3 — Transport absence

Probe the real binary for **undeclared** transports. A transport that exists and
is not declared is RED.

## G4.4 — Transport invariance (`invariance.py`)

Stand every transport up **simultaneously**, derive the verb list **from the
binary**, invoke one verb through all live transports, compare byte-for-byte.

Three properties, each load-bearing:

- **Simultaneous, not sequential.** Sequential cannot distinguish "the transports
  agree" from "the transports share a process-global only one may hold at a time."
  All-live is the configuration a real client fleet produces, and the one that
  surfaces a shared listener, a shared lock, or a single-owner runtime.
- **Derived, not hand-written.** A hand-written probe tests the verbs someone
  remembered. A derived one tests the surface that shipped and grows with it.
- **Identical AND valid.** Two identically-wrong strings must not pass. The payload
  is parsed as JSON after the equality check.

Vacuity guards: an empty verb list dies (`a parity check over an empty surface is
vacuous`); a probe verb absent from the binary's own list dies; fewer than two
invocable transports reports `INVARIANCE_SKIP`, never PASS.

Readiness probes by **connecting**, never by binding — a probe that binds competes
with the server it waits for and can starve it.

**Known live defect this gate should already be catching:** seven native routes,
including the banner-advertised `/stream/generate` and `/batch/generate`, always
fail with "No model available" on the standard `apr serve run model.gguf` path
while `/generate` on the same server works (#2376, P0, **closed COMPLETED
2026-08-13** — the ledger row still shows no `Fixed by`; re-probe rather than
assume, and treat this as the regression the gate now ratchets).

---

# Phase 5 — Fleet hardware matrix

`--fleet` / `--release`. Resolves the 427 UNKNOWN `verified_hardware` rows.

| Host | Compute | Backend | Role |
|---|---|---|---|
| `lambda-labs` | RTX 4090, 24 GB | CUDA sm_89 | primary dev/control; **not a CI runner** (retired 2026-05-10, do not revive) |
| `gx10` | GB10 Blackwell, 120 GB unified | CUDA 13.0 **sm_121** | aarch64-linux |
| `mini` | Apple M4, 16 GB unified | Metal | aarch64-macos; cowork-first, selective CI |
| `intel` | Xeon W-3245, 283 GB | CPU | clean-room CI runner, 8 concurrent, memory-bound |

## G5.1 — `verified_hardware` provenance

A row may claim a hardware target **only** if a CI job actually ran the feature on
that target, or a Cargo feature gate compiles it. **Never infer hardware from a
feature's name.** A CUDA kernel with no job that runs it is `UNKNOWN`.

Live example of why: `dispatch.rs:180-182` tells the user only the root crate's
feature chain can enable the CUDA path. That is false —
`crates/apr-cli/Cargo.toml:89` defines `cuda` directly, and
`.github/workflows/qwen-story-daily.yml:62` builds the nightly CUDA binary with
`cargo install --path crates/apr-cli --force --features cuda` — bypassing the root
facade entirely. A user-facing message asserting a
false hardware constraint is itself a defect.

## G5.2 — Canonical benchmark

One model, one quant, one pinned `llama.cpp` commit, four hosts. See
`APR-BENCH-RFC-001` — **which does not exist**: zero hits across the whole tree at
HEAD. Author it before this gate can run; until then G5.2 reports SKIP naming the
missing RFC, never PASS. Report `pp512` and `tg128` **separately** — GB10 legitimately
loses ~4× on decode while winning prefill, and a blended figure reports a correct
machine as broken.

`gx10` requires `-DGGML_CUDA_ARCHITECTURES=121`. Omitting it JITs from stale PTX —
a large, **silent** loss that reads as "GB10 is just slow."

## G5.3 — Orchestration is an OPEN DESIGN QUESTION

Two of four hosts are not general CI runners; `intel` is the clean-room runner this
would contend with. A four-host release-blocking gate has no obvious CI home.
**Escalate; do not default.** Record the decision and its rationale in the ticket.

---

# Phase 6 — Receipt and verdict

## Receipt

Deterministic, byte-identical for the same tree. No timestamps or durations in the
body. Contains:

1. Identity — crate, version, HEAD SHA, feature set used, exclusions **named**
2. Surface — counts per kind, with the vacuity threshold beside each
3. **Coverage — the Phase 2 table, overall / per binary / per band, vs. baseline**
3b. **The allocation table (below) — every cluster, gate count, share of gate
    effort, EARNED vs inherited gates, and cluster coverage REPORTED WITH THE
    FEATURE FRACTION on the same line.** The pairing is checked mechanically:
    point the gate at the receipt and it will refuse an unpaired ratio —
    `DOGFOOD_RECEIPT=<path> bash scripts/check_dogfood_coverage.sh`
3c. **Cluster reassignments — every feature that changed `cluster_label` since
    the comparand, with from, to and reason.** Empty is the normal case. A move
    makes a cluster's gate INHERITED, and an inherited gate does not close a zero
4. **The 44 — uncovered features with quality ≤ 4, enumerated**
5. Gate results — every gate, PASS/FAIL/SKIP/WARN, with the blocker named on SKIP
6. Transport matrix — declared, parity, absence, invariance
7. Fleet matrix — hardware verification per host (`--fleet`)
8. Gaps — every enumeration not completed, and the exact artifact to close it

## The allocation table — belongs IN the receipt, every run

A standing reminder of where the marginal gate is worth least. Measured
2026-08-22 from `docs/audits/surface_audit.csv`; re-derive with
`python3 scripts/dogfood_baseline.py  # section: per_cluster` — **the CSV wins over
this table**, which is a dated sample.

| cluster | n | gates | share of all gate effort | cluster coverage |
|---|---:|---:|---:|---:|
| `apr-lint-diag` | 68 | 55 | 38.7% | 80.9% |
| `http-apr-serve` | 44 | 39 | 27.5% | 88.6% |
| `apr-core-commands` | 109 | 38 | 26.8% | 34.9% |
| *(11 others)* | 611 | 10 | 7.0% | 1.6% |

**93.0% of gate effort sits over 26.6% of the surface. 142 gates / 832 features.**

Nobody chose that allocation; it accreted. Clustering is what makes it visible.
Adding a 56th gate to `apr-lint-diag` buys less than the FIRST gate in
`contracts-pv` (0 of 76) or `http-orchestrate-banco` (0 of 95).

The nine clusters at zero, largest first: `http-orchestrate-banco` (95),
`contracts-pv` (76), `data-pipeline` (76), `orchestrate-agent-stack` (56),
`test-harness` (49), `rag-eval` (44), `qa-cgp` (37), `simulation` (18),
`orchestrate-pacha-secrets` (17).

**Report both numbers or neither (T2).** "5 of 14 clusters gated (35.7%)" without "143 of 833 features gated (17.2%)" beside it is a proxy masquerading as coverage, and the gate refuses to emit it — *on this line too*.

The rule is about the NUMBER, not about a phrasing, and it is enforced on every
surface that can emit one: the gate's own report, the receipt, the output of
`scripts/dogfood_baseline.py`, and this file. A ratio whose denominator is the
cluster count is a cluster-level claim whatever words surround it, and it must
carry a ratio whose denominator is the feature count. Both denominators are
derived from the ledger at run time; neither is a literal in a gate file.

    python3 scripts/lib/dogfood_coverage_gate.py \
        --head docs/audits/surface_audit.csv \
        --pair-scan .claude/skills/apr-dogfood/SKILL.md \
        --pair-scan <your-receipt>

**Honest limit:** the pairing is per LINE, because "beside it" is what makes the
two numbers readable together. A cluster ratio on one line and its feature ratio
forty lines away satisfies neither a reader nor this rule. A line that genuinely
must state one number alone opts out by saying so, with a reason, in the same
shape as the T1 pragma: `t2-pairing allow (<reason>)`. Silence is not an opt-out.

## Verdict

| | Condition |
|---|---|
| **GO** | Every gate green; coverage ≥ baseline; every per-cluster floor held; the 44 empty or fully waived |
| **WARN** | All gates green; a coverage floor unchanged but not improved |
| **NO-GO** | Any gate red; **or** coverage below baseline; **or** a cluster lost a gate; **or** an unwaived quality-≤4 uncovered feature; **or** (at `--release`) a cluster with zero gates |

Coverage regression alone is NO-GO. That is the whole design: the suite can be
incomplete, and it may not silently become *more* incomplete.

---

# Mutation registry

**A gate with no registered mutation is inadmissible.** Every gate names an
injectable change that must turn it RED, and the pre-fix GREEN is captured as
before-evidence. Discrimination check on every one: RED for the mutation, GREEN for
a no-op rebuild of the same commit. A gate that fires on both measures nothing.

| Gate | Mutation → RED |
|---|---|
| G0.1 provenance | Put a stale `apr` earlier on `PATH` |
| G0.2 identity | Set `version.workspace = true` and drop the fallback |
| G0.5 dup binary | Add an allocator attribute to one entry point only |
| G1.1 vacuity | Truncate the `--help` parse to 2 commands |
| G1.3 value_parser | Remove `BACKEND_VALUES` from one of the three sites |
| G1.4 help/dispatch | Add a dispatch arm without documenting it |
| G2.1 freshness | Backdate the CSV behind HEAD |
| G2.2 reconciliation | Delete one CSV row for a live command |
| G2.3 floors | Flip one `in_dogfood_skill` from `yes` to `no` |
| G2.5 per-cluster | Move a cluster's **only** gate to another cluster — totals unchanged, so only the per-cluster floor can explain the RED (asserted by finding text, not exit code) |
| G2.5 membership | Move a **gated** feature into a zero-gate cluster and write one new gate in the cluster it left. Every count still holds — the pre-fix gate printed `clusters gated 3/3 (100.0%)` and exited 0 — so only the membership ratchet can explain the RED [t2-pairing allow (quoting the pre-fix verdict)] |
| G2.5 earned | **Declare** that same move and arm `--release`: still RED, because the gate that walked in is inherited, not earned. Paired GREEN: at the same arm, *write* a gate in the zero cluster instead |
| G2.5 / T2 pairing | Delete the feature fraction from the report emitter → RED |
| T2 pairing, receipt | State a cluster ratio in the **receipt** with no feature fraction → RED. The rule was once applied to one generated string; a receipt is a different channel and looked stricter while measuring less |
| T1 id ban | Key a contract on `cluster_id` instead of `cluster_label` → RED |
| T1 allowlist | Key on the id with a form no blacklist listed — a pandas-style `groupby` on the id column → RED. Twelve such constructs were GREEN under the four-syntax version. (Writing that call out here would itself be a keying line: the guard is an allowlist and prose is exempt only when the token is *directly* backticked, so `key: ` + the id inside a code span still fails. That is the rule working, not a false positive.) |
| G3.0 sweep | Close a defect ticket in cluster X with X's uncovered members unenumerated → the ticket does not close |
| P6 / T2.9 parity | Stride the GPU cache by `q_dim` (the #749 bug) → red by step 8 |
| P7 NaN sentinel | Disarm the threshold comparison |
| T2.2 metamorphic | Perturb `absmax` on quant roundtrip 2 |
| G4.1 decl | Remove one transport from `[package.metadata.transports]` |
| G4.2 parity | Point one `e2e` at a target with 0 tests |
| G4.4 invariance | Return a differing field on one transport only |
| G5.1 hardware | Claim `nvidia-cuda` for a feature no job runs |
| T3.5 clean-room | Introduce a sibling-path dependency |

---

# Anti-patterns — do not do these

| | Why |
|---|---|
| Write the surface list into this file | The defect this repo keeps re-finding — 36/77/103/111, every one a stale list failing silently |
| Assert `is_ok()` on invalid input | Locks the defect in. Found in the 0.63.0 audit |
| Treat exit 0 as PASS | 16 open P0s (of 24 total) include commands that exit 0 while printing FAIL |
| Single-sample a temporal path | CF-4 / #1864 — green gate, gibberish output |
| Blend `pp512` and `tg128` | Reports a correct GB10 as broken |
| Reconcile a denominator mismatch silently | Turns the ledger back into a list |
| `--all-features` as a fallback | Measures the `broken-tests` quarantine; permanent RED that everyone walks past |
| Drop a blocked gate from the receipt | A vanished gate reads as a passed one |
| Resolve a bare `apr` | #2384 — a 0.63.0 process ran a 0.60.0 backend; P0, closed 2026-08-11 by #2424, now a ratchet |
| Set a coverage threshold before measuring | Either never fires or fires constantly |
| Report cluster coverage without the feature % | T2 — the vacuity failure one level up, looking stricter while measuring less |
| Enforce a reporting rule on ONE channel | The receipt, the baseline printer and this file are three more. One call site is three ways around it |
| Move a gated feature into a zero-gate cluster | The zero stops EXISTING instead of being closed. Same move as deleting a losing benchmark row (d7e08043b, the only one in this repo's history) |
| Re-cluster silently | Clusters are derived, so moves are legitimate — *declared* ones. `docs/audits/cluster_reassignments.yaml`, with from, to and a reason |
| Key a contract, gate or waiver on `cluster_id` | T1 — k-means labels permute; the obligation silently re-points |
| Ban a token by listing its syntaxes | "Nothing may key on this" is universal; no finite list of syntaxes carries a universal claim. Allowlist the legitimate positions instead |
| Derive `quality_1_10` from cluster membership | T3 — clustering is a prior, never evidence |
| Regenerate `cluster_label` from a re-run | The labels are human-owned; regenerating re-points every obligation citing them |
| Keep a second, clustered copy of the ledger | Two ledgers over one surface; the copy goes stale in silence |
| Close a cluster-X defect without sweeping X | 201 findings collapsed to 37 causes — ~5.4 per cause. The siblings are already broken |
| Bypass a red gate to ship | Stop the line. Five-whys to the owning module |

---

# Escalation

File, don't fix. This skill is read-only.

- `pmat work add` first, then branch → PR → `ci / gate`
- Contract (`pv`) in the same PR as the fix
- Six-part DoD: merged green · gate exists · mutation observed RED · `pv` contract ·
  discrimination confirmed · invalidated doc claims updated in the same PR
- Escaped defect → case file, append-only, plus a **permanent** falsifier. The
  regression ratchet is the point

Ambiguous design decisions escalate. They do not get resolved silently.

---

# Provenance of this file — read before editing

Written 2026-08-22 against `paiml/aprender` `main`.

**Sections I authored from full sources:** Phase 0 (from `dogfood.sh` — its inline
post-mortems are quoted, not paraphrased), Phase 2 (from `surface_audit.csv`, 830
rows, all tallies computed not estimated), Phase 4 (from `dogfood.sh` transport
sections and `invariance.py` in full), the mutation registry, the anti-pattern
table.

**The carry-over is COMPLETE.** Tier 0 (Gates 4–7), Tier 1 (Gates 2–3 / P1–P12),
Tier 2 (Gates 8–12, 14–16, 18), Tier 3 (Gate 17) and G0.3 (Gate 13) all carry
their v2.0 bodies **verbatim**, spliced under PMAT-742 rather than re-derived. A
re-derived body loses the specific mutation each one was hardened against, which
is why they exist in that exact wording.

**Numbers requiring verification before this file is committed:**

| Claim | Source | Status |
|---|---|---|
| 830 features / 28 binaries | `surface_audit.csv` | computed from the file |
| 17.1% coverage, 27 binaries at 0% | same | computed |
| 44 uncovered with quality ≤ 4 | same | computed |
| 427 UNKNOWN hardware, 204 low-conf uncovered | same | computed |
| 201 findings / 37 clusters, 171 open, 16 open P0 | `dogfood-0.63.0-ledger.md` | computed |
| G1.3 substantive claim (serve lacks `value_parser`) | HEAD, 2026-08-22 | **VERIFIED** — `serve_commands.rs:73`, 0 `BACKEND_VALUES` |
| G1.3 `extended_commands.rs:85`, `serve/types.rs:57` | HEAD | **VERIFIED** — note `:85` is the field, the `#[arg]` line is `:84` |
| G1.3 `commands_enum.rs:110` | HEAD | **WAS STALE, FIXED** → `:154` (`#[arg(… value_parser = BACKEND_VALUES)]`), field `:155`, const `:10` |
| G1.4 (10 arms, 2 documented) | HEAD, `dispatch_analysis.rs:1383-1419` | **VERIFIED** — exactly 10 `Some(..)` arms, `classify` at `:1383` is `#[cfg(feature = "training")]`-gated; default `_ =>` perplexity arm at `:1427`; `extended_commands.rs:138` documents 2 |
| G0.5 (`dhat` divergence) | HEAD, `crates/apr-cli/src/main.rs:10-12` | **VERIFIED** — `src/bin/apr.rs:9-10` delegates to `cli_main()` |
| G5.1 `dispatch.rs:180-182` | HEAD | **VERIFIED** — the "build the root, not `-p apr-cli`" claim is at `:180`; it is false |
| G5.1 `apr-cli/Cargo.toml:89` | HEAD | **VERIFIED** — `cuda = ["inference", "realizar/cuda", …]` |
| G5.1 `qwen-story-daily.yml:63` | HEAD | **WAS STALE, FIXED** → `:62`, and the invocation is `--path crates/apr-cli --features cuda`, not `-p apr-cli` |
| `APR-BENCH-RFC-001` (G5.2) | tree-wide grep | **STALE — DOES NOT EXIST**; marked to-be-authored inline |
| `[package.metadata.transports]` (G4.1) | HEAD `Cargo.toml`s | **ABSENT** — G4.1 is RED at HEAD; recorded inline |
| #2384 "still open", #2376 "open" | `gh issue view` | **STALE — both CLOSED COMPLETED** (2026-08-11, 2026-08-13); corrected in three places |
| "24 open P0s" | ledger table | **STALE** — 24 is the P0 *total*; **16** are open. Corrected |
| "644 features scored 6" | CSV | **IMPRECISE** — 672 scored 6; 644 of those are *uncovered*. Corrected |
| 830 rows / 28 binaries / 142 covered / 17.1% | CSV, recomputed | **VERIFIED** |
| 44 quality ≤ 4 uncovered (all in `apr`) | CSV, recomputed | **VERIFIED** |
| 427 UNKNOWN hardware, 204 low-conf uncovered | CSV, recomputed | **VERIFIED** |
| band splits 58/80, 27/49, 29/673, 8/8, 20/20 | CSV, recomputed | **VERIFIED** |
| 201 findings / 37 clusters / 171 open / 16 open P0 | ledger, recomputed | **VERIFIED** |
| `MAX_SKIP_PCT` bound in `dogfood_surfaces.sh` | HEAD `:70` | **VERIFIED** — default 40 |
| Baseline threshold values in G2.3 | CSV, 2026-08-22 @ `4bbfeb07f` | **MEASURED AND COMMITTED** |
| v2.0 gate bodies (Gates 2–18) | `origin/main` SKILL.md | **SPLICED VERBATIM** — contiguous-block equality asserted for all 12 slices |

The CSV is a snapshot. The moment the surface moves, the coverage number is stale —
which is exactly what G2.1 exists to detect.

---

## Cleanup

```bash
rm -f /tmp/apr-qa-*.{gguf,apr,enc,jsonl,safetensors}
```
