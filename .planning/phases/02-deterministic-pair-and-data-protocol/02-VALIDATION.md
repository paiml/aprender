---
phase: 2
slug: deterministic-pair-and-data-protocol
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-08
updated: 2026-08-08 (revised after cross-AI review; re-revised after plan-checker verification)
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) + proptest + trybuild + pv (contract validation) |
| **Config file** | Cargo.toml (workspace) / .pmat-gates.toml |
| **Quick run command** | `cargo test -p <crate-under-change> --lib` |
| **Full suite command** | `make tier2` then tier3 contract validation over $(CONTRACTS) |
| **Estimated runtime** | ~60 seconds (quick), ~5 min (full) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-under-change>` (new crate is seconds)
- **After every plan wave:** Run `make tier2` + `make contrastive-data-boundary` (once it exists)
- **Before `/gsd:verify-work`:** `make tier3` green including pv validate AND `make contract-audit-phase2` (binding coverage) over BOTH phase contracts

### Feedback latency — TIERED, and the numbers are honest

The previous single "< 120 s" ceiling was not true of two tasks in this phase, and the sign-off
box below ticked it anyway (checker warning 6). The ceiling is now stated per class of task:

| Class | Ceiling | Tasks | Why |
|-------|---------|-------|-----|
| Per-task inner loop | **120 s** | every task except the two below | `cargo test -p aprender-contrastive-data <module>` and `cargo test -p apr-cli --lib data_*` are seconds on a warm tree |
| Gate sweep | **900 s** | 02-08.T3 | Chains `make tier2` + `contrastive-data-boundary` + `contract-audit-phase2` + `setfit-feature-matrix` + `cargo check --workspace`. This is a deliberate end-of-phase sweep, not an inner loop; it runs once per wave-6 pass |
| Full-crate CLI suite | **600 s** | 02-09.T3 | `cargo test -p apr-cli --lib` is 4,158 tests. **Fast signal returns first:** every 02-09 task's `<automated>` runs `cargo test -p apr-cli --lib data_contrastive` (seconds) BEFORE the full suite, and 02-09.T2 does not run the full suite at all — only T3 does, once, as the plan's exit check |
| Mutation run | **2700 s** (bounded, reported partial on expiry) | 02-08.T3 | `timeout 2700 cargo mutants -p aprender-contrastive-data --timeout 60`; expiry is reported as a PARTIAL result, never as a pass |

No task's INNER loop exceeds 120 s. The three exceptions above are once-per-plan exit sweeps and
are declared as such rather than hidden behind a ceiling they do not meet.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01.T1 | 02-01 | 1 | DATA-01/02 | T-02-01/02/33 | Baseline lands hash-attested in its own PR, tests green | unit | `shasum -a 256 -c /tmp/d06-baseline.sha256 > /tmp/v.log 2>&1; rc=$?` + `cargo test -p apr-cli --lib data_tweeteval > /tmp/t.log 2>&1; rc=$?` | ✅ (uncommitted → committed by task) | ⬜ pending |
| 02-01.T2 | 02-01 | 1 | DATA-01/02 | T-02-06/34 | Contract pv-valid + tier3-reachable; CLAUDE.md `pv diff` corrected | contract | `cargo run --release -p aprender-contracts-cli --bin pv -- validate contracts/tweet-eval-stance-benchmark-v1.yaml > /tmp/pv.log 2>&1; rc=$?` | ✅ | ⬜ pending |
| 02-02.T1 | 02-02 | 2 | DATA-02..06 | T-02-04/36 | Publishable crate, 3 workspace deps added, module skeleton, typed error surface | unit + scripts + package | `cargo check -p aprender-contrastive-data && cargo package --no-verify -p aprender-contrastive-data` (statuses captured separately) | ❌ W0 (created by task) | ⬜ pending |
| 02-02.T2 | 02-02 | 2 | DATA-02..06 | T-02-06 | Contract part 1: 24 equations + proof obligations, no dangling contract references | contract (structure) | `grep -v '^ *#' contracts/contrastive-pair-protocol-v1.yaml \| grep -c "rand-philox-v1"` == 0 + equation/obligation ID presence | ❌ W0 (created by task) | ⬜ pending |
| 02-02.T2b | 02-02 | 2 | DATA-02..06 | T-02-06 | Contract part 2: falsification tests + Kani declarations + qa_gate → FIRST pv-green state | contract | `pv -- validate contracts/contrastive-pair-protocol-v1.yaml > /tmp/pv.log 2>&1; rc=$?` | ❌ W0 (created by 02-02.T2) | ⬜ pending |
| 02-02.T3 | 02-02 | 2 | DATA-05 | T-02-05/35 | D-04 boundary enforced by positive allowlist + src-wide symbol ban, inside a tier | make gate | `make contrastive-data-boundary > /tmp/cdb.log 2>&1; rc=$?` | ❌ W0 (created by task) | ⬜ pending |
| 02-03.T1 | 02-03 | 3 | DATA-01/02 | T-02-08 | deny_unknown_fields; dual hashes; identity-complete fingerprint | unit/property | `cargo test -p aprender-contrastive-data schema` / `hash` | ❌ new | ⬜ pending |
| 02-03.T2 | 02-03 | 3 | DATA-02/06 | T-02-07/10 | Typestate gate ladder (pub(crate) ctors) + persistable ledger | unit | `cargo test -p aprender-contrastive-data split` / `ledger` | ❌ new | ⬜ pending |
| 02-03.T3 | 02-03 | 3 | DATA-02/06 | T-02-09/37/38 | Union-find coalesced dedup + PreparedDataset profile typestate (D-27 both halves) | unit + doctest | `cargo test -p aprender-contrastive-data dedup` / `prepared` / `--doc` | ❌ new | ⬜ pending |
| 02-04.T1 | 02-04 | 3 | DATA-04/05 | T-02-12/39 | Measured-vs-contracted fixture families incl. K=N + env attestation | fixture gen | `cd .../setfit_reference && shasum -a 256 -c manifest.sha256 > /tmp/f.log 2>&1; rc=$?` | ❌ new | ⬜ pending |
| 02-04.T2 | 02-04 | 3 | DATA-04/05 | T-02-11 | Working-dir-independent fixture integrity + shared test models | integration | `cargo test -p aprender-contrastive-data --test reference_fixtures` | ❌ new | ⬜ pending |
| 02-05.T1 | 02-05 | 4 | DATA-03 | T-02-14/16 | Indexed pure RNG, frozen LE encoding golden, NonZeroU64 bound | unit/property | `cargo test -p aprender-contrastive-data rng` | ❌ new | ⬜ pending |
| 02-05.T2 | 02-05 | 4 | DATA-03 | T-02-15 | Labeled selection over ONE PreparedDataset<Canonical>, ten contracted seeds, **plus the digest-free SelectionPayload the seed loop asserts on — checker blocker 3** | unit/property | `cargo test -p aprender-contrastive-data select` + `... buckets` + `... payload` | ❌ new | ⬜ pending |
| 02-05.T3 | 02-05 | 4 | DATA-03 | T-02-13/40/41 | Manifest envelope + ledger-consistency guard + strict replay (12 DISTINCT rejection tests) + include_bytes! goldens | unit + golden | `cargo test -p aprender-contrastive-data manifest` | ❌ new | ⬜ pending |
| 02-06.T1 | 02-06 | 4 | DATA-01/02 | T-02-42/43 | Dataset attestation boundary (9 rejection tests) + crate-owned schema-version constants. **Reordered ahead of the adapter — checker blocker 2** | unit | `cargo test -p aprender-contrastive-data attestation` | ❌ new | ⬜ pending |
| 02-06.T2 | 02-06 | 4 | DATA-01/02 | T-02-17/20 | Thin adapter on the D-05 seam, byte parity, schema v2 policy — consumes T1's `DatasetAttestation` | unit + golden | `cargo test -p apr-cli --lib data_tweeteval` | ✅ (refactor) | ⬜ pending |
| 02-06.T3 | 02-06 | 4 | DATA-02 | T-02-18/19 | Real-duplicate live golden (exactly one group) + contract growth | opt-in network + contract | `cargo test -p apr-cli --lib data_tweeteval -- --ignored` + pv validate | ✅ (extended) | ⬜ pending |
| 02-07.T1 | 02-07 | 5 | DATA-04/05 | T-02-22 | Fallible capacity, binding hard cap, total degenerate policy | unit + fixture | `cargo test -p aprender-contrastive-data pairs` AND `cargo test -p aprender-contrastive-data --test pair_counts` (the fixture-driven host target — checker warning 4) | ❌ new (both) | ⬜ pending |
| 02-07.T2 | 02-07 | 5 | DATA-04/05 | T-02-21/44/45 | O(K) streaming sampler, public state diagnostics, untrusted-pair validation | property | `cargo test -p aprender-contrastive-data pairs` | ❌ new | ⬜ pending |
| 02-07.T3 | 02-07 | 5 | DATA-04/05 | T-02-23/24 | Tuple-committing streamed hash + dump round-trip + goldens | unit + golden | `cargo test -p aprender-contrastive-data manifest` | ❌ new | ⬜ pending |
| 02-08.T1 | 02-08 | 6 | DATA-05/06 | T-02-25/26 | In-band negatives RED + mirrors GREEN, incl. K=N adversarial | in-band negative | `cargo test -p aprender-contrastive-data --test negative_leaky` / `--test negative_materializing` | ❌ new | ⬜ pending |
| 02-08.T2 | 02-08 | 6 | DATA-06 | T-02-27 | Leakage not constructible (5 trybuild cases vs the profile typestate) | trybuild | `cargo test -p aprender-contrastive-data --test ui` | ❌ new | ⬜ pending |
| 02-08.T3 | 02-08 | 6 | DATA-05/06 | T-02-28/46/47 | Binding coverage gate + bounded mutation + full gate sweep + packaging | mutation + gates | `make tier2 && make contrastive-data-boundary && make contract-audit-phase2 && make setfit-feature-matrix` (statuses captured separately; gate-sweep ceiling 900 s, mutation 2700 s bounded) | ❌ new | ⬜ pending |
| 02-09.T1 | 02-09 | 6 | DATA-03/04/05 | T-02-50 | CLI surface routed; `data_contrastive` module CREATED here so `cargo check` is satisfiable; `--seed` required proven non-vacuously (checker warning 12) | unit | `cargo check -p apr-cli` + `--seed <SEED>` in usage + no `[default:` on the `--seed` line | ❌ new | ⬜ pending |
| 02-09.T2 | 02-09 | 6 | DATA-03/04/05 | T-02-30/31/48/50 | `run_select`: attested ingest, seed policy, shared atomic-write helper + its three write-safety proofs | unit + fixture | `cargo test -p apr-cli --lib data_contrastive` (fast signal only — the full CLI suite runs once, in T3) | ❌ new | ⬜ pending |
| 02-09.T3 | 02-09 | 6 | DATA-03/04/05 | T-02-29/32/49 | `run_pairs`: strict replay, budget/cap resolution, dump via the shared helper, workflow docs | unit + fixture | `cargo test -p apr-cli --lib data_contrastive` then `cargo test -p apr-cli --lib` (plan exit sweep, 600 s ceiling) | ❌ new | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Commit the in-flight D-06 baseline work currently uncommitted on the branch (data_tweeteval.rs, eval diff, contract, docs example) — **covered by plan 02-01 Task 1 (Wave 1)**
- [ ] Restructure `contracts/tweet-eval-stance-benchmark-v1.yaml` so `pv validate` passes (PROVABILITY-001: add proof_obligations/kani_harnesses per `setfit-encoder-conformance-v1.yaml` shape precedent) — **covered by plan 02-01 Task 2 (Wave 1)**
- [ ] Append phase contracts to the Makefile `$(CONTRACTS)` explicit list so tier3 exercises them — **covered by plans 02-01 Task 2 (tweet-eval) and 02-02 Task 3 (contrastive-pair-protocol)**

*Existing cargo test infrastructure covers unit/property testing; every task above has an `<automated>` command whose prerequisites are created either by Wave-1/2 plans (contract/crate/Makefile) or by the task itself (test files land with the code).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pinned TweetEval source acquisition from network | DATA-01 | Requires network fetch of pinned upstream source; CI stays offline (SAFE-02) | Run `cargo test -p apr-cli --lib data_tweeteval -- --ignored` with network; verify SHA-256 hashes match the manifest and the exclusion record shows exactly the train:70 ≡ validation:3 duplicate (plan 02-06 Task 3 — the real-duplicate golden; Task 2 is the adapter refactor) |

*All other phase behaviors have automated verification.*

---

## Known-Red Gate (phase-exit caveat, NOT a regression)

`pre-release` **Gate 5** — the VERIFYING `cargo package -p apr-cli` — is red from wave 2 through
phase exit, because `apr-cli` gains a dependency on `aprender-contrastive-data`, which is not on
crates.io until the human-approved publish cascade lands it (RESEARCH Pitfall 8 / Finding F5).

- What IS gated and must stay green: `cargo package --no-verify` for BOTH crates (plans 02-02 T1
  and 02-08 T3).
- Exit condition: publish `aprender-contrastive-data` BEFORE `apr-cli`. Human-approved release
  action only — CLAUDE.md forbids self-serving the publish cascade.
- Recorded durably in `must_haves.caveats` of plans 02-02 and 02-08 and in STATE.md
  Blockers/Concerns, so `/gsd:verify-work` reads it as expected state rather than a regression
  (checker warning 11).

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (plans 02-01/02-02)
- [x] No watch-mode flags
- [x] Feedback latency: **per-task inner loop < 120 s for every task.** Three once-per-plan exit
      sweeps exceed it and are declared with their own ceilings in the Sampling Rate section
      (02-08.T3 gate sweep 900 s, 02-08.T3 mutation 2700 s bounded, 02-09.T3 full CLI suite
      600 s). The previous flat "< 120 s" tick was untrue of those three and is corrected here
      rather than left ticked (checker warning 6)
- [x] Fast signal precedes slow: every 02-09 task runs `cargo test -p apr-cli --lib data_contrastive`
      before any full-suite invocation, and only 02-09.T3 runs the full suite
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending execution
