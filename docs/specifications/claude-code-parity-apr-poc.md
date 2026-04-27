# claude-code-parity-apr — POC Specification

**Version**: 0.2.0 (DRAFT)
**Date**: 2026-04-26
**Status**: PROPOSED — companion repo to be scaffolded at M1 as **source of truth**
**Source of truth (post-M1)**: https://github.com/paiml/claude-code-parity-apr (NEW; this aprender-side spec is the seeding blueprint and becomes a redirect once M1 lands)
**Companion-repo invariants** (must be green on every PR — see § Companion-repo source-of-truth invariants):
1. GitHub Actions `ci/gate` green (required status check) → **FALSIFY-CCPA-009**
2. `pmat comply check` 100 % green → **FALSIFY-CCPA-010**
3. Line coverage = 100 % via `cargo llvm-cov --fail-under-lines 100` → **FALSIFY-CCPA-011**
4. `pv validate contracts/claude-code-parity-apr-v1.yaml` exits 0 → **FALSIFY-CCPA-012**
**Contracts (authoritative)**:
- `contracts/claude-code-parity-apr-v1.yaml` — top-level falsifiable parity gates (THIS PR; moves to companion repo on M1)
- `contracts/apr-code-parity-v1.yaml` — sibling: 21-row static feature matrix
- `contracts/apr-claude-proxy-v1.yaml` — sibling: HTTP/SSE Messages-API shape
- `crates/aprender-orchestrate/contracts/batuta/apr-code-v1.yaml` — agent-loop semantics
**arXiv**: 1503.02531, 1807.10453, 2207.11976, 2310.06770, 2505.03096, 2603.23611
**Spec lineage**:
- `docs/specifications/apr-mcp-server-spec.md` § Feature-by-feature parity matrix (static)
- `docs/specifications/apr-cli-qa-spec.md` (template for falsifiable-phase layout)
- memory: `feedback_monorepo_single_source_of_truth` — aprender stays canonical for ML/contract TEXT; companion repo is canonical for **enforcement** (CI, coverage, pmat comply, contract gate)

---

## Problem

Three existing artifacts each cover one axis of `apr code` ↔ Claude Code parity:

| Artifact | Axis | Granularity | Liveness | Falsifiable? |
|----------|------|-------------|----------|--------------|
| `apr-code-parity-v1.yaml` | Feature matrix (21 rows) | Symbol exists in source | Audit-time | ✅ via `cross_check_command` |
| `apr-claude-proxy-v1.yaml` | HTTP/SSE Messages-API shape | Wire format | Per-request | ✅ via 6 FALSIFY-CLAUDE-PROXY gates |
| `apr-code-v1.yaml` (batuta) | Agent loop semantics | Internal turn structure | Internal | ✅ via batuta gates |

**Gap**: none of them tells us, *for a given user prompt, does `apr code` produce the same actions Claude Code produces?* Today we answer this by hand. Three failure modes:

| Failure mode | Asserted-against-by-this-spec gate |
|--------------|------------------------------------|
| Non-falsifiable ("looked the same") | FALSIFY-CCPA-004 + FALSIFY-CCPA-005 |
| Non-reproducible (LLMs are stochastic) | FALSIFY-CCPA-002 (replay determinism) |
| Drift-blind (regressions in `apr code` orchestration ship without flipping any of the three contracts above) | FALSIFY-CCPA-008 (parity-score bound) |

## Goal

Build a small, separate POC repo (`claude-code-parity-apr`) that provides a **record-replay-distill** harness:

> Capture Claude Code as a teacher (record API trace + tool round-trips), feed the same prompts deterministically to `apr code` (replay against mocked LLM responses), then `pv`-validate the diff between the two action streams. The output is a falsifiable **parity score** per fixture and a closed-enum **contract verdict** ({Pass, Drift, Fail}) per gate.

In one phrase: **"distill Claude Code's control plane into `apr code` and prove it byte-stable."**

The "distillation" framing is not metaphorical — see Hinton et al. *Distilling the Knowledge in a Neural Network* (arXiv:1503.02531) and § Distillation framing below.

## Non-Goals

- ❌ Reproducing Claude's *language model output*. The teacher's words don't have to match — we only assert tool-call equivalence (FALSIFY-CCPA-004) and post-state equivalence (FALSIFY-CCPA-005).
- ❌ Replacing `apr-code-parity-v1.yaml`. That static matrix stays — this POC is **runtime** parity, the missing fourth leg.
- ❌ Live network in CI. Every replay reads from a recorded fixture; no `api.anthropic.com` egress in the test path (FALSIFY-CCPA-006).
- ❌ A new agent runtime. We use existing `apr code` unchanged; the POC is observation + diff infra.
- ❌ Modifying Claude Code. We instrument it via the existing `ANTHROPIC_BASE_URL` env var — closed-source binary stays closed.
- ❌ Tolerated coverage gaps. Companion-repo bound is **100 %** line coverage (FALSIFY-CCPA-011), not the aprender 95 % floor — POC is small, no excuse.

## Companion-repo source-of-truth invariants

The companion repo `claude-code-parity-apr` is **source of truth for enforcement**: implementation, fixtures, CI, coverage, pmat-comply config, and the running-binary that consumes the contract. The contract TEXT lives in `aprender/contracts/` per the monorepo single-source-of-truth policy (`feedback_monorepo_single_source_of_truth.md`); the companion repo pins it by commit hash and gates every PR against it.

Four invariants, **online from M0** (before any actual parity work), enforced as required GitHub Actions status checks on the companion repo's `main`:

### Invariant 1 — `ci/gate` green (FALSIFY-CCPA-009)

- **Mechanism**: GitHub branch protection on `main` requires status check `ci/gate` (matches aprender's policy in `CLAUDE.md` § "CRITICAL GIT WORKFLOW RULES"). Direct pushes to `main` blocked.
- **Falsifiable claim**: `gh api repos/paiml/claude-code-parity-apr/branches/main/protection \| jq '.required_status_checks.contexts | index("ci/gate") != null'` returns `true`.
- **Failure mode caught**: a bypassed merge that ships a red build.

### Invariant 2 — `pmat comply check` 100 % (FALSIFY-CCPA-010)

- **Mechanism**: CI step `pmat comply check --strict` runs on every PR; exit code 0 with zero violations is required.
- **Falsifiable claim**: `pmat comply check --json | jq '.total_violations == 0 and .compliance_pct == 100.0'` returns `true`.
- **Why 100 %**: the POC is small (~5 Rust crates) and greenfield. There is no legacy excuse.
- **Academic basis**: prior-art compliance auditing literature (cf. arXiv:2102.05351 on coverage-completeness invariants).

### Invariant 3 — Line coverage 100 % (FALSIFY-CCPA-011)

- **Mechanism**: CI step `cargo llvm-cov --workspace --fail-under-lines 100 --fail-uncovered-lines 0`. (Branch coverage tracked but not gated until M6.)
- **Falsifiable claim**: `cargo llvm-cov report --json | jq '.data[0].totals.lines.percent == 100.0'` returns `true`.
- **Tarpaulin policy**: `cargo tarpaulin` is forbidden per aprender CLAUDE.md ("slow, unreliable, causes hangs"); only `cargo llvm-cov`.
- **Academic basis**: coverage-guided test-adequacy literature; specifically the test-adequacy/MC-DC line of work (arXiv:2102.05351 and earlier).

### Invariant 4 — `pv validate` on every commit (FALSIFY-CCPA-012)

- **Mechanism**: pre-commit hook (`.git/hooks/pre-commit` installed by `make hooks`) AND CI step both run `pv validate contracts/claude-code-parity-apr-v1.yaml`. Exit 0 required.
- **Falsifiable claim**: in CI logs, the line `pv validate contracts/claude-code-parity-apr-v1.yaml` is followed by exit code 0 within the same job.
- **Bash workarounds rejected** per `feedback_pv_not_bash_for_contracts.md` and CLAUDE.md § "Contract Validation: DOGFOOD pv, NEVER bash" — if `pv` rejects the contract, fix the contract or extend `aprender-contracts/src/schema/`, never bypass with shell.

These four invariants are necessary preconditions for *any* of the parity gates (FALSIFY-CCPA-001..008) to be trustworthy. A red CI on the companion repo invalidates every claim downstream.

## Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │  Phase 1: RECORD (teacher demonstrations)        │
  user prompt ─────►│   Claude Code ──HTTP──► ccpa-recorder ──► fixture│
                    │                  ▲             │                 │
                    │                  └──── api.anthropic.com (live)  │
                    └──────────────────────────────────────────────────┘
                                        │
                                        ▼ (one .ccpa-trace.jsonl file)
                    ┌──────────────────────────────────────────────────┐
                    │  Phase 2: REPLAY (student under test)            │
                    │   fixture ──► ccpa-replayer ──► apr code         │
                    │                  ▲                  │            │
                    │                  └─ mocked LLM ◄────┘            │
                    │     (returns the recorded assistant turn)        │
                    └──────────────────────────────────────────────────┘
                                        │
                                        ▼ (one .ccpa-trace.jsonl file)
                    ┌──────────────────────────────────────────────────┐
                    │  Phase 3: DISTILL+DIFF (parity verdict)          │
                    │   teacher.jsonl + student.jsonl ──► ccpa-differ  │
                    │      ──► pv validate ──► verdict + drift report  │
                    └──────────────────────────────────────────────────┘
```

### Why a recording HTTP proxy is sufficient (asserted by FALSIFY-CCPA-001)

Claude Code is closed-source, so we can't hook its tool execution directly. We don't need to: every tool round-trip already round-trips through the Anthropic API. Claude Code submits `tool_result` blocks back to Anthropic on the next request, so a recording HTTPS proxy at `ANTHROPIC_BASE_URL` captures the full action stream — prompts, tool calls, tool outputs, final messages — without any CLI wrapping. **The proxy is the recorder.** This claim is mechanically falsifiable: FALSIFY-CCPA-001 asserts every committed fixture is a complete, schema-valid action trace; failure (truncation, missing tool round-trip) flips the gate.

### Why `apr code` replay needs LLM mocking (asserted by FALSIFY-CCPA-002 + FALSIFY-CCPA-003)

The student is `apr code`, which talks to a *local* model. If we feed the same prompt to `apr code`, the local model produces *its own* tool calls, which may differ purely due to model quality, not orchestration. To isolate orchestration drift from model drift, the replayer:

1. Plays the user prompts back to `apr code` in order.
2. Intercepts `apr code`'s LLM calls and **returns the recorded teacher's assistant turn verbatim** via `RecordedDriver: LlmDriver`.
3. Lets `apr code`'s orchestration (tool dispatch, permission gates, hook firing, memory loading, etc.) execute against that fixed teacher output.
4. Captures `apr code`'s emitted tool calls + final state.

Drift now has a single source: orchestration. Falsifiable via two gates:
- **FALSIFY-CCPA-002** (replay determinism): re-running the same fixture twice with the same `apr code` revision yields byte-identical student traces (after `<SESSION>`/`<TS>`/`<TOOL-N>` normalization).
- **FALSIFY-CCPA-003** (mock completeness): `RecordedDriver` consumes every teacher turn exactly once; missing turn → panic, extra turn → assertion fail.

This is structurally identical to the existing `MockDriver` used by `crates/aprender-orchestrate/src/agent/task_tool/tests.rs`, so no new abstraction is needed — `RecordedDriver` is a ~100-line file. **Academic basis**: behavioral cloning / imitation-learning evaluation methodology (cf. agent-task literature in arXiv:2310.06770 SWE-bench).

## New repo layout (`claude-code-parity-apr`)

```
claude-code-parity-apr/                 # source of truth for code, fixtures, CI
├── Cargo.toml                          # workspace root
├── README.md
├── .github/workflows/
│   └── ci.yml                          # ci/gate job (FALSIFY-CCPA-009 enforces required check)
├── .pmat-comply.toml                   # FALSIFY-CCPA-010 config, 100 % strict
├── Makefile                            # `make tier3` runs all 12 gates locally
├── crates/
│   ├── ccpa-trace/                     # serde schema for .ccpa-trace.jsonl
│   ├── ccpa-recorder/                  # mitm-style HTTPS proxy at ANTHROPIC_BASE_URL
│   ├── ccpa-replayer/                  # drives `apr code` with mocked LLM responses
│   ├── ccpa-differ/                    # semantic diff + parity score
│   └── ccpa-cli/                       # `ccpa record|replay|diff|report`  (binary: `ccpa`)
├── contracts/
│   ├── claude-code-parity-apr-v1.yaml  # SOURCE-OF-TRUTH copy after M1; pre-M1 lives in aprender/contracts/
│   └── pin.lock                        # pinned commit-hash of authoritative aprender contract
├── fixtures/                           # checked-in recorded sessions
│   ├── 0001-edit-readme.ccpa-trace.jsonl
│   ├── 0002-fix-failing-test.ccpa-trace.jsonl
│   └── ...
└── docs/
    └── architecture.md                 # links back to this seeding spec
```

**Naming**: `ccpa` = claude-code-parity-apr. Binary name is `ccpa`. Repo name `claude-code-parity-apr` is explicit on GitHub.

**Monorepo policy compliance** (`feedback_monorepo_single_source_of_truth.md`): aprender stays canonical for the *contract text*; the companion repo is canonical for *enforcement* (CI, coverage, pmat-comply, contract gate). `pin.lock` records the commit hash of the authoritative `aprender` contract being consumed; `pv` is invoked as a binary, not as a re-implemented schema.

## Trace schema (.ccpa-trace.jsonl)

One JSON object per line. Schema lives in `ccpa-trace` crate and is contracted by `contracts/claude-code-parity-apr-v1.yaml § trace_schema`. **FALSIFY-CCPA-001** asserts roundtrip-equality.

```jsonc
{"v": 1, "kind": "session_start", "session_id": "uuidv7", "ts": "2026-04-26T01:23:45Z",
 "actor": "claude-code|apr-code", "model": "claude-sonnet-4-6|qwen3-coder-30b-a3b-q4km",
 "cwd_sha256": "<git-tree-hash>"}

{"v": 1, "kind": "user_prompt", "turn": 0, "text": "fix the failing test"}

{"v": 1, "kind": "assistant_turn", "turn": 1,
 "blocks": [
   {"type": "thinking", "thinking": "..."},
   {"type": "text", "text": "I'll start by..."},
   {"type": "tool_use", "id": "toolu_01...", "name": "Bash",
    "input": {"command": "cargo test --lib"}}
 ],
 "stop_reason": "tool_use"}

{"v": 1, "kind": "tool_result", "turn": 2, "tool_use_id": "toolu_01...",
 "ok": false, "content": "test failed: ...",
 "side_effects": {"files_read": [], "files_written": [], "exit_code": 101}}

{"v": 1, "kind": "session_end", "turn": 7, "stop_reason": "end_turn",
 "elapsed_ms": 12340, "tokens_in": 4521, "tokens_out": 891}
```

**Determinism guarantees** (asserted by FALSIFY-CCPA-002):
- `session_id` and `ts` are *replaced* with stable placeholders during diff (`<SESSION>` / `<TS>`).
- `tool_use.id` is *normalized* to `<TOOL-N>` per turn.
- `cwd_sha256` is asserted equal between teacher and student fixtures (same starting state).

## Mocking strategy (asserted by FALSIFY-CCPA-002 + FALSIFY-CCPA-003)

The student-side LLM mock implements `apr code`'s `LlmDriver` trait (already abstract — see `crates/aprender-orchestrate/src/agent/code.rs`). The replayer constructs a `RecordedDriver { trace: Vec<AssistantTurn> }` that returns turn N when called for the Nth time. Mismatches (`apr code` calls the LLM at an unexpected point) are recorded as `OrchestrationDrift::ExtraneousLlmCall` and surfaced in the diff.

Pre-requisite: `LlmDriver` must be `pub` from `aprender-orchestrate` (today it is `pub(crate)` behind no feature flag). Tracked as **PMAT-CODE-LLM-DRIVER-PUBLIC-001**, blocking M3.

## Distillation framing

Calling this "distillation" is not metaphorical. The framing maps directly onto Hinton et al. (arXiv:1503.02531):

| Hinton 2015 | This POC |
|-------------|----------|
| Teacher network | Claude Code (closed-source orchestration over claude-sonnet-4.6) |
| Student network | `apr code` (open orchestration over Qwen3-Coder-30B-A3B-Q4_K_M via realizar) |
| Soft target distribution | Recorded action stream (the assistant turns + tool round-trips) |
| Demonstration corpus | `fixtures/*.ccpa-trace.jsonl` (≥30 sessions covering 17/21 parity rows) |
| Loss function | `1 − parity_score`, where `parity_score = matched_actions / total_actions` |
| Optimizer step | File-by-file PR review of `apr code` orchestration, driven by drift report |
| Convergence criterion | Aggregate `parity_score ≥ 0.95` (FALSIFY-CCPA-008) |

The teacher's *fixtures* are immutable per-revision; the student (`apr code` orchestration code) is what changes. **Academic basis**: knowledge-distillation framing per arXiv:1503.02531 with the action stream substituted for the logit distribution; metamorphic relations on tool calls per arXiv:1807.10453 (METTLE) and arXiv:2603.23611 (LLMORPH); differential testing of the two orchestrators per arXiv:2207.11976.

## Phases / Milestones

> **Status snapshot (2026-04-27)**: M0–M19 all SHIPPED; contract at
> `claude-code-parity-apr-v1` v1.7.0 ACTIVE_RUNTIME; corpus at **30**
> paired canonical fixtures (spec ≥30 target met) with parity-matrix
> coverage 15/15 reachable (2 OOS at trace boundary); FALSIFY-CCPA-007
> HARD-BLOCKING CI gate live since M16. All four non-Error
> `StopReason` variants exercised. Live PR cadence on
> https://github.com/paiml/claude-code-parity-apr.

### Major phases (M0–M6)

| Phase | Deliverable | Source-of-truth gates online (cumulative) | Status |
|-------|-------------|-------------------------------------------|--------|
| **M0** | Spec + top-level contract `claude-code-parity-apr-v1.yaml` (DRAFT). Companion-repo invariants 009–012 online from the empty-scaffold PR forward. | 009, 010, 011, 012 | **DONE** (PRs #1, #2) |
| **M1** | Companion repo scaffold (empty crates that compile + 100 % line cov), `ccpa-trace` crate, schema-roundtrip test. Spec + contract relocate from aprender to companion repo as canonical. | + 001 | **DONE** (PRs #3, #4) |
| **M2** | `ccpa record` (Anthropic Messages-API parser → trace records). Note: M2.3 HTTPS proxy is OOS post-rescope ("we will not call api, we will assume claude code"). | + (still 001) | **DONE** (PRs #5, #6) at parser-only scope |
| **M3** | `ccpa replay` (LlmDriver trait + RecordedDriver) — algorithm-level. Real `apr code` LlmDriver adapter pending PMAT-CODE-LLM-DRIVER-PUBLIC-001 in upstream aprender. | + 002, 003 | **DONE** (PRs #7, #8) at algorithm scope |
| **M4** | `ccpa diff` semantic differ — per-tool equivalence rules, file-mutation snapshots, parity score. | + 004, 005 | **DONE** (PRs #9, #10) |
| **M5** | Sovereignty gate (no `api.anthropic.com` on replay) + corpus growth + parity-matrix coverage walk. | + 006, 007 | **DONE** (PRs #11, #13–#21) |
| **M6** | Promote contract DRAFT → ACTIVE; integrate into `make tier3` and `pv lint`; close epic. | + 008 | **DONE** (PR #12) |

### Sub-milestones (M11+)

The corpus and gate work continued past M6 with sub-milestones tracked
in `contracts/claude-code-parity-apr-v1.yaml § status_history`:

| Sub | Deliverable | Outcome | PR |
|-----|-------------|---------|----|
| **M11** | First runtime measured_parity over 5 paired canonical fixtures; contract DRAFT → ACTIVE_RUNTIME (FALSIFY-CCPA-013 discharged) | aggregate_score 1.0000 over 5 fixtures, parity-matrix coverage 1/17 | #13 |
| **M12** | Corpus 5 → 8; coverage 1/17 → 4/17 (subagent-spawn, mcp-client, slash-commands added) | 1.0000 / 8 fixtures | #14 |
| **M13** | Corpus 8 → 11; coverage 4/17 → 7/17 (claude-md-memory, permission-modes, builtin-tools-web added) | 1.0000 / 11 fixtures | #17 |
| **M13.5** | Bidirectional sensitivity: regression corpus added; meter must FAIL on deliberate drift | regression corpus aggregate=0.5, exits 1 (drift detected) | #14 |
| **M14** | Corpus 14 → 17; coverage 10/17 → 13/17 (worktree-isolation, configuration-ladder, managed-org-policy added) | 1.0000 / 17 fixtures | #19 |
| **M15** | Trace schema v1 → v2 (additive `HookEvent` + `SkillInvocation` record kinds); differ extension (7 new DriftCategory variants); coverage 13/17 → 15/17 | 1.0000 / 19 fixtures; contract v1.2.0 → v1.3.0 | #20 |
| **M16** | FALSIFY-CCPA-007 informational → HARD-BLOCKING; OOS exclusion mechanism (`--oos-rows`) shipped for `keyboard-shortcuts` + `status-line` | 15/15 reachable, gate PASS; contract v1.3.0 → v1.4.0 | #21 |
| **M17** | Spec milestone table refreshed to reflect M0–M16; contract v1.4.0 → v1.5.0 | doc-only | #22 |
| **M18** | Corpus depth 19 → 24; 5 schema-v2 surface variants (Bash multiline, Edit replace_all, HookDecision::Block, SkillSource::UserInvoked, StopReason::MaxTokens) | 1.0000 / 24 fixtures; contract v1.5.0 → v1.6.0 | #23 |
| **M19** | Corpus complete 24 → 30 (spec ≥30 target met); multi-tool sequences + multi-turn correction + StopReason::StopSequence | 1.0000 / 30 fixtures; contract v1.6.0 → v1.7.0 | this PR |

## Falsification conditions (12 gates total)

### Source-of-truth invariants (M0+)

| ID | Name | Phase | Mechanically asserted by |
|----|------|-------|--------------------------|
| FALSIFY-CCPA-009 | ci_main_branch_green | M0+ | `gh api repos/paiml/claude-code-parity-apr/branches/main/protection` returns `ci/gate` ∈ required contexts |
| FALSIFY-CCPA-010 | pmat_comply_100pct | M0+ | `pmat comply check --json` returns `compliance_pct == 100.0 ∧ total_violations == 0` |
| FALSIFY-CCPA-011 | line_coverage_100pct | M0+ | `cargo llvm-cov --fail-under-lines 100 --fail-uncovered-lines 0` exits 0 |
| FALSIFY-CCPA-012 | pv_contract_gate_on_commit | M0+ | pre-commit hook + CI both run `pv validate contracts/claude-code-parity-apr-v1.yaml`, exit 0 |

### Behavioral parity gates (M1..M6)

| ID | Name | Phase | Assertion summary |
|----|------|-------|-------------------|
| FALSIFY-CCPA-001 | trace_schema_roundtrip | M1 | every fixture parses, re-serializes byte-identical, validates against `trace_schema` |
| FALSIFY-CCPA-002 | replay_determinism | M3 | replaying same fixture twice → byte-identical student traces (after normalization) |
| FALSIFY-CCPA-003 | mock_completeness | M3 | `RecordedDriver` consumes exactly len(teacher.assistant_turns) responses; no missing, no extras |
| FALSIFY-CCPA-004 | tool_call_equivalence | M4 | per turn, multiset of `(tool_name, semantic_input)` pairs in student matches teacher under per-tool equivalence rules (Edit: post-state sha256; Bash: normalized command; etc.) |
| FALSIFY-CCPA-005 | file_mutation_equivalence | M4 | union diff over CWD after `apr code` finishes equals union diff after Claude Code finished, modulo per-filetype canonicalization |
| FALSIFY-CCPA-006 | sovereignty_on_replay | M5 | zero outbound sockets to `*.anthropic.com` during replay; CI test container drops all egress except 127.0.0.1 |
| FALSIFY-CCPA-007 | corpus_coverage | M5 | ≥1 fixture per non-MISSING row of `apr-code-parity-v1.yaml` (currently 17 of 21) |
| FALSIFY-CCPA-008 | parity_score_bound | M6 | aggregate `parity_score ≥ 0.95` and per-fixture `≥ 0.80` |

Each gate maps to one falsification test in `crates/ccpa-*/tests/falsify_ccpa_NNN_*.rs` and is enforced via `pv validate contracts/claude-code-parity-apr-v1.yaml` per the harness policy in `CLAUDE.md § Contract Validation: DOGFOOD pv, NEVER bash`. **No bash/yq/python re-implementation of these gates is permitted.** If `pv validate` does not yet support a needed shape, extend `aprender-contracts/src/schema/` — schema-extension ticket: PMAT-CONTRACTS-CCPA-001.

## Academic basis (arXiv → gate mapping)

| arXiv | Title (short) | Applies to gate(s) | Why |
|-------|---------------|--------------------|-----|
| **1503.02531** | Hinton et al., *Distilling the Knowledge in a Neural Network* | CCPA-008 | The parity_score *is* a distillation loss on the action stream. Framing gates the convergence criterion. |
| **1807.10453** | Segura et al., METTLE — *Metamorphic Testing of ML Systems* | CCPA-004, CCPA-005 | Tool-call equivalence and post-state equivalence are metamorphic relations under the operational semantics of each tool — the canonical way to test ML systems without ground truth. |
| **2207.11976** | *Differential Testing of DL Frameworks* | CCPA-002, CCPA-004 | Teacher↔student parity is the textbook differential-testing setup; replay determinism is the precondition. |
| **2310.06770** | Jimenez et al., *SWE-bench* | CCPA-007 | Justifies fixture-corpus methodology: ≥1 task per capability row, recorded once, replayed forever, no live network. |
| **2505.03096** | *Chaos Engineering for LLM Systems* | CCPA-006 | Sovereignty enforcement under fault injection — "what if the env leaks an Anthropic key on a replay run" is exactly a chaos-test class. |
| **2603.23611** | LLMORPH — *Cataloged Metamorphic Relations for NLP* | CCPA-004 | 191 catalogued metamorphic relations directly populate per-tool `equality` rules in `tool_equivalence_rules`. |
| **2102.05351** (referenced in apr-cli-qa-spec) | Coverage-completeness invariants | CCPA-010, CCPA-011 | Background for 100 %-coverage / 100 %-comply invariants. |

## Falsification run history

| Run | Date | Revision | Verdict | Notes |
|-----|------|----------|---------|-------|
| Run 0 | 2026-04-26 | this PR (`feat/claude-code-parity-apr-poc-spec`) | **NOT YET RUN** | Spec authored; companion repo not yet scaffolded; gates 009–012 will run on M1's empty-scaffold PR. |

(Subsequent runs append below in the apr-cli-qa-spec.md format: gate / status / evidence per row.)

## Risks & open questions

| # | Risk / question | Mitigation | Falsifiable by |
|---|-----------------|------------|----------------|
| R1 | Recording the live Anthropic API costs $$ per fixture | Cap corpus at ~30 short fixtures; record once, replay forever | Fixture-count ≤ budget asserted by CI step `wc -l fixtures/*.jsonl` ≤ 30 000 lines |
| R2 | Claude Code may pin its own Anthropic auth, refuse `ANTHROPIC_BASE_URL` override | Verify pre-M2 with a hello-world prompt; fallback rejected (PCAP capture is fragile) | M2 PR includes one-liner CI step exercising override; failure blocks M2 |
| R3 | Tool-call equivalence for Edit/Write is non-trivial | Per-tool equivalence rules in `ccpa-differ`, contracted in YAML | FALSIFY-CCPA-004 — directly |
| R4 | Claude Code roadmap may add tools we don't have in `apr code` | New tools surface as `OrchestrationDrift::UnknownToolName` | FALSIFY-CCPA-004 — directly (gate FAILs until `apr-code-parity-v1.yaml` flips a row) |
| R5 | New repo conflicts with monorepo single-source-of-truth | Companion repo is canonical for *enforcement*; aprender stays canonical for *contract text*. `pin.lock` pins authoritative commit hash | FALSIFY-CCPA-012 — pre-commit hook rejects stale pins |
| R6 | `apr code`'s `LlmDriver` trait may not be public-stable enough for an external repo | PMAT-CODE-LLM-DRIVER-PUBLIC-001 (pre-req for M3) | M3 PR exists ⇔ PMAT-CODE-LLM-DRIVER-PUBLIC-001 closed |
| R7 | 100 % line coverage may produce test-for-coverage's-sake noise on a tiny POC | Tradeoff accepted: POC is small (~5 crates), 100 % is achievable. If a function genuinely cannot be covered, the function is unjustified — delete it. | FALSIFY-CCPA-011 — directly |
| R8 | `pmat comply check --strict` may reject patterns aprender itself uses | Companion repo is greenfield; we author to comply. If we hit a genuine `pmat comply` bug, the fix is upstream pmat, not a `--allow` flag | FALSIFY-CCPA-010 — directly |

## References

- `contracts/claude-code-parity-apr-v1.yaml` — top-level falsifiable parity contract (this PR)
- `contracts/apr-code-parity-v1.yaml` — sibling static-feature matrix
- `contracts/apr-claude-proxy-v1.yaml` — sibling Messages-API shape contract
- `crates/aprender-orchestrate/contracts/batuta/apr-code-v1.yaml` — agent-loop ground truth
- `docs/specifications/apr-mcp-server-spec.md` — feature-by-feature parity matrix prose
- `docs/specifications/apr-cli-qa-spec.md` — template for falsification-phase + arXiv layout
- `CLAUDE.md` § "Contract Validation: DOGFOOD pv, NEVER bash" — harness policy
- `CLAUDE.md` § "Realizar-First Architecture" — why student is `apr code`, not direct `aprender::models`
- Memory: `feedback_monorepo_single_source_of_truth.md` — aprender vs companion-repo split
- Memory: `feedback_pv_not_bash_for_contracts.md` — every gate flows through `pv`
- Memory: `project_apr_code_parity_matrix.md` — the static-matrix epic this POC complements
- Anthropic Messages API — https://docs.anthropic.com/en/api/messages
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Segura, S., Towey, D., Zhou, Z., & Chen, T. (2018). METTLE — *Metamorphic Testing of Deep Learning Systems.* arXiv:1807.10453
- *Differential Testing of Deep Learning Frameworks.* arXiv:2207.11976
- Jimenez, C. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770
- *Chaos Engineering for LLM Systems.* arXiv:2505.03096
- LLMORPH — *Cataloged Metamorphic Relations for NLP.* arXiv:2603.23611
