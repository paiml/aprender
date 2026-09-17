# IMPL receipt — PMAT-3429 — PP-QUANT-001 M3: regression fixtures on tiny synthetic GGUFs

## Identity
| field | value |
|---|---|
| ticket | PMAT-3429 (GitHub #3429), kind=code |
| branch | `PMAT-3429-regression-fixtures`, cut from `origin/main` @ `dc6fb687c` (the #3448 merge) |
| discover.json | sha256 prefix `2fd044d8ec5086af`; `gate_cmd=make gate`, `gate_cmd_fallback=false`, `required_check=ci / gate,workspace-test`, `contracts_dir=contracts` |
| orchestrator model | admitted `opus-5` by `model-gate.sh` (basis=file). The session opened on `fable-5-1`, which the gate refused (exit 1: the ticket carries no `orch:fable`); the operator relaunched on Opus. `phase-boundary.sh` printed `change=none` at every boundary |
| worktree | a separate worktree off `origin/main`. The shared checkout was on another branch with 1,700 staged paths and was left untouched |
| status-line join | `[U]` not measured this run (statusLine `session_id` = hook `session_id`, `tasks[].id` = `agent_id`, `transcript_path` on subagentStatusLine stdin) — gap G-5 |

## Premise corrections (measured before planning)
1. The issue says "no fixture in the tree reproduces any of them". **False for #3341**: `moe_load_contract_refuses_q4_0_experts` already builds a synthetic Q4_0-expert GGUF and goes RED under the inverse of 102a154c9. All three plan lanes found the same thing independently. It is cited, not duplicated.
2. #1789 already had 7 unit tests on `validate_matmul_weight_shape`, but they call the guard **directly**, so deleting its call site leaves all 7 GREEN. The discriminating surface is the serve route (Option B), not the guard.
3. The plan's #3091 fixture as first written had an IQ2_XXS payload sized for `token_embd` but attached to `ffn_down`. It passed the pin, but only because `tensor_byte_size` refused it first. The R5 mutation (adding an IQ2_XXS byte-size arm) exposed it: "data range exceeds file size". The payload was resized and re-pinned. Under the same mutation the fixture now LOADS, so #3432 can invert the pin.
4. The first #1749 test asserted `run(...)` is Ok. That passes only above spec H12's **10 tok/s wall-clock floor**. It was rewritten to assert the generated token count on `run_realizar_benchmark`, removing the timing dependency.

## Plan, routing, triggers
| phase | work | trigger | `route.sh` printed | dispatched | acceptance |
|---|---|---|---|---|---|
| P1 | revert-probe matrix on EXISTING tests + plan grill | Q1 (\|M\|=6), Q2 | `route=agy-grillme w=1.00 basis=absent effort=1[U]` | delegate grillme w3 ∥ sonnet worker (probe worktree) | `jq` shape check on `matrix.json` + probe tree clean → exit 0 |
| P2 | serve fixtures: #2535, #3091 pin, `add_raw_tensor`, sha256 pins | — | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true` | delegate goal w1 writes=true | `cargo test -p aprender-serve --lib -- gguf::regression_fixtures moe_load_contract` exit 0 + R3/R4/R5 RED |
| P3 | #1789 MoE serve route | — | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true` | sonnet worker (R-4 fallback, see ledger) | `… -- regression_1789` exit 0 + R2b RED |
| P4 | #1749 apr-cli bench route + real-file exit script | — | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true` | sonnet worker (one-writer rule: P2's lane held the writer slot) | `cargo test -p apr-cli --lib -- regression_1749` exit 0 + R1 RED; `bashrs lint` exit 0 |
| P5 | contracts, final matrix on HEAD, pre-PR review, gate | Phase-4 pre-PR | `route=self` (orchestration) + review quorum | direct + delegate quorum w3 | see verification table |

Quorum decisions adopted (plan grill, `agreed=false`: lanes 1 and 2 `do-not-implement-as-written`, lane 3 `PASS`; all gemini, author claude):
- Both blocking premises were amended into the plan: `GGUFBuilder` gains `add_raw_tensor` and reuses `GgmlQuantType::IQ2XXS` (no parallel const); #3341 is cited.
- #1789 tests the serve route (lanes 2 and 3 measured `try_qwen3_moe_backend` is not cuda-gated).
- Q-a: sha256 of builder bytes, no committed `.gguf` (`.gitignore:47` is `*.gguf`).
- Q-b: characterization pin for #3091 — lanes 1 and 3 for, lane 2 against (dissent recorded: "asserts the defect"; mitigated by the pin's message naming #3432 and by the R5 proof that the fixture loads once the arm exists).
- Q-c: all three lanes rejected duplicating the builder, and each proposed a different fix: `#[path]` (does not compile, since `super::types` imports; same shape as the #3425 clean-room red), a new crate (out of scope), or a test-only feature. **Decided:** #1749 uses apr-cli's own public writer `export_tensors_to_gguf` with its own tensor table and sha256 pin. The builder is not copied; the table duplication is recorded as G-3. The pre-PR review (2 PASS lanes) called the duplication acceptable.

## Dispatch ledger
| # | description | executor / model | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| 1 | `PMAT-3429/ph1.delegate` grillme w3 | paiml-agy-delegate / opus | 30 + resume | yes | once | lanes ran; receipt on resume. agy conversations `bb3578d4…`, `69d182c0…`, `71ce0020…` (child_conversations=3) |
| 2 | `PMAT-3429/ph1.probe` worker B | paiml-impl-worker / sonnet | 40 + 40 | yes ×2 | once | R2a/R3/R4 measured, R1/R2b/R5 and `matrix.json` finished by the orchestrator |
| 3 | `PMAT-3429/ph2.delegate` goal w1 writes=true | paiml-agy-delegate / opus → agy gemini-3.1-pro-high | 15 | no | no | **agy-lane exit 3** (see jidoka J-1); edits left uncommitted in the kept lane worktree; `outcome=achieved` contradicted by the tree. The orchestrator salvaged the diff and re-verified it. Conversation `d0063bfa…` |
| 4 | `PMAT-3429/ph4.cli` worker B | paiml-impl-worker / sonnet | 40 + 40 | yes ×2 | once | test and script written; the orchestrator removed the wall-clock dependency, the deleted logs and the bogus suppressions, then ran the mutation |
| 5 | `PMAT-3429/ph3.serve` worker B | paiml-impl-worker / sonnet | 40 | yes | no | `partial=true`: the fixture lacked `qwen3moe.expert_feed_forward_length` (out of the worker's scope). The orchestrator fixed the builder and re-pinned |
| 6 | `PMAT-3429/ph5.delegate` review w3 | paiml-agy-delegate / opus | 16 | no | no | lanes 1 and 3 PASS, lane 2 **BLIND** (void); all three exit 3 from a foreign `git fetch` poller (J-2). Conversations `67fcff36…`, `bfc65914…`, `faa50254…` |

R-4 fallbacks, named: P3 and P4 went to `sonnet-worker`. P4's reason is the one-writer rule (P2's goal lane was live). P3's reason is that P2's writes lane had just been voided by an environment exit 3 that recurred in every later lane (J-1, J-2).

## Slots, denials, I-3
`transcript-gate.sh` (run from the session's project dir): `PASS attempted=9 denied=0 stalled=0 running_peak=2 slots=3 (agent_calls=6 resumes=3 workflow_started=0)`. The same script run from the worktree cwd reported `attempted=0`, a vacuous PASS against the wrong project directory (G-6). Hook events: `SubagentStart` 9, `SubagentStop` 4, `reap` 6, denials 0.

## Revert matrix — the ticket's falsifier, re-run by the orchestrator on HEAD `8011cc963`
Each row applies the inverse of the fix's behavioural hunk (whole-commit reverts do not apply; the files moved) in a separate worktree, runs the fixtures, and restores.

| row | issue | fix | inverse applied | test that goes RED | exit | message names |
|---|---|---|---|---|---|---|
| R1 | #1749 | 541c2477d | delete the `is_moe_gguf` dispatch in `benchmark.rs` | `regression_1749_moe_gguf_bench_takes_the_moe_route` (generated 0 tokens, want 1) | 101 | #1749 (measured at `f0fca3811`; apr-cli test code unchanged since except a comment) |
| R2b | #1789 | Option B (after 9c974524f) | skip `try_qwen3_moe_backend` at the chat handler | `regression_1789_moe_gguf_chat_completions_takes_the_moe_route` (HTTP 500, #1790 empty-buffer refusal) | 101 | #1789 |
| R3 | #2535 | 33e881e5b | `validate_quantized_tensors` always checks dense FFN | `regression_2535_complete_moe_passes_validate_quantized_tensors` (+ the #1789 route, which loads the same model) | 101 | #2535 |
| R4 | #3341 | 102a154c9 | skip `validate_moe_layer_tensors` at load | pre-existing `moe_load_contract_refuses_q4_0_experts` | 101 | #3341 |
| R5 | #3091 | unfixed (#3432) | add an IQ2_XXS arm to `tensor_byte_size` | `regression_3091_iq2_xxs_in_shared_base_refused_until_3432` (the file now loads) | 101 | #3091 |
| R2a | #1789 | 2a9520579 | delete the `validate_matmul_weight_shape` call site | none of the 7 existing guard tests (they call the fn directly) — superseded by R2b | 0 | — |

No new fixture stayed GREEN under its revert, so none was deleted.

## Verification table (claimed vs re-run)
| check | claimed by | claim | orchestrator re-run |
|---|---|---|---|
| serve fixtures + contract tests | ph2 lane | achieved (no exit codes) | exit 0, 13 passed (HEAD `8011cc963`) |
| #2535 mutation | ph2 lane | "mutated to `false`" (not the brief's inverse) | R3 exit 101 naming #2535 |
| #1749 test | ph4 worker | 3 passed | exit 0 after the rewrite; R1 exit 101 |
| #1789 test | ph3 worker | exit 101 (partial) | exit 0 after the builder fix; R2b exit 101 |
| plan lanes | lane 2 | "measured 9 passed" | not relied on; R4 re-run exit 101 |
| review lanes | lane 3 | "removing try_qwen3_moe_backend causes HTTP 500" | R2b measured: 500 |
| `cargo fmt --all -- --check` | — | — | exit 0 |
| `cargo clippy -p aprender-serve --lib -- -D warnings` | — | — | exit 0 |
| `cargo clippy -p {aprender-serve,apr-cli} --lib --tests -- -D warnings` | — | — | exit 101 on main-carried debt; 0 diagnostics located in files this branch adds or edits |
| `cargo test -p aprender-contracts --lib` | — | — | exit 0, 1538 passed |
| `cargo deny check advisories` | — | — | exit 0 |
| `pv validate` + `pv lint` on both edited contracts | — | — | exit 0 / 0 errors (identical to origin/main's copies) |
| `bashrs lint scripts/pp_quant_exit_fixtures.sh` | worker | clean | exit 0, 0 warnings (two documented false-positive suppressions) |
| `check_apr_bin_pinned.sh` | — | — | exit 0; with a bare `apr run` appended to the script → exit 1 (the guard scans it) |
| `pp_quant_exit_fixtures.sh --verify-only` (empty cache) | — | — | exit 1, both `(absent) FAIL` as documented |
| `make gate` | — | — | see Gate below |
| `kind-gate.sh` (DoD) | — | — | exit 0, kind=code |

## Gate
`make gate` at HEAD `4ecf043a7`, comparand `origin/main@d79a6875b`: exit 2 (make), classified **environment, not code**:
- `pmat verify --skip satd --skip tests`: `ok=null stages_measured=2` (format, complexity, clippy reported with no status) — not a PASS; recorded as measured-but-unjudged.
- `scripts/guard_tree.sh --no-cargo`: 2 FAIL, both `tool_version` instrument mismatches. `scripts/cb200_baseline.txt` and `scripts/complexity_baseline.txt` were recorded under pmat 3.40.1, the fleet's declared version; this box runs 3.40.2. The complexity measurement itself is 673 → 673 functions over threshold on base and merge. Every other baseline "did not grow". Every other guard PASS or skipped as wired elsewhere.
- `scripts/gate_touched_crates.sh`, run directly because make stops at the first failing line: touched `apr-cli aprender-serve`; Cargo.lock touched → `cargo check --workspace --tests` exit 0.
The fleet's `ci / gate` on PR #3456 is the judge of record for the instrument rows.

## Jidoka log
- **J-1** (ph2 goal lane, exit 3). Defect: `agy-lane` isolation assertion on the SHARED `.git/config` and refs. Owner: the harness plus the host's shared object store. Whys: (1) `.git/config` hash changed mid-lane → (2) mtime equals a `git push -u origin PMAT-3445-t0-milestone-gate` from a sibling worktree in the same second → (3) all ~355 worktrees share one `.git` → (4) `agy-lane.sh` asserts repo-wide config and refs, not the lane's own → (5) concurrent sessions on one host are routine here. No code defect in this ticket; the lane's work was salvaged and every claim re-run.
- **J-2** (ph5 review lanes, all exit 3). Same class: a foreign `git fetch origin --quiet` poller about every 10 min moved remote-tracking refs. One lane was also BLIND (never read its workspace); its verdict is void.
- **J-3** (fixture). The IQ2_XXS payload was mis-sized (premise correction 3). Caught by the R5 mutation, fixed, re-pinned.
- **J-4** (test design). The #1749 green path depended on the H12 10 tok/s floor (premise correction 4). Fixed before commit.

## Estimates
| K̂ | K | actual (k_measured, whole session incl. 9 Fable turns) | basis |
|---|---|---|---|
| 24 | 48 | 120 at the receipt commit (distinct assistant message ids in the session transcript) | `first-run[U]`. `estimate.sh aprender` exits 2: 47 ledger rows and none enters a total |

The status blocks' `global=` values were my running estimates, not measurements (`k_measured` printed `?`). The measured value is the one above.

## Gaps (NotRun, with what closes them)
- **G-1** Real-file exit check not run end to end (≈900 MB download plus a HEAD-built `apr`). The operator observed both RED on 0.68.0 (#3429 comment). Closed by running `scripts/pp_quant_exit_fixtures.sh` in a non-required lane. Wiring it into a workflow needs `.github/workflows` edits, which are operator-gated.
- **G-2** No contract row for #1749 (bench route) or #3091 (qtype byte size): the qtype table is owned by #3431/#3432.
- **G-3** The #1749 tensor table is duplicated from the serve builder (the builder itself is not copied). Each copy is pinned by its own sha256.
- **G-4** The Q-b dissent (lane 2) stands on record. #3432 must invert `regression_3091_…`.
- **G-5** Status-line join table not measured.
- **G-6** `transcript-gate.sh` from a linked worktree's cwd reads the wrong project dir and PASSes vacuously (`attempted=0`). A harness defect to file.
- **G-7** Pre-PR review is 2 independent PASS (plus 1 blind), not 3. The merge helper's three-PASS rule is not met by this quorum.

## Verdict
**PARTIAL(escalate)** — the code deliverable is complete and verified. Every fixture was observed RED under its revert and GREEN on HEAD, with contracts and the exit script in this PR (#3456, draft). Not DONE because:
1. The DoD's "merged green on `ci / gate`, `workspace-test`" is not yet reached.
2. The pre-PR quorum is 2 independent PASS plus 1 blind lane, so the three-PASS merge rule is unmet (G-7). A re-review needs lanes that are not voided by other sessions' `.git` writes (J-1, J-2).
3. Wiring `scripts/pp_quant_exit_fixtures.sh` into a non-required lane is a workflow edit and needs the operator (G-1).
