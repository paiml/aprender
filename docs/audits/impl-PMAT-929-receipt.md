# impl receipt — PMAT-929 (PP-LLAMA-001 v3.1 run 4: the software rows re-verified, and the 0.65.0 release drive)

## Identity
- ticket: PMAT-929 (umbrella; minted by hand — see jidoka 1). Linked: PMAT-930, 931, 932, 933, 934, 935 (PR #2861), PMAT-936, 937, 938, 939..948 (PR #2860)
- target: `docs/specifications/PP-LLAMA-001-MASTER.md` (v3.1 on origin/main). `[A]` the v3.0 copy in `~/Downloads/files(12).zip` (2026-09-02 09:28) is superseded by the v3.1 that #2851/#2854 landed; it was not added (a second source of truth). The audit in the zip is already on main at `docs/audits/parity-spec-audit-2026-09-02.md`.
- base: origin/main 68b059ca9 (0.65.0 in Cargo.toml; crates.io max 0.64.0 — the release is 0.65.0)
- discover.json sha256 (prefix):  · gate_cmd_fallback=true (`cargo test --workspace`); required_check = `ci / gate`, `workspace-test`; quorum_tool = agy 1.1.25
- worktrees: /mnt/nvme-raid0/agent-wt/pp4 (main, verification), pp4-930 (PR #2861), batch (PR #2860), pp4-work (durable artifacts — the session scratchpad under /tmp was wiped twice mid-run)

## Plan (Phase 1), routing and trigger
| phase | rows | route | trigger | acceptance |
|---|---|---|---|---|
| P1 verify LANDED rows | 0d 0a 0e 0b 0c 3 AppC 10 5 9 11 12 | direct | — | every §6-named case present on its surface (`spec_conformance.sh` 33 rows / 83 cases / 0 missing) + a live must-fire mutation per shell surface + the six rs cases |
| P2 fix the defect P1 found | Appendix C (PP-9) | subagent:sonnet then direct | Q3 | `a930.sh` (selftest names, live scan LEDGER 6 6, the escape shapes, contract, RATIONALE) |
| P3 review rounds | PMAT-930…935 | delegate (quorum:3 + §3.E arm) ×5 | Q2 pre-PR review | each round's findings re-run by the orchestrator before they counted |
| P4 land | #2857 #2859 #2809 #2861 #2860 | direct | — | `ci / gate` + `workspace-test` green, merge queue |
| P5 dogfood gates | PMAT-936/937/938 on #2860 | subagent:opus ×2 then direct; delegate ×1 | Q1 (|M|≥3) | `pmat verify` satd ok; CB-200 Warn at baseline; shell ratchet per file |
| P6 release | tag · cascade through `check_publish_preflight.sh` · post-publish dogfood | direct | — | version on crates.io + dogfood GO |

## Dispatch ledger
| # | mode | agent | turns | maxTurns hit | resumed | lane / width / conversations |
|---|---|---|---|---|---|---|
| 1 | subagent:sonnet | a4d7de30c9b0a8c44 (paiml-impl-worker, PMAT-930) | 40+40 | yes ×2 | once | — |
| 2 | delegate:opus | addeafcacfec1dda6 (round 1, PMAT-930) | 28 | no | — | quorum 3 + §3.E: df32ef55, f41060f3, 02623f0a, b2af861a |
| 3 | delegate:opus | a9e2e1a86b46445f8 (round 2, PMAT-931) | 33 | no | — | 519483fd, cb269b2c, 804fdc9e, 972bd0c8 |
| 4 | delegate:opus | ad7e590bc2dc7e3da (round 3, PMAT-932) | 30+ | yes | once | 611de8db, 87ecbcb2, 12f3e8e4 (lane 2 absent) |
| 5 | delegate:opus | aaf1244c23eda1b98 (round 4, PMAT-933) | 30+ | yes | once | f5451570, d7910b93, 76cbe3d0, dc6aa69e |
| 6 | delegate:opus | acdc40aa89dfb46d0 (round 5, PMAT-934) | 30+ | yes | once | 044542d6, 1ca23c06, 27e31188, 4c1a7eac |
| 7 | (round 6 dispatch rejected by the operator: "release autonomously in as few iterations as possible") | — | — | — | — | — |
| 8 | subagent:opus | ae9b62779413fe0b5 (paiml-impl-worker, PMAT-938 sweep) | 40+40 | yes ×2 | once | — |
| 9 | subagent:opus | a715768cc986a449c (paiml-impl-worker, PMAT-938 finetune.rs) | 40+ | yes | once | — |
| 10 | delegate:opus | adf8ed8e205b9e649 (#2860 review) | 30+ | yes | once | b78b3a0e, 8ebb2947, e411e800, 96e45c67 |
Peak concurrent Claude subagents: 1 (the invocation brief, verbatim: "≤1 subagent; agy for width on a logged Q-trigger"; the standing rule is recorded in project memory on 2026-09-02 as "ONE sub-agent at a time; prefer agy for fan-out" and was not re-stated in this session; ultracode was set by /effort mid-run and left as an `[A]` — width went through agy). agy child_conversations: 0 (quorum lanes spawn none).

## Verification table (claimed vs my rerun)
| claim | who claimed | my rerun | verdict |
|---|---|---|---|
| rows 0a/0b/0c/0d/0e/3/5/9/10/11/12 LANDED (spec §12) | spec v3.1 | 12 selftest surfaces rc=0 on 68b059ca9; every §6-named case found; live RED for 0d (boolean gpu_layers), 0e (case renamed), 10 (uncited ratio), 12 (cancel-in-progress); rs cases 6/6 ok; row 3 and 5 live mutations unreachable with no MEASURED cell (selftest cases are the evidence; the matrix schema guard turns RED on a cell without owner) | confirmed |
| Appendix C ledger re-spend refused | spec (`respend_same_key`) | duplicating row 6 as row 7 → PASS "no cell was spent twice" (LEDGER 4 4) | **refuted → PMAT-930** |
| round-1 escapes (no leading pipe, backticked id, trailing \|\|, extra pipe) | delegate lanes | no-leading-pipe: no violation; extra pipe: no violation; backtick/trailing inside the table: L1 still fired | 2 confirmed on the tree, 2 confirmed on the source → PMAT-931 |
| round-2: dummy first line; blank line in §13 flags 11 rows; crash counted as kill; bold id | delegate lanes | confirmed, confirmed (L2 6..16), confirmed (22 BROKE / 21 errored); bold id **refuted** (L1,L2 fired) | → PMAT-932 |
| round-3: foreign-header run skipped; row three columns off dropped | delegate | confirmed; confirmed once the probe carried the tier | → PMAT-933 |
| round-4: non-row-id first cell; backticked tier | delegate | confirmed; confirmed | → PMAT-934 |
| round-5: __RECORDED__; __lambda__ / code tags / zero-width key; CONFORMANT never deduped; harness lacked L1 mutant; LEDGER counts from two sets | delegate | all confirmed | → PMAT-935 |
| decomposition of 10 scanner functions is behaviour-preserving | me / lanes | scan TSV byte-identical vs main's scanner on the live tree at every head; every diff on mutated ledgers is a new rule (delegate measured round 5) | confirmed |
| PMAT-938 sweep: 33 → 0 strict SATD | worker | `pmat verify` satd ok on e1aed1972; `pmat analyze satd --strict` total 0 | confirmed |
| finetune.rs decompositions behaviour-preserving | worker + #2860 review | 7208 apr-cli lib tests; zero string literals lost (delegate); runner.rs breaks → returns, termination unchanged | confirmed |
| shell-lint ratchet number is deterministic | me | single invocation: main 13 (fleet) / 53 (local), batch 57 / 12; per file: main 12, batch 9 on both | confirmed → PMAT-936 |
| CB-200 baseline ratcheted | #2860 review | no script compared .pmat-gates.toml to origin/main | confirmed → mirrored file + pair check (must-fire 610 vs 609 → FAIL) |
| clean-room on the batch head 7fb86a59d (own target dir, CARGO_BUILD_JOBS=2, no sccache, doc --test-threads=4, --no-fail-fast) | the invocation brief, verbatim: "clean-room green FIRST (CARGO_BUILD_JOBS=2, doctest --test-threads=4)" | lib 88,530 passed / 0 failed; doc 1,140 passed / 0 failed — after fixing seven dark doctests (aprender-orchestrate `use aprender_orchestrate::` → `batuta`, aprender-profile-core `use renacer::` → `renacer_core`) and two racy orchestrate tests (ETXTBSY fork/exec window; a fake MCP server exiting before the request was written) | green on the batch head; to be re-run on the final main |
| full dogfood on the batch head (main's dogfood.sh, no --phase) | release gate | NO-GO on the by-construction rows #2859 defers (publish-dry-run, version-unpublished, multiplatform receipts) plus two real rows fixed here: cli-surface probed the words of wrapped `--help` lines as subcommands ("existing", "producer", "yet" …); bashrs counted two unguarded `rm -rf` in dogfood.sh | fixed on #2860 (92b7188bb) |
| `intel-clean-room-6` failing every job at "Set up runner" | CI | root-owned `_work/_temp/_github_home` residue from a container; six of sixteen runners carried it; ownership restored on the host per ci.yml's own remediation | ENV; #2861 and #2809 re-queued by close/reopen |
| #2857 / #2859 / #2809 red runs | CI | check-run annotations: zero failed steps, "runner lost communication" (outage 09:29–11:21Z and ~16:00Z; the intel host rebooted, 16 services inactive) | ENV; fresh runs by close/reopen, no rerun issued |

| #2860 run 33827395851: lint/coverage/workspace-test/guard failures | CI | lint = rustup download failed (env); coverage = runner-6 ContainerId null (env); workspace-test = `test_readme_contract_count_matches_workspace` 1805≠1807 (real); guard = FALSIFY-README-002 1805≠1807 (real, same defect) | one defect, README count |
| README 1807 fixes both (the two contracts are this PR's own: apr-complexity-ratchet-v1.yaml, pv-artifact-kinds-v1.yaml; commit c91b1c6f6's message blamed #2857 — wrong, origin/main counts 1805) | me | `check_readme_claims.sh` PASS 1807; `check_no_claim_literals.sh` PASS; `check_perf_claims_cite_receipts.sh` PASS | PASS |
| #2860 run 33838507493: PERF-009 `count=4 baseline=0` | CI | reproduced locally on the batch tree; origin/main counts 0; the four flagged scripts are exactly the bashrs-edited ones | real |
| PERF-009 root cause | me | ship-002/ship-008 used `date -u +%s` on main (guard regex `date \+%s` never matched); eval-shard/ex-06/probe only stamp filenames and gained `$(date +%s)` from the DET002 idiom | guard blind spot + false positives |
| PERF-009 fix (PMAT-949, e70b00760) | me | guard `count=0 baseline=0 OK`, selftest 0 BROKE; mutation 1 (revert `( -u)?`) → `BROKE date -u +%s counts as timing`, rc=1; mutation 2 (drop ship-002 allowlist) → `COMPETING ship-002`, count=1; bashrs error lines 0 on the three scripts; `check_baseline_ratchets.sh` PASS; `check_shell_lint_ratchet.sh` PASS | PASS, both mutations RED |
| pmat 3.36.0 re-pin (teammate message) | teammate session | no in-repo version pin exists on main (CI runs `cargo install pmat --locked`); the comment-divider stopgap is not on origin/main; local pmat is 3.36.0 | nothing to change in-repo |

| predicted-main clean-room (8100ecd06 = main+#2809+#2860+#2861) | me | lib 6568/1 FAILED: batuta `test_stdio_transport_process_exit_failure` (`write stdin: Broken pipe`), doc phase pending | one real race, PMAT-950 |
| PMAT-950 transport fix | me | 40/40 green on both stdio tests; mutation (revert transport change) → new test RED `got: write stdin: Broken pipe` | PASS, mutation RED |
| dogfood --phase pre-publish preview on 8100ecd06 | me | NO-GO on exactly one gate: bashrs 7 SEC/DET/IDEM over 238 files (DET002 ×2 mine, SEC010 ×4 #2861, SEC010 ×1 #2859's preflight); every other row PASS/DEFER/SKIP as designed | 7 → 0 after fixes on both branches |
| #2861 roadmap relocation | me | real `git merge --no-commit` of #2861 onto #2860 and onto #2809: 0 conflicts | saves one queue cycle |

| #2860 run e24051530 guard-runner-labels: complexity ratchet `moved backwards` | CI | CI measures the PR's MERGE commit: main (with #2809) brought `cublas_prefill/attention.rs::batched_gemv_or_gemm` under both thresholds → 691 vs 692 rows, one STALE. Reproduced on intel with the fleet's pmat 3.31.0 on the bare branch (692, no diff) and locally on the merged tree; not a version skew | ratchet correct; row deleted with main merged |
| predicted-main clean-room #2 (8cd505610 = main+#2860 e24051530+#2861 cec7150a9) | me | lib 88541/0, doc 1140/0 | GREEN |

| perf_gate.sh --phase release on the release sha 587ad0797 | me | `VERDICT PASS host=lambda phase=release workload=W1` — RELABELLED phase=pre-publish-checkout (RELABELLED 2026-09-05, PMAT-967): the receipt the gate graded is evidence/perf-gate-001-w1-lambda/receipt.r1.json — commit 745fa8588, a dev checkout built at /mnt/nvme-raid0/perf-gate/target-cuda/release/apr, feature_set [cuda], binary sha256 9d0b08b015e22fb3e2cade8f0862d31b03e970c16ac8f2abf77140b7b8b63ed5, W1 7B, ledger row 3 (SPENT, subject lane invalid, unsigned). It is not the artifact that was published (crates.io `cargo install aprender`, default features = cli, CPU-only) and cannot be quoted as a release verdict; its PASS meant only that no W1 arm was ARMED; one SKIP (ArmE is W2-only) | PASS-as-printed, relabelled pre-publish-checkout |
| clean-room on 587ad0797, run 1 | me | lib phase HUNG 20+ min: 49 threads in futex_wait in the aprender-compute lib tests; gdb: ABBA deadlock DEVICE_INIT_LOCK ↔ shared_instance OnceLock (PMAT-952); killed | RED, root-caused |
| PMAT-952 fix (#2862, 2f36db3e5) | me | fresh-process probe: fix 2 passed; mutant (mutex restored) FAILED, probe child exit 101 at 30 s; fix again 2 passed; agy §3.E review implement-as-written, 4 confirmations; receipt validated rc=0 | PASS, mutation RED |
| clean-room on 587ad0797, run 2 | me | lib 88537 passed / 4 failed (batuta stdio MCP tests, clean-exit `write stdin: Broken pipe`, PMAT-953); doc 1140/0 | RED, root-caused |
| PMAT-953 fix (#2863, 95ea02252) | me | fix 20 passed; mutant (stdout-empty guard reverted) FAILED `got: Err("write stdin: Broken pipe")`; fix again 1 passed; agy review implement-as-written, 3 confirmations; receipt validated rc=0 | PASS, mutation RED |
| dogfood --phase pre-publish on 587ad0797 | me | `VERDICT: GO`, 41 rows, no FAIL; DEFER rows are the post-publish ones by design; receipt `.dogfood/receipt-20260904T153728Z.json` | GO |
| tag v0.65.0 | me | annotated tag at 587ad0797, pushed 16:13Z; version from cargo metadata; 0.65.0 absent on crates.io before the push (measured) | done |
| check_publish_preflight.sh (F-9, R1–R5) | me | R1 clean tree, R2 version 0.65.0, R3 tag at HEAD, R4 ancestor of origin/main, R5 dogfood GO for 587ad0797 → PASS; cascade-drain started with no --allow-dirty (verified on origin/main: `cargo publish "${sel[@]}" --locked`) | PASS |

| cascade-drain (0.65.0) | me | preflight R1–R5 PASS; pass 1 published 45, pass 2 none: `STUCK 48/74`, no reason in the log (the drain filters DEFER lines: PMAT-954). `cascade-publish.sh --only-tier 2` by hand: `aprender-core DEFER: failed to select a version for aprender-data ^0.65.0`; aprender-data's dev-dep `aprender-test-lib ^0.65.0` (workspace alias carries the version) needs aprender-core: a version-level cycle (PMAT-955). 48 crates live at 0.65.0, the root and 25 others at 0.64.0, consistent | STOPPED; fix on #2864 |
| PMAT-955 fix | me | five sibling dev-deps rewritten path-only; `cargo metadata`: versioned sibling dev-deps = []; `cargo check --tests` on the four crates: Finished; preflight R6 added: selftest 18/18, mutation (R6 never fails) → `BROKE versioned_sibling_devdep_refuses`, 17/18; drain keeps DEFER; workspace bumped to 0.65.1 (`bump-version.sh --check` ok) | PASS, mutation RED |

| PMAT-954/955 PR #2864 (f95045f18) | me | preflight selftest 18/18, R6 mutation 17/18 BROKE; `check_workspace_siblings_pathed.sh` PASS; `check_baseline_ratchets.sh` PASS; `check_no_claim_literals.sh` PASS after the CHANGELOG insert was withdrawn (line-keyed baseline refused the coordinate shift: PMAT-956 filed); `cargo test -p aprender-contracts --lib` 1501/0; version 0.65.1 (`bump-version.sh --check` ok) | PASS |

| batch #2865 (307b8655c = main + #2864 + #2862 + #2863, merge commits) | me | roadmap keep-both union: yaml loads, 703 items, 0 duplicate ids, PMAT-952..956 present; `cargo metadata`: versions all 0.65.1, versioned sibling dev-deps []; preflight selftest 18/18; `check_baseline_ratchets.sh` PASS; fmt ok; agy composition lane implement-as-written (2 measured confirmations), 12 prior findings from the three lanes carried in the SARIF; receipt validated rc=0 | PASS |
| fleet priority (operator: "renice our builds and rmedia (deprioritize)") | me | `/home/noah/data/ci-nice.sh` on intel: aprender runner cgroups + job containers at nice −5, rmedia at +15, re-applied every 60 s for 8 h (`ci-nice.stop` kills it); verified: 7 aprender / 5 rmedia busy runners, top rustc processes at nice −5 | applied |

| #2865 merged → main 752f55346 (0.65.1) | CI | merge-group run 33906769395 success; tree hash 9e663aaae67e identical to the batch head dd7eebe88 on which the pre-built clean-room ran | landed |
| pre-built clean-room on dd7eebe88 (tree = release tree) | me | lib 85962 passed / 0 failed / 1 target crashed: aprender-gpu (trueno_gpu) SIGSEGV in the CUDA driver tests; doc 1140/0 | RED on one crate (PMAT-957) |
| PMAT-957 characterization (same binary, same box) | me | 48 threads: 3/10 runs fail (CUDA_ERROR_UNKNOWN 901/906 in cuda_graph/gpu_buffer; one SIGSEGV in test_context_creation_device_0); 4 threads: 0/10; old tree 587ad0797: 3/3 green (too small to attribute); the crate #2865 touched for wgpu init is not the crate that fails | pre-existing-class race, filed |
| perf_gate.sh --phase release on 752f55346 | me | `VERDICT PASS host=lambda phase=release workload=W1` — RELABELLED phase=pre-publish-checkout (RELABELLED 2026-09-05, PMAT-967): the receipt the gate graded is evidence/perf-gate-001-w1-lambda/receipt.r1.json — commit 745fa8588, a dev checkout built at /mnt/nvme-raid0/perf-gate/target-cuda/release/apr, feature_set [cuda], binary sha256 9d0b08b015e22fb3e2cade8f0862d31b03e970c16ac8f2abf77140b7b8b63ed5, W1 7B, ledger row 3 (SPENT, subject lane invalid, unsigned). It is not the artifact that was published (crates.io `cargo install aprender`, default features = cli, CPU-only) and cannot be quoted as a release verdict; its PASS meant only that no W1 arm was ARMED; one SKIP (ArmE is W2-only) | PASS-as-printed, relabelled pre-publish-checkout |

| clean-room on the release sha 752f55346 | me | lib 88544 passed / 0 failed (76 crates, rc=0; aprender-gpu green this run); doc 1140/0 | GREEN |
| dogfood --phase pre-publish on 752f55346 | me | `VERDICT: GO`, WARN rows: changelog (no 0.65.1 entry, PMAT-956), reachability (standing); receipt `receipt-20260904T191717Z.json` commit 752f55346 version 0.65.1 | GO |
| tag v0.65.1 | me | annotated tag at 752f55346 pushed 19:39Z; 0.65.1 absent on crates.io before the push (measured) | done |
| check_publish_preflight.sh R1–R6 on 752f55346 | me | R1 clean, R2 0.65.1, R3 tag at HEAD, R4 ancestor of origin/main, R5 dogfood GO for 752f55346, **R6 no sibling dev-dependency carries a version** → PASS; cascade-drain started (no --allow-dirty) | PASS |

| cascade-drain 0.65.1 | me | 9 passes, then `STUCK 67/74`; 64 crates verified live at 0.65.1 on crates.io (incl. aprender-core, aprender-serve, aprender-train); DEFER reasons now visible (PMAT-954): `aprender-test-lib … could not compile (lib)` → `src/perf_gate/protocol.rs:93 include_str!("../../../../scripts/perf-matrix.yaml")` cannot be in the package tarball (PMAT-958); blocked behind it: aprender-test-cli, aprender-orchestrate (optional jugar-probar dep), apr-cli, **aprender (root)**, aprender-monte-carlo, aprender-tsp; transient `failed to publish` (viz, registry, train-*) resolved on later passes | STOPPED at 64/74; fix on 0.65.2 |
| PMAT-958 fix | me | build.rs copies scripts/perf-matrix.yaml into OUT_DIR in the workspace, embeds the vendored copy in a published crate, and refuses to build when the two differ; see the mutation and dry-run rows below | — |

| PMAT-958 build.rs + vendored copy (ed995c709) | me | `cargo build -p aprender-test-lib -vv`: build script ran, `OUT_DIR/perf-matrix.yaml` written; mutation (vendored copy drifted) → `failed to run custom build command … PMAT-958 … differs`; `cargo publish -p aprender-test-lib --dry-run --locked` at 0.65.1 deps: Packaged 242 files, Verifying, Finished, upload aborted by dry run (rc=0) — the same command failed on origin/main with `couldn't read src/perf_gate/../../../../scripts/perf-matrix.yaml` | PASS, mutation RED |

| CB-510 guard widening (PMAT-958) | me | `check_package_includes.sh --self-test` rows 5–8 ok; mutation (escape predicate disabled) → `FAIL row 5`; on the tree: `PASS … all 10 crate(s)`; origin/main's protocol.rs is flagged by the resolver (`../../scripts/perf-matrix.yaml src/perf_gate/protocol.rs`); residual found: aprender-present-lib wasm32-only showcase.rs includes a path that exists only under crates/aprender-present/ (PMAT-959, skipped visibly, not silently) | PASS, mutation RED |

| #2866 §3.E review round 1 (agy, review schema) | agy lane | verdict do-not-implement-as-written, 6 non-note findings: rerun-if-changed on a missing path (real: cargo reruns the script every build in a published crate — confirmed from cargo's fingerprint rule), false drift panic in a foreign tree (real: a path/git consumer with an unrelated ../../scripts/perf-matrix.yaml), truncation at the first #[cfg(test)] hides later production code (real), commented `use wasm_bindgen` skips a file (real), block comments not stripped (real), concat!(env!(CARGO_MANIFEST_DIR)) escapes missed (real) | all six re-verified and applied (3952ebd38): rows 9–13, mutation still RED, tree PASS 10 crates |
| build.rs after the review | me | workspace build Finished; drift mutation → `PMAT-958 … differs`; rerun-if-changed now emitted only for existing paths; the workspace file trusted only when ../../Cargo.toml is the aprender workspace manifest; `--escapes` on the tree: 0 (artifact.rs's include lives under `#[cfg(all(test, feature = "setfit"))]`, now recognised) | PASS |

| #2866 ci / lint on 4168cab3f | CI | Clippy: `manual_assert` on build.rs:47 (pedantic under -D warnings; the local check had run `cargo build`, not clippy) | real; fixed e6ab06c0d (assert!), drift mutation still RED, clippy Finished |

| pre-built clean-room on #2866's head e6ab06c0d (tree = the 0.65.2 release tree once squashed) | me | lib 88544 passed / 0 failed (76 crates, aprender-gpu green this run); doc 1140/0 | GREEN |

| #2866 merged → main 8e1e9ad40 (0.65.2) | CI | merge-group run 33921821174 success (workspace-test, guard-runner-labels, ci / gate) | landed |

| perf_gate.sh --phase release on 8e1e9ad40 (0.65.2) | me | `VERDICT PASS host=lambda phase=release workload=W1` — RELABELLED phase=pre-publish-checkout (RELABELLED 2026-09-05, PMAT-967): the receipt the gate graded is evidence/perf-gate-001-w1-lambda/receipt.r1.json — commit 745fa8588, a dev checkout built at /mnt/nvme-raid0/perf-gate/target-cuda/release/apr, feature_set [cuda], binary sha256 9d0b08b015e22fb3e2cade8f0862d31b03e970c16ac8f2abf77140b7b8b63ed5, W1 7B, ledger row 3 (SPENT, subject lane invalid, unsigned). It is not the artifact that was published (crates.io `cargo install aprender`, default features = cli, CPU-only) and cannot be quoted as a release verdict; its PASS meant only that no W1 arm was ARMED; one SKIP (ArmE is W2-only); release tree hash 7149b5cf4031 identical to the pre-built clean-room's | PASS |
| clean-room on the release sha 8e1e9ad40 | me | lib 88544 passed / 0 failed (76 crates, rc=0); doc 1140/0 | GREEN |
| dogfood --phase pre-publish on 8e1e9ad40 | me | `VERDICT: GO`; receipt `receipt-20260904T221607Z.json` commit 8e1e9ad40 version 0.65.2 | GO |
| tag v0.65.2 | me | annotated tag at 8e1e9ad40 pushed 22:38Z; 0.65.2 absent on crates.io before the push | done |
| check_publish_preflight.sh R1–R6 on 8e1e9ad40 | me | all six ok → PASS; cascade-drain started (no --allow-dirty) | PASS |

## Jidoka log (`.pmat/jidoka.jsonl`, entries 3–25 of this run)
1. pmat work add minted PMAT-744 / PMAT-497 (live elsewhere) and rewrote 2046 roadmap lines → tickets minted by hand; paiml/paiml-mcp-agent-toolkit#1169
2. PMAT-930: PP-9 live scan read only the first pipe table
3. PMAT-931: L2 was a whitelist of one row shape
4. PMAT-932: the run-skip rule failed both ways; the harness had no baseline
5. PMAT-933: header/width conditions were still author-satisfiable
6. PMAT-934: the row id was required and the tier read without the shared normalisation
7. PMAT-935: emphasis/code-tag/zero-width renderings of the key and tier; L1 deduped only RECORDED
8. PMAT-936: the shell-lint ratchet ratcheted a nondeterministic number
9. PMAT-938 (P5): the pre-commit hook refuses any edit to a debt-carrying file (11 functions decomposed as the price of 33 marker rewrites)

- runner-6 root-owned residue killed 8 jobs in 75 min at "Set up runner" (two merge-queue evictions, #2809 and #2861); filed paiml/paiml-mcp-agent-toolkit#1185 (pmat's container jobs write root files with no chown step); ownership restored on all 16 runners at 06:44Z, five still carry paths chown could not reach.

- CUDA nightly on gx10 (cuda-nightly.yml, fdc340e0f, 06:21Z): PP-26 witness PASS at c=1/4/8, FAIL at c=16 (9 of 16 slots diverge at chunk 31 < declared_min 64; `perf041-parity: exit 1`). Identical to the 2026-09-03 run (bfe5439ac) and every nightly since 08-30: a standing INVALID-CORRECTNESS band, not a regression from this run. Not in the §12 kill table; the gx10 cells in scripts/perf-matrix.yaml are UNMEASURED/NA with anchors (rows 15/18, hardware window). Owner: #2753 / #2809 (batched decode). Action: after #2809 lands, dispatch cuda-nightly.yml on main to re-take the witness before quoting any gx10 c>1 figure.

- runner-6 root cause (jidoka 11): the fleet pre-job hook probes `cargo --version` from the STALE job workspace; runner-6's `_work/aprender/aprender/rust-toolchain.toml` was 0 bytes (killed job, Sep 3 16:47), rustup refused the empty override, the hook reported `missing: cargo`, and no job could reach checkout to repair it. Replicated: probe PASS from the runner root and the pmat checkout, FAIL only from that workspace. Restored from git at 08:40Z (`cargo 1.93.0` there now). The root-owned residue (pmat#1185) was real but not this cause. 14 aprender jobs and three merge-queue runs died on it between 00:28Z and 08:40Z.


## Decisions recorded as the operator's (quoted verbatim) and as mine [A]

- Operator, 2026-09-04: "we will cherry pick this and not stop release" (on PMAT-952) and "remember to autonomously release". Applied: v0.65.0 tagged at 587ad0797 with the release clean-room NOT green on that sha (run 1 hung on PMAT-952, run 2 failed 4 batuta stdio tests on PMAT-953); both defects are root-caused, fixed with must-fire mutations RED on PRs #2862 and #2863 armed for auto-merge behind the tag. The two predicted-main clean-rooms earlier the same day (8100ecd06, 8cd505610) were lib 88541/0 and doc 1140/0 on the same test code; the release-sha runs differed only in load and scheduling.
- Operator, 2026-09-04: "load is dropping and I have told other projects to stop … keeping what we have, not dropping" — all three PRs landed in order (#2809 10:47Z, #2861 14:00Z, #2860 15:04Z).
- [A] The §3.E cross-vendor review lanes for #2862 and #2863 were run by me directly through `agy` (one lane each, review schema) rather than through the `paiml-agy-delegate`, because the spec-review workflow already held the single subagent slot. Conservative reading of the operator's "ONE sub-agent at a time; prefer agy for fan-out".
- Operator (verbatim): "[continue release in parallel and much this request to a background agent and ensure \"agy teamework and agy plan are used to review\"]" — so the 0.66 spec grill ran as a background Workflow (worker → agy teamwork-preview → agy plan quorum → worker apply). [A] The lane composition (teamwork-preview then a 3-lane plan quorum) was my choice within that instruction.

## Estimates

| K̂ | K | actual (my turns) | basis |
|---|---|---|---|
| 90 | 150 (operator cap; the operator re-issued the run with "release autonomously in as few iterations as possible", which reset the andon) | ≈385 across two context windows | `docs/audits/impl-estimates.jsonl` L2 (prior run 88/6 phases) for the plan; the overrun is the seven rows below |

Per-phase rows appended to `docs/audits/impl-estimates.jsonl` (est → actual): P1 verify 13 → 24 · P2/P3 ledger scanner 15 → 58 · P5 dogfood gates 15 → 34 · P4 land five PRs 10 → 62 · P5 cut ×3 7 → 48 · P5b eight cut-day defects 0 → 46 · spec/report writing (operator messages mid-run, verbatim: "while waiting I need you to create a docs/reports/work-history-delay-optimization-report.md" and "in parallel write a detailed docs/specifications/0.66-performance-parity-report.md") 0 → 12. The estimator counts phases; the cost was CI cycles (~1 h per merge, 2.3 PR runs per PR) and mid-cascade discoveries that no pre-publish gate could see. `docs/reports/work-history-delay-optimization-report.md` quantifies both.

## Gaps (NotRun lanes and the artifact that closes each)
- 0.66 spec review (Workflow wf_2faf0298-d56): the agy `/teamwork-preview` lane returned nothing parseable within its 25-minute budget (recorded as null); the 3-lane agy `--mode plan` quorum ran (consensus implement-with-changes; 3/3 lanes on the missing owner column; the delegate refuted lane 2's unsourced-figures claim by citing §12) and its findings were applied by the worker (owner column, single release-criteria statement, B-W5/B-M4/B-S4 rows, root manifest in the rename-guard universe, zero-slack expiries). Receipt: evidence/spec-review/0.66/plan-quorum-receipt.json. A re-run of the teamwork lane is owed before 0.66 starts.
- Round-6 review of #2861's final head (d9d77704a + receipt/signature commits): NotRun. Operator (verbatim): "we will cherry pick this and not stop release" and "release autonomously in as few iterations as possible"; [A] reading those as 'no further review round on an already-reviewed ancestor' was mine. The receipt reviews 3c646f55b (an ancestor) and dispositions every later fix.
- #2860: the CUDA-docs consultation is `unreachable` (MCP endpoint ETIMEOUT); verdict DEGRADED, recorded.
- `cargo mutants --in-diff`: not applicable to any PR of this run (no Rust in #2861; #2860's Rust is decomposition-only and covered by crate tests + the committed shell mutation sets).
- Rows 2, 13–21: hardware windows; `perf-matrix.yaml` carries UNMEASURED{owner, expires_after} for lambda/gx10/intel W1/W2 and NA{decided_by} for mini; nothing was measured from this host, no cell fabricated.
- Release steps after the merges (perf_gate release phase on main, clean-room, dogfood --phase pre-publish GO, tag, cascade through check_publish_preflight.sh, post-publish dogfood): see the verdict section.

## Verdict
**PARTIAL(escalate)** — release sha `8e1e9ad40`; crates.io 74/74 crates at the released version; the publish is complete (0.65.0 reached 48/74, 0.65.1 64/74, 0.65.2 74/74; GitHub release v0.65.2) and the pre-publish dogfood was GO (41 rows, no FAIL), but the POST-publish dogfood is NO-GO on measured evidence: every host's parity row FAILs (lambda cpu lane decode 0.59x/prefill 0.005x and cuda lane decode 0.69x/prefill 0.18x against llama.cpp 39173bcac at c=1; every c>1 band on every host is INVALID-CORRECTNESS(#2753/#2776) because no PP-26 witness was run, so no c>1 ratio exists; intel/gx10/mini could not emit a block because the published binary fails every request at c>=8 on the two aarch64 hosts and one c=16 replicate on intel; gx10 3.5 tok/s and mini 4.5 tok/s at c=1 are 0.046x/0.05x (medians vs comparator medians 75.7/90.4)), plus the PMAT-960 gate polarity defect on version-unpublished. Receipts: evidence/dogfood/0.65.2/{lambda,intel,gx10,mini}.json and VERDICT.md; dogfood receipts evidence/release/0.65.2/dogfood-postpublish-receipt-run2.json (claim-literal RED on the 0.66 report, fixed by receipting every figure) and run 3 (final, appended when it lands). Tickets filed from the receipts: PMAT-960 (gate polarity), PMAT-961 (pin resolver refused every non-CUDA comparator; fixed in #2867, MERGED), PMAT-962 (CPU prefill at decode speed), PMAT-963 (request failures at c>=8), PMAT-964 (published aarch64 binaries below their own H12 floor). DECIDED (decided_by: noah, 2026-09-05, verbatim): "0.65.2 stays published. No yank, no 0.65.3. These four host receipts are the 0.66 baseline." The 0.66 discovery baseline is 8e1e9ad40. cuda-default / W-I work is design ticket R-2 in PP-066, not a re-cut. Transcript-gate PASS (peak subagent concurrency 1 <= 3, 11 intervals, session adf433f2); status-lint PASS (19 blocks, all with basis=)

Re-run policy honoured: no `--skip`, no `--allow-dirty`, no `gh run rerun` (every fresh CI run came from close/reopen), no `git reset --hard`. Every claim in the verification table above carries my own re-run.

PASS status-lint: 19 blocks, all with basis=
PASS transcript-gate: 11 subagent interval(s), peak concurrency 1 ≤ 3 (session adf433f2-0c65-4a24-be88-e6e752517df7, registered by hook)
