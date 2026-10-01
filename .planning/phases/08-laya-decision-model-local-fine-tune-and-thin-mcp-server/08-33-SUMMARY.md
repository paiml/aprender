---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 33
subsystem: contract-hygiene-repair
tags: [contract-hygiene, sigma, valid-under, formal-vocabulary, pv-validate, tracked-graph, strict-test-binding, gap-closure]
status: complete
outcome: complete-locally
gap_closure: true

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-32's measured reds (five in aprender-contracts-cli) and its owner decision 'Repair all (new gap plan)'"
provides:
  - "13 contracts conforming to upstream's contract-hygiene rules, every upstream gate that reads the corpus green with nothing loosened"
  - "08-33-FORMAL-REWRITES.json: 105 claim-identical formal rewrites (96 by recorded rule, 9 by hand with reasons)"
  - "08-33-HYGIENE-EVIDENCE.json: before/after gate numbers, premise corrections, baseline integrity, planted-violation pairs"
  - "contracts/contracts.nt regenerated from a clean export (the working tree cannot reproduce it, see Deviations)"
affects: [08-34]

actuals:
  tokens: 106211
  tasks: 4
  commits: 7
plan_head_before: c028b4eec24c2c023d3f6ed249022342f128b2b8

tech-stack:
  added: []
  patterns:
    - "declared-not-executed kani harnesses naming an existing falsification test (spectral-indices-v1, after laya-parity-v1)"
    - "generate the tracked graph in a clean git clone, never in a working tree that carries untracked worktrees"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-33-FORMAL-REWRITES.json
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-33-HYGIENE-EVIDENCE.json
  modified:
    - contracts/chronos-bolt-parity-v1.yaml
    - contracts/decide-apr-v1.yaml
    - contracts/decide-tool-boundary-v1.yaml
    - contracts/forecast-tool-boundary-v1.yaml
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/laya-parity-v1.yaml
    - contracts/neon-blis-v1.yaml
    - contracts/neuralprophet-parity-v1.yaml
    - contracts/prophet-parity-v1.yaml
    - contracts/setfit-benchmark-claims-v1.yaml
    - contracts/setfit-train-lifecycle-v1.yaml
    - contracts/spectral-indices-v1.yaml
    - contracts/tweet-eval-stance-benchmark-v1.yaml
    - contracts/contracts.nt
    - crates/aprender-forecast/src/bolt.rs
    - crates/aprender-forecast/src/chronos.rs
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "Implication glyph is U+21D2, if-and-only-if is U+2194: the plan named 'the implication glyph' without picking one of the two declared implication glyphs (U+21D2 and U+27F9); the recorded rules fix the choice."
  - "No qualifier on any valid_under: no contract states a host/toolchain/backend/feature restriction as a claim of the contract itself."
  - "pv diff's minor-bump suggestion for spectral-indices-v1 was taken (1.0.0 to 1.1.0), and no other contract's version changed."
  - "contracts.nt is generated from a clean clone, not the working tree."

requirements-completed: [D-07, D-10, D-13, D-17]

coverage:
  - id: D1
    description: "Every contract conforms to upstream's hygiene rules and upstream's corpus gates are green: pv lint contracts Pass (only the unarmed reverse-coverage skipped), pv validate 0 of 1850 failed, census fresh, README count synced, ratchet green"
    requirement: "D-07"
    verification:
      - kind: other
        ref: "08-33-PLAN.md Task 4 verify payload run to exit 0 (verify4c, two lines adapted, see Deviations)"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-contracts-cli in a clean clone of 01b4d9f8d: 394 passed, 0 failed"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-contracts --lib (1671 passed) --test validate_contracts --test kani_harness_generation --test probar_test_generation: 0 failures"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every hygiene edit is claim-identical: 105 formal rewrites reproduce from the recorded rules or carry a reason; pv diff identical for 12 contracts and exactly the suggested minor bump for spectral-indices-v1"
    requirement: "D-13"
    verification:
      - kind: other
        ref: "Task 1-4 verify payloads: ledger reproduction assert, pv diff identical loop, 133 laya_claims anchors present"
        status: pass
    human_judgment: true
    rationale: "Notation-only equivalence of nine hand-written formals is a reading judgement no test asserts; each is recorded with its reason and the owner may overrule per contract."
  - id: D3
    description: "Nothing loosened: baselines, ontology and all gate sources byte-identical to 2817c6d97; each repair category shown to bite by a planted violation at zero headroom"
    requirement: "D-10"
    verification:
      - kind: other
        ref: "git diff --quiet 2817c6d97 over the gate surface; planted must-fail/must-pass pairs in the Task 4 verify"
        status: pass
    human_judgment: false
  - id: D4
    description: "Phase 8 pins hold: just laya-claims-check OK 133 rows, contract-audit-phase8 exit 0, decide crates and apr-format lib tests pass"
    requirement: "D-17"
    verification:
      - kind: other
        ref: "just laya-claims-check: LAYA CLAIMS OK 133 rows; make contract-audit-phase8 exit 0"
        status: pass
    human_judgment: false

duration: 39min
completed: 2026-09-29
---

# Phase 8 Plan 33: Contract-hygiene repair Summary

**Thirteen contracts brought into upstream's hygiene vocabulary (105 formals restated claim-identically, twelve worlds declared, two `pv validate` errors and nine dangling test citations cleared, the graph regenerated) so upstream's Sigma, valid-under, validate, graph and strict-binding gates all land exactly at their shrink-only baselines with nothing loosened, committed locally and unpushed.**

## Performance

- **Duration:** 39 min (2026-09-29T21:05:06Z to 21:44Z)
- **Tasks:** 4 (Task 1 tracer)
- **Files:** 19 changed (13 contracts, the graph, two Rust test modules, ledger, evidence, deferred-items)

## Accomplishments

- Task 1 (tracer): reproduced the reds verbatim (fragments 336/338/455 and the hidden ones), repaired laya-parity-v1 alone and measured the gates move by exactly its share (formal_prose 1569 to 1565, without-world 398 to 397).
- Task 2: Phase 8's other three contracts (21 formals): 1544 and 394.
- Task 3: the seven Phase 3-6 contracts (80 formals, nine by hand): 1464 and 387; audits 2-5 green, audit 6 still exactly its nine unbound equations.
- Task 4: neon-blis-v1 and spectral-indices-v1 validate, the five `#[test]` attributes reordered, graph regenerated, every gate re-run, integrity and bite proved.

## Gate table (before, after)

| Measure | Before (3c0f2cf00) | After | Baseline / ceiling |
|---|---|---|---|
| `formal_prose` | 1569 | 1464 | 1464 |
| `contracts_without_valid_under` | 398 | 386 | 386 |
| strict-binding dangling total | not measured (VACUOUS); chronos-bolt-parity-v1 9 vs baseline 0 once unmasked | 27 (no contract above its own line) | sum 27 |
| `contract_data_integrity` | 445 | 444 | ceiling 444 |
| `pv validate contracts` | 2 of 1850 failed | 0 failed | 0 |
| `pv lint contracts` | stops at validate | Pass, all armed gates ran | Pass |
| ont2b (fragment 336) | 6 passed, 2 failed | green | |
| ont4b (fragment 338) | 10 passed, 1 failed | green (in a clean clone, see Deviations 1) | |
| ont7 (fragment 455) | 11 passed, 2 failed | green | |
| `aprender-contracts-cli` | 389 passed, 5 failed | 394 passed, 0 failed (clean clone) | |
| `aprender-contracts --lib` | 1665 passed, 6 failed | 1671 passed, 0 failed | |
| `aprender-contracts --test validate_contracts` | 8 passed, 2 failed | 10 passed, 0 failed | |
| `book_coverage` | passing (premise correction) | passing | |

Task-boundary gate counts (sigma / valid-under): base 1569 / 398; after Task 1 1565 / 397; after Task 2 1544 / 394; after Task 3 1464 / 387; after Task 4 1464 / 386.

## Kind and world table

All twelve kernel-kind contracts without a world now carry `metadata.kind: kernel` written out and `metadata.valid_under: {world: committed}` with no qualifier. Each is a kernel in this schema's sense: it carries equations, proof obligations, falsification tests and kani harness declarations, exactly what `schema/kind.rs` says `kernel` demands. Nothing was reclassified (that would exit PROVABILITY-001, a loosening).

| Contract | kind (was) | Why kernel | World |
|---|---|---|---|
| chronos-bolt-parity-v1 | kernel (declared) | frozen parity bars: equations, obligations, tests, harnesses | committed |
| decide-apr-v1 | kernel (declared) | the .apr artifact schema and load ladder, with obligations and tests | committed |
| decide-tool-boundary-v1 | kernel (declared) | the tool boundary's constants and refusals as equations and obligations | committed (Lambda tier not expressible in the closed qualifier set, so none) |
| forecast-tool-boundary-v1 | kernel (declared) | bounds and refusals as equations and obligations | committed (see quote below) |
| laya-finetune-gate-v1 | kernel (declared) | the fail-closed gate's claims | committed |
| laya-parity-v1 | kernel (declared) | D-13's parity bars | committed |
| neuralprophet-parity-v1 | kernel (declared) | parity bars | committed |
| prophet-parity-v1 | kernel (declared) | parity bars | committed |
| setfit-benchmark-claims-v1 | kernel (declared) | benchmark-claim rules with obligations and tests. **Sits oddly as a kernel: it is a benchmark-claims contract.** | committed |
| setfit-train-lifecycle-v1 | kernel (was the default, now written out) | lifecycle rules with obligations and tests | committed |
| tweet-eval-stance-benchmark-v1 | kernel (was the default, now written out) | dataset rules with obligations and tests. **Sits oddly as a kernel: it is a dataset-evaluation contract.** | committed |
| spectral-indices-v1 | kernel (was the default, now written out) | index formulas with obligations and tests; gained declared-not-executed kani harnesses | committed |

setfit-benchmark-claims-v1 and tweet-eval-stance-benchmark-v1 are a claims contract and a dataset-evaluation contract that sit oddly as `kernel`; the tool-boundary contracts and decide-apr-v1 are further `pattern`/`schema` candidates. Reclassifying any of the twelve is an **owner option, recorded as D-ITEM-08-33-B**, and was NOT done here because reclassifying away from kernel exits the provability invariant they satisfy today.

The one place a contract names a host: forecast-tool-boundary-v1 proof_obligations[20] states "... >= constants.pool_speedup_min_x10 / 10 (evaluated by `just forecast-pool-ratio`, aarch64 release only)". That says where one obligation is evaluated (an evidence note on the pool speed-up), not a restriction of the contract, whose bounds and refusals hold everywhere; no `host_class` qualifier was added. neon-blis-v1 is registry-flagged and got no kind (D-ITEM-08-33-A).

## Hand rewrites (9 of 105; no `moved` entries)

96 entries are `substitution` and reproduce from the 15 recorded unambiguous-operator rules; nine are `hand`, each with its reason in 08-33-FORMAL-REWRITES.json:

1. chronos-bolt-parity-v1 [12]: comma-and joining two equations becomes the conjunction glyph, every other token kept.
2. chronos-bolt-parity-v1 [16]: the English "and" between two behavioural clauses of the second run becomes the conjunction glyph.
3. forecast-tool-boundary-v1 [9]: "and" between additionalProperties == false and the required set becomes the conjunction glyph.
4. neuralprophet-parity-v1 [3]: the spaced ASCII minus becomes the declared minus glyph (U+2212).
5. neuralprophet-parity-v1 [5]: "no Tensor::new on the value path; the mask carries no requires_grad" restated as a negated existence conjoined with a negated requires_grad.
6. neuralprophet-parity-v1 [7]: "length 0 after train() and after a completed forecast()" stated after each event and conjoined.
7. neuralprophet-parity-v1 [9]: "depends on day only through X" restated as an existential function f of day minus X.
8. setfit-train-lifecycle-v1 [6]: "occurs within the loop and not in run_tuning's body" restated as membership and non-membership.
9. tweet-eval-stance-benchmark-v1 [1]: two identifiers asserted identical with the declared identical-to glyph.

Read-through note: neuralprophet-parity-v1 [11] keeps a "; conjunction" seam that the original had as "; AND" (rule-applied, meaning unchanged); all arithmetic in decide-tool-boundary-v1 (800 x 12 + 15000 + 1100 + 4000 = 29700 <= 30000) changed only the `<=` glyph.

## Claim identity and readers

`pv diff` against 3c0f2cf00: identical for 12 contracts; spectral-indices-v1 takes exactly the suggested minor bump (1.0.0 to 1.1.0) and adds only KANI-SPECTRAL-001..004 (DECLARED, NOT EXECUTED, each naming its existing `test_falsify_spectral_00n_*`); the SCHEMA-013 (no qa_gate) warning on it stays, no qa_gate invented. All 133 laya_claims.tsv anchors present, `just laya-claims-check` OK 133 rows, `make contract-audit-phase8` exit 0, audits 2-5 exit 0, audit 6 the same nine BIND-001 equations. Files naming each stem (readers; none reads `formal` or `valid_under`, only `constants`, `equations`, tolerances): see the `git grep -l` lists for chronos-bolt-parity-v1 (aprender-forecast bolt.rs, chronos.rs; aprender-mcp-chronos lib.rs), forecast-tool-boundary-v1 (forecast.rs, np.rs, prophet.rs, types.rs; aprender-mcp-chronos lib.rs), neuralprophet-parity-v1 (events.rs, forecast.rs, np.rs), prophet-parity-v1 (events.rs, forecast.rs, np.rs, prophet.rs, reg_parity.rs, regressors/tracer.rs), setfit-benchmark-claims-v1 (apr-cli setfit_bench.rs, setfit_commands.rs; aprender-core stats/hypothesis.rs; aprender-train bench_gate.rs, bench_metrics.rs, bench_row.rs; scripts/run_bench_cells.sh, gen_claims_fixtures.py), setfit-train-lifecycle-v1 (apr-cli setfit_train.rs, inspect_setfit_tests.rs; aprender-train setfit/*), tweet-eval-stance-benchmark-v1 (aprender-train bench_metrics.rs, bench_metrics_tests.rs, bench_row.rs). Owner crates re-run: aprender-forecast lib 239 passed, 15 ignored (the five cited tests are still tests, ignored with the unarmed reason), aprender-mcp-chronos, aprender-decide, aprender-mcp-decide, aprender-mcp-decide-lambda, apr-format lib green.

## Baseline integrity

`git diff --quiet 2817c6d97` over lint-baseline.json, ontology.yaml, contract_test_binding_baseline.txt, aprender-contracts, aprender-contracts-cli, tests/fixtures/ont, the ratchet and binding scripts, ci/sections.yml and .github/workflows: exit 0. `ci/explicit-test-commands.d` differs by exactly the additions 510 and 520. No baseline raised, no Sigma symbol added, no gate loosened, nothing reclassified. Planted violations (scratch export of HEAD, with untouched controls that stayed green): a prose formal turns the sigma gate red (PV-ONT-004), a kernel without a world turns valid-under red (PV-ONT-016), a top-level `kind: KernelContract` fails pv validate (SCHEMA-018), a metadata.kind change fails `pv extract --check` (exit 1).

## Task Commits

1. Task 1 tracer: `e465421c4` fix(08-33): laya-parity-v1 in the hygiene vocabulary (+ledger, +evidence before-state)
2. Task 2: `84299911e` fix(08-33): Phase 8's decide-apr, decide-tool-boundary, laya-finetune-gate
3. Task 3: `dc963665b` fix(08-33): the seven Phase 3-6 contracts
4. Task 4: `adeeb5111` fix(08-33): neon-blis-v1, spectral-indices-v1, five `#[test]` positions, deferred-items; `01b4d9f8d` fix(08-33): the graph from a clean export; `4f23d0338` and `3b75bfd74` docs(08-33): evidence after-numbers and planted-violation record

`commits: 7` is measured from the plan-head ledger at SUMMARY write time (`git rev-list --count c028b4eec..HEAD`), before this SUMMARY's own commit. Nothing was pushed: `origin/gsd/phase-2-contract-gate` is still 3c0f2cf00.

## Deviations from Plan

**1. [Rule 1 - Bug] contracts.nt generated from a clean clone, not the working tree.** Found during Task 4 (the plan's own untouched-copy control went red). `pv extract contracts` embeds filesystem paths in unresolved-symbol reasons and walks the untracked `.claude/worktrees/agent-*` checkouts on this host: the graph generated in the working tree carried 76 `.claude/worktrees/...` lines that CI's clean checkout cannot reproduce, so `the_tracked_repo_graph_is_fresh` would have failed in CI while passing here. Fix: commit the code first, run `pv extract contracts` over a `git archive` of HEAD, copy `contracts.nt` back (shapes.ttl unchanged, shapes_n 18, zero `.claude` or absolute paths). Consequence to know: in THIS dirty working tree `pv extract --check` and `ont4b_shapes_gate::the_tracked_repo_graph_is_fresh` report a diff by construction; in a clean `git clone` of the branch both pass and `aprender-contracts-cli` reads 394 passed, 0 failed. Commit `01b4d9f8d`. Recorded in the evidence file (`graph_environment_note`).

**2. [Rule 3 - Blocking] two lines of the Task 4 verify adapted.** (a) `pv diff` always prints `Suggested bump: <structural change>` regardless of the version the new file already carries (`diff.rs:21`), so the plan's `! grep 'Suggested bump: minor'` is unsatisfiable while the harness additions exist; the intended condition (the version delta IS the suggested bump: 1.0.0 to 1.1.0 is exactly minor) is what was asserted. (b) The graph-freshness and full `aprender-contracts-cli` steps read their result from a clean clone (deviation 1). Everything else in the payload ran verbatim and exited 0.

**3. [Documented, out of scope] Task 3 verify's aprender-train lib step cannot pass on macOS.** 21 failures, all in `gpu::guard`/`gpu::ledger`/`gpu::wait`, all Linux `/proc` (ledger.rs:77-79); `crates/aprender-train` is byte-identical to the base; 7647 passed including every setfit and tweet-eval reader. Logged as D-ITEM-08-33-E; every other Task 3 assertion passed.

**4. [Choice within plan] Glyph picks.** Implication U+21D2 and if-and-only-if U+2194 (the plan said "the implication glyph" without choosing between two declared ones); kani bounds/strategies for spectral-indices-v1 (8/8/8/16; stub_float, bounded_int) are declarative and chosen by me since the file has no harness to derive them from.

**Total deviations:** 4 (1 bug fixed, 1 blocking, 1 documented out-of-scope, 1 choice). **Impact:** none on claims or baselines; deviation 1 changes where the graph must be regenerated in future (a clean clone), worth knowing for 08-34 and for anyone editing a contract on this host.

## Owner decisions surfaced

- D-ITEM-08-33-A (neon-blis-v1 registry-flagged kernel: leave or upgrade to a proven kernel), D-ITEM-08-33-B (kind alternatives for the two benchmark contracts, the tool-boundary contracts and decide-apr-v1), D-ITEM-08-33-E (macOS-only aprender-train GPU-ledger tests). D-ITEM-08-33-C and -D are reserved for 08-34. D-ITEM-08-32-A and D-ITEM-08-01-A closed; D-ITEM-08-32-B repaired locally (closes when 08-34 observes workspace-test).
- Untracked `.claude/worktrees/agent-*` checkouts pollute any `pv extract` run on this host; they are not this plan's to delete.

## Known Stubs

None. (The kani harnesses in spectral-indices-v1 are honestly declared NOT EXECUTED and claim no proof; that is the branch convention, not a stub.)

## Threat Flags

None: contract YAML annotations and notation, two attribute reorders in test code; no new endpoint, auth path, file access or schema at a trust boundary.

## Next

Ready for 08-34 (CI-order audit, regression composition, push, maintainer approval, CI evidence). 08-34 must regenerate contracts.nt from a clean clone if it edits any contract.

## Self-Check: PASSED

All 16 created or modified files exist; all 7 task commits found; remote-tracking ref of gsd/phase-2-contract-gate still 3c0f2cf00 (nothing pushed).
