# PMAT-3704 — receipt

**Ticket:** PMAT-3704 (issue #3704) — ONT-4c3: pc_extract draws the parity-receipt positive control every gate run, and every extractor the gate runs has a control. **Kind:** code (`kind:code`).
**Branch:** `PMAT-3704-ont-4c3-pc-extract` from `origin/main` at `225b2a9ab` (release 0.69.0, #3698), in the worktree `~/.cache/paiml-implement/wt/aprender-ont4c3`; `~/src/aprender` untouched.
**Spec:** `paiml/infra docs/specifications/paiml-ontology.md` v4.10 — §2 R-3 (*"`pc_extract` (one planted defect per registered extractor, every run)"*) and §5 ONT-4c3, whose probe asks `.pc_extract["parity-receipt"]=="fired"`.
**discover.json:** `repo_root` the worktree, `default_branch=main`, `required_check=ci / gate,workspace-test`, `gate_cmd=make gate`, `quorum_tool=agy`, `code_search=pmat query`, `gate_cmd_fallback=false`.

orch_model: opus-5 [V]   orch_class: opus   orch_decision: admit   orch_basis: file

Phase 0: `kind-gate.sh PMAT-3704 docs/roadmaps/roadmap.yaml --base origin/main` → `kind=code ticket=PMAT-3704 files=0`; `model-gate.sh` → `model=opus-5 class=opus decision=admit basis=file`; `config-lint.sh` → `slots=3 gh_calls_per_min=30 bank=3`; `target-guard.sh` PASS.

## How this ticket was found

The session was launched to implement ONT-001. R-24's next row was ONT-4c3, and its implementation (#3600) had landed in the 0.69 batch (#3669, squash `a877fa056`), so the next unit looked like the infra ledger carrier. The carrier's own first check refused it: with `pv` built at `a877fa056`, the gate on `contracts/` reports `verdict=Pass`, `by_entity_type["parity-receipt"]=7` and `pc_extract={apr-model, code, gguf, lean}` — no `parity-receipt` key. `extract_controls()` is byte-identical on #3600's head `242bdc3b5`, on `a877fa056` and on `225b2a9ab`: the batch did not drop the control, it was never wired. `parity_receipt::positive_control` had one caller, `parity_receipt_tests.rs`.

## Five whys

1. The probe is RED → `pc_extract` has no `parity-receipt` key.
2. → `extract_controls()` does not call `parity_receipt::positive_control`.
3. → the control list is a hand-written array beside Σ, and adding an extractor does not force adding its control. Measured: Σ implements **seven** entity types (`pv-contract`, `json`, `code`, `lean`, `apr-model`, `gguf`, `parity-receipt`); the array had **four**. `json` (#3516) shipped without one too.
4. → no test compared the two sets — the shape of #3624 (`by_entity_type`, one map over) and of bashrs#266 (two lists, nothing tying them).
5. → so the fix is the tie, not the missing entry: a test that makes the key set EQUAL Σ's implemented entity types, and the three controls it then demands.

## Plan and routing

| phase | what | route | A_i |
|---|---|---|---|
| ph1 | RED: the Σ-tie lib test + the CLI mirror of the probe | `route=self` (direct) | both tests red on main: 4 keys vs 7; `pc_extract` has no `parity-receipt` |
| ph2 | GREEN: three controls, wired | `route=agy-goal w=1.00 basis=absent` → **refused at spawn**, fell back to direct | both tests green unedited; gate `Pass`, 7/7 fired |
| ph3 | mutation | `route=self` | four mutants, each RED (below) |
| ph4 | gate, receipt, diff quorum, PR | `route=agy-quorum w=1.00 basis=absent` (review); push/PR `route=self` | `make gate` rc 0 on the committed tree; quorum 3/3 |

Trigger: Q1 (`|M|=4`: `lint/shapes_gate.rs`, `ontology/extract/{parity_receipt,json,pv_contract}.rs`).

## Dispatch ledger

- **ph2.goal** (`paiml-agy-delegate`, goal lane, writes=true) and **ph1.grill** (`paiml-agy-delegate`, grillme quorum width 3) — both **refused at spawn** by the `SubagentStart` gate: `kind-gate refused PMAT-3704 (exit 2) — … not filed in /home/noah/src/infra/docs/roadmaps/roadmap.yaml`. The hook resolves the roadmap from the SESSION cwd (`~/src/infra`, where ONT-001 is launched), not from the ticket's repository. Not retried. Filed as **paiml/paiml-implement#326** with the repro.
- Consequence: ph2 was implemented direct, and the ph1 plan grill is **NotRun** — the diff quorum in ph4 is the review of record.
- slots used: 0 running; attempted=2 denied=2 running_peak=0 slots=3.
- `pmat hooks install --strict --force` was NOT run: aprender's hooks directory is the common git dir shared by every aprender worktree and session. Every commit on this branch carries `Pmat-Ticket: PMAT-3704` by hand.

## What landed

- **`lint/shapes_gate.rs`** — `extract_controls()` carries seven controls keyed by entity type, and the module doc names what each plants. New test `every_implemented_entity_type_in_sigma_has_an_extract_control_and_it_fires`: reads `contracts/ontology.yaml` through `Sigma::from_yaml`, asserts the implemented set is non-empty (a vacuous comparison is refused), that the key set EQUALS it, and that every value is `fired`.
- **`ontology/extract/parity_receipt.rs`** — `control_sample()`: a minimal `apr-parity-receipt/v2` record from a literal plus `SCHEMA` (inserted, so the two cannot drift). It fails closed: a literal that did not parse is `Null`, and the control cannot fire on `Null`. Built without `serde_json::json!`, whose expansion calls `to_value(..).unwrap()` and trips the crate's `disallowed_methods` (14 clippy errors on the first draft).
- **`ontology/extract/json.rs`** — `extract_text()`: the post-read body of `extract_into`, factored so the control runs the gate's code rather than a copy. `positive_control()`, in memory: a mapped nested key extracts typed by its class, and an unmapped one is refused as `Unmapped { key: "orphan" }`.
- **`ontology/extract/pv_contract.rs`** — `positive_control()`: `metadata.kind` and a `depends_on` relation come out as `ont:kind` and the typed edge, and the copy without `metadata` carries no `ont:kind`.
- **`crates/aprender-contracts-cli/tests/ont4c3_parity_receipts.rs`** — `the_parity_extractor_control_is_drawn_by_the_gate_every_run`: the probe's question, at the top level of the report, on the repo's own `contracts/`.
- Unit tests: `json::the_positive_control_fires`, `pv_contract::the_positive_control_fires`, `parity_receipt::the_gate_sample_is_a_record_the_control_fires_on`, and `parity_receipt::a_sample_without_a_comparator_cannot_fire_the_control` (discrimination: a sample with no comparator edge to lose must not fire).

## Verification (claimed = what the tree said when written; rerun = the orchestrator's own run)

| cmd | claimed | rerun | note |
|---|---|---|---|
| `cargo test -p aprender-contracts --lib lint::shapes_gate::tests::every_implemented…` on `d542db3cd` (RED) | 101 | 101 | `left: {apr-model, code, gguf, lean}` · `right: {apr-model, code, gguf, json, lean, parity-receipt, pv-contract}` |
| `cargo test -p aprender-contracts-cli --test ont4c3_parity_receipts the_parity_extractor_control…` on `d542db3cd` (RED) | 101 | 101 | `pc_extract carries no fired parity-receipt control` |
| the same two on `9f6b2e02c` (GREEN) | 0 | 0 | unedited |
| `cargo test -p aprender-contracts --lib ontology::extract` | 0 | 0 | 49 passed |
| `cargo test -p aprender-contracts --lib lint::shapes_gate` | 0 | 0 | 15 passed |
| `cargo test -p aprender-contracts-cli` over every `tests/ont*.rs` (11 files) | 0 | 0 | 8+11+8+11+12+10+7+6+6+7+9 passed |
| `cargo clippy -p aprender-contracts --all-targets -- -D warnings` · same for `-cli` | 0 | 0 | after the `json!` rewrite above |
| `cargo fmt --all -- --check` | 0 | 0 | no diff |
| `pmat analyze complexity` on the new functions | — | — | `extract_controls` 2/1 · `pv_contract::positive_control` 4/3 · `parity_receipt::positive_control` 3/2 · `control_sample` 2/1 · `json::extract_text` 2/3 · `json::positive_control` 3/2 (cyclomatic/cognitive) |
| `bash scripts/check_ont_ratchet.sh` | 0 | 0 | PASS |
| `pv lint contracts/ --gate shapes --format json` (pv from this tree) | 0 | 0 | `verdict=Pass`, `pc_extract` 7/7 `fired`, `pc_shape=fired`, `by_entity_type["parity-receipt"]=7` |
| the ONT-4c3 probe's gate conjunct: `jq -e '.verdict=="Pass" and .by_entity_type["parity-receipt"]==7 and .pc_extract["parity-receipt"]=="fired"'` | 0 | 0 | was 1 at `a877fa056` and at `225b2a9ab` |

## Mutations (each applied with a one-match assertion, then `git checkout`)

| mutant | Σ-tie test | gate on `contracts/` |
|---|---|---|
| M1 — `parity_receipt::emit` drops the comparator edge | 101 | exit 2, `positive control pc_extract.parity-receipt did not fire` · `decline: PositiveControlFailed` |
| M2 — `json::nested_class` infers a class for an unmapped key | 101 | exit 2, `positive control pc_extract.json did not fire` |
| M3 — `pv_contract::extract_one` defaults `metadata.kind` to `kernel` (the inference ONT-6b made visible) | 101 | exit 2, `positive control pc_extract.pv-contract did not fire` |
| M4 — the `json` entry removed from `extract_controls()` | 101 on the assertion (`left` lacks `json`), not a compile error | — |

Each gate decline names the control that was broken, and no other. After the four, the tree is clean and the rebuilt `pv` reports `Pass` with 7/7.

## ont-delta

`none — adds pc_extract positive controls for three already-implemented extractors (pv-contract, json, parity-receipt); no entity type, shape, relation or verdict reason is added or changed.`

## Estimates

K̂ = 64 (`estimate.sh aprender 4`, basis `docs/audits/impl-estimates.jsonl:L49-L52`), K = 80 (`1.25 K̂`, ONT-001 §6).

## Gaps

- **ph1 plan grill: NotRun** — refused at spawn (paiml-implement#326). The ph4 diff quorum is the review of record.
- The ONT-4c3 ledger binding is paiml/infra's row, written after this merges, from a `pv` built at this PR's squash sha (R-23).
- #3624 (`by_entity_type` is a hand-written list parallel to Σ) is the same class, one map over, and is NOT closed here: `by_entity_type` also has no `json` key today.

## Verdict

`PARTIAL(open)` until merged green on `ci / gate` + `workspace-test`.
