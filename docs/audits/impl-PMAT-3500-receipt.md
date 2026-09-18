# PMAT-3500 — receipt

**Ticket:** PMAT-3500 (issue #3500) — ONT-4b: `pv extract` (pv-contract) → `contracts.nt`; the in-house shapes validator over the SHACL-Core subset; `pv lint --gate shapes` with `pc_shape`. **Spec:** paiml/infra `docs/specifications/paiml-ontology.md` v4.5 §5 ONT-4b (after the split), §3.6, §3.7, §4.1, R-2, R-3, R-13, R-15, R-18, **R-24**. **Kind:** code.
**Branch:** `PMAT-3500-ont-4b-shapes` off `origin/main` `acab2e754` (ONT-4's squash), in the persistent clone; `~/src/aprender` untouched.
**discover.json:** `repo_root` the clone, `default_branch=main`, `required_check=ci / gate,workspace-test`, `gate_cmd=make gate`.

orch_model: opus-5 [V]   orch_class: opus   orch_decision: admit   orch_basis: file
fable_binding: false   quota_age_h: absent   quota_mark: ?   k_measured_at_set: 0

`kind-gate.sh PMAT-3500 --base main` → `kind=code files=0` (after `make roadmap-aggregate`: pmat 3.41.1 writes the fragment, not the aggregate); `model-gate.sh` → `model=opus-5 class=opus decision=admit basis=file`. `goal.sh set` refused (one ticket per session). Turns for this ticket: **54 at this receipt**, from the `gh issue create` for #3500.

**Why this row:** R-24 — ONT-4 was bound by infra#696 (`6262a7ab`) during this session, ONT-4b became eligible, and it is the row the operator's §0.4 ruling names: the first usable shapes gate.

routes:
  ph1  class=impl           route=self w=100.00 basis=absent   (rdf.rs, extract/pv_contract.rs, shapes.rs — from the spec's tables)
  ph2  class=impl           route=self w=100.00 basis=absent   (shapes_gate.rs, pv extract, --gate shapes, the first shape, fixtures, CLI test)
  ph3  class=orchestration  route=self w=100.00 basis=absent
  ph4  class=review         route=agy-quorum w=1.00 basis=absent effort=1[U]

verification:
  cmd=cargo test -p aprender-contracts --lib 'ontology::' (rdf 5, extract 2, shapes 10) and lint::shapes_gate (5)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=cargo test -p aprender-contracts-cli --test ont4b_shapes_gate (11)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=MUTATION minCount removed from ont:id in contracts/ont-shapes-v1.yaml; pv lint contracts/ --gate shapes  claimed_exit=2  rerun_exit=2  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=MUTATION metadata.kind: novel-kind on ont-relations-v1; pv lint contracts/ --gate shapes  claimed_exit=1  rerun_exit=1  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=the spec's ONT-4b probe text, every conjunct but `merged`  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=pv extract contracts --check (R-18: tracked == fresh)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=pv lint contracts/ --format json (12 gates, shapes armed)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0
  cmd=lib 1610 · cli 271 · clippy -D warnings both crates · make contracts · check_ont_ratchet · check_baseline_ratchets · check_tree_reader_tests · check_roadmap_fragment_required · check_tool_versions  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3500-receipt.md  sha256=0

## What landed

- **`ontology/rdf.rs`** — `Graph` (a `BTreeSet<Triple>`), `Term` with no blank-node variant by construction, N-Triples in byte order, IRIs `https://ont.paiml.dev/v1alpha1/{contract,…}/<id>` percent-encoded (R-15).
- **`ontology/extract/pv_contract.rs`** — every contract → `ont:Contract` + `prov:Entity`, `ont:id` (the stem), `ont:file`, `ont:kind`, `ont:name`/`version`/`status`, `ont:evidenceLevel`, `ont:entityType`/`entityRef`, and ONT-4's relations as `ont:<role>` edges. Nothing inferred: a key the contract lacks yields no triple, so a `minCount` over it is a constraint, not a tautology. Σ: `pv-contract` and `pv_contract` are `implemented: true`.
- **`ontology/shapes.rs`** — §3.6's subset **exactly**: `targetClass`, `minCount`/`maxCount`, `datatype`/`class`/`nodeKind`, `in`/`pattern`/`minLength`/`maxLength`, `node` (one level), `closed`/`ignoredProperties`, single-predicate paths; `targetNode`, the `qualifiedValueShape` family, `languageIn`, sequence/alternative paths, `or`/`and`/`not`, a second `node` level and **any key not in the table** are refused at parse by name — a component ignored is a shape that reports conforms for something it never checked. The `sh:ValidationReport`-shaped result names focus node, shape, path and component; `severity: warning` per property; a Turtle export for the oracle.
- **`lint/shapes_gate.rs`** — gate 12 in every `pv lint` (R-8); `pv lint --gate shapes`; the exit vocabulary (Pass 0 · reject 1 naming focus and shape · `NoShapes`/`NoFocus`/`PositiveControlFailed`/`Warn` 2 · unsupported 3). **The plant:** a bare `ont:Contract` node every run must draw ≥1 violation — exactly 1 on this corpus, from `ont:id minCount` — else `Unknown{PositiveControlFailed}`. The single-gate JSON now carries the gate's fields at the top level beside `verdict`, which is where the spec's probes read them.
- **`pv extract [dir] [--check]`** — writes `contracts/contracts.nt` (9231 triples, 1.5 MB, sha over the bytes) and `contracts/shapes.ttl`; `--check` exits 1 on drift; `make contracts` runs it (R-18).
- **`contracts/ont-shapes-v1.yaml`** — the row's contract AND the first shape: `entity: {type: pv-contract}`, so its `shape:` targets `ont:Contract` and **all 1731 contracts are focus nodes on day one**. It constrains `ont:id` (min 1, max 1, pattern) and `ont:kind` (`in` the 13 measured values) — and deliberately NOT `name`/`status`/`version` (absent from 1156/931/841 contracts: a `minCount` there is a migration by refusal) nor `ont:file maxCount 1` (55 stems are shared with a subdirectory; that is the duplicate-stems gate's baselined finding, not this gate's to report twice). `armed_gates` += `shapes` (11); `ont.contracts_shaped` 1.
- Fixtures `tests/fixtures/ont/shapes-{ok,violation,warn,nofocus,unsupported,noplant}`; `tests/ont4b_shapes_gate.rs` (11 legs); census 1796, README synced; the tree-reader registry.

## Two things measured that the spec did not state

1. **The spec's probe reads `.shapes_n` at the top level of the single-gate report**, where `.gate` and `.verdict` live; ONT-2b's report put per-gate fields under `.extra`. The report now carries both (a serde flatten; one `type` key rides along), and the spec's ONT-4b probe text passes verbatim minus `merged` — measured, not assumed, because ONT-4's probe would have passed either way and this one would not.
2. **55 stems are shared between `contracts/` and a subdirectory** (`contracts/work/…` mostly): one subject, two `ont:file` values. That is what the duplicate-stems gate already baselines; the first shape leaves `ont:file` unconstrained so the two gates do not report one condition twice.

## Deviations, named

- Two `impl` phases routed `self`, not `agy-goal`: the spec's §3.6 table is the specification of `shapes.rs` line by line, and the day's lane record (a lane deleted a tracked line, a lane launched the 50-minute suite) is in the ONT-4 receipt.
- The pmat pre-commit complexity gate refused nothing this time because the functions were split before the first commit (max cognitive 19, after `check_value`/`turtle_node`/`parse_property` went over on first measurement).
- Fifth PR of the session on the ONT program. Precedent and reason as before; the operator's stated priority is this row.

## Dogfood (R-23), pv built from this branch at `edd7f90a`

  aprender: cmd="pv lint contracts/ --gate shapes --format json" pv=edd7f90a exit=0 verdict="Pass — shapes_n=1 focus_nodes_n=1731 pc_shape=fired plant_violations=1 violations=0"
  rmedia: cmd="pv lint crates/rmedia-core/contracts/ --gate shapes --format json" pv=edd7f90a exit=2 verdict="decline: NoShapes — 66 contracts at 2f1ced1, none carries shape:; tree untouched"

The carrier re-runs both from the squash sha. rmedia's first `shape:` block is one `entity` line and two properties away from a real verdict there — that is the row R-23 files after its third decline, and the reason the gate exists.

## Gaps

- ONT-4b2: the oracle differential and the vendored W3C cases; `code`/`lean` extraction. ONT-4c: the first document contracts. ONT-4d: inheritance down Σ's `subsumes`.
- The `in` list on `ont:kind` is a measured snapshot (13 values at `acab2e754`); a new kind is a one-line change to the shape, and the gate will name the contract that needs it.
