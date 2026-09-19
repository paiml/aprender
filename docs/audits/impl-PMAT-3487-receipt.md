# PMAT-3487 — receipt

**Ticket:** PMAT-3487 (issue #3487) — ONT-4: typed relations. **Spec:** paiml/infra `docs/specifications/paiml-ontology.md` v4.5 (sha 34bdc7ea…, on infra main at `204a0314`) §5 ONT-4, §3.1, R-2, R-5, R-6, R-8, R-11, **R-24**. **Kind:** code (`kind-gate.sh PMAT-3487 --base main` → `kind=code files=0`).
**Branch:** `PMAT-3487-ont-4-relations` off `origin/main` `3409b29d2`, in the persistent clone `~/.cache/paiml-implement/wt/aprender-PMAT-3451`; `~/src/aprender` untouched.
**discover.json:** `repo_root` the clone, `default_branch=main`, `required_check=ci / gate,workspace-test`, `gate_cmd=make gate`, `contracts_dir=contracts`.

orch_model: opus-5 [V]   orch_class: opus   orch_decision: admit   orch_basis: file
fable_binding: false   quota_age_h: absent   quota_mark: ?   k_measured_at_set: 0

`model-gate.sh` → `model=opus-5 class=opus decision=admit basis=file` (ticket `kind:code,orch:fable,orch-basis:state`). `estimate.sh aprender 4` → `K_HAT=4 BASIS=first-run[U]` (43 rows excluded — the ledger's unit does not fit; the spec's K̂ 60 is the operative figure). `goal.sh set` refused (one ticket per session).

**Deviation, recorded:** second PR of this session — infra#691 (the ONT-D carrier, merged `7e0bcf23` at 15:11Z) and this. Precedent: session `f3799744` bound ONT-P and filed and bound ONT-0 in one day; the operator's §0.4 ruling asks for speed on the shapes path, and a carrier's quorum and CI are idle time. Turns for this ticket: **64 at this receipt**, measured from the `gh issue create` for #3487.

## Why this row

**R-24.** With ONT-2b bound and v4.5 merged, ONT-4 is the first row of the shapes path and the selector's required pick; the session receipt for the ONT-D carrier records the measurement (every `ONT-*.sh` probe run against main: six rc 0, ONT-D bound by #691, ONT-4 eligible). ONT-4b (the validator, K̂ 90) depends on it.

routes:
  ph1  class=impl           route=self w=100.00 basis=absent   (the gate, from the sigma gate's template)
  ph2  class=impl           route=self w=100.00 basis=absent   (fixtures, CLI test, contract, corpus seeding)
  ph3  class=orchestration  route=self w=100.00 basis=absent
  ph4  class=review         route=agy-quorum w=1.00 basis=absent effort=1[U]

verification:
  cmd=cargo test -p aprender-contracts --lib relations_gate  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=cargo test -p aprender-contracts-cli --test ont4_relations_gate  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=MUTATION the acyclic check skipped (if true { continue }); cargo test … relations_gate  claimed_exit=1  rerun_exit=1  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=MUTATION the symmetric closure dropped (if false); cargo test … relations_gate  claimed_exit=1  rerun_exit=1  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=pv lint contracts/ --gate relations --format json (the ONT-4 probe's predicate)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=pv lint contracts/ --format json (all gates, relations armed)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=cargo test -p aprender-contracts --lib (1588 passed) · cargo test -p aprender-contracts-cli (260 passed)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0
  cmd=make contracts · cargo fmt --check · cargo clippy --all-targets -D warnings (both crates) · cargo deny check advisories  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3487-receipt.md  sha256=0

## What landed

- **`lint/relations_gate.rs`** — five rules over the raw `relations:` block (PV-ONT-005 undeclared role · 006 not Contract→Contract · 007 not a list · 008 dangling · 009 cycle through an acyclic role, named as the closing path) plus the legacy ratchet (PV-ONT-010); `contradicts` materialized both ways; `relations_n == 0` → `NoRelations`, exit 2 (R-2). `pv lint --gate relations`; gate 11 in every `run_lint` (R-8); `NAMED_GATES` = relations · sigma · validate.
- **Σ** gains `supersedes` (acyclic) and `contradicts` (symmetric), both Contract→Contract; `readers.roles` names `lint/relations_gate.rs` beside the sigma gate (R-11).
- **The corpus, measured, not rewritten (R-5):** at `3409b29d` **0** contracts carried `relations:`; **188** carry `metadata.depends_on` — **354 edges, 111 distinct targets, 8 unresolved** after normalizing the `contracts/<x>.yaml` spelling (13 by naive stem match). The 354 are counted as `legacy_depends_on`; the 8 are the ratchet's baseline (`ont.legacy_unresolved_depends_on: 8`). Typed edges on the real corpus: **3** — `ont-relations-v1` `depends_on` [`ont-sigma-v1`, `ont-verdict-lattice-v1`] and `ont-sigma-v1` `depends_on` [`ont-verdict-lattice-v1`], each true of the code. **No `supersedes` or `contradicts` instance exists in the corpus** (no `-v2` beside a `-v1` anywhere); both are exercised on fixtures only, and the receipt says so rather than inventing one.
- **`contracts/ont-relations-v1.yaml`** — the feature's contract (6 invariants, 6 falsification tests); `armed_gates` += `relations` (10); census regenerated (1795 files), README count synced.
- Fixtures `tests/fixtures/ont/relations-{ok,dangling,cycle,domain,malformed,legacy,legacy-rise}`; `tests/ont4_relations_gate.rs` (11 CLI legs over the exit vocabulary 0/1/2/3).

## Three things the tools refused, in order

1. **The sigma gate refused the ONT-4 contract.** `invariants[3].formal` carried `…`, which Σ does not declare (PV-ONT-003). Rewritten as `acyclic(r) ⇒ ¬∃ c ∈ corpus: reach(r, c, c)`. ONT-2b's gate did its job on the next row's own contract.
2. **ONT-2b's `sigma-ok` fixture carried `relations: {binds: [a::symbol]}`** — a contract→symbol edge under a `relations:` key. Under ONT-4 that is PV-ONT-006 (`binds` is Contract→Code); a `relations:` block relates contracts, and a symbol binding is `binding.yaml`'s (ONT-3a). The fixture drops the block; `sigma-ok` is now also the R-2 witness (a corpus with Σ and no typed relation declines).
3. **pmat's pre-commit refused the first commit** — `run_relations_gate` at cyclomatic 30 / cognitive 25 against the threshold; then again at cognitive 30 (a nested DFS `visit`) and 29 (`check_block`). Split into `read_corpus`, `check_block`, `check_targets`, `cycle_sweep`, a top-level `visit`: max cyclomatic 9, max cognitive 17.

Five existing tests pinned the gate count (10) and the armed set; each was updated the way ONT-2b updated them for `sigma` (`mod_tests.rs` ×3, `ont6_lint_verdict.rs` ×2).

## Dogfood (R-23), pv built from this branch at `a60edc16`

  aprender: cmd="pv lint contracts/ --gate relations --format json" pv=a60edc16 exit=0 verdict="Pass — relations_n=3 contracts_with_relations=2 legacy_depends_on=354 legacy_unresolved_depends_on=8"
  rmedia: cmd="pv lint crates/rmedia-core/contracts/ --gate relations --format json" pv=a60edc16 exit=2 verdict="decline: NoCheckable — no contracts/ontology.yaml; 66 contracts at 2f1ced1; tree untouched"

The carrier that binds this row re-runs both from the merged sha and records them under `dogfood:` (R-23); this block is the implementation session's own measurement.

## Gaps

- `--gate relations` and `--gate sigma` both decline with `NoCheckable` when Σ is absent — the reason still does not say *why* (the ONT-D receipt's open item, unchanged).
- ONT-4e (Liskov on `refines`) and ONT-5 (what `contradicts` means to pv-sat) consume these edges; neither is touched here.
