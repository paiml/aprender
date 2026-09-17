# PMAT-3471 — ONT-2b plan v2 (grilled once; replaced by the receipt at PR time)

Spec: paiml/infra `docs/specifications/paiml-ontology.md` v4.4 (sha 87d6d9aeb9f2) §5 ONT-2b, §4.1, §4.2, R-8.
Issue aprender#3471. Base `origin/main` f30c67de3. `depends_on ONT-1, ONT-6` — both merged and BOUND
(ONT-1 `2fb79ff0b`, ONT-6 `2911bdcde`; carrier infra#660 `ac94de5a7`, probe rc 0 on main's ledger).

## The row, verbatim

- RED: undeclared role → exit 1 · undeclared symbol in `formal:` without `prose` → exit 1 · `entity.type` not in Σ
  `entity_types` → exit 1 · an `entity_types` entry naming no extractor → exit 3 · Σ key with no reader → exit 3 ·
  `not_expressible` without `reader` → exit 3 · `extractors[]` entry without `reader` → exit 3.
- Change: Σ per §4.1; symbol-level parser only; `formal_prose` and `unanchored_but_bindable` baselines recorded.
- Mutation: drop a role the corpus uses → RED.
- probe (infra `scripts/ont/done_when/ONT-2b.sh`): `tracked contracts/ontology.yaml && present '^entity_types:' &&
  present '^extractors:' && "$PV" lint contracts/ --gate sigma --format json > s.json && json_object s.json &&
  jq -e '.gate=="sigma" and .verdict=="Pass"' && merged ONT-2b`

## Measured facts (this clone at `f30c67de3`; every row re-measured by the orchestrator after the grill)

| fact | value | why it matters |
|---|---|---|
| contracts censused | 1792 | `census.json.n_files`; **1844 `*.yaml` files exist on disk** — the difference is quarantine + sidecars, and a count that conflates them is a different number (grill lane 2) |
| `by_anchoring` | unanchored 1792, class 0, instance 0 | no contract carries `entity:` — "entity.type ∉ Σ" cannot fire today |
| files with `relations:` | **0** | "undeclared role → exit 1" cannot fire today, and neither can the row's NAMED mutation |
| files with `prose:` | **0** | the opt-out marker does not exist yet |
| files with `formal:` / expressions | **489 / 2330** | the only RED rule with corpus weight |
| `pv lint --gate` | absent | new CLI surface |
| `lint-baseline.json.ont.unanchored_but_bindable` | 297 | already recorded AND ratcheted (`check_ont_ratchet.sh` compares it) |
| ratchet comparison covers | `contracts_anchored`, `contracts_shaped`, `unanchored_but_bindable` only | **`formal_prose` would have no ratchet** unless this row adds one |
| ratchet `entity_types_registered` / `extractors_implemented` | 0 / 0, counted from RUST (`EntityType::X`, `impl Extractor`) | Σ is YAML; the ratchet does not read it |
| `scripts/ont/` in aprender | **absent** | the probes live in **infra**; Phase 5 runs infra's copy with `WT` = this clone |
| `jq -e` on empty stdin | **exit 0** | a `… \| jq -e` acceptance passes when pv never runs — the row's probe already avoids this by writing a file first |

## Rulings — the six open points, decided (grill round 1: 2 FAIL + 1 do-not-implement-as-written, 3/3 non-PASS)

1. **`prose` is an EXPLICIT per-entry marker**, never inferred. An inferred rule cannot fail on a malformed string
   and cannot be shrunk deliberately. Existing expressions are not rewritten by this row: they are COUNTED in the
   `formal_prose` baseline, which is shrink-only. (Grill lane 1 ruled the same.)
2. **Symbol-level means tokens against Σ `symbols[]`, and nothing else.** NOT checked, stated so the boundary is
   falsifiable: grammar, parenthesis balance, arity, types, scope, variable binding, or whether the expression is
   true. A token is an identifier (`isFinite`, `len`, `modifies`, `RoPE`, `Backends::GL`) or a declared operator glyph
   (`∀ ∧ ∈ ⊆ ∩ ∅ ≥ ≈ ‖ ε`). Bare variables matching `[a-z][A-Za-z0-9_]*(_[a-z0-9]+)?` are not symbols.
3. **Σ's `reader` is a declared name carried with `implemented: true|false`**, not a Rust path that must resolve now.
   §4.1 gives `entity_types[{name, extractor, implemented}]` and ONT-4c says Σ "marks these four `implemented: true`
   and the rest `false`", so a Σ that could only name built extractors could not be written until ONT-4c. The exit-3
   rules still bite: an entry naming NO extractor, an `extractors[]`/`not_expressible[]` entry without a `reader`, or
   a Σ key no declared reader claims, are all malformed-Σ errors.
4. **Both vacuous rules get committed positive-control fixtures**, and so does the row's named mutation. A rule that
   cannot fire on the corpus is this fleet's signature defect; the fixtures make "it can fire" a measured claim:
   `tests/fixtures/ont/sigma-undeclared-role.yaml`, `…/sigma-entity-type-not-in-sigma.yaml`,
   `…/sigma-undeclared-symbol.yaml`, and their Σ. The row's mutation is run as written — drop a role the FIXTURE
   corpus uses → RED — and the receipt says plainly that the real corpus uses zero roles today.
5. **`sigma` JOINS `armed_gates` in this PR.** R-8 and §3.5 leave it to the repo ("new gates computed everywhere,
   armed where listed"), and aprender is the repo that authors Σ: an unarmed gate here would report a verdict nothing
   acts on — the shape this fleet keeps refusing. Arming is monotone (ONT-6: dropping it later is exit 3). Cost, named:
   from this PR on, every `pv lint contracts/` — `make contracts`, `dogfood.sh`, CI — fails if a new contract carries an
   undeclared symbol without `prose`. That is the intent. The armed default run must be rc 0 at merge, and A_3 proves it.
6. **ONT-2b does NOT add Rust `EntityType`/`impl Extractor` symbols**, so `check_ont_ratchet.sh` keeps counting 0/0
   (two grill lanes ruled this independently; the third measured the same two grep expressions). Fabricating an empty
   enum to move a counter is the decoration ONT-1 refused. The receipt records the consequence out loud: **the ratchet
   claims to count entity types and extractors while Σ declares them in YAML that it never reads** — a real gap, owned
   here, and the honest place to close it is the row that implements the first extractor (ONT-4b/4c), not this one.

## Phases

| P | Scope | Change | A_i (a command, and it can fail) | Mutation (must go RED) |
|---|---|---|---|---|
| 1 | `crates/aprender-contracts/src/ontology/sigma.rs`, `contracts/ontology.yaml`, `tests/fixtures/ont/sigma-*.yaml` | Σ types + loader per §4.1; the four malformed-Σ classes as one typed `SigmaError` (exit 3) | `cargo test -p aprender-contracts --lib ontology::sigma` | delete one extractor's `reader` → its integrity test RED |
| 2 | `crates/aprender-contracts-cli/src/{cli,lib}.rs`, `commands/lint.rs`, `lint/sigma_gate.rs`, `contract_walk.rs` | `pv lint --gate <name>`: run one named gate, emit `{gate, verdict, …}`, map the exit through ONT-6's lattice (Pass 0 · reject 1 · decline 2 · malformed Σ 3); sigma's corpus rules that are not the symbol check | `cargo test -p aprender-contracts-cli --test ont2b_sigma_gate` | `--gate sigma` mapping a malformed Σ to 1 → the exit-3 test RED |
| 3 | `lint/sigma_symbols.rs`, `contracts/lint-baseline.json`, `scripts/check_ont_ratchet.sh` | the symbol parser + the explicit `prose` opt-out; `formal_prose` MEASURED, recorded **and ratcheted** (the ratchet compares only three counters today, so this row adds the fourth); `sigma` added to `armed_gates` | `cargo test -p aprender-contracts --lib ontology::sigma_symbols && . scripts/pv_bin.sh && "$PV" lint contracts/ --gate sigma --format json > /tmp/s.json && jq -e '.gate=="sigma" and .verdict=="Pass"' /tmp/s.json && "$PV" lint contracts/ >/dev/null; echo rc=$?` (rc 0 with sigma ARMED) | drop from Σ a role the FIXTURE corpus uses → RED (the row's named mutation); raise `formal_prose` by one → ratchet RED |
| 4 | `contracts/ont-sigma-v1.yaml`, `contracts/census.json`, `README.md` | the row's own pv contract; census/README regen (census grows by the new contracts) | `"$PV" validate contracts/ont-sigma-v1.yaml && make contracts && make readme-sync-check` | drop a `kani_harnesses` entry's `obligation` → `pv validate` rc≠0 |
| 5 | infra probe (read-only from here) + receipt + PR | run infra's `scripts/ont/done_when/ONT-2b.sh` with `WT` = this clone, every conjunct except `merged`; write the receipt | `WT=<this clone> LEDGER=<infra main ledger> bash <infra>/scripts/ont/done_when/ONT-2b.sh; echo rc=$?` (1 expected, on `merged` alone, until the carrier lands) | — |

**Phase 5 correction (grill lanes 1 and 2, verified):** `scripts/ont/` does not exist in aprender. The probe is
infra's file, run from the infra checkout against this clone; v1's acceptance command named a path that cannot exist
here, and a command that cannot run cannot fail.

**Phase 3 acceptance correction (grill lane 2, verified: `printf '' | jq -e '.x' ; echo $?` → 0):** every acceptance
that reads pv's JSON writes it to a file first and checks the file, exactly as the row's own probe does. A pipeline
into `jq -e` passes when pv never ran.

## Routing

ph1 impl → `route.sh --phase-class impl` · ph2 impl · ph3 impl · ph4 mechanical · ph5 orchestration (`route=self`).
Phase 3 quorum trigger: Q2 (spec artifact) fired for this plan; the pre-PR diff quorum is the Phase 4 trigger.
