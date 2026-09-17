# PMAT-3471 — ONT-2b plan v1 (for the plan grill; replaced by the receipt at PR time)

Spec: paiml/infra `docs/specifications/paiml-ontology.md` v4.4 (sha 87d6d9aeb9f2) §5 ONT-2b, §4.1, §4.2.
Issue aprender#3471. Base `origin/main` f30c67de3. `depends_on ONT-1, ONT-6` — both merged and BOUND
(ONT-1 `2fb79ff0b`, ONT-6 `2911bdcde`; carrier infra#660 `ac94de5a7`, probe rc 0 on main's ledger).

## The row, verbatim

- RED: undeclared role → exit 1 · undeclared symbol in `formal:` without `prose` → exit 1 · `entity.type` not in Σ
  `entity_types` → exit 1 · an `entity_types` entry naming no extractor → exit 3 · Σ key with no reader → exit 3 ·
  `not_expressible` without `reader` → exit 3 · `extractors[]` entry without `reader` → exit 3.
- Change: Σ per §4.1; symbol-level parser only; `formal_prose` and `unanchored_but_bindable` baselines recorded.
- Mutation: drop a role the corpus uses → RED.
- probe: `tracked contracts/ontology.yaml && present '^entity_types:' && present '^extractors:' &&
  "$PV" lint contracts/ --gate sigma --format json > s.json && json_object s.json &&
  jq -e '.gate=="sigma" and .verdict=="Pass"' && merged ONT-2b`

## Measured facts (main `2911bdcde` / `f30c67de3`, this clone)

| fact | value | why it matters |
|---|---|---|
| contracts | 1792 | the corpus the gate must pass over |
| `by_anchoring` | unanchored 1792, class 0, instance 0 | **no contract carries `entity:`** — "entity.type not in Σ" is vacuous today |
| files with `relations:` | **0** | "undeclared role → exit 1" is vacuous today |
| files with `prose:` | **0** | the opt-out marker does not exist yet |
| files with `formal:` | **489** | |
| `formal:` expressions | **2330** | the only RED rule with real corpus weight |
| `pv lint --gate` | **absent** | new CLI surface: a single named gate emitting `{gate, verdict, …}` |
| `lint-baseline.json.ont.unanchored_but_bindable` | **297** | already ratcheted by `check_ont_ratchet.sh` |
| ratchet `entity_types_registered` / `extractors_implemented` | **0 / 0** | both count RUST symbols (`EntityType::X`, `impl Extractor`) under `src/ontology/`, **not** Σ's YAML |
| gate names today | validate, audit, score, verify, enforce, enforcement-level, reverse-coverage, duplicate-stems, composition, strict-test-binding | `sigma` is the 11th |

## Phases

| P | Scope | Change | A_i (a command) | Mutation (must go RED) |
|---|---|---|---|---|
| 1 | `crates/aprender-contracts/src/ontology/sigma.rs`, `contracts/ontology.yaml` | Σ types + loader per §4.1 (`concepts`, `roles{domain,range,symmetric?,acyclic?}`, `symbols[]`, `worlds`, `agents`, `entity_types[{name,extractor,implemented}]`, `extractors[{name,reader}]`, `not_expressible[{key,reader}]`); Σ-integrity errors, all exit 3: entity_type naming no extractor, Σ key with no reader, `not_expressible`/`extractors[]` without `reader` | `cargo test -p aprender-contracts --lib ontology::sigma` | delete the `reader` of one extractor → its integrity test RED |
| 2 | `crates/aprender-contracts-cli/src/{cli,lib}.rs`, `commands/lint.rs`, `lint/sigma_gate.rs` | `pv lint --gate <name>`: run one named gate, emit `{gate, verdict, …}` and map the exit through ONT-6's lattice (Pass 0 · reject 1 · decline 2 · `SigmaMalformed` 3); the `sigma` gate runs the corpus rules that are not the symbol check (`entity.type ∉ Σ`, undeclared role) | `cargo test -p aprender-contracts-cli --test ont2b_sigma_gate` | `--gate sigma` mapping a malformed Σ to exit 1 → the exit-3 test RED |
| 3 | `lint/sigma_symbols.rs`, `contracts/lint-baseline.json` | symbol-level parser over `formal:` (symbols only — never arity, types or well-formedness); undeclared symbol without the `prose` opt-out → reject; `formal_prose` baseline recorded and shrink-only | `cargo test -p aprender-contracts --lib ontology::sigma_symbols && "$PV" lint contracts/ --gate sigma --format json \| jq -e '.gate=="sigma" and .verdict=="Pass"'` | drop a role/symbol the corpus uses from Σ → the gate rejects (the row's named mutation) |
| 4 | `contracts/ont-sigma-v1.yaml`, `contracts/census.json`, `README.md` | the row's own pv contract (kernel kind) + census/README regen if counts move | `"$PV" validate contracts/ont-sigma-v1.yaml && make contracts && make readme-sync-check` | drop an obligation's `formal` → `pv validate` rc≠0 |
| 5 | infra probe + receipts | run `scripts/ont/done_when/ONT-2b.sh` conjuncts except `merged`; receipt; PR | each conjunct rc 0 in this clone | — |

## Open points for the grill (decide these, do not assume)

1. **What makes an expression exempt.** The RED line says "undeclared symbol in `formal:` **without `prose`**". Is `prose`
   an explicit per-entry key the author writes (and the `formal_prose` baseline counts), or is "prose" inferred? An
   inferred rule cannot be shrunk deliberately; an explicit one needs 2330 expressions triaged before the gate can pass.
   **Consequence either way:** the probe demands `verdict == "Pass"` on today's corpus, so whatever the rule is, the
   corpus must satisfy it on day one without the pass being vacuous.
2. **Symbol granularity, stated as what is NOT checked.** "symbol-level parser only": identifiers and operators
   declared in Σ `symbols[]`, with no arity, type, scope or well-formedness check. Where exactly does the line fall for
   `isFinite(x_i)`, `‖RoPE(x, m)‖`, `Backends::GL`, `d mod 2 = 0`, `∀i:`? Which of those are symbols, which are noise?
3. **Bootstrapping "Σ key with no reader → exit 3".** Every Σ key must name a reader, and today `sigma.rs` is the only
   reader that will exist. Does `reader` name a Rust path that must resolve NOW (fail-closed, and then most of Σ cannot
   be declared until ONT-4b/4c implement the extractors), or a declared-and-ratcheted name with `implemented: false`
   (§4.1's own shape, and ONT-4c's "Σ `entity_types` marks these four `implemented: true` and the rest `false`")?
4. **Two of the four exit-1 rules are vacuous on today's corpus** (0 contracts with `entity:`, 0 with `relations:`).
   A rule that cannot fire is the fleet's signature defect. Does each vacuous rule need a committed fixture that makes
   it fire (a positive control), and does the row need to say so in the receipt?
5. **Is `sigma` armed?** ONT-6 §3.9: only gates in `lint-baseline.json.armed_gates` enter the repo's meet. If `sigma`
   is armed, every `pv lint contracts/` must pass it and the arming is monotone (dropping it later is exit 3). If it is
   not armed, `--gate sigma` still reports, but a failure does not fail a default run. The probe only exercises
   `--gate sigma` explicitly, so the row does not force the answer.
6. **The ratchet counts Rust, Σ declares YAML.** `entity_types_registered` counts `EntityType::X` in `src/ontology/`
   and `extractors_implemented` counts `impl Extractor` — neither reads `contracts/ontology.yaml`. Does ONT-2b add the
   Rust enum/trait (moving counters 0 → N and making the ratchet meaningful), or leave both at 0 and note that Σ's
   declarations are not yet counted by the ratchet that claims to count them?
