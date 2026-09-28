# impl receipt — PMAT-4070 (ONT-4d, aprender#4070)

- ticket: PMAT-4070 · kind: code · branch: PMAT-4070-ont-4d-subsumption · stacked on PMAT-4071 (ONT-2c) @ 76b0f6e87
  (base for the full diff: origin/main 49fe19c28; ONT-4d's own delta: `git diff 76b0f6e87...HEAD`)
- spec: paiml/infra docs/specifications/paiml-ontology.md @ 948ae923, row ONT-4d (line 619), R-19
- session model: claude-opus-5-5 (model-gate.sh measured)

## Row → where

| row clause | where | evidence |
|---|---|---|
| Σ `subsumes[]` | sigma.rs `Subsumes`, `subsumes` (readers-claimed), `supers()` / `subs()` | contracts/ontology.yaml: `Kernel ⊑ Contract`, `Symbol ⊑ Code` (read off the concepts' own docs) |
| cycle → exit 3 `error: subsumes cycle <path>` | sigma.rs `check_subsumes` + `find_cycle` (integrity, exit 3) | `pv lint tests/fixtures/ont/subsumption-cycle --gate sigma` → rc 3 `error: subsumes cycle Code -> Contract -> Kernel -> Code` |
| extract:pv-contract materializes the type closure | pv_contract types kernel-kind contracts `ont:Kernel` (absent kind = kernel, ContractKind default); `extract::all` → `materialize_type_closure` after every extractor | tracked contracts.nt: +1003 lines; every Kernel node typed Contract, every Symbol node typed Code |
| shapes resolve targetClass through it | shapes.rs selects focus nodes by rdf:type, so the materialized closure is the resolution | `subsumption-inherit`: Code shape focus_nodes_n 1 == Code ∪ subs (0 direct + 1 Kernel), rc 1 |
| sub-shape removing a super constraint → exit 1 `reject: <sub> weakens <super>.<constraint>` | lint/subsumption.rs `weakenings` (PV-ONT-013) | `subsumption-weaken`: rc 1 `reject: kernel-shape weakens contract-shape.name.minCount` (the only violation) |
| rdf:type closure in contracts.nt for every super-concept | tracked contracts.nt | `the_tracked_contracts_nt_carries_the_type_closure` |
| export byte-identical across two runs | fixture extract twice | `a_fixture_extract_carries_the_closure_and_is_byte_identical_twice` |
| OWL writer emits SubClassOf | owl.rs (subsumes → SubClassOf + intended) | contracts/ontology.ofn has both SubClassOf lines; tbox entailed 2, unintended 0 |
| `by_concept` in census | census.rs | contracts/census.json `by_concept` (Code 168 from Symbol only; Kernel 835 ⊂ Contract 1764) |
| probe: `.inherited_shapes_applied>0 and .verdict=="Pass"` | shapes gate `inherited_shapes_applied` / `inherited_by_shape` | repo: Pass, 835, `ont-shapes-v1 <- Kernel=835` |
| mutation: drop closure materialization → Kernel-under-Code fixture stops firing | extract/mod.rs | MUTANT planted: 4 of 8 ont4d tests RED incl. `a_code_shape_rejects_a_kernel_instance_through_the_closure`; restored (git checkout, cmp == snapshot, 0 MUTANT anchors), 8/8 green |

## Honest notes

- On aprender's corpus `inherited_shapes_applied` = 835 counts the Contract shape APPLIED to Kernel instances; those
  nodes are ALSO asserted `ont:Contract` directly (pv_contract keeps it so a Σ-less extraction does not lose it). So
  on the real corpus it demonstrates application, not necessity. The `subsumption-inherit` fixture is where the
  closure is the ONLY path (no contract is typed Code directly) — and is the one the mutation turns RED.
- `Symbol ⊑ Code` adds `ont:Code` to 168 symbols; no shape targets Code in the corpus, so it changes no verdict.
- Σ's `Kernel` concept is the kernel-KIND contract. ONT-4c4's future `ont:Kernel ⊂ ont:Symbol` (cuda-oxide
  `#[kernel]` symbols) would reuse the IRI in a different sense; with this Σ that edge would type kernel contracts as
  symbols. Flagged for ONT-4c4's owner — not decided here.
- GateExtra::Shapes: the ONT-4b2 counters and ONT-4d inheritance moved into a `#[serde(flatten)] Box<ShapesCounters>`
  (JSON keys unchanged; ont4b2 CLI tests still read them) to stay under clippy `large_enum_variant`.
- The worktree lives under the main checkout's `.claude/worktrees/`, so cargo inherits the main checkout's
  `.cargo/config.toml` (`[patch]`, shared target dir): every build added `[[patch.unused]]` to Cargo.lock. Those were
  discarded each time and are NOT in this diff; builds used a private CARGO_TARGET_DIR.

## Measured (lambda, this branch)

- aprender-contracts --lib 1724 passed; aprender-contracts-cli: all targets green (ont4d_subsumption 8, ont2c 9,
  ont4b2 11, ont4b 8, …). clippy `--all-targets -D warnings` clean (both crates; `--lib` alone had hidden a `type_complexity` in the lint tests and a disallowed `unwrap` inside `json!` in the CLI test — fixed at the post-receipt head). fmt clean.
- Guards: complexity ratchet PASS 49fe19c28 vs 4fadd5ed4 (check_subsumes cognitive 28 refactored); tree-reader
  registry 146 == derived; explicit-test-commands PASS (lane 470); guards-wired PASS; include-files OK; roadmap
  sorted + aggregate idempotent; `pv extract contracts --check` fresh; `pv lint --gate tbox` decline Advisory.
- Oracle (ONT-2c's, re-run): kinds 38/38 admitted; ELK agree (consistent, 2 entailed), positive control RED.

verdict: DONE (code) — awaiting quorum; not armed.
