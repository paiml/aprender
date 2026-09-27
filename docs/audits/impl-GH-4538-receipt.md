---
status: complete
ticket: GH-4538
github_issue: 4538
part: "Branch 6b/4538-sigma-crate-contracts on origin/main @aca6f2d7f6. Post-0.70 (off the car), coordinated with infra-83."
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: GH-4538, Σ walks crates/*/contracts

## READ FIRST

Ruling (infra-ont, 10:20Z): Σ walks `crates/*/contracts`. A stem is emitted only if it is unique or byte-identical;
otherwise it is refused by name (PV-DUP-001) and never unioned. infra-83 confirmed the design and added a caveat:
"a binding whose file names a dropped crate copy must be refused by name, never quietly resolved to the top-level
node with different content. Add a fixture for that case."

| file | change |
|---|---|
| `ontology/extract/pv_contract.rs` | `corpus()`: top-level `contracts/` is taken whole, as before. Then every `crates/<c>/contracts` whose crate has a `Cargo.toml` (the manifest-less staging tree is excluded) is added. A crate stem is admitted when it brings no bytes the other copies lack, and a byte-identical copy is skipped. A stem whose crate copies add a distinct content is refused (`RefusedStem {stem, paths}`), and its crate copies are dropped. `documents()`, the one walk every extractor shares, reads `corpus().files` |
| `ontology/extract/code.rs` | `registries()` also walks the crate contract dirs. A binding whose contract resolves (relative to its registry file) to a refused copy is refused by name into `CodeStats::refused_bindings`, with no symbol edge. A binding's contract IRI is now its file stem: 5 edges from `contracts/aprender/binding.yaml` named `../<stem>.yaml` and pointed at `contract/..%2F<stem>`, which no node carries. They now bind the existing top-level node |
| `ontology/extract/mod.rs`, `commands/extract_rdf.rs` | `Extraction::refused`. `pv extract` prints `refused: PV-DUP-001 …` for each refused stem and each refused binding |
| `lint/duplicate_stems.rs`, `lint/mod.rs` | Gate 8 judges two ratchets: the top-level divergent stems as before, and (new) `scan_crate_refusals` (the same `corpus()` walk, so the gate and the graph cannot disagree) against `scripts/contract_crate_stem_refused_baseline.txt` |
| `scripts/contract_crate_stem_refused_baseline.txt` | new shrink-only baseline, 17 stems (33 at 0cfb192a1b, 16 pruned by e4bed01811, see below). `check_baseline_ratchets.sh` classifies it `set` |

What this does NOT do:
- `legacy/` and `kaizen/` stay excluded, as the ruling scopes `crates/*/contracts` only.
- The interim `ont:OutOfCensusContract` class was never landed (0 hits), so there is nothing to delete.

## Effect on the tracked graph

`contracts/contracts.nt`: +295 / −12. 59 crate contracts are admitted and none are removed. The 12 removed lines are the
`..%2F` edges, rewritten to stem IRIs. Before: 5 symbol edges pointed at contract IRIs with no `ont:Contract` node.
After: 0 (measured by a script over both files). `shapes.ttl` is unchanged. The shapes gate passes on the admitted contracts.

## Acceptance

| criterion | state | proof |
|---|---|---|
| Σ walks `crates/*/contracts` (crates only) | MET | `crate_corpus_tests::a_unique_crate_contract_is_admitted_with_its_crate_path`, `a_directory_without_a_manifest_is_not_a_crate` |
| Unique or byte-identical only; top level authoritative | MET | `an_identical_crate_copy_is_skipped_and_the_top_level_file_kept` |
| Differing copy refused by name, never unioned | MET | `a_differing_crate_copy_is_refused_by_name_and_not_unioned`: the refusal names both paths, the graph has no triple from the crate copy, and `scan_crate_refusals` reports 2 variants. `a_crate_only_stem_with_two_contents_is_refused_whole` |
| Binding naming a refused copy refused by name (infra-83) | MET | `a_binding_that_names_a_refused_crate_copy_is_refused_by_name_not_resolved_to_the_top_level_node`: no edge to `contract/diff-v1`; the admitted binding and a `../../../contracts/top-v1.yaml` path binding are kept |
| Refusals reported and ratcheted | MET | `pv lint` Gate 8: `81 ambiguous stems, 81 baselined, 0 unbaselined, 0 stale` (48 top-level + 33 crate) at 0cfb192a1b; `65 … 65 baselined, 0 stale` (48 + 17) at e4bed01811 |

### Mutation, measured 2026-09-27 on a detached worktree at 0cfb192a1b

The runner is `/mnt/nvme-raid0/tmp/6b-4538/mut.sh`. It runs `cargo test -p aprender-contracts --lib crate_corpus_tests`:

```
M0 unmutated                                           rc 0    6 passed
M1 refusal condition → `false && …` (union instead)    rc 101  3 FAILED (differing_copy, binding, crate_only_stem)
M2 binding refusal lookup → `false && …`               rc 101  1 FAILED (binding test, crate_corpus_tests.rs:139)
M3 Cargo.toml requirement dropped (staging = crate)    rc 101  1 FAILED (a_directory_without_a_manifest_is_not_a_crate)
worktree after restore: git status --porcelain empty
```

## Measured at 0cfb192a1b (private target `/mnt/nvme-raid0/cargo-targets/6b-4538`)

```
cargo fmt --all -- --check                                                rc 0
cargo test -p aprender-contracts --lib                                    rc 0  (1727 passed, 5 ignored)
cargo test -p aprender-contracts-cli                                      rc 0  (incl. the_tracked_repo_graph_is_fresh)
cargo clippy -p aprender-contracts -p aprender-contracts-cli --lib --tests -- -D warnings   rc 0
pv extract contracts --check                                              rc 0
pv lint contracts                                                         rc 0
scripts/check_baseline_ratchets.sh                                        rc 1: the ONLY FAIL row is
    `shell_lint_baseline.txt recorded under bashrs 7.4.1, runner has bashrs 7.4.2` — a host tool-version
    row this diff does not touch. The new baseline reports NOT ARMED (introduced by this commit), as designed.
```

## Census on the same walk (e4bed01811, infra-83's class gap)

infra-83 flagged that Σ would admit crate contracts while `pv census` walked `contracts/` only: two walks and nothing
tying them. Now `census.rs::census_files` takes the top level plus `pv_contract::corpus(dir).files`, so both sides
count one corpus.

Measured first: that census rejected 21 parse errors, crate YAMLs like `crates/aprender-compute/contracts/cgp/*.yaml`
("missing field `metadata`"). Σ had admitted them because it parses raw YAML. The fix: Σ's crate admission now
requires a typed `Contract` parse. 38 crate-local files are not contracts (model-families, cgp kernel sheets, crate
`binding.yaml`). They are listed by name in `Corpus::unparsed` and in `pv extract` stderr (`skipped: 38 …`), and
counted by neither side. 16 crate stems had conflicted only through those non-contracts, so the ratchet dropped
33 → 17. contracts.nt: −125 lines. census.json and README were regenerated.

```
M0 unmutated                                               rc 0
M1 census_files returns the top level only                 rc 101  the_census_counts_the_crate_contracts_sigma_admits_and_not_the_ones_it_refuses FAILED
M2 Σ admits untyped crate files (typed = true)             rc 101  a_crate_yaml_that_is_not_a_typed_contract_is_skipped_by_name_not_admitted FAILED
worktree after restore: git status --porcelain empty
```

Gates at e4bed01811: fmt 0, `pv extract contracts` 0, `pv census` 0, readme_sync --write/--check 0/0,
aprender-contracts-cli tests 0, clippy (both crates) 0, `extract --check` 0, `pv lint` 0 (Gate 8 ✓),
aprender-contracts lib 0 (1728 passed), readme_contract 0, check_baseline_ratchets rc 1 (only the same bashrs 7.4.1/7.4.2 host row).

Not done here: the crate `binding.yaml` `../X` path handling overlaps 0d's #3559 (out-of-census rows). It was
coordinated, not changed.

## Quorum round 1 → fix

Round 1 (on 0cfb192a1b) returned sonnet FAIL, sonnet FAIL, haiku PASS. Both FAILs make the same claim: the
refused-binding lookup compared `dir.join(b.contract)` against the refused paths by `Path` equality, which works
component by component and does not resolve anything. A binding that reaches a refused copy through `../` (for example
`../../k/contracts/./diff-v1.yaml` from another crate) therefore missed the refusal. The fix is `code.rs::lexical`,
which normalizes `.`/`..` lexically on both sides. The test is
`a_binding_that_reaches_a_refused_copy_through_dot_dot_is_still_refused`.

```
M0 fixed                                         rc 0    8 passed
M4 lookup without lexical() (the round-1 code)   rc 101  a_binding_that_reaches_a_refused_copy_through_dot_dot_is_still_refused FAILED
clippy -p aprender-contracts --lib --tests -D warnings rc 0; fmt rc 0; pv extract contracts --check rc 0
```
