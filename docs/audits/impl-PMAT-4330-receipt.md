---
status: partial
partial_reason: "The implementation is done and every gate below is green. Two owed items are events this PR cannot cause: (1) the ONT-4f probe's `merged ONT-4f` clause is satisfiable only after merge; (2) aprender#4323 (ONT-4c, open, not merged) adds Σ `entity_type_target_class`, and its sigma gate requires an entry for every implemented entity type, so whichever of #4323 / this PR merges second must add `repo: ont:Repo, issue: ont:Issue, pull-request: ont:PullRequest, milestone: ont:Milestone`."
ticket: PMAT-4330
row: ONT-4f
issue: 4330
model: "claude-opus-5-5 (1M context)"
---
# impl-PMAT-4330 — ONT-4f · GitHub repos, issues, pull requests and milestones as focus nodes

## Identity
Spec: paiml/infra `docs/specifications/paiml-ontology.md` v4.16 §5 row ONT-4f. Branch `ONT-4f-github-entities`,
stacked on `fold/b3-contracts-ont-docs` (#4317).

## What changed
- **Σ** (`contracts/ontology.yaml`): concepts `Json`, `Repo`, `Issue`, `PullRequest`, `Milestone`; entity types
  `repo`, `issue`, `pull-request`, `milestone` on the `json` extractor, each with a `vocabulary`
  (`prefix`, `root_class`, `version`: `sha` for repo, `updatedAt` for the rest); `Repo|Issue|PullRequest|Milestone ⊑ Json`.
  `sigma.rs` parses `vocabulary` (deny_unknown_fields) and refuses an empty prefix/version or a `root_class`
  that is not a declared concept (`SigmaError::VocabularyMalformed`).
- **Extractor** `ontology/extract/json/github.rs`: a submodule of `json`, not a new extractor. It reads only committed
  snapshots `evidence/github/<type>/<ref-slug>.json` and makes no network call. It refuses by name (PV-ONT-012, Fail):
  a ref whose version disagrees with the snapshot's own `sha`/`updatedAt` (both values named); `state: merged` with no
  `mergedAt`; an identity or file-name contradiction; a duplicate; a nested object; a stray dir or a non-json file.
  `resolves:` works between tracked snapshots only. A hit is an IRI edge. A miss is a literal on `<key>Unresolved`, which
  the armed shape holds at `maxCount: 0`, so it fails closed and is never Unknown. A unit test asserts the RESOLVES
  table equals the contract's `resolves:` declarations.
- **Contract** `contracts/github-entities-v1.yaml`: four shapes (`github-repo`, `github-issue`,
  `github-pull-request`, `github-milestone`), all four armed in `lint-baseline.json`.
- **Gate** `shapes_gate.rs`: `by_entity_type` carries the four types; `pc_extract` fires a positive control per type.
- **Snapshots**: `paiml/aprender@aa7c6ef0…`, issue #3022, PR #3706 (merged, baseRepo → the repo), milestone #3.
- **Tool fix, in scope** (`scripts/check_ont_ratchet.sh`): `make ont-ratchet` DELETED `underived_proved_claims`
  (the proved-is-derived ratchet) from `lint-baseline.json` while printing PASS. The cause was that the carried
  top-level keys were a hand-kept allowlist that the proved-is-derived gate never joined. Inverted: every top-level
  key the script does not own is carried, and a multi-line value is refused, never dropped. Two new self-test rows.

## Measured
| check | result |
|---|---|
| `pv lint contracts/ --gate shapes --format json` | `Pass`; by_entity_type repo 1, issue 1, pull-request 1, milestone 1; pc_extract all four `fired` |
| ONT-4f fixture harness (`done-when-test.sh --suite ont --row ONT-4f`) | rows=1 cases=12 failed=0 |
| `cargo test -p aprender-contracts --lib` | 1920 passed, 0 failed, 5 ignored |
| every `aprender-contracts-cli` `ont*` test (18 suites incl. new `ont4f_github_entities`, 6 tests) | all ok |
| `cargo fmt --check`, `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` | clean |
| `cargo deny check advisories` | advisories ok |
| `check_ont_ratchet.sh` / `--self-test` | PASS / 33 passed 0 failed |
| regenerated | census.json, contracts.nt, shapes.ttl, ontology.ofn, tbox-report.json, witness (pv-sat), README count 1836 |

## RED (each fix reverted, the named test fails, fix restored)
| mutation | red tests |
|---|---|
| drop the `mergedAt` check | `mutation_a_merged_pull_request_with_no_merged_at_is_refused_naming_the_field`, `every_positive_control_fires…` |
| drop the version check | `mutation_a_repo_whose_sha_disagrees_with_its_ref_is_refused_naming_both`, `a_milestone_updated_at_mismatch…`, `every_positive_control_fires…` |
| write an untracked value as a plain literal on the key (no `Unresolved`) | `mutation_an_issue_naming_an_untracked_milestone_fails_closed_never_unknown`, `every_positive_control_fires…` |
| old ratchet allowlist | self-test `--write keeps a top-level ratchet no list names` (got 0) and `a multi-line top-level value is refused` (got carried): 31 passed 2 failed |

## Where the spec's wording differs from the code (recorded here, not a STOP; the contract follows the code)
- Σ `subsumes` uses `sup:`, not `super:`. The extractor is named `json`, not `extract:json`.
- Σ has no ParityReceipt ⊑ Json edge; parity-receipt has its own extractor module. GitHub is a `json` submodule, as the row asks.
- ontology.ofn and tbox-report.json both change (four classes, four SubClassOf). `scripts/batch_fold.sh --regen` does not
  regenerate tbox-report.json or the witness, so a fold with a Σ change needs `pv ontology tbox --write` and `pv-sat contracts` too.
- A corpus whose ONLY snapshot is refused has zero focus nodes and declines `Unknown{NoFocus}` (exit 2, not a pass)
  before the refusal is reported. The sha-mismatch fixture therefore carries a second, valid snapshot.

## Residuals
1. `merged ONT-4f`: unmet until merge.
2. #4323 `entity_type_target_class`: add the four entries when the second of the two merges (see partial_reason).
3. The ont4b counters `shapes_n` (18 → 22) and `plant_violations` (3 → 26, the four armed shapes' minCounts) are shared,
   so a sibling PR that adds a shape raises them again at merge.
