---
status: complete-pending-merge
ticket: PMAT-3847
row: ONT-4c
issue: 3847
model: "claude-opus-5-5 (infra ONT-001 session): extractors, gate wiring, ratchets, contracts, fixtures"
---
# impl-PMAT-3847 — ONT-4c · the first non-code contracts: README.md, CLAUDE.md, one .apr, one CSV (#3847; ONT-001 v4.16 §5 ONT-4c, B.4–B.8)

## What lands
- `ontology::extract::{claims, readme, llm_context, csv}`:
  - `claims` holds the shared CommonMark machinery: frontmatter, headings, code spans, and fences at any nesting depth. A **claim fence** is one whose first info token, lower-cased with `{`/`.` stripped, is `bash`, `sh` or `shell`. The **CI set** is every normalised `run:` line of a merge-path workflow (push, pull_request, pull_request_target, merge_group), excluding steps under `if: false`.
  - `readme` builds a `readme:Readme` node:
    - `readme:verifiedCommand` has `resolves: ci-step`.
    - `readme:entrypoint` has `resolves: path`.
    - `readme:contractCount` has `resolves: census`.
  - `llm_context` builds an `llm:LlmContext` node:
    - The four role predicates come from Σ `llm_context_role_synonyms` (D5b).
    - `llm:referencedPath` has `resolves: path`.
    - `llm:command` has `resolves: ci-step`.
    - It also emits `llm:declaredTool` and `llm:neverRule`.
  - `csv` builds a `csv:Dataset` node carrying header, columns, rows, dtypes, sha256 and a `producer` with `resolves: path`. A ragged row is refused, naming the file and the row.
  - Each extractor has an in-memory positive control (`pc_extract.readme` / `llm-context` / `csv`).
- `ontology::measured_sets` and the gate:
  - The committed `lint-baseline.json` `readme` / `claude_md` `{verified_commands, withdrawn}` are measured.
  - **F-33** (PV-ONT-013, library): the committed set must equal the live one, both directions, named.
  - **F-34** (PV-ONT-014, CLI `lint_arming::measured_ratchet`) checks `withdrawn(HEAD)\withdrawn(cmp) == set(cmp)\set(live)` against the merge-base. The three comparand cases:
    - no work tree → `not-checked`
    - a work tree with no ref → RED
    - corpus untracked at the comparand → ∅
  - The report carries `readme`, `claude_md` and `ratchets.measured_sets`.
- Σ (D-T1/D-T2):
  - `entity_type_target_class` gives a shape with no `targetClass` its default.
  - `apr_model` types `model:AprModel` as well as `model:Model`.
  - gguf ladder rungs are typed `model:LadderRung`.
  - `ladder-measured` is retargeted to `model:LadderRung`; that retarget is ONT-4c's change, and ONT-4c1's probe is untouched.
  - The map is cross-checked, not hand-kept (quorum #4323 follow-up): the sigma gate refuses, exit 3 named, a map key no `entity_types` entry declares and an `implemented: true` type with neither a class nor an explicit `~`. Σ now carries `json`/`code`/`lean`/`parity-receipt`/`release-evidence: ~`. Shown RED against the pre-fix code: `sigma_refuses_a_target_class_key_that_no_entity_type_declares`, `sigma_refuses_an_implemented_type_with_neither_a_mapping_nor_an_explicit_unmapped_entry` and the Σ half of the next test all FAILED (13 passed, 3 failed) before `check_target_classes` existed.
  - The spec Mutation "remove an `entity.type` from the map → exit 3 malformed, named" is a committed test: `removing_an_entity_type_from_the_target_class_map_exits_3_naming_the_shape_and_the_type` (shapes gate names `readme-doc` / `no targetClass`; sigma gate names `readme`).
- Contracts (kind `pattern`, all armed): `readme-root` (B.4), `claude-md` (B.5), `model-setfit-slice` (B.6, `slice_model.apr` by sha256), `csv-train` (B.8, producer `prepare_data.sh`).
- README.md and CLAUDE.md:
  - Each gains frontmatter; README's `contract_count` is written by `readme_sync.sh`.
  - Every claim fence no merge-path step runs was relabelled `text`, leaving one verified claim each: `bash scripts/check_readme_claims.sh` and `bash scripts/check_package_includes.sh`.
- `check_ont_ratchet.sh --write` measures the two sets (withdrawn = merge-base set \ current). Self-test: 24 rows.
- Tests and CI:
  - `tests/fixtures/ont/docs-green/` is a whole repo in miniature. `crates/aprender-contracts-cli/tests/ont4c_doc_contracts.rs` has 12 cases, each one edit to a copy. They cover: README/CLAUDE.md bad command, fence-tag variants, the `if: false` step, a bad path, a missing Purpose section, CSV column mismatch, F-33, F-34 relabel-drop with and without `withdrawn[]`, withdrawal-without-drop, bootstrap with the keys absent, and a work tree with no ref.
  - CI: `ci/explicit-test-commands.d/530-aprender-contracts-cli-ont4c-doc-contracts.cmd`. Open PRs hold 455–520 and 600.

## Probe (§5 ONT-4c, v4.16), this tree, built pv
Every conjunct before `merged ONT-4c` is GREEN. The measured values:
- `by_entity_type`: readme 1, llm-context 1, apr-model 1, csv 1
- `pc_extract`: all `fired`
- `readme.verified_commands` = 1 and `claude_md.verified_commands` = 1
- `ratchets.measured_sets` = `checked`
- `verdict` = `Pass`
- `present '^---' README.md` is true

`merged ONT-4c` is RED until this merges.

## Mutations (each on the built pv, `pv lint contracts/ --gate shapes`)
| mutation | verdict | named |
|---|---|---|
| unrun claim (`make ghost-target` in a bash fence) | Fail | `readme:verifiedCommand unresolved: make ghost-target` |
| relabel the verified README claim to `text`, no `withdrawn[]` | Fail | `readme-root` minCount on `readme:verifiedCommand`, and F-33. F-34 is shown in git by `f34_relabel_drops_claim_…` |
| one byte flipped in `slice_model.apr` | Fail | `model-setfit-slice` (in) on `model:sha256` |
| CSV header `label`→`target` | Fail | `csv-train` (in) on `csv:header` |
| `## Project Overview` renamed | Fail | `claude-md` minCount on `llm:purposeSection` |
| F-34 neutered (`dropped = ∅`) | test RED | `f34_relabel_drops_claim_without_withdrawn_is_red_and_with_it_passes` |

## Named residuals
- The README and CLAUDE.md examples that CI does not run are now `text` fences: illustrations, not claims. Wiring any of them into a merge-path workflow would promote it back to a verified claim.
- `GateExtra` carries `#[allow(clippy::large_enum_variant)]`. It is one value per run.
- Two new `formal:` entries use `≡` so that `formal_prose` does not rise.
- The roadmap status is not moved: the aggregator is Python, which is banned here.
- `lint::tests::lint_empty_dir` (pre-existing, not this branch) fails under `TMPDIR=/tmp` on lambda-labs because a stray `/tmp/scripts/contract_duplicate_stem_baseline.txt` exists there: the duplicate-stems gate walks up from its empty tempdir, finds that file, and reports 48 "no longer diverges" findings. The test is not hermetic; CI runners have no such file, so CI should not hit it. It passes under `TMPDIR=/mnt/nvme-raid0/tmp`.

## Verification (this tree)
| check | result |
|---|---|
| `cargo test -p aprender-contracts --lib` | 1730 passed, 0 failed |
| `cargo test -p aprender-contracts-cli --test ont4c_doc_contracts` / `ont4c1_model_receipts` / `ont6_lint_verdict` / `ont4b2_code_lean_w3c` / `ont_release_readiness` | 16 / 12 / 7 / 8 / 34 (+ `ont2b_sigma_gate` 8), all pass |
| `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` · `cargo fmt --all --check` · `cargo deny check advisories` | clean · 0 · ok |
| `make contracts` (pv lint 11/11 armed, census, `pv extract --check`, readme_sync, provenance, engine tests) | PASS |
| check_ont_ratchet --check · explicit_test_commands · tree_reader_tests · readme_claims · package_includes | all PASS |
