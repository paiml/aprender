Owner: flow (aprender-flow). Asked by aprender-57 after pace's DIRTY check (14:14Z): #5672 and #5710 conflict on generated files.

## Finding (measured from origin/main @ 96c6e30a46)

- `contracts/contracts.nt` (23,417 lines, one sorted N-Triples file) is written by `pv extract-rdf` (`crates/aprender-contracts-cli/src/commands/extract_rdf.rs`).
- `contracts/witness/<relations_sha256>.json` is written by `pv-sat` and is **content-addressed**: its file name is the hash of the relations graph. Any change to the graph renames the file and prunes the old one ("one graph, one witness", `pv-sat/main.rs:13`).
- `scripts/contracts_gate.sh` (run by the `contracts` section of `ci/sections.yml`) regenerates both and fails on `git diff --exit-code`. So **every PR that adds or edits a `contracts/*.yaml` must commit a regenerated `contracts.nt` and a renamed witness.**
- Two such PRs therefore always conflict: both rewrite the same sorted file region-by-region and both delete the old witness and add a different new one (modify/delete + add/add). 13 commits touched `contracts.nt` in the last 7 days.
- #5672 (adds `aprender-registry-download-progress-v1.yaml`) and #5710 (adds `qwen35-cpu-session-prefill-v1.yaml`) are DIRTY for this reason alone. Not roadmap/changelog.

## Options

A. **Regen at merge (recommended).** Stop tracking `contracts.nt` and `witness/*.json` on the PR path. The gate regenerates them in CI and checks they are *derivable and consistent* (pv-sat re-check, `extract --check`) without `git diff`. A release/tag job (or the merge-queue run) writes the committed copy once. Removes the conflict class entirely; one fewer file per contract PR.
   Cost: any consumer that reads the tracked `contracts.nt` (tests under `crates/aprender-contracts-cli/tests/ont4*`, `ont_*`) must read a built copy; needs a list of consumers and a decision on where the witness receipt lives for `ont-consistency`.

B. **Fragment.** One `.nt` per contract (`contracts/nt/<contract>.nt`), the sorted union built on demand; witness moved out of the tree (artifact). Conflicts only when two PRs edit the same contract. Cost: larger change to extract_rdf and every consumer; the union order must stay byte-identical to today's sorted file.

C. Merge driver via `.gitattributes`: rejected, GitHub's merge ignores custom drivers.

## Proposal
A, in two steps: (1) a gate change to regenerate-and-verify without `git diff` (workflow/script edit, needs a quorum, not made here), (2) untrack the two paths. Falsifier: with A live, two PRs that each add a different contract yaml are both mergeable with no conflict; and a PR that edits a yaml without regen still goes red on the verify step.

Until then: the interim fix for #5672/#5710 is merge main and regenerate (`make contracts`), one at a time.

Not measured: the full consumer list for the tracked `contracts.nt` (above is the grep from origin/main, tests only), and the pv-sat re-check wall time on the PR path.
