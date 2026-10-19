# #5762 revised: contracts.nt / shapes.ttl / witness / README count are generated shared files

Owner: flow (aprender-flow). Supersedes the 14:30Z text after the 3-lane Opus quorum (receipt:
aprender-pace q5762/receipt.md): "regen-at-merge, untrack the files" was NO as written. Four blockers, answered below.

## What is generated (corrected: four things, not two)
`contracts/contracts.nt`, `contracts/shapes.ttl` (`contracts_gate.sh` GENERATED=), `contracts/witness/<relations_sha256>.json`
(pv-sat; the name is the hash of the graph), and the contract count inside README.md (`scripts/readme_sync.sh`).
Two PRs that each add a contract conflict on them because each commits its own regenerated copy.

## Blockers from the quorum and the answer
1. step_diff / ont-consistency starve if the files are untracked: **agreed. Nothing is untracked.** The gate is unchanged.
2. shapes.ttl omitted: **added** (and README.md).
3. No workflow can push to protected main, so "a job writes the committed copy" is impossible: **dropped.** No bot, no admin skip.
4. The verify step stays a blocking GATE with a RED mutant row: **unchanged**; `contracts_gate.sh` is not edited, so its
   existing row "a contract added WITHOUT regenerating is RED on diff" (`contracts_gate.sh` --self-test) keeps covering it.

## Proposal: option "merge-main + regen helper" (quorum's lower-risk alternative 7)
`scripts/contracts_resolve.sh` (new, 40 lines, a developer tool, NOT on any CI path, not gate content):
after `git merge origin/main` stops, it refuses unless every conflict is a generated path, drops those paths, regenerates
them with the gate's own `regen` step + `pv-sat` + `readme_sync.sh --write`, and stages them. The gate stays the judge.

Measured (throwaway merge of #5710 head with origin/main 14:50Z, never pushed):
- before: conflict on `contracts/witness/811c7818...json`; README count stale.
- `bash scripts/contracts_resolve.sh`: 54 s wall (pv build cached), pv-sat 2 ms.
- then commit, `bash scripts/contracts_gate.sh`: `8 of 8 step(s) RAN, 0 FAILED` (diff + consistency + readme green).
- first attempt without the README step: `2 FAILED: census readme` -> that is why README.md is in the set.
Not measured: #5672 (same shape, not run), a merge where a non-generated file also conflicts (the helper refuses by design).

## What this does not fix
The conflict still happens; it becomes a 1-minute mechanical step per PR instead of a hand job. Removing it needs
option B (fragment: one `.nt` per contract, witness out of tree), a bigger, one-way change that I am not proposing
until the consumers listed by the quorum (`lint_passes_on_real_contracts`, `ont4d_subsumption`, `ont5_consistency_gate`,
`tests/oracle/jena`, `batch_fold.sh`) are costed. Falsifier for this proposal: a DIRTY contract PR that
`contracts_resolve.sh` leaves red on `contracts_gate.sh`.

## Ask
No quorum needed for the script alone (no workflow, no gate content). Please confirm that reading; if you want it quorumed
anyway, the receipt can be on this head.
