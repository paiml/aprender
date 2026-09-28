# PMAT-3715 proposal: enforce `release-readiness-v1` as a never-worse ratchet (for quorum)

Author: aprender-0d (claude-opus-5-5), 2026-09-28. Assigned by the cop (aprender-77), verbatim:

> "OPERATOR plan: you own #3715 today. Propose enforcing it as a never-worse ratchet at the current count (1101),
> with burn-down tickets. Run a quorum (sonnet + agy + haiku, never the author's model id) on whether that meets
> the ONT-10 row wording. If the quorum says yes, land the ratchet; if not, fix it for real."

**This diff is a proposal, not code.** The lanes judge ONE question (section 5). A `PASS` means "a ratchet meets the
wording, so land it". A `FAIL` means "it does not, so fix it for real" (section 4).

## 1. What is measured today

Measured by aprender-e6 with pv built from the tree, on the 0.69.1 receipts, at `fold/3715-on-b3` @2ccb51778:

| class | violations |
|-------|-----------:|
| cell (no `ont:release/row` for a model x host x verb x think x rung cell) | 1008 |
| coverage | 87 |
| model | 28 |
| tokenizer | 28 |
| context | 5 |
| host | 2 |
| release | 1 |
| **total** (`pv lint --gate shapes --shape release-readiness-v1`, rc 1) | **1101** |

Why: the #3712 `cells[]` producer exists, but only `model_ladder.sh --cells` runs it, from the nightly.
`scripts/release/autopilot.sh` never calls `model_ladder.sh --cells`. So release receipts carry no cells, and every
cell is missing. `scripts/release/release_readiness.sh` is committed as `DEFAULT_MODE=report`: a Fail prints a WARN row
and the train continues (this branch's base, B5 `ont/0.70-c2` @aa2ed6776).

## 2. The proposal: not a stored 1101 but a head-vs-base ratchet

A literal ratchet "at 1101" is a number committed to the repo. The operator ruled on 2026-09-27 that never-worse
counts are gated **head vs base, same scanner, same run**. They are never gated against a stored limit, and there is
no re-baseline and no waiver. So the proposal, restated in that form:

- **R8 ratchet mode.** At T-1 (autopilot `models`) and T-4 (`check_publish_preflight.sh` R8), `release_readiness.sh`
  runs the same pv binary twice in the same run:
  - on HEAD: this release's receipts at the release commit;
  - on BASE: the previous release tag's committed receipts, graded against that tag's commit.
- **Keys.** Each violation is keyed version-independently as (class, model sha256, host, verb, think, rung,
  sh:resultPath). The `apr_sha` and version are left out of the key, so "stale because it is a new version" does not
  count as new.
- **Verdict.** Pass iff `keys(HEAD) ⊆ keys(BASE)`. A new key STOPs the tag and names it. A cell that passed at BASE and
  is missing at HEAD is a new key.
- **Tests.** A planted fixture (BASE plus one new missing cell) turns it RED. BASE equal to HEAD is GREEN.
  HEAD ⊂ BASE is GREEN and prints the burn-down delta.
- **Burn-down tickets.** These are proposals; the cop mints them:
  - **BD-1, cell 1008:** the autopilot `models` step runs `model_ladder.sh --cells` on lambda and gx10, and commits
    `cells[]` in the release receipts. This is the real fix.
  - **BD-2, coverage 87.**
  - **BD-3, model 28 and tokenizer 28:** the inventory-model rows the ladder does not cover.
  - **BD-4, context 5, host 2, release 1.**
- **The flip to `enforce`** (every violation STOPs) happens when `keys(HEAD) = ∅` on a release.

## 3. The words it is judged against

**ONT-10 row**, infra `docs/specifications/paiml-ontology.md` v4.17, line 714, verbatim:

> ONT-10: release aprender-contracts-cli with ONT-0..9 (incl. 4d, 4e) — CI tags, operator publishes (RP-001) ·
> depends_on ONT-0..9 (incl. ONT-4b2, ONT-4g), ONT-D, ONT-G, ONT-11, PVL EV-12 · ONT-12 deliberately NOT here.
> Clean-room first; CI green for aprender and dependents; RP-001 no-publish-in-ci green; `make oracle` green on the
> tag; release receipt records previous_pin; STOP(PUBLISH…); probe = cargo search version == pkgid && merged ONT-10.

The row never names #3715 or `release-readiness`. #3715 reaches 0.70 through its `ont-0.70` label and the operator's
0.70 gate: "FULL ontology spec + ALL aprender binaries working".

**The spec's own mechanism for a shape that is RED by construction**, v4.6, line 458: `armed_shapes[]` in
`contracts/lint-baseline.json`. A shape not listed there is computed and reported (`not_armed_shapes[]`) and does not
feed the verdict. `ont.shapes_unarmed` is recorded, not ratcheted.

**#3715 done_when**, verbatim fragments (full text in `docs/audits/impl-PMAT-3715-receipt.md` §1):
- item 4: "a violation STOPs before the tag and refus[es]…"
- item 5: "On 0.69.1 it passes with N/N cells named."
- the title: "the tag is REFUSED unless every … cell has a Pass receipt — a missing cell is a violation, never a skip"

## 4. The alternative, "fix it for real"

- Wire BD-1 now: the autopilot `models` step runs `model_ladder.sh --cells` (T-1).
- Produce one `--cells` receipt set on lambda and gx10 for the 0.70 release commit.
- Grade it. If it is Pass, flip `DEFAULT_MODE=enforce` in one reviewed commit.
- What does not grade Pass is listed by key. It is either fixed, or moved out of 0.70 by a cop ruling on the inventory
  (a model that is not in the inventory is not owed). It is never waived.

## 5. The question for each lane

Judge the proposal in section 2 against the words in section 3.

1. Does it meet the ONT-10 row wording? The row is silent on #3715. Does silence make a ratchet sufficient for ONT-10,
   or does the 0.70 "FULL ontology spec" gate carry #3715's own wording in with it?
2. Does it meet #3715's done_when items 4 and 5 and its title ("REFUSED unless every cell … Pass", "never a skip")?
3. Is a head-vs-base ratchet that tolerates the BASE's 1101 keys distinguishable from the `armed_shapes[]` report-only
   state the spec already gives? Or is it an un-armed shape under another name?

**PASS** = a ratchet meets the wording and may be landed as #3715's enforcement.
**FAIL** = it does not, and section 4 is the path.

Author's disclosure, for the lanes to confirm or refute:
- I read items 4 and 5 as requiring a STOP on any violation and N/N on 0.69.1. A ratchet meets neither.
- ONT-10's row neither requires nor forbids it.
