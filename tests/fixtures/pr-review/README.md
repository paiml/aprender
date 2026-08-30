# pr-review fixtures — PR-REVIEW-SKILL-002 v2 §6.3

Fixtures for `scripts/check_pr_review_receipt.sh`, exercised by `tests/pr-review.bats`.

```bash
bats tests/pr-review.bats          # all rows, both case-table polarities
```

## The table

Every row is a committed directory holding the three artifacts the guard reads:
`receipt.intoto.jsonl`, `findings.sarif`, and the detached `receipt.intoto.jsonl.minisig`.
Each RED row also names the blocking class from `contracts/pr-review-skill-v2.yaml`,
because a rejection with no named class is how a guard grows a rule nothing governs.

| # | fixture | verdict | class | what is wrong |
|---|---------|---------|-------|----------------|
| 1 | `row-01-cuda-not-triggered-on-cuda-diff` | RED | B1 | `cuda.status: not-triggered` while the diff touches `src/cuda/kernel.cu` |
| 2 | `row-02-mutation-attempted-zero` | RED | B1 | `mutation.attempted: 0` with `status: consulted` |
| 3 | `row-03-cited-empty-excerpt` | RED | B1 | a `cited` finding whose `excerpt` is empty |
| 4 | `row-04-comparative-claim-no-comparator` | RED | B4 | `2.93x Ollama` with no comparator `command` or `artifact_sha256` |
| 5 | `row-05-unreachable-pmat-verdict-pass` | RED | B1 | `pmat.status: unreachable` with `verdict: PASS` |
| 6 | `row-06-unreachable-pmat-verdict-degraded` | **GREEN** | — | the same unreachable source, honestly recorded as `DEGRADED` |
| 7 | `row-07-honest-docs-only-all-not-triggered` | **GREEN** | — | a docs-only PR where nothing triggers, and nothing pretends to |
| 8 | `row-08-self-review` | RED | B2 | `reviewer_actor.id == author_actor.id` |
| 9 | `row-09-stale-index-verdict-pass` | RED | B6 | `index_commit` is not an ancestor of `head_sha`, verdict `PASS` |
| 10 | `row-10-base-sha-not-merge-base` | RED | B1 | `base_sha` names main's tip, not the fork point |
| 11 | `row-11-empty-failure-scenario` | RED | B1 | a finding with an empty `failure_scenario` |
| 12 | `row-12-excerpt-digest-mismatch` | RED | B1 | `excerpt_sha256` != `sha256(excerpt)` |
| 13 | `row-13-invalid-signature` | RED | B1 | a real Ed25519 signature over different bytes |
| 14 | `row-14-complete-gpu-review` | **GREEN** | — | all four consulted, a verified citation, a complete comparator |
| 15 | `row-15-finding-with-no-grounding-mark` | RED | B1 | a finding carrying **no** `properties.grounding` at all |

**Rows 6, 7 and 14 are discrimination cases.** Without them, a guard that refuses every
receipt reads green — the over-reach a discrimination case already caught in PERF-055
and in the #2766 delta-gate work. Row 14 is the widest: it carries a `cited` finding
whose digest matches, a `measured` finding, and a comparative claim with a full
comparator, so a guard that refuses correct work fails on it.

**Row 15 is not in §6.3.** It is owed to PRREV-003 by `contracts/pr-review-skill-v2.yaml`,
whose falsification test `F-PRREV-001` is recorded LIVE-PENDING on exactly this case:
rows 3, 11 and 12 cover a *malformed* grounding mark, but nothing covered a *missing*
one — the single §8 metric (`unmarked_claims = 0`) the fourteen rows left asserted.

**A missing receipt is RED, not skipped**, and so is a run over zero receipts. Both are
bats tests rather than table rows.

## The positive controls (§6.1)

`positive-control/{self-review,findings-digest,cost-missing}/` plus `positive-control.pub`.

The guard validates these **before** anything real and requires each to be rejected under
a named class *and* with a named reason. They are schema-valid and correctly signed, so
they can only be rejected by reaching the semantic branch each one pins.

Three of them, not one, because they are a **mutation-kill set**: measured, a
schema-depth control alone left eleven deleted branches undetected, and asserting only
the *class* left a twelfth undetected (deleting the in-toto schema gate made the schema
control fire on the signature branch while still reporting `B1`). Their SHAs are all-zero
on purpose — every class they pin is evaluated before the merge-base check, so they need
no git repository and run identically in CI, in a worktree, and on a laptop.

## Why a synthesized git repository

`make-fixture-repo.sh` builds the repo the receipts are written against:

```
C1 ---- C2 ---- C3          <- main, and refs/remotes/origin/main
 \
  +---- F1                  <- gpu-pr head   (adds src/cuda/kernel.cu)
  \
   +--- D1                  <- docs-pr head  (adds docs/note.md)
```

Rows 1, 9, 10 and 14 are not exercisable against aprender's own history: for **any**
commit `X` reachable from `origin/main`, `git merge-base origin/main X` is `X` itself.
`base_sha` would be forced to equal `head_sha` and every diff would be empty, so row 10's
comparison and row 1's path trigger would both pass vacuously — the same shape as
`pv lint <FILE>` returning PASS over zero contracts. C2 and C3 stand in for the unrelated
PRs another agent lands on main while a review is open, which is the diff-scope pollution
§2's merge-base boundary exists to keep out.

The SHAs are deterministic (fixed identity and dates, no hooks, no signing) and the
script **asserts** them against `expected-shas.txt`, so a drift fails loudly instead of
silently validating a different repository.

## Case tables

`cuda-path-cases.tsv`, `cuda-message-cases.tsv`, `comparative-claim-cases.tsv` — every
regex in the guard ships must-match **and** must-not-match rows, driven through
`--match-path` / `--match-message` / `--match-comparative`. This repository's guard
patterns have been wrong six times; a case table caught every one and review caught none.

Three deliberate gaps are pinned as NO-MATCH rows rather than silently widened:

- `crates/aprender-compute/src/backends/gpu/device/mod.rs` — §3.B names
  `crates/aprender-gpu/**` and `*cuda*`/`*ptx*`/`*cublas*`/`*fp8*`/`*nvrtc*`, never
  `*gpu*`. A real GPU file does not trigger.
- `fix: CUDA build flags` — the path trigger is case-insensitive; §3.B's *message*
  regexes (`cu[A-Z]\w+`, `cuda[A-Z]\w+`) are not, so an all-caps `CUDA` matches none.
- `Turing` and `Pascal` are absent from the architecture list, because "Turing complete"
  and "PascalCase" are commoner in this repository's commit messages than the parts. A
  blocking class must hold >=90% precision to stay armed (§7 admission rule).

Widening the patterns here would put the guard silently out of step with its spec.

## Regenerating

```bash
tests/fixtures/pr-review/build-fixtures.sh     # rebuilds and re-signs every fixture
```

Every digest inside a fixture is computed by that script, never typed: `findings_ref.sha256`
is `sha256(findings.sarif)` and `excerpt_sha256` is `sha256(excerpt)`. A hand-typed digest
is a fixture that passes for the wrong reason the first time someone edits the file it
describes.

`keys/pr-review-test-TEST-ONLY.key` is an unencrypted **test-only** minisign key. It signs
fixtures and nothing else. The production key referenced by §4.3 is `.github/pr-review.pub`,
which PRREV-005/006 owes; the guard rejects a receipt when the public key is absent, because
an unverifiable signature is not a verified one.
