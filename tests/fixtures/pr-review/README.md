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

## Branch probes and the mutation set (§6.4)

`scripts/mutate-guard.sh` is the guard's own falsifier. §6.4 requires 100% kill and §8
records `guard_mutation_score` as **1.00 with no ratchet** — the one place §7's
narrowness does not apply, because every other verdict rests on the guard.

```bash
scripts/mutate-guard.sh --list       # the catalogue
scripts/mutate-guard.sh --jobs 16    # the sweep; exit 0 only at 100% kill
```

Two operators, applied at every `reject B<n>` site the guard contains, and the site list
is **derived by scanning the guard on every run** rather than written down — a catalogue
that has to be remembered falls behind the file it mutates:

| operator | edit | what it proves |
|---|---|---|
| `drop` | `reject B` → `true B` | the rule is *tested*: something must go RED→GREEN |
| `flip` | `\|\| return 1` → `&& return 1` | the branch's *sense* is right: a receipt satisfying the rule is now rejected, so a discrimination row must go GREEN→RED |

plus named single-line edits to the control machinery, which is not a uniform `reject`
site and cannot be derived. Each named entry must match **exactly one** line of the
guard or the sweep refuses to run: an entry that matches nothing mutates nothing and
scores a kill it never earned.

Excluded on purpose: the two `case` gates around `rm -rf` and the `EXIT` trap. They are
destructive-op guards, not validation branches — no receipt can reach them, so no
fixture can kill them, and including them would park a permanent survivor in a score
§8 fixes at one.

### Why the probes exist

The fifteen rows pin fifteen branches; the guard has fifty-odd. Every branch the rows
leave untripped came back from the first sweep as a **surviving mutant** — a rule the
guard states and nothing tests. `tests/pr-review.bats` therefore carries a second
family, named for the branch rather than for a spec row:

- shape and gates: `findings.sarif` absent, two JSON records, unparseable receipt,
  unparseable SARIF, SARIF that parses but fails the vendored schema
- signature material: the public key absent, the receipt unsigned
- predicate identity: wrong `predicateType`, an `attestation_level` claiming more than
  `L1-self`, `head_sha`/`base_sha`/`author_actor.id`/`reviewer_actor.id` absent, a
  `subject[0].digest` that is not the head under review, a verdict outside the four,
  `findings_ref.path` pointing elsewhere
- diff boundary: an unresolvable head; a head sharing **no** history with `origin/main`
- consultations: a status absent, a status outside the three-state vocabulary,
  `attempted`/`killed` that are not counts, `index_commit` absent or unresolvable, and
  `index_is_ancestor` *misreported* on a fresh index
- grounding: an invented fourth category, a `cited` finding with an empty `source` or no
  `excerpt_sha256`, an `asserted` finding classed `blocking`
- comparative claims: a competitor ratio stated in a finding while
  `comparative_claims[]` is empty — the never-ran-Ollama shape with one extra step
- the guard's own preconditions: a needed tool off `PATH`, no repository at all, a
  positive control that is *accepted*, a control measured against a permissive schema,
  and each control's own `|| exit 1`

A probe is **derived** from a committed row by one `jq` edit and **re-signed** with the
committed test-only key, because the signature is verified before any semantic branch is
reached — an unsigned probe would be rejected at the signature and would pin nothing.
The derivation, the base bytes and the key are all committed, so a probe is reproducible
from this tree: it is a shorter way of writing a fixture, not a weaker one.

### Every case now asserts its reason, not just its class

`B1` covers thirty-odd branches. A case that trips a *different* `B1` branch than the
one it exists to pin still reports `B1` and still exits 1 — it passes for the wrong
reason, and the mutant that dropped its branch lives.

**Measured, not argued.** A counter-sweep ran the whole set against a copy of
`tests/pr-review.bats` with every assertion made reason-blind: **110/119**, so **nine
mutants die only because the reason is asserted**. They sit on seven guard branches —
`head_sha` absent, `base_sha` absent, an unresolvable head, a non-numeric
`mutation.attempted`, `index_commit` absent or unresolvable, and a `cited` finding with
no `excerpt_sha256` — each of which falls through to a *neighbouring* `B1` branch and,
class-only, reads as a correct rejection.

The first candidate tested was **not** one of them, which is why this says *measured*:
dropping the empty-excerpt check was predicted to leave row 3 rejected on the digest
branch, and it does not. Row 3's `excerpt_sha256` is `sha256("")` on purpose, so with
the check gone the receipt is **accepted** and the exit code alone kills the mutant. The
prediction was wrong; the counter-sweep is what stands.

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
