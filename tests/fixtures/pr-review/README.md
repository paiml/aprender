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
| 7 | `row-07-honest-docs-only-pmat-consulted` | **GREEN** | — | a docs-only PR: `pmat` consulted and empty, nothing else triggered |
| 8 | `row-08-self-review` | RED | B2 | `reviewer_actor.id == author_actor.id` |
| 9 | `row-09-stale-index-verdict-pass` | RED | B6 | `index_commit` is not an ancestor of `head_sha`, verdict `PASS` |
| 10 | `row-10-base-sha-not-merge-base` | RED | B1 | `base_sha` names main's tip, not the fork point |
| 11 | `row-11-empty-failure-scenario` | RED | B1 | a finding with an empty `failure_scenario` |
| 12 | `row-12-excerpt-digest-mismatch` | RED | B1 | `excerpt_sha256` != `sha256(excerpt)` |
| 13 | `row-13-invalid-signature` | RED | B1 | a real Ed25519 signature over different bytes |
| 14 | `row-14-complete-gpu-review` | **GREEN** | — | all four consulted, a verified citation, a complete comparator |
| 15 | `row-15-finding-with-no-grounding-mark` | RED | B1 | a finding carrying **no** `properties.grounding` at all |
| 16 | `row-16-comparative-claim-only-in-the-diff` | RED | B4 | the diff publishes `2.93× Ollama` in `book/`; `comparative_claims` is empty |
| 17 | `row-17-comparative-claim-recorded` | **GREEN** | — | the same diff, the same ratio, recorded with a complete comparator |
| 18 | `row-18-cuda-consulted-no-queries` | RED | B1 | `cuda.status: consulted` with `queries: []` |
| 19 | `row-19-pmat-not-triggered-on-a-code-diff` | RED | B1 | `pmat.status: not-triggered`, though §3.A is unconditional |
| 20 | `row-20-mutation-not-triggered-on-a-code-diff` | RED | B1 | `mutation.status: not-triggered` on a diff changing Rust source |
| 21 | `row-21-crux-not-triggered-on-a-claim-diff` | RED | B1 | `crux.status: not-triggered` on a diff publishing a competitor ratio |
| 22 | `row-22-printed-ratio-not-the-quoted-one` | RED | B4 | one `.rs` file, the same ratio twice: a `format!` a user reads **fires**, the `//` comment two lines above it **does not** |

**Rows 6, 7, 14 and 17 are discrimination cases.** Without them, a guard that refuses every
receipt reads green — the over-reach a discrimination case already caught in PERF-055
and in the #2766 delta-gate work. Row 14 is the widest: it carries a `cited` finding
whose digest matches, a `measured` finding, and a comparative claim with a full
comparator, so a guard that refuses correct work fails on it.

**Row 15 is not in §6.3.** It is owed to PRREV-003 by `contracts/pr-review-skill-v2.yaml`,
whose falsification test `F-PRREV-001` was recorded LIVE-PENDING on exactly this case:
rows 3, 11 and 12 cover a *malformed* grounding mark, but nothing covered a *missing*
one — the single §8 metric (`unmarked_claims = 0`) the fourteen rows left asserted. The
row exists and is wired, so PRREV-008 discharges `F-PRREV-001` in the contract: a metric
whose `check:` still reads *"NOT YET EXERCISED BY ANY FIXTURE"* while a fixture exercises
it is a ledger that has stopped tracking the tree, and the next reader believes whichever
of the two they happen to open.

**Rows 16–22 are not in §6.3 either.** They are PRREV-008's, one per defect the §9 step-7
backtest measured against this guard, and every one of them was **ACCEPTED** before:

| defect | what was wrong | rows |
|---|---|---|
| F1 | `match_comparative` had one call site, over findings *the reviewer wrote*, so B4 never read the diff. A signed discrimination pair proved the verdict turned on the reviewer's candour, not on the diff. | 16, 17, 22 |
| F2 | Only cuda's trigger was recomputed and only mutation's emptiness was checked; **no** consultation had both. | 18, 20, 21 + probes |
| F3 | `pmat: not-triggered` was accepted on a code PR, though §3.A calls pmat unconditional — and row 7 *blessed* it with a `trigger_reason` reading "not-triggered is never correct for it". | 19, and row 7 rebuilt |
| F5 | B4's pattern allowed a **zero-word** gap where #2763 measured **five**, so `36.9x over FasterTransformer` — the spelling APR-PERF-GATE-001 §0.1 uses — did not match. | the case table below |

**Row 22 is the scope, and the scope was measured twice.** B4's diff half was first written
over every changed `.rs` line and all of `docs/**`. Run against the last **300 commits of
`origin/main`** it fires five times, on two commits — and **three of the five quote a
fabricated claim in order to ban it**, two of them in `docs/benchmarking-gate-spec.md`
(*"2.93× Ollama from a harness that never ran Ollama"*) and one in a `//` comment
(*"// #2696: this printed \"Performance: 800+ tok/s (2.8x Ollama)\""*).

Those three have **no honest remedy.** §3.C.1's exit is a recorded comparator command,
version and log, and there is no log for a number nobody measured. 2 of 5 is **40%
measured precision** against §7's ≥90% admission bar, so the class may not block there —
a gate whose only exit is to fabricate the evidence it demands is worse than the hole it
closes, and that is #2757 and #2766 exactly.

So B4 blocks on `book/**.md` and on **printed literals and doc comments** in shipped
`.rs` — the surfaces `check_no_claim_literals.sh` measured as user-facing, and where
*"2.93× Ollama"* was actually published. Over the same 300 commits that scope fires
**zero** times: no measured false positives **and** no measured true positives. That is
not evidence of precision; it is evidence that this repository has not published a
competitor ratio to the book in 300 commits, and it is written down as such.

**Residual, recorded rather than hidden:** a comparative claim added to `docs/` prose or
to a plain `//` comment is not blocked. Two real ones are named in the guard —
`0.097× llama.cpp at c=16` and `// 15.7 tok/s decode, 0.099x llama.cpp`.

Every consultation now carries **both halves**, which is the shape the audit found missing:

| consultation | trigger recomputed from the diff | emptiness checked |
|---|---|---|
| `pmat` | unconditional (§3.A) — `not-triggered` is never legal | the four §3.A arrays must be present *as arrays* |
| `cuda` | path + commit message (§3.B) | `queries[]` non-empty and well-formed |
| `crux` | surface declaration + comparative claim (§3.C) | `surfaces[]` or `comparative_claims[]` non-empty; `crux_coverage`/`gap_effect` in vocabulary |
| `mutation` | file shape (§3.D) | `attempted > 0`, `killed <= attempted`, survivors match the arithmetic |

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

The twenty-two rows pin twenty-two branches; the guard has seventy-odd. Every branch the rows
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
  +---- F1                  <- gpu-pr head    (adds src/cuda/kernel.cu)
  \
  +---- D1                  <- docs-pr head   (adds docs/note.md)
  \
  +---- G1                  <- claim-pr head  (adds book/…/apr-cli.md: 2.93× Ollama)
  \
   +--- S1                  <- code-pr head   (adds a plain .rs file)
```

`G1` and `S1` are PRREV-008's. B4's diff half cannot be exercised without a head that
**publishes** a ratio on a surface a user reads, and §3.D's trigger cannot be exercised
without a head that touches Rust source — `F1` adds a `.cu` and `D1` adds markdown, so
`mutation: not-triggered` was true on every head the fixtures had. A rule tested only
against receipts is the circularity F1 was opened for.

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

Eight tables — `cuda-path`, `cuda-message`, `comparative-claim`, `shipped-surface`,
`rs-published`, `crux-surface`, `mutation-trigger`, `target` — each driven through its
own `--match-*` predicate. Every regex in the guard ships must-match **and** must-not-match rows. This
repository's guard patterns have been wrong six times; a case table caught every one and
review caught none.

**The comparative table was itself an instance of the failure it exists to prevent.** Its
first thirteen rows were written from `COMPARATIVE_RE`'s own vocabulary, so they passed
13/13 over a hole: the pattern allowed a **zero-word gap** between the ratio and the
competitor, where #2763/PERF-049 had already *measured* a five-word bound for the same
rule. `36.9x over FasterTransformer` — the spelling APR-PERF-GATE-001 §0.1 uses, and the
literal #2763 hardened `check_no_claim_literals.sh` to catch — did not match. A guard
universe built from the wrong side.

The pattern is now **`check_no_claim_literals.sh`'s `RATIO_RE`**, with the two competitor
lists unioned, rather than a second implementation of the same rule. Measured over the
6909-file shipped surface at `origin/main` `745fa8588`, read from a pristine worktree of
that ref: **61 hits before, 73 after, none of the 61 lost, and all 12 additions real
comparative claims.** On the table itself the old pattern missed **seven** must-match rows
and produced zero spurious matches, so the change closes holes rather than trading
precision for recall. The 16 CORPUS rows are the backtest's own table
(`evidence/pr-review/backtest/comparative-claim-backtest-cases.tsv`), drawn from this
repository's real claim corpus rather than from the pattern; the transcript of both
measurements is `evidence/prrev-008/comparative-pattern-measurement.txt`.

Deliberate gaps are pinned as NO-MATCH rows rather than silently widened:

- `crates/aprender-compute/src/backends/gpu/device/mod.rs` — §3.B names
  `crates/aprender-gpu/**` and `*cuda*`/`*ptx*`/`*cublas*`/`*fp8*`/`*nvrtc*`, never
  `*gpu*`. A real GPU file does not trigger.
- `fix: CUDA build flags` — the path trigger is case-insensitive; §3.B's *message*
  regexes (`cu[A-Z]\w+`, `cuda[A-Z]\w+`) are not, so an all-caps `CUDA` matches none.
- `Turing` and `Pascal` are absent from the architecture list, because "Turing complete"
  and "PascalCase" are commoner in this repository's commit messages than the parts. A
  blocking class must hold >=90% precision to stay armed (§7 admission rule).

- `Command::new` is **not** a `crux-surface` token though clap uses it: 752 hits across
  293 files, overwhelmingly `std::process::Command`. This class only fires on a receipt
  claiming the surface did *not* change, so a false positive calls an honest reviewer a
  liar.
- `OutputFormat` and `#[serde(rename …)]` are not tokens either, so §3.C's *config key*
  and *output format* routes are uncovered — 822 hits across 103 files for the first, and
  no spelling of the second measured precisely enough to block on.
- The **PR body** is outside B4's diff recomputation. Commit messages are not a
  substitute: this repository's own commit messages quote the banned ratios in order to
  ban them, so scanning them would red the very commits that fix the defect.
- A comparative claim in `tests/`, `benches/`, `examples/`, a fixture, `docs/**` or a
  plain `//` comment is out of scope — a target is not a claim, and a document quoting a
  banned literal in order to ban it is not publishing it. Asserted by a bats test, both
  because without that scope this guard could never be edited again without a comparator
  for its own case table, and because the measurement above says the class would
  otherwise block at 40% precision with no honest exit.

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
