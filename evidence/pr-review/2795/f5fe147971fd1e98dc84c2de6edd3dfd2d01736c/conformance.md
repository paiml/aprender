# PRREV-012 dogfood — the skill's first run against a real PR, and it is its own

**Subject:** PR #2795, head `f5fe147971fd1e98dc84c2de6edd3dfd2d01736c`, base
`745fa8588783413669f4ac6170d4645e39c15be4` (`git merge-base origin/main HEAD`, computed in
this review's worktree — never GitHub's `baseRefOid`).

**Why this subject.** The skill had never reviewed a real PR, its own included. The backtest
(`evidence/pr-review/backtest/`) reconstructed merged PRs, which proves the reasoning and not
that the thing runs in the workflow. Before this run there were **zero** `receipt.intoto.jsonl`
files anywhere under `evidence/pr-review/` — `find evidence/pr-review -name receipt.intoto.jsonl`
returned nothing. This is the first.

## The guard accepted it — with one asterisk that is itself finding 01

```
$ PR_REVIEW_PUBKEY=<throwaway>.pub bash scripts/check_pr_review_receipt.sh \
    evidence/pr-review/2795/f5fe147971fd1e98dc84c2de6edd3dfd2d01736c
positive-control  schema-invalid  fired (B1: receipt fails schemas/in-toto-statement-v1.json ...)
positive-control  self-review     fired (B2: reviewer_actor.id = author_actor.id ...)
positive-control  findings-digest fired (B1: findings_ref.sha256 ...)
positive-control  cost-missing    fired (B1: predicate.cost must carry numeric ...)
ACCEPT  evidence/pr-review/2795/f5fe147971fd1e98dc84c2de6edd3dfd2d01736c
```

All four positive controls fired before the verdict, so the ACCEPT is a verdict and not a count
of files.

**The asterisk.** `PR_REVIEW_PUBKEY` had to be overridden. The guard defaults it to
`.github/pr-review.pub`, that file does not exist, and line 380 rejects B1 on its absence — so
**no receipt can be accepted with the repository's own default today**, this one included. A
throwaway keypair was generated for this run. Its **public** half is committed beside this file
as `receipt-verification-key-THROWAWAY.pub`, so the `.minisig` here can actually be checked
(`minisign -V -m receipt.intoto.jsonl -p receipt-verification-key-THROWAWAY.pub`); its
**secret** half never left the review's scratchpad and is not committed. That key is not the
repository's signing key and must never be treated as one. That is finding `PRREV-DOGFOOD-01`, and it is exactly the state PRREV-005's
own conformance run recorded: the mechanics are proven, CI provenance is not.

## The ACCEPT is not vacuous — six negative controls, each re-signed

Each mutation below was applied to **this** receipt (not a synthetic fixture), re-signed so it
could only fail on the branch it names, and re-run:

| mutation of the accepted receipt | guard |
|---|---|
| `reviewer_actor.id := author_actor.id` | REJECT **[B2]** — *"a self-review is not a review (S5)"* |
| `cuda.status := not-triggered` | REJECT **[B1]** — *"its S3.B trigger fires on this diff (path tests/fixtures/pr-review/cuda-message-cases.tsv)"* |
| `verdict := PASS` with `pmat.status := unreachable` | REJECT **[B1]** — *"an unreachable source must be DEGRADED, not clean (S3.0)"* |
| a finding's `properties.grounding` deleted | REJECT **[B1]** — *"an unmarked claim is a defect in the review (S1)"* |
| a finding's `failure_scenario := ""` | REJECT **[B1]** — *"a finding that cannot name the failure it permits is a comment"* |
| the `asserted` finding's `precision_class := blocking` | REJECT **[B1]** — *"an asserted claim never blocks (S1)"* |

A no-op control (receipt rewritten through `jq` and re-signed, nothing changed) stayed **ACCEPT**,
so the guard is discriminating rather than refusing everything.

## What each consultation actually did

Every trigger was **recomputed from the diff with the guard's own predicates**, never judged by
eye, and both polarities of each predicate were controlled at the real input size first.

| consultation | status | how the trigger was decided |
|---|---|---|
| `pmat` | consulted (transport `cli`; `mcp: ConnectionRefused`) | unconditional (S3.A) |
| `cuda` | **consulted** | `--match-path` fires on **8 of 195** paths — all pr-review fixture filenames containing `cuda`. The message trigger does **not** fire over 88,051 bytes of commit message. `not-triggered` would have been a fixture-row-1 rejection, and was: see the negative-control table. |
| `crux` | consulted | `--match-crux-surface` fires on 12 of 18,776 added lines, all inside the guard's own `CRUX_SURFACE_RE` and its case table |
| `mutation` | consulted, scope `guard` | `--match-mutation-trigger` fires on 6 of 195 paths — the blocking 100%-kill row of S3.D |

The CUDA docs server **was asked**, not assumed: one query on `cudaStreamNonBlocking` /
`CU_STREAM_NON_BLOCKING`, which returned CUDA C++ Programming Guide S2.5.8 and **supports** the
one device-behaviour claim this diff carries (in `evidence/pr-review/backtest/`). Recorded as a
`cited` query with a verified `excerpt_sha256`, and raised as no finding, so that "asked and
confirmed" stays distinguishable from "did not ask".

## Why the verdict is DEGRADED and not PASS

Two measured reasons, both in `degraded_reason`:

1. The **pmat MCP transport was ConnectionRefused for the whole session.** The CLI answered and
   produced every pmat number here, so `status` is honestly `consulted` with `transport` and
   `transport_unavailable` both recorded — a live fallback must not erase a dead transport.
2. **pmat's quality analyzers reach 0 of the 195 changed files.** The diff has no `.rs` at all;
   `pmat analyze tdg` puts all 8 changed shell scripts in `ungraded_files` with the reason
   *"this build has no TDG analyzer for .sh"*, and its `language_distribution` has no shell entry.
   So `complexity_delta`, `tdg_delta` and `satd_introduced` are empty for lack of coverage, not
   for lack of defects. That is finding `PRREV-DOGFOOD-04`.

Spec S3.0 names the refused pmat MCP as its day-one example of row 3 (DEGRADED); SKILL.md S3.0(b)
says a CLI that answers makes it `consulted`. Both readings were true at once here. The receipt
resolves it toward the **spec**, which SKILL.md itself declares authoritative, and records the
divergence as finding `PRREV-DOGFOOD-06` rather than silently taking the reading with the cleaner
verdict.

## Author / reviewer separation, stated honestly

`reviewer_actor.id != author_actor.id` holds, and the separation is real at the level of session
and context: this review ran as a separate invocation in its own worktree with no access to the
authoring lanes' reasoning traces. It is **not** independent at the level of model family — the
same family wrote the diff — and S5's own cited rationale about self-preference applies to this
receipt. That is recorded in `reviewer_actor.note` rather than omitted, because presenting this as
full independence would be the self-flattery S5 exists to prevent.

## What this run does NOT show

- On this PR the merge-base and GitHub's `baseRefOid` are the **same commit** (the branch is 0
  commits behind `origin/main`), so the run exercises the merge-base path but does not
  discriminate it from the method that made PRREV-007 unreproducible. Only a PR that has fallen
  behind main would.
- `cost.input_tokens` / `cost.output_tokens` are `0` with `tokens_measured: false`. Neither the
  skill nor the guard gives a reviewer any way to read its own token usage (finding
  `PRREV-DOGFOOD-05`), so those zeros are flagged rather than presented as measurements.
- The signature proves whoever held the throwaway key produced this file. It does **not** prove
  the review was honest or complete. `attestation_level` is `L1-self` for that reason.

---

## Addendum — PRREV-013, 2026-08-31: this receipt now verifies under the repository default

Appended, not rewritten. Everything above is the transcript of the run as it happened at
`f5fe147`, including the two sentences that say the signature was a throwaway's — which
was true then and is the finding that produced this addendum.

**What changed:** `.github/pr-review.pub` is committed (PRREV-013), and
`receipt.intoto.jsonl` has been **re-signed with the secret half of that key**. The receipt
BYTES are unchanged — only the detached `.minisig` is new — so nothing above is restated by
the re-signing; it is the same document under a key a reader of this repository can actually
obtain.

```
$ bash scripts/check_pr_review_receipt.sh evidence/pr-review/2795/f5fe147971fd1e98dc84c2de6edd3dfd2d01736c
ACCEPT  evidence/pr-review/2795/f5fe147971fd1e98dc84c2de6edd3dfd2d01736c        # rc=0
```

with **no `PR_REVIEW_PUBKEY` override**. Before this addendum the same command was
`REJECT [B1] public key .github/pr-review.pub is absent`.

`receipt-verification-key-THROWAWAY.pub` is **deleted**: it can no longer verify the
signature beside it, and a public key that verifies nothing sitting next to a receipt is a
worse artifact than no key at all. The superseded signature line is preserved here so the
original signing event is not erased by its replacement:

```
# superseded signature over receipt.intoto.jsonl, throwaway key 4401F077A1B6314F
RURPMbahd/ABRJ5e1EKyoZmro1sVrz5W+fHFtgpxLp0T0T8tvFq1oqKozyeDsK7j9iEtTxGoZXnDNawLKfLn1NT6qvW/UbFn2wc=
```

Two sentences above are now historical rather than current, and are left standing rather
than edited: *"the signature proves whoever held the throwaway key produced this file"*, and
the verification instruction naming the throwaway. What has NOT changed is the sentence that
matters — `attestation_level` is still `L1-self`, and the signature still proves provenance
and not diligence.
