---
status: partial
ticket: R-5
issue: 2908
kind: code
branch: agent/R-5
model: claude-fable-5-1
---
# impl receipt — R-5: the cascade (1), the promotion gate (2), the asset workflow (3)

`status: partial`: the cascade script, its case table, its mutations and its contract are done and
measured. The rest of R-5 (five release assets, sha256 + minisign manifest, base-owned promotion
from four host receipts) and the merge are fleet- and queue-gated. Branched from `origin/main`
directly; the cascade depends on no other row.

## Claim
The cascade derives its publish set, orders it by dependency, refuses every unsafe precondition,
and is idempotent per crate.

## The publish set is derived, never listed
Every `cargo metadata` workspace member with `publish != false`, ordered so each crate follows the
workspace crates it depends on. Measured on this workspace:

| fact | value |
|---|---|
| crates in the derived set | 71 |
| ordering violations over normal and build deps | 0 |
| exit code of `--list` | 0 |

**A defect my own first cut had, found by running it on the real workspace rather than a fixture.**
The first version ordered on *all* dependencies and reported
`aprender-compute -> aprender-core -> aprender-compute` as a cycle, refusing to publish a workspace
that publishes fine. Dev-dependencies are not ordering edges: siblings here are path-only, cargo
drops them from the published manifest, and they are legally cyclic. The ordering now uses normal
and build dependencies only, and the case table gained a row asserting a dev-dep cycle is **not** a
cycle. A fixture-only test would have shipped the bug.

## Refusals (live mode publishes only if all five hold)
`git describe --exact-match` equals the tag · HEAD detached · tree clean · the GitHub release is not
a prerelease · `CARGO_REGISTRY_TOKEN` unset. An unknown release lookup refuses rather than assuming.
crates.io has no unpublish, so each of these guards something unrecoverable.

## Acceptance (`.pr/R-5/accept.sh`) — 5/5 GREEN
`--help` names the flags · `--self-test` 15/15 · `pv validate` 0 errors 0 warnings · `--list` derives
the set · `bashrs lint` 0 errors.

## Case table: 15 rows, both polarities
preflight all-clear · the four registered mutations, each on its own · wrong tag · no tag ·
unknown release · topological and drops `publish=false` · a private crate never appears · a real
cycle is exit 2 · a dev-dep cycle is not a cycle · malformed metadata is exit 2 · an
already-published version is skipped · a new one is not.

## Mutation (measured, not asserted)
| mutation | result |
|---|---|
| remove the `CARGO_REGISTRY_TOKEN` check | row 5 RED (15/15 → 14/15) |
| remove the detached-HEAD check | row 2 RED (15/15 → 14/15) |

The dirty-tree and prerelease checks are covered by rows 3 and 4 by the same construction; the two
above were run to show the construction actually flips rather than asserting that it would.

## A stop is a report, never a retry
The first non-zero `cargo publish` ends the run and prints the crate, what is live and what remains.
Re-running into a half-published set is how one release becomes two; every crate already on
crates.io at its workspace version is skipped, so a re-run after a fix is safe by construction.

## Part 2: `scripts/check_promotion_receipts.sh` — the base-owned promotion gate
A release is promoted from prerelease to stable only over four host receipts that each prove the
DOWNLOADED asset is the TESTED binary. Per host, all required: a receipt exists · its
`binary_sha256` is in the release's minisign-verified sha256 manifest · its C14 parity status is
PASS · its parity was not skipped (status `skipped`, or `SKIP_PARITY_GATE` in the recorded
command, refuses even beside a PASS — the 0.65.2 defect exactly). The manifest must verify
against the repository's public key with minisign, the scheme `.github/pr-review.pub` already
uses (S0-19: no third scheme); unsigned or unverifiable promotes nothing.

| leg | result |
|---|---|
| `--self-test` | 15/15, both polarities |
| `bashrs lint` | 0 errors |
| `pv validate contracts/apr-release-assets-v1.yaml` | valid, 0 errors 0 warnings |
| bare run (how `guard_tree.sh` invokes every `check_*.sh`) | `NOT-RUN … exit 0` — argument-wired; the promotion step supplies the tag |

The three registered mutations, **measured**:

| mutation (whole statement removed, `bash -n` clean, self-test rc checked) | rows that go RED |
|---|---|
| missing: the no-receipt branch becomes a silent `continue` | row 2 (15/15 → 14/15) |
| tampered: the manifest membership test removed | rows 3 and 11 (15/15 → 13/15) |
| skipped: the skipped test removed | row 5 (15/15 → 14/15) |

Row 4 (status `skipped`) does NOT depend on the skipped test: its status also falls into the
status case-arm's catch-all refusal, so it stays RED under this mutant. Row 5 (status PASS with
`SKIP_PARITY_GATE` in the recorded command) is the one only the skipped test catches, and it is
the one that matters — a receipt that says PASS while the gate was bypassed is the 0.65.2 shape.

**A defect in how I first measured this, recorded so it is not repeated.** The guard is a
one-line `if …; then …; fi`. My first mutant replaced only the `if` head, leaving a dangling
`fi`, so the self-test died with a **syntax error** (rc 2) and printed zero FAIL rows — which
reads exactly like "nothing went red". I had already written "rows 4 and 5 RED" into this receipt
from that non-result. Re-measured with the whole statement replaced, `bash -n` asserted clean, and
the self-test's own exit code checked. A mutant that does not parse is not a mutant.

A defect in my own first cut, caught by the case table: the public-key path was bound when the
script loaded, so the self-test's override was never seen and two rows refused on "public key
absent". Read lazily now. The real-signature row (15) is exercised with verification ON: a fake
`.minisig` does not verify, so the bypass the case table uses for the other rows cannot leak into
a live promotion.

What this half does NOT do: build or sign the assets, or produce the receipts. The public key
`.github/release-assets.pub` does not exist yet; it is created with the first signed manifest at
release time, and until then the gate refuses (absent key ⇒ nothing promotes), which is the
correct default.

## Part 3: `.github/workflows/release-assets.yml`
On a published release: `apr` is built for the five nightly.yml targets on hosted runners
(`--locked`; Windows without the visualization feature, as nightly does), plus a `--features cuda`
build on the gx10 runner. One sha256 manifest is written over every archive and signed with
minisign under the scheme the repo already uses for PR-review receipts — committed
`.github/release-assets.pub`, secret `RELEASE_ASSETS_SIGNING_KEY_B64` materialised to a file before
`minisign -S` (S0-19: no third scheme) — and verified against the committed public key BEFORE
upload. Archives, manifest and signature go to the release with `gh release upload`.

A separate, manually dispatched `promote` job downloads the signed manifest FROM THE RELEASE,
verifies it, runs `check_promotion_receipts.sh <tag> --manifest …` over the four host receipts
committed on main, and only on PASS runs `gh release edit --prerelease=false`. The gate therefore
reads nothing this workflow produced except the signed manifest, whose signature is what makes it
base-owned (I2).

| check | result |
|---|---|
| `actionlint` on release-assets.yml and ci.yml | clean, rc 0 |
| `cargo publish` literal in release-assets.yml | 0 (the phrase is kept out of every workflow on purpose) |
| `check_guards_are_wired.sh` | PASS (ratcheted): 110 scanned, 3 unwired, unchanged vs the baseline |
| merge onto the advanced main (505fd159e), throwaway-worktree oracle | CLEAN |

**Two facts that bound this half, recorded not hidden.**
- **No x86_64 cuda runner exists.** The registered cuda-capable runner is gx10 (aarch64). The
  x86_64 cuda asset D-10 would want cannot be produced by any workflow today; that asset is
  absent from the manifest by construction, not silently mislabelled. Building it on lambda
  requires registering lambda as a runner, which is an infra decision, not this row's.
- **The public key does not exist yet.** `.github/release-assets.pub` is created with the first
  signed manifest at release time (`minisign -G -W`, commit the .pub, set the secret). Until then
  the manifest job refuses (unset secret ⇒ exit 1) and the promotion gate refuses (absent key ⇒
  REFUSE). Both defaults are the correct direction.

**The cascade script is a named guard, and that is right.** `check_guards_are_wired.sh` derives
its universe from any script with a `--self-test` mode — "a claim to be a guard is what makes it
one" — and went RED on `publish_cascade.sh`. The honest wiring is its CASE TABLE, run in ci's
guard job on every PR; the live mode is never named by a workflow. A defect in how I first wired
it: my anchor step existed only on a sibling stack, the assertion fired, and a chained commit
landed WITHOUT the edit. Re-anchored on the tree being edited; the guard now passes.

**Owed at release, not here:** the receipts' directory for an rc tag (`evidence/dogfood/0.66.0-rc1`
vs `0.66.0`) is [U] until fleet-verify writes the first one; `check_multiplatform_dogfood.sh
--from-release <tag>` (named in #2908's acceptance) is an R-2-file change and is not in this PR.

## An observation for the DONE-IF, not a change
The driver's check "no workflow runs cargo publish" is written as `grep -rL 'cargo publish'`. That
literal grep FAILS here, but the intent HOLDS: all three occurrences
(`binary-release.yml:5`, `ci.yml:1051`, `ci.yml:1517`) are **comments**, not commands. Same
classifier class as PMAT-1074 — a textual token test over prose. Recorded rather than worked around.

## Gaps
`.github/release-assets.pub` + the secret (created at the first signed release); the x86_64 cuda
asset (no runner); `--from-release` in the dogfood script (R-2's file); `--dry-run` against
the real registry (it runs `cargo publish --dry-run` per crate, which needs a full build of 71
crates and is a release-time step); the live cascade itself at RELEASE; the 3-lane quorum; the CI
RED→GREEN mutation pair; the merge. Queue is under another session's lock by operator instruction.
