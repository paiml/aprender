---
status: partial
ticket: R-5
issue: 2908
kind: code
branch: agent/R-5
model: claude-fable-5-1
---
# impl receipt — R-5 (part 1): scripts/publish_cascade.sh

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

## An observation for the DONE-IF, not a change
The driver's check "no workflow runs cargo publish" is written as `grep -rL 'cargo publish'`. That
literal grep FAILS here, but the intent HOLDS: all three occurrences
(`binary-release.yml:5`, `ci.yml:1051`, `ci.yml:1517`) are **comments**, not commands. Same
classifier class as PMAT-1074 — a textual token test over prose. Recorded rather than worked around.

## Gaps
The five assets, the sha256 + minisign manifest and the base-owned promotion; `--dry-run` against
the real registry (it runs `cargo publish --dry-run` per crate, which needs a full build of 71
crates and is a release-time step); the live cascade itself at RELEASE; the 3-lane quorum; the CI
RED→GREEN mutation pair; the merge. Queue is under another session's lock by operator instruction.
