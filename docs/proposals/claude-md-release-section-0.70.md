# Proposed CLAUDE.md release-section text (#4045 M6): for OPERATOR review, not applied

The release cop asked for CLAUDE.md's release section to reflect the 0.70+ gate. A session does not edit CLAUDE.md on a
peer's request, so this is a proposal. The operator applies it, edits it, or declines it at merge.

Proposed addition, under "## CI/CD" (or a new "## Release gate (0.70+)"):

```markdown
## Release gate (0.70+) — APR-RELEASE-001 §14

A release is judged by `scripts/check_model_ladder.sh --scope release --nightly <root> --crux <smoke dir>`:
CRUX smoke on the RELEASE BINARY (every host x certified model x admitted mode, positive control GREEN) AND a GREEN
nightly long certification (`scripts/certify_nightly.sh`: full ladder + full CRUX, both lanes) <= 24 h old at an
ancestor of the cut, bound to the cut by equivalence or the #4037 carry-forward. The full sweep runs NIGHTLY, never on
release night. From the freeze, every publish-blocking gate runs continuously on the candidate (shift-left, §14.3):
real gates andon, bookkeeping gates auto-fix and never block the publish.
```
