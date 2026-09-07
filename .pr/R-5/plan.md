# R-5 plan (part 1 of the row) — scripts/publish_cascade.sh

**Issue** #2908 (inst:A). No blockers: the cascade script depends on no other row.

## Claim
`scripts/publish_cascade.sh` publishes the workspace to crates.io in dependency order, derives the
publish set rather than listing it, refuses every unsafe precondition, and is idempotent per crate.

## Rules (from the driver, each a falsifier)
- publish set = `cargo metadata` members with `publish != false`, **dependency-topological**, derived.
- `--dry-run` runs `cargo publish --dry-run` per crate and writes docs/audits/publish-cascade-dryrun.md.
- live mode **refuses** unless all of: `git describe --exact-match` == the tag · HEAD detached ·
  tree clean · `gh release view <tag> --json isPrerelease` false · `CARGO_REGISTRY_TOKEN` unset.
- per crate: `cargo search` already shows the version ⇒ skip; else ONE `cargo publish -p <crate>`;
  stop on the first non-zero; wait for index availability.
- a stop is a REPORT (crate, error, live set, remaining), never a retry into a half-published set.

## Acceptance (accept.sh)
1. **A1** `--help` names `--dry-run`, `--self-test` and the five refusals.
2. **A2** `--self-test` — case table, both polarities, including the four registered mutations.
3. **A3** `pv validate contracts/apr-publish-cascade-v1.yaml` through the pin.
4. **A4** the publish set is derived and topological: the script's `--list` prints members with
   `publish != false` and every crate appears after its intra-workspace dependencies.
5. **A5** bashrs lint: 0 errors.

## Registered mutations
branch (not detached) → refuse · dirty tree → refuse · prerelease release → refuse ·
`CARGO_REGISTRY_TOKEN` set → refuse. Each must flip its own self-test row.

## Owed (not in this PR)
The five release assets, sha256 + minisign manifest, and base-owned promotion from four receipts.
`--dry-run` against the real registry and the live cascade run at RELEASE time.

## K̂
[U] — second receipt of this class for A (R-2 was the first).
