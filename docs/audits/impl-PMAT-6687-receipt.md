# PMAT-6687 implementation receipt (PR #6685)

Code head under test: 77e2c9761 (parent of the commit that adds this file). Run from the repo root, worktree detached at that commit.

## Full run

    $ bash scripts/release/falsify_package_lane.sh; echo rc=$?
    falsify_package_lane: 77 rows held, 0 failed
    rc=0

## Mutants run

    $ bash scripts/release/falsify_package_lane.sh --mutants; echo rc=$?
    falsify_package_lane: 77 rows held, 0 failed
    control: unmutated copy passes every row (falsify_package_lane: 77 rows held, 0 failed)
    mutant 1 (verify comparator inverted) caught by: FAIL chain rc -- rc=1, wanted 0: a (#3) is listed before its dependency b (#2) b (#2) is listed before its dep
    mutant 2 (dev edges counted) caught by: FAIL dev-dependency cycle is not a cycle -- rc=1, wanted 0: package_lane: dependency cycle (or downstream of o
    mutant 3 (level ordering dropped) caught by: FAIL chain rc -- rc=1, wanted 0: a (#1) is listed before its dependency b (#2) b (#2) is listed before its dep
    mutant 4 (missing check dead) caught by: FAIL missing crate refused -- rc=0, wanted 1: package_lane: order ok: 2 crate(s) in /tmp/tmp.pim9W7Do4G/fx/mis
    mutant 5 (--no-verify dropped) caught by: FAIL cargo flags -- every crate must be packaged with --no-verify --locked, in that order
    mutant 6 (--locked dropped) caught by: FAIL cargo flags -- every crate must be packaged with --no-verify --locked, in that order
    mutant 7 (vacuity floor dead) caught by: FAIL below --min-crates is red -- rc=0, wanted 1: package_lane: order ok: 3 crate(s) in /tmp/tmp.iHhbZHKJlI/o1
    mutant 8 (publish filter dropped) caught by: FAIL publish=false / other registry excluded, outside deps ignored -- order '', wanted 'a z'
    mutant 9 (scope always full) caught by: FAIL receipt scope -- an --only run must carry scope only:a,c, got "full"
    mutant 10 (publishable() always true) caught by: FAIL publish=false / other registry excluded, outside deps ignored -- order '', wanted 'a z'
    mutant 11 (TIERS[] table deleted) caught by: FAIL real TIERS[] table present -- no TIERS[] rows in /tmp/tmp.NfeX8LlNQv/mroot/scripts/cascade-nontiers.sh: a
    mutants: 11 of 11 caught
    rc=0

Counts: 77 rows held, 11 of 11 mutants caught (mutant 11 deletes TIERS[] from a fixture copy of cascade-publish.sh and is caught by the counted row 'real TIERS[] table present').
