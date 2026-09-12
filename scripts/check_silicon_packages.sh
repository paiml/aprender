#!/usr/bin/env bash
# Every package the silicon lane says it tests must exist in this workspace.
#
# WHY THIS EXISTS.
#
# `Silicon Nightly` shipped naming `-p aprender-primitives` on BOTH of its test
# legs. There has never been a package by that name in this repo — `git log -S`
# over all of history finds it only in the workflow file that introduced it. So
# the lane's first scheduled run (2026-08-30 03:30 UTC) did this on each leg:
#
#     cargo test -p aprender-compute --release --lib   # 3510 passed
#     cargo test -p aprender-primitives --release --lib
#     error: package ID specification `aprender-primitives` did not match any packages
#     ##[error]Process completed with exit code 101
#
# Both legs green on the real work, both legs red on a name. The lane has never
# had a healthy run and could not have had one, and nothing in this repo noticed
# — it surfaced five days later in ANOTHER repo, as `aprender/Silicon Nightly is
# DEAD (no-healthy 999d)` from paiml/infra's dead-man's switch (PMAT-185).
#
# WHAT THE EXISTING GUARD DID NOT COVER. `check_silicon_coverage.sh` probes the
# INSTRUMENT thoroughly — it self-tests its label matcher against committed
# fixtures and refuses to report on an empty runner listing. It asks "can a
# runner serve this axis?". Nothing asked "is the work this lane performs even
# resolvable?", so the lane passed every probe it had and then died on a typo
# nine minutes into a 90-minute leg.
#
# The check is cheap and belongs in the `coverage` job, which is first and
# blocking and runs on intel: a name that cannot resolve should cost seconds on
# a CPU runner, not a GPU leg at 03:30.
#
# ONE SOURCE OF TRUTH, DELIBERATELY. The package names are read out of the
# workflow file itself, never kept in a list beside it. A second list is what
# this class of defect is made of.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW="${SILICON_WORKFLOW:-$REPO_ROOT/.github/workflows/silicon-nightly.yml}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

# The `-p <name>` arguments the text on stdin asks cargo for, deduplicated.
# Stdin, not a path, so the self-test needs no temp file to leak.
package_refs() {
    grep -oE '(^|[[:space:]])-p[[:space:]]+[A-Za-z0-9_.-]+' 2>/dev/null \
        | awk '{ print $NF }' \
        | sort -u
}

# The packages this workspace actually defines. `--no-deps` keeps it to members;
# `--locked` keeps a read-only guard read-only — without it `cargo metadata`
# rewrites Cargo.lock (measured here: 44 added `[[patch.unused]]` lines), which
# a guard has no business doing and a lockfile gate would rightly fail on.
workspace_packages() {
    cargo metadata --no-deps --format-version 1 --locked --manifest-path "$REPO_ROOT/Cargo.toml" 2>/dev/null \
        | jq -r '.packages[].name' \
        | sort -u
}

# PROBE THE INSTRUMENT WITH AN INPUT IT MUST REJECT. A guard that has never been
# shown capable of failing is not evidence of anything: this one's whole job is
# to say NO to a name, so it is handed a name to say no to, and a real one it
# must accept. `--self-test` alone runs only this and exits.
self_test() {
    local fails=0 got fixture
    fixture='      - name: a leg naming one real package and one invented one
        run: |
          cargo test -p aprender-compute --release --lib
          cargo test -p definitely-not-a-real-package --release --lib'
    got="$(package_refs <<<"$fixture" | paste -sd' ' -)"
    # sort -u, so the expectation is alphabetical, not source order.
    if [ "$got" != "aprender-compute definitely-not-a-real-package" ]; then
        echo "  self-test: package_refs -> '$got'" >&2
        echo "  expected 'aprender-compute definitely-not-a-real-package'" >&2
        fails=$((fails + 1))
    fi
    # Text with no `-p` at all must yield nothing, not everything. A reader that
    # returns the empty set for every input would pass the check above too.
    fixture='run: cargo test --release --lib'
    got="$(package_refs <<<"$fixture" | paste -sd' ' -)"
    if [ -n "$got" ]; then
        echo "  self-test: text with no -p yielded '$got'" >&2
        fails=$((fails + 1))
    fi
    if [ "$fails" -ne 0 ]; then
        echo "check-silicon-packages: SELF-TEST FAILED ($fails case(s))" >&2
        return 2
    fi
    echo "check-silicon-packages: self-test OK (2 cases, the instrument can say no)"
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

self_test || exit 2

if [ ! -f "$WORKFLOW" ]; then
    echo "::error::check-silicon-packages: no workflow at $WORKFLOW"
    echo "  A guard that cannot find its input has not checked anything." >&2
    exit 2
fi

members="$(workspace_packages)"
member_n="$(grep -c . <<<"$members")"
# BLIND IS A NO-GO, not a pass. Zero members means cargo metadata failed, and
# "no package matched" would then be true of every name including the real ones.
if [ "$member_n" -eq 0 ]; then
    echo "::error::check-silicon-packages: cargo metadata listed 0 workspace members"
    echo "  Cannot tell a bad package name from a broken manifest; refusing to report." >&2
    exit 2
fi

refs="$(package_refs <"$WORKFLOW")"
ref_n=0
[ -n "$refs" ] && ref_n="$(grep -c . <<<"$refs")"

bad=0
while IFS= read -r pkg; do
    [ -n "$pkg" ] || continue
    if ! grep -qxF -- "$pkg" <<<"$members"; then
        echo "::error::check-silicon-packages: $(basename "$WORKFLOW") tests \`-p $pkg\`, which is not a package in this workspace"
        bad=$((bad + 1))
    fi
done <<<"$refs"

# PRINT THE DENOMINATOR. "0 violations" over 0 references is the shape this
# guard exists to end, so say what was actually examined.
line="check-silicon-packages: $ref_n package reference(s) in $(basename "$WORKFLOW") against $member_n workspace member(s), $bad unresolvable"
echo "$line"
echo "$line" >>"$SUMMARY"

if [ "$bad" -ne 0 ]; then
    echo "  A lane cannot test a package that does not exist. Fix the name, or"
    echo "  drop the leg — but do not leave the lane asserting against nothing."
    exit 1
fi

# A lane that names NO packages is not passing this check, it is skipping it.
if [ "$ref_n" -eq 0 ]; then
    echo "::error::check-silicon-packages: the lane names no packages at all"
    echo "  Either the legs stopped testing anything, or the reader stopped reading." >&2
    exit 1
fi

echo "OK: every package the silicon lane names exists in this workspace."
