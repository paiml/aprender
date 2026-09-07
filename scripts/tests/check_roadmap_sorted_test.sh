#!/usr/bin/env bash
# check_roadmap_sorted_test.sh — falsifier for scripts/check_roadmap_sorted.sh
# (BSE-09a, PMAT-1065).
#
# Runs the guard against synthetic fixtures under mktemp -d (never against
# the real docs/roadmaps/roadmap.yaml — that's `check_roadmap_sorted.sh`'s
# own --self-test plus the plain invocation, both exercised here too), and
# proves the duplicate-id check is load-bearing by running a MUTANT copy of
# the guard — its duplicate-detection arm deleted — against the duplicate
# fixture and asserting the mutant goes GREEN where the real guard is RED.
# A falsifier that cannot fail against its own mutant is not evidence of
# anything (see the repo's capability-gated-tests-are-vacuous lesson).
#
#   bash scripts/tests/check_roadmap_sorted_test.sh
#
# Refs: PMAT-1065, BSE-09a.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD="$ROOT/scripts/check_roadmap_sorted.sh"

if [ ! -f "$GUARD" ]; then
    printf 'ENV - %s is missing (the box cannot answer)\n' "$GUARD" >&2
    exit 2
fi

TD=$(mktemp -d "${TMPDIR:-/tmp}/rmsorted-test.XXXXXX") || exit 2
cleanup() {
    local victim=${TD:-}
    case "$victim" in
        *rmsorted-test.*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
        *) return 0 ;;
    esac
}
trap cleanup EXIT

export GIT_TERMINAL_PROMPT=0
mkdir -p "$TD/.empty-template"
mkgitrepo() { # mkgitrepo DIR
    mkdir -p "$1"
    git -C "$1" init -q --template="$TD/.empty-template"
    git -C "$1" config user.email test@example.com
    git -C "$1" config user.name test
}

n=0
red=0

row() { # row WANT_RC LABEL MUST_MATCH BIN FILE
    local want=$1 label=$2 pat=$3 bin=$4 f=$5 rc=0 out
    n=$((n + 1))
    out=$(bash "$bin" "$f" 2>&1); rc=$?
    if [ "$rc" = "$want" ] && printf '%s' "$out" | grep -qE -- "$pat"; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=1
    fi
}

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
mkgitrepo "$TD/sorted"
cat >"$TD/sorted/roadmap.yaml" <<'YAML'
roadmap_version: '1.0'
github_enabled: true
github_repo: paiml/aprender
roadmap:
- id: PMAT-1
  title: one
- id: PMAT-2
  title: two
- id: PMAT-9
  title: nine
- id: PMAT-10
  title: ten
- id: GH-3
  title: gh three
- id: GH-4
  title: gh four
- id: Legacy freeform title, predates the id scheme
  title: legacy
YAML

mkgitrepo "$TD/unsorted"
cat >"$TD/unsorted/roadmap.yaml" <<'YAML'
roadmap:
- id: PMAT-945
  title: nine forty five
- id: PMAT-745
  title: seven forty five
- id: PMAT-744
  title: seven forty four
YAML

mkgitrepo "$TD/dup"
# The two PMAT-905 entries are ADJACENT and share the same numeral, so this
# fixture trips ONLY the duplicate-id check, never the sort-order check
# (905 is not < 905) — required so the mutation test below isolates the
# duplicate-check arm instead of accidentally being caught by sort-order.
cat >"$TD/dup/roadmap.yaml" <<'YAML'
roadmap:
- id: PMAT-905
  title: nine oh five
- id: PMAT-905
  title: nine oh five again, the duplicate arm this file exists to catch
- id: PMAT-906
  title: nine oh six
YAML

mkgitrepo "$TD/bak"
cat >"$TD/bak/roadmap.yaml" <<'YAML'
roadmap:
- id: PMAT-1
  title: one
YAML
cp "$TD/bak/roadmap.yaml" "$TD/bak/roadmap.yaml.bak"
touch "$TD/bak/roadmap.yaml.lock"
git -C "$TD/bak" add roadmap.yaml roadmap.yaml.bak roadmap.yaml.lock

# ---------------------------------------------------------------------------
# The real guard, both polarities per rule.
# ---------------------------------------------------------------------------
row 0 "sorted fixture: mixed PMAT/GH prefixes + one legacy freeform id" \
    '^PASS  .*7 top-level id' "$GUARD" "$TD/sorted/roadmap.yaml"
row 1 "out-of-order id (PMAT-945, PMAT-745, PMAT-744 — the BSE-09b shape)" \
    'FAIL.*sorts before' "$GUARD" "$TD/unsorted/roadmap.yaml"
row 1 "duplicate id (PMAT-905 twice)" \
    'FAIL  duplicate id PMAT-905' "$GUARD" "$TD/dup/roadmap.yaml"
row 1 "a tracked roadmap.yaml.bak / .lock (git add, no commit needed)" \
    'FAIL  tracked backup/lock file' "$GUARD" "$TD/bak/roadmap.yaml"
row 2 "a missing roadmap file is ENV (exit 2), never a pass" \
    'ENV' "$GUARD" "$TD/absent/roadmap.yaml"

# ---------------------------------------------------------------------------
# The guard's own --self-test must also be green (it is exercised in CI via
# this same file, not just ad hoc).
# ---------------------------------------------------------------------------
n=$((n + 1))
self_out=$(bash "$GUARD" --self-test 2>&1)
self_rc=$?
if [ "$self_rc" = 0 ] && printf '%s' "$self_out" | grep -qE '^[0-9]+/[0-9]+ checks, 0 failed$'; then
    printf 'ok    row %-2s rc=0  scripts/check_roadmap_sorted.sh --self-test is itself green\n' "$n"
else
    printf 'FAIL  row %-2s rc=%s  scripts/check_roadmap_sorted.sh --self-test\n' "$n" "$self_rc"
    printf '%s\n' "$self_out" | sed 's/^/        /'
    red=1
fi

# ---------------------------------------------------------------------------
# MUTATION: delete the guard's duplicate-detection arm (between its own
# marker comments) and confirm the resulting mutant goes GREEN on the exact
# duplicate fixture the real guard just went RED on. If it does not — if the
# mutant is still RED — this falsifier is not discriminating and the "RED"
# row above could be passing for an unrelated reason.
# ---------------------------------------------------------------------------
MUTANT="$TD/check_roadmap_sorted.mutant.sh"
sed '/# --- DUP-CHECK-BEGIN ---/,/# --- DUP-CHECK-END ---/d' "$GUARD" >"$MUTANT"
chmod +x "$MUTANT"

if ! diff -q "$GUARD" "$MUTANT" >/dev/null 2>&1; then
    n=$((n + 1))
    mut_out=$(bash "$MUTANT" "$TD/dup/roadmap.yaml" 2>&1)
    mut_rc=$?
    if [ "$mut_rc" = 0 ]; then
        printf 'ok    row %-2s rc=0  mutant (duplicate-check arm deleted) is GREEN on the duplicate fixture — falsifier discriminates\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s  mutant should have gone GREEN (rc=0) on the duplicate fixture but did not — the fixture proves nothing\n' "$n" "$mut_rc"
        printf '%s\n' "$mut_out" | sed 's/^/        /'
        red=1
    fi
else
    n=$((n + 1))
    printf 'FAIL  row %-2s  sed found no DUP-CHECK-BEGIN/END markers to delete — the mutant is byte-identical to the guard\n' "$n"
    red=1
fi

printf '%s/%s checks, %s failed\n' "$((n - red))" "$n" "$red"
[ "$red" = 0 ]
