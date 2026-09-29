#!/usr/bin/env bash
# ladder_skip_is_red.sh — planted falsifier: a SKIPPED provable-ladder must be RED (EV-9,
# finding a2-fat-driver-skip-is-success; operator 2026-09-28 "skipped ≠ pass").
#
# Runs the real ci/sections.yml `provable-ladder` section through the real fat driver with
# LADDER_FORCE_SKIP=1, in a throwaway clone of HEAD, and requires the section result to be
# `failure`. Then plants the mutant (the "No verdict is not a pass" step deleted) and
# requires the same run to come back `success` — so the check is proven able to see the
# defect it guards, not just green.
#
# EXIT  0 forced skip is RED and the mutant is caught · 1 otherwise · 64 usage
#
#   bash scripts/ci/ladder_skip_is_red.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
STEP='No verdict is not a pass -- a skipped ladder is RED'
TD="$(mktemp -d)"
trap 'rm -rf "${TD:?}"' EXIT

# result_of <clone dir> -> the provable-ladder section's result, forced to skip.
result_of() {
    local ws="$1" rt="$TD/rt-$(basename "$1")"
    mkdir -p "$rt"
    ( cd "$ws" && env -u GITHUB_OUTPUT -u GITHUB_STEP_SUMMARY \
        GITHUB_WORKSPACE="$ws" RUNNER_TEMP="$rt" LADDER_FORCE_SKIP=1 \
        python3 scripts/ci/fat_driver.py run --sections provable-ladder \
        --results "$rt/results.json" > "$rt/driver.log" 2>&1 ) || true
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provable-ladder"]["result"])' \
        "$rt/results.json" 2>/dev/null || echo "no-result"
}

clone_at_head() {
    git clone -q --local "$ROOT" "$1"
    git -C "$1" -c advice.detachedHead=false checkout -q "$(git -C "$ROOT" rev-parse HEAD)"
}

fail=0
clone_at_head "$TD/real"
got="$(result_of "$TD/real")"
if [ "$got" = failure ]; then echo "PASS forced skip -> section $got (RED)"
else echo "FAIL forced skip -> section $got, want failure (a skip read as a pass)"; fail=1; fi

clone_at_head "$TD/mutant"
python3 - "$TD/mutant/ci/sections.yml" "$STEP" <<'PY'
import re, sys
p, name = sys.argv[1], sys.argv[2]
s = open(p).read()
pat = re.compile(r"      - name: " + re.escape(name) + r"\n(?:        .*\n|\s*\n)*?(?=      - name: )")
s2, n = pat.subn("", s)
if n != 1: sys.exit(f"mutant: step {name!r} matched {n} times")
open(p, "w").write(s2)
PY
git -C "$TD/mutant" -c core.hooksPath=/dev/null -c user.name=mutant -c user.email=m@x commit -qam "mutant: skip passes"
got="$(result_of "$TD/mutant")"
if [ "$got" = success ]; then echo "PASS mutant (step deleted) -> section $got: the check sees the defect"
else echo "FAIL mutant -> section $got, want success (the falsifier cannot see a skip-pass)"; fail=1; fi

exit "$fail"
