#!/usr/bin/env bash
# check_determinism_wait.sh — T42: the determinism job never holds an ARM64 runner waiting for an X64
# raster that cannot come, and a missing raster is still a RED.
#
# Before T42 the determinism job started with the run, built its ARM64 raster in about a minute, then
# polled this run's artifact list every 20 s for up to FAT_ARTIFACT_WAIT_S (3600 s) for the X64 raster
# that x86-main uploads mid-run. When x86-main was queued past that hour or failed before uploading,
# one of the two ARM64 clean-room runners was held for the whole hour (~181 REST calls) and the job then
# went red on the timeout. Contract: contracts/ci-determinism-wait-v1.yaml.
#
# What the determinism job in ci.yml must say (rule W1-W5, `check`):
#   W1 needs: x86-main            the job starts only once the X64 producer has concluded
#   W2 if: ${{ !cancelled() }}    a failed x86-main still runs the job: missing X64 = RED, never a skip
#   W3 FAT_ARTIFACT_WAIT_S: "0"   one read of the artifact list, no poll loop
#   W4 FAT_EXPECT_ARTIFACTS lists determinism-X64 and determinism-ARM64   (the compare's inputs)
#   W5 the sections step still runs determinism[ARM64] and determinism-compare   (the verdict stays)
# The comparison itself is scripts/ci/determinism-compare.sh, unchanged; rows C1-C3 run it on planted
# receipts so a hash mismatch and a missing X64 receipt are measured RED here too.
#
# Usage:
#   check_determinism_wait.sh [ci.yml]   rules W1-W5 on that file (default .github/workflows/ci.yml); prints
#                                        the ARM64 seconds held per missing raster and REST calls per wait
#   check_determinism_wait.sh --selftest   case table (planted ci.yml copies + planted receipts)
#   check_determinism_wait.sh --mutants    each mutant of this checker must turn the table RED
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/check_determinism_wait.sh"
CMP="${DETERMINISM_COMPARE:-"$ROOT/scripts/ci/determinism-compare.sh"}"
POLL_S=20   # fat_driver.py uses_download sleeps 20 s between artifact-list reads
DRIVER_WAIT_DEFAULT=3600

# job <ci.yml>: the determinism job's lines, from its key to the next job key.
job() { awk '/^  determinism:$/{p=1; print; next} p && /^  [A-Za-z0-9_-]+:/{exit} p' "$1"; }

# wait_s <job-text>: the artifact wait the driver will use (its default when the job sets none).
wait_s() {
    local v
    v=$(printf '%s\n' "$1" | sed -n -E 's/^      FAT_ARTIFACT_WAIT_S: "?([0-9]+)"?[[:space:]]*$/\1/p' | head -n 1)
    printf '%s' "${v:-"$DRIVER_WAIT_DEFAULT"}"
}

check() {   # check <ci.yml>: rc 0 when W1-W5 hold
    local f=$1 j bad=0 w held calls
    [ -f "$f" ] || { printf 'ENV   no such file: %s\n' "$f"; return 2; }
    j=$(job "$f")
    [ -n "$j" ] || { printf 'FAIL  no determinism job in %s\n' "$f"; return 1; }
    printf '%s\n' "$j" | grep -q -E '^    needs: \[([a-z0-9-]+, )*x86-main(, [a-z0-9-]+)*\]$' ||   # m:noneeds
        { printf 'FAIL  W1 the determinism job does not need x86-main: it starts before the X64 raster can exist\n'; bad=1; }
    printf '%s\n' "$j" | grep -q -E '^    if: \$\{\{ !cancelled\(\)( && [^}]*)? \}\}$' ||   # m:noif
        { printf 'FAIL  W2 the determinism job is not if: !cancelled(): a failed x86-main would skip it, not RED it\n'; bad=1; }
    w=$(wait_s "$j")
    [ "$w" = 0 ] ||   # m:nowait
        { printf 'FAIL  W3 FAT_ARTIFACT_WAIT_S is %s, not "0": the job polls for a raster after its producer ended\n' "$w"; bad=1; }
    printf '%s\n' "$j" | grep -q -E '^      FAT_EXPECT_ARTIFACTS: determinism-X64,determinism-ARM64$' ||   # m:noexpect
        { printf 'FAIL  W4 FAT_EXPECT_ARTIFACTS no longer names both rasters\n'; bad=1; }
    printf '%s\n' "$j" | grep -q -F -e "--sections 'determinism[ARM64],determinism-compare'" ||   # m:nocompare
        { printf 'FAIL  W5 the job no longer runs determinism[ARM64] and determinism-compare\n'; bad=1; }
    # The numbers T42 is measured by: ARM64 seconds held by one raster that never comes, REST calls per wait.
    held="$w"   # with or without needs: x86-main, a raster that never comes holds the runner for the whole wait
    calls=$(( w / POLL_S + 1 ))
    printf 'held_s_per_missing_x64=%s rest_calls_per_wait=%s starts_after_x86_main=%s\n' \
        "$held" "$calls" "$(printf '%s\n' "$j" | grep -q -E '^    needs: .*x86-main' && echo yes || echo no)"
    [ "$bad" = 0 ] && printf 'ok    determinism job: starts after x86-main, never skipped, one artifact read\n'
    return "$bad"
}

# ---------------------------------------------------------------- self-test
PASS=0; FAIL=0
row() {   # row <id> <want: GREEN|RED> <got-rc> <text>
    local got=GREEN; [ "$3" = 0 ] || got=RED
    if [ "$got" = "$2" ]; then PASS=$((PASS + 1)); printf 'PASS  %s %s\n' "$1" "$4"
    else FAIL=$((FAIL + 1)); printf 'FAIL  %s %s (want %s, got %s rc %s)\n' "$1" "$4" "$2" "$got" "$3"; fi
}

# plant <out> <sed-expr>: a copy of the real ci.yml with one edit inside the determinism job; an edit that
# changes nothing is an error (exit 2), never a row that passes for the wrong reason.
plant() {
    sed -E "/^  determinism:\$/,/^  mac-check:\$/{$2}" "$CI" > "$1"
    if cmp -s "$1" "$CI"; then printf 'ENV   planted edit did not apply: %s\n' "$2"; exit 2; fi
}

receipt() {   # receipt <dir> <arch> <svg-hex-char>
    jq -n --arg a "$2" --arg s "$(printf '%064d' 0 | tr 0 "$3")" --arg p "$(printf '%064d' 0 | tr 0 b)" \
        '{host_arch:$a, os:"linux", svg_sha256:$s, svg_bytes:16754, png_sha256:$p, png_bytes:98390,
          manifest_root:$s, coord_grid:0.001, libm:"pure-rust"}' > "$1/determinism-$2.json"
}

selftest() {
    local d rc
    command -v jq > /dev/null || { printf 'ENV   jq is required\n'; exit 2; }
    d=$(mktemp -d); trap 'rm -rf "${d:?}"' RETURN
    CI="$ROOT/.github/workflows/ci.yml"

    check "$CI" > "$d/out" 2>&1; rc=$?
    row W0 GREEN "$rc" "the real ci.yml determinism job holds W1-W5"
    grep -q -x 'held_s_per_missing_x64=0 rest_calls_per_wait=1 starts_after_x86_main=yes' "$d/out"; rc=$?
    row N0 GREEN "$rc" "real ci.yml: 0 ARM64 seconds held, 1 REST call per wait, starts after x86-main"

    plant "$d/w1.yml" 's/^    needs: \[x86-main\]$/    needs: []/'
    check "$d/w1.yml" > /dev/null 2>&1; row W1 RED $? "determinism that does not need x86-main"
    plant "$d/w2.yml" '/^    if: \$\{\{ !cancelled\(\) \}\}$/d'
    check "$d/w2.yml" > /dev/null 2>&1; row W2 RED $? "no if: (default success()) skips the job when x86-main fails"
    plant "$d/w2b.yml" 's/^    if: \$\{\{ !cancelled\(\) \}\}$/    if: ${{ success() }}/'
    check "$d/w2b.yml" > /dev/null 2>&1; row W2b RED $? "if: success() skips the job when x86-main fails"
    plant "$d/w3.yml" '/^      FAT_ARTIFACT_WAIT_S: "0"$/d'
    check "$d/w3.yml" > /dev/null 2>&1; row W3 RED $? "no FAT_ARTIFACT_WAIT_S: the driver waits its 3600 s default"
    plant "$d/w3b.yml" 's/^      FAT_ARTIFACT_WAIT_S: "0"$/      FAT_ARTIFACT_WAIT_S: "600"/'
    check "$d/w3b.yml" > /dev/null 2>&1; row W3b RED $? "FAT_ARTIFACT_WAIT_S 600 polls after the producer ended"
    plant "$d/w4.yml" 's/^      FAT_EXPECT_ARTIFACTS: determinism-X64,determinism-ARM64$/      FAT_EXPECT_ARTIFACTS: determinism-ARM64/'
    check "$d/w4.yml" > /dev/null 2>&1; row W4 RED $? "the X64 raster dropped from the expected artifacts"
    plant "$d/w5.yml" "s/determinism\\[ARM64\\],determinism-compare/determinism[ARM64]/"
    check "$d/w5.yml" > /dev/null 2>&1; row W5 RED $? "the compare section dropped from the job"

    # The pre-T42 shape (no needs, no if, no wait override): the before numbers.
    plant "$d/old.yml" '/^    needs: \[x86-main\]$/d; /^    if: \$\{\{ !cancelled\(\) \}\}$/d; /^      FAT_ARTIFACT_WAIT_S: "0"$/d'
    check "$d/old.yml" > "$d/old.out" 2>&1; row B0 RED $? "the pre-T42 determinism job"
    grep -q -x 'held_s_per_missing_x64=3600 rest_calls_per_wait=181 starts_after_x86_main=no' "$d/old.out"; rc=$?
    row B1 GREEN "$rc" "pre-T42 numbers: 3600 ARM64 seconds held, 181 REST calls per missing X64 raster"

    # The comparison's verdicts, unchanged, on planted receipts (scripts/ci/determinism-compare.sh).
    mkdir -p "$d/c1" "$d/c2" "$d/c3"
    receipt "$d/c1" x86_64 a; receipt "$d/c1" aarch64 a
    bash "$CMP" "$d/c1" "$d/c1.json" > /dev/null 2>&1; row C1 GREEN $? "identical SVG digests on both hosts compare green (control)"
    receipt "$d/c2" x86_64 a; receipt "$d/c2" aarch64 d
    bash "$CMP" "$d/c2" "$d/c2.json" > /dev/null 2>&1; row C2 RED $? "a planted SVG hash mismatch is RED"
    receipt "$d/c3" aarch64 a
    bash "$CMP" "$d/c3" "$d/c3.json" > /dev/null 2>&1; row C3 RED $? "the X64 receipt missing is RED, never a pass"

    printf 'check_determinism_wait.sh --selftest: %s PASS, %s FAIL\n' "$PASS" "$FAIL"
    [ "$FAIL" = 0 ]
}

# ---------------------------------------------------------------- mutants
# Each mutant is a sed edit of this checker (or of the compare it runs); the planted copy's self-test
# must FAIL. An edit that does not apply is an ERROR (exit 2), never a survivor.
MUTANTS=(noneeds noif nowait noexpect nocompare cmpnosvg)   # each check line carries its tag: # m:<name>

mutants() {
    local d m name killed=0 total=0 s
    d=$(mktemp -d); trap 'rm -rf "${d:?}"' RETURN
    for name in "${MUTANTS[@]}"; do
        total=$((total + 1)); s="$d/$name.sh"
        if [ "$name" = cmpnosvg ]; then
            # Mutants of the comparison this table measures: a compare that calls every SVG pair identical must turn row C2 RED.
            m="$d/$name-cmp.sh"
            sed -e "s/svg_identical: (\[\.\[\]\[\$key\]\] | unique | length == 1),/svg_identical: true,/" "$CMP" > "$m"
            if cmp -s "$m" "$CMP"; then printf 'ERROR mutant %s did not apply\n' "$name"; exit 2; fi
            DETERMINISM_COMPARE="$m" bash "$SELF" --selftest > "$d/$name.out" 2>&1
        else
            sed -E "/# m:$name\$/s/^( +).*\$/\\1true ||   # mutated/" "$SELF" > "$s"
            if cmp -s "$s" "$SELF"; then printf 'ERROR mutant %s did not apply\n' "$name"; exit 2; fi
            mkdir -p "$d/$name-root/scripts/ci" "$d/$name-root/.github/workflows"
            cp "$s" "$d/$name-root/scripts/check_determinism_wait.sh"
            cp "$CMP" "$d/$name-root/scripts/ci/determinism-compare.sh"
            cp "$ROOT/.github/workflows/ci.yml" "$d/$name-root/.github/workflows/ci.yml"
            bash "$d/$name-root/scripts/check_determinism_wait.sh" --selftest > "$d/$name.out" 2>&1
        fi
        if [ $? != 0 ]; then killed=$((killed + 1)); printf 'killed    %s\n' "$name"
        else printf 'SURVIVED  %s\n' "$name"; fi
    done
    printf 'check_determinism_wait.sh --mutants: %s/%s killed\n' "$killed" "$total"
    [ "$killed" = "$total" ]
}

case "${1:-}" in
    --selftest) selftest ;;
    --mutants) mutants ;;
    -h|--help) sed -n '2,24p' "$SELF" ;;
    *) check "${1:-$ROOT/.github/workflows/ci.yml}" ;;
esac
