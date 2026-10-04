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
# What the determinism job in ci.yml must say (rules W1-W5, `check`):
#   W1 needs: x86-main            the job starts only once the X64 producer has concluded
#   W2 if: ${{ !cancelled() }}    a failed x86-main still runs the job: missing X64 = RED, never a skip
#   W3 FAT_ARTIFACT_WAIT_S: "0"   one read of the artifact list, no poll loop (no step-level override either)
#   W4 FAT_EXPECT_ARTIFACTS lists determinism-X64 and determinism-ARM64   (the compare's inputs)
#   W5 the sections step still runs determinism[ARM64] and determinism-compare   (the verdict stays)
# and the rest of the chain that turns a missing raster into a RED, read as source (structural, not run):
#   W6 fat_driver.py uses_download reads FAT_ARTIFACT_WAIT_S and returns failure on the first read past
#      its deadline, before any sleep
#   W7 the gate needs determinism, runs if: always(), and reads DET:determinism-compare
#   W8 the determinism-compare section in ci/sections.yml carries no continue-on-error
#   W9 its download pattern matches determinism-X64 and determinism-ARM64 and not its own
#      determinism-receipt, which a re-run of the run still lists
# The comparison itself is scripts/ci/determinism-compare.sh, unchanged; rows C1-C3 run it on planted
# receipts so a hash mismatch and a missing X64 receipt are measured RED here too.
#
# Usage:
#   check_determinism_wait.sh [ci.yml [fat_driver.py [sections.yml]]]   rules W1-W9 (defaults: this repo's
#                                        files); prints the ARM64 seconds held per missing raster once the
#                                        job has started, and REST calls per wait
#   check_determinism_wait.sh --selftest   case table (planted ci.yml copies + planted receipts)
#   check_determinism_wait.sh --mutants    each mutant of this checker must turn the table RED
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/check_determinism_wait.sh"
CMP="${DETERMINISM_COMPARE:-"$ROOT/scripts/ci/determinism-compare.sh"}"
POLL_S=20   # fat_driver.py uses_download sleeps 20 s between artifact-list reads
DRIVER_WAIT_DEFAULT=3600
DRIVER="${FAT_DRIVER:-"$ROOT/scripts/ci/fat_driver.py"}"
SECTIONS="${FAT_SECTIONS:-"$ROOT/ci/sections.yml"}"

# job <ci.yml>: the determinism job's lines, from its key to the next job key.
job() { awk '/^  determinism:$/{p=1; print; next} p && /^  [A-Za-z0-9_-]+:/{exit} p' "$1"; }

# wait_s <job-text>: the artifact wait the driver will use (its default when the job sets none).
wait_s() {
    local v
    v=$(printf '%s\n' "$1" | sed -n -E 's/^      FAT_ARTIFACT_WAIT_S: "?([0-9]+)"?[[:space:]]*$/\1/p' | head -n 1)
    printf '%s' "${v:-"$DRIVER_WAIT_DEFAULT"}"
}

check() {   # check <ci.yml>: rc 0 when W1-W5 hold
    local f=$1 drvf="${2:-"$DRIVER"}" secf="${3:-"$SECTIONS"}" j g nc bad=0 w held calls n_step drv n_gate n_coe pat
    [ -f "$f" ] || { printf 'ENV   no such file: %s\n' "$f"; return 2; }
    [ -f "$drvf" ] && [ -f "$secf" ] || { printf "ENV   missing %s or %s\n" "$drvf" "$secf"; return 2; }
    j=$(job "$f")
    [ -n "$j" ] || { printf 'FAIL  no determinism job in %s\n' "$f"; return 1; }
    grep -q -E '^    needs: \[([a-z0-9-]+, )*x86-main(, [a-z0-9-]+)*\]$' <<<"$j" ||   # m:noneeds
        { printf 'FAIL  W1 the determinism job does not need x86-main: it starts before the X64 raster can exist\n'; bad=1; }
    grep -q -E '^    if: \$\{\{ !cancelled\(\)( && [^}]*)? \}\}$' <<<"$j" ||   # m:noif
        { printf 'FAIL  W2 the determinism job is not if: !cancelled(): a failed x86-main would skip it, not RED it\n'; bad=1; }
    w=$(wait_s "$j")
    [ "$w" = 0 ] ||   # m:nowait
        { printf 'FAIL  W3 FAT_ARTIFACT_WAIT_S is %s, not "0": the job polls for a raster after its producer ended\n' "$w"; bad=1; }
    grep -q -E '^      FAT_EXPECT_ARTIFACTS: determinism-X64,determinism-ARM64$' <<<"$j" ||   # m:noexpect
        { printf 'FAIL  W4 FAT_EXPECT_ARTIFACTS no longer names both rasters\n'; bad=1; }
    nc=$(grep -v -E '^[[:space:]]*#' <<<"$j")
    grep -q -F -e "--sections 'determinism[ARM64],determinism-compare'" <<<"$nc" ||   # m:nocompare
        { printf 'FAIL  W5 the job no longer runs determinism[ARM64] and determinism-compare\n'; bad=1; }
    # W3 at step level: a step env that sets the wait to anything but 0 overrides the job's "0".
    n_step=$(printf '%s\n' "$j" | grep -E '^ +FAT_ARTIFACT_WAIT_S:' | grep -c -v -E 'FAT_ARTIFACT_WAIT_S: "?0"?[[:space:]]*$')
    [ "$n_step" = 0 ] ||   # m:nostepwait
        { printf 'FAIL  W3 a step-level FAT_ARTIFACT_WAIT_S overrides the job-level "0"\n'; bad=1; }
    # W6 the driver: the deadline is exactly now + FAT_ARTIFACT_WAIT_S, and the first read past it returns
    # failure with no sleep before it (missing raster = RED).
    awk '/^def uses_download\(/{p=1; next} p && /^def /{exit}
         p && /^    deadline = time\.time\(\) \+ float\(os\.environ\.get\("FAT_ARTIFACT_WAIT_S", "3600"\)\)$/{dl=1; next}
         p && dl && !br && /time\.sleep\(/{exit}
         p && dl && /if CANCELLED\.is_set\(\) or time\.time\(\) > deadline:/{br=1; next}
         p && br && /time\.sleep\(/{exit}
         p && br && /^ +return False, \{\}$/{ok=1; exit}
         END{exit !ok}' "$drvf"; drv=$?
    [ "$drv" = 0 ] ||   # m:nodriver
        { printf 'FAIL  W6 %s uses_download no longer fails on the first read past its deadline\n' "$drvf"; bad=1; }
    # W7 the gate: it still needs determinism, runs always(), and reads DET:determinism-compare.
    g=$(awk '/^  gate:$/{p=1; print; next} p && /^  [A-Za-z0-9_-]+:/{exit} p' "$f" | grep -v -E '^[[:space:]]*#')
    n_gate=$(printf '%s\n' "$g" | grep -c -E '^    needs: \[(.*, )?determinism(, .*)?\]$|^    if: always\(\)$|^ +for pair in .*DET:determinism-compare')
    [ "$n_gate" = 3 ] ||   # m:nogate
        { printf 'FAIL  W7 the gate no longer needs determinism under always() and reads DET:determinism-compare\n'; bad=1; }
    # W8 the compare section: a continue-on-error would turn a failed download into a pass.
    n_coe=$(awk '/^  determinism-compare:$/{p=1; next} p && /^  [A-Za-z0-9_-]+:/{exit} p' "$secf" |
        grep -v -E '^[[:space:]]*#' | grep -c 'continue-on-error')
    [ "$n_coe" = 0 ] ||   # m:nocoe
        { printf 'FAIL  W8 the determinism-compare section carries continue-on-error\n'; bad=1; }
    # W9 the compare's download: both host receipts and nothing else. The compare uploads its own
    # determinism-receipt, which stays listed for the run, so `determinism-*` hands a re-run 3 receipts.
    pat=$(awk '/^  determinism-compare:$/{p=1; next} p && /^  [A-Za-z0-9_-]+:/{exit} p' "$secf" |
        sed -n -E 's/^ +pattern: ([^ #]+)[[:space:]]*(#.*)?$/\1/p' | head -n 1)
    # shellcheck disable=SC2053 # $pat is a glob on purpose: the driver's fnmatch reads it the same way
    [[ -n "$pat" && determinism-X64 == $pat && determinism-ARM64 == $pat && determinism-receipt != $pat ]] ||   # m:nopattern
        { printf 'FAIL  W9 the compare download pattern %s is not exactly the two host receipts\n' "${pat:-<none>}"; bad=1; }
    # The numbers T42 is measured by: ARM64 seconds held by one raster that never comes, REST calls per wait.
    held="$w"   # derived from the wait W3 checks, not measured; that the driver honours it is W6 (structural)
    calls=$(( w / POLL_S + 1 ))
    printf 'held_s_per_missing_x64=%s rest_calls_per_wait=%s starts_after_x86_main=%s\n' \
        "$held" "$calls" "$(grep -q -E '^    needs: .*x86-main' <<<"$j" && echo yes || echo no)"
    [ "$bad" = 0 ] && printf 'ok    determinism job: starts after x86-main, never skipped, one artifact read; driver, gate and compare section read it RED\n'
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

plant_file() {   # plant_file <src> <out> <sed-expr>: the same, for any file; a no-op edit exits 2
    sed -E "$3" "$1" > "$2"
    if cmp -s "$1" "$2"; then printf 'ENV   planted edit did not apply to %s: %s\n' "$1" "$3"; exit 2; fi
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

    plant "$d/w3c.yml" "s/^( +)(.*--sections 'determinism\\[ARM64\\],determinism-compare'.*)\$/\\1\\2\\n          FAT_ARTIFACT_WAIT_S: \"3600\"/"
    check "$d/w3c.yml" > /dev/null 2>&1; row W3c RED $? "a step-level FAT_ARTIFACT_WAIT_S 3600 under the job's \"0\""
    plant "$d/w5b.yml" "s/^( +)(.*)--sections 'determinism\\[ARM64\\],determinism-compare'(.*)\$/\\1# --sections 'determinism[ARM64],determinism-compare'\\n\\1\\2--sections 'determinism[ARM64]'\\3/"
    check "$d/w5b.yml" > /dev/null 2>&1; row W5b RED $? "the compare section named only in a comment"

    # The rest of the chain that makes a missing raster RED: driver, gate, compare section.
    plant_file "$DRIVER" "$d/d1.py" '/^def uses_download\(/,/^def /{s/^( +)return False, \{\}$/\1return True, {}/}'
    check "$CI" "$d/d1.py" > /dev/null 2>&1; row D1 RED $? "a driver that passes the download when the raster never came"
    plant_file "$DRIVER" "$d/d2.py" '/^def uses_download\(/,/^def /{s/FAT_ARTIFACT_WAIT_S/FAT_ARTIFACT_WAIT_SECONDS/}'
    check "$CI" "$d/d2.py" > /dev/null 2>&1; row D2 RED $? "a driver that no longer reads FAT_ARTIFACT_WAIT_S (the job's \"0\" would be ignored)"
    plant_file "$DRIVER" "$d/d3.py" '/^def uses_download\(/,/^def /{s/^(    deadline = time\.time\(\) \+ float\(os\.environ\.get\("FAT_ARTIFACT_WAIT_S", "3600"\)\))$/\1 + 7200/}'
    check "$CI" "$d/d3.py" > /dev/null 2>&1; row D3 RED $? "a driver deadline padded past the job's wait (+ 7200)"
    plant_file "$DRIVER" "$d/d4.py" '/^def uses_download\(/,/^def /{s/^(        if CANCELLED\.is_set\(\) or time\.time\(\) > deadline:)$/        time.sleep(20)\n\1/}'
    check "$CI" "$d/d4.py" > /dev/null 2>&1; row D4 RED $? "a driver that sleeps before its deadline check"
    plant_file "$CI" "$d/g1.yml" '/^  gate:$/,/^  [a-z]/{s/ DET:determinism-compare//}'
    check "$d/g1.yml" > /dev/null 2>&1; row G1 RED $? "a gate that no longer reads DET:determinism-compare"
    plant_file "$CI" "$d/g2.yml" '/^  gate:$/,/^    steps:$/{s/^    if: always\(\)$/    if: success()/}'
    check "$d/g2.yml" > /dev/null 2>&1; row G2 RED $? "a gate that is skipped when determinism fails"
    plant_file "$SECTIONS" "$d/s1.yml" '/^  determinism-compare:$/a\    continue-on-error: true'
    check "$CI" "$DRIVER" "$d/s1.yml" > /dev/null 2>&1; row S1 RED $? "a compare section with continue-on-error"
    plant_file "$SECTIONS" "$d/s2.yml" '/^  determinism-compare:$/,/^  [a-z]/{s/^( +pattern: ).*$/\1determinism-*/}'
    check "$CI" "$DRIVER" "$d/s2.yml" > /dev/null 2>&1; row S2 RED $? "a compare download of determinism-* (also takes its own earlier receipt)"
    plant_file "$SECTIONS" "$d/s3.yml" '/^  determinism-compare:$/,/^  [a-z]/{s/^( +pattern: ).*$/\1determinism-A*64/}'
    check "$CI" "$DRIVER" "$d/s3.yml" > /dev/null 2>&1; row S3 RED $? "a compare download that misses determinism-X64"

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
    # What W9 prevents: the compare's own receipt from an earlier attempt beside the two host receipts.
    mkdir -p "$d/c4"; receipt "$d/c4" x86_64 a; receipt "$d/c4" aarch64 a; bash "$CMP" "$d/c1" "$d/c4/determinism-receipt.json" > /dev/null 2>&1
    bash "$CMP" "$d/c4" "$d/c4.json" > /dev/null 2>&1; row C4 RED $? "an earlier attempt's determinism-receipt beside both host receipts is RED"

    printf 'check_determinism_wait.sh --selftest: %s PASS, %s FAIL\n' "$PASS" "$FAIL"
    [ "$FAIL" = 0 ]
}

# ---------------------------------------------------------------- mutants
# Each mutant is a sed edit of this checker (or of the compare it runs); the planted copy's self-test
# must FAIL. An edit that does not apply is an ERROR (exit 2), never a survivor.
MUTANTS=(noneeds noif nowait noexpect nocompare nostepwait nodriver nogate nocoe nopattern cmpnosvg)   # each check line carries its tag: # m:<name>

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
            mkdir -p "$d/$name-root/ci"; cp "$DRIVER" "$d/$name-root/scripts/ci/fat_driver.py"; cp "$SECTIONS" "$d/$name-root/ci/sections.yml"
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
    -h|--help) sed -n '2,30p' "$SELF" ;;
    *) check "${1:-$ROOT/.github/workflows/ci.yml}" ;;
esac
