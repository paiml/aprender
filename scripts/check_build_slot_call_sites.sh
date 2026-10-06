#!/usr/bin/env bash
# check_build_slot_call_sites.sh -- every heavy cargo step on intel's CI pool takes
# a host build slot (C313/10-06 item 3; tool: scripts/ci/build_slot.sh).
#
# A slot only bounds the host if every cargo on it takes one. One unadmitted
# workspace build beside four admitted ones is the herd the pool exists to stop,
# and nothing about it looks wrong in a diff. So, text only (no build):
#
#   R1  In each section a fat job on the X64 clean-room pool runs (read from the
#       `--sections` lists in .github/workflows/ci.yml), a step that runs a heavy
#       cargo subcommand, or a known heavy driver (ci_guards.sh guard-cargo,
#       ci_run_explicit_test_commands.sh), has
#         shell: bash scripts/ci/build_slot.sh run -- <exactly its old shell> {0}
#   R2  The shell after `--` is one of the two the driver already runs
#       (`bash -e {0}` = no shell, `bash --noprofile --norc -eo pipefail {0}` =
#       `shell: bash`), so admission never changes how a step runs.
#   R3  The tool's test-only knobs (FLEET_BUILD_PRIO, FLEET_BUILD_SLOT_DIR,
#       FLEET_BUILD_SLOTS_FILE, FLEET_BUILD_LEDGER, FLEET_BUILD_SLOT_POLL_S)
#       appear in no workflow or section file: a tier comes from the event, never
#       from a line someone can type.
#
# Known gap, printed every run: sov.* sections run in a job `container:`, so their
# steps cannot see the host lock until the lock dir is provisioned and mounted
# (infra ticket T1). They are listed, not judged.
#
# R4 (C313): a new check lands report-only. Findings print and rc is 0; with
# --enforce (or BUILD_SLOT_SITES_ENFORCE=1) findings are rc 1. A file that
# cannot be read, or zero sections found, is rc 2: not measured, never a pass.
#
#   bash scripts/check_build_slot_call_sites.sh [--enforce] [ROOT]
#   bash scripts/check_build_slot_call_sites.sh --self-test
set -uo pipefail

HEAVY_RE='(^|[^A-Za-z0-9_-])cargo( \+[a-z0-9.-]+)? (build|b|test|t|nextest|check|c|clippy|mutants|run|r|install|llvm-cov|bench|doc|miri|rustc)([^A-Za-z0-9_-]|$)|scripts/ci_guards\.sh guard-cargo|scripts/ci_run_explicit_test_commands\.sh'
SHELL_OK_RE='^ *shell: bash scripts/ci/build_slot\.sh run (--label [A-Za-z0-9._:/@=+-]+ )?-- (bash -e|bash --noprofile --norc -eo pipefail) \{0\} *$'
KNOB_RE='FLEET_BUILD_(PRIO|SLOT_DIR|SLOTS_FILE|LEDGER|SLOT_POLL_S)'

# Sections run by fat jobs on [self-hosted, Linux, X64, clean-room], one per line,
# matrix pins stripped (`determinism[X64]` -> determinism, `ws?1?3?` -> ws).
pool_sections() {
    awk '
        /^  [a-z0-9-]+:$/ { pool = 0 }
        /^    runs-on:/ { pool = ($0 ~ /X64/ && $0 ~ /clean-room/ && $0 !~ /cuda/) }
        pool && /--sections / {
            s = $0; sub(/.*--sections +\047/, "", s); sub(/\047.*/, "", s)
            n = split(s, a, ",")
            for (i = 1; i <= n; i++) { x = a[i]; sub(/[\[?].*/, "", x); print x }
        }' "$1" | sort -u
}

# One line per step of the named sections: "<section>\t<line>\t<heavy 0|1>\t<shell line>".
steps_of() {
    HRE="$HEAVY_RE" awk -v want="$2" '
        BEGIN { hre = ENVIRON["HRE"]; n = split(want, w, " "); for (i = 1; i <= n; i++) ok[w[i]] = 1 }
        function flush() { if (cur != "") printf "%s\t%d\t%d\t%s\n", job, cur, heavy, sh; cur = "" }
        /^jobs:/ { inj = 1; next }
        inj && /^  [a-z][a-z0-9._-]*:$/ { flush(); job = substr($1, 1, length($1) - 1); insteps = 0; next }
        inj && /^    steps:/ { insteps = ok[job]; next }
        insteps && /^      - / { flush(); cur = NR; heavy = 0; sh = "" }
        insteps && cur != "" {
            s = $0; sub(/(^|[ \t])#.*/, "", s)
            if (s ~ hre) heavy = 1
            if (s ~ /^        shell:/) sh = s
        }
        END { flush() }' "$1"
}

# check ROOT -> prints findings, sets FINDINGS and GAPS; rc 2 when not measured.
check() {
    local root="$1" ci="$1/.github/workflows/ci.yml" sec="$1/ci/sections.yml" secs want f line
    FINDINGS=0 GAPS=0
    [ -r "$ci" ] && [ -r "$sec" ] || { printf 'NOT MEASURED: cannot read %s or %s\n' "$ci" "$sec"; return 2; }
    secs=$(pool_sections "$ci")
    [ -n "$secs" ] || { printf 'NOT MEASURED: no X64 clean-room --sections list in %s\n' "$ci"; return 2; }
    want=$(printf '%s\n' "$secs" | grep -v '^sov\.' | tr '\n' ' ')
    for f in $(printf '%s\n' "$secs" | grep '^sov\.'); do
        printf 'gap: %s runs in a job container; not judged until the lock dir is mounted (infra T1)\n' "$f"
        GAPS=$((GAPS + 1))
    done
    while IFS=$'\t' read -r job line heavy sh; do
        if [ "$heavy" = 1 ] && ! printf '%s\n' "$sh" | grep -Eq "$SHELL_OK_RE"; then
            printf 'R1 %s:%s %s: heavy cargo step without a build slot (shell: %s)\n' "ci/sections.yml" "$line" "$job" "${sh:-none}"
            FINDINGS=$((FINDINGS + 1))
        elif [ "$heavy" = 0 ] && printf '%s\n' "$sh" | grep -q 'build_slot\.sh' && ! printf '%s\n' "$sh" | grep -Eq "$SHELL_OK_RE"; then
            printf 'R2 %s:%s %s: build slot with a shell the driver does not run today (%s)\n' "ci/sections.yml" "$line" "$job" "$sh"
            FINDINGS=$((FINDINGS + 1))
        fi
    done < <(steps_of "$sec" "$want")
    while IFS= read -r line; do
        printf 'R3 %s: test-only knob in a workflow\n' "${line%%:*}:$(printf '%s' "$line" | cut -d: -f2)"
        FINDINGS=$((FINDINGS + 1))
    done < <(cd "$root" && grep -nE "$KNOB_RE" ci/sections.yml ci/vendor/*.yml .github/workflows/*.yml 2> /dev/null | grep -vE '^[^:]+:[0-9]+: *#')
    return 0
}

self_test() {
    local tmp fails=0 rows=0 rc out
    tmp=$(mktemp -d) || return 2
    row() { rows=$((rows + 1)); if [ "$2" = "$3" ]; then printf '  ok    %s\n' "$1"; else printf '  FAIL  %s: got [%s] want [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi; }
    fixture() { # fixture DIR STEP_SHELL STEP_RUN [EXTRA_CI_LINE]
        mkdir -p "$1/.github/workflows" "$1/ci/vendor"
        printf 'jobs:\n  x86-main:\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps:\n      - run: >\n          fat --sections %ssov.*,build[X64],light%s\n%s\n  gx10:\n    runs-on: [self-hosted, Linux, ARM64, cuda]\n    steps:\n      - run: fat --sections %sarmonly%s\n' "'" "'" "${4:-}" "'" "'" > "$1/.github/workflows/ci.yml"
        printf 'jobs:\n  build:\n    steps:\n      - name: heavy\n%s        run: %s\n      - name: light\n        run: echo hi\n  armonly:\n    steps:\n      - name: arm\n        run: cargo build --release\n' "${2:+        shell: $2
}" "$3" > "$1/ci/sections.yml"
        : > "$1/ci/vendor/sovereign-ci.yml"
    }
    run_case() { out=$(bash "$ME" --enforce "$1" 2>&1); rc=$?; }

    fixture "$tmp/1" "bash scripts/ci/build_slot.sh run -- bash -e {0}" "cargo test --workspace"
    run_case "$tmp/1"; row "admitted cargo step: green" "$rc" 0
    row "sov.* listed as a gap, not judged" "$(printf '%s' "$out" | grep -c '^gap: sov\.\*')" 1
    fixture "$tmp/2" "" "cargo nextest run --lib"
    run_case "$tmp/2"; row "unadmitted cargo step: R1 red" "$rc/$(printf '%s' "$out" | grep -c '^R1 ')" 1/1
    fixture "$tmp/3" "bash" "cargo +nightly build -p x"
    run_case "$tmp/3"; row "shell: bash without a slot: R1 red" "$rc" 1
    fixture "$tmp/4" "bash scripts/ci/build_slot.sh run -- bash -eo pipefail {0}" "cargo build"
    run_case "$tmp/4"; row "slot with a new shell (adds pipefail): red" "$rc" 1
    fixture "$tmp/5" "bash scripts/ci/build_slot.sh run --label build -- bash --noprofile --norc -eo pipefail {0}" "docker run img cargo clippy -p a"
    run_case "$tmp/5"; row "labelled slot, shell: bash form: green" "$rc" 0
    fixture "$tmp/6" "" "bash scripts/ci_guards.sh guard-cargo"
    run_case "$tmp/6"; row "indirect heavy driver unadmitted: red" "$rc" 1
    fixture "$tmp/7" "" "cargo fmt --all -- --check"
    run_case "$tmp/7"; row "cargo fmt is light: green" "$rc" 0
    fixture "$tmp/8" "" "echo # cargo build is only in a comment"
    run_case "$tmp/8"; row "cargo in a comment: green" "$rc" 0
    fixture "$tmp/9" "" "ls ~/.cargo/registry; echo cargo-home"
    run_case "$tmp/9"; row "a cargo path is not a cargo run: green" "$rc" 0
    fixture "$tmp/10" "bash scripts/ci/build_slot.sh run -- bash -e {0}" "cargo test" "    env: {FLEET_BUILD_PRIO: 0}"
    run_case "$tmp/10"; row "FLEET_BUILD_PRIO in a workflow: R3 red" "$rc/$(printf '%s' "$out" | grep -c '^R3 ')" 1/1
    fixture "$tmp/11" "bash scripts/ci/build_slot.sh run -- bash -e {0}" "cargo test" "    # FLEET_BUILD_SLOT_DIR is test-only"
    run_case "$tmp/11"; row "a knob named in a comment: green" "$rc" 0
    out=$(bash "$ME" "$tmp/2" 2>&1); rc=$?
    row "report-only: findings print, rc 0" "$rc/$(printf '%s' "$out" | grep -c '^R1 ')" 0/1
    mkdir -p "$tmp/12"
    run_case "$tmp/12"; row "missing files: not measured (2), never a pass" "$rc" 2
    fixture "$tmp/13" "" "cargo test"; sed -i 's/X64, clean-room/ARM64, clean-room/' "$tmp/13/.github/workflows/ci.yml"
    run_case "$tmp/13"; row "no X64 pool section list: not measured (2)" "$rc" 2

    rm -rf "${tmp:?}"
    printf 'check_build_slot_call_sites self-test: %s rows, %s failed\n' "$rows" "$fails"
    [ "$rows" -ge 15 ] || { printf 'only %s rows ran (want 15): NOT MEASURED\n' "$rows"; return 2; }
    [ "$fails" = 0 ]
}

ME="${BASH_SOURCE[0]}"
ENFORCE="${BUILD_SLOT_SITES_ENFORCE:-0}"
case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --enforce) ENFORCE=1; shift ;;
esac
ROOT="${1:-$(cd "$(dirname "$ME")/.." && pwd)}"
check "$ROOT"; rc=$?
[ "$rc" = 0 ] || exit "$rc"
if [ "$FINDINGS" = 0 ]; then
    printf 'PASS: every heavy cargo step on the X64 clean-room pool takes a build slot (%s gap(s) listed)\n' "$GAPS"
    exit 0
fi
if [ "$ENFORCE" = 1 ]; then
    printf 'FAIL: %s step(s) above (C313 item 3)\n' "$FINDINGS"
    exit 1
fi
printf 'REPORT-ONLY (C313 R4 ratchet): %s finding(s) above; blocks after three green nights\n' "$FINDINGS"
exit 0
