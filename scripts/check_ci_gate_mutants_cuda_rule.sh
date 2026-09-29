#!/usr/bin/env bash
# check_ci_gate_mutants_cuda_rule.sh -- the `gate` job is RED on a pull_request whose CPU mutants section
# deferred a NOT_MEASURED set, unless mutants-cuda succeeded and measured that same set (#4621, operator E1).
#
# cargo-mutants on clean-room builds default features, so a mutant in `#[cfg(feature = "cuda")]` code is never
# compiled there. The CPU section defers those (outputs not_measured, not_measured_sha); the yoga shard
# mutants-cuda measures the cuda-gated files (outputs measured_sha). The rule compares the two shas.
#
# This guard does not re-implement the rule. It EXTRACTS the block between the GATE-MUTANTS-CUDA-RULE-BEGIN/END
# markers from .github/workflows/ci.yml and executes it for every row of the table, so it judges the code the
# gate runs. A missing or empty block is ENV rc=2, never a pass.
#
#   check_ci_gate_mutants_cuda_rule.sh              the case table over ci.yml's block
#   check_ci_gate_mutants_cuda_rule.sh --self-test  planted weaker rules: each must go RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WF="${GATE_RULE_WORKFLOW:-$ROOT/.github/workflows/ci.yml}"

# extract <workflow> -> the rule block, dedented; empty when the markers are absent
extract() {
    awk '/GATE-MUTANTS-CUDA-RULE-BEGIN/{f=1; next} /GATE-MUTANTS-CUDA-RULE-END/{f=0} f' "$1" | sed -E 's/^ {10}//'
}

# x86 <not_measured> <sha> -> the x86-main results JSON ("-" = the output is absent)
x86() {
    if [ "$1" = - ]; then printf '{"mutants":{"result":"success","outputs":{}}}'
    else printf '{"mutants":{"result":"success","outputs":{"not_measured":"%s","not_measured_sha":"%s"}}}' "$1" "$2"; fi
}
# ym <result> <sha> -> the yoga-mutants results JSON ("none" = the job was skipped: no outputs at all)
ym() {
    if [ "$1" = none ]; then printf ''
    else printf '{"mutants-cuda":{"result":"%s","outputs":{"measured":"7","measured_sha":"%s"}}}' "$1" "$2"; fi
}

verdict() { # verdict <block> <evt> <nm> <nm_sha> <ym_result> <ym_sha>; a sha of "x" is the empty string
    local a=$4 b=$6
    [ "$a" = x ] && a=""
    [ "$b" = x ] && b=""
    EVT=$2 X86=$(x86 "$3" "$a") YM=$(ym "$5" "$b") bash -c "$1" > /dev/null 2>&1
}

table() { # table <workflow> -> 0 iff every row holds
    local blk bad=0 n=0 evt nm nms yr ms want got
    blk=$(extract "$1")
    [ -n "$blk" ] || { printf 'ENV   no GATE-MUTANTS-CUDA-RULE block in %s -- cannot judge, not a pass\n' "$1" >&2; return 2; }
    command -v jq > /dev/null || { printf 'ENV   no jq -- cannot judge, not a pass\n' >&2; return 2; }
    while read -r evt nm nms yr ms want; do
        [ -n "$evt" ] || continue
        n=$((n + 1)); got=0; verdict "$blk" "$evt" "$nm" "$nms" "$yr" "$ms" || got=1
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-13s nm=%-2s %-5s shard=%-9s %-5s -> %s\n' "$n" "$evt" "$nm" "$nms" "$yr" "$ms" "$([ "$want" = 0 ] && echo pass || echo RED)"
        else printf 'FAIL  row %-2s %-13s nm=%-2s %-5s shard=%-9s %-5s wanted %s, got %s\n' "$n" "$evt" "$nm" "$nms" "$yr" "$ms" "$want" "$got" >&2; bad=1; fi
    done <<'ROWS'
pull_request  0  e3b0  none       x     0
pull_request  -  x     none       x     0
pull_request  7  aa11  success    aa11  0
pull_request  7  aa11  success    bb22  1
pull_request  7  aa11  success    x     1
pull_request  7  aa11  failure    aa11  1
pull_request  7  aa11  cancelled  aa11  1
pull_request  7  aa11  skipped    aa11  1
pull_request  7  aa11  none       x     1
pull_request  7  x     success    x     1
pull_request  junk aa11 none      x     1
merge_group   -  x     none       x     0
push          -  x     none       x     0
ROWS
    # END TO END: the x86 JSON above is hand-written and carries `outputs`. The one the gate really receives
    # is what fat_driver.emit_results_output writes; when that dropped `outputs` this rule read "" on every
    # run and could never be RED (#4621). So feed the rule the driver's own emission.
    local emitted rc=0
    emitted=$(python3 "$ROOT/scripts/ci/emit_results_sample.py" 2>/dev/null) || rc=$?
    [ "$rc" = 0 ] && [ -n "$emitted" ] || { printf 'ENV   could not run fat_driver.emit_results_output -- cannot judge, not a pass\n' >&2; return 2; }
    n=$((n + 1)); got=0
    EVT=pull_request X86=$emitted YM= bash -c "$blk" > /dev/null 2>&1 || got=1
    if [ "$got" = 1 ]; then printf 'ok    row %-2s driver-emitted results, 3 deferred, no shard -> RED\n' "$n"
    else printf 'FAIL  row %-2s driver-emitted results carry no outputs: the rule read them as empty and passed\n' "$n" >&2; bad=1; fi
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== gate mutants-cuda rule: every weaker rule must turn the table RED ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/gate-cuda-rule.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    [ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
    real=$(awk '/GATE-MUTANTS-CUDA-RULE-BEGIN/,/GATE-MUTANTS-CUDA-RULE-END/' "$WF")
    [ -n "$real" ] || { printf 'ENV   no rule block in %s\n' "$WF" >&2; exit 2; }
    bad=0 i=0
    if table "$WF" > "$d/out" 2>&1; then printf 'ok    the real block passes the table\n'
    else printf 'FAIL  the real block fails its own table\n'; bad=1; fi
    # name~from~to: each plants one weakening of the REAL block
    while IFS='~' read -r name from to; do
        [ -n "$name" ] || continue
        i=$((i + 1))
        printf '%s\n' "$real" > "$d/m$i.yml"
        python3 - "$d/m$i.yml" "$from" "$to" <<'PY' || { printf 'FAIL  mutation %s: its anchor is gone from the block\n' "$name"; bad=1; continue; }
import sys
p, a, b = sys.argv[1:]
s = open(p).read()
if s.count(a) != 1:
    sys.exit(1)
open(p, "w").write(s.replace(a, b))
PY
        if table "$d/m$i.yml" > "$d/out" 2>&1; then printf 'FAIL  weaker rule %s passed the table\n' "$name"; bad=1
        else printf 'ok    weaker rule %-22s is RED (%s row(s))\n' "$name" "$(grep -c '^FAIL' "$d/out")"; fi
    done <<'MUT'
shard-result-ignored~[ "$YR" = success ] ||~true ||
sha-not-compared~[ -n "$NMS" ] && [ "$MS" = "$NMS" ] ||~true ||
empty-sha-matches~[ -n "$NMS" ] && [ "$MS" = "$NMS" ] ||~[ "$MS" = "$NMS" ] ||
rule-off~[ "$EVT" = pull_request ] && [ "${NM:-0}" != 0 ]~false
numeric-only~[ "${NM:-0}" != 0 ]~[ "${NM:-0}" -gt 0 ] 2> /dev/null
MUT
    printf 'jobs: {}\n' > "$d/none.yml"
    rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && printf 'ok    a workflow with no rule block is ENV rc=2, never a pass\n' || { printf 'FAIL  no rule block gave rc=%s\n' "$rc"; bad=1; }
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== gate needs mutants-cuda to measure the deferred NOT_MEASURED set (check_ci_gate_mutants_cuda_rule.sh) ==="
[ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
table "$WF"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
