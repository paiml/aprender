#!/usr/bin/env bash
# conleche.sh -- PVL-F7 (#3142): an INDEPENDENT re-check of ProvableContracts by con-leche. Advisory.
#
# WHY THIS EXISTS
# ---------------
# Every other verdict on our Lean tree comes from Lean's own C++ kernel (lake build, leanchecker). con-leche
# (leanprover/con-leche) is a second checker, written in Lean and proven in Lean not to accept a proof of False,
# with its own term representation. It rejects every axiom beyond propext / Classical.choice / Quot.sound, so it
# also re-checks the escape allowlist being empty (#4347) with a tool that shares no code with pv.
#
#   the tree's lean-toolchain -> the lean4export commit pinned for it (EXPORT_PINS; oleans are toolchain-exact)
#   build.sh                  fresh oleans first. A stale .lake exported the pre-#4347 axiom and con-leche
#                             declined it (measured 2026-09-25: olean 03:56, source 08:43)
#   lean4export ProvableContracts -> NDJSON (format 3.1.0) -> con-leche --verified
#   controls, every run: a one-file module using an axiom must come back RED, a trivial theorem must be accepted.
#   An oracle that cannot say no proves nothing.
#
# Verdicts (exit code):
#   0  ACCEPT      con-leche accepted N declarations
#   1  RED         a declaration was rejected, or it names a non-standard axiom (an escape)
#   2  NOT A VERDICT  con-leche declined an unsupported feature, ran out of memory, a pin could not be
#                  resolved, a control failed, or the build/export failed. Never a pass.
#
#   ./conleche.sh                     fetch pins, build, export, check  (lambda 2026-09-25: con-leche build
#                                     395 s once; export 232 s / 5.9 GB; check 154 s / 4.1 GB; 402163 decls)
#   ./conleche.sh --judge <rc> <log>  judge a saved con-leche run
#   ./conleche.sh --self-test         the verdict case table
#
# Heavy steps run under `systemd-run --user --scope --slice=agent.slice -p MemoryMax -p CPUQuota` when the user
# manager is reachable (the host must stay usable for other work, #4348), and always with LEAN_NUM_THREADS capped.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROG=conleche

# lean-toolchain line -> lean4export commit whose lean-toolchain is exactly that line.
# Find a new one with: gh api 'repos/leanprover/lean4export/commits?path=lean-toolchain' (the "bump toolchain" commit).
EXPORT_PINS='leanprover/lean4:v4.29.0-rc4 86c42ee28d1c0cc7ced06fe286d23f0a0fd83d50'
EXPORT_REPO=https://github.com/leanprover/lean4export
CONLECHE_REPO=https://github.com/leanprover/con-leche
CONLECHE_SHA=ae0c0c4e4ce6a0081648aff03fe9c39d002c4526
CACHE="${PVL_CONLECHE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/pvl-conleche}"
MEM="${PVL_CONLECHE_MEMORY_MAX:-16G}"
CPU="${PVL_CONLECHE_CPU_QUOTA:-800%}"
export LEAN_NUM_THREADS="${LEAN_NUM_THREADS:-8}"

say() { printf '%s: %s\n' "$PROG" "$*"; }
notverdict() { say "NOT A VERDICT -- $*"; exit 2; }

# judge <con-leche rc> <log> -> prints the verdict, returns 0 ACCEPT / 1 RED / 2 not a verdict
judge() {
    local rc=$1 log=$2 n ax line
    line=$(grep -m1 -E 'accepted [0-9]+ declarations|non-standard axiom|not implemented yet|rejected|INTERNAL PANIC|error' "$log" || true)
    case "$rc" in
        0)  n=$(grep -oE 'accepted [0-9]+ declarations' "$log" | grep -oE '[0-9]+' | tail -1)
            if [ -n "$n" ]; then say "ACCEPT: con-leche accepted $n declarations (--verified)"; return 0; fi
            say "NOT A VERDICT -- exit 0 but no 'accepted N declarations' line"; return 2 ;;
        1)  if grep -q 'out of memory' "$log"; then say "NOT A VERDICT -- out of memory under MemoryMax=$MEM"; return 2; fi
            if grep -q 'INTERNAL PANIC' "$log"; then say "NOT A VERDICT -- con-leche panicked: $line"; return 2; fi
            say "RED: a declaration was rejected: $line"; return 1 ;;
        2)  ax=$(grep -oE 'non-standard axiom \([^)]*\)' "$log" | head -1)
            if [ -n "$ax" ]; then say "RED: $ax -- an axiom beyond propext/Classical.choice/Quot.sound"; return 1; fi
            say "NOT A VERDICT -- con-leche declined an unsupported feature: $line"; return 2 ;;
        *)  say "NOT A VERDICT -- con-leche exit $rc (usage, malformed input or internal failure): $line"; return 2 ;;
    esac
}

self_test() {
    local td n=0 red=0 got
    td=$(mktemp -d "${TMPDIR:-/tmp}/conleche.XXXXXX") || return 2
    [ -n "$td" ] && [ -d "$td" ] || return 2
    row() { # row <want> <con-leche rc> <log text>
        n=$((n + 1)); printf '%s\n' "$3" > "$td/log"; got=0
        judge "$2" "$td/log" > "$td/out" || got=$?
        if [ "$got" = "$1" ]; then printf 'ok    row %-2s %s  rc=%s  %s\n' "$n" "$1" "$2" "$(cat "$td/out")"
        else printf 'FAIL  row %-2s got %s wanted %s  rc=%s  %s\n' "$n" "$got" "$1" "$2" "$3"; red=1; fi
    }
    row 0 0 'con-leche: accepted 402163 declarations (--verified)'
    row 2 0 'con-leche: 49 inductive blocks modelled in-process'
    row 1 2 'con-leche: not implemented yet: non-standard axiom (ProvableContracts.DPO.dpo_gradient_formula) [at axiom ProvableContracts.DPO.dpo_gradient_formula, fold position 235754] (--verified) t=92.6s'
    row 1 2 'con-leche: not implemented yet: non-standard axiom (sorryAx) [at axiom sorryAx]'
    row 2 2 'con-leche: not implemented yet: nested inductive with indices'
    row 1 1 'con-leche: declaration Foo.bar rejected: type mismatch'
    row 2 1 'INTERNAL PANIC: out of memory'
    row 2 1 'INTERNAL PANIC: index out of bounds'
    row 2 3 'usage: con-leche [--verified|--trusted] FILE.ndjson'
    row 2 137 ''
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ -n "$td" ] && [ "$td" != / ] && rm -rf -- "$td"
    [ "$red" = 0 ]
}

# capped <cmd...>: under the agent.slice caps when a user manager answers, else plain (LEAN_NUM_THREADS still caps)
capped() {
    if systemd-run --user --scope --quiet --slice=agent.slice true >/dev/null 2>&1; then
        systemd-run --user --scope --quiet --slice=agent.slice -p MemoryMax="$MEM" -p CPUQuota="$CPU" "$@"
    else
        "$@"
    fi
}

# pin <repo> <sha> <dir>: a checkout of exactly <sha>, or not a verdict
pin() {
    local repo=$1 sha=$2 dir=$3
    [ -d "$dir/.git" ] || git clone -q "$repo" "$dir" || notverdict "git clone $repo failed"
    git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin || notverdict "git fetch $repo failed"
    git -C "$dir" checkout -q --force --detach "$sha" || notverdict "$repo has no commit $sha"
    # A reused cache clone must BE the pin: no edited tracked file, no stray source. .lake is lake's build cache,
    # keyed on source hashes, and keeping it saves the 395 s con-leche build.
    git -C "$dir" reset -q --hard "$sha" && git -C "$dir" clean -q -fdx -e .lake || notverdict "cannot reset $dir to $sha"
    [ -z "$(git -C "$dir" status --porcelain --ignored=no)" ] || notverdict "$dir differs from $sha after reset"
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || notverdict "$dir is not at $sha"
}

# check <ndjson> <log>: run con-leche, return its exit code
check() {
    local rc=0
    capped "$CACHE/con-leche/.lake/build/bin/con-leche" --verified --jobs="$LEAN_NUM_THREADS" "$1" > "$2" 2>&1 || rc=$?
    return "$rc"
}

# controls <sysroot>: the oracle must say no to an axiom and yes to a trivial theorem
controls() {
    local sr=$1 d="$CACHE/controls" rc
    [ -n "$CACHE" ] && [ "$CACHE" != / ] || notverdict "cache dir is empty or /"
    rm -rf -- "${CACHE:?}/controls"; mkdir -p "$d" || notverdict "cannot create $d"
    printf 'axiom escape_ax : False\ntheorem uses_escape : False := escape_ax\n' > "$d/Neg.lean"
    printf 'theorem ok_thm : True := trivial\n' > "$d/Pos.lean"
    for m in Neg Pos; do
        (cd "$d" && elan run "$TC" lean -o "$m.olean" "$m.lean") || notverdict "control $m did not compile"
        LEAN_PATH="$d:$sr/lib/lean" "$EXPORTER" "$m" > "$d/$m.ndjson" || notverdict "control $m did not export"
    done
    # The negative control must fail for the PLANTED reason: exit 2 naming escape_ax. A crash (exit 1) or any other
    # RED would also judge 1, and then the control would pass without con-leche ever having seen the axiom.
    rc=0; check "$d/Neg.ndjson" "$d/Neg.log" || rc=$?
    [ "$rc" = 2 ] && grep -q 'non-standard axiom (escape_ax)' "$d/Neg.log" \
        || notverdict "negative control: con-leche exit $rc without naming escape_ax: $(tail -1 "$d/Neg.log")"
    judge "$rc" "$d/Neg.log" > /dev/null; rc=$?
    [ "$rc" = 1 ] || notverdict "negative control: judge did not call the escape_ax module RED (judge $rc)"
    rc=0; check "$d/Pos.ndjson" "$d/Pos.log" || rc=$?
    judge "$rc" "$d/Pos.log" > /dev/null; rc=$?
    [ "$rc" = 0 ] || notverdict "positive control: a trivial theorem was not accepted (judge $rc): $(tail -1 "$d/Pos.log")"
    say "controls ok: axiom module RED, trivial theorem accepted"
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --judge) [ $# = 3 ] || { echo "usage: conleche.sh --judge <rc> <log>" >&2; exit 2; }; judge "$2" "$3"; exit $? ;;
    "") ;;
    *) echo "usage: conleche.sh [--self-test | --judge <rc> <log>]   (no argument: build, export, check)" >&2; exit 2 ;;
esac

TC=$(tr -d '[:space:]' < "$HERE/lean-toolchain")
EXPORT_SHA=$(printf '%s\n' "$EXPORT_PINS" | awk -v tc="$TC" '$1 == tc { print $2 }')
[ -n "$EXPORT_SHA" ] || notverdict "no lean4export commit pinned for $TC: add it to EXPORT_PINS"
mkdir -p "$CACHE" || notverdict "cannot create $CACHE"
pin "$EXPORT_REPO" "$EXPORT_SHA" "$CACHE/lean4export"
[ "$(tr -d '[:space:]' < "$CACHE/lean4export/lean-toolchain")" = "$TC" ] || notverdict "lean4export@$EXPORT_SHA is not on $TC"
pin "$CONLECHE_REPO" "$CONLECHE_SHA" "$CACHE/con-leche"
(cd "$CACHE/lean4export" && capped lake build > "$CACHE/build-lean4export.log" 2>&1) || notverdict "lean4export build failed: $CACHE/build-lean4export.log"
(cd "$CACHE/con-leche" && capped lake build > "$CACHE/build-con-leche.log" 2>&1) || notverdict "con-leche build failed: $CACHE/build-con-leche.log"
EXPORTER="$CACHE/lean4export/.lake/build/bin/lean4export"
SYSROOT=$(elan run "$TC" lean --print-prefix) || notverdict "no $TC sysroot"
controls "$SYSROOT"

brc=0; capped "$HERE/build.sh" || brc=$?
case "$brc" in
    0) ;;
    2) notverdict "build.sh declined (exit 2, e.g. a Mathlib cache miss): it gave no verdict on the tree, so nothing is exported" ;;
    *) notverdict "build.sh exited $brc: the tree does not build, and an export of its old oleans would check a stale tree" ;;
esac
erc=0; (cd "$HERE" && capped lake env "$EXPORTER" ProvableContracts > "$CACHE/ProvableContracts.ndjson" 2> "$CACHE/export.err") || erc=$?
[ "$erc" = 0 ] || notverdict "lean4export exited $erc: $(tail -1 "$CACHE/export.err")"
say "checking $(git -C "$HERE" rev-parse --short=12 HEAD 2>/dev/null || echo '<no git>') on $TC with lean4export@${EXPORT_SHA:0:12} con-leche@${CONLECHE_SHA:0:12}"
crc=0; check "$CACHE/ProvableContracts.ndjson" "$CACHE/check.log" || crc=$?
rm -f -- "$CACHE/ProvableContracts.ndjson"
judge "$crc" "$CACHE/check.log"
