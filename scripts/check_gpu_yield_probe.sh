#!/usr/bin/env bash
# check_gpu_yield_probe.sh -- the yield-to-training probe must not read "[N/A]" as a busy GPU (#4879).
#
# WHY THIS EXISTS
#   gpu-quick (ci/sections.yml) and the cuda-nightly `decide` step ask nvidia-smi two
#   questions before they spend a GPU leg -- how much memory is in use, and which
#   compute processes are on the card -- and YIELD (skip the leg, job GREEN) when either
#   answer says busy. The process question was counted with `grep -c .`: one output line
#   was one busy process. A Jetson Orin answers BOTH questions with the literal text
#   `[N/A]` (the nvgpu driver has no per-process accounting), so the probe counted ONE
#   busy process on an idle card, skipped every GPU leg and reported success. The lane
#   was green because it did not run, which is the one thing a GPU lane must never mean.
#
# THE RULE
#   A reading counts only when it is a bare integer. Anything else is "cannot measure":
#   neither busy nor idle, never shown as 0, never a reason to yield. When NOTHING could
#   be measured the step says so with a ::notice::. Numeric answers (gx10's GB10 prints a
#   number for the process question and [N/A] for memory; a discrete card prints both)
#   decide exactly as before.
#
# WHAT IT JUDGES
#   1. each of ci/sections.yml and .github/workflows/cuda-nightly.yml carries exactly one
#      `# BEGIN yield-probe` / `# END yield-probe` block;
#   2. the two blocks are byte-identical once their indentation is removed (two
#      hand-maintained copies of one rule is how the defect lived in both);
#   3. no other line of .github/workflows/*.yml or ci/*.yml queries compute-apps outside
#      such a block (a third, unpinned copy of the old probe);
#   4. each block, EXECUTED under `bash -e` against a fake nvidia-smi, gives the table's
#      answer for every row (used / procs / busy / exit code / notice);
#   5. the PRE-FIX probe, replayed on the same rows, agrees on every `same` row (numeric
#      answers behave exactly as before) and DISAGREES on every `fixed` row (so the row
#      can tell the fix from the bug; a row the old probe also passes proves nothing).
#
# USAGE
#   bash scripts/check_gpu_yield_probe.sh              # judge this tree
#   bash scripts/check_gpu_yield_probe.sh --root DIR   # judge another tree (fixtures)
#   bash scripts/check_gpu_yield_probe.sh --self-test  # fixture trees: one GOOD, the rest must be RED
# EXIT: 0 every rule holds; 1 a violation; 2 cannot judge (file unreadable, no tool).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
BEGIN_RE='^[[:space:]]*# BEGIN yield-probe[[:space:]]*$'
END_RE='^[[:space:]]*# END yield-probe[[:space:]]*$'
FILES=(ci/sections.yml .github/workflows/cuda-nightly.yml)
WORK=""

cleanup() { if [ -n "$WORK" ]; then rm -rf "${WORK:?}"; fi; }
trap cleanup EXIT

usage() {
    cat <<'EOF'
usage: check_gpu_yield_probe.sh [--root DIR | --self-test | --help]
  (no argument)  judge this tree: both yield probes pinned, equal, executed against a fake nvidia-smi
  --root DIR     judge the tree at DIR instead
  --self-test    judge fixture trees; the good one must pass and every broken one must be red
EOF
}

setup_work() {
    if [ -z "$WORK" ]; then
        WORK=$(mktemp -d) || return 2
        case $WORK in /?*) ;; *) echo "ENV   mktemp gave an unusable directory: '$WORK'" >&2; return 2 ;; esac
        mkdir -p "$WORK/bin" "$WORK/j" "$WORK/fx"
        write_fake_smi "$WORK/bin"
    fi
}

# The fake answers from FAKE_SMI_* and exits as told. \n in a cell becomes a second row.
write_fake_smi() { # write_fake_smi DIR
    cat > "$1/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    --query-gpu=*)
        if [ -n "${FAKE_SMI_MEM:-}" ]; then printf '%b\n' "$FAKE_SMI_MEM"; fi
        exit "${FAKE_SMI_MEM_RC:-0}" ;;
    --query-compute-apps=*)
        if [ -n "${FAKE_SMI_APPS:-}" ]; then printf '%b\n' "$FAKE_SMI_APPS"; fi
        exit "${FAKE_SMI_APPS_RC:-0}" ;;
esac
exit 64
EOF
    chmod +x "$1/nvidia-smi"
}

# extract FILE -> the block between the markers, de-indented, on stdout.
# rc 0 one well-formed pair; 1 markers missing, repeated or unclosed; 2 unreadable.
extract() {
    [ -r "$1" ] || return 2
    awk -v b="$BEGIN_RE" -v e="$END_RE" '
        $0 ~ b { nb++; if (nb == 1) { match($0, /[^ ]/); ind = RSTART - 1; inblk = 1 } next }
        $0 ~ e { ne++; inblk = 0; next }
        inblk  { print substr($0, ind + 1) }
        END    { if (nb != 1 || ne != 1 || inblk) exit 3 }
    ' "$1" || return 1
}

# stray_probes FILE -> "FILE:LINE" for every compute-apps query that sits outside a block.
stray_probes() {
    awk -v f="$1" -v b="$BEGIN_RE" -v e="$END_RE" '
        $0 ~ b { inblk = 1; next }
        $0 ~ e { inblk = 0; next }
        /query-compute-apps/ && !inblk { print f ":" NR }
    ' "$1"
}

# The pre-fix probe, verbatim, as a replay fixture. This is the DEFECT, kept so the table
# can be shown to tell it from the fix; it is not a second copy of the rule.
old_probe() {
    cat <<'EOF'
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -cd '0-9')
used=${used:-0}
procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
procs=${procs:-0}
busy=0
if [ "${used:-0}" -gt 2000 ] || [ "${procs:-0}" -gt 0 ]; then busy=1; fi
EOF
}

# Columns: name|kind|mem answer|pid answer|mem exit|pid exit|want exit|want used|want procs|want busy|want notices
# kind  same  = the pre-fix probe must reach the same busy verdict (a numeric answer, unchanged)
#       fixed = the pre-fix probe must reach a DIFFERENT busy verdict (the row that tells fix from bug)
# want exit  0, or `die` (the step must fail, as it does today when nvidia-smi itself fails)
# `\n` inside an answer is a second output row.
table() {
    cat <<'EOF'
gx10-idle|same|[N/A]||0|0|0|n/a|0|0|0
gx10-busy|same|[N/A]|4242\n4343|0|0|0|n/a|2|1|0
rtx-idle|same|512||0|0|0|512|0|0|0
rtx-busy-memory|same|8192||0|0|0|8192|0|1|0
rtx-memory-at-threshold|same|2000||0|0|0|2000|0|0|0
rtx-memory-over-threshold|same|2001||0|0|0|2001|0|1|0
rtx-busy-process|same|512|777|0|0|0|512|1|1|0
two-gpus-first-row-counts|same|100\n9000||0|0|0|100|0|0|0
empty-answers|same|||0|0|0|n/a|0|0|0
orin-na-both|fixed|[N/A]|[N/A]|0|0|0|n/a|0|0|1
orin-padded|fixed| [N/A] | [N/A]|0|0|0|n/a|0|0|1
orin-not-supported|fixed|[Not Supported]|[Not Supported]|0|0|0|n/a|0|0|1
orin-na-per-process-rows|fixed|[N/A]|[N/A]\n[N/A]|0|0|0|n/a|0|0|1
orin-numeric-memory-na-pid|fixed|1500|[N/A]|0|0|0|1500|0|0|0
numeric-memory-busy-na-pid|same|3000|[N/A]|0|0|0|3000|0|1|0
na-row-beside-a-real-pid|same|[N/A]|[N/A]\n999|0|0|0|n/a|1|1|0
nvidia-smi-memory-query-fails|same|||9|0|die|-|-|-|-
nvidia-smi-pid-query-fails-memory-numeric|same|512||0|9|0|512|0|0|0
nvidia-smi-pid-query-fails-memory-na|same|[N/A]||0|9|0|n/a|0|0|1
EOF
}

# make_harness BLOCKFILE OUT -> a step body the way the runner would run it
make_harness() {
    {
        printf 'set -uo pipefail\nwho=harness\n'
        cat "$1"
        cat <<'EOF'
printf 'RESULT used=%s procs=%s busy=%s\n' "$used" "$procs" "$busy"
EOF
    } > "$2"
}

# run_step SCRIPT MEM APPS MEM_RC APPS_RC -> "rc=N" then the step's output, under bash -e
# (fat_driver runs a step as `bash -e FILE`; the runner's default shell is `bash -e {0}`).
run_step() {
    local out rc=0
    out=$(PATH="$WORK/bin:$PATH" FAKE_SMI_MEM="$2" FAKE_SMI_APPS="$3" FAKE_SMI_MEM_RC="$4" \
        FAKE_SMI_APPS_RC="$5" bash -e "$1" 2>&1 < /dev/null) || rc=$?
    printf 'rc=%s\n%s\n' "$rc" "$out"
}

field_rc() { sed -n '1s/^rc=//p'; }
field_result() { sed -n 's/^RESULT //p'; }
field_busy() { sed -n 's/^RESULT .* busy=\([0-9]*\)$/\1/p'; }
count_notices() { grep -c '^::notice::harness: cannot measure' || true; }

# judge ROOT -> prints findings; rc 0 holds, 1 violation, 2 cannot judge
judge() {
    local root=$1 jd="$WORK/j" f i=0 bad=0 rows=0 fixed=0 same=0 g
    rm -rf "${jd:?}"; mkdir -p "$jd"
    local -a names=()
    for f in "${FILES[@]}"; do
        extract "$root/$f" > "$jd/block.$i"
        case $? in
            0) ;;
            2) echo "ENV   $f is not readable under $root - cannot judge, and that is not a pass"; return 2 ;;
            *) echo "FAIL  $f: needs exactly one '# BEGIN yield-probe' ... '# END yield-probe' pair (markers missing, repeated or unclosed)"; bad=1 ;;
        esac
        names[i]=$f
        i=$((i + 1))
    done
    [ "$bad" -eq 0 ] || return 1

    if ! cmp -s "$jd/block.0" "$jd/block.1"; then
        echo "FAIL  the two yield-probe blocks differ (${names[0]} vs ${names[1]}):"
        diff "$jd/block.0" "$jd/block.1" | head -n 12 | sed 's/^/        /'
        bad=1
    fi

    for g in "$root"/.github/workflows/*.yml "$root"/ci/*.yml; do
        [ -f "$g" ] || continue
        stray_probes "$g" | sed "s|^$root/||; s|^|FAIL  stray compute-apps probe outside a yield-probe block: |"
        [ -z "$(stray_probes "$g")" ] || bad=1
    done

    old_probe > "$jd/old.block"
    make_harness "$jd/old.block" "$jd/old.sh"
    local name kind mem apps mrc arc wrc wused wprocs wbusy wnote
    local out oldout rc res notes ob
    while IFS='|' read -r name kind mem apps mrc arc wrc wused wprocs wbusy wnote; do
        oldout=$(run_step "$jd/old.sh" "$mem" "$apps" "$mrc" "$arc")
        i=0
        for f in "${FILES[@]}"; do
            make_harness "$jd/block.$i" "$jd/new.$i.sh"
            out=$(run_step "$jd/new.$i.sh" "$mem" "$apps" "$mrc" "$arc")
            rc=$(printf '%s\n' "$out" | field_rc)
            res=$(printf '%s\n' "$out" | field_result)
            notes=$(printf '%s\n' "$out" | count_notices)
            rows=$((rows + 1))
            if [ "$wrc" = die ]; then
                if [ "$rc" = 0 ]; then
                    echo "FAIL  $f row '$name': nvidia-smi failed and the step went on (rc=0); a broken probe must stop the step as it does today"
                    bad=1
                fi
            elif [ "$rc" != "$wrc" ] || [ "$res" != "used=$wused procs=$wprocs busy=$wbusy" ] || [ "$notes" != "$wnote" ]; then
                echo "FAIL  $f row '$name': got rc=$rc '${res:-<no result>}' notices=$notes, want rc=$wrc 'used=$wused procs=$wprocs busy=$wbusy' notices=$wnote"
                bad=1
            fi
            i=$((i + 1))
        done
        if [ "$wrc" = die ]; then
            [ "$(printf '%s\n' "$oldout" | field_rc)" != 0 ] || { echo "FAIL  replay row '$name': the pre-fix probe survived a failing nvidia-smi, so 'same' is wrong"; bad=1; }
            same=$((same + 1))
            continue
        fi
        ob=$(printf '%s\n' "$oldout" | field_busy)
        case $kind in
            same)
                same=$((same + 1))
                [ "$ob" = "$wbusy" ] || { echo "FAIL  row '$name': the pre-fix probe says busy=${ob:-?} and the new one busy=$wbusy - a numeric answer changed its verdict"; bad=1; } ;;
            fixed)
                fixed=$((fixed + 1))
                [ "$ob" != "$wbusy" ] || { echo "FAIL  row '$name': the pre-fix probe reaches the same verdict (busy=$ob), so this row cannot tell the fix from the bug"; bad=1; } ;;
            *) echo "FAIL  row '$name': kind '$kind' is neither same nor fixed"; bad=1 ;;
        esac
    done < <(table)

    if [ "$rows" -eq 0 ]; then
        echo "ENV   executed 0 rows - a guard that ran nothing is not a pass"
        return 2
    fi
    if [ "$bad" -ne 0 ]; then return 1; fi
    echo "OK    yield probe: ${#FILES[@]} blocks identical, no stray copy, $rows step run(s) matched ($same same-rows, $fixed fixed-rows the pre-fix probe fails)"
    return 0
}

# splice FILE BODYFILE -> FILE with the yield-probe block body replaced, indentation kept
splice() {
    awk -v b="$BEGIN_RE" -v e="$END_RE" -v rf="$2" '
        $0 ~ b { print; match($0, /[^ ]/); ind = sprintf("%" (RSTART - 1) "s", "")
                 while ((getline line < rf) > 0) print ind line
                 close(rf); skip = 1; next }
        $0 ~ e { skip = 0; print; next }
        !skip  { print }
    ' "$1"
}

# mk_fixture NAME -> $WORK/fx/NAME with the two real files, then the mutation NAME names
mk_fixture() {
    local fx="$WORK/fx/$1" f body
    rm -rf "${fx:?}"; mkdir -p "$fx/ci" "$fx/.github/workflows"
    for f in "${FILES[@]}"; do cp "$REPO_ROOT/$f" "$fx/$f"; done
    extract "$REPO_ROOT/${FILES[0]}" > "$WORK/fx/real.block" || return 2
    body="$WORK/fx/$1.body"
    case $1 in
        good|empty-root) ;;
        old-logic)
            old_probe > "$body"
            for f in "${FILES[@]}"; do splice "$REPO_ROOT/$f" "$body" > "$fx/$f"; done ;;
        na-counted-as-a-process)
            sed 's/\*\[!0-9\]\*) apps_ok=0 ;; //' "$WORK/fx/real.block" > "$body"
            for f in "${FILES[@]}"; do splice "$REPO_ROOT/$f" "$body" > "$fx/$f"; done ;;
        notice-removed)
            sed 's/echo "::notice::\${who}: cannot measure[^"]*"/:/' "$WORK/fx/real.block" > "$body"
            for f in "${FILES[@]}"; do splice "$REPO_ROOT/$f" "$body" > "$fx/$f"; done ;;
        nightly-drifted)
            { cat "$WORK/fx/real.block"; echo '# a comment only one copy has'; } > "$body"
            splice "$REPO_ROOT/${FILES[1]}" "$body" > "$fx/${FILES[1]}" ;;
        markers-removed)
            sed -e '/# BEGIN yield-probe/d' -e '/# END yield-probe/d' "$REPO_ROOT/${FILES[1]}" > "$fx/${FILES[1]}" ;;
        markers-repeated)
            { cat "$REPO_ROOT/${FILES[1]}"; printf '# BEGIN yield-probe\n# END yield-probe\n'; } > "$fx/${FILES[1]}" ;;
        stray-copy)
            printf 'jobs:\n  x:\n    steps:\n      - run: |\n          procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)\n' \
                > "$fx/ci/extra.yml" ;;
    esac
    if [ "$1" = empty-root ]; then rm -rf "${fx:?}/ci" "${fx:?}/.github"; fi
    printf '%s' "$fx"
}

self_test() {
    local case_ want_rc token fx out rc fail=0 n=0
    while IFS='|' read -r case_ want_rc token; do
        fx=$(mk_fixture "$case_") || { echo "ENV   cannot build fixture '$case_'"; return 2; }
        out=$(judge "$fx") && rc=0 || rc=$?
        n=$((n + 1))
        if [ "$rc" != "$want_rc" ]; then
            echo "FAIL  self-test '$case_': rc=$rc, want $want_rc"; printf '%s\n' "$out" | head -n 4 | sed 's/^/        /'; fail=$((fail + 1))
        elif [ -n "$token" ] && ! printf '%s\n' "$out" | grep -qF -- "$token"; then
            echo "FAIL  self-test '$case_': red, but not for the reason it names (no '$token' in the output)"; printf '%s\n' "$out" | head -n 4 | sed 's/^/        /'; fail=$((fail + 1))
        else
            echo "ok    self-test '$case_' -> rc=$rc"
        fi
    done <<'EOF'
good|0|
old-logic|1|row 'orin-na-both': got
na-counted-as-a-process|1|orin-na-both
notice-removed|1|notices=0, want rc=0 'used=n/a
nightly-drifted|1|yield-probe blocks differ
markers-removed|1|needs exactly one
markers-repeated|1|needs exactly one
stray-copy|1|stray compute-apps probe
empty-root|2|is not readable
EOF
    echo "self-test: $n case(s), $fail failed"
    [ "$fail" -eq 0 ]
}

main() {
    local root=$REPO_ROOT
    case "${1:-}" in
        -h|--help) usage; return 0 ;;
        --self-test) setup_work || return 2; self_test; return $? ;;
        --root) [ -n "${2:-}" ] || { usage >&2; return 2; }; root=$2 ;;
        '') ;;
        *) usage >&2; return 2 ;;
    esac
    setup_work || return 2
    judge "$root"
}

main "$@"
