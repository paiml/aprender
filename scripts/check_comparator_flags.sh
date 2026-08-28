#!/usr/bin/env bash
#
# check_comparator_flags.sh -- the comparator's DECLARED knobs and its ACTUAL
# invocations may not disagree (aprender#2737, epic APR-PERF-GATE-001 #2706).
#
# WHY THIS EXISTS
# ---------------
# scripts/llama_pin.toml declared `batch_size = 1`. Three places acted on that,
# and they were three unjoined copies:
#
#   llama_pin.toml:92          batch_size = 1                     declaration
#   llama_pin.toml:112         ... -b {batch_size} ...            substituted
#   llama_pin.toml:173         ... -c 4096 -t 8 -b 1 ...          RETYPED
#   parity_host_receipt.sh:108 ... -c 4096 -t 8 -b 1 ...          RETYPED AGAIN
#
# Two of the four retyped values that the other two declared. Nothing compared
# them, so `context_length` and `threads` were declared keys that no execution
# path read, and `-b 1` was an execution that no declaration governed. Either
# half can move without the other and no gate would have said a word.
#
# What `-b 1` did, measured on gx10 against the pinned build, medians of 2:
# it switched llama.cpp's batching off, moving the c=16 aggregate ratio from
# 2.03x to 4.85x -- a 2.39x overstatement in apr's favour. The c=1 arm was
# unchanged (0.3%), which is what proves it was the batching path and not a
# general slowdown. A protocol that flatters itself by crippling the baseline
# is the receipt-rule defect in its purest form.
#
# WHAT THIS GUARD ASSERTS
# -----------------------
#   1. llama_comparator_server_flags -- the single producer -- turns a
#      declaration into the right flags, and REFUSES a malformed one rather
#      than emitting a partial list.
#   2. The two invocations of record in llama_pin.toml agree, flag by flag,
#      with what that producer emits from this repo's own declaration.
#   3. scripts/parity_host_receipt.sh does not carry a hardcoded knob literal.
#      It is the only thing that actually runs llama-server, so a literal there
#      outranks anything the declaration says.
#
# The case table runs on every invocation, in front of the verdict, because a
# guard that has not shown it can fail has not shown anything.
#
#   bash scripts/check_comparator_flags.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0
printf -- '--- comparator flags: declaration vs invocation ----------------------\n'

# shellcheck source=scripts/llama_bin.sh
. scripts/llama_bin.sh >/dev/null 2>&1 || true
if ! command -v llama_comparator_server_flags >/dev/null 2>&1; then
    printf 'FAIL  scripts/llama_bin.sh defines no llama_comparator_server_flags\n'
    exit 1
fi

# Value of <flag> in a whitespace-separated invocation, or the word ABSENT.
# Tokenised and walked rather than pattern-matched, so `-t` never matches
# inside `--top-p` and a flag at end-of-string cannot borrow the next line.
flag_value() {
    local want="$1" str="$2" tok prev="" out="ABSENT"
    # A HERESTRING, NEVER A PIPE. `printf ... | while read` both subshells the
    # loop (so `out` is lost) and, under `set -o pipefail`, hands the pipeline
    # printf's SIGPIPE 141 when the reader stops early -- a false failure that
    # depends on input size and so is green locally and red in CI.
    while read -r tok; do
        if [ "$prev" = "$want" ]; then out="$tok"; break; fi
        if [ "$tok" = "$want" ]; then out="PRESENT"; fi
        prev="$tok"
    done <<< "$(tr ' ' '\n' <<< "$str" | grep -v '^$')"
    printf '%s' "$out"
}

# ---------------------------------------------------------------------------
printf '\nthe producer, against fixture declarations\n'
TD=$(mktemp -d) || exit 2
trap 'rm -rf "${TD:?}"' EXIT

mkpin() { # mkpin <file> <batch_size> <comparator_parallel> <ctx> <threads>
    printf 'batch_size = %s\ncomparator_parallel = %s\ncontext_length = %s\nthreads = %s\n' \
        "$2" "$3" "$4" "$5" > "$1"
}

row() { # row <name> <b> <np> <ctx> <thr> <expected-rc> <expected-flags>
    local name="$1" want_rc="$6" want="$7" got got_rc
    mkpin "$TD/pin.toml" "$2" "$3" "$4" "$5"
    got=$(llama_comparator_server_flags 999 "$TD/pin.toml"); got_rc=$?
    if [ "$got_rc" != "$want_rc" ]; then
        printf 'FAIL  %-46s rc=%s, expected rc=%s\n' "$name" "$got_rc" "$want_rc"; rc=1; return
    fi
    if [ "$got" != "$want" ]; then
        printf 'FAIL  %-46s got [%s], expected [%s]\n' "$name" "$got" "$want"; rc=1; return
    fi
    printf 'ok    %-46s rc=%s [%s]\n' "$name" "$got_rc" "$got"
}

#   name                                           b          np       ctx    thr  rc  expected
row 'default/default emits NO -b and NO -np'       '"default"' '"default"' 4096 8   0 '-ngl 999 -c 4096 -t 8 --no-warmup'
row 'a numeric batch_size DOES emit -b'            1           '"default"' 4096 8   0 '-ngl 999 -c 4096 -t 8 -b 1 --no-warmup'
row 'a numeric comparator_parallel emits -np'      '"default"' 4           4096 8   0 '-ngl 999 -c 4096 -t 8 -np 4 --no-warmup'
row 'both numeric emits both'                      2048        4           4096 8   0 '-ngl 999 -c 4096 -t 8 -b 2048 -np 4 --no-warmup'
row 'ctx and threads come from the declaration'    '"default"' '"default"' 8192 32  0 '-ngl 999 -c 8192 -t 32 --no-warmup'
# The refusals. A partial flag list is worse than none: it runs, and it runs
# a comparator nobody declared.
row 'a missing context_length REFUSES'             '"default"' '"default"' ''   8   3 ''
row 'a missing threads REFUSES'                    '"default"' '"default"' 4096 ''  3 ''
row 'a non-numeric, non-default batch_size REFUSES' '"auto"'   '"default"' 4096 8   3 ''
row 'a non-numeric comparator_parallel REFUSES'    '"default"' '"auto"'    4096 8   3 ''

# An absent declaration file must refuse too, not default to something.
if out=$(llama_comparator_server_flags 999 "$TD/does-not-exist.toml") && [ -n "$out" ]; then
    printf 'FAIL  %-46s a missing declaration produced [%s]\n' 'a missing pin file REFUSES' "$out"; rc=1
else
    printf 'ok    %-46s refused\n' 'a missing pin file REFUSES'
fi

# ---------------------------------------------------------------------------
printf '\nthis repo: the declaration vs the invocations of record\n'

PIN=scripts/llama_pin.toml
[ -f "$PIN" ] || { printf 'FAIL  %s is missing\n' "$PIN"; exit 2; }

BUILT=$(llama_comparator_server_flags 999 "$PIN") || {
    printf 'FAIL  %s does not yield a comparator invocation; see the rows above\n' "$PIN"
    exit 1
}
printf 'ok    built from the declaration: [%s]\n' "$BUILT"

decl_of() { llama_pin_get_raw "$1" "$PIN"; }
cmd_of() {
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"\(.*\)\"[[:space:]]*\$/\1/p" "$PIN" | head -1
}

SERVE=$(cmd_of comparator_serve_command)
BENCH=$(cmd_of comparator_command)
[ -n "$SERVE" ] || { printf 'FAIL  comparator_serve_command is empty\n'; rc=1; }
[ -n "$BENCH" ] || { printf 'FAIL  comparator_command is empty\n'; rc=1; }

# Render {key} placeholders from the declaration, so a template that SAYS
# {threads} is compared on the value it resolves to, not on the word.
render() {
    local s="$1" k v
    for k in context_length threads batch_size comparator_parallel gpu_layers; do
        v=$(decl_of "$k")
        s=${s//\{$k\}/$v}
    done
    printf '%s' "$s"
}
SERVE_R=$(render "$SERVE")
# RENDER BOTH. Mutation M3 -- declaration and both invocations all saying 1 --
# went RED because only SERVE was rendered, so `-b {batch_size}` was compared as
# the literal string "{batch_size}". A guard that cannot go green on a correct
# tree is as broken as one that cannot go red on a wrong one.
BENCH_R=$(render "$BENCH")

# Flag by flag, both directions, naming BOTH sides on a mismatch.
for f in -c -t -b -np; do
    want=$(flag_value "$f" "$BUILT")
    got=$(flag_value "$f" "$SERVE_R")
    if [ "$want" = "$got" ]; then
        printf 'ok    comparator_serve_command %-4s %s\n' "$f" "$want"
    else
        printf 'FAIL  comparator_serve_command %s is %s but the declaration builds %s\n' \
            "$f" "$got" "$want"
        printf '      declaration: batch_size=%s comparator_parallel=%s context_length=%s threads=%s\n' \
            "$(decl_of batch_size)" "$(decl_of comparator_parallel)" \
            "$(decl_of context_length)" "$(decl_of threads)"
        printf '      invocation : %s\n' "$SERVE"
        rc=1
    fi
done

# The llama-bench differential lane carries no -b either. It is not the Arm B
# comparator (spec 4.4.8, I-15) but it must not contradict the declaration.
b_bench=$(flag_value -b "$BENCH_R")
b_want=$(flag_value -b "$BUILT")
if [ "$b_bench" = "$b_want" ]; then
    printf 'ok    comparator_command      -b   %s\n' "$b_bench"
else
    printf 'FAIL  comparator_command -b is %s but the declaration builds %s\n' "$b_bench" "$b_want"
    printf '      invocation : %s\n' "$BENCH"
    rc=1
fi

# gpu_layers is declared "all"; llama.cpp spells that -ngl 999.
gl=$(decl_of gpu_layers)
ngl_serve=$(flag_value -ngl "$SERVE_R")
case "$gl" in
    all) want_ngl=999 ;;
    ''|*[!0-9]*) printf 'FAIL  gpu_layers=%s is neither "all" nor a number\n' "${gl:-<unset>}"; rc=1; want_ngl="$ngl_serve" ;;
    *) want_ngl="$gl" ;;
esac
if [ "$ngl_serve" = "$want_ngl" ]; then
    printf 'ok    comparator_serve_command -ngl %s (gpu_layers = %s)\n' "$ngl_serve" "$gl"
else
    printf 'FAIL  comparator_serve_command -ngl is %s but gpu_layers = %s wants %s\n' \
        "$ngl_serve" "$gl" "$want_ngl"
    rc=1
fi

# ---------------------------------------------------------------------------
printf '\nthis repo: the executor carries no hardcoded knob\n'

EXEC=scripts/parity_host_receipt.sh
[ -f "$EXEC" ] || { printf 'FAIL  %s is missing\n' "$EXEC"; exit 2; }

# The line that actually launches the comparator. Comments are stripped first:
# this file DOCUMENTS the old `-c 4096 -t 8 -b 1` in its own header, and a scan
# that reads its own explanation as a violation is a guard that can only fail.
launch=$(sed 's/^[[:space:]]*#.*$//' "$EXEC" | grep -n 'LLAMA_SERVER' | grep -v '^\s*$')
apr_launch=$(sed 's/^[[:space:]]*#.*$//' "$EXEC" | grep -n 'serve run' | grep -v '^\s*$')
if [ -z "$launch" ]; then
    printf 'FAIL  %s no longer launches $LLAMA_SERVER; this guard has lost its subject\n' "$EXEC"
    rc=1
else
    bad=""
    for f in -b -np -c -t -ngl; do
        v=$(flag_value "$f" "$launch")
        case "$v" in
            ABSENT|PRESENT) ;;
            \$*|\"\$*)      ;;   # a shell expansion is pin-derived; that is the point
            *[!0-9]*)       ;;   # not a bare number, so not a retyped literal
            # `${f}=${v}` NOT `$f=$v`: bashrs parses the latter as an
            # assignment to a variable named by $f and reports SC1066 (error).
            *)              bad="$bad ${f}=${v}" ;;
        esac
    done
    if [ -n "$bad" ]; then
        printf 'FAIL  %s hardcodes:%s\n' "$EXEC" "$bad"
        printf '      Every knob must come from llama_comparator_server_flags, or the\n'
        printf '      declaration and the run can disagree again (#2737).\n'
        rc=1
    else
        printf 'ok    %s builds its flags from the declaration\n' "$EXEC"
    fi
    # EXECUTION, NOT MENTION. This grepped the raw file, and the file names the
    # producer in its own explanatory comment -- so mutation M8, which deleted
    # the actual call, stayed GREEN. Same defect check_guards_are_wired.sh
    # documents one level up. Comments are stripped first.
    if grep -q 'llama_comparator_server_flags' <<< "$(sed 's/^[[:space:]]*#.*$//' "$EXEC")"; then
        printf 'ok    %s calls llama_comparator_server_flags\n' "$EXEC"
    else
        printf 'FAIL  %s does not call llama_comparator_server_flags\n' "$EXEC"
        rc=1
    fi
    # apr's own lane, same rule. A literal here and a declaration there is how
    # the two sides get run at different context lengths while both look pinned.
    actx=$(flag_value --context-length "$apr_launch")
    case "$actx" in
        ABSENT|PRESENT|\$*|\"\$*|*[!0-9]*)
            printf 'ok    apr lane --context-length is declaration-derived (%s)\n' "$actx" ;;
        *)
            printf 'FAIL  %s hardcodes --context-length=%s; context_length = %s is declared\n' \
                "$EXEC" "$actx" "$(decl_of context_length)"
            rc=1 ;;
    esac
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the declaration is the only source of the comparator knobs, and\n'
    printf '      the producer refuses a declaration it cannot read.\n'
else
    printf 'FAIL  see rows above (#2737).\n'
fi
exit "$rc"
