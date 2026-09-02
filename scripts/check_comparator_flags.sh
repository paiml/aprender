#!/usr/bin/env bash
#
# check_comparator_flags.sh -- the comparator's DECLARED knobs and its ACTUAL
# invocations may not disagree, and no accelerator flag is a BOOLEAN
# (aprender#2737; PP-15, PP-20, §5.3 of PP-LLAMA-001 v3.0).
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
# AND THE GUARD BLESSED IT. Row :98 of this file asserted that a declared
# `batch_size = 1` DOES emit `-b 1`, rc 0 -- so the exact configuration §5.3
# names as the cripple passed every check in the tree, and the refusal the spec
# requires had no producer. That row is now `batch_size = 1 REFUSES`.
#
# WHAT THIS GUARD ASSERTS
# -----------------------
#   1. llama_comparator_server_flags -- the single producer -- turns a
#      declaration plus a BAND into the right flags, and REFUSES a malformed
#      one, a cripple, and a per-slot context too small for the workload,
#      rather than emitting a partial list.
#   2. The invocation of record in llama_pin.toml agrees, flag by flag, with
#      what that producer emits from this repo's own declaration.
#   3. scripts/parity_host_receipt.sh does not carry a hardcoded knob literal.
#      It is the only thing that actually runs llama-server, so a literal there
#      outranks anything the declaration says.
#   4. PP-15: no BOOLEAN accelerator flag (`--gpu`, `--no-gpu`) survives in a
#      harness command or in the executor. `--gpu` is a promise the published
#      binary did not keep -- it was accepted, ignored, and the run silently
#      went to CPU at 15.7 tok/s. A quantity (`--gpu-layers all|N|0`) resolves
#      to a number the server prints about itself, so the lane can be read from
#      the resolution instead of from the request.
#
# The case table runs in front of the verdict on every invocation, because a
# guard that has not shown it can fail has not shown anything.
#
#   bash scripts/check_comparator_flags.sh              case table + this repo
#   bash scripts/check_comparator_flags.sh --selftest   case table only
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0

# shellcheck source=scripts/llama_bin.sh
. scripts/llama_bin.sh >/dev/null 2>&1 || true
if ! command -v llama_comparator_server_flags >/dev/null 2>&1; then
    printf 'FAIL  scripts/llama_bin.sh defines no llama_comparator_server_flags\n'
    exit 1
fi

# The band the templates and the repo comparison are rendered at. 16 is the top
# of http_concurrency_bands and the band where `-np` matters most: at the pin's
# auto slot count of 4 the comparator queues 12 of 16 requests.
RENDER_C=16

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

# Is <flag> PRESENT as a bare token? `flag_value` always reads the NEXT token,
# which is right for `-c 4096` and WRONG for a switch: it reported `-cb` as
# `--no-warmup`, so a template that had lost `-cb` entirely and one that still
# carried it were indistinguishable to a comparison written in terms of
# flag_value. A boolean flag needs a predicate, not an accessor.
flag_present() { # flag_present <flag> <invocation>
    if tr ' ' '\n' <<< "$2" | grep -qxF -- "$1"; then
        printf 'PRESENT'
    else
        printf 'ABSENT'
    fi
}

# The `serve run` lines of a script, with the two things that are NOT
# invocations removed: full-line comments (this file and
# check_readme_claims.sh both document the banned flag in prose) and calls to
# `accel_row`, which are this guard's own must-fire FIXTURES. Both exclusions
# are STRUCTURAL, not a filename allowlist -- a real `--gpu` anywhere else in
# any file, this one included, still REDs, and the selftest proves exactly that
# by injecting one into a copy of this file.
serve_run_stripped() { # serve_run_stripped <file>
    sed -e 's/^[[:space:]]*#.*$//' -e '/^[[:space:]]*accel_row[[:space:]]/d' "$1"
}
serve_run_lines() { # serve_run_lines <file>
    serve_run_stripped "$1" | grep 'serve run' || true
}

# The `serve run` lines that are EXECUTIONS, not MENTIONS, `<lineno>:<text>`.
#
# `serve_run_lines` above is PP-15's universe and deliberately keeps both: a
# boolean accelerator flag is banned in a printed example too, because a reader
# copies it. The --context-length rule is a different question -- what was this
# lane actually RUN at -- and reading a banner as the run is exactly how the
# assertion below came to prove nothing.
#
# MEASURED, not supposed. scripts/parity_host_receipt.sh prints
#
#     printf '  subject   : %s serve run %s ... --context-length %s\n' ...   (:245)
#
# in its --dry-run banner, ~50 lines ABOVE the line that actually launches apr
# (:346). The old check joined every `serve run` line in the file into one
# string and asked `flag_value` for `--context-length`; `flag_value` walks
# tokens in order and stops at the FIRST occurrence, so it answered with the
# BANNER's `%s\n'` -- not a bare number, therefore "declaration-derived",
# therefore ok -- and the executing line was never read at all. A hardcoded
# `--context-length 4096` there passed this guard green. `ctx_literal_*` below
# is that mutation, and it REDs.
#
# The exclusion is STRUCTURAL: a line whose first word is a printing builtin is
# a mention. Not a filename allowlist and not a line number.
executing_serve_run_lines() { # executing_serve_run_lines <file>
    serve_run_stripped "$1" | grep -n 'serve run' \
        | grep -vE '^[0-9]+:[[:space:]]*(printf|echo|print)([[:space:]]|$)' || true
}

# apr's own lane, same rule as the comparator's: a knob literal on the line that
# RUNS outranks anything the declaration says, because the two can then disagree
# while both look pinned (#2737). Prints `<lineno>=<value>` per violation, or
# nothing. A shell expansion is pin-derived, which is the point.
apr_ctx_literals() { # apr_ctx_literals <file>
    local sr n body v out=""
    while IFS= read -r sr; do
        [ -n "$sr" ] || continue
        n=${sr%%:*}
        body=${sr#*:}
        v=$(flag_value --context-length "$body")
        case "$v" in
            ABSENT|PRESENT) ;;
            \$*|\"\$*)      ;;   # a shell expansion is pin-derived
            *[!0-9]*)       ;;   # not a bare number, so not a retyped literal
            # `${n}=${v}` NOT `$n=$v`: bashrs parses the latter as an assignment
            # to a variable named by $n and reports SC1066 (error).
            *)              out="$out ${n}=${v}" ;;
        esac
    done <<< "$(executing_serve_run_lines "$1")"
    printf '%s' "${out# }"
}

# PP-15's PREDICATE, shared by the case table and the repo scan so the rows
# below prove the thing that runs. Full-line comments are stripped first: this
# file and check_readme_claims.sh both DOCUMENT the banned flag in prose, and a
# scan that reads its own explanation as a violation is a guard that can only
# fail. Prints the offending tokens, or nothing.
boolean_accel_tokens() { # boolean_accel_tokens <text>
    sed 's/^[[:space:]]*#.*$//' <<< "$1" \
        | tr ' \t' '\n\n' \
        | sed -e 's/^"//' -e 's/"$//' \
        | grep -xE -- '--gpu|--no-gpu' \
        | sort -u \
        | tr '\n' ' ' \
        | sed 's/[[:space:]]*$//'
}

# ---------------------------------------------------------------------------
selftest() {
    printf -- '--- comparator flags: declaration vs invocation ----------------------\n'
    printf '\nthe producer, against fixture declarations\n'
    TD=$(mktemp -d) || exit 2
    trap 'rm -rf "${TD:?}"' EXIT

    # A COMPLETE declaration by default, so each row states ONLY what it
    # changes. `key=DELETE` removes a key; that is how "a missing X REFUSES" is
    # expressed without every row restating ten values it does not care about.
    mkpin() { # mkpin <file> [key=value|key=DELETE ...]
        local f="$1" kv k v; shift
        {
            printf 'batch_size = "default"\n'
            printf 'comparator_parallel = "band"\n'
            printf 'context_length = 4096\n'
            printf 'threads = 8\n'
            printf 'n_ctx_slot = 1024\n'
            printf 'flash_attention = "auto"\n'
            printf 'n_ubatch = 512\n'
            printf 'cache_type_k = "f16"\n'
            printf 'cache_type_v = "f16"\n'
            printf 'cont_batching = true\n'
            printf 'slot_save = false\n'
        } > "$f"
        for kv in "$@"; do
            k=${kv%%=*}; v=${kv#*=}
            grep -v "^${k} = " "$f" > "$f.tmp" && mv "$f.tmp" "$f"
            [ "$v" = "DELETE" ] || printf '%s = %s\n' "$k" "$v" >> "$f"
        done
    }

    row() { # row <name> <c> <expected-rc> <expected-flags> [key=value ...]
        local name="$1" c="$2" want_rc="$3" want="$4" got got_rc
        shift 4
        mkpin "$TD/pin.toml" "$@"
        got=$(llama_comparator_server_flags 999 "$c" "$TD/pin.toml"); got_rc=$?
        if [ "$got_rc" != "$want_rc" ]; then
            printf 'FAIL  %-48s rc=%s, expected rc=%s\n' "$name" "$got_rc" "$want_rc"; rc=1; return
        fi
        if [ "$got" != "$want" ]; then
            printf 'FAIL  %-48s got [%s], expected [%s]\n' "$name" "$got" "$want"; rc=1; return
        fi
        printf 'ok    %-48s rc=%s [%s]\n' "$name" "$got_rc" "$got"
    }

    local BAND16='-ngl 999 -c 16384 -t 8 -np 16 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup'

    # §5.3, the decided comparator: relaunched per band, serving the band.
    row 'band mode emits -np c and -c c*n_ctx_slot' 16 0 "$BAND16"
    row 'band mode at c=1'                          1 0 \
        '-ngl 999 -c 1024 -t 8 -np 1 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup'
    row 'band mode at c=4'                          4 0 \
        '-ngl 999 -c 4096 -t 8 -np 4 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup'
    # The REPORT-only second lane (§5.3, last paragraph): llama.cpp as a user
    # gets it. Never the comparator, but expressible.
    row 'default mode emits NO -np, declared -c'    16 0 \
        '-ngl 999 -c 4096 -t 8 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'comparator_parallel="default"'
    row 'a numeric comparator_parallel emits -np'   16 0 \
        '-ngl 999 -c 4096 -t 8 -np 4 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'comparator_parallel=4'

    # THE CRIPPLE. rc 4, and NOTHING printed: a caller that word-splits the
    # output must get an empty invocation, never a partial one.
    printf '\nthe cripple, and the per-slot context floor (§5.3)\n'
    row 'batch_size = 1 REFUSES (the cripple)'      16 4 '' 'batch_size=1'
    row 'batch_size = 0 REFUSES (the cripple)'      16 4 '' 'batch_size=0'
    row 'batch_size = 2048 emits -b 2048'           16 0 \
        '-ngl 999 -c 16384 -t 8 -b 2048 -np 16 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'batch_size=2048'
    row 'batch_size = 2 is NOT the cripple'         16 0 \
        '-ngl 999 -c 16384 -t 8 -b 2 -np 16 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'batch_size=2'
    # W1 is 512 prompt + 128 generated: below 640 the slot truncates and the
    # band measures a different workload while reporting this one.
    row 'n_ctx_slot below 640 REFUSES'              16 3 '' 'n_ctx_slot=512'
    row 'n_ctx_slot exactly 640 is accepted'        16 0 \
        '-ngl 999 -c 10240 -t 8 -np 16 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'n_ctx_slot=640'

    # The refusals. A partial flag list is worse than none: it runs, and it runs
    # a comparator nobody declared.
    printf '\nrefusals: an incomplete or malformed declaration\n'
    row 'a missing context_length REFUSES'          16 3 '' 'context_length=DELETE'
    row 'a missing threads REFUSES'                 16 3 '' 'threads=DELETE'
    row 'a missing n_ctx_slot REFUSES'              16 3 '' 'n_ctx_slot=DELETE'
    row 'a missing n_ubatch REFUSES'                16 3 '' 'n_ubatch=DELETE'
    row 'a missing cache_type_k REFUSES'            16 3 '' 'cache_type_k=DELETE'
    row 'a missing cache_type_v REFUSES'            16 3 '' 'cache_type_v=DELETE'
    row 'a missing cont_batching REFUSES'           16 3 '' 'cont_batching=DELETE'
    row 'a missing slot_save REFUSES'               16 3 '' 'slot_save=DELETE'
    row 'a missing flash_attention REFUSES'         16 3 '' 'flash_attention=DELETE'
    row 'a non-numeric batch_size REFUSES'          16 3 '' 'batch_size="auto"'
    row 'a non-numeric comparator_parallel REFUSES' 16 3 '' 'comparator_parallel="auto"'
    # `flash_attention = false` is what the pin USED to declare. llama.cpp's
    # -fa takes on|off|auto, so a boolean there could never have been passed --
    # which is why the key sat unread for months while reading as enforced.
    row 'flash_attention = false REFUSES'           16 3 '' 'flash_attention=false'
    row 'flash_attention = off emits -fa off'       16 0 \
        '-ngl 999 -c 16384 -t 8 -np 16 -fa off -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'flash_attention="off"'
    # slot_save = true has no pinned path; inventing one would be a knob the
    # declaration does not govern.
    row 'slot_save = true REFUSES'                  16 3 '' 'slot_save=true'
    row 'cont_batching = false emits -nocb'         16 0 \
        '-ngl 999 -c 16384 -t 8 -np 16 -fa auto -ub 512 -ctk f16 -ctv f16 -nocb --no-warmup' \
        'cont_batching=false'
    row 'cache types come from the declaration'     16 0 \
        '-ngl 999 -c 16384 -t 8 -np 16 -fa auto -ub 512 -ctk q8_0 -ctv q8_0 -cb --no-warmup' \
        'cache_type_k="q8_0"' 'cache_type_v="q8_0"'
    row 'ctx and threads come from the declaration' 16 0 \
        '-ngl 999 -c 8192 -t 32 -fa auto -ub 512 -ctk f16 -ctv f16 -cb --no-warmup' \
        'comparator_parallel="default"' 'context_length=8192' 'threads=32'

    # The band is an ARGUMENT, and a missing or non-numeric one is a caller
    # error (rc 2), distinct from a bad declaration (rc 3).
    mkpin "$TD/pin.toml"
    for probe in "999::a missing band REFUSES" "999:abc:a non-numeric band REFUSES"; do
        pname=${probe#*:*:}
        pargs=${probe%%:*}
        pband=$(printf '%s' "$probe" | cut -d: -f2)
        out=$(llama_comparator_server_flags "$pargs" "$pband" "$TD/pin.toml"); prc=$?
        if [ "$prc" = 2 ] && [ -z "$out" ]; then
            printf 'ok    %-48s rc=2\n' "$pname"
        else
            printf 'FAIL  %-48s rc=%s [%s], expected rc=2 and no output\n' "$pname" "$prc" "$out"; rc=1
        fi
    done

    # An absent declaration file must refuse too, not default to something.
    if out=$(llama_comparator_server_flags 999 16 "$TD/does-not-exist.toml") && [ -n "$out" ]; then
        printf 'FAIL  %-48s a missing declaration produced [%s]\n' 'a missing pin file REFUSES' "$out"; rc=1
    else
        printf 'ok    %-48s refused\n' 'a missing pin file REFUSES'
    fi

    # ── PP-15: the predicate, both directions ──────────────────────────────
    printf '\nPP-15: no BOOLEAN accelerator flag (the predicate, both directions)\n'
    accel_row() { # accel_row <name> <text> <BAD|OK>
        local name="$1" text="$2" want="$3" got verdict
        got=$(boolean_accel_tokens "$text")
        if [ -n "$got" ]; then verdict=BAD; else verdict=OK; fi
        if [ "$verdict" = "$want" ]; then
            printf 'ok    %-48s %s%s\n' "$name" "$verdict" "${got:+ [$got]}"
        else
            printf 'FAIL  %-48s %s, expected %s (tokens: [%s])\n' "$name" "$verdict" "$want" "$got"
            rc=1
        fi
    }
    accel_row 'boolean_flag'                    'apr serve run m.gguf --gpu --port 8080'          BAD
    accel_row 'boolean_flag (--no-gpu)'         'run_lane cpu "--no-gpu" 0'                       BAD
    accel_row 'boolean_flag (a quoted token)'   'apr_serve_command = "apr serve run {model} --gpu --port {port}"' BAD
    accel_row 'quantity_flag'                   'apr serve run m.gguf --gpu-layers all --port 8080' OK
    accel_row 'quantity_flag (numeric)'         'apr serve run m.gguf --gpu-layers 12'            OK
    accel_row 'quantity_flag (zero, cpu lane)'  'run_lane cpu "--gpu-layers 0" 0'                 OK
    # The scan must not read its own explanation as a violation, and must not
    # be fooled by a flag that merely CONTAINS the banned token.
    accel_row 'a comment naming --gpu is prose' '# --gpu was ignored and the run went to CPU'     OK
    accel_row '--gpu-layers is not --gpu'       '--gpu-layers all'                                OK
    accel_row '--gpuish is not --gpu'           'apr serve run m.gguf --gpuish'                   OK

    # THE EXCLUSION IS STRUCTURAL, PROVED BY MUTATION. The rows above are
    # written as `accel_row '...' 'apr serve run ... --gpu ...' BAD`, so the
    # repo scan would read this guard's own must-fire fixtures as violations
    # unless it drops `accel_row` lines. That drop must not become a blanket
    # pass for this file, so the same scan is run against THIS file twice: as
    # it is, and with a real `--gpu` injected onto a `serve run` line that is
    # not a fixture. A filename allowlist would go green on both.
    self_probe() { # self_probe <name> <file> <BAD|OK>
        local name="$1" got verdict
        got=$(boolean_accel_tokens "$(serve_run_lines "$2")")
        if [ -n "$got" ]; then verdict=BAD; else verdict=OK; fi
        if [ "$verdict" = "$3" ]; then
            printf 'ok    %-48s %s%s\n' "$name" "$verdict" "${got:+ [$got]}"
        else
            printf 'FAIL  %-48s %s, expected %s\n' "$name" "$verdict" "$3"; rc=1
        fi
    }
    # ── The apr lane's --context-length, read from the line that RUNS ──────
    printf '\napr lane: --context-length is read from the EXECUTING line, not the banner\n'
    ctx_row() { # ctx_row <name> <BAD|OK> <line...>
        local name="$1" want="$2" got verdict
        shift 2
        printf '%s\n' "$@" > "$TD/exec-probe.sh"
        got=$(apr_ctx_literals "$TD/exec-probe.sh")
        if [ -n "$got" ]; then verdict=BAD; else verdict=OK; fi
        if [ "$verdict" = "$want" ]; then
            printf 'ok    %-48s %s%s\n' "$name" "$verdict" "${got:+ [$got]}"
        else
            printf 'FAIL  %-48s %s, expected %s (hits: [%s])\n' "$name" "$verdict" "$want" "$got"
            rc=1
        fi
    }
    # MUST FIRE, and this is the exact shape the old check was blind to: a
    # banner carrying `%s` FIRST, the executing line carrying a literal SECOND.
    # `flag_value` over the joined blob answered with the banner and stopped.
    ctx_row 'ctx_literal_behind_a_banner' BAD \
        "        printf '  subject   : %s serve run %s --context-length %s\\n' \\" \
        '            "$APR" "$MODEL" "$CTX"' \
        '    "$APR" serve run "$MODEL" --port 8090 --context-length 4096 \' \
        '        > "$WORK/apr.log" 2>&1 &'
    # MUST FIRE with no banner at all, so the row above is not passing for the
    # accidental reason that a banner was present.
    ctx_row 'ctx_literal_on_the_executing_line' BAD \
        '    "$APR" serve run "$MODEL" --port 8090 --context-length 4096'
    # MUST NOT FIRE: the same two lines with the declaration threaded through.
    # Single variable against `ctx_literal_behind_a_banner`: the literal.
    ctx_row 'ctx_from_the_declaration' OK \
        "        printf '  subject   : %s serve run %s --context-length %s\\n' \\" \
        '            "$APR" "$MODEL" "$CTX"' \
        '    "$APR" serve run "$MODEL" --port 8090 --context-length "$CTX" \' \
        '        > "$WORK/apr.log" 2>&1 &'
    # A literal inside a BANNER is out of scope BY DECISION, and the decision is
    # asserted rather than left implicit: the banner is not the run, and PP-15's
    # own universe (serve_run_lines) still reads printf lines for the boolean
    # flag, so nothing about printed examples is generally exempt here.
    ctx_row 'a literal in a dry-run banner is not the run' OK \
        "        printf '  subject   : %s serve run %s --context-length 4096\\n' \"\$APR\" \"\$MODEL\""
    # VACUITY. A file the scan finds no executing `serve run` line in yields no
    # hits, which is indistinguishable from a clean one -- so the gate must
    # assert the subject EXISTS, and this row proves the emptiness is real.
    ctx_row 'a file with no executing serve-run line is empty' OK \
        '    echo "nothing here"'
    if [ -n "$(executing_serve_run_lines "$TD/exec-probe.sh")" ]; then
        printf 'FAIL  %-48s the emptiness row is not empty; the vacuity check is unproven\n' \
            'the vacuity row really is empty'; rc=1
    else
        printf 'ok    %-48s no executing serve-run line, as intended\n' 'the vacuity row really is empty'
    fi

    self_probe 'this guard passes its own scan' "$0" OK
    # The injected line is ASSEMBLED, never written literally: a source line
    # here carrying both `serve run` and the bare flag would make this file
    # violate its own rule, which is how the first draft of this probe turned
    # the row above BAD.
    mut_flag='--gpu'
    { cat "$0"; printf 'apr serve run m.gguf %s --port 8080\n' "$mut_flag"; } > "$TD/mutated.sh"
    if [ "$(serve_run_lines "$TD/mutated.sh" | wc -l)" -le "$(serve_run_lines "$0" | wc -l)" ]; then
        printf 'FAIL  %-48s the mutation did not apply; the row below proves nothing\n' \
            'the mutation applies'; rc=1
    else
        printf 'ok    %-48s injected a boolean flag on a serve-run line\n' 'the mutation applies'
    fi
    self_probe 'an injected boolean in THIS file REDs' "$TD/mutated.sh" BAD
}

# ---------------------------------------------------------------------------
gate() {
    printf '\nthis repo: the declaration vs the invocations of record\n'

    PIN=scripts/llama_pin.toml
    [ -f "$PIN" ] || { printf 'FAIL  %s is missing\n' "$PIN"; exit 2; }

    BUILT=$(llama_comparator_server_flags 999 "$RENDER_C" "$PIN") || {
        printf 'FAIL  %s does not yield a comparator invocation at c=%s; see the rows above\n' \
            "$PIN" "$RENDER_C"
        exit 1
    }
    printf 'ok    built from the declaration at c=%s: [%s]\n' "$RENDER_C" "$BUILT"

    decl_of() { llama_pin_get_raw "$1" "$PIN"; }
    cmd_of() {
        sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"\(.*\)\"[[:space:]]*\$/\1/p" "$PIN" | head -1
    }

    SERVE=$(cmd_of comparator_serve_command)
    [ -n "$SERVE" ] || { printf 'FAIL  comparator_serve_command is empty\n'; rc=1; }
    # The llama-bench CLI-differential lane (`comparator_command`) is RETIRED
    # (§5.3, I-15: llama-bench is never the comparator). Its absence is asserted
    # rather than assumed, so re-adding it silently is not possible.
    if [ -n "$(cmd_of comparator_command)" ] || [ -n "$(cmd_of apr_command)" ]; then
        printf 'FAIL  the RETIRED llama-bench differential lane is back in %s.\n' "$PIN"
        printf '      §5.3 says llama-bench is never the comparator; `apr run` has no\n'
        printf '      --gpu-layers, so that lane cannot satisfy PP-15 either.\n'
        rc=1
    else
        printf 'ok    the retired llama-bench differential lane is absent\n'
    fi

    # Render {key} placeholders from the declaration, so a template that SAYS
    # {threads} is compared on the value it resolves to, not on the word. {c}
    # and {c_ctx} are RUN-TIME, not declared: they are the band and the derived
    # per-band context, which is exactly what §5.3 made vary.
    render() {
        local s="$1" k v slot
        for k in context_length threads batch_size comparator_parallel gpu_layers \
                 flash_attention n_ubatch cache_type_k cache_type_v n_ctx_slot; do
            v=$(decl_of "$k")
            s=${s//\{$k\}/$v}
        done
        slot=$(decl_of n_ctx_slot)
        case "$slot" in ''|*[!0-9]*) slot=0 ;; esac
        s=${s//\{c_ctx\}/$((RENDER_C * slot))}
        s=${s//\{c\}/$RENDER_C}
        printf '%s' "$s"
    }
    SERVE_R=$(render "$SERVE")

    # Flag by flag, both directions, naming BOTH sides on a mismatch. `-b` stays
    # in the list although the declaration says "default": ABSENT on both sides
    # is the assertion that the cripple is not being passed anyway.
    for f in -c -t -b -np -fa -ub -ctk -ctv; do
        want=$(flag_value "$f" "$BUILT")
        got=$(flag_value "$f" "$SERVE_R")
        if [ "$want" = "$got" ]; then
            printf 'ok    comparator_serve_command %-4s %s\n' "$f" "$want"
        else
            printf 'FAIL  comparator_serve_command %s is %s but the declaration builds %s\n' \
                "$f" "$got" "$want"
            printf '      declaration: batch_size=%s comparator_parallel=%s context_length=%s\n' \
                "$(decl_of batch_size)" "$(decl_of comparator_parallel)" "$(decl_of context_length)"
            printf '                   n_ctx_slot=%s threads=%s flash_attention=%s n_ubatch=%s\n' \
                "$(decl_of n_ctx_slot)" "$(decl_of threads)" \
                "$(decl_of flash_attention)" "$(decl_of n_ubatch)"
            printf '      invocation : %s\n' "$SERVE"
            rc=1
        fi
    done

    # cont_batching is a BOOLEAN mapped to one of two flags, so it cannot be a
    # {key} substitution and needs its own comparison.
    cbv=$(decl_of cont_batching)
    cb_built=$(flag_present -cb "$BUILT")
    cb_serve=$(flag_present -cb "$SERVE_R")
    nocb_serve=$(flag_present -nocb "$SERVE_R")
    case "$cbv" in
        true)  want_cb=PRESENT; want_nocb=ABSENT ;;
        false) want_cb=ABSENT;  want_nocb=PRESENT ;;
        *)     printf 'FAIL  cont_batching=%s is neither true nor false\n' "${cbv:-<unset>}"
               rc=1; want_cb="$cb_serve"; want_nocb="$nocb_serve" ;;
    esac
    if [ "$cb_serve" = "$want_cb" ] && [ "$nocb_serve" = "$want_nocb" ] && [ "$cb_built" = "$want_cb" ]; then
        printf 'ok    comparator_serve_command -cb/-nocb match cont_batching = %s\n' "$cbv"
    else
        printf 'FAIL  cont_batching = %s wants -cb %s / -nocb %s; the template has %s / %s\n' \
            "$cbv" "$want_cb" "$want_nocb" "$cb_serve" "$nocb_serve"
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

    # -----------------------------------------------------------------------
    printf '\nthis repo: the executor carries no hardcoded knob\n'

    EXEC=scripts/parity_host_receipt.sh
    [ -f "$EXEC" ] || { printf 'FAIL  %s is missing\n' "$EXEC"; exit 2; }

    # The line that actually launches the comparator. Comments are stripped
    # first: this file DOCUMENTS the old `-c 4096 -t 8 -b 1` in its own header,
    # and a scan that reads its own explanation as a violation is a guard that
    # can only fail.
    launch=$(sed 's/^[[:space:]]*#.*$//' "$EXEC" | grep -n 'LLAMA_SERVER' | grep -v '^\s*$')
    if [ -z "$launch" ]; then
        printf 'FAIL  %s no longer launches $LLAMA_SERVER; this guard has lost its subject\n' "$EXEC"
        rc=1
    else
        bad=""
        for f in -b -np -c -t -ngl -ub -ctk -ctv; do
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
        # EXECUTION, NOT MENTION. This grepped the raw file, and the file names
        # the producer in its own explanatory comment -- so mutation M8, which
        # deleted the actual call, stayed GREEN. Same defect
        # check_guards_are_wired.sh documents one level up. Comments stripped.
        if grep -q 'llama_comparator_server_flags' <<< "$(sed 's/^[[:space:]]*#.*$//' "$EXEC")"; then
            printf 'ok    %s calls llama_comparator_server_flags\n' "$EXEC"
        else
            printf 'FAIL  %s does not call llama_comparator_server_flags\n' "$EXEC"
            rc=1
        fi
    fi

    # apr's own lane, same rule, EVERY EXECUTING LINE. A literal here and a
    # declaration there is how the two sides get run at different context
    # lengths while both look pinned.
    #
    # THE SUBJECT MUST EXIST BEFORE ITS ABSENCE CAN BE READ AS CLEAN. An empty
    # scan and a clean scan produce the same empty hit list, so the count is
    # asserted first: if the executor stops launching apr, or the printf
    # exclusion widens until it swallows the real line, that is a FAIL and not
    # a silent pass.
    exec_srl=$(executing_serve_run_lines "$EXEC")
    if [ -z "$exec_srl" ]; then
        printf 'FAIL  %s has no EXECUTING `serve run` line; this scan has lost its\n' "$EXEC"
        printf '      subject and its silence means nothing\n'
        rc=1
    else
        printf 'ok    %s launches apr on %s executing `serve run` line(s)\n' \
            "$EXEC" "$(printf '%s\n' "$exec_srl" | grep -c .)"
        actx_bad=$(apr_ctx_literals "$EXEC")
        if [ -n "$actx_bad" ]; then
            printf 'FAIL  %s hardcodes --context-length on the line(s) that RUN: %s\n' "$EXEC" "$actx_bad"
            printf '      (format is <lineno>=<value>; context_length = %s is declared in %s)\n' \
                "$(decl_of context_length)" "$PIN"
            rc=1
        else
            printf 'ok    every executing `serve run` line takes --context-length from the declaration\n'
        fi
    fi

    # -----------------------------------------------------------------------
    printf '\nthis repo: PP-15, no boolean accelerator flag\n'
    #
    # THE UNIVERSE, and why it is these three parts. A PR that closes §12 row 0d
    # by editing the pin alone changes nothing that runs: no execution path
    # reads `apr_serve_command`. So the scan covers the DECLARATION (what a
    # reader reproduces from), the EXECUTOR in full (the only file that actually
    # launches apr, including the lines that build its lane flags -- the
    # boolean lived at `run_lane accel "--gpu" 999`, not on the `serve run`
    # line), and every other `serve run` line in scripts/.
    pp15_bad=""
    pin_cmds=$(sed -n 's/^[[:space:]]*[A-Za-z_]*_command[[:space:]]*=[[:space:]]*\(.*\)$/\1/p' "$PIN")
    hit=$(boolean_accel_tokens "$pin_cmds")
    [ -z "$hit" ] || pp15_bad="$pp15_bad $PIN:[$hit]"
    hit=$(boolean_accel_tokens "$(sed '/^[[:space:]]*accel_row[[:space:]]/d' "$EXEC")")
    [ -z "$hit" ] || pp15_bad="$pp15_bad $EXEC:[$hit]"
    for f in scripts/*.sh; do
        [ "$f" = "$EXEC" ] && continue
        srlines=$(serve_run_lines "$f")
        [ -n "$srlines" ] || continue
        hit=$(boolean_accel_tokens "$srlines")
        [ -z "$hit" ] || pp15_bad="$pp15_bad $f:[$hit]"
    done
    if [ -n "$pp15_bad" ]; then
        printf 'FAIL  boolean accelerator flag(s) survive:%s\n' "$pp15_bad"
        printf '      PP-15: an accelerator setting is a QUANTITY. `--gpu` was accepted,\n'
        printf '      ignored, and the run went to CPU at 15.7 tok/s with no banner and\n'
        printf '      no VRAM -- a lane labelled by intent. `--gpu-layers all|N|0`\n'
        printf '      resolves to a number the server prints about itself.\n'
        rc=1
    else
        printf 'ok    no --gpu/--no-gpu token in the pin, the executor, or any\n'
        printf '      `serve run` line in scripts/\n'
    fi
}

case "${1:-}" in
    --selftest) selftest ;;
    "")         selftest; gate ;;
    *)          printf 'usage: %s [--selftest]\n' "$0" >&2; exit 2 ;;
esac

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the declaration is the only source of the comparator knobs, the\n'
    printf '      producer refuses a cripple and an incomplete declaration, and no\n'
    printf '      accelerator flag is a boolean.\n'
else
    printf 'FAIL  see rows above (#2737, PP-15, §5.3).\n'
fi
exit "$rc"
