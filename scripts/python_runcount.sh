#!/usr/bin/env bash
# python_runcount.sh -- count the Python a command RUNS, not how its scripts spell it.
#
# WHY (#4684 -> this)
# -------------------
# The operator's ruling is that the build is Rust only. A text scan of how
# scripts invoke tools (R12, #4684) did not converge: Python reached through env,
# a variable, a heredoc or an absolute #! line reads differently every time.
# This measures at run time instead, with two independent mechanisms:
#
#   1. A PATH shim for python, python3, uv and uvx, first on PATH. It logs every
#      call with its caller, then execs the real tool.
#   2. An execve trace (strace -f) of the whole command. It catches what the shim
#      cannot: an absolute interpreter path, a versioned python3.NN, a script run
#      by its own #! line.
#
# Both feed ONE key function (lib/python_runcount/keys.awk), so a call seen by
# both is one entry point, not two. The count is the number of DISTINCT entry
# points: a script, a module, or inline/stdin code keyed by the script holding it.
#
# MODES
#   --run OUT -- CMD...   run CMD under the shim and the trace; OUT gets shim.log,
#                         trace.raw, meta. Exits with CMD's status.
#   --count OUT           print the summary line, then the entry points:
#                           PYRUN count=N shim=S trace=T trace=measured|not_measured rc=R
#                         Exit 2 when the measurement itself failed, never a count.
#   --compare BASE HEAD   never worse: HEAD's count <= BASE's. Exit 0 GREEN,
#                         1 RED (the new entry points are listed), 2 NOT_MEASURED
#                         when either side has no trace -- a shim-only count misses
#                         absolute paths, so it is never a pass -- or the head lane
#                         exited other than 0 or as base did.
#   --self-test           the plant table: env, variable, heredoc, -c, -m, uv,
#                         an absolute #! line, a versioned interpreter, a cd, exec -a,
#                         a reused pid, and broken measurements that must read 2.
#   --mutants             drop each check in a copy; every one must turn a named
#                         --self-test row RED. NOT_MEASURED if the table is not green unmutated.
#
# Report-only until it has three green nights; making it block is a sign-off.
set -euo pipefail
export LC_ALL=C   # sort and comm order keys the same on every host

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"
SELF="$HERE/${BASH_SOURCE[0]##*/}"
LIB="$HERE/lib/python_runcount"
ROOT="${PYRUN_ROOT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." > /dev/null 2>&1 && pwd )}"
SHIM_NAMES=(python python3 uv uvx)

usage() {
    printf 'usage: %s --run OUT -- CMD... | --count OUT | --compare BASE HEAD | --self-test | --mutants\n' "${0##*/}" >&2
    exit 2
}

trace_usable() { # strace present and allowed to trace a child here
    [ "${PYRUN_NO_TRACE:-0}" != 1 ] || return 1
    command -v strace > /dev/null 2>&1 || return 1
    command -v pgrep > /dev/null 2>&1 || return 1   # stop_tracer finds the tracer with it
    strace -f -qq -o /dev/null true > /dev/null 2>&1
}

tracer_pids() { # tracer_pids TRACEFILE -> the pid of each strace writing TRACEFILE
    local p c us=$'\037'
    while IFS= read -r p; do
        c=$(tr '\0' '\037' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"$us-o$us$1$us"*) printf '%s\n' "$p" ;; esac
    done < <(pgrep -x strace || :)
}

# strace runs detached (-D), so the command's own exit returns here. A
# descendant that daemonized would keep the tracer alive for ever (a guard's
# fake server once did, and the lane never ended): give the tracer a moment to
# finish by itself, then detach it (-I 2: strace blocks SIGTERM by default when
# it writes -o FILE). The descendant keeps running; it is not ours.
stop_tracer() { # stop_tracer OUT
    local p i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        [ -n "$(tracer_pids "$1/trace.raw")" ] || { printf 'detached=no\n' >> "$1/meta"; return 0; }
        sleep 0.5
    done
    while IFS= read -r p; do
        printf 'python_runcount: a descendant outlived the command; detaching the trace (pid %s)\n' "$p" >&2
        kill -TERM "$p" 2>/dev/null || :
        timeout 30 tail --pid="$p" -f /dev/null || :
    done < <(tracer_pids "$1/trace.raw")
    printf 'detached=yes\n' >> "$1/meta"
}

run_cmd() { # run_cmd OUT CMD...
    local out="$1" n rc=0
    shift
    [ "$#" -gt 0 ] || usage
    mkdir -p "$out/shim"
    : > "$out/shim.log"
    : > "$out/trace.raw"
    for n in "${SHIM_NAMES[@]}"; do
        ln -sf "$LIB/shim" "$out/shim/$n"
    done
    out="$( cd "$out" && pwd )"
    if trace_usable; then
        printf 'trace=measured\ncwd=%s\ntmpdir=%s\n' "$PWD" "${TMPDIR:-}" > "$out/meta"
        PATH="$out/shim:$PATH" PYRUN_LOG="$out/shim.log" PYRUN_SHIM_DIR="$out/shim" \
            strace -D -I 2 -f -q -s 4096 -e signal=none -e trace=execve,chdir,clone,clone3,fork,vfork -o "$out/trace.raw" "$@" || rc=$?
        stop_tracer "$out"
    else
        printf 'trace=not_measured\ncwd=%s\ntmpdir=%s\n' "$PWD" "${TMPDIR:-}" > "$out/meta"
        PATH="$out/shim:$PATH" PYRUN_LOG="$out/shim.log" PYRUN_SHIM_DIR="$out/shim" "$@" || rc=$?
    fi
    printf 'rc=%s\n' "$rc" >> "$out/meta"
    return "$rc"
}

meta() { # meta OUT KEY
    sed -n "s/^$2=//p" "$1/meta" | head -n 1
}

shebang_interp() { # shebang_interp FILE -> python*/uv/uvx if FILE's #! line runs one
    local line tok b
    line=$(head -c 256 "$1" 2>/dev/null | tr -d '\000' | head -n 1) || return 1
    case "$line" in '#!'*) ;; *) return 1 ;; esac
    line="${line#\#!}"
    read -r -a tok <<< "$line"
    [ "${#tok[@]}" -gt 0 ] || return 1
    b="${tok[0]##*/}"
    if [ "$b" = env ]; then
        b=""
        for t in "${tok[@]:1}"; do
            case "$t" in -*|*=*) continue ;; esac
            b="${t##*/}"; break
        done
    fi
    case "$b" in python|python[0-9]*|uv|uvx) printf '%s' "$b" ;; *) return 1 ;; esac
}

err2() { printf 'ERROR: %s\n' "$*" >&2; return 2; }

trace_ran() { # trace_ran TRACE -> true iff the trace is whole: a successful execve and the lane's exit
    # A real trace always holds the lane's own exec; none means strace never ran it.
    # The first pid in it is the lane. A tracer killed mid-run leaves no exit line
    # for that pid, and the execs it missed would read as a low count.
    local root
    root=$(head -n 1 "$1" | cut -d' ' -f1)
    case "$root" in ''|*[!0-9]*) return 1 ;; esac
    grep -q -E '^[0-9]+ +(execve\(|<\.\.\. execve resumed>).* = 0$' "$1" &&
        grep -q -E "^$root +\+\+\+ (exited with [0-9]+|killed by SIG[A-Z0-9]+( \(core dumped\))?) \+\+\+$" "$1"
}

# count is called as `count X || ...`, where set -e does not apply: a failed
# stage would leave an empty key list that reads as count 0. So every stage
# checks its own status, and a failure is exit 2, never a count.
count() { # count OUT -> OUT/keys, OUT/summary; prints the summary line
    local out="$1" f i n s t
    [ -r "$out/meta" ] || { printf 'ENV: %s has no meta (not a --run dir)\n' "$out" >&2; return 2; }
    if ! cat "$out/shim.log" > "$out/records"; then err2 "cannot read $out/shim.log"; return; fi
    if [ "$(meta "$out" trace)" = measured ]; then
        trace_ran "$out/trace.raw" ||
            { err2 "$out says trace=measured, but its trace is not whole (no execve, or no exit line for the lane)"; return; }
        awk -v cwd0="$(meta "$out" cwd)" -f "$LIB/trace.awk" "$out/trace.raw" "$out/trace.raw" >> "$out/records" ||
            { err2 "trace.awk failed on $out"; return; }
    fi
    awk -F'\t' '$1 == "trace" && NF >= 5 { print $5 }' "$out/records" | sort -u > "$out/execs" ||
        { err2 "cannot list the exec'd files of $out"; return; }
    : > "$out/shebangs"
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        i=$(shebang_interp "$f") || continue
        printf '%s\t%s\n' "$f" "$i" >> "$out/shebangs"
    done < "$out/execs"
    awk -v root="$ROOT" -v tmpd="$(meta "$out" tmpdir)" -f "$LIB/keys.awk" "$out/shebangs" "$out/records" > "$out/calls" ||
        { err2 "keys.awk failed on $out"; return; }
    # One line per entry point: key, then which mechanisms saw it, then its callers.
    awk -F'\t' '{ s[$1] = s[$1] (index(s[$1], $2) ? "" : (s[$1] ? "+" : "") $2)
                  c[$1] = c[$1] (index(c[$1], $3) ? "" : (c[$1] ? "," : "") $3) }
                END { for (k in s) printf "%s\t%s\t%s\n", k, s[k], c[k] }' "$out/calls" | sort > "$out/keys" ||
        { err2 "cannot aggregate the keys of $out"; return; }
    n=$(wc -l < "$out/keys") && s=$(awk -F'\t' '$2 ~ /shim/' "$out/keys" | wc -l) &&
        t=$(awk -F'\t' '$2 ~ /trace/' "$out/keys" | wc -l) || { err2 "cannot count $out/keys"; return; }
    printf 'PYRUN count=%s shim=%s trace=%s trace=%s rc=%s\n' "${n// /}" "${s// /}" "${t// /}" "$(meta "$out" trace)" "$(meta "$out" rc)" > "$out/summary" ||
        { err2 "cannot write $out/summary"; return; }
    cat "$out/summary"
}

compare() { # compare BASE HEAD
    local b="$1" h="$2" bc hc br hr
    count "$b" > /dev/null || return 2
    count "$h" > /dev/null || return 2
    if [ "$(meta "$b" trace)" != measured ] || [ "$(meta "$h" trace)" != measured ]; then
        printf 'NOT_MEASURED: no execve trace on %s -- a shim-only count is not a pass\n' "$( [ "$(meta "$b" trace)" = measured ] && printf head || printf base )"
        return 2
    fi
    # A head lane that stopped early never reached its later Python and would
    # read low. Its exit status is the evidence: 0, or the same as base's (a
    # lane that ends red on base too), else the head count is not a measurement.
    br=$(meta "$b" rc); hr=$(meta "$h" rc)
    if [ -z "$hr" ] || { [ "$hr" != 0 ] && [ "$hr" != "$br" ]; }; then
        printf 'NOT_MEASURED: the head lane exited %s (base %s), so it may have stopped before its Python\n' \
            "${hr:-?}" "${br:-?}"
        return 2
    fi
    bc=$(wc -l < "$b/keys" | tr -d ' ')
    hc=$(wc -l < "$h/keys" | tr -d ' ')
    cut -f1 "$b/keys" > "$h/base.keylist"
    cut -f1 "$h/keys" | comm -13 "$h/base.keylist" - > "$h/new.keylist"
    if [ "$hc" -le "$bc" ]; then
        printf 'GREEN python entry points head=%s base=%s\n' "$hc" "$bc"
        # The ratchet is on the count; a swap (one out, one in) is still named.
        [ ! -s "$h/new.keylist" ] || { printf 'new (not counted against the ratchet):\n'; sed 's/^/  /' "$h/new.keylist"; }
        return 0
    fi
    printf 'RED python entry points head=%s > base=%s; new:\n' "$hc" "$bc"
    sed 's/^/  /' "$h/new.keylist"
    return 1
}

# ---------------------------------------------------------------- self-test
# Each plant is a lane (a bash script) that reaches Python one way. The fake
# interpreters live in a bin dir AFTER the shim on PATH, so nothing real runs.
pyrun_cleanup() { # release the daemon plant on ANY exit (opened read-write, the write never blocks)
    [ ! -p "${PYRUN_TD:?}/fifo" ] || timeout 5 bash -c 'exec 3<> "$1"; printf x >&3' _ "$PYRUN_TD/fifo" 2>/dev/null || :
    rm -rf "${PYRUN_TD:?}"
}
rc_of() { # rc_of CMD... -> CMD's exit status, its output dropped
    local r=0
    "$@" > /dev/null 2>&1 || r=$?
    printf '%s' "$r"
}
self_test() {
    local fails=0 rows=0 rc td n f p k o
    PYRUN_TD=$(mktemp -d "${TMPDIR:-/tmp}/pyrun.XXXXXX"); td=$PYRUN_TD
    trap 'pyrun_cleanup' EXIT   # td is local; the trap runs after it is gone
    # The fake interpreters open what CPython opens (lib/python_runcount/fakes/python).
    # They sit AFTER the shim on PATH, so nothing real runs.
    mkdir -p "$td/bin" "$td/w/sub" "$td/w/lanes" "$td/lib/encodings/__pycache__" "$td/lib/json/__pycache__" "$td/lib/__pycache__"
    for f in encodings/__pycache__/__init__ json/__pycache__/__init__ json/__pycache__/tool __pycache__/compileall; do
        : > "$td/lib/$f.cpython-313.pyc"
    done
    for n in python3 python3.11 python3.13t pythonw pypy3 python3.12d; do
        ln -sf "$LIB/fakes/python" "$td/bin/$n"
    done
    for n in uv uvx; do
        ln -sf "$LIB/fakes/uv" "$td/bin/$n"
    done
    mkfifo "$td/fifo"
    # The repo the lanes run in. Its tracked files key by path, anything else by content.
    for f in tool a b c d; do
        printf '# %s\n' "$f" > "$td/w/sub/$f.py"
    done
    printf '# t\n' > "$td/w/sub/$(printf 't\303\266\303\266l')".py
    printf '# w\n' > "$td/w/sub/two words.py"
    printf '#!%s/bin/python3\n' "$td" > "$td/w/abs_tool"
    chmod 755 "$td/w/abs_tool"
    cp "$LIB"/plants/*.sh "$td/w/lanes/"
    git -C "$td/w" init -q && git -C "$td/w" add -A ||
        { printf 'NOT_MEASURED: cannot make the plant repo (git)\n'; return 2; }
    _eq() {
        rows=$((rows + 1))
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"
        else fails=$((fails + 1)); printf 'FAIL  %s: got "%s", wanted "%s"\n' "$1" "$3" "$2"; fi
    }
    # plant NAME NO_TRACE(0|1) OUT: run lanes/NAME.sh in the plant repo, into o_OUT
    plant() {
        local o="$td/o_$3"
        ( cd "$td/w" && env "PATH=$td/bin:$PATH" "PYRUN_NO_TRACE=$2" "PYRUN_TEST_FIFO=$td/fifo" "PYRUN_FAKE_LIB=$td/lib" "PYRUN_FAKE_BIN=$td/bin" "PYRUN_ROOT=$td/w" timeout 60 bash "$SELF" --run "$o" -- bash "lanes/$1.sh" ) < /dev/null > /dev/null 2>&1 || :
        PYRUN_ROOT=/nonexistent count "$o" > /dev/null 2>&1 || :   # the root is the run's (meta)
    }
    synth() { # synth OUT META < TRACE -> a --run dir holding that trace, counted
        mkdir -p "$td/o_$1"; printf '%s\n' "$2" > "$td/o_$1/meta"; : > "$td/o_$1/shim.log"
        sed "s|@W@|$td/w|g" > "$td/o_$1/trace.raw"
        count "$td/o_$1" > /dev/null 2>&1 || :
    }
    dup() { cp -r "$td/o_$1" "$td/o_$2"; }   # dup FROM TO: a copy of run o_FROM to break
    keys() { cut -f1 "$td/o_$1/keys" 2>/dev/null | paste -sd' ' -; }
    srcs() { cut -f2 "$td/o_$1/keys" 2>/dev/null | paste -sd' ' -; }
    h() { printf "$1" | head -c 65536 | sha256sum | cut -c1-12; }   # the content key of a temp script
    if ! trace_usable; then
        printf 'NOT_MEASURED: strace cannot trace here; the trace rows cannot run\n'
        return 2
    fi
    local -a lanes=(none env_call var_call heredoc inline module uv_run abs_shebang versioned twice cd_call renamed tmp_script tmp_caller
        multiline flag_forms daemon mflag tmp_names uv_flags freethreaded interp_names octal first_match gone same_content two_inline)
    for p in "${lanes[@]}"; do
        plant "$p" 0 "$p"
    done
    plant tmp_script 0 tmp_script2
    plant tmp_caller 0 tmp_caller2
    plant tmp_names 0 tmp_names2
    plant abs_shebang 1 abs_noshim_trace
    plant none 1 none_notrace
    _eq 'P0 a lane with no Python counts 0, measured (the keys file exists)' '0 measured' "$(wc -l < "$td/o_none/keys" | tr -d ' ') $(meta "$td/o_none" trace)"
    _eq 'P1 env python3 tool.py counts the script' 'script:sub/tool.py' "$(keys env_call)"
    _eq 'P2 "$PY" (a variable) counts the script' 'script:sub/tool.py' "$(keys var_call)"
    _eq 'P3 a heredoc on stdin counts, keyed by its lane' 'inline@lanes/heredoc.sh' "$(keys heredoc)"
    _eq 'P4 python3 -c counts, keyed by its lane' 'inline@lanes/inline.sh' "$(keys inline)"
    _eq 'P5 python3 -m counts the module it opened' '-m:json.tool' "$(keys module)"
    _eq 'P6 uv run counts uv at its call site, and the python it starts' 'script:sub/tool.py uv@lanes/uv_run.sh' "$(keys uv_run)"
    _eq 'P7 an absolute #! line counts (the trace sees it)' 'script:abs_tool' "$(keys abs_shebang)"
    _eq 'P8 an absolute #! line is trace-only: the shim alone misses it' 'trace' "$(srcs abs_shebang)"
    _eq 'P9 a versioned python3.11 (not shimmed) counts' 'script:sub/tool.py' "$(keys versioned)"
    _eq 'P10 the shim and the trace agree on the process (one entry point, both saw it)' 'shim+trace' "$(srcs env_call)"
    _eq 'P11 the same script run twice is one entry point' '1' "$(wc -l < "$td/o_twice/keys" | tr -d ' ')"
    _eq 'P12 with no trace the absolute #! line is missed and says not_measured' '0 not_measured' "$(wc -l < "$td/o_abs_noshim_trace/keys" | tr -d ' ') $(meta "$td/o_abs_noshim_trace" trace)"
    _eq 'P13 a lane that changes dir first: the trace follows chdir, both agree on one key' 'script:sub/tool.py shim+trace' "$(keys cd_call) $(srcs cd_call)"
    _eq 'P14 a renamed argv[0] (exec -a) still counts: the trace keys on the file' 'script:sub/tool.py trace' "$(keys renamed) $(srcs renamed)"
    _eq 'P15 a script in a fresh mktemp dir keys by its content, so reruns agree' "script:#$(h '')" "$(keys tmp_script)"
    _eq 'P16 inline Python in a mktemp script keys by one stable caller' 'inline@~tmp' "$(keys tmp_caller)"
    _eq 'P17 inline code with a newline and a tab is one shim record of 3 fields, one key' 'inline@lanes/multiline.sh shim+trace 1 0' \
        "$(keys multiline) $(srcs multiline) $(wc -l < "$td/o_multiline/shim.log" | tr -d ' ') $(awk -F'\t' 'NF != 3' "$td/o_multiline/shim.log" | wc -l | tr -d ' ')"
    _eq 'P18 -W/-X values, joined or not, are not -c/-m; code under bash -ec belongs to its lane' 'inline@lanes/flag_forms.sh script:sub/tool.py' "$(keys flag_forms)"
    _eq 'P19 -m joined to other flags or to its module is the module (-mcompileall, -Im); -Bc is inline' '-m:compileall -m:json.tool inline@lanes/mflag.sh' "$(keys mflag)"
    local want20
    want20=$(for n in a b c d; do printf 'script:%s%s\n' '#' "$(h "\\043 $n\\n")"; done; printf 'inline@~tmp\n')
    _eq 'P20 -XXXXXX suffixes, pid names, a mktemp dir and a mktemp file in the repo key by content' \
        "$(printf '%s\n' "$want20" | sort | paste -sd' ' -)" "$(keys tmp_names)"
    _eq 'P21 uv options that take a value are never read: uv keys by its call site' 'script:sub/tool.py uv@lanes/uv_flags.sh' "$(keys uv_flags)"
    _eq 'P22 a free-threaded python3.13t counts' 'script:sub/tool.py' "$(keys freethreaded)"
    _eq 'P23 pythonw, pypy3, python3.12d and env -u each count' 'script:sub/a.py script:sub/b.py script:sub/c.py script:sub/d.py' "$(keys interp_names)"
    _eq 'P24 a name strace escapes (octal bytes, a space) keys as the shim saw it' \
        "script:sub/two words.py script:sub/$(printf 't\303\266\303\266l').py shim+trace shim+trace" "$(keys octal) $(srcs octal)"
    _eq 'P25 the entry point is the first file opened that the argv names, not a data file after it' '-m:compileall script:sub/tool.py' "$(keys first_match)"
    _eq 'P26 a temp script no shim hashed, deleted before --count, keys as gone by its caller' 'script:gone@lanes/gone.sh' "$(keys gone)"
    _eq 'N1 two temp scripts with the same code are one entry point (the contract: content, not name)' "script:#$(h '# same\n')" "$(keys same_content)"
    _eq 'N2 inline code is keyed by the lane that holds it: two -c and a heredoc are one entry point' 'inline@lanes/two_inline.sh' "$(keys two_inline)"
    _eq 'S1 two runs of the same lanes give the same keys (fresh mktemp names each run)' 'yes' \
        "$( [ "$(keys tmp_script) $(keys tmp_caller) $(keys tmp_names)" = "$(keys tmp_script2) $(keys tmp_caller2) $(keys tmp_names2)" ] && echo yes || echo no)"
    # A descendant that daemonized must not hold the run open, nor leave the tracer behind.
    _eq 'D1 a daemonized descendant does not hang --run (rc, key)' '0 script:sub/tool.py' "$(meta "$td/o_daemon" rc) $(keys daemon)"
    _eq 'D2 the tracer is detached, not left running' 'yes 0' "$(meta "$td/o_daemon" detached) $(tracer_pids "$td/o_daemon/trace.raw" | wc -l | tr -d ' ')"
    timeout 5 bash -c 'printf x > "$1"' _ "$td/fifo" || :   # release the daemon
    strace -qq -o "$td/d3.raw" sleep 30 > /dev/null 2>&1 &
    p=$!
    for n in 1 2 3 4 5 6 7 8 9 10; do
        ! tracer_pids "$td/d3.raw" | grep -q . || break; sleep 0.2
    done
    _eq 'D3 tracer_pids finds a live strace by its -o file (pgrep)' '1' "$(tracer_pids "$td/d3.raw" | wc -l | tr -d ' ')"
    kill "$p" 2>/dev/null || :; wait "$p" 2>/dev/null || :
    # --compare: never worse, head vs base; a shim-only side is NOT_MEASURED.
    rc=0; compare "$td/o_none" "$td/o_env_call" > "$td/c1" 2>&1 || rc=$?
    _eq 'C1 head adds an entry point -> RED (1), and names it' '1 yes' "$rc $(grep -q -e '^  script:sub/tool.py$' "$td/c1" && echo yes || echo no)"
    rc=0; compare "$td/o_env_call" "$td/o_none" > /dev/null 2>&1 || rc=$?
    _eq 'C2 head removes one -> GREEN (0)' '0' "$rc"
    rc=0; compare "$td/o_env_call" "$td/o_var_call" > /dev/null 2>&1 || rc=$?
    _eq 'C3 head = base -> GREEN (0)' '0' "$rc"
    rc=0; compare "$td/o_none_notrace" "$td/o_env_call" > /dev/null 2>&1 || rc=$?
    _eq 'C4 a side with no trace -> NOT_MEASURED (2), never a pass' '2' "$rc"
    rc=0; compare "$td/o_env_call" "$td/o_heredoc" > "$td/c5" 2>&1 || rc=$?
    _eq 'C5 a swap (one out, one in) is GREEN (0) and still names the new one' '0 yes' "$rc $(grep -q -e '^  inline@lanes/heredoc.sh$' "$td/c5" && echo yes || echo no)"
    rc=0
    ( cd "$td/w" && PATH="$td/bin:$PATH" bash "$SELF" --run "$td/o_rc" -- bash -c 'exit 7' ) > /dev/null 2>&1 || rc=$?
    _eq 'R1 --run exits with the command status' '7' "$rc"
    # Synthetic traces: what a live run cannot be made to do on demand.
    k="trace=measured"$'\n'"cwd=$td/w"$'\n'"root=$td/w"$'\n'"tmpdir=/scratch/q"$'\n'"rc=0"
    # A pid the kernel reused is a new process: it must not inherit the old one's cwd or caller.
    synth reuse "$k" <<'SYN'
100 execve("/bin/bash", ["bash", "/opt/lanes/lane1.sh"], 0x7ffd /* 5 vars */) = 0
100 openat(AT_FDCWD, "/opt/lanes/lane1.sh", O_RDONLY) = 3
100 clone(child_stack=NULL, flags=SIGCHLD) = 200
200 chdir("/elsewhere") = 0
200 +++ exited with 0 +++
100 clone(child_stack=NULL, flags=SIGCHLD) = 300
300 execve("/bin/bash", ["bash", "/opt/lanes/lane2.sh"], 0x7ffd /* 5 vars */) = 0
300 openat(AT_FDCWD, "/opt/lanes/lane2.sh", O_RDONLY) = 3
300 clone(child_stack=NULL, flags=SIGCHLD) = 200
200 execve("/usr/bin/python3", ["python3", "sub/tool.py"], 0x7ffd /* 5 vars */) = 0
200 openat(AT_FDCWD, "@W@/sub/tool.py", O_RDONLY|O_CLOEXEC) = 3
200 +++ exited with 0 +++
300 +++ exited with 0 +++
100 +++ exited with 0 +++
SYN
    _eq 'T1 a reused pid keys by its own cwd and caller, not the dead process'"'"'s' 'script:sub/tool.py lane2.sh' "$(cut -f1,3 "$td/o_reuse/keys" 2>/dev/null | tr '\t' ' ')"
    # The temp dir is the one the lane ran with (meta), not whatever TMPDIR --count sees.
    mkdir -p "$td/o_tmpd"; printf '%s\n' "$k" > "$td/o_tmpd/meta"; : > "$td/o_tmpd/shim.log"
    cat > "$td/o_tmpd/trace.raw" <<'SYN'
100 execve("/bin/bash", ["bash", "/opt/lanes/lane.sh"], 0x7ffd /* 5 vars */) = 0
100 openat(AT_FDCWD, "/opt/lanes/lane.sh", O_RDONLY) = 3
100 clone(child_stack=NULL, flags=SIGCHLD) = 200
200 execve("/usr/bin/python3", ["python3", "/scratch/q/x7/cell.py"], 0x7ffd /* 5 vars */) = 0
200 openat(AT_FDCWD, "/scratch/q/x7/cell.py", O_RDONLY|O_CLOEXEC) = 3
200 +++ exited with 0 +++
100 +++ exited with 0 +++
SYN
    TMPDIR=/elsewhere count "$td/o_tmpd" > /dev/null 2>&1 || :
    _eq 'T2 a script under the TMPDIR of the run is a temp script, whatever TMPDIR --count has' 'script:gone@lane.sh' "$(keys tmpd)"
    _eq 'T3 the repo root is the run'"'"'s (meta), not PYRUN_ROOT at count time' 'script:sub/tool.py' \
        "$(PYRUN_ROOT=/nonexistent bash "$SELF" --count "$td/o_env_call" 2>/dev/null | sed -n '2p' | cut -f1)"
    synth fchdir "$k" <<'SYN'
100 execve("/bin/bash", ["bash", "/opt/lanes/lane.sh"], 0x7ffd /* 5 vars */) = 0
100 openat(AT_FDCWD, "/opt/lanes/lane.sh", O_RDONLY) = 3
100 openat(AT_FDCWD, "sub", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
100 fchdir(4) = 0
100 clone(child_stack=NULL, flags=SIGCHLD) = 200
200 execve("/usr/bin/python3", ["python3", "tool.py"], 0x7ffd /* 5 vars */) = 0
200 openat(AT_FDCWD, "@W@/sub/tool.py", O_RDONLY|O_CLOEXEC) = 3
200 +++ exited with 0 +++
100 +++ exited with 0 +++
SYN
    _eq 'T4 fchdir moves the working dir like chdir' 'script:sub/tool.py' "$(keys fchdir)"
    synth execveat "$k" <<'SYN'
100 execve("/bin/bash", ["bash", "/opt/lanes/lane.sh"], 0x7ffd /* 5 vars */) = 0
100 openat(AT_FDCWD, "/opt/lanes/lane.sh", O_RDONLY) = 3
100 clone(child_stack=NULL, flags=SIGCHLD) = 200
200 execveat(AT_FDCWD, "/usr/bin/python3", ["python3", "sub/tool.py"], 0x7ffd /* 5 vars */, 0) = 0
200 openat(AT_FDCWD, "@W@/sub/tool.py", O_RDONLY|O_CLOEXEC) = 3
200 +++ exited with 0 +++
100 +++ exited with 0 +++
SYN
    _eq 'T5 execveat starts a process like execve' 'script:sub/tool.py' "$(keys execveat)"
    synth split "$k" <<'SYN'
100 execve("/bin/bash", ["bash", "/opt/lanes/lane.sh"], 0x7ffd /* 5 vars */) = 0
100 openat(AT_FDCWD, "/opt/lanes/lane.sh", O_RDONLY) = 3
100 clone(child_stack=NULL, flags=SIGCHLD <unfinished ...>
200 execve("/usr/bin/python3", ["python3", "sub/tool.py"], 0x7ffd /* 5 vars */ <unfinished ...>
100 <... clone resumed>) = 200
200 <... execve resumed>) = 0
200 openat(AT_FDCWD, "@W@/sub/tool.py", O_RDONLY|O_CLOEXEC <unfinished ...>
100 wait4(-1 <unfinished ...>
200 <... openat resumed>) = 3
200 +++ exited with 0 +++
100 <... wait4 resumed>) = 200
100 +++ exited with 0 +++
SYN
    _eq 'T6 a call strace split (<unfinished ...> / resumed) is read whole' 'script:sub/tool.py lane.sh' "$(cut -f1,3 "$td/o_split/keys" 2>/dev/null | tr '\t' ' ')"
    _eq 'L1 keys do not depend on the locale --count is started in (strace octal bytes)' "$(keys octal)" \
        "$(LC_ALL=C.UTF-8 bash "$SELF" --count "$td/o_octal" 2>/dev/null | sed '1d' | cut -f1 | paste -sd' ' -)"
    # Never a count from a broken measurement: each is exit 2, not GREEN 0.
    dup none e
    mkdir -p "$td/broken/lib"; cp "$SELF" "$td/broken/runcount.sh"; cp -r "$LIB" "$td/broken/lib/"
    printf '}\n' >> "$td/broken/lib/python_runcount/keys.awk"
    rc=$(rc_of bash "$td/broken/runcount.sh" --compare "$td/o_env_call" "$td/o_e")
    _eq 'E1 a failing key stage is NOT_MEASURED (2), never count 0' '2' "$rc"
    : > "$td/o_e/trace.raw"
    _eq 'E2 trace=measured with no execve in the trace is NOT_MEASURED (2)' '2' "$(rc_of compare "$td/o_env_call" "$td/o_e")"
    rm -rf "${td:?}/o_e"; dup none e
    sed -i 's/^rc=0$/rc=3/' "$td/o_e/meta"
    _eq 'E3 a head lane that exited 3 where base exited 0 is NOT_MEASURED (2)' '2' "$(rc_of compare "$td/o_env_call" "$td/o_e")"
    dup env_call e_base; sed -i 's/^rc=0$/rc=3/' "$td/o_e_base/meta"
    _eq 'E4 head and base ending with the same status are measured (GREEN 0)' '0' "$(rc_of compare "$td/o_e_base" "$td/o_e")"
    # A tracer killed mid-run leaves a cut trace: execs, but no exit line for the lane.
    dup env_call e5
    head -n "$(( $(wc -l < "$td/o_env_call/trace.raw") / 2 ))" "$td/o_env_call/trace.raw" > "$td/o_e5/trace.raw"
    _eq 'E5 a cut trace (no exit line for the lane, rc 0) is NOT_MEASURED (2)' '2' "$(rc_of compare "$td/o_env_call" "$td/o_e5")"
    dup env_call e6; sed -i "s|^root=.*|root=$td/bin|" "$td/o_e6/meta"
    _eq 'E6 a repo root git cannot list (tracked files unknown) is NOT_MEASURED (2)' '2' "$(rc_of compare "$td/o_env_call" "$td/o_e6")"
    _eq 'M0 --mutants with no green unmutated table is NOT_MEASURED (2), never killed=N' '2' "$(PYRUN_NO_TRACE=1 rc_of bash "$SELF" --mutants)"
    printf '\n%s row(s), %s red / %s\n' "$rows" "$fails" "$( [ "$fails" -eq 0 ] && echo GREEN || echo RED )"
    [ "$fails" -eq 0 ]
}

# ---------------------------------------------------------------- mutants
# Each mutant removes one thing a row depends on, in a copy of this script and
# its lib. A patch that does not change its file is an ERROR, never a kill.
mutants() {
    local killed=0 total=0 errors=0 id rel expr out row
    PYRUN_MT=$(mktemp -d "${TMPDIR:-/tmp}/pyrun-mut.XXXXXX")
    trap 'rm -rf "${PYRUN_MT:?}"' EXIT
    # A mutant is judged against a GREEN unmutated table. If the table cannot
    # run here (no strace, ptrace denied), every mutant would "fail" and read killed.
    if ! bash "$SELF" --self-test > "$PYRUN_MT/base" 2>&1; then
        printf 'NOT_MEASURED: --self-test is not green unmutated, so no mutant can be judged:\n'
        tail -n 3 "$PYRUN_MT/base"
        return 2
    fi
    while IFS='|' read -r id rel expr; do
        [ -n "$id" ] || continue
        total=$((total + 1))
        rm -rf "${PYRUN_MT:?}/t"; mkdir -p "$PYRUN_MT/t/lib"
        cp "$SELF" "$PYRUN_MT/t/"; cp -r "$LIB" "$PYRUN_MT/t/lib/"
        cp "$PYRUN_MT/t/$rel" "$PYRUN_MT/orig"
        sed -i -e "$expr" "$PYRUN_MT/t/$rel"
        if cmp -s "$PYRUN_MT/orig" "$PYRUN_MT/t/$rel"; then
            errors=$((errors + 1)); printf 'ERROR     %s: the patch did not apply\n' "$id"; continue
        fi
        out="$PYRUN_MT/out"
        if bash "$PYRUN_MT/t/${SELF##*/}" --self-test > "$out" 2>&1; then
            printf 'SURVIVED  %s\n' "$id"
        elif row=$(grep -m1 -o -e '^FAIL  [A-Z][0-9]*' "$out"); then
            killed=$((killed + 1))   # a kill is a named red row, nothing less
            printf 'killed    %s by %s\n' "$id" "${row#FAIL  }"
        else
            errors=$((errors + 1))
            printf 'ERROR     %s: the self-test failed with no red row: %s\n' "$id" "$(tail -n 1 "$out")"
        fi
    done <<'TABLE'
M1 drop chdir|lib/python_runcount/trace.awk|s/    cwd\[pid\] = norm(unq(d), cwd\[pid\]); next/    next/
M2 no inheritance from the parent|lib/python_runcount/trace.awk|s/cwd\[pid\] = (p != "" ? cwd\[p\] : cwd0); argv\[pid\] = (p != "" ? argv\[p\] : "?")/cwd[pid] = cwd0; argv[pid] = "?"/
M3 no shebang table|python_runcount.sh|s/i=$(shebang_interp "$f") || continue/continue/
M4 drop the shim log|python_runcount.sh|s/^    if ! cat "$out\/shim.log" > "$out\/records"; then/    if ! : > "$out\/records"; then/
M5 not_measured passes|python_runcount.sh|/a shim-only count is not a pass/{n;s/return 2/return 0/}
M6 count calls, not entry points|python_runcount.sh|s/END { for (k in s) printf "%s\\t%s\\t%s\\n", k, s\[k\], c\[k\] }/{ print }/
M7 only python and python3|lib/python_runcount/keys.awk|s/\^python\[0-9.\]\*t?\$/^python3?$/
M8 a silent shim|lib/python_runcount/shim|s/>> "$log"$/> \/dev\/null/
M9 NR == FNR on an empty shebang file|lib/python_runcount/keys.awk|s/^FILENAME == ARGV\[1\]/NR == FNR/
M10 equal counts read RED|python_runcount.sh|s/\[ "$hc" -le "$bc" \]/[ "$hc" -lt "$bc" ]/
M11 PYRUN_NO_TRACE ignored|python_runcount.sh|s/\[ "${PYRUN_NO_TRACE:-0}" != 1 \] || return 1/:/
M12 stdin code loses its lane|lib/python_runcount/keys.awk|s/^\(        if (x == "-") return "stdin\)@" who/\1"/
M13 argv[0] alone names the interpreter|lib/python_runcount/keys.awk|s/if (src == "trace" \&\& !interp(name) \&\& interp(base(file))) name = base(file)/if (0) name = base(file)/
M14 temp paths keyed raw|lib/python_runcount/keys.awk|s/^    if (!rel \&\& tmpath(p)) return/    if (0) return/
M15 mktemp names kept|lib/python_runcount/keys.awk|s/^function stable(s, lvl,    out, r) {/function stable(s, lvl,    out, r) { return s/
M16 strace attached (not -D)|python_runcount.sh|s/strace -D -I 2 -f/strace -I 2 -f/
M17 tracer left running|python_runcount.sh|s/^        stop_tracer "$out"$/        :/
M18 strace ignores SIGTERM (-I 3, its default with -o)|python_runcount.sh|s/strace -D -I 2 -f/strace -D -I 3 -f/
M19 newlines kept in a shim record|lib/python_runcount/shim|\%^    c=\${c//\$'\\n'%d
M20 tabs kept in a shim record|lib/python_runcount/shim|\%^    c=\${c//\$'\\t'%d
M21 a failed key stage reads as a count|python_runcount.sh|s/{ err2 "keys.awk failed on $out"; return; }/:/
M22 a trace with no execve passes|python_runcount.sh|s/^trace_ran() { # /trace_ran() { return 0; # /
M23 the head lane's exit status ignored|python_runcount.sh|s/\[ "$hr" != "$br" \]; }/false; }/
M24 a reused pid keeps the dead process's state|lib/python_runcount/trace.awk|/^rest ~ \/^\\+\\+\\+ (exited|killed) \/ {/d
M25 joined -W/-X values read as -c/-m|lib/python_runcount/keys.awk|/-Wonce, -Xutf8/d
M26 bash -ec is not an inline-shell caller|lib/python_runcount/keys.awk|s/if (a\[i\] ~ \/^-\[A-Za-z\]\*c\$\/) return b " -c"/if (a[i] == "-c") return b " -c"/
M27 a cut trace (no exit line for the lane) passes|python_runcount.sh|s/ grep -q -E "^$root / true || grep -q -E "^$root /
M28 -c tested before -m in a joined flag|lib/python_runcount/keys.awk|s/j = match(f, \/\[cm\]\/)/j = match(f, \/c\/)/
M29 -XXXXXX suffixes kept|lib/python_runcount/keys.awk|s/    if (lvl >= 1 \&\& length(r) >= 6/    if (0 \&\& length(r) >= 6/
M30 pid names kept|lib/python_runcount/keys.awk|s/if (r ~ \/^\[0-9\]+\$\/ \&\& length(r) >= 5) return 1/if (0) return 1/
M31 a mktemp dir in the repo keyed raw|lib/python_runcount/keys.awk|s/p = p stable(parts\[i\], 1) "\/"/p = p parts[i] "\/"/
M32 TMPDIR read at count time, not from the run|python_runcount.sh|s/ -v tmpd="$(meta "$out" tmpdir)"//
M33 a caller in a temp dir keyed raw|lib/python_runcount/keys.awk|s/^function cname(p) { return stable(base(p), tmpath(p) ? 2 : 0) }/function cname(p) { return stable(base(p), 0) }/
M34 uv long options with a value read as the script|lib/python_runcount/keys.awk|s/|color|index-strategy|/|/
M35 uv short options with a value read as the script|lib/python_runcount/keys.awk|s/\^-\[pwcrifCP\]\$/^-[pw]$/
M36 a repo-local tmp.XXXXXXXXXX kept|lib/python_runcount/keys.awk|s/    if (length(r) == 10 \&\& before/    if (0 \&\& before/
TABLE
    printf '\nkilled=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

case "${1:-}" in
    --run)
        [ "$#" -ge 4 ] && [ "$3" = "--" ] || usage
        out="$2"; shift 3
        run_cmd "$out" "$@"
        ;;
    --count) [ "$#" -eq 2 ] || usage; count "$2" && cat "$2/keys" ;;
    --compare) [ "$#" -eq 3 ] || usage; compare "$2" "$3" ;;
    --self-test) self_test ;;
    --mutants) mutants ;;
    *) usage ;;
esac
