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
        printf 'trace=measured\ncwd=%s\n' "$PWD" > "$out/meta"
        PATH="$out/shim:$PATH" PYRUN_LOG="$out/shim.log" PYRUN_SHIM_DIR="$out/shim" \
            strace -D -I 2 -f -q -s 4096 -e signal=none -e trace=execve,chdir,clone,clone3,fork,vfork -o "$out/trace.raw" "$@" || rc=$?
        stop_tracer "$out"
    else
        printf 'trace=not_measured\ncwd=%s\n' "$PWD" > "$out/meta"
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

trace_ran() { # trace_ran TRACE -> true iff the trace holds a successful execve
    # A real trace always holds the lane's own exec; none means strace never ran it.
    grep -q -E '^[0-9]+ +(execve\(|<\.\.\. execve resumed>).* = 0$' "$1"
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
            { err2 "$out says trace=measured, but its trace holds no execve"; return; }
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
    awk -v root="$ROOT" -f "$LIB/keys.awk" "$out/shebangs" "$out/records" > "$out/calls" ||
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
    local fails=0 rows=0 rc td
    PYRUN_TD=$(mktemp -d "${TMPDIR:-/tmp}/pyrun.XXXXXX"); td=$PYRUN_TD
    trap 'pyrun_cleanup' EXIT   # td is local; the trap runs after it is gone
    mkdir -p "$td/bin" "$td/w/sub"
    mkfifo "$td/fifo"
    for n in python3 python3.11 uv; do
        printf '#!/usr/bin/env bash\ncat > /dev/null 2>&1 || :\nexit 0\n' > "$td/bin/$n"
        chmod 755 "$td/bin/$n"
    done
    printf '#!%s/bin/python3\n' "$td" > "$td/w/abs_tool"
    chmod 755 "$td/w/abs_tool"
    _eq() {
        rows=$((rows + 1))
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"
        else fails=$((fails + 1)); printf 'FAIL  %s: got "%s", wanted "%s"\n' "$1" "$3" "$2"; fi
    }
    plant() { # plant NAME NO_TRACE(0|1) OUT -> runs plants/NAME.sh in $td/w into $td/o_OUT
        local o="$td/o_$3"
        ( cd "$td/w" && PATH="$td/bin:$PATH" PYRUN_NO_TRACE="$2" PYRUN_TEST_FIFO="$td/fifo" timeout 60 bash "$SELF" --run "$o" -- bash "$LIB/plants/$1.sh" ) < /dev/null > /dev/null 2>&1 || :
        count "$o" > /dev/null 2>&1 || :
    }
    keys() { cut -f1 "$td/o_$1/keys" 2>/dev/null | paste -sd' ' -; }
    srcs() { cut -f2 "$td/o_$1/keys" 2>/dev/null | paste -sd' ' -; }
    if ! trace_usable; then
        printf 'NOT_MEASURED: strace cannot trace here; the trace rows cannot run\n'
        return 2
    fi
    ROOT="$td/w"
    for p in none env_call var_call heredoc inline module uv_run abs_shebang versioned twice cd_call renamed tmp_script tmp_caller multiline flag_forms daemon; do
        plant "$p" 0 "$p"
    done
    plant tmp_script 0 tmp_script2
    plant tmp_caller 0 tmp_caller2
    plant abs_shebang 1 abs_noshim_trace
    plant none 1 none_notrace
    _eq 'P0 a lane with no Python counts 0, measured (the keys file exists)' '0 measured' "$(wc -l < "$td/o_none/keys" | tr -d ' ') $(meta "$td/o_none" trace)"
    _eq 'P1 env python3 tool.py counts the script' 'script:sub/tool.py' "$(keys env_call)"
    _eq 'P2 "$PY" (a variable) counts the script' 'script:sub/tool.py' "$(keys var_call)"
    _eq 'P3 a heredoc on stdin counts, keyed by its lane' 'stdin@heredoc.sh' "$(keys heredoc)"
    _eq 'P4 python3 -c counts, keyed by its lane' '-c@inline.sh' "$(keys inline)"
    _eq 'P5 python3 -m counts the module' '-m:json.tool' "$(keys module)"
    _eq 'P6 uv run tool.py counts' 'uv:run:script:sub/tool.py' "$(keys uv_run)"
    _eq 'P7 an absolute #! line counts (the trace sees it)' 'script:abs_tool' "$(keys abs_shebang)"
    _eq 'P8 an absolute #! line is trace-only: the shim alone misses it' 'trace' "$(srcs abs_shebang)"
    _eq 'P9 a versioned python3.11 (not shimmed) counts' 'script:sub/tool.py' "$(keys versioned)"
    _eq 'P10 the shim and the trace agree on the key (one entry point, both saw it)' 'shim+trace' "$(srcs env_call)"
    _eq 'P13 a lane that changes dir first: the trace follows chdir, both agree on one key' 'script:sub/tool.py shim+trace' "$(keys cd_call) $(srcs cd_call)"
    _eq 'P14 a renamed argv[0] (exec -a) still counts: the trace keys on the file' 'script:sub/tool.py trace' "$(keys renamed) $(srcs renamed)"
    _eq 'P15 a script in a fresh mktemp dir keys by name, so reruns agree' 'script:~tmp/cell.py' "$(keys tmp_script)"
    _eq 'P16 inline Python in a mktemp script keys by one stable caller name' '-c@tmp.X' "$(keys tmp_caller)"
    _eq 'P17 inline code with a newline and a tab is one shim record of 4 fields, one key' '-c@multiline.sh shim+trace 1 0' \
        "$(keys multiline) $(srcs multiline) $(wc -l < "$td/o_multiline/shim.log" | tr -d ' ') $(awk -F'\t' 'NF != 4' "$td/o_multiline/shim.log" | wc -l | tr -d ' ')"
    _eq 'P11 the same script run twice is one entry point' '1' "$(wc -l < "$td/o_twice/keys" | tr -d ' ')"
    _eq 'P12 with no trace the absolute #! line is missed and says not_measured' '0 not_measured' "$(wc -l < "$td/o_abs_noshim_trace/keys" | tr -d ' ') $(meta "$td/o_abs_noshim_trace" trace)"
    # A descendant that daemonized must not hold the run open, nor leave the tracer behind.
    _eq 'D1 a daemonized descendant does not hang --run (rc, key)' '0 script:sub/tool.py' "$(meta "$td/o_daemon" rc) $(keys daemon)"
    _eq 'D2 the tracer is detached, not left running' 'yes 0' "$(meta "$td/o_daemon" detached) $(tracer_pids "$td/o_daemon/trace.raw" | wc -l | tr -d ' ')"
    timeout 5 bash -c 'printf x > "$1"' _ "$td/fifo" || :   # release the daemon
    # --compare: never worse, head vs base; a shim-only side is NOT_MEASURED.
    rc=0; compare "$td/o_none" "$td/o_env_call" > "$td/c1" 2>&1 || rc=$?
    _eq 'C1 head adds an entry point -> RED (1), and names it' '1 yes' "$rc $(grep -q -e '^  script:sub/tool.py$' "$td/c1" && echo yes || echo no)"
    rc=0; compare "$td/o_env_call" "$td/o_none" > /dev/null 2>&1 || rc=$?
    _eq 'C2 head removes one -> GREEN (0)' '0' "$rc"
    rc=0; compare "$td/o_env_call" "$td/o_var_call" > /dev/null 2>&1 || rc=$?
    _eq 'C3 head = base -> GREEN (0)' '0' "$rc"
    rc=0; compare "$td/o_none_notrace" "$td/o_env_call" > /dev/null 2>&1 || rc=$?
    _eq 'C4 a side with no trace -> NOT_MEASURED (2), never a pass' '2' "$rc"
    rc=0
    ( cd "$td/w" && PATH="$td/bin:$PATH" bash "$SELF" --run "$td/o_rc" -- bash -c 'exit 7' ) > /dev/null 2>&1 || rc=$?
    _eq 'R1 --run exits with the command status' '7' "$rc"
    _eq 'P18 joined -W/-X values do not read as -c/-m; bash -ec is an inline-shell caller' '-c@bash -c -c@flag_forms.sh script:sub/tool.py' "$(keys flag_forms)"
    _eq 'S1 two runs of the same lanes give the same keys (fresh mktemp names each run)' 'yes' "$( [ "$(keys tmp_script) $(keys tmp_caller)" = "$(keys tmp_script2) $(keys tmp_caller2)" ] && echo yes || echo no)"
    # A pid the kernel reused is a new process: it must not inherit the old one's cwd or caller.
    mkdir -p "$td/o_reuse"
    printf 'trace=measured\ncwd=%s\nrc=0\n' "$td/w" > "$td/o_reuse/meta"
    : > "$td/o_reuse/shim.log"
    { printf '100 execve("/bin/bash", ["bash", "lane1.sh"], 0x7ffd /* 5 vars */) = 0\n'
      printf '100 clone(child_stack=NULL, flags=SIGCHLD) = 200\n'
      printf '200 chdir("/elsewhere") = 0\n'
      printf '200 +++ exited with 0 +++\n'
      printf '100 clone(child_stack=NULL, flags=SIGCHLD) = 300\n'
      printf '300 execve("/bin/bash", ["bash", "lane2.sh"], 0x7ffd /* 5 vars */) = 0\n'
      printf '300 clone(child_stack=NULL, flags=SIGCHLD) = 200\n'
      printf '200 execve("/usr/bin/python3", ["python3", "tool.py"], 0x7ffd /* 5 vars */) = 0\n'
    } > "$td/o_reuse/trace.raw"
    count "$td/o_reuse" > /dev/null 2>&1 || :
    _eq 'T1 a reused pid keys by its own cwd and caller, not the dead process'"'"'s' 'script:tool.py lane2.sh' "$(cut -f1,3 "$td/o_reuse/keys" 2>/dev/null | tr '\t' ' ')"
    # Never a count from a broken measurement: each is exit 2, not GREEN 0.
    cp -r "$td/o_none" "$td/o_e"
    mkdir -p "$td/broken/lib"; cp "$SELF" "$td/broken/runcount.sh"; cp -r "$LIB" "$td/broken/lib/"
    printf '}\n' >> "$td/broken/lib/python_runcount/keys.awk"
    rc=$(rc_of env PYRUN_ROOT="$td/w" bash "$td/broken/runcount.sh" --compare "$td/o_env_call" "$td/o_e")
    _eq 'E1 a failing key stage is NOT_MEASURED (2), never count 0' '2' "$rc"
    : > "$td/o_e/trace.raw"
    _eq 'E2 trace=measured with no execve in the trace is NOT_MEASURED (2)' '2' "$(rc_of compare "$td/o_env_call" "$td/o_e")"
    rm -rf "${td:?}/o_e"; cp -r "$td/o_none" "$td/o_e"
    sed -i 's/^rc=0$/rc=3/' "$td/o_e/meta"
    _eq 'E3 a head lane that exited 3 where base exited 0 is NOT_MEASURED (2)' '2' "$(rc_of compare "$td/o_env_call" "$td/o_e")"
    cp -r "$td/o_env_call" "$td/o_e_base"; sed -i 's/^rc=0$/rc=3/' "$td/o_e_base/meta"
    _eq 'E4 head and base ending with the same status are measured (GREEN 0)' '0' "$(rc_of compare "$td/o_e_base" "$td/o_e")"
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
M14 temp paths keyed raw|lib/python_runcount/keys.awk|s/ p = "~tmp\/" stable(base(p))$/ p = p/
M15 mktemp names kept|lib/python_runcount/keys.awk|s/^function stable(s) { gsub(.*, "tmp.X", s); return s }/function stable(s) { return s }/
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
