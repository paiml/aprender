#!/usr/bin/env bash
# ci_measured.sh - run a CI command and collect its MEASUREMENT ARTIFACTS,
# whatever that command's outcome (ARBITER-001 row R2a, PMAT-532, aprender#3133).
#
#   bash scripts/ci_measured.sh <name> <cmd> [args...]
#   bash scripts/ci_measured.sh --self-test
#
# WHY THIS EXISTS
# ---------------
# Measured 2026-09-12 by `gh api`: the newest completed `main` push run of `CI`
# in EVERY roster repo carries ZERO artifacts. aprender already WRITES a junit
# file - `.config/nextest.toml` `[profile.ci.junit] path = "junit.xml"` puts it
# at <target>/nextest/ci/junit.xml - but nothing uploads it, and `workspace-test`
# runs nextest more than once per job, so the second run OVERWRITES the first
# one's junit before anybody could. sccache's stats live in the sccache SERVER,
# which lives inside each `docker run --rm` and dies with it, so the capture has
# to happen in the SAME invocation as cargo, not in a host-side step afterwards.
#
# So: one wrapper, called at each test step. It runs the command it is given,
# and then - whatever happened - MOVES the junit to a per-step name and writes
# the sccache stats beside it under <target>/ci-artifacts/, which the host-side
# `upload-artifact` step reads.
#
# WHY MOVE THE JUNIT AND NOT COPY IT
# ----------------------------------
# A copy leaves <target>/nextest/ci/junit.xml in place, so the NEXT step in the
# same job starts with a stale file it did not write. If that step's nextest
# never gets to write its own (a compile error, a signal, a timeout), the
# collector would pick the previous step's results up and upload them under the
# new step's name: a measurement attributed to a run that never produced it,
# which is this fleet's signature defect. Moving makes that impossible - the
# absence of a junit is then reported as `junit=none`.
#
# WHAT IT CANNOT DO, STATED RATHER THAN HIDDEN
# --------------------------------------------
# It cannot save a junit nextest never wrote. If the wrapped command dies before
# nextest finishes (signal, timeout, compile failure), there is no junit and the
# summary says `junit=none`. The sccache stats are still collected, because the
# sccache server answers whether or not the build finished.
#
# EXIT CODE
# ---------
# The child's exit code, unchanged, AFTER the collectors have run - or 128+N
# when the child died of signal N. There is no `set -e` here and the collectors
# never overwrite the code: a wrapper that turned a red step green, or a green
# step red, would be worse than no measurement at all.
#
# SIGNALS
# -------
# TERM, INT and HUP are trapped and FORWARDED to the child, then the wrapper
# waits for the child and runs the collectors anyway. GitHub cancels a job with
# a signal, and a cancelled job is exactly when the sccache stats are most worth
# having. Self-test case 6 kills a wrapped `sleep 30` after one second and
# asserts the sccache file exists and the wrapper exits 143 in about a second.
set -u -o pipefail

ART_SUBDIR="ci-artifacts"
CHILD=0

usage() {
    printf 'usage: %s <name> <cmd> [args...]\n' "$0" >&2
    printf '       %s --self-test\n' "$0" >&2
    printf 'name must match ^[A-Za-z0-9._-]+$ and is used in the artifact file names\n' >&2
}

target_dir() { # where cargo writes, and where ci-artifacts/ is created
    printf '%s' "${CI_MEASURED_TARGET_DIR:-${CARGO_TARGET_DIR:-target}}"
}

forward() { # send the signal we caught on to the child, never to ourselves
    if [ "$CHILD" -ne 0 ]; then
        kill -s "$1" "$CHILD" 2>/dev/null || true
    fi
}

collect() { # $1 name, $2 art dir, $3 junit source -> prints the summary fields
    local name=$1 art=$2 junit_src=$3 junit_dst=none sccache_dst=none part
    if [ -f "$junit_src" ]; then
        if mv -f "$junit_src" "$art/junit-$name.xml"; then
            junit_dst="$art/junit-$name.xml"
        else
            printf 'ci_measured: could not move %s; no junit artifact for %s\n' "$junit_src" "$name"
        fi
    fi
    if command -v sccache >/dev/null 2>&1; then
        part="$art/.sccache-$name.json.part"
        if sccache --show-stats --stats-format json > "$part" 2>/dev/null; then
            mv -f "$part" "$art/sccache-$name.json"
            sccache_dst="$art/sccache-$name.json"
        else
            rm -f "$part"
            printf 'ci_measured: sccache --show-stats failed; no sccache artifact for %s, exit code unchanged\n' "$name"
        fi
    fi
    printf 'ci_measured: name=%s rc=%s junit=%s sccache=%s\n' "$name" "$RC" "$junit_dst" "$sccache_dst"
}

run_measured() { # $1 name, rest: the command, passed through UNPARSED
    local name=$1 td art junit_src
    shift
    td=$(target_dir)
    art="$td/$ART_SUBDIR"
    junit_src="$td/nextest/ci/junit.xml"
    mkdir -p "$art" || return 2

    trap 'forward TERM' TERM
    trap 'forward INT' INT
    trap 'forward HUP' HUP

    # The child runs in the BACKGROUND so that a signal arriving at this shell
    # interrupts `wait` and runs the trap. A foreground child would make this
    # shell die of the signal with the collectors never reached.
    "$@" &
    CHILD=$!
    RC=0
    while : ; do
        wait "$CHILD"
        RC=$?
        # A status above 128 from `wait` means either the child died of a signal
        # or a trapped signal interrupted the wait. Only the second leaves the
        # child alive, and only then is there anything left to wait for.
        if [ "$RC" -gt 128 ] && kill -0 "$CHILD" 2>/dev/null; then continue; fi
        break
    done
    trap - TERM INT HUP

    collect "$name" "$art" "$junit_src"
    return "$RC"
}

self_test() {
    local td n=0 red=0 rc out T
    T=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-measured.XXXXXX") || return 1
    trap 'rm -rf "${td:?}"' RETURN

    # A fake sccache on a PRIVATE path, so the case table measures the wrapper
    # and not whether this box happens to have a build cache daemon.
    mkdir -p "$td/good" "$td/bad"
    cat > "$td/good/sccache" <<'SCCACHE_OK'
#!/usr/bin/env bash
printf '{"stats":{"compile_requests":1}}\n'
SCCACHE_OK
    cat > "$td/bad/sccache" <<'SCCACHE_BAD'
#!/usr/bin/env bash
printf 'sccache: error: failed to connect to server\n' >&2
exit 1
SCCACHE_BAD
    chmod +x "$td/good/sccache" "$td/bad/sccache"

    sanitized_path() { # PATH with every directory that holds an sccache removed
        local d out=""
        while IFS= read -r d; do
            if [ -z "$d" ]; then continue; fi
            if [ -x "$d/sccache" ]; then continue; fi
            out="${out:+$out:}$d"
        done < <(printf '%s' "$PATH" | tr ':' '\n')
        printf '%s' "$out"
    }
    NO_SCCACHE_PATH=$(sanitized_path)

    mkcase() { # $1 a fresh fake target dir, $2 = junit to seed nextest's output
        rm -rf "${1:?}"
        mkdir -p "$1/nextest/ci"
        if [ "${2:-}" = junit ]; then printf '<testsuites/>\n' > "$1/nextest/ci/junit.xml"; fi
    }
    # One invocation, many assertions: capture the run once, replay its stdout
    # and its exit code for every row that asks a question about it.
    cap() { local name=$1 r=0; shift; "$@" > "$td/$name.out" 2>&1 || r=$?; printf '%s' "$r" > "$td/$name.rc"; }
    replay() { printf 'cat "%s/%s.out"; exit "$(cat "%s/%s.rc")"' "$td" "$1" "$td" "$1"; }
    row() { local want=$1 label=$2 pat=$3; shift 3; n=$((n + 1)); rc=0; out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then
            printf 'ok    case %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else
            printf 'FAIL  case %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
            printf '%s\n' "$out" | sed 's/^/        /'
            red=$((red + 1))
        fi
    }
    exists() { if [ -e "$1" ]; then printf 'PRESENT %s\n' "$1"; else printf 'ABSENT %s\n' "$1"; return 1; fi; }
    absent() { if [ -e "$1" ]; then printf 'PRESENT %s\n' "$1"; return 1; else printf 'ABSENT %s\n' "$1"; fi; }

    # --- case 1: the happy path. Both artifacts, exit 0, and the junit MOVED.
    mkcase "$td/c1" junit
    cap c1 env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c1" bash "$T" c1 true
    row 0 "rc 0 with a junit present: exit 0 and the summary names both artifacts" \
        'name=c1 rc=0 junit=.*/ci-artifacts/junit-c1\.xml sccache=.*/ci-artifacts/sccache-c1\.json' bash -c "$(replay c1)"
    row 0 "  ...the junit was MOVED, so nextest's own path is now empty" '^ABSENT' absent "$td/c1/nextest/ci/junit.xml"
    row 0 "  ...and the per-step copy exists" '^PRESENT' exists "$td/c1/ci-artifacts/junit-c1.xml"
    row 0 "  ...and the sccache file holds the server's JSON, not an empty file" 'compile_requests' cat "$td/c1/ci-artifacts/sccache-c1.json"

    # --- case 2: a FAILING child still gets measured, and keeps its code.
    mkcase "$td/c2" junit
    cap c2 env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c2" bash "$T" c2 bash -c 'exit 7'
    row 7 "child exits 7: the collectors still run and rc 7 is returned unchanged" 'name=c2 rc=7' bash -c "$(replay c2)"
    row 0 "  ...junit collected despite the failure" '^PRESENT' exists "$td/c2/ci-artifacts/junit-c2.xml"
    row 0 "  ...sccache collected despite the failure" '^PRESENT' exists "$td/c2/ci-artifacts/sccache-c2.json"

    # --- case 3: no junit is reported as none, never invented.
    mkcase "$td/c3"
    cap c3 env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c3" bash "$T" c3 true
    row 0 "no junit produced: junit=none and exit 0" 'name=c3 rc=0 junit=none' bash -c "$(replay c3)"
    row 0 "  ...and no junit file was written" '^ABSENT' absent "$td/c3/ci-artifacts/junit-c3.xml"

    # --- case 4: no sccache on PATH is not an error and not a file.
    mkcase "$td/c4" junit
    cap c4 env PATH="$NO_SCCACHE_PATH" CI_MEASURED_TARGET_DIR="$td/c4" bash "$T" c4 bash -c 'exit 5'
    row 5 "sccache absent from PATH: sccache=none and the child's rc 5 survives" 'name=c4 rc=5 .*sccache=none' bash -c "$(replay c4)"
    row 0 "  ...no sccache file was invented" '^ABSENT' absent "$td/c4/ci-artifacts/sccache-c4.json"
    row 0 "  ...and the junit was collected anyway" '^PRESENT' exists "$td/c4/ci-artifacts/junit-c4.xml"

    # --- case 5: the command is passed through UNPARSED. `false && true` is a
    # bash -c STRING; a wrapper that re-split or re-quoted it would not exit 1.
    mkcase "$td/c5"
    cap c5 env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c5" bash "$T" c5 bash -c 'false && true'
    row 1 "a bash -c string is run as given: 'false && true' exits 1 and the 1 survives" 'name=c5 rc=1' bash -c "$(replay c5)"

    # --- case 5b: an argument holding a space stays ONE argument.
    mkcase "$td/c5b"
    cap c5b env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c5b" bash "$T" c5b printf '[%s]' 'a b'
    row 0 "an argument with a space arrives as one argument, not two" '^\[a b\]' bash -c "$(replay c5b)"

    # --- case 6: a cancelled job. TERM to the WRAPPER must reach the child,
    # the collectors must still run, and the wrapper must not sit out the sleep.
    term_case() {
        local start end pid r=0
        mkcase "$td/c6"
        start=$SECONDS
        env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c6" bash "$T" c6 sleep 30 > "$td/c6-wrapper.out" 2>&1 &
        pid=$!
        sleep 1
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" || r=$?
        end=$SECONDS
        printf 'rc=%s elapsed=%s\n' "$r" "$((end - start))"
        cat "$td/c6-wrapper.out"
    }
    cap c6 term_case
    row 0 "a wrapped sleep 30 TERMed after 1s: exit 143 within 3s, not 30" '^rc=143 elapsed=[0-3]$' bash -c "$(replay c6)"
    row 0 "  ...and the summary was still printed, naming rc=143" 'name=c6 rc=143' bash -c "$(replay c6)"
    row 0 "  ...and the sccache stats were collected after the signal" '^PRESENT' exists "$td/c6/ci-artifacts/sccache-c6.json"
    row 0 "  ...while junit is honestly none: nextest never ran, so there is nothing to save" 'junit=none' bash -c "$(replay c6)"

    # --- case 7: a bad name is refused BEFORE anything is run or created.
    mkcase "$td/c7"
    cap c7 env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c7" bash "$T" a/b true
    row 2 "a name outside ^[A-Za-z0-9._-]+\$ exits 2 with usage" 'usage:' bash -c "$(replay c7)"
    row 0 "  ...and nothing was created: no ci-artifacts directory" '^ABSENT' absent "$td/c7/ci-artifacts"
    mkcase "$td/c7b"
    cap c7b env PATH="$td/good:$PATH" CI_MEASURED_TARGET_DIR="$td/c7b" bash "$T" c7b
    row 2 "a name with no command exits 2 with usage" 'usage:' bash -c "$(replay c7b)"

    # --- case 8: a BROKEN sccache writes nothing and changes no exit code.
    mkcase "$td/c8" junit
    cap c8 env PATH="$td/bad:$PATH" CI_MEASURED_TARGET_DIR="$td/c8" bash "$T" c8 bash -c 'exit 3'
    row 3 "sccache --show-stats fails: sccache=none and the child's rc 3 survives" 'name=c8 rc=3 .*sccache=none' bash -c "$(replay c8)"
    row 3 "  ...and it says so on stdout rather than failing silently" 'sccache --show-stats failed' bash -c "$(replay c8)"
    row 0 "  ...no sccache file, not even an empty one" '^ABSENT' absent "$td/c8/ci-artifacts/sccache-c8.json"
    row 0 "  ...and the junit was still collected" '^PRESENT' exists "$td/c8/ci-artifacts/junit-c8.xml"

    # --- case 9: CARGO_TARGET_DIR is the fallback, which is what the six call
    # sites in ci.yml actually set inside the container.
    mkcase "$td/c9" junit
    cap c9 env PATH="$td/good:$PATH" CARGO_TARGET_DIR="$td/c9" bash "$T" c9 true
    row 0 "CARGO_TARGET_DIR is honoured when CI_MEASURED_TARGET_DIR is unset" 'junit=.*/c9/ci-artifacts/junit-c9\.xml' bash -c "$(replay c9)"

    printf '\nci_measured --self-test: %s cases, %s failed\n' "$n" "$red"
    [ "$red" -eq 0 ]
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
if [ $# -lt 2 ]; then usage; exit 2; fi
NAME=$1
shift
case "$NAME" in
    *[!A-Za-z0-9._-]*) printf 'ci_measured: bad name "%s"\n' "$NAME" >&2; usage; exit 2 ;;
esac
run_measured "$NAME" "$@"
exit $?
