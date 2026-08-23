#!/usr/bin/env bash
# Every python helper on a release-gate path must run under the MINIMUM Python
# the fleet provides -- not merely under the one on the box where it was written.
#
# aprender#2635. scripts/lib/facade_pin.py was written and verified on a dev box
# running python 3.13 and used `tomllib`, which entered the stdlib in 3.11. The
# CI runner host runs 3.10.12. On the runner every invocation died with
# ModuleNotFoundError, the helper printed NOTHING, facade_edges() produced no
# ordering edges, and R3 -- the rule that stops a facade being published before
# its upstream -- had nothing left to check. The cascade-coverage case table
# caught it because it asserts a REJECTION; the live scan next to it would have
# reported "ok R3 nothing to order" and gone quietly inert.
#
# The class is not "tomllib is 3.11+". It is a release gate carrying an
# INTERPRETER ASSUMPTION that was never checked against the machine the gate runs
# on -- the same family as the apr-binary rule already in this programme: the
# thing that ran is not the thing that was reasoned about.
#
# WHAT THIS GATE CAN AND CANNOT KNOW, stated plainly because a gate that implies
# coverage it does not have is the same defect one level up:
#
#   RULE A (static, host-independent, exact everywhere).  Every import in every
#   helper is extracted with `ast` -- never by importing the file -- and checked
#   against a table of when each module entered the stdlib. This runs identically
#   on a 3.13 dev box and on a 3.10 runner, so it does not depend on the very
#   thing it is auditing. It is what makes this gate meaningful when it happens
#   to execute on a NEWER interpreter than the floor.
#
#   RULE B (dynamic, exact only for the interpreter actually present).  Every
#   helper is byte-compiled and every import resolved with importlib.find_spec
#   under `python3` AS THIS HOST RESOLVES IT. Where the gate runs on the runner
#   this is exact coverage of the real target. Where it runs on a newer box it
#   proves less, and the gate SAYS SO in its output rather than implying the
#   floor was tested.
#
# The universe is derived, not listed. It is scripts/lib/*.py -- the directory
# that exists only to serve gate scripts -- plus every `scripts/**.py` path
# literal named by any shell script that a workflow names. Path LITERALS, not
# `python3 <path>` invocations: a helper reached through a shell variable
# (`P="$REPO_ROOT/scripts/lib/x.py"; python3 "$P"`) drops out of an
# invocation-shaped scan, and a universe assembled from the wrong side is the
# recurring way a guard ends up iterating fewer items than it claims.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The minimum Python the fleet provides. MEASURED, not assumed: the self-hosted
# runner host that executes the `run: bash scripts/...` steps in ci.yml is
# Ubuntu 22.04, whose /usr/bin/python3 is 3.10.12. Those steps run on the HOST,
# not in the sovereign-ci container (which has no python3 at all -- irrelevant
# here, and not something to "fix"). Raise this only when every host is raised.
PY_FLOOR_MAJOR=3
PY_FLOOR_MINOR=10

# module<TAB>major<TAB>minor -- the release the module entered the stdlib.
# Only entries ABOVE the floor can ever fire; the rest document the boundary.
# aprender#2635: `tomllib` is the row that was missing when it mattered.
# A function rather than a multi-line string: bashrs cannot parse the latter
# (SC1078), and a guard whose own linter cannot read it is not maintainable.
stdlib_since() {
    printf 'tomllib\t3\t11\n'
    printf 'dbm.sqlite3\t3\t13\n'
    printf 'graphlib\t3\t9\n'
    printf 'zoneinfo\t3\t9\n'
}

floor_str() { printf '%s.%s' "$PY_FLOOR_MAJOR" "$PY_FLOOR_MINOR"; }

# --------------------------------------------------------------------------
# Universe.
gate_shell_scripts() {  # root -- scripts a workflow actually names
    local root="$1"
    if [ -d "$root/.github/workflows" ]; then
        grep -rhoE 'scripts/[A-Za-z0-9_./-]+\.sh' "$root/.github/workflows" 2>/dev/null | sort -u
    fi
}

py_universe() {  # root
    local root="$1" rel
    {
        # 1. the shared gate-helper directory, in full.
        if [ -d "$root/scripts/lib" ]; then
            find "$root/scripts/lib" -maxdepth 1 -name '*.py' -printf 'scripts/lib/%f\n' 2>/dev/null
        fi
        # 2. every scripts/**.py named by a shell script a workflow names.
        while IFS= read -r rel; do
            [ -n "$rel" ] || continue
            [ -f "$root/$rel" ] || continue
            grep -ohE 'scripts/[A-Za-z0-9_./-]+\.py' "$root/$rel" 2>/dev/null
        done < <(gate_shell_scripts "$root")
    } | sort -u | while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        [ -f "$root/$rel" ] && printf '%s\n' "$rel"
    done
}

# --------------------------------------------------------------------------
# RULE A: static, host-independent.
rule_a() {  # root  -> 0 ok, 1 violation
    local root="$1" rc=0 rel line mod kind since_major since_minor row found
    local marked_shown=0
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        local imports
        imports="$(python3 "$root/scripts/lib/py_imports.py" "$root/$rel" 2>&1)"
        if [ $? -ne 0 ]; then
            printf 'FAIL  A %s: could not be parsed\n%s\n' "$rel" "$imports"
            rc=1
            continue
        fi
        while IFS=$'\t' read -r line mod kind; do
            [ -n "$mod" ] || continue
            found=""
            while IFS=$'\t' read -r row since_major since_minor; do
                [ -n "$row" ] || continue
                [ "$row" = "$mod" ] || continue
                found=1
                if [ "$since_major" -gt "$PY_FLOOR_MAJOR" ] \
                   || { [ "$since_major" -eq "$PY_FLOOR_MAJOR" ] \
                        && [ "$since_minor" -gt "$PY_FLOOR_MINOR" ]; }; then
                    if [ "$kind" = "marked" ]; then
                        printf 'note  A %s:%s imports %s (stdlib %s.%s) -- EXCUSED by an\n' \
                            "$rel" "$line" "$mod" "$since_major" "$since_minor"
                        printf '        explicit `# min-python-ok` marker. The import must be\n'
                        printf '        guarded; it is named here so the excuse is never silent.\n'
                        marked_shown=$((marked_shown + 1))
                    else
                        printf 'FAIL  A %s:%s imports %s, which entered the stdlib in %s.%s.\n' \
                            "$rel" "$line" "$mod" "$since_major" "$since_minor"
                        printf '        The fleet floor is python %s. On a floor host this helper\n' "$(floor_str)"
                        printf '        dies at import, prints nothing, and the gate that reads it\n'
                        printf '        goes inert instead of red.\n'
                        rc=1
                    fi
                fi
            done < <(stdlib_since)
            : "$found"
        done <<< "$imports"
    done < <(py_universe "$root")
    return "$rc"
}

# --------------------------------------------------------------------------
# RULE B: dynamic, under python3 as THIS host resolves it.
rule_b() {  # root  -> 0 ok, 1 violation
    local root="$1" rc=0 rel out
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        out="$(python3 "$root/scripts/lib/py_can_run.py" "$root/$rel" 2>&1)"
        if [ -n "$out" ]; then
            printf 'FAIL  B %s under python3 %s on this host:\n' \
                "$rel" "$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
            printf '%s\n' "$out" | sed 's/^/        /'
            rc=1
        fi
    done < <(py_universe "$root")
    return "$rc"
}

# --------------------------------------------------------------------------
# Self-test.
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
    trap 'rm -rf "${TD:?}"' EXIT

    mkfix() {  # name -- build a miniature repo at $TD/$name
        local r="$TD/$1"
        mkdir -p "$r/.github/workflows" "$r/scripts/lib"
        printf 'jobs:\n  g:\n    steps:\n      - run: bash scripts/check_thing.sh\n' \
            > "$r/.github/workflows/ci.yml"
        printf '#!/usr/bin/env bash\npython3 "$REPO_ROOT/scripts/lib/helper.py" read\n' \
            > "$r/scripts/check_thing.sh"
        cp "$REPO_ROOT/scripts/lib/py_imports.py" "$r/scripts/lib/py_imports.py"
        printf '%s' "$r"
    }

    row() {  # label want_rc root needle
        local label="$1" want="$2" root="$3" needle="$4" out rc
        out="$(rule_a "$root" 2>&1)"; rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL  %s: exit %s, expected %s\n%s\n' "$label" "$rc" "$want" "$out"
            fails=1; return
        fi
        if [ -n "$needle" ] && ! grep -q -- "$needle" <<< "$out"; then
            printf 'FAIL  %s: exit %s as expected but did not name %s\n%s\n' \
                "$label" "$rc" "$needle" "$out"
            fails=1; return
        fi
        printf 'ok    %s\n' "$label"
    }

    # Row 1 CONTROL. Without a passing case every row below is satisfied by a
    # checker that fails unconditionally.
    R="$(mkfix control)"
    printf 'import os\nimport re\nimport sys\n' > "$R/scripts/lib/helper.py"
    row 'row 1 a helper importing only ancient stdlib passes' 0 "$R" ''

    # Row 2 THE DEFECT, reproduced: the exact #2635 import.
    R="$(mkfix tomllib)"
    printf 'import sys\nimport tomllib\n' > "$R/scripts/lib/helper.py"
    row 'row 2 `import tomllib` in a gate helper is REJECTED' 1 "$R" 'tomllib'

    # Row 3 the other spelling of the same import.
    R="$(mkfix fromform)"
    printf 'from tomllib import load\n' > "$R/scripts/lib/helper.py"
    row 'row 3 `from tomllib import ...` is REJECTED too' 1 "$R" 'tomllib'

    # Row 4 a nested import is still an import. The first draft of the extractor
    # looked only at module level, which would have missed exactly this.
    R="$(mkfix nested)"
    printf 'def f():\n    import tomllib\n    return tomllib\n' > "$R/scripts/lib/helper.py"
    row 'row 4 an import nested inside a function is REJECTED' 1 "$R" 'tomllib'

    # Row 5 the declared-intent marker excuses the line it is on...
    R="$(mkfix marked)"
    printf 'try:\n    import tomllib  # min-python-ok\nexcept ImportError:\n    tomllib = None\n' \
        > "$R/scripts/lib/helper.py"
    row 'row 5 a guarded import carrying the marker is allowed' 0 "$R" 'EXCUSED'

    # Row 6 ...and ONLY the line it is on. An escape hatch that leaks to the
    # rest of the file is a worse defect than the one it excuses.
    R="$(mkfix marked_leak)"
    printf 'import tomllib  # min-python-ok\nimport dbm.sqlite3\n' > "$R/scripts/lib/helper.py"
    row 'row 6 the marker does NOT excuse a different unmarked import' 1 "$R" 'dbm.sqlite3'

    # Row 7 THE UNIVERSE. A helper reached only through a shell VARIABLE must
    # still be scanned -- an invocation-shaped scan would drop it, and the gate
    # would pass while never looking at the file carrying the defect.
    R="$(mkfix viavar)"
    printf 'import os\n' > "$R/scripts/lib/helper.py"
    printf '#!/usr/bin/env bash\nP="$REPO_ROOT/scripts/other.py"\npython3 "$P"\n' \
        > "$R/scripts/check_thing.sh"
    printf 'import tomllib\n' > "$R/scripts/other.py"
    row 'row 7 a helper reached via a shell variable is still in the universe' 1 "$R" 'scripts/other.py'

    # Row 8 scope. A .py that no gate-wired script names is NOT this gate's
    # business -- research scripts legitimately import torch. A gate that fails
    # on files it does not govern gets disabled, which is how coverage dies.
    R="$(mkfix outofscope)"
    printf 'import os\n' > "$R/scripts/lib/helper.py"
    printf 'import tomllib\n' > "$R/scripts/research_only.py"
    row 'row 8 a .py no gate-wired script names is out of scope' 0 "$R" ''

    # Row 9 VACUITY in the universe. A gate over zero files reports success.
    R="$(mkfix empty)"
    rm -f "$R/scripts/lib/helper.py" "$R/scripts/lib/py_imports.py" "$R/scripts/check_thing.sh"
    _u="$(py_universe "$R")"
    if [ -z "$_u" ]; then
        printf 'ok    row 9 an empty universe is detectable (main scan rejects it)\n'
    else
        printf 'FAIL  row 9 py_universe invented %s files for an empty tree\n' \
            "$(printf '%s\n' "$_u" | wc -l)"
        fails=1
    fi

    # Row 10 the universe over the REAL tree is non-empty and contains the file
    # this whole ticket is about. A case table that only ever runs on fixtures
    # cannot notice that the real scan stopped finding anything.
    _u="$(py_universe "$REPO_ROOT")"
    if grep -qx 'scripts/lib/facade_pin.py' <<< "$_u"; then
        printf 'ok    row 10 the real universe contains scripts/lib/facade_pin.py (%s files)\n' \
            "$(printf '%s\n' "$_u" | wc -l | tr -d ' ')"
    else
        printf 'FAIL  row 10 the real universe does NOT contain scripts/lib/facade_pin.py\n'
        fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (10/10)\n'
    exit 0
fi

# --------------------------------------------------------------------------
printf '=== python helpers on release-gate paths must run on the fleet floor ===\n\n'

HOST_PY="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
if [ -z "$HOST_PY" ]; then
    printf 'FAIL  no python3 on PATH -- every helper below is unrunnable here\n'
    exit 1
fi
printf 'declared fleet floor : python %s\n' "$(floor_str)"
printf 'this host`s python3  : python %s (%s)\n' "$HOST_PY" "$(command -v python3)"

UNIVERSE="$(py_universe "$REPO_ROOT")"
if [ -z "$UNIVERSE" ]; then
    printf '\nFAIL  the helper universe is EMPTY -- the ENUMERATION is broken, and a\n'
    printf '      gate over zero files reports success on a broken tree.\n'
    exit 1
fi
printf 'helpers in scope     : %s\n\n' "$(printf '%s\n' "$UNIVERSE" | wc -l | tr -d ' ')"

RC=0
if rule_a "$REPO_ROOT"; then
    printf 'ok    A no helper imports a stdlib module newer than python %s\n' "$(floor_str)"
else
    RC=1
fi

if rule_b "$REPO_ROOT"; then
    printf 'ok    B every helper compiles and resolves its imports under python %s\n' "$HOST_PY"
else
    RC=1
fi

# Say exactly what was and was not observed. The floor is what matters; this
# host is only sometimes at it.
HOST_MINOR="${HOST_PY#*.}"
printf '\ncoverage: '
if [ "$HOST_MINOR" -lt "$PY_FLOOR_MINOR" ]; then
    printf 'this host (%s) is BELOW the declared floor (%s).\n' "$HOST_PY" "$(floor_str)"
    printf '          The floor declaration is wrong for a machine that runs these gates.\n'
    RC=1
elif [ "$HOST_MINOR" -eq "$PY_FLOOR_MINOR" ]; then
    printf 'rule B ran ON the floor interpreter (python %s), so the\n' "$HOST_PY"
    printf '          dynamic half is EXACT coverage of the real target.\n'
else
    printf 'rule B ran on python %s, ABOVE the floor (%s). It therefore\n' \
        "$HOST_PY" "$(floor_str)"
    printf '          proves these helpers run HERE, NOT that they run on the floor.\n'
    printf '          Only rule A -- which is host-independent -- covers the floor from\n'
    printf '          this host. This is stated rather than implied on purpose: the\n'
    printf '          defect being guarded was exactly a floor claim made from a\n'
    printf '          newer interpreter.\n'
fi

exit "$RC"
