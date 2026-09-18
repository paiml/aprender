#!/usr/bin/env bash
# check_explicit_test_commands.sh -- the explicit integration-test list stays FRAGMENTS (PMAT-3313).
#
# WHY. workspace-test's "Integration tests" step was one physical ci.yml line:
# `bash -c '<39 commands joined by &&>'`, ~4000 characters. Every PR adding a test
# target edited that same line, so any two of them conflicted. A single
# one-command-per-line file only MOVED the lock (two end-of-file appends still
# conflict -- measured), so the commands are fragments, the docs/roadmaps/entries/
# shape: ci/explicit-test-commands.d/NNN-<slug>.cmd, one command per file, run in
# `LC_ALL=C sort` order by scripts/ci_run_explicit_test_commands.sh.
#
# WHAT THIS REFUSES (default mode, over ROOT, default `.`):
#   1. a missing or empty directory (vacuity);
#   2. a fragment that does not hold exactly one non-comment command line;
#   3. the same command in two fragments;
#   4. two fragments sharing an ordinal prefix (ambiguous order);
#   5. an entry not named ^[0-9]{3}-[a-z0-9-]+\.cmd$;
#      (1-5 are the runner's own --list parser: the CI step refuses the same trees)
#   6. no workflow EXECUTING the runner on the directory (a `#` comment is a
#      mention, not wiring) -- every fragment would be dark;
#   7. any workflow line chaining two cargo commands with `&&` -- the mega-line.
#
# --equivalence OLD_CI_YML [DIR]: the representation change added, dropped and
#   reordered nothing. Extracts the ONE `&&`-chained cargo line from OLD_CI_YML,
#   splits and trims it, and compares it to DIR's commands in sort order.
#   rc 0 equal, 1 differ, 2 OLD_CI_YML has no (or several) such lines / DIR refused.
#
# Usage:
#   bash scripts/check_explicit_test_commands.sh [--check ROOT]
#   bash scripts/check_explicit_test_commands.sh --equivalence OLD_CI_YML [DIR]
#   bash scripts/check_explicit_test_commands.sh --self-test   # case table (also the runner's)
set -euo pipefail

DIR_REL="ci/explicit-test-commands.d"
RUNNER_REL="scripts/ci_run_explicit_test_commands.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

# is_megaline LINE -- rc 0 when LINE (comment-stripped) chains two cargo commands with &&.
is_megaline() {
    local code="${1%%#*}"
    grep -qE '(^|[^A-Za-z0-9_-])cargo[[:space:]].*&&[[:space:]]*cargo[[:space:]]' <<< "$code"
}

# is_wiring LINE -- rc 0 when LINE (comment-stripped) runs the runner in --run mode on the directory.
is_wiring() {
    local code="${1%%#*}"
    case "$code" in
        *"ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d"*) return 0 ;;
        *) return 1 ;;
    esac
}

workflow_files() { # workflow_files ROOT -> one path per line
    local f
    for f in "$1"/.github/workflows/*.yml "$1"/.github/workflows/*.yaml; do
        [ -f "$f" ] && printf '%s\n' "$f"
    done
    return 0
}

check() {
    local root=$1 rc=0 n wf line lineno wired=0 megas=0 out
    local runner="$root/$RUNNER_REL"
    [ -f "$runner" ] || runner="$HERE/ci_run_explicit_test_commands.sh"
    if out=$(bash "$runner" --list "$root/$DIR_REL" 2>&1); then
        n=$(grep -c . <<< "$out")
        printf 'ok    %s: %s fragment(s), one command each, no shared ordinal, no duplicate command\n' "$DIR_REL" "$n"
    else
        printf 'FAIL  %s refused:\n%s\n' "$DIR_REL" "$(sed 's/^/        /' <<< "$out")"
        rc=1
    fi
    while IFS= read -r wf; do
        lineno=0
        while IFS= read -r line || [ -n "$line" ]; do
            lineno=$((lineno + 1))
            if is_wiring "$line"; then wired=$((wired + 1)); fi
            if is_megaline "$line"; then
                megas=$((megas + 1))
                printf 'FAIL  %s:%s chains cargo commands with && on one line:\n        %.160s...\n' "${wf#"$root"/}" "$lineno" "$(sed 's/^[[:space:]]*//' <<< "$line")"
                printf '      Put each command in its own %s/NNN-<slug>.cmd instead (PMAT-3313).\n' "$DIR_REL"
            fi
        done < "$wf"
    done < <(workflow_files "$root")
    if [ "$megas" -gt 0 ]; then rc=1; else printf 'ok    no workflow line chains cargo commands with &&\n'; fi
    if [ "$wired" -eq 0 ]; then
        printf 'FAIL  no workflow runs `%s --run %s` (outside a comment): every fragment is dark\n' "$RUNNER_REL" "$DIR_REL"
        rc=1
    else
        printf 'ok    %s workflow invocation(s) run the directory\n' "$wired"
    fi
    [ "$rc" -eq 0 ] && printf 'PASS  check_explicit_test_commands\n'
    return "$rc"
}

# old_commands OLD_CI_YML -> the ordered commands of its single mega-line; rc 2 if not exactly one.
old_commands() {
    local yml=$1 line hits=0 found="" body
    while IFS= read -r line || [ -n "$line" ]; do
        if is_megaline "$line"; then hits=$((hits + 1)); found=$line; fi
    done < "$yml"
    if [ "$hits" -ne 1 ]; then
        printf 'ENV   %s has %s &&-chained cargo line(s); equivalence needs exactly 1\n' "$yml" "$hits" >&2
        return 2
    fi
    body=$(sed -E "s/^[[:space:]]*(bash -c ')?//; s/'[[:space:]]*\$//" <<< "$found")
    sed 's/&&/\n/g' <<< "$body" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^$' || true
}

equivalence() {
    local old=$1 dir=$2 a b na nb
    a=$(old_commands "$old") || return 2
    b=$(bash "$HERE/ci_run_explicit_test_commands.sh" --list "$dir") || return 2
    na=$(grep -c . <<< "$a"); nb=$(grep -c . <<< "$b")
    if [ "$a" = "$b" ]; then
        printf 'PASS  equivalent: %s command(s) in the old line == %s fragment command(s) in %s (LC_ALL=C sort order), byte for byte\n' "$na" "$nb" "$dir"
        return 0
    fi
    printf 'FAIL  NOT equivalent: old line %s command(s), %s %s command(s) (<: old only, >: fragments only):\n' "$na" "$dir" "$nb"
    diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") | sed 's/^/        /' || true
    return 1
}

self_test() {
    local td n=0 red=0 R T out rc
    td=$(mktemp -d); trap 'rm -rf "${td:?}"' RETURN
    R="$HERE/ci_run_explicit_test_commands.sh"; T="$HERE/check_explicit_test_commands.sh"
    row() { # row WANT_RC LABEL MUST_MATCH -- CMD...
        local want=$1 label=$2 pat=$3; shift 3; n=$((n + 1))
        rc=0; out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && grep -qE -- "$pat" <<< "$out"; then
            printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
            sed 's/^/        /' <<< "$out"; red=1
        fi
    }
    fact() { local label=$1; shift; n=$((n + 1))
        if "$@"; then printf 'ok    row %-2s        %s\n' "$n" "$label"
        else printf 'FAIL  row %-2s        %s\n' "$n" "$label"; red=1; fi
    }
    notfact() { local label=$1; shift; n=$((n + 1))
        if "$@"; then printf 'FAIL  row %-2s        %s\n' "$n" "$label"; red=1
        else printf 'ok    row %-2s        %s\n' "$n" "$label"; fi
    }
    frags() { # frags DIR NAME=BODY... (BODY is printf %b)
        local d=$1 kv; shift; rm -rf "${d:?}"; mkdir -p "$d"
        for kv in "$@"; do printf '%b' "${kv#*=}" > "$d/${kv%%=*}"; done
    }

    # ── the runner: parsing ──────────────────────────────────────────────
    frags "$td/p" '020-b.cmd=  echo b  \n' '010-a.cmd=# why\n\n   true\n' '100-c.cmd=echo c'
    row 0 "--list reads fragments in LC_ALL=C sort order, skips # and blank lines, trims" '^true$' bash "$R" --list "$td/p"
    fact "  ...and yields exactly [true, echo b, echo c] (010 < 020 < 100; no trailing newline still counts)" test "$(bash "$R" --list "$td/p" | tr '\n' '|')" = "true|echo b|echo c|"
    # ── refusals: vacuity ────────────────────────────────────────────────
    row 2 "REFUSE missing directory (vacuity)" 'no such directory' bash "$R" --run "$td/absent"
    frags "$td/empty"
    row 2 "REFUSE empty directory (vacuity)" 'EMPTY' bash "$R" --run "$td/empty"
    # ── refusals: one command per fragment ───────────────────────────────
    frags "$td/zero" '010-a.cmd=true\n' '020-b.cmd=# only a comment\n\n'
    row 2 "REFUSE a fragment with ZERO command lines" '020-b.cmd holds 0 command' bash "$R" --list "$td/zero"
    frags "$td/two" '010-a.cmd=true\necho second\n'
    row 2 "REFUSE a fragment with TWO command lines" 'holds 2 command' bash "$R" --list "$td/two"
    # ── refusals: duplicates and ordinals ────────────────────────────────
    frags "$td/dup" '010-a.cmd=echo same\n' '020-b.cmd=  echo same\n'
    row 2 "REFUSE the same command in two fragments (after trimming)" 'same command is in 010-a.cmd and 020-b.cmd' bash "$R" --list "$td/dup"
    frags "$td/ord" '010-a.cmd=echo a\n' '010-b.cmd=echo b\n'
    row 2 "REFUSE two fragments sharing an ordinal" 'ordinal 010 is shared' bash "$R" --list "$td/ord"
    # ── sharding (PACK-001): every M-th command from the N-th; refusals, never a vacuous pass ──
    row 0 "--shard 1/2 takes the 1st and 3rd of three commands" 'shard 1/2: 2 of 3' bash "$R" --run "$td/p" --shard 1/2
    row 0 "--shard 2/2 takes the 2nd" 'shard 2/2: 1 of 3' bash "$R" --run "$td/p" --shard 2/2
    fact "  ...and 1/2 ran exactly [true, echo c]; 2/2 ran [echo b] (a command is on ONE shard)" \
        test "$(bash "$R" --run "$td/p" --shard 1/2 2>/dev/null | grep -cE '^\[|^(b|c)$')$(bash "$R" --run "$td/p" --shard 2/2 2>/dev/null | grep -c '^b$')" = "11"
    row 2 "REFUSE --shard with N > M" 'N exceeds M' bash "$R" --run "$td/p" --shard 3/2
    row 2 "REFUSE a shard that selects zero commands (more shards than commands)" 'selects 0 of 3' bash "$R" --run "$td/p" --shard 4/4
    row 2 "REFUSE a malformed --shard" 'must look like N/M' bash "$R" --run "$td/p" --shard x
    row 2 "REFUSE an unknown --run option" 'usage' bash "$R" --run "$td/p" --bogus
    # ── refusals: filename shape (must-match / must-not-match, rule 7) ───
    local bad
    for bad in '10-a.cmd' '0100-a.cmd' '010-A.cmd' '010-a_b.cmd' '010-.cmd' '010-a.txt' '010a.cmd' 'README.md' '.gitkeep' '010-a.cmd.orig'; do
        frags "$td/name" '020-ok.cmd=true\n' "$bad=echo x\n"
        row 2 "REFUSE filename <$bad>" 'not a regular file named' bash "$R" --list "$td/name"
    done
    frags "$td/name" '000-a.cmd=true\n' '999-z-9.cmd=echo z\n' '050-apr-cli-falsify-auth-001-002-003.cmd=echo m\n'
    row 0 "ACCEPT filenames 000-a / 050-apr-cli-falsify-auth-001-002-003 / 999-z-9" '^echo z$' bash "$R" --list "$td/name"
    mkdir -p "$td/name/010-sub.cmd"
    row 2 "REFUSE a directory named like a fragment" 'not a regular file' bash "$R" --list "$td/name"
    notfact "  ...and a refused tree prints NO commands on stdout" test -n "$(bash "$R" --list "$td/name" 2>/dev/null)"
    # ── the runner: execution ────────────────────────────────────────────
    frags "$td/ff" "010-a.cmd=touch $td/a\n" '020-b.cmd=exit 7\n' "030-c.cmd=touch $td/c\n"
    row 7 "FAIL-FAST: the first non-zero command's status is the step's status" 'exit 7' bash "$R" --run "$td/ff"
    fact "  ...the command before it ran" test -e "$td/a"
    notfact "  ...the command after it did NOT run" test -e "$td/c"
    frags "$td/in" "010-a.cmd=cat > $td/swallowed\n" "020-b.cmd=touch $td/after\n"
    row 0 "a command reading stdin gets /dev/null" '^PASS  2/2' bash "$R" --run "$td/in"
    fact "  ...and the next command still ran" test -e "$td/after"
    fact "  ...and it read nothing" test ! -s "$td/swallowed"
    frags "$td/ok" '010-a.cmd=echo one\n' '020-b.cmd=echo two\n'
    row 0 "a group header per command, numbered" '::group::\[2/2\] echo two' bash "$R" --run "$td/ok"
    frags "$td/bad-run" "010-a.cmd=touch $td/ran\n" '010-b.cmd=echo b\n'
    row 2 "a REFUSED tree runs nothing (--run)" 'ordinal 010' bash "$R" --run "$td/bad-run"
    notfact "  ...not even its first valid fragment" test -e "$td/ran"
    row 2 "no mode -> usage, rc 2" 'usage' bash "$R"

    # ── megaline matcher: must-match / must-not-match (rule 7) ───────────
    local want l
    while IFS='|' read -r want l; do
        [ -n "$want" ] || continue
        if [ "$want" = M ]; then fact "megaline MATCH:     $l" is_megaline "$l"
        else notfact "megaline NO-MATCH:  $l" is_megaline "$l"; fi
    done <<'TABLE'
M|            bash -c 'cargo test -p a --test x && cargo test -p b --test y'
M|        run: cargo build && cargo test
M|  cargo test -p a 2>&1 && cargo check --benches
M|  cargo test -p a &&cargo test -p b
N|            bash -c 'cargo test -p a --test x'
N|  # bash -c 'cargo test -p a && cargo test -p b'
N|  run: echo x # cargo test -p a && cargo test -p b
N|  which cargo-mutants || (echo installing && cargo install cargo-mutants)
N|  cargo test -p a && echo done
N|  bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d
TABLE
    fact "wiring MATCH: the runner on the directory" is_wiring "            bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d"
    notfact "wiring NO-MATCH: commented out" is_wiring "  # bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d"
    notfact "wiring NO-MATCH: --list is parsing, not running" is_wiring "  bash scripts/ci_run_explicit_test_commands.sh --list ci/explicit-test-commands.d"

    # ── check mode over fixture trees ────────────────────────────────────
    mk() { # mk ROOT WORKFLOW_BODY NAME=BODY...
        local r=$1 wf=$2; shift 2
        mkdir -p "$r/.github/workflows" "$r/scripts"; cp "$R" "$r/scripts/"
        printf '%b' "$wf" > "$r/.github/workflows/ci.yml"
        frags "$r/$DIR_REL" "$@"
    }
    local wire='jobs:\n  t:\n    steps:\n      - run: |\n          bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d\n'
    mk "$td/good" "$wire" '010-a-x.cmd=cargo test -p a --test x\n' '020-b-y.cmd=cargo test -p b --test y\n'
    row 0 "check: wired, valid fragments, no megaline -> PASS" '^PASS' bash "$T" --check "$td/good"
    mk "$td/mega" "$wire      - run: bash -c 'cargo test -p a --test x && cargo test -p b --test y'\n" '010-a-x.cmd=cargo test -p a --test x\n'
    row 1 "check: a megaline reintroduced in a workflow -> RED" 'chains cargo commands' bash "$T" --check "$td/mega"
    mk "$td/dark" 'jobs:\n  t:\n    steps:\n      # bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d\n      - run: true\n' '010-a-x.cmd=cargo test -p a --test x\n'
    row 1 "check: runner only in a comment -> RED (dark)" 'every fragment is dark' bash "$T" --check "$td/dark"
    mk "$td/vac" "$wire"
    row 1 "check: empty directory -> RED (vacuity)" 'EMPTY' bash "$T" --check "$td/vac"
    rm -rf "${td:?}/vac/$DIR_REL"
    row 1 "check: directory missing -> RED" 'no such directory' bash "$T" --check "$td/vac"
    mk "$td/same" "$wire" '030-a-x.cmd=cargo test -p a --test x\n' '030-b-y.cmd=cargo test -p b --test y\n'
    row 1 "check: two fragments share an ordinal -> RED" 'ordinal 030 is shared' bash "$T" --check "$td/same"
    mk "$td/dupc" "$wire" '010-a-x.cmd=cargo test -p a --test x\n' '020-a-x-again.cmd=cargo test -p a --test x\n'
    row 1 "check: the same command in two fragments -> RED" 'same command' bash "$T" --check "$td/dupc"
    mk "$td/multi" "$wire" '010-a-x.cmd=cargo test -p a --test x\ncargo test -p b --test y\n'
    row 1 "check: a fragment with two commands -> RED" 'holds 2 command' bash "$T" --check "$td/multi"
    mk "$td/badname" "$wire" '010-a-x.cmd=cargo test -p a --test x\n' '020_b.cmd=cargo test -p b --test y\n'
    row 1 "check: a badly named fragment -> RED" 'not a regular file named' bash "$T" --check "$td/badname"

    # ── equivalence ──────────────────────────────────────────────────────
    printf "jobs:\n  t:\n    steps:\n      - run: |\n          docker run img \\\\\n            bash -c 'cargo test -p a --lib f && cargo test -p b --test y --test z --no-fail-fast && cargo build --examples'\n" > "$td/old.yml"
    frags "$td/eq" '010-a-lib-f.cmd=cargo test -p a --lib f\n' '020-b-y-z.cmd=cargo test -p b --test y --test z --no-fail-fast\n' '030-build-examples.cmd=cargo build --examples\n'
    row 0 "equivalence: same commands, same order -> PASS" 'equivalent: 3 command' bash "$T" --equivalence "$td/old.yml" "$td/eq"
    rm -f "$td/eq/020-b-y-z.cmd"
    row 1 "equivalence: a DROPPED fragment -> RED" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/eq"
    frags "$td/eq" '030-a-lib-f.cmd=cargo test -p a --lib f\n' '020-b-y-z.cmd=cargo test -p b --test y --test z --no-fail-fast\n' '040-build-examples.cmd=cargo build --examples\n'
    row 1 "equivalence: REORDERED by ordinal -> RED" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/eq"
    frags "$td/eq" '010-a-lib-f.cmd=cargo test -p a --lib f\n' '020-b-y-z.cmd=cargo test -p b --test y --test z --no-fail-fast\n' '030-build-examples.cmd=cargo build --examples\n' '040-c.cmd=cargo test -p c\n'
    row 1 "equivalence: an ADDED fragment -> RED" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/eq"
    frags "$td/eq" '010-a-lib-f.cmd=cargo test -p a --lib f\n' '020-b-y-z.cmd=cargo test -p b  --test y --test z --no-fail-fast\n' '030-build-examples.cmd=cargo build --examples\n'
    row 1 "equivalence: an inner-whitespace edit -> RED (byte for byte)" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/eq"
    row 2 "equivalence: an old workflow with NO megaline -> ENV, never a pass" 'exactly 1' bash "$T" --equivalence "$td/good/.github/workflows/ci.yml" "$td/good/$DIR_REL"

    if [ "$red" -ne 0 ]; then printf 'FAIL  check_explicit_test_commands self-test: %s rows, at least one RED\n' "$n"; return 1; fi
    printf 'PASS  check_explicit_test_commands self-test: %s rows\n' "$n"
}

case "${1:-}" in
    --self-test) self_test ;;
    --equivalence)
        [ -n "${2:-}" ] || { printf 'usage: %s --equivalence OLD_CI_YML [DIR]\n' "$0" >&2; exit 2; }
        equivalence "$2" "${3:-$DIR_REL}" ;;
    --check) check "${2:-.}" ;;
    "") check . ;;
    -h|--help) printf 'usage: %s [--check ROOT] | --equivalence OLD_CI_YML [DIR] | --self-test\n' "$0" ;;
    *) printf 'usage: %s [--check ROOT] | --equivalence OLD_CI_YML [DIR] | --self-test\n' "$0" >&2; exit 2 ;;
esac
