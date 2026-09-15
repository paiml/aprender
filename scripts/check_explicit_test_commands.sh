#!/usr/bin/env bash
# check_explicit_test_commands.sh -- the explicit integration-test list stays a FILE (PMAT-3313).
#
# WHY. workspace-test's "Integration tests" step was one physical ci.yml line:
# `bash -c '<39 commands joined by &&>'`, ~4000 characters. Every PR adding a test
# target edited that same line, so any two of them conflicted -- three of three
# conflicts in one backlog drain. Operator ruling 4 made it a file:
# ci/explicit-test-commands.txt, one full command per line, run in order by
# scripts/ci_run_explicit_test_commands.sh. Appends on different lines merge clean.
#
# WHAT THIS ASSERTS (default mode, over ROOT, default `.`):
#   1. the file parses (via the runner's own --list, one parser) to >= 1 command;
#   2. some workflow EXECUTES the runner on that file (a `#` comment is a mention,
#      not wiring) -- otherwise every command in it is dark;
#   3. NO workflow line chains two cargo commands with `&&` -- the mega-line may
#      not come back. A new command is a new line in the file.
#
# --equivalence OLD_CI_YML [FILE]: the representation change added, dropped and
#   reordered nothing. Extracts the ONE `&&`-chained cargo line from OLD_CI_YML,
#   splits it, trims, and compares the ordered list to FILE's parsed commands.
#   rc 0 equal, 1 differ, 2 OLD_CI_YML has no (or several) such lines.
#
# Usage:
#   bash scripts/check_explicit_test_commands.sh [--check ROOT]
#   bash scripts/check_explicit_test_commands.sh --equivalence OLD_CI_YML [FILE]
#   bash scripts/check_explicit_test_commands.sh --self-test   # case table (also the runner's)
set -euo pipefail

FILE_REL="ci/explicit-test-commands.txt"
RUNNER_REL="scripts/ci_run_explicit_test_commands.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

# is_megaline LINE -- rc 0 when LINE (comment-stripped) chains two cargo commands with &&.
# A bare `cargo` word followed by whitespace, later `&&`, then another such word.
is_megaline() {
    local code="${1%%#*}"
    grep -qE '(^|[^A-Za-z0-9_-])cargo[[:space:]].*&&[[:space:]]*cargo[[:space:]]' <<< "$code"
}

# is_wiring LINE -- rc 0 when LINE (comment-stripped) invokes the runner in --run mode on the file.
is_wiring() {
    local code="${1%%#*}"
    case "$code" in
        *"ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.txt"*) return 0 ;;
        *) return 1 ;;
    esac
}

workflow_files() { # workflow_files ROOT -> one path per line (nullglob-safe)
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
    if out=$(bash "$runner" --list "$root/$FILE_REL" 2>&1); then
        n=$(grep -c . <<< "$out")
        printf 'ok    %s parses to %s command(s)\n' "$FILE_REL" "$n"
    else
        printf 'FAIL  %s does not parse to a non-empty command list:\n%s\n' "$FILE_REL" "$(sed 's/^/        /' <<< "$out")"
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
                printf '      Put each command on its own line in %s instead (PMAT-3313).\n' "$FILE_REL"
            fi
        done < "$wf"
    done < <(workflow_files "$root")
    if [ "$megas" -gt 0 ]; then rc=1; else printf 'ok    no workflow line chains cargo commands with &&\n'; fi
    if [ "$wired" -eq 0 ]; then
        printf 'FAIL  no workflow runs `%s --run %s` (outside a comment): every command in it is dark\n' "$RUNNER_REL" "$FILE_REL"
        rc=1
    else
        printf 'ok    %s workflow invocation(s) run the file\n' "$wired"
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
    local old=$1 file=$2 a b na nb
    a=$(old_commands "$old") || return 2
    b=$(bash "$HERE/ci_run_explicit_test_commands.sh" --list "$file") || return 2
    na=$(grep -c . <<< "$a"); nb=$(grep -c . <<< "$b")
    if [ "$a" = "$b" ]; then
        printf 'PASS  equivalent: %s command(s) in the old line == %s command(s) in %s, same order, byte for byte\n' "$na" "$nb" "$file"
        return 0
    fi
    printf 'FAIL  NOT equivalent: old line %s command(s), %s %s command(s) (<: old only, >: file only):\n' "$na" "$file" "$nb"
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
    fact() { # fact LABEL -- CMD... (rc 0 == holds)
        local label=$1; shift; n=$((n + 1))
        if "$@"; then printf 'ok    row %-2s        %s\n' "$n" "$label"
        else printf 'FAIL  row %-2s        %s\n' "$n" "$label"; red=1; fi
    }
    notfact() { local label=$1; shift; n=$((n + 1))
        if "$@"; then printf 'FAIL  row %-2s        %s\n' "$n" "$label"; red=1
        else printf 'ok    row %-2s        %s\n' "$n" "$label"; fi
    }

    # ── the runner: parsing ──────────────────────────────────────────────
    printf '# header\n\n   # indented comment\n  true  \n\necho b\n' > "$td/p.txt"
    row 0 "--list skips blank and # lines and trims" '^true$' bash "$R" --list "$td/p.txt"
    fact "  ...and yields exactly [true, echo b]" test "$(bash "$R" --list "$td/p.txt" | tr '\n' '|')" = "true|echo b|"
    printf 'true\necho last' > "$td/nonl.txt"
    fact "a last line without a trailing newline is still a command" test "$(bash "$R" --list "$td/nonl.txt" | grep -c .)" = 2
    # ── the runner: vacuity ──────────────────────────────────────────────
    row 2 "VACUITY: missing file -> rc 2, never a pass" 'no such file' bash "$R" --run "$td/absent.txt"
    : > "$td/empty.txt"
    row 2 "VACUITY: empty file -> rc 2" 'ZERO commands' bash "$R" --run "$td/empty.txt"
    printf '# only\n\n  # comments\n' > "$td/comments.txt"
    row 2 "VACUITY: comments-only file -> rc 2" 'ZERO commands' bash "$R" --run "$td/comments.txt"
    # ── the runner: execution ────────────────────────────────────────────
    printf 'touch %s/a\nexit 7\ntouch %s/c\n' "$td" "$td" > "$td/ff.txt"
    row 7 "FAIL-FAST: the first non-zero command's status is the step's status" 'exit 7' bash "$R" --run "$td/ff.txt"
    fact "  ...the command before it ran" test -e "$td/a"
    notfact "  ...the command after it did NOT run" test -e "$td/c"
    printf 'cat > %s/swallowed\ntouch %s/after\n' "$td" "$td" > "$td/stdin.txt"
    row 0 "a command reading stdin gets /dev/null" '^PASS  2/2' bash "$R" --run "$td/stdin.txt"
    fact "  ...and did not swallow the next line (it ran)" test -e "$td/after"
    fact "  ...and read nothing" test ! -s "$td/swallowed"
    printf 'echo one\necho two\n' > "$td/ok.txt"
    row 0 "a group header per command, numbered" '::group::\[2/2\] echo two' bash "$R" --run "$td/ok.txt"
    row 2 "no mode -> usage, rc 2" 'usage' bash "$R"

    # ── megaline matcher: must-match / must-not-match (rule 7) ───────────
    local l
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
N|  bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.txt
TABLE
    fact "wiring MATCH: the runner on the file" is_wiring "            bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.txt"
    notfact "wiring NO-MATCH: commented out" is_wiring "  # bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.txt"
    notfact "wiring NO-MATCH: --list is parsing, not running" is_wiring "  bash scripts/ci_run_explicit_test_commands.sh --list ci/explicit-test-commands.txt"

    # ── check mode over fixture trees ────────────────────────────────────
    mk() { # mk DIR WORKFLOW_BODY FILE_BODY
        mkdir -p "$1/.github/workflows" "$1/ci" "$1/scripts"
        cp "$R" "$1/scripts/"
        printf '%b' "$2" > "$1/.github/workflows/ci.yml"; printf '%b' "$3" > "$1/$FILE_REL"
    }
    local wire='jobs:\n  t:\n    steps:\n      - run: |\n          bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.txt\n'
    mk "$td/good" "$wire" 'cargo test -p a --test x\ncargo test -p b --test y\n'
    row 0 "check: wired, non-empty, no megaline -> PASS" '^PASS' bash "$T" --check "$td/good"
    mk "$td/mega" "$wire      - run: bash -c 'cargo test -p a --test x && cargo test -p b --test y'\n" 'cargo test -p a --test x\n'
    row 1 "check: a megaline reintroduced in a workflow -> RED" 'chains cargo commands' bash "$T" --check "$td/mega"
    mk "$td/dark" 'jobs:\n  t:\n    steps:\n      # bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.txt\n      - run: true\n' 'cargo test -p a --test x\n'
    row 1 "check: runner only in a comment -> RED (dark)" 'every command in it is dark' bash "$T" --check "$td/dark"
    mk "$td/vac" "$wire" '# nothing\n'
    row 1 "check: file with zero commands -> RED (vacuity)" 'non-empty command list' bash "$T" --check "$td/vac"
    rm -f "$td/vac/$FILE_REL"
    row 1 "check: file missing -> RED" 'no such file' bash "$T" --check "$td/vac"

    # ── equivalence ──────────────────────────────────────────────────────
    printf "jobs:\n  t:\n    steps:\n      - run: |\n          docker run img \\\\\n            bash -c 'cargo test -p a --lib f && cargo test -p b --test y --test z --no-fail-fast && cargo build --examples'\n" > "$td/old.yml"
    printf '# hdr\ncargo test -p a --lib f\ncargo test -p b --test y --test z --no-fail-fast\ncargo build --examples\n' > "$td/eq.txt"
    row 0 "equivalence: same commands, same order -> PASS" 'equivalent: 3 command' bash "$T" --equivalence "$td/old.yml" "$td/eq.txt"
    printf 'cargo test -p a --lib f\ncargo build --examples\n' > "$td/drop.txt"
    row 1 "equivalence: a DROPPED command -> RED" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/drop.txt"
    printf 'cargo test -p b --test y --test z --no-fail-fast\ncargo test -p a --lib f\ncargo build --examples\n' > "$td/reorder.txt"
    row 1 "equivalence: REORDERED -> RED" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/reorder.txt"
    printf 'cargo test -p a --lib f\ncargo test -p b --test y --test z --no-fail-fast\ncargo build --examples\ncargo test -p c\n' > "$td/add.txt"
    row 1 "equivalence: an ADDED command -> RED" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/add.txt"
    printf 'cargo test -p a --lib f\ncargo test -p b  --test y --test z --no-fail-fast\ncargo build --examples\n' > "$td/ws.txt"
    row 1 "equivalence: an inner-whitespace edit -> RED (byte for byte)" 'NOT equivalent' bash "$T" --equivalence "$td/old.yml" "$td/ws.txt"
    row 2 "equivalence: an old workflow with NO megaline -> ENV, never a pass" 'exactly 1' bash "$T" --equivalence "$td/good/.github/workflows/ci.yml" "$td/eq.txt"

    if [ "$red" -ne 0 ]; then printf 'FAIL  check_explicit_test_commands self-test: %s rows, at least one RED\n' "$n"; return 1; fi
    printf 'PASS  check_explicit_test_commands self-test: %s rows\n' "$n"
}

case "${1:-}" in
    --self-test) self_test ;;
    --equivalence)
        [ -n "${2:-}" ] || { printf 'usage: %s --equivalence OLD_CI_YML [FILE]\n' "$0" >&2; exit 2; }
        equivalence "$2" "${3:-$FILE_REL}" ;;
    --check) check "${2:-.}" ;;
    "") check . ;;
    -h|--help) printf 'usage: %s [--check ROOT] | --equivalence OLD_CI_YML [FILE] | --self-test\n' "$0" ;;
    *) printf 'usage: %s [--check ROOT] | --equivalence OLD_CI_YML [FILE] | --self-test\n' "$0" >&2; exit 2 ;;
esac
