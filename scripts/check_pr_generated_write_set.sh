#!/usr/bin/env bash
# check_pr_generated_write_set.sh — no PR writes docs/roadmaps/roadmap.yaml or the
# README census; main regenerates both (operator doctrine P4, #4417).
#
# WHY THIS EXISTS
# ---------------
# roadmap.yaml is GENERATED from docs/roadmaps/entries/ (#3297), and the README's
# counts are GENERATED or MEASURED (scripts/readme_sync.sh, check_readme_claims.sh).
# While every PR also carries its own regeneration, any two open PRs conflict on the
# same three files, and each merge leaves the others DIRTY. Seven of the ten open PRs
# did on 2026-09-25, including all four batches. G-11 (check_row_pr_write_set.sh)
# refuses these writes only for ROW PRs on agent/<row> branches. Every other branch
# walked through, and check_roadmap_fragment_required.sh rule 2 *required* the
# aggregate to be regenerated in the PR. P4 says a PR writes its fragment, and the
# aggregate and the README counts are written once, on the way to main.
#
# THE RULE, over merge-base..head (pull_request and merge_group shapes):
#   1. RED if the diff writes docs/roadmaps/roadmap.yaml. --no-renames is used, so
#      renaming the file away counts as a write.
#   2. RED if the diff adds or removes a README.md census line. A census line is a
#      CONTRACT_COUNT marker block, or any line the check_readme_claims.sh extractors
#      read as a claim: `**N** workspace crates`, `**N** CLI commands`, or
#      `N [one or two words] contract(s)`.
#   The ONE exemption is the single-writer regen: a diff whose EVERY path is in the
#   generated set scripts/batch_fold.sh regen() writes (roadmap.yaml, contracts.nt,
#   shapes.ttl, README.md), AND whose head roadmap.yaml is the aggregator's fixed
#   point (roadmap_fragments.py aggregate --check, the same function
#   `make roadmap-aggregate` writes with). A regen diff that carries one other path,
#   or a hand edit in place of a regeneration, is not a regen. The final CLI for the
#   single writer (`pmat roadmap sync`, pmat#1370) is to be agreed with its owner;
#   this exemption is shaped by content, not by branch name, so it holds whichever
#   tool writes the commit.
#
# PUSH SHAPE IS NOT JUDGED: a push to main IS the single writer's landing. The guard
# REPORTs the shape and exits 0, stated, never silent.
#
#   bash scripts/check_pr_generated_write_set.sh [--base <ref>] [--head <ref>] [--event <name>]
#   bash scripts/check_pr_generated_write_set.sh --self-test
#
# Exit: 0 clean · 1 a PR writes a generated file · 2 cannot judge (never a pass).
# Refs: #4417, #3650, #3709, pmat#1370, scripts/check_row_pr_write_set.sh (G-11).
set -uo pipefail
ROOT="${PRGENWS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # PRGENWS_ROOT: the mutant copies run outside the tree
PROG=check_pr_generated_write_set
PY_LIB="$ROOT/scripts/lib/roadmap_fragments.py"
ROADMAP_FILE="docs/roadmaps/roadmap.yaml"
ENTRIES_DIR="docs/roadmaps/entries"
README_FILE="README.md"
# = scripts/batch_fold.sh regen()'s generated set; a regen writes these and nothing else
GEN_SET_RE='^(docs/roadmaps/roadmap\.yaml|contracts/contracts\.nt|contracts/shapes\.ttl|README\.md)$'
# = the check_readme_claims.sh extractors (claimed_crate_count, claimed_contract_counts,
# claimed_cli_command_count) plus the generated CONTRACT_COUNT block; matched case-blind
# because claimed_contract_counts is (`grep -oiE`)
CENSUS_RE='CONTRACT_COUNT_START|\*\*[0-9]+\*\* +(workspace crates?|CLI commands?)|[0-9]+\*{0,2}( +[a-z]+){0,2} +contracts?\b'

usage() { printf 'usage: %s [--base <ref>] [--head <ref>] [--event <name>] | --self-test\n' "$PROG" >&2; exit 2; }

# regen_fixed_point <repo> <head> -> 0 when head's roadmap.yaml == aggregate(head's entries/)
regen_fixed_point() {
    local repo=$1 head=$2 td name rc
    td=$(mktemp -d "${TMPDIR:-/tmp}/prgenws.XXXXXX") || return 2
    mkdir -p "$td/entries"
    git -C "$repo" show "$head:$ROADMAP_FILE" > "$td/roadmap.yaml" 2>/dev/null || : > "$td/roadmap.yaml"
    while IFS= read -r name; do
        case "$name" in *.yaml) git -C "$repo" show "$head:$name" > "$td/entries/${name##*/}" || { rm -rf -- "${td:?}"; return 2; } ;; esac
    done < <(git -C "$repo" ls-tree -r --name-only "$head" -- "$ENTRIES_DIR" 2>/dev/null)
    python3 "$PY_LIB" aggregate --check --roadmap "$td/roadmap.yaml" --entries "$td/entries" > "$td/out" 2>&1; rc=$?
    [ "$rc" = 0 ] || sed 's/^/        /' "$td/out" | head -8
    rm -rf -- "${td:?}"
    return "$rc"
}

judge() { # judge <repo> <base> <head> <event> -> 0 clean · 1 RED · 2 ENV
    local repo=$1 base=$2 head=$3 event=$4 changed others hits rc=0
    case "$event" in
        push)
            printf 'REPORT %s: push shape; a push to main is the single writer landing, the write set was judged on the pull_request run. Not a verdict.\n' "$PROG"
            return 0 ;;
    esac
    git -C "$repo" rev-parse --verify -q "$base^{commit}" > /dev/null || { printf '%s: ENV - base %s is not a commit here (never a pass)\n' "$PROG" "$base" >&2; return 2; }
    git -C "$repo" rev-parse --verify -q "$head^{commit}" > /dev/null || { printf '%s: ENV - head %s is not a commit here (never a pass)\n' "$PROG" "$head" >&2; return 2; }
    changed=$(git -C "$repo" diff --no-renames --name-only "$base" "$head" --) || { printf '%s: ENV - git diff %s %s failed\n' "$PROG" "$base" "$head" >&2; return 2; }
    if ! printf '%s\n' "$changed" | grep -qxE "$ROADMAP_FILE|$README_FILE"; then
        printf 'PASS  %s: the diff writes neither %s nor %s\n' "$PROG" "$ROADMAP_FILE" "$README_FILE"; return 0
    fi
    # the single-writer regen: every path generated, and the aggregate at its fixed point
    others=$(printf '%s\n' "$changed" | grep -v '^$' | grep -vE "$GEN_SET_RE" || true)
    if [ -z "$others" ]; then
        if regen_fixed_point "$repo" "$head"; then
            printf 'PASS  %s: a regen-only diff (every path in the generated set) and %s is aggregate(%s/) at head: the single writer\n' "$PROG" "$ROADMAP_FILE" "$ENTRIES_DIR"; return 0
        fi
        printf 'FAIL  %s: the diff writes only generated paths but %s is NOT aggregate(%s/) at head, so it is a hand edit, not a regeneration\n' "$PROG" "$ROADMAP_FILE" "$ENTRIES_DIR"; return 1
    fi
    if printf '%s\n' "$changed" | grep -qxF -- "$ROADMAP_FILE"; then
        printf 'FAIL  %s: this PR writes %s. Write docs/roadmaps/entries/<ID>.yaml only; main regenerates the aggregate (P4, #4417)\n' "$PROG" "$ROADMAP_FILE"; rc=1
    fi
    if printf '%s\n' "$changed" | grep -qxF -- "$README_FILE"; then
        hits=$(git -C "$repo" diff "$base" "$head" -- "$README_FILE" | grep -E '^[-+][^-+]' | grep -iE -- "$CENSUS_RE" || true)
        if [ -n "$hits" ]; then
            printf 'FAIL  %s: this PR edits a README census line; main regenerates the counts (P4, #4417):\n' "$PROG"
            printf '%s\n' "$hits" | head -6 | cut -c1-200 | sed 's/^/        /'; rc=1
        fi
    fi
    [ "$rc" = 0 ] && printf 'PASS  %s: README.md prose changed, no census line and no aggregate write\n' "$PROG"
    [ "$rc" = 0 ] || printf '      remedy: restore them to the base, e.g. `git checkout %.9s -- %s` (plus README.md count lines), and keep the fragment\n' "$base" "$ROADMAP_FILE"
    return "$rc"
}

# ---------------------------------------------------------------------------
# --self-test: a fixture repo, both polarities per rule, then planted mutants
# ---------------------------------------------------------------------------
self_test() {
    local script=${1:-$0} TD R BASE n=0 red=0
    TD=$(mktemp -d "${TMPDIR:-/tmp}/prgenws-selftest.XXXXXX") || return 2
    # shellcheck disable=SC2064
    trap "case '$TD' in *prgenws-selftest.*) rm -rf -- '$TD' ;; esac" RETURN
    R="$TD/repo"; mkdir -p "$R/$ENTRIES_DIR" "$R/crates/x/src" "$R/contracts"
    git -C "$R" init -q . && git -C "$R" config user.email t@t && git -C "$R" config user.name t && git -C "$R" config core.hooksPath /dev/null
    printf -- "roadmap_version: '1.0'\nroadmap:\n- id: PMAT-100\n  title: first\n  status: planned\n" > "$R/$ROADMAP_FILE"
    printf -- '- id: PMAT-100\n  title: first\n  status: planned\n' > "$R/$ENTRIES_DIR/PMAT-100.yaml"
    # a fragment that landed with the aggregate LAGGING, the P4 shape a regen catches up
    printf -- '- id: PMAT-200\n  title: lagging\n  status: planned\n' > "$R/$ENTRIES_DIR/PMAT-200.yaml"
    printf '# apr\n\n| Workspace crates | **80** workspace crates |\n| CLI commands | **110** CLI commands |\n| Contracts | **<!-- CONTRACT_COUNT_START -->1830<!-- CONTRACT_COUNT_END -->** provable contracts |\n\nThe tree carries 1767 contracts across inference.\n\nprose line\n' > "$R/$README_FILE"
    printf 'fn main() {}\n' > "$R/crates/x/src/lib.rs"
    printf '<a> <b> <c> .\n' > "$R/contracts/contracts.nt"
    git -C "$R" add -A && git -C "$R" commit -qm base
    BASE=$(git -C "$R" rev-parse HEAD)
    local regen="python3 '$PY_LIB' aggregate --write --roadmap $ROADMAP_FILE --entries $ENTRIES_DIR >/dev/null"
    row() { # row <want rc> <must-contain> <label> <event> <shell mutating the tree>
        local want=$1 pat=$2 label=$3 event=$4 mut=$5 rc=0 out
        n=$((n + 1))
        ( cd "$R" && git checkout -q --detach "$BASE" && bash -c "$mut" && git add -A && git commit -qm "$label" --allow-empty ) > /dev/null 2>&1 \
            || { printf 'FAIL  row %-2s fixture mutation failed  %s\n' "$n" "$label"; red=1; return; }
        out=$(bash "$script" --repo "$R" --base "$BASE" --head "$(git -C "$R" rev-parse HEAD)" --event "$event" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && grep -qF -- "$pat" <<< "$out"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must contain: %s)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; sed 's/^/        /' <<< "$out" | head -8; red=1; fi
    }
    row 0 'writes neither'      "crate code only: PASS"                                      pull_request 'echo "// x" >> crates/x/src/lib.rs'
    row 0 'writes neither'      "a fragment with the aggregate left lagging: PASS (P4)"      pull_request 'printf -- "- id: PMAT-300\n  title: new\n  status: planned\n" > docs/roadmaps/entries/PMAT-300.yaml; echo "// x" >> crates/x/src/lib.rs'
    row 1 "writes $ROADMAP_FILE" "fragment + regenerated aggregate in a code PR: RED (the registered mutation)" pull_request "printf -- '- id: PMAT-300\n  title: new\n  status: planned\n' > docs/roadmaps/entries/PMAT-300.yaml; echo '// x' >> crates/x/src/lib.rs; $regen"
    row 1 "writes $ROADMAP_FILE" "a merge_group run of the same: RED"                        merge_group "echo '// x' >> crates/x/src/lib.rs; $regen"
    row 1 'census line'         "README CONTRACT_COUNT bump: RED"                            pull_request 'sed -i "s/>1830</>1831</" README.md; echo "// x" >> crates/x/src/lib.rs'
    row 1 'census line'         "README **N** workspace crates bump: RED"                    pull_request 'sed -i "s/\*\*80\*\*/**81**/" README.md; echo "// x" >> crates/x/src/lib.rs'
    row 1 'census line'         "README **N** CLI commands bump: RED"                        pull_request 'sed -i "s/\*\*110\*\*/**111**/" README.md; echo "// x" >> crates/x/src/lib.rs'
    row 1 'census line'         "an authored 'N contracts across' prose count: RED"          pull_request 'sed -i "s/1767 contracts/1768 contracts/" README.md; echo "// x" >> crates/x/src/lib.rs'
    row 0 'prose changed'       "README prose with no count: PASS"                           pull_request 'sed -i "s/prose line/prose line edited/" README.md; echo "// x" >> crates/x/src/lib.rs'
    row 0 'single writer'       "regen-only diff at the fixed point (aggregate + README count): PASS" pull_request "$regen; sed -i 's/>1830</>1831</' README.md"
    row 1 'hand edit'           "regen-only paths but a hand-edited aggregate: RED"          pull_request 'printf -- "- id: PMAT-999\n  title: hand\n  status: planned\n" >> docs/roadmaps/roadmap.yaml'
    row 1 "writes $ROADMAP_FILE" "a regen plus one code path is not a regen: RED"            pull_request "$regen; echo '// x' >> crates/x/src/lib.rs"
    row 1 "writes $ROADMAP_FILE" "renaming the aggregate away is a write (--no-renames): RED" pull_request 'git mv docs/roadmaps/roadmap.yaml docs/roadmaps/moved.yaml; echo "// x" >> crates/x/src/lib.rs'
    row 0 'REPORT'              "push shape: REPORT, exit 0, stated"                         push        "echo '// x' >> crates/x/src/lib.rs; $regen"
    n=$((n + 1)); local rc=0
    bash "$script" --repo "$R" --base 0000000000000000000000000000000000000000 --head "$BASE" --event pull_request > /dev/null 2>&1 || rc=$?
    if [ "$rc" = 2 ]; then printf 'ok    row %-2s rc=2  a base that is not a commit is ENV, never a pass\n' "$n"
    else printf 'FAIL  row %-2s rc=%s (wanted 2)  a missing base\n' "$n" "$rc"; red=1; fi
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ]
}

mutants() { # every rule deleted in a copy must turn a row RED
    local TD bad=0 name expr
    TD=$(mktemp -d "${TMPDIR:-/tmp}/prgenws-mut.XXXXXX") || return 2
    # shellcheck disable=SC2064
    trap "case '$TD' in *prgenws-mut.*) rm -rf -- '$TD' ;; esac" RETURN
    while IFS='|' read -r name expr; do
        sed "$expr" "$ROOT/scripts/check_pr_generated_write_set.sh" > "$TD/$name.sh"
        if cmp -s "$ROOT/scripts/check_pr_generated_write_set.sh" "$TD/$name.sh"; then printf 'FAIL  mutant %s did not apply: it proves nothing\n' "$name"; bad=1; continue; fi
        if PRGENWS_ROOT="$ROOT" bash "$TD/$name.sh" --self-test-only "$TD/$name.sh" > "$TD/$name.out" 2>&1; then printf 'FAIL  mutant %s SURVIVED every row\n' "$name"; bad=1
        else printf 'ok    mutant %s killed by: %s\n' "$name" "$(grep -m1 '^FAIL' "$TD/$name.out" | cut -c7-90)"; fi
    done <<'EOF'
roadmap-rule|s/grep -qxF -- "\$ROADMAP_FILE"; then/false; then/
census-rule|s/grep -iE -- "\$CENSUS_RE" || true)/grep -iE -- "NEVERMATCHES" || true)/
fixed-point|s/if regen_fixed_point "\$repo" "\$head"; then/if true; then/
regen-only|s/grep -vE "\$GEN_SET_RE" || true)/grep -vE "." || true)/
push-report|s/^        push)$/        push-never)/
EOF
    [ "$bad" = 0 ]
}

REPO="$ROOT"; BASE=""; HEAD_REF="HEAD"; EVENT="${GITHUB_EVENT_NAME:-pull_request}"
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test "$0" || exit 1; echo "== mutants (each must turn a row RED)"; mutants || exit 1; exit 0 ;;
        --self-test-only) self_test "${2:-$0}"; exit $? ;;
        --repo) REPO=$2; shift 2 ;;
        --base) BASE=$2; shift 2 ;;
        --head) HEAD_REF=$2; shift 2 ;;
        --event) EVENT=$2; shift 2 ;;
        *) usage ;;
    esac
done
[ "$EVENT" = push ] && { judge "$REPO" "" "" push; exit $?; }
if [ -z "$BASE" ]; then
    # shellcheck source=scripts/lib/resolve_base.sh
    REPO_ROOT="$REPO"; . "$ROOT/scripts/lib/resolve_base.sh" || exit 2
    resolve_base "$HEAD_REF" || { printf '%s: no base can be named; pass --base (never a pass)\n' "$PROG" >&2; exit 2; }
    case "$BASE_HOW" in
        *"push shape"*) printf 'REPORT %s: %s (%s); a commit on main is the single writer landing. Not a verdict.\n' "$PROG" "${BASE_REF:0:9}" "$BASE_HOW"; exit 0 ;;
    esac
    BASE="$BASE_REF"; printf '               comparand: %s (%s)\n' "${BASE:0:9}" "$BASE_HOW"
fi
judge "$REPO" "$BASE" "$HEAD_REF" "$EVENT"
exit $?
