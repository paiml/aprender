#!/usr/bin/env bash
# ci_change_class.sh — what KIND of change is this diff? (aprender#4472)
#
# WHY
#   A PR that touches only Markdown paid for the whole fat CI surface: the sov.*
#   matrix (test, lint, coverage, bench, security, provenance), the diff-scoped
#   mutants section, both determinism rasters, the provable ladder and mac-check.
#   None of those can read a Markdown file that no Rust source compiles in. This
#   script decides ONCE, in the `change-class` job of ci.yml, and every job that
#   scales down reads that job's output. The verdict jobs (`gate`, `ci / gate`)
#   read the same output, so a section is absent only BECAUSE the classifier said
#   docs, and they say so by name ("not-triggered: docs-only"), never silently.
#
# WHAT STILL RUNS FOR class=docs (the checks that read doc CONTENT)
#   guard-tree and guard-cargo (README claims, claim literals, spec conformance,
#   receipts, ...), vendored-schemas, pr-review-shadow, pr-review-sign, and every
#   workspace-test shard (its own tier decision runs the tree-reader targets,
#   readme_contract among them). Nothing here changes what those accept.
#
# THE DOCS CLASS — every rule must hold for EVERY path in the diff:
#   1. status A or M. A delete, rename, copy or type change is `full`: a removed
#      file is read by somebody's test, and a rename is a delete.
#   2. the path ends in `.md`.
#   3. the path is not under .github/ (a workflow-adjacent file is CI config).
#   4. no Rust source or build script compiles it in: a "....md" literal inside
#      include_str!/include_bytes!/include! (resolved against the .rs file's own
#      directory), or any ".md" literal in a build.rs. A literal built with
#      env!/concat! cannot be resolved here, so its BASENAME is matched instead,
#      the conservative direction (it can only turn a docs diff `full`).
#   Anything else is `full`. An empty or underivable diff is exit 2 (ENV).
#
# OUTPUT (KEY=VALUE on stdout, for `>> "$GITHUB_OUTPUT"`)
#   class=docs | full
#   reason=<one line>
#
# EXIT
#   0  a decision was reached (either polarity)
#   2  ENV — no diff derivable, or it is empty. NEVER guess. ci.yml treats a
#      failed decision as `full`: every job that scales down tests == 'docs'.
#
#   bash scripts/ci_change_class.sh --event pull_request             # HEAD^1..HEAD (the merge ref)
#   bash scripts/ci_change_class.sh --event merge_group --base SHA   # SHA..HEAD
#   bash scripts/ci_change_class.sh --event push                     # always full
#   bash scripts/ci_change_class.sh --diff-from FILE [--repo-root D] # name-status lines (tests)
#   bash scripts/ci_change_class.sh --self-test                      # the case table
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVENT="" BASE="" DIFF_FROM="" SELF_TEST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --event) EVENT="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --diff-from) DIFF_FROM="$2"; shift 2 ;;
        --repo-root) REPO_ROOT="$2"; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        *) printf 'ci_change_class.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Every repo-relative .md path a Rust file compiles in, one per line; a literal
# that cannot be resolved is printed as `?<basename>`.
included_md() {
    local f lit dir
    while IFS= read -r f; do
        [ -f "$REPO_ROOT/$f" ] || continue
        dir=$(dirname "$f")
        # Up to three lines after the macro name: rustfmt splits a long include.
        while IFS= read -r lit; do
            case "$lit" in
                RESOLVE:*) realpath -m --relative-to="$REPO_ROOT" "$REPO_ROOT/$dir/${lit#RESOLVE:}" ;;
                BASENAME:*) printf '?%s\n' "$(basename "${lit#BASENAME:}")" ;;
            esac
        done < <(grep -A3 -E 'include(_str|_bytes)?!' "$REPO_ROOT/$f" 2> /dev/null |
            awk '/^--$/ { dyn = 0; next }
                 /env!|concat!/ { dyn = 1 }
                 { while (match($0, /"[^"]*\.md"/)) {
                       print (dyn ? "BASENAME:" : "RESOLVE:") substr($0, RSTART + 1, RLENGTH - 2)
                       $0 = substr($0, RSTART + RLENGTH) } }')
    done < <(git -C "$REPO_ROOT" grep -lE '\.md"' -- '*.rs' 2> /dev/null)
    # A build script that names a .md reads it at build time: match by basename.
    while IFS= read -r f; do
        { grep -oE '"[^"]*\.md"' "$REPO_ROOT/$f" 2> /dev/null || true; } | tr -d '"' |
            while IFS= read -r lit; do printf '?%s\n' "$(basename "$lit")"; done
    done < <(git -C "$REPO_ROOT" ls-files 'build.rs' '*/build.rs')
    return 0
}

derive_diff() {
    case "$EVENT" in
        pull_request)
            # actions/checkout leaves the PR's merge commit: HEAD^1 is the base tip.
            git -C "$REPO_ROOT" rev-parse --verify --quiet 'HEAD^1' > /dev/null || return 1
            git -C "$REPO_ROOT" diff --no-renames --name-status 'HEAD^1..HEAD'
            ;;
        merge_group)
            [ -n "$BASE" ] || return 1
            git -C "$REPO_ROOT" diff --no-renames --name-status "$BASE..HEAD"
            ;;
        *) return 1 ;;
    esac
}

decide() {
    local diff status path n=0 incl
    if [ -n "$DIFF_FROM" ]; then
        [ -f "$DIFF_FROM" ] || { printf 'reason=ENV: --diff-from names no file: %s\n' "$DIFF_FROM" >&2; return 2; }
        diff=$(cat "$DIFF_FROM")
    else
        case "$EVENT" in
            pull_request | merge_group) ;;
            *)
                printf 'class=full\nreason=event %s: only a pull_request or merge_group diff is classified\n' "${EVENT:-<none>}"
                return 0
                ;;
        esac
        diff=$(derive_diff) || {
            printf 'reason=ENV: no diff derivable for %s (base=%s), refusing to guess\n' "$EVENT" "${BASE:-<none>}" >&2
            return 2
        }
    fi
    diff=$(printf '%s\n' "$diff" | sed '/^[[:space:]]*$/d')
    [ -n "$diff" ] || { printf 'reason=ENV: the diff is EMPTY: undecidable, not docs\n' >&2; return 2; }
    incl=$(included_md)
    while IFS=$'\t' read -r status path; do
        n=$((n + 1))
        case "$status" in
            A | M) ;;
            *) printf 'class=full\nreason=%s %s: only an added or modified file can be docs\n' "$status" "$path"; return 0 ;;
        esac
        case "$path" in
            .github/*) printf 'class=full\nreason=%s: under .github/ (CI config)\n' "$path"; return 0 ;;
            *.md) ;;
            *) printf 'class=full\nreason=%s: not a Markdown file\n' "$path"; return 0 ;;
        esac
        if printf '%s\n' "$incl" | grep -qxF -e "$path" -e "?$(basename "$path")"; then
            printf 'class=full\nreason=%s: compiled into a Rust crate (include_str!/include!/build.rs)\n' "$path"
            return 0
        fi
    done <<< "$diff"
    printf 'class=docs\nreason=all %s path(s) are added/modified Markdown outside .github/ that no Rust source compiles in\n' "$n"
}

self_test() {
    local td fails=0 rows=0 T=$'\t'
    td=$(mktemp -d)
    [ -n "$td" ] && [ -d "$td" ] || { printf 'self-test: mktemp -d gave no directory\n' >&2; return 2; }
    trap 'rm -rf "${td:?}"' RETURN
    git -C "$td" init -q
    mkdir -p "$td/crates/x/src" "$td/crates/y" "$td/docs/audits"
    printf 'pub const P: &str = include_str!("../../../docs/audits/plan.md");\n' > "$td/crates/x/src/lib.rs"
    printf 'pub const S: &str = include_str!(\n    "../../../docs/split.md"\n);\n' > "$td/crates/x/src/split.rs"
    printf '#[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/GUIDE.md"))]\npub fn g() {}\n' > "$td/crates/x/src/dyn.rs"
    printf 'fn main() { let _ = std::fs::read("NOTES.md"); }\n' > "$td/crates/y/build.rs"
    # A build.rs naming no .md, sorted LAST: a no-match grep must not end the scan (it once exited 1 here).
    mkdir -p "$td/crates/z" && printf 'fn main() {}\n' > "$td/crates/z/build.rs"
    git -C "$td" add -A
    row() { # want, label, name-status lines...
        local want="$1" label="$2" got rc=0
        shift 2
        printf '%s\n' "$@" > "$td/diff.txt"
        rows=$((rows + 1))
        got=$(bash "$SELF" --diff-from "$td/diff.txt" --repo-root "$td" 2> /dev/null) || rc=$?
        got=$(printf '%s\n' "$got" | sed -n 's/^class=//p')
        [ "$rc" -eq 2 ] && got=ENV
        if [ "$got" = "$want" ]; then
            printf 'ok   %-5s %s\n' "$got" "$label"
        else
            printf 'FAIL want=%s got=%s (rc %s): %s\n' "$want" "${got:-<none>}" "$rc" "$label"
            fails=$((fails + 1))
        fi
    }
    row docs "README + two book pages (the #4467 shape)" "M${T}README.md" "M${T}book/src/introduction.md" "A${T}book/src/getting-started/installation.md"
    row docs "a doc beside, not equal to, an included one" "M${T}docs/audits/other.md"
    row full "docs + one Rust file" "M${T}README.md" "M${T}crates/x/src/lib.rs"
    row full "docs + Cargo.toml" "M${T}README.md" "M${T}Cargo.toml"
    row full "a non-Markdown file under docs/ (yaml)" "M${T}docs/roadmaps/entries/PMAT-1.yaml"
    row full "a deleted Markdown file" "D${T}docs/old.md"
    row full "a renamed Markdown file (R status)" "R100${T}docs/a.md"
    row full "Markdown under .github/" "M${T}.github/PULL_REQUEST_TEMPLATE.md"
    row full "a Markdown file include_str!-ed by a crate" "M${T}docs/audits/plan.md"
    row full "include_str! split over lines by rustfmt" "M${T}docs/split.md"
    row full "env!/concat! include: basename match" "M${T}crates/x/GUIDE.md"
    row full "a .md named by a build.rs" "M${T}crates/y/NOTES.md"
    row ENV "an empty diff is undecidable, never docs" ""
    rows=$((rows + 1))
    if bash "$SELF" --event push | grep -qx 'class=full'; then
        printf 'ok   full  a push is never classified\n'
    else
        printf 'FAIL a push did not answer class=full\n'
        fails=$((fails + 1))
    fi
    printf '%s/%s rows\n' "$((rows - fails))" "$rows"
    [ "$fails" -eq 0 ]
}

if [ "$SELF_TEST" -eq 1 ]; then
    self_test
else
    decide
fi
