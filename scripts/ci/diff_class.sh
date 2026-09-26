#!/usr/bin/env bash
# diff_class.sh -- what KIND of change a diff is (#4472). ONE classifier, read by
# every consumer that scales CI to the change: scripts/ci_test_tier.sh (which test
# tier workspace-test owes) and scripts/check_pr_review_receipt.sh (whether the
# pr-review quorum may take its docs tier). Before this file each of them kept its
# own path list -- ci_test_tier.sh's docs/roadmaps+docs/audits allowlist and the
# receipt guard's per-arm trigger regexes -- and the two had drifted: PR #4467
# (README.md + two book/src pages) was docs to a reader and full-weight to both.
#
# Output (KEY=VALUE lines, exit 0):
#   class=docs     EVERY touched path is documentation (DOCS PATHS below) AND exists
#                  at the head revision. A deleted doc is NOT docs: readme_contract
#                  (a tree reader) asserts that every path CLAUDE.md cites exists, so
#                  a delete can turn a test red.
#   subclass=ledger  (with class=docs) every path is under docs/roadmaps/ or
#                  docs/audits/ -- the #3658 set that no Rust test reads at run time.
#   subclass=prose   (with class=docs) anything else docs: README.md, book pages, the
#                  rest of docs/ -- tree readers (readme_contract, the BEATS drift gate,
#                  book contracts) DO read these, so their tests still run.
#   class=code     anything else, and the reason names the first non-docs path.
#   class=empty    the diff lands no path.
#   paths=N        the number of touched paths.
#   reason=...     one line, human-readable.
# Exit 2 (ENV) when the diff cannot be derived: an unreadable --diff-from, a failed
# git diff. A consumer must treat 2 as "not docs" (fail closed, towards more CI).
#
# DOCS PATHS. Chosen so that no path on the list is COMPILED or EXECUTED by any
# build, test harness or workflow -- only READ:
#   <root>/*.md         README.md, CHANGELOG.md, CONTRIBUTING.md, ...  (not */x.md:
#                       crates/*/README.md is `include_str!`-ed into rustdoc, so a
#                       crate's README is part of its doc-tests)
#   book/**/*.md        the mdBook sources (Book workflow still compiles their code
#                       blocks: it is keyed on book/** paths, not on this class)
#   docs/**             prose, specs, roadmaps, audits, evidence notes
# NOT docs, deliberately: .claude/** (skills and agent memory are read by contract
# tests), contracts/** (pv-validated, and several are CI inputs), evidence/** (model
# ledgers read by gates), *.yaml/*.toml anywhere at the root, and every .github/ file.
#
# Usage:
#   diff_class.sh --diff-from FILE [--repo R] [--head REV]   paths, one per line
#   diff_class.sh --base REV [--head REV] [--repo R]         git diff --no-renames
#   diff_class.sh --self-test                                case table
set -uo pipefail

usage() { sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; }

is_docs_path() { # $1 = path -> 0 iff on the DOCS PATHS list
    case "$1" in
        */*) ;;
        *.md) return 0 ;;
        *) return 1 ;;
    esac
    case "$1" in
        book/*.md|docs/*) return 0 ;;
        *) return 1 ;;
    esac
}

is_ledger_path() { case "$1" in docs/roadmaps/*|docs/audits/*) return 0 ;; *) return 1 ;; esac; }

# classify <paths file> <repo> <head rev>
classify() {
    local list=$1 repo=$2 head=$3 p n=0 ledger=1
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        n=$((n + 1))
        if ! is_docs_path "$p"; then
            printf 'class=code\npaths=%s\nreason=%s is not a docs path (root *.md, book/**/*.md, docs/**) -- the diff can reach a build, a test or a workflow\n' \
                "$(grep -c . "$list")" "$p"
            return 0
        fi
        if ! git -C "$repo" cat-file -e "$head:$p" 2>/dev/null; then
            printf 'class=code\npaths=%s\nreason=%s is deleted (absent at %s) -- a tree reader asserts cited paths exist, so a doc delete is not docs\n' \
                "$(grep -c . "$list")" "$p" "$head"
            return 0
        fi
        is_ledger_path "$p" || ledger=0
    done < "$list"
    if [ "$n" -eq 0 ]; then
        printf 'class=empty\npaths=0\nreason=the diff lands no path\n'
        return 0
    fi
    if [ "$ledger" -eq 1 ]; then
        printf 'class=docs\nsubclass=ledger\npaths=%s\nreason=all %s path(s) are under docs/roadmaps/ or docs/audits/ and present at %s\n' "$n" "$n" "$head"
    else
        printf 'class=docs\nsubclass=prose\npaths=%s\nreason=all %s path(s) are docs (root *.md, book/**/*.md, docs/**) and present at %s\n' "$n" "$n" "$head"
    fi
}

self_test() {
    local td rc=0 pass=0 fail=0 me
    me=$(readlink -f "$0")
    td=$(mktemp -d "${TMPDIR:-/tmp}/diff-class.XXXXXX") || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${td:?}'" RETURN
    git -C "$td" init -q -b main r && cd "$td/r" || return 2
    git config user.email t@t && git config user.name t && git config core.hooksPath /dev/null
    mkdir -p book/src/g docs/roadmaps docs/audits docs/specifications crates/x/src .claude/skills contracts evidence/models .github/workflows
    for f in README.md CHANGELOG.md book/src/g/install.md book/src/SUMMARY.md docs/roadmaps/r.yaml docs/audits/a.md \
             docs/specifications/s.md docs/BEATS.md crates/x/README.md crates/x/src/lib.rs .claude/skills/s.md \
             contracts/c.yaml evidence/models/supported.yaml .github/workflows/ci.yml Cargo.toml book/book.toml codecov.yaml; do
        printf 'x\n' > "$f"
    done
    git add -A && git commit -qm base && git rm -q docs/specifications/s.md && git commit -qm del
    row() { # expected-regex description paths...
        local want=$1 what=$2 out; shift 2
        printf '%s\n' "$@" > "$td/d.txt"
        out=$(bash "$me" --diff-from "$td/d.txt" --repo "$td/r" 2>&1)
        if printf '%s\n' "$out" | grep -Eq -- "$want"; then pass=$((pass + 1)); printf 'ok   %s\n' "$what"
        else fail=$((fail + 1)); printf 'FAIL %s\n     want /%s/ got: %s\n' "$what" "$want" "$(printf '%s' "$out" | tr '\n' ' ')"; fi
    }
    # --- docs (#4467's exact shape first) ---
    row '^subclass=prose$' "#4467 shape: README + two book pages -> docs/prose" README.md book/src/g/install.md book/src/SUMMARY.md
    row '^class=docs$' "root CHANGELOG.md -> docs" CHANGELOG.md
    row '^subclass=ledger$' "roadmap + audit only -> docs/ledger (#3658 set)" docs/roadmaps/r.yaml docs/audits/a.md
    row '^subclass=prose$' "ledger + BEATS -> prose, not ledger" docs/roadmaps/r.yaml docs/BEATS.md
    # --- not docs: every row here must stay code ---
    row '^class=code$' "#4467 as merged (+ evidence/models/supported.yaml) -> code" README.md evidence/models/supported.yaml
    row '^class=code$' "crate README is include_str!-ed into rustdoc -> code" crates/x/README.md
    row '^class=code$' ".rs -> code" crates/x/src/lib.rs
    row '^class=code$' ".claude/ skill markdown -> code (contract tests read it)" .claude/skills/s.md
    row '^class=code$' "contracts/ yaml -> code" contracts/c.yaml
    row '^class=code$' "workflow -> code" .github/workflows/ci.yml
    row '^class=code$' "root Cargo.toml -> code" Cargo.toml
    row '^class=code$' "root config yaml -> code" codecov.yaml
    row '^class=code$' "book.toml is not a page -> code" book/book.toml
    row '^class=code$' "deleted doc -> code (tree readers assert cited paths exist)" docs/specifications/s.md
    row '^class=code$' "one code path among docs -> code" README.md docs/audits/a.md crates/x/src/lib.rs
    row '^class=empty$' "empty diff -> empty" ''
    # --- ENV: an unreadable diff is exit 2, never a class ---
    if bash "$me" --diff-from "$td/nope.txt" --repo "$td/r" >/dev/null 2>&1; then fail=$((fail + 1)); printf 'FAIL missing --diff-from must exit 2\n'
    else [ $? -eq 2 ] && { pass=$((pass + 1)); printf 'ok   missing --diff-from -> exit 2\n'; } || { fail=$((fail + 1)); printf 'FAIL missing --diff-from exit was not 2\n'; }; fi
    # --- --base derivation uses --no-renames: a rename INTO docs/ lists the source too ---
    git mv crates/x/src/lib.rs docs/audits/lib.rs && git commit -qm mv
    out=$(bash "$me" --base HEAD^1 --repo "$td/r")
    if printf '%s\n' "$out" | grep -qx 'class=code'; then pass=$((pass + 1)); printf 'ok   rename crates/x/src/lib.rs -> docs/audits/ -> code (#3664)\n'
    else fail=$((fail + 1)); printf 'FAIL rename into docs/ read as %s\n' "$(printf '%s' "$out" | tr '\n' ' ')"; fi
    printf 'diff_class self-test: %s pass, %s fail\n' "$pass" "$fail"
    [ "$fail" -eq 0 ] || rc=1
    return $rc
}

main() {
    local repo='' base='' head=HEAD diff_from='' tmp rc
    while [ $# -gt 0 ]; do
        case "$1" in
            --repo) repo=$2; shift 2 ;;
            --base) base=$2; shift 2 ;;
            --head) head=$2; shift 2 ;;
            --diff-from) diff_from=$2; shift 2 ;;
            --self-test) self_test; return $? ;;
            -h|--help) usage; return 0 ;;
            *) printf 'diff_class: unknown argument %s\n' "$1" >&2; return 2 ;;
        esac
    done
    [ -n "$repo" ] || repo=$(git rev-parse --show-toplevel 2>/dev/null) || { printf 'ENV: not in a git repo and no --repo\n' >&2; return 2; }
    if [ -n "$diff_from" ]; then
        [ -r "$diff_from" ] || { printf 'ENV: --diff-from %s is not readable\n' "$diff_from" >&2; return 2; }
        classify "$diff_from" "$repo" "$head"; return $?
    fi
    [ -n "$base" ] || { printf 'diff_class: need --diff-from or --base\n' >&2; return 2; }
    tmp=$(mktemp "${TMPDIR:-/tmp}/diff-class.XXXXXX") || return 2
    # --no-renames (#3664): both sides of a rename are touched paths.
    if ! git -C "$repo" diff --no-renames --name-only "$base" "$head" > "$tmp" 2>/dev/null; then
        rm -f -- "${tmp:?}"; printf 'ENV: git diff %s %s failed\n' "$base" "$head" >&2; return 2
    fi
    classify "$tmp" "$repo" "$head"; rc=$?
    rm -f -- "${tmp:?}"
    return $rc
}

main "$@"
