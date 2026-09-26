#!/usr/bin/env bash
# check_changelog_covers_merged.sh - a release's CHANGELOG section must cover
# every PR merged since the last release, and nothing that shipped before it.
#
# THE DEFECT (#3183). dogfood.sh's changelog row was `grep -qF "$VERSION"
# CHANGELOG.md`: a heading with an empty body passed, and so did a section that
# described some other release's work. Measured on v0.69.1..v0.69.3: 14 merged
# PRs with a trailing (#N), a [0.69.3] section citing two refs, and a PASS.
#
# THE PREDICATE, both directions, over `git log --first-parent LAST_TAG..CUT`:
#   MISSING   a merged commit whose subject ends in `(#PR)` is covered when the
#             version's section cites #PR or ANY `#N` in that subject (the
#             sections are prose that cite the issue a PR fixes, not a row per
#             PR). A subject with no trailing (#N) is not a PR merge; skipped.
#   FALSE ROW a ref the section cites whose `(#R)` merge commit is an ancestor
#             of LAST_TAG - work the previous release already shipped.
# --notes FILE applies the same predicate to a release-notes file, whole.
#
# Usage: check_changelog_covers_merged.sh [--changelog F] [--notes F]
#                                         LAST_TAG CUT VERSION
#        check_changelog_covers_merged.sh --self-test
# Exit 0 = covered. 1 = a missing or false row (each printed). 2 = usage/env.

set -euo pipefail

CHANGELOG=CHANGELOG.md
NOTES=""
usage() { sed -n '19,23p' "$0" >&2; exit 2; }

section_refs() { # section_refs FILE VERSION|"" -> sorted unique ref numbers
    local file=$1 ver=$2
    if [ -n "$ver" ]; then
        awk -v v="$ver" '/^## \[/{p=index($0,"["v"]")>0} p' "$file"
    else
        cat "$file"
    fi | grep -oE '#[0-9]+' | tr -d '#' | sort -u || true
}

check_one() { # check_one LABEL FILE VERSION|"" LAST CUT -> prints rows, returns 1 on any
    local label=$1 file=$2 ver=$3 last=$4 cut=$5 bad=0 subj pr n covered r
    local -A cited=() shipped=()
    local hdr
    hdr=$(printf '## [%s' "$ver")
    if [ -n "$ver" ] && ! grep -qF -- "$hdr]" "$file"; then
        printf 'MISSING %s: no "## [%s]" section\n' "$label" "$ver"
        return 1
    fi
    while IFS= read -r r; do
        [ -n "$r" ] || continue
        cited["$r"]=1
    done < <(section_refs "$file" "$ver")

    # Merge commits the previous release already carried, by their (#PR).
    while IFS= read -r subj; do
        [[ $subj =~ \(\#([0-9]+)\)$ ]] && shipped[${BASH_REMATCH[1]}]=1
    done < <(git log --first-parent --format=%s "$last")

    while IFS= read -r subj; do
        [[ $subj =~ \(\#([0-9]+)\)$ ]] || continue
        pr=${BASH_REMATCH[1]}
        covered=0
        for n in $(grep -oE '#[0-9]+' <<< "$subj" | tr -d '#'); do
            if [ -n "${cited[$n]:-}" ]; then covered=1; break; fi
        done
        if [ "$covered" -eq 0 ]; then
            printf 'MISSING %s: #%s merged in %s..%s is not cited: %s\n' "$label" "$pr" "$last" "$cut" "$subj"
            bad=1
        fi
    done < <(git log --first-parent --format=%s "$last..$cut")

    for r in "${!cited[@]}"; do
        if [ -n "${shipped[$r]:-}" ]; then
            printf 'FALSE-ROW %s: #%s cited, but its merge already shipped in %s\n' "$label" "$r" "$last"
            bad=1
        fi
    done
    return "$bad"
}

self_test() {
    local td rc fail=0 self out
    self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    td=$(mktemp -d "${TMPDIR:-/tmp}/clcov-self.XXXXXX") || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${td:?}'" EXIT
    (
        cd "$td"
        git init -q --template="$td/.none" . && git config user.email t@t && git config user.name t
        c() { git commit -q --allow-empty -m "$1"; }
        c "base"; c "feat: old thing (#10)"; git tag v1.0.0
        c "fix(#20): the bug (#21)"; c "feat: new thing (#22)"; c "chore: no pr number"
    ) >/dev/null 2>&1 || { echo "self-test: fixture repo failed" >&2; exit 2; }
    run() { # run WANT MUST_MATCH LABEL CHANGELOG_BODY
        printf '# Changelog\n\n## [Unreleased]\n\n%b\n## [1.0.0]\n- old (#10)\n' "$4" > "$td/CHANGELOG.md"
        out=$(cd "$td" && bash "$self" --changelog CHANGELOG.md v1.0.0 HEAD 1.1.0 2>&1) && rc=0 || rc=$?
        if [ "$rc" -eq "$1" ] && grep -qE -- "$2" <<< "$out"; then printf 'ok   %s (exit %s)\n' "$3" "$rc"
        else printf 'FAIL %s: want exit %s + /%s/, got %s: %s\n' "$3" "$1" "$2" "$rc" "$out"; fail=1; fi
    }
    run 0 '^OK:'              "covered by PR number and by the cited issue" '## [1.1.0]\n- fix #20\n- new thing (#22)\n'
    run 1 '^MISSING.*#22 '    "a deleted row is RED"                         '## [1.1.0]\n- fix #20\n'
    run 1 '^MISSING.*#21 '    "a PR citing neither its number nor its issue is RED" '## [1.1.0]\n- new (#22)\n'
    run 1 '^FALSE-ROW.*#10 '  "a false row (shipped in v1.0.0) is RED"       '## [1.1.0]\n- fix #20\n- new (#22)\n- old (#10)\n'
    run 1 'no "## \[1.1.0\]"' "an absent version section is RED"             ''
    run 1 '^MISSING.*#21 '    "a bare heading (the old grep PASS) is RED"    '## [1.1.0]\n'
    return "$fail"
}

case "${1:-}" in --self-test) self_test; exit $? ;; esac
while [ $# -gt 0 ]; do
    case "$1" in
        --changelog) [ $# -ge 2 ] || usage; CHANGELOG=$2; shift 2 ;;
        --notes) [ $# -ge 2 ] || usage; NOTES=$2; shift 2 ;;
        -h|--help) usage ;;
        *) break ;;
    esac
done
[ $# -eq 3 ] || usage
LAST=$1 CUT=$2 VER=$3
git rev-parse -q --verify "$LAST^{commit}" >/dev/null || { echo "no such ref: $LAST" >&2; exit 2; }
git rev-parse -q --verify "$CUT^{commit}" >/dev/null || { echo "no such ref: $CUT" >&2; exit 2; }
[ -f "$CHANGELOG" ] || { echo "no changelog: $CHANGELOG" >&2; exit 2; }

rc=0
check_one "$CHANGELOG [$VER]" "$CHANGELOG" "$VER" "$LAST" "$CUT" || rc=1
if [ -n "$NOTES" ]; then
    [ -f "$NOTES" ] || { echo "no release notes: $NOTES" >&2; exit 2; }
    check_one "$NOTES" "$NOTES" "" "$LAST" "$CUT" || rc=1
fi
[ "$rc" -eq 0 ] && printf 'OK: %s [%s] covers every PR merged in %s..%s\n' "$CHANGELOG" "$VER" "$LAST" "$CUT"
exit "$rc"
