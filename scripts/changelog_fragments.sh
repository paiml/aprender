#!/usr/bin/env bash
# changelog_fragments.sh -- changelog.d/<TICKET>.md fragments (FLOW-009 RH-1/RH-2, #5761).
#
#   check [DIR]                 schema of every fragment; exit 1 on a bad one
#   aggregate [DIR] [CHANGELOG] fold fragments into CHANGELOG [Unreleased], then delete them
#   test                        plant each falsifier; must PASS/FAIL as named
set -euo pipefail

SECTIONS="Added Changed Fixed Performance Security Removed"

# One defect per line: "FILE: reason". Empty output = valid.
check_dir() {
    local dir=$1 f name
    for f in "$dir"/*.md; do
        [ -e "$f" ] || continue
        name=$(basename "$f")
        [ "$name" = README.md ] && continue
        if ! [[ "$name" =~ ^[A-Za-z]+-[0-9]+\.md$ ]]; then
            echo "$name: file name must be <TICKET>.md (e.g. PMAT-1234.md)"
            continue
        fi
        awk -v name="$name" -v secs="$SECTIONS" '
            BEGIN { n = split(secs, s, " "); for (i = 1; i <= n; i++) ok[s[i]] = 1 }
            /^[[:space:]]*$/ { next }
            /^### / { h++; sec = substr($0, 5); if (!(sec in ok)) printf "%s: unknown section \"%s\"\n", name, sec; if (b) printf "%s: heading after bullets\n", name; next }
            /^- / { b++; if (!h) printf "%s: bullet before the ### heading\n", name; next }
            { printf "%s: line is neither a heading nor a \"- \" bullet: %s\n", name, $0 }
            END { if (h != 1) printf "%s: needs exactly one ### heading (found %d)\n", name, h; if (!b) printf "%s: needs at least one bullet\n", name }
        ' "$f"
    done
}

aggregate() {
    local dir=$1 log=$2 tmp sec f body
    [ -f "$log" ] || { echo "no $log" >&2; return 1; }
    local bad
    bad=$(check_dir "$dir")
    [ -z "$bad" ] || { printf '%s\n' "$bad" >&2; return 1; }
    tmp=$(mktemp)
    body=""
    for sec in $SECTIONS; do
        local lines=""
        for f in "$dir"/*.md; do
            [ -e "$f" ] || continue
            [ "$(basename "$f")" = README.md ] && continue
            if grep -qx "### $sec" "$f"; then
                lines="$lines$(grep '^- ' "$f")"$'\n'
            fi
        done
        [ -z "$lines" ] || body="$body### $sec"$'\n\n'"$lines"$'\n'
    done
    [ -n "$body" ] || { echo "no fragments" >&2; rm -f "${tmp:?}"; return 0; }
    awk -v body="$body" '
        { print }
        /^## \[Unreleased\]/ && !done { print ""; printf "%s", body; done = 1 }
    ' "$log" >"$tmp"
    mv "$tmp" "$log"
    for f in "$dir"/*.md; do
        [ "$(basename "$f")" = README.md ] || rm -f -- "${f:?}"
    done
}

self_test() {
    local d rc=0
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    expect() { # name want(PASS|FAIL) content filename [needle]
        mkdir -p "${d:?}/$1"
        printf '%s\n' "$3" >"$d/$1/${4:-T-1.md}"
        local out got=PASS
        out=$(check_dir "$d/$1")
        [ -z "$out" ] || got=FAIL
        if [ "$got" != "$2" ] || { [ -n "${5:-}" ] && [[ "$out" != *"$5"* ]]; }; then
            echo "SELF-TEST FAIL: $1 wanted $2 ${5:-}, got $got: $out" >&2
            rc=1
        fi
    }
    expect ok PASS $'### Fixed\n\n- fix(x): y (#1)'
    expect bad-name FAIL $'### Fixed\n\n- a' notes.md 'file name must be'
    expect no-heading FAIL '- a' T-1.md 'needs exactly one ### heading'
    expect bad-section FAIL $'### Misc\n\n- a' T-1.md 'unknown section'
    expect no-bullet FAIL '### Fixed' T-1.md 'at least one bullet'
    expect prose FAIL $'### Fixed\n\nsome prose' T-1.md 'neither a heading'
    expect two-headings FAIL $'### Fixed\n- a\n### Added\n- b' T-1.md 'needs exactly one'
    # aggregate: both fragments land under [Unreleased], in section order, and the files go.
    mkdir -p "$d/agg"
    printf '# Changelog\n\n## [Unreleased]\n\n## [1.0.0]\n' >"$d/CHANGELOG.md"
    printf '### Fixed\n\n- fix a (#1)\n' >"$d/agg/PMAT-1.md"
    printf '### Added\n\n- add b (#2)\n' >"$d/agg/PMAT-2.md"
    aggregate "$d/agg" "$d/CHANGELOG.md"
    local want got
    want=$(printf '# Changelog\n\n## [Unreleased]\n\n### Added\n\n- add b (#2)\n\n### Fixed\n\n- fix a (#1)\n\n\n## [1.0.0]')
    got=$(cat "$d/CHANGELOG.md")
    [ "$got" = "$want" ] || { echo "SELF-TEST FAIL: aggregate wrote: $got" >&2; rc=1; }
    [ ! -e "$d/agg/PMAT-1.md" ] || { echo "SELF-TEST FAIL: fragment not deleted" >&2; rc=1; }
    [ "$rc" -eq 0 ] && echo "changelog_fragments self-test: 8 cases ok"
    return "$rc"
}

cmd=${1:-}
case "$cmd" in
    test) self_test ;;
    check)
        out=$(check_dir "${2:-changelog.d}")
        if [ -n "$out" ]; then printf '%s\n' "$out" >&2; exit 1; fi
        ;;
    aggregate) aggregate "${2:-changelog.d}" "${3:-CHANGELOG.md}" ;;
    *) echo "usage: $0 check [DIR] | aggregate [DIR] [CHANGELOG] | test" >&2; exit 2 ;;
esac
