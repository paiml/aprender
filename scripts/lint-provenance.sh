#!/usr/bin/env bash
# lint-provenance.sh — ONT-001 R-10, interim: a claim with a number carries its mark.
# usage: lint-provenance.sh [--self-test] <file>...   exit 0 clean · 1 unmarked claims · 2 usage
set -euo pipefail
MARKS='\[(V|C|A|U|X)([^]]*)\]'
lint_file() {
    local f="$1" n=0 line
    [ -r "$f" ] || { printf 'lint-provenance: %s is not readable\n' "$f" >&2; return 2; }
    while IFS= read -r line; do
        case "$line" in \#*|'') continue ;; esac
        printf '%s' "$line" | rg -q '[0-9]' || continue
        printf '%s' "$line" | rg -q "$MARKS" && continue
        printf 'unmarked: %s: %s\n' "$f" "$(printf '%s' "$line" | cut -c1-90)"
        n=$((n + 1))
    done < <(rg -N '^\s*(-|\|)\s*\S' "$f" || true)
    [ "$n" -eq 0 ]
}
self_test() {
    local t rc=0
    t=$(mktemp -d "${TMPDIR:-/tmp}/lint-prov.XXXXXX")
    printf -- '- the corpus holds 1790 contracts [V 2026-09-16]\n' > "$t/green.md"
    printf -- '- the corpus holds 1790 contracts\n' > "$t/red.md"
    lint_file "$t/green.md" >/dev/null || { echo 'self-test: a marked claim must pass'; rc=1; }
    lint_file "$t/red.md" >/dev/null && { echo 'self-test: an unmarked claim must fail'; rc=1; }
    case "$t" in /*/lint-prov.??????) rm -rf "${t:?}" ;; esac
    [ "$rc" -eq 0 ] && echo 'lint-provenance: self-test ok (marked passes, unmarked fails)'
    return "$rc"
}
[ $# -gt 0 ] || { echo 'usage: lint-provenance.sh [--self-test] <file>...' >&2; exit 2; }
[ "$1" != --self-test ] || { self_test; exit $?; }
rc=0
for f in "$@"; do lint_file "$f" || rc=1; done
exit "$rc"
