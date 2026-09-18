#!/usr/bin/env bash
# lint-provenance.sh — ONT-001 R-10, interim: a claim with a number carries its mark.
# usage: lint-provenance.sh [--self-test] <file>...   exit 0 clean · 1 unmarked claims · 2 usage
#
# MEASURED 2026-09-16 (quorum round 7, two independent lanes, then re-run here): the first
# version fed its loop from `rg -N '^\s*(-|\|)\s*\S'` — Markdown list and table rows ONLY.
# Its one production target is contracts/external-corpora.yaml, whose claims are YAML
# key-values, so it examined exactly ONE line (`- name: provable-contracts`), that line held
# no digit, and it exited 0. Appending `unmarked_total: 4242` also exited 0. A detector that
# cannot see the form it is pointed at is indistinguishable from a file with nothing to find:
# "0 violations over 0 files", this fleet's signature defect, aimed at its own subject.
#
# The self-test did not catch it because BOTH fixtures were .md — a fixture per FORM, where
# the rule is a fixture per FORM VARIANT. It now carries a yaml pair and an exempt-key case,
# and it asserts the count EXAMINED is non-zero, because an exit code alone cannot tell a
# clean file from an unread one.
set -euo pipefail
MARKS='\[(V|C|A|U|X)([^]]*)\]'
# Identifiers and commands, not measurements: a sha-shaped id is a claim (head:), a schema
# URI and the command that counted something are not.
EXEMPT='^(schema|ref|repo|name|mark|counted_by|item_type|id)$'
SCANNED=0
lint_file() {
    local f="$1" n=0 scanned=0 line key
    [ -r "$f" ] || { printf 'lint-provenance: %s is not readable\n' "$f" >&2; return 2; }
    while IFS= read -r line; do
        case "$line" in \#*|'') continue ;; esac
        key=$(printf '%s' "$line" | sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*:.*/\1/p')
        if [ -n "$key" ] && printf '%s' "$key" | rg -q "$EXEMPT"; then continue; fi
        printf '%s' "$line" | rg -q '[0-9]' || continue
        scanned=$((scanned + 1))
        printf '%s' "$line" | rg -q "$MARKS" && continue
        printf 'unmarked: %s: %s\n' "$f" "$(printf '%s' "$line" | cut -c1-90)"
        n=$((n + 1))
    done < <(rg -N -e '^\s*(-|\|)\s*\S' -e '^\s*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:[[:space:]]*\S' "$f" || true)
    printf 'lint-provenance: %s: %d numeric claim(s) examined, %d unmarked\n' "$f" "$scanned" "$n"
    SCANNED=$scanned
    [ "$n" -eq 0 ]
}
self_test() {
    local t rc=0
    t=$(mktemp -d "${TMPDIR:-/tmp}/lint-prov.XXXXXX")
    printf -- '- the corpus holds 1790 contracts [V 2026-09-16]\n'  > "$t/green.md"
    printf -- '- the corpus holds 1790 contracts\n'                 > "$t/red.md"
    printf -- 'n_files: 397  # [V 2026-09-16]\n'                    > "$t/green.yaml"
    printf -- 'n_files: 397\n'                                      > "$t/red.yaml"
    printf -- 'schema: ont.paiml.dev/external-corpora/v1alpha1\ncounted_by: gh api x?per_page=100\n' > "$t/exempt.yaml"
    lint_file "$t/green.md"   >/dev/null || { echo 'self-test: a marked markdown claim must pass'; rc=1; }
    lint_file "$t/red.md"     >/dev/null && { echo 'self-test: an unmarked markdown claim must fail'; rc=1; }
    lint_file "$t/green.yaml" >/dev/null || { echo 'self-test: a marked YAML claim must pass'; rc=1; }
    lint_file "$t/red.yaml"   >/dev/null && { echo 'self-test: an unmarked YAML claim must FAIL — the form-variant this guard was blind to'; rc=1; }
    # anti-vacuity: the red YAML fixture must have been READ, not merely judged
    lint_file "$t/red.yaml" >/dev/null 2>&1 || true
    [ "$SCANNED" -ge 1 ] || { echo "self-test: the YAML fixture examined $SCANNED claims — a guard that reads nothing reports nothing"; rc=1; }
    lint_file "$t/exempt.yaml" >/dev/null || { echo 'self-test: a schema URI and a counted_by command are identifiers, not claims'; rc=1; }
    case "$t" in /*/lint-prov.??????) rm -rf "${t:?}" ;; esac
    [ "$rc" -eq 0 ] && echo 'lint-provenance: self-test ok (markdown + yaml, marked passes, unmarked fails, exempt keys pass, count non-zero)'
    return "$rc"
}
[ $# -gt 0 ] || { echo 'usage: lint-provenance.sh [--self-test] <file>...' >&2; exit 2; }
[ "$1" != --self-test ] || { self_test; exit $?; }
rc=0
for f in "$@"; do lint_file "$f" || rc=1; done
exit "$rc"
