#!/usr/bin/env bash
# check_guards_nightly_manifest.sh — every step scripts/guards_nightly_manifest.txt
# says moved to guards-nightly.yml must actually be a step in it.
#
# THE DEFECT THIS FIXES. The check lived inline in the workflow and matched:
#
#     grep -Fq -- "- name: $name" .github/workflows/guards-nightly.yml
#
# A fixed-string match against the BARE spelling. But a step name containing a
# colon-space must be QUOTED in YAML, and one of them does:
#
#     manifest:  Test-tier decision case table (BSE-17): quick/full/reuse, drift is ENV
#     workflow:  - name: "Test-tier decision case table (BSE-17): quick/full/reuse, drift is ENV"
#
# so the matcher looked for `- name: Test-tier...` and the file said
# `- name: "Test-tier...`. guards-nightly has been RED on this since the name was
# quoted — reporting a step that runs on every nightly as one the workflow does
# not run. A guard that names a real step as missing is worse than no guard: it
# trains the reader to ignore it.
#
# THE MATCH IS NOW ON THE PARSED NAME, not on a spelling. All three YAML forms of
# the same string are the same step, so all three match, and a name that is
# genuinely absent still fails. The case table below pins both directions.
#
#   bash scripts/check_guards_nightly_manifest.sh
#   bash scripts/check_guards_nightly_manifest.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW="${GUARDS_NIGHTLY_WORKFLOW:-$REPO_ROOT/.github/workflows/guards-nightly.yml}"
MANIFEST="${GUARDS_NIGHTLY_MANIFEST:-$REPO_ROOT/scripts/guards_nightly_manifest.txt}"

# Every `- name:` in a workflow, with YAML quoting removed. One per line.
step_names() { # step_names WORKFLOW
    sed -nE 's/^[[:space:]]*-[[:space:]]+name:[[:space:]]*(.*)$/\1/p' "$1" \
        | sed -E 's/[[:space:]]+$//' \
        | sed -E 's/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/'
}

check() {
    local n=0 bad=0 secs name names
    [ -s "$MANIFEST" ] || { printf 'FAIL: %s missing or empty\n' "$MANIFEST" >&2; return 1; }
    [ -s "$WORKFLOW" ] || { printf 'FAIL: %s missing or empty\n' "$WORKFLOW" >&2; return 1; }
    names="$(step_names "$WORKFLOW")"
    while IFS=$'\t' read -r secs name; do
        case "${secs:-}" in ''|'#'*) continue ;; esac
        [ -n "${name:-}" ] || { printf 'FAIL: manifest line has no step name: %s\n' "$secs"; bad=1; continue; }
        n=$((n + 1))
        if ! grep -Fqx -- "$name" <<<"$names"; then
            printf 'FAIL: manifest names a step this workflow does not run: %s\n' "$name"
            bad=1
        fi
    done < <(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST")
    # Vacuity: a manifest that parsed almost nothing would pass by reading nothing.
    [ "$n" -ge 5 ] || { printf 'FAIL (vacuity): only %s manifest entries parsed\n' "$n" >&2; return 1; }
    [ "$bad" -eq 0 ] || return 1
    printf 'ok  %s manifest entries, every one a step in this workflow\n' "$n"
}

self_test() {
    local t pass=0 fail=0 rc
    t="$(mktemp -d)"
    case "$t" in /tmp/*|/var/tmp/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok    %-48s %s\n' "$1" "$3"
            else fail=$((fail+1)); printf '  FAIL  %-48s want=%s got=%s\n' "$1" "$3" "$2"; fi; }

    cat > "$t/wf.yml" <<'YML'
jobs:
  x:
    steps:
      - name: Plain step
      - name: "Quoted step (BSE-17): has a colon"
      - name: 'Single quoted step'
      - name: Trailing space step   
YML
    # THE PARSER, four rows: the defect was here and nowhere else.
    # step_names is in scope here; invoking it through `bash -c 'source ...'`
    # re-ran this file's own dispatch and returned nothing, which is a broken
    # HARNESS reporting a working parser as broken — the row that matters most
    # is the one most easily faked, in either direction.
    local got
    got="$(step_names "$t/wf.yml")"
    grep -Fqx 'Plain step' <<<"$got" && row "bare name parsed" ok ok || row "bare name parsed" no ok
    grep -Fqx 'Quoted step (BSE-17): has a colon' <<<"$got" && row "DOUBLE-quoted name unquoted (the defect)" ok ok || row "DOUBLE-quoted name unquoted (the defect)" no ok
    grep -Fqx 'Single quoted step' <<<"$got" && row "single-quoted name unquoted" ok ok || row "single-quoted name unquoted" no ok
    grep -Fqx 'Trailing space step' <<<"$got" && row "trailing whitespace stripped" ok ok || row "trailing whitespace stripped" no ok

    # END TO END, both directions.
    printf '1\tPlain step\n2\tQuoted step (BSE-17): has a colon\n3\tSingle quoted step\n4\tTrailing space step\n5\tPlain step\n' > "$t/ok.txt"
    set +e; GUARDS_NIGHTLY_WORKFLOW="$t/wf.yml" GUARDS_NIGHTLY_MANIFEST="$t/ok.txt" bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; rc=$?; set -e
    row "manifest fully covered -> pass" "$rc" "0"

    printf '1\tPlain step\n2\tQuoted step (BSE-17): has a colon\n3\tSingle quoted step\n4\tTrailing space step\n5\tA step that is not there\n' > "$t/bad.txt"
    set +e; GUARDS_NIGHTLY_WORKFLOW="$t/wf.yml" GUARDS_NIGHTLY_MANIFEST="$t/bad.txt" bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; rc=$?; set -e
    row "a genuinely absent step -> STILL FAILS" "$rc" "1"

    printf '1\tPlain step\n' > "$t/thin.txt"
    set +e; GUARDS_NIGHTLY_WORKFLOW="$t/wf.yml" GUARDS_NIGHTLY_MANIFEST="$t/thin.txt" bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; rc=$?; set -e
    row "vacuity floor: <5 entries -> refuse" "$rc" "1"

    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf "$t"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    "") check ;;
    *) printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
