#!/usr/bin/env bash
# check_sections_claimed.sh -- every fat_driver section is run by some workflow (#3668).
#
# A section in ci/sections.yml (or a job of the vendored sovereign-ci workflow,
# section `sov.<job>`) runs only when a workflow passes it to
# `fat_driver.py run --sections '...'`. #3668 moved five sections between two
# ci.yml jobs; one left out of both lists would never run again, and nothing
# would read RED for it: a section that does not run reports no result at all.
#
# RED when a section is named by no `--sections` part outside a comment. A
# `${{ ... }}` in a part reads as `*` (the shard matrix); a part matches a
# section by equality or as a glob, as fat_driver's resolve_section_names does.
# Exempt: a manifest (`if: false`, never run by anything) and NOT_DRIVEN below.
#
#   bash scripts/check_sections_claimed.sh              # check the repo
#   bash scripts/check_sections_claimed.sh --self-test  # case table + planted mutants
#
# Text only (bash + awk); no build, no network.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="${BASH_SOURCE[0]}"

# Sections no workflow hands to the driver, each with its reason.
NOT_DRIVEN='workspace-test'   # ci.yml runs it as its own job, after the shards

# sections_from_sections_yml FILE -- one runnable section name per line:
# jobs: entries, a matrix-pins entry expanded to name[v1,v2], `if: false` dropped.
sections_from_sections_yml() {
    awk '
        /^[^ #]/ { top = $0; sub(/:.*/, "", top) }
        top == "matrix-pins" && /^  [A-Za-z0-9_.-]+:[ \t]*$/ { pin = $1; sub(/:$/, "", pin); next }
        top == "matrix-pins" && /^    - \{/ {
            s = $0; sub(/^    - \{/, "", s); sub(/\}[ \t]*$/, "", s)
            n = split(s, kv, /,[ \t]*/); label = ""
            for (i = 1; i <= n; i++) { v = kv[i]; sub(/^[^:]*:[ \t]*/, "", v); label = label (i > 1 ? "," : "") v }
            pins[pin] = pins[pin] "\n" pin "[" label "]"; next
        }
        top == "jobs" && /^  [A-Za-z0-9_.-]+:[ \t]*$/ { job = $1; sub(/:$/, "", job); order[++nj] = job; next }
        top == "jobs" && job != "" && /^    if:[ \t]*false[ \t]*$/ { off[job] = 1 }
        END {
            for (i = 1; i <= nj; i++) {
                j = order[i]
                if (off[j]) continue
                if (j in pins) printf "%s\n", substr(pins[j], 2)
                else print j
            }
        }' "$1"
}

# sections_from_sov FILE -- `sov.<job>` for each job of the vendored workflow.
sections_from_sov() {
    awk '
        /^[^ #]/ { top = $0; sub(/:.*/, "", top) }
        top == "jobs" && /^  [A-Za-z0-9_.-]+:[ \t]*$/ { j = $1; sub(/:$/, "", j); print "sov." j }
    ' "$1"
}

# spec_parts FILE... -- every comma-separated part of every --sections '...'
# outside a comment, `${{ ... }}` read as `*`.
spec_parts() {
    awk '
        /^[ \t]*#/ { next }
        {
            line = $0
            while (match(line, /--sections[ \t]+\x27[^\x27]*\x27/)) {
                spec = substr(line, RSTART, RLENGTH)
                line = substr(line, RSTART + RLENGTH)
                sub(/^--sections[ \t]+\x27/, "", spec); sub(/\x27$/, "", spec)
                gsub(/\$\{\{[^}]*\}\}/, "*", spec)
                n = split(spec, p, ",")
                for (i = 1; i <= n; i++) { s = p[i]; gsub(/^[ \t]+|[ \t]+$/, "", s); if (s != "") print s }
            }
        }' "$@"
}

# unclaimed SECTIONS_YML SOV_YML WORKFLOW... -- prints each unclaimed section.
unclaimed() {
    local sec="$1" sov="$2"; shift 2
    local -a names parts
    mapfile -t names < <(sections_from_sections_yml "$sec"; sections_from_sov "$sov")
    mapfile -t parts < <(spec_parts "$@")
    [ "${#names[@]}" -gt 0 ] || { echo "no sections read from $sec / $sov" >&2; return 2; }
    local n p hit
    for n in "${names[@]}"; do
        [[ " $NOT_DRIVEN " == *" $n "* ]] && continue
        hit=0
        for p in "${parts[@]}"; do
            # shellcheck disable=SC2053  # $p is a glob on purpose
            if [ "$n" = "$p" ] || [[ "$n" == $p ]]; then hit=1; break; fi
        done
        [ "$hit" = 1 ] || printf '%s\n' "$n"
    done
}

check_repo() {
    local -a wfs=("$ROOT"/.github/workflows/*.yml)
    local out rc=0
    out="$(unclaimed "$ROOT/ci/sections.yml" "$ROOT/ci/vendor/sovereign-ci.yml" "${wfs[@]}")" || rc=$?
    [ "$rc" = 0 ] || { echo "FAIL: could not read the sections (rc=$rc)"; return 2; }
    if [ -n "$out" ]; then
        printf '::error::section %s is named by no workflow --sections; it never runs\n' $out
        return 1
    fi
    local total
    total=$( { sections_from_sections_yml "$ROOT/ci/sections.yml"; sections_from_sov "$ROOT/ci/vendor/sovereign-ci.yml"; } | wc -l)
    echo "PASS: all $total runnable sections are named by a workflow --sections (exempt: $NOT_DRIVEN, if: false manifests)"
}

self_test() {
    local d bad=0 n=0
    d="$(mktemp -d)"
    trap 'rm -rf "${d:?}"' RETURN
    local sec="$ROOT/ci/sections.yml" sov="$ROOT/ci/vendor/sovereign-ci.yml" ci="$ROOT/.github/workflows/ci.yml"
    local -a others=()
    local f
    for f in "$ROOT"/.github/workflows/*.yml; do [ "$f" = "$ci" ] || others+=("$f"); done

    row() {  # label want got
        n=$((n + 1))
        if [ "$2" = "$3" ]; then printf 'ok   %s\n' "$1"
        else printf 'BAD  %s\n     want: %s\n     got:  %s\n' "$1" "$2" "$3"; bad=1; fi
    }
    j() { tr '\n' ' ' | sed 's/ $//'; }

    # --- extractor case table: must-match / must-not-match -------------------
    printf '%s\n' \
        "          --sections 'a,b'" \
        "  run: python3 x run --sections 'c' --results r" \
        "          --sections 'w-shard?\${{ matrix.shard }}?3?'" \
        "          --sections ' d , e '" > "$d/m.yml"
    row "extractor: folded, inline, matrix-expression and spaced parts are read" \
        "a b c w-shard?*?3? d e" "$(spec_parts "$d/m.yml" | j)"
    printf '%s\n' \
        "      # --sections 'commented'" \
        "    # --sections splits on commas" \
        "          --sections \"dq\"" \
        "          --section 'singular'" > "$d/nm.yml"
    row "extractor: a comment, a double-quoted spec and --section are not read" \
        "" "$(spec_parts "$d/nm.yml" | j)"

    # --- section listing ------------------------------------------------------
    local names
    names="$(sections_from_sections_yml "$sec" | j)"
    row "sections.yml: the shard matrix expands from matrix-pins" \
        "1" "$(grep -c 'workspace-test-shard\[1,3\] workspace-test-shard\[2,3\] workspace-test-shard\[3,3\]' <<<"$names")"
    row "sections.yml: determinism expands to both arches" \
        "1" "$(grep -c 'determinism\[X64\] determinism\[ARM64\]' <<<"$names")"
    row "sections.yml: if: false manifests are not runnable sections" \
        "0" "$(grep -cE '(^| )guard-(tree|cargo)-steps( |$)' <<<"$names")"
    row "sections.yml: the five #3668 sections are listed" \
        "5" "$(tr ' ' '\n' <<<"$names" | grep -cxE 'guard-tree|guard-cargo|vendored-schemas|pr-review-shadow|pr-review-sign')"
    row "sov: sov.gate is a section" "1" "$(sections_from_sov "$sov" | grep -cx 'sov.gate')"

    # --- the real tree, then planted defects ---------------------------------
    row "repo: every runnable section is claimed" "" "$(unclaimed "$sec" "$sov" "$ci" "${others[@]}" | j)"

    sed "s/'guard-tree,guard-cargo,/'guard-tree,/" "$ci" > "$d/drop.yml"
    row "MUTANT guard-cargo dropped from the guards job reads RED" \
        "guard-cargo" "$(unclaimed "$sec" "$sov" "$d/drop.yml" "${others[@]}" | j)"

    sed "s/,pr-review-sign'/'/" "$ci" > "$d/drop2.yml"
    row "MUTANT pr-review-sign dropped (last in its list) reads RED" \
        "pr-review-sign" "$(unclaimed "$sec" "$sov" "$d/drop2.yml" "${others[@]}" | j)"

    sed "s/'sov\.\*,/'sov.gate,/" "$ci" > "$d/sov.yml"
    row "MUTANT x86-main's sov.* narrowed to sov.gate reads RED" \
        "$(sections_from_sov "$sov" | grep -vx 'sov.gate' | j)" \
        "$(unclaimed "$sec" "$sov" "$d/sov.yml" "${others[@]}" | j)"

    local -a no_nightly=()
    for f in "${others[@]}"; do grep -q "'provable-ladder'" "$f" || no_nightly+=("$f"); done
    row "MUTANT the nightly ladder workflow removed reads RED" \
        "provable-ladder" "$(unclaimed "$sec" "$sov" "$ci" "${no_nightly[@]}" | j)"

    sed "s/^\( *\)--sections 'workspace-test-shard?/\1# --sections 'workspace-test-shard?/" "$ci" > "$d/cmt.yml"
    row "MUTANT the shard spec only in a comment reads RED for all three shards" \
        "workspace-test-shard[1,3] workspace-test-shard[2,3] workspace-test-shard[3,3]" \
        "$(unclaimed "$sec" "$sov" "$d/cmt.yml" "${others[@]}" | j)"

    sed "s/^    if: false\$/    if: true/" "$sec" > "$d/sec_on.yml"
    row "MUTANT a manifest switched on (if: true) and unclaimed reads RED" \
        "guard-tree-steps guard-cargo-steps" "$(unclaimed "$d/sec_on.yml" "$sov" "$ci" "${others[@]}" | j)"

    local rc=0
    : > "$d/empty.yml"
    unclaimed "$d/empty.yml" "$d/empty.yml" "$ci" > /dev/null 2>&1 || rc=$?
    row "no sections read (empty files) is an error, never a vacuous pass" "2" "$rc"

    echo "check_sections_claimed self-test: $n rows, $([ "$bad" = 0 ] && echo 0 || echo some) bad"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") check_repo ;;
    *) echo "usage: $SELF [--self-test]" >&2; exit 2 ;;
esac
