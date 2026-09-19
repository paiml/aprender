#!/usr/bin/env bash
# check_ont_ratchet.sh — APR-RELEASE-001 §11.2, the five counters, measured.
#
# §11.2 gives five counters and a direction, and records "Today" as 0/0/0/0/[U].
# Those zeros lived in a spec table: nothing computed them, nothing could fail on
# them, and `git ls-files contracts/lint-baseline.json` was empty. A ratchet whose
# counters are typed into prose is a ratchet that cannot turn.
#
# THE TRAP THIS GUARD EXISTS TO REFUSE, measured 2026-09-14:
#
#   $ printf 'entity: kernel\n' | cat - contracts/<any>.yaml > /tmp/a.yaml
#   $ pv validate /tmp/a.yaml
#   0 error(s), 0 warning(s)   Contract is valid.
#
# `pv validate` ACCEPTS `entity:` — by ignoring it. There is no `pv census`, no
# `pv extract`, no `crates/aprender-contracts/src/ontology/`, and nothing in
# either contracts crate reads the key:
#
#   $ grep -rn 'get("entity")\|\["entity"\]\|entity_type\|by_entity' \
#         crates/aprender-contracts/src crates/aprender-contracts-cli/src
#   (nothing)
#
# So anchoring the corpus today would write 1818 rows that every gate calls valid
# and no code reads, while `ont.contracts_anchored` climbed from 0 to 1818. That
# is a counter measuring its own decoration. ONT-001's own R-2 — "zero is a
# decline, never an accept" — applies to the instrument as much as to a verdict:
# an ANCHOR WITH NO CONSUMER IS NOT PROGRESS, and this guard reports it as
# `inert` rather than as a number, and REFUSES a rise in it.
#
# When ONT-1 lands (`pv census`, the consumer), `consumer_present` flips to true
# on its own — it is derived from `pv --help`, never declared — and the counter
# starts meaning what §11.2 says it means.
#
#   bash scripts/check_ont_ratchet.sh            # check against the baseline
#   bash scripts/check_ont_ratchet.sh --write    # restamp (ONT R-6: make ont-ratchet)
#   bash scripts/check_ont_ratchet.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO_ROOT/contracts/lint-baseline.json"

# ── the consumer probe ───────────────────────────────────────────────────────
# Derived from the binary's own surface, never from a list here. `pv census` is
# ONT-1; until it exists, `entity:` is a key nothing reads.
ont_consumer_present() {
    local pvbin help_out
    pvbin="$(command -v pv 2>/dev/null || true)"
    [ -n "$pvbin" ] || return 1
    help_out="$("$pvbin" --help 2>&1 || true)"
    # HERE-STRING, never a pipe into a quiet grep. Under pipefail the quiet grep
    # exits on the first match, the producer takes SIGPIPE and returns 141, and the
    # PIPELINE is 141 -- so the probe would read "no census" precisely when census
    # EXISTS. check_no_pipe_into_grep_q.sh caught this line; it then caught the
    # COMMENT that replaced it, because the scanner reads text and a warning that
    # spells the banned construct IS the banned construct as far as it can tell.
    grep -qE '^[[:space:]]+census[[:space:]]' <<<"$help_out" || return 1
    return 0
}

# ── the five counters ───────────────────────────────────────────────────────
# `|| true` is LOAD-BEARING, not defensive noise. Under `set -euo pipefail` a
# grep that matches NOTHING exits 1, pipefail propagates that through
# `grep | wc | tr`, and the command substitution kills the script -- silently,
# with no output and exit 1. Every counter here starts at zero, so the FIRST
# real measurement is the one that dies. Measured: `--write` produced no file
# and no error until `bash -x` showed it stopping one line after `anchored=0`.
count_anchored() { { grep -rlE '^entity:' "$REPO_ROOT/contracts" --include='*.yaml' 2>/dev/null || true; } | wc -l | tr -d ' '; }
# ONT-4c1: a contract carries ONE `shape:` (id = its stem) or a `shapes:` list of named shapes; either anchors it.
count_shaped()   { { grep -rlE '^shapes?:' "$REPO_ROOT/contracts" --include='*.yaml' 2>/dev/null || true; } | wc -l | tr -d ' '; }

# Σ lives in crates/aprender-contracts/src/ontology/ (ONT decision 4). Absent
# directory is 0 registered, not an error — 0 is the honest reading.
count_entity_types() {
    local d="$REPO_ROOT/crates/aprender-contracts/src/ontology"
    [ -d "$d" ] || { printf '0\n'; return 0; }
    { grep -rhoE '^[[:space:]]*EntityType::[A-Za-z]+' "$d" 2>/dev/null || true; } | sort -u | wc -l | tr -d ' '
}
count_extractors() {
    local d="$REPO_ROOT/crates/aprender-contracts/src/ontology"
    [ -d "$d" ] || { printf '0\n'; return 0; }
    { grep -rlE 'impl[[:space:]]+Extractor' "$d" 2>/dev/null || true; } | wc -l | tr -d ' '
}

# ONT R-5, verbatim: "Only contracts that *should* be anchored (kernel-kind with
# a binding, and any contract naming a file) count against the
# `unanchored_but_bindable` ratchet."
#
# Read it as TWO disjuncts, and the first is a CONJUNCTION:
#   (kernel-kind AND a binding)  OR  (names a file)
#
# The first draft here spelled it `kernel OR binding OR file`, which admits a
# kernel-kind contract with no binding. Both spellings return 297 on today's
# corpus because that set is EMPTY right now — so the number agreed while the
# RULE did not, and it would have diverged silently the first time such a
# contract was written. An unanchored contract is not a defect (R-5: `entity:`
# is optional by design, and many contracts are laws, patterns or policies with
# nothing to anchor); only the bindable ones are the backlog this ↓ counter
# drains.
count_unanchored_bindable() {
    local n=0 f is_kernel has_binding names_file
    while IFS= read -r f; do
        grep -qE '^entity:' "$f" 2>/dev/null && continue
        is_kernel=0; has_binding=0; names_file=0
        grep -qE '^kind:[[:space:]]*Kernel' "$f" 2>/dev/null && is_kernel=1
        grep -qE '^[[:space:]]*binding:' "$f" 2>/dev/null && has_binding=1
        grep -qE '^[[:space:]]*(file|path|source_file):' "$f" 2>/dev/null && names_file=1
        if { [ "$is_kernel" -eq 1 ] && [ "$has_binding" -eq 1 ]; } || [ "$names_file" -eq 1 ]; then
            n=$((n+1))
        fi
    done < <(find "$REPO_ROOT/contracts" -name '*.yaml' -type f 2>/dev/null)
    printf '%s\n' "$n"
}

# ONT-6 (PMAT-3451): `armed_gates` is the arming declaration `pv lint` reads, not
# a counter. `--write` restamps the counters and must carry the declaration over
# verbatim: resetting it to [] disarms every gate (the meet declines, exit 2).
# The array must sit on ONE line; any other shape is refused rather than
# guessed at, because a partial read would rewrite the declaration silently.
# No file at all prints [] - the fail-closed reading (a decline, never an accept).
armed_gates_of() { # armed_gates_of FILE -> the one-line JSON array
    local line
    [ -f "$1" ] || { printf '[]'; return 0; }
    line="$({ grep -E '"armed_gates"' "$1" || true; } | head -1)"
    case "$line" in
        '')
            printf 'NO-GO: %s has no armed_gates; declare it before restamping\n' "$1" >&2
            return 2 ;;
        *'"armed_gates"'*:*'['*']'*)
            printf '%s' "$line" | sed -E 's/.*"armed_gates"[[:space:]]*:[[:space:]]*(\[[^]]*\]).*/\1/' ;;
        *)
            printf 'NO-GO: %s spreads armed_gates over several lines; keep the array on one\n' "$1" >&2
            return 2 ;;
    esac
}

# ONT-4c1 (PMAT-3508): `armed_shapes` is the per-shape arming declaration (ONT-001 v4.6 §3.9). Optional — a
# baseline without it arms every shape (ONT-4b's behaviour). When present it rides through `--write` verbatim
# on ONE line, exactly like armed_gates; a multi-line array is refused for the same reason. Prints nothing
# when the key is absent, so measure() can omit it.
armed_shapes_of() { # armed_shapes_of FILE -> the one-line JSON array, or nothing
    local line
    [ -f "$1" ] || return 0
    line="$({ grep -E '"armed_shapes"' "$1" || true; } | head -1)"
    case "$line" in
        '') return 0 ;;
        *'"armed_shapes"'*:*'['*']'*)
            printf '%s' "$line" | sed -E 's/.*"armed_shapes"[[:space:]]*:[[:space:]]*(\[[^]]*\]).*/\1/' ;;
        *)
            printf 'NO-GO: %s spreads armed_shapes over several lines; keep the array on one\n' "$1" >&2
            return 2 ;;
    esac
}
# Shapes declared in the corpus: one per `^shape:` (id = the stem) plus one per `- id:` entry under a
# `^shapes:` list. `shapes_unarmed` = declared − armed when armed_shapes is present, else 0 (all armed). It is
# RECORDED, not ratcheted: a new shape ships reported-first (ladder-green), so the count may rise; what the
# baseline gives a reviewer is a diff, not a silence.
count_shapes_declared() {
    local n=0 f
    while IFS= read -r f; do
        n=$((n + $(awk 'BEGIN{c=0;inl=0} /^shape:/{c++} /^shapes:/{inl=1;next} inl&&/^[^ ]/{inl=0} inl&&/^  - id:/{c++} END{print c}' "$f")))
    done < <(find "$REPO_ROOT/contracts" -name '*.yaml' -type f 2>/dev/null)
    printf '%s\n' "$n"
}
count_shapes_unarmed() { # count_shapes_unarmed BASELINE_FILE
    local armed declared
    armed="$(armed_shapes_of "$1")" || return 2
    [ -n "$armed" ] || { printf '0\n'; return 0; }
    declared="$(count_shapes_declared)"
    # entries = commas + 1 in a non-empty array
    local entries
    case "$armed" in '[]'|'[ ]') entries=0 ;; *) entries=$(( $(printf '%s' "$armed" | tr -cd ',' | wc -c) + 1 )) ;; esac
    [ "$declared" -ge "$entries" ] && printf '%s\n' $((declared - entries)) || printf '0\n'
}
measure() { # prints the JSON document
    local anchored shaped types extractors bindable consumer total armed armed_shapes shapes_line unarmed
    armed="$(armed_gates_of "$BASELINE")" || return 2
    armed_shapes="$(armed_shapes_of "$BASELINE")" || return 2
    shapes_line=""
    [ -z "$armed_shapes" ] || shapes_line="$(printf '  "armed_shapes": %s,\n' "$armed_shapes")
"
    unarmed="$(count_shapes_unarmed "$BASELINE")" || return 2
    anchored="$(count_anchored)"; shaped="$(count_shaped)"
    types="$(count_entity_types)"; extractors="$(count_extractors)"
    bindable="$(count_unanchored_bindable)"
    total="$({ find "$REPO_ROOT/contracts" -name '*.yaml' -type f 2>/dev/null || true; } | wc -l | tr -d ' ')"
    if ont_consumer_present; then consumer=true; else consumer=false; fi
    cat <<JSON
{
  "_spec": "APR-RELEASE-001 §11.2 — moves only through \`make ont-ratchet\` (ONT R-6)",
  "armed_gates": $armed,
${shapes_line}  "ont": {
    "consumer_present": $consumer,
    "contracts_total": $total,
    "entity_types_registered": $types,
    "extractors_implemented": $extractors,
    "contracts_anchored": $anchored,
    "contracts_shaped": $shaped,
    "unanchored_but_bindable": $bindable,
    "shapes_unarmed": $unarmed
  }
}
JSON
}

# Reads one scalar out of the counter document. Deliberately tolerant of BOTH
# shapes: the pretty file `make ont-ratchet` writes, and the one-line fixtures the
# self-test builds. `sed 's/[",]//g'` alone left the `}}` of a one-liner attached
# and every row compared "7" against "7}}" — a stripper that stops at the wrong
# characters is how a green-looking table measures nothing.
field() { # field JSON_FILE NAME
    { grep -E "\"$2\"" "$1" || true; } | head -1 \
        | sed -E "s/.*\"$2\"[[:space:]]*:[[:space:]]*//" \
        | sed -E 's/[^A-Za-z0-9._-].*$//'
}

self_test() {
    local t pass=0 fail=0
    t="$(mktemp -d)"
    case "$t" in /tmp/*|/var/tmp/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT WANT
        if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok    %-44s %s\n' "$1" "$3"
        else fail=$((fail+1)); printf '  FAIL  %-44s want=%s got=%s\n' "$1" "$3" "$2"; fi
    }
    printf 'check_ont_ratchet self-test\n'
    printf '{"ont":{"contracts_anchored": 7}}\n' > "$t/j.json"
    row "field reads a number"          "$(field "$t/j.json" contracts_anchored)" "7"
    printf '{"ont":{"consumer_present": false}}\n' > "$t/k.json"
    row "field reads a bool"            "$(field "$t/k.json" consumer_present)" "false"
    # ONT-6 (PMAT-3451): --write restamps the counters and must NOT reset the arming declaration.
    printf '{\n  "armed_gates": ["validate", "audit"],\n  "ont": {}\n}\n' > "$t/armed.json"
    BASELINE="$t/armed.json" measure > "$t/pres.json"
    row "measure() preserves armed_gates" "$(grep -o '"armed_gates": *\[[^]]*\]' "$t/pres.json" | tr -d ' ')" '"armed_gates":["validate","audit"]'
    # ...and so does --write, which reads and replaces the SAME file.
    cp "$t/armed.json" "$t/w.json"
    set +e
    BASELINE="$t/w.json" main --write >/dev/null 2>&1
    set -e
    row "--write preserves armed_gates in place" "$(grep -o '"armed_gates": *\[[^]]*\]' "$t/w.json" | tr -d ' ')" '"armed_gates":["validate","audit"]'
    # ONT-4c1 (PMAT-3508): armed_shapes rides through --write the same way, and its absence stays absent.
    printf '{\n  "armed_gates": ["validate"],\n  "armed_shapes": ["ont-shapes-v1", "ladder-measured"],\n  "ont": {}\n}\n' > "$t/ws.json"
    set +e
    BASELINE="$t/ws.json" main --write >/dev/null 2>&1
    set -e
    row "--write preserves armed_shapes in place" "$(grep -o '"armed_shapes": *\[[^]]*\]' "$t/ws.json" | tr -d ' ')" '"armed_shapes":["ont-shapes-v1","ladder-measured"]'
    row "--write writes shapes_unarmed"            "$(grep -c '"shapes_unarmed"' "$t/ws.json")" "1"
    row "--write does not invent armed_shapes"     "$(grep -c '"armed_shapes"' "$t/w.json")" "0"
    printf '{\n  "armed_gates": ["validate"],\n  "armed_shapes": [\n    "a"\n  ]\n}\n' > "$t/multis.json"
    set +e
    BASELINE="$t/multis.json" measure >/dev/null 2>&1
    row "a multi-line armed_shapes is refused" "$?" "2"
    set -e
    printf '{\n  "armed_gates": [\n    "validate"\n  ]\n}\n' > "$t/multi.json"
    set +e
    BASELINE="$t/multi.json" measure >/dev/null 2>&1
    row "a multi-line armed_gates is refused" "$?" "2"
    set -e
    # the measurement must be valid JSON and carry every §11.2 counter
    measure > "$t/m.json"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import json,sys;json.load(open('$t/m.json'))" >/dev/null 2>&1 \
            && row "measure() emits valid JSON" ok ok || row "measure() emits valid JSON" bad ok
    fi
    local c
    for c in entity_types_registered extractors_implemented contracts_anchored contracts_shaped unanchored_but_bindable consumer_present; do
        grep -q "\"$c\"" "$t/m.json" && row "counter present: $c" ok ok || row "counter present: $c" missing ok
    done
    # THE DECIDING ROW: with no consumer, a rise in contracts_anchored is refused.
    printf '{"ont":{"consumer_present": false,"contracts_anchored": 0}}\n' > "$t/base.json"
    local rc
    set +e
    BASELINE="$t/base.json" _ONT_FORCE_ANCHORED=5 _ONT_FORCE_CONSUMER=false compare_against "$t/base.json" >/dev/null 2>&1
    rc=$?
    set -e
    row "anchored rises with NO consumer -> refused" "$rc" "1"
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf "$t"
    [ "$fail" -eq 0 ]
}

compare_against() { # compare_against BASELINE_FILE -> 0 ok, 1 violation
    local base="$1" cur anchored_now anchored_was consumer_now shaped_now shaped_was bind_now bind_was rc=0
    cur="$(mktemp)"
    if [ -n "${_ONT_FORCE_ANCHORED:-}" ]; then
        printf '{"ont":{"consumer_present": %s,"contracts_anchored": %s,"contracts_shaped": 0,"unanchored_but_bindable": 0,"entity_types_registered":0,"extractors_implemented":0}}\n' \
            "${_ONT_FORCE_CONSUMER:-false}" "$_ONT_FORCE_ANCHORED" > "$cur"
    else
        measure > "$cur"
    fi
    anchored_now="$(field "$cur" contracts_anchored)"; anchored_was="$(field "$base" contracts_anchored)"
    shaped_now="$(field "$cur" contracts_shaped)";     shaped_was="$(field "$base" contracts_shaped)"
    bind_now="$(field "$cur" unanchored_but_bindable)"; bind_was="$(field "$base" unanchored_but_bindable)"
    consumer_now="$(field "$cur" consumer_present)"

    # THE GATE. An anchor nothing reads is decoration, and a counter that rises on
    # decoration is the theater §11 exists to stop. Refuse the rise, name the fix.
    if [ "$consumer_now" != "true" ] && [ "${anchored_now:-0}" -gt "${anchored_was:-0}" ]; then
        printf 'FAIL  contracts_anchored rose %s -> %s while consumer_present=false.\n' "${anchored_was:-0}" "${anchored_now:-0}"
        printf '      `pv census` does not exist, nothing in either contracts crate reads\n'
        printf '      `entity:`, and `pv validate` calls such a contract VALID by ignoring\n'
        printf '      the key. Anchoring now writes rows no code reads while the counter\n'
        printf '      climbs. Land ONT-1 (the consumer) first; the probe flips on its own.\n'
        rc=1
    fi
    # ↑ counters may not fall.
    [ "${anchored_now:-0}" -lt "${anchored_was:-0}" ] && { printf 'FAIL  contracts_anchored FELL %s -> %s (§11.2 is a ratchet)\n' "$anchored_was" "$anchored_now"; rc=1; }
    [ "${shaped_now:-0}"   -lt "${shaped_was:-0}"   ] && { printf 'FAIL  contracts_shaped FELL %s -> %s\n' "$shaped_was" "$shaped_now"; rc=1; }
    # ↓ counter may not rise.
    [ "${bind_now:-0}"     -gt "${bind_was:-0}"     ] && { printf 'FAIL  unanchored_but_bindable ROSE %s -> %s (§11.2: this one goes DOWN)\n' "$bind_was" "$bind_now"; rc=1; }
    rm -f "$cur"
    return "$rc"
}

main() {
    case "${1:---check}" in
        --self-test) self_test; return $? ;;
        --write)
            mkdir -p "$(dirname "$BASELINE")"
            # Never `measure > "$BASELINE"`: the shell truncates the file BEFORE
            # measure() reads armed_gates out of it, so the declaration read back
            # empty and was rewritten as [].
            local tmp
            tmp="$(mktemp "$BASELINE.XXXXXX")"
            measure > "$tmp" || { rm -f "$tmp"; return 2; }
            mv "$tmp" "$BASELINE"
            printf 'wrote %s\n' "${BASELINE#"$REPO_ROOT"/}"
            sed -n '/"ont"/,/}/p' "$BASELINE"
            return 0 ;;
        --check|"") ;;
        *) printf 'usage: %s [--check|--write|--self-test]\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    printf '== ONT ratchet (APR-RELEASE-001 §11.2) ==\n'
    if [ ! -f "$BASELINE" ]; then
        printf 'NO-GO: %s does not exist. §11.2 records the counters there;\n' "${BASELINE#"$REPO_ROOT"/}" >&2
        printf 'without it every counter is unmeasured and this guard would pass vacuously.\n' >&2
        printf 'Create it: make ont-ratchet\n' >&2
        return 2
    fi
    measure | sed -n '/"ont"/,/^  }/p'
    if compare_against "$BASELINE"; then printf 'PASS\n'; return 0; fi
    printf 'FAILED: §11.2 — the ratchet turns one way, through `make ont-ratchet` only.\n'
    return 1
}

main "$@"
