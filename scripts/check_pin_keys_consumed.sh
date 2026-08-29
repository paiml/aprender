#!/usr/bin/env bash
#
# check_pin_keys_consumed.sh -- a declared pin field that nothing reads is a
# setting that does not exist (aprender#2743, epic APR-PERF-GATE-001 #2706).
#
# WHY THIS EXISTS
# ---------------
# scripts/llama_pin.toml declared `flash_attention = false`. Nothing passed
# `-fa`. Neither invocation of record carried the flag, so the file described a
# configuration that had never run -- and in the PINNED era of llama.cpp (7746,
# 39173bcac) `-fa` defaults to `auto`, so the comparator may well have been
# running flash attention ON while the declaration said `false`.
#
# It is the `-b 1` defect (#2737) with the sign reversed. There a declared value
# WAS enforced and handicapped the comparator 2.4x. Here a declared value was
# NOT enforced. Both make the receipt's own record of how a number was measured
# untrue, which is this epic's subject.
#
# THE EXISTING GUARDS COULD NOT HAVE CAUGHT IT, AND IT IS WORTH BEING PRECISE
# ABOUT WHY -- this is the "recorded 18,292 times, never COMPARED" pattern:
#
#   check_bench_protocol.sh   names flash_attention in a bare word list and asks
#                             `grep -qE "^\s*flash_attention\s*="`. That proves
#                             the key is DECLARED. It never compares it to any
#                             invocation. Collection was flawless; only the
#                             comparison was missing.
#   check_comparator_flags.sh compares declaration against invocation properly,
#                             but its universe was the five flags that had
#                             already bitten us: -c -t -b -np -ngl. It was
#                             written for the field that failed and was blind to
#                             its siblings -- the recurring shape in this repo.
#
# So the answer to "is the existing guard already general?" is NO, measured:
# before #2743 it covered 5 of the 35 keys the pin declares. This guard is the
# general one, and it is deliberately about a different question -- not "does
# the invocation match the declaration" (that is check_comparator_flags.sh) but
# "does anything read this declaration AT ALL".
#
# WHAT COUNTS AS CONSUMPTION
# --------------------------
# Three ways, in decreasing order of strength:
#
#   PRODUCER      Perturbing the key changes what llama_comparator_server_flags
#                 emits. This is BEHAVIOURAL -- the key is proven to reach the
#                 comparator invocation, not merely to be mentioned near it.
#                 Immune to wrapper functions, indirection and renamed locals.
#   SUBSTITUTED   `{key}` appears inside the pin's own *_command templates, so
#                 rendering the declared invocation consumes it.
#   READ          Some file outside the pin extracts the key's VALUE.
#
# MENTION IS NOT CONSUMPTION, AND THAT DISTINCTION IS THE WHOLE GUARD. A guard
# that counted any occurrence of the key name would be vacuous: every key
# appears in check_bench_protocol.sh's REQUIRED list, so every key would look
# consumed and this file would pass on the exact tree that shipped the defect.
#
# The rule is mechanical: after stripping comments, a line whose ENTIRE content
# is the key name is a word-list entry -- a mention. A line that contains the
# key name alongside anything else is extracting or acting on it -- a read.
# `n_low` alone on a line is the REQUIRED list; `low=$(sed -n 's/...n_low...')`
# is a read. See the case table below, which runs on every invocation.
#
# THE UNIVERSE, WHICH MATTERS MORE THAN THE PATTERN
# -------------------------------------------------
# Two universes, and they fail in opposite directions:
#
#   KEYS      from llama_pin.toml itself. A key the extractor MISSES gets a free
#             pass, so the extractor ships a must-match/must-not-match table.
#   CONSUMERS from a working-tree `find`, never `git ls-files`. An untracked
#             consumer would otherwise be invisible -- though note this
#             direction is the safe one: missing a consumer makes a key look
#             DEAD (false RED), while missing a KEY is a silent free pass.
#
# THE LEDGER
# ----------
# Eleven keys are declared and read by nothing. They are recorded in
# scripts/pin_unconsumed_ledger.txt with a per-key reason, not waved through:
# the ratchet is TWO-WAY. A new dead key FAILS (debt may not grow), and a
# ledgered key that becomes consumed FAILS until its line is deleted (the
# ledger may not rot into a list of things that are secretly fine).
#
#   bash scripts/check_pin_keys_consumed.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PIN=scripts/llama_pin.toml
LEDGER=scripts/pin_unconsumed_ledger.txt
rc=0

[ -f "$PIN" ] || { printf 'FAIL  %s is missing\n' "$PIN"; exit 2; }
[ -f "$LEDGER" ] || { printf 'FAIL  %s is missing\n' "$LEDGER"; exit 2; }

# --------------------------------------------------------------------------
# The key extractor. A key this misses is a key that gets a free pass, so it is
# the one regex in this file with a must-match / must-not-match table.
# --------------------------------------------------------------------------
KEY_RE='^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*'

# LC_ALL=C: locale collation ignores underscores, so `_leading` sorted after
# `indented` under en_US and before it under C. A guard whose universe depends
# on the ambient locale reports a different set on the runner than on the dev
# box.
extract_keys() { sed -n "s/${KEY_RE}/\1/p" "$1" | LC_ALL=C sort -u; }

printf -- '--- pin key extractor: case table ------------------------------------\n'
CT=$(mktemp -d) || exit 2
trap 'rm -rf "${CT:?}"' EXIT

# must-match: these ARE declarations and must be seen
cat > "$CT/must.toml" <<'EOF'
plain = 1
quoted = "x"
tight=2
  indented = 3
with_digits9 = 4
_leading = 5
spaced   =   6
list = [1, 2]
trailing = 7   # comment after a value
EOF
# must-NOT-match: none of these declares a key
cat > "$CT/mustnot.toml" <<'EOF'
# commented = 1
#commented_tight = 2
[section]
   # indented_comment = 3
prose mentioning equals = signs is not indented like a key
9leading_digit = 4
EOF
want_must='_leading indented list plain quoted spaced tight trailing with_digits9'
got_must=$(extract_keys "$CT/must.toml" | tr '\n' ' ' | sed 's/ $//')
if [ "$got_must" = "$want_must" ]; then
    printf 'ok    must-match: all 9 declaration forms extracted\n'
else
    printf 'FAIL  must-match: got [%s], want [%s]\n' "$got_must" "$want_must"; rc=1
fi
got_mustnot=$(extract_keys "$CT/mustnot.toml" | tr '\n' ' ' | sed 's/ $//')
# `prose mentioning equals` has no leading-key shape; `9leading_digit` starts
# with a digit; the rest are comments or a [section] header.
if [ -z "$got_mustnot" ]; then
    printf 'ok    must-not-match: comments, sections and prose extract nothing\n'
else
    printf 'FAIL  must-not-match: extracted [%s] from a file declaring nothing\n' "$got_mustnot"; rc=1
fi

# --------------------------------------------------------------------------
# The mention-vs-read classifier: the distinction the whole guard rests on.
# --------------------------------------------------------------------------
# Strip comments, then a line whose entire remaining content is the bare key is
# a word-list entry. Anything else containing the key acts on it.
#
# A HERESTRING, NEVER A PIPE INTO grep -q. `producer | grep -q X` returns 141
# when grep matches and exits early: pipefail then hands back the producer's
# SIGPIPE and the match reads as NO MATCH. Five instances of that were found in
# this repo; it is input-size dependent, so it is green locally and red in CI.
reads_key() { # reads_key <key> <file>  -> 0 if the file extracts the key's value
    local key="$1" file="$2" stripped hits
    # Comments stripped and both ends trimmed in ONE sed, so what follows is
    # a herestring at every step and this function contains no pipeline at all.
    stripped=$(sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        "$file" 2>/dev/null)
    # Lines that mention the key, as a whole token...
    hits=$(grep -E "(^|[^A-Za-z0-9_])${key}([^A-Za-z0-9_]|$)" <<< "$stripped")
    # ...minus the lines that are ONLY the key, which are word-list entries.
    hits=$(grep -vxF "$key" <<< "$hits")
    [ -n "$hits" ]
}

printf -- '\n--- mention vs read: case table --------------------------------------\n'
cat > "$CT/mention.sh" <<'EOF'
REQUIRED="
flash_attention
prompt_id
"
EOF
cat > "$CT/read.sh" <<'EOF'
fa=$(llama_pin_get_raw flash_attention "$file")
EOF
cat > "$CT/comment.sh" <<'EOF'
# flash_attention same on both sides, whichever it is
EOF
cat > "$CT/wrapper.sh" <<'EOF'
gl=$(decl_of flash_attention)
EOF
cat > "$CT/substring.sh" <<'EOF'
my_flash_attention_helper=1
EOF
ct_row() { # ct_row <name> <file> <key> <expect 0|1>
    if reads_key "$3" "$2"; then got=0; else got=1; fi
    if [ "$got" = "$4" ]; then
        printf 'ok    %-52s %s\n' "$1" "$([ "$4" = 0 ] && echo READ || echo 'not a read')"
    else
        printf 'FAIL  %-52s expected %s, got %s\n' "$1" "$4" "$got"; rc=1
    fi
}
ct_row 'a bare word-list entry is a MENTION'            "$CT/mention.sh"   flash_attention 1
ct_row 'a comment is a MENTION'                         "$CT/comment.sh"   flash_attention 1
ct_row 'llama_pin_get_raw <key> is a READ'              "$CT/read.sh"      flash_attention 0
ct_row 'a wrapper function arg is a READ'               "$CT/wrapper.sh"   flash_attention 0
ct_row 'the key as a SUBSTRING of an identifier is not' "$CT/substring.sh" flash_attention 1
ct_row 'an absent key is not a read'                    "$CT/read.sh"      prompt_id       1

# --------------------------------------------------------------------------
# Behavioural: does perturbing the key change the comparator invocation?
# --------------------------------------------------------------------------
# shellcheck source=scripts/llama_bin.sh
. scripts/llama_bin.sh >/dev/null 2>&1 || true
if ! command -v llama_comparator_server_flags >/dev/null 2>&1; then
    printf 'FAIL  scripts/llama_bin.sh defines no llama_comparator_server_flags\n'
    exit 1
fi

BASE=$(llama_comparator_server_flags 999 "$PIN") || BASE=""
[ -n "$BASE" ] || { printf 'FAIL  the pin yields no comparator invocation at all\n'; exit 1; }

# Perturb one key to a sentinel and see whether the producer notices. A key that
# reaches the invocation cannot leave the output unchanged; a key that does not
# cannot change it. Two sentinels, because a numeric knob and a tri-state knob
# accept different values and a single sentinel would refuse one of them.
producer_notices() { # producer_notices <key>
    local key="$1" v out
    for v in 4242 '"true"' 'true'; do
        sed "s|^[[:space:]]*${key}[[:space:]]*=.*|${key} = ${v}|" "$PIN" > "$CT/perturbed.toml"
        out=$(llama_comparator_server_flags 999 "$CT/perturbed.toml") || out="REFUSED"
        [ "$out" = "$BASE" ] || return 0
    done
    return 1
}

# --------------------------------------------------------------------------
# The verdict, over every declared key.
# --------------------------------------------------------------------------
printf -- '\n--- every declared key must be consumed -------------------------------\n'

# CONSUMERS from a working-tree find, NOT `git ls-files`: an untracked consumer
# would otherwise be invisible and its key would read as dead.
#
# NARROWED TO FILES THAT NAME THE PIN, and that is a correctness fix rather than
# an optimisation. Scanning every script made two keys pass on coincidence:
# `name` matched the `name:` of every step in every workflow, and `quantization`
# matched a receipt field of the same name in bench_receipt.py. Neither file
# reads the pin. A file cannot read this declaration without naming it -- it has
# to construct the path -- so naming it is a necessary condition, and requiring
# it removes a whole class of same-token coincidence.
#
# THIS FILE EXCLUDES ITSELF. Its case tables mention flash_attention and
# prompt_id as fixture data, and counting those made `prompt_id` -- a genuinely
# dead key -- report READ (scripts/check_pin_keys_consumed.sh). A guard that
# reads its own fixtures as evidence is green on its own defect, which is
# exactly the shape #2733 was filed for.
#
# WORKFLOWS ARE NOT IN THE UNIVERSE, and that was checked rather than assumed:
# every .github/workflows file that names llama_pin does so on a `run:` line
# invoking a guard script (nightly-bench.yml:67, ci.yml:1022) and none parses a
# pin value. Meanwhile every workflow is dense with `- name:` step keys, which
# made the pin's own `name = "llama.cpp"` report READ (.github/workflows/ci.yml)
# on pure token collision. Re-check this if a workflow ever reads the pin
# directly.
SELF=check_pin_keys_consumed.sh
mapfile -t CONSUMERS < <(find scripts -type f \
    \( -name '*.sh' -o -name '*.py' -o -name '*.awk' \) \
    ! -name "$SELF" 2>/dev/null \
    | xargs grep -l 'llama_pin' 2>/dev/null | LC_ALL=C sort)
[ "${#CONSUMERS[@]}" -ge 5 ] || {
    printf 'FAIL  only %s files name %s; the search universe collapsed\n' \
        "${#CONSUMERS[@]}" "$PIN"
    exit 1
}
printf 'universe: %s scripts name the pin (%s excluded: its own fixtures)\n' \
    "${#CONSUMERS[@]}" "$SELF"

# The pin's own command templates, for the SUBSTITUTED test.
TEMPLATES=$(grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*_command[[:space:]]*=' "$PIN")

# The ledger: key in field 1, reason after it. Comments and blanks ignored.
ledger_keys=$(sed -e 's/#.*$//' "$LEDGER" | awk 'NF {print $1}' | sort -u)

KEYS=$(extract_keys "$PIN")
n_keys=0; n_dead=0; dead_new=""; ledger_stale=""
for key in $KEYS; do
    n_keys=$((n_keys + 1))
    verdict=""
    if producer_notices "$key"; then
        verdict=PRODUCER
    elif grep -qF "{$key}" <<< "$TEMPLATES"; then
        verdict=SUBSTITUTED
    else
        for f in "${CONSUMERS[@]}"; do
            if reads_key "$key" "$f"; then verdict="READ ($f)"; break; fi
        done
    fi

    in_ledger=no
    grep -qxF "$key" <<< "$ledger_keys" && in_ledger=yes

    if [ -n "$verdict" ]; then
        if [ "$in_ledger" = yes ]; then
            ledger_stale="$ledger_stale $key"
            printf 'FAIL  %-26s %s -- but it is still in the ledger\n' "$key" "$verdict"
        else
            printf 'ok    %-26s %s\n' "$key" "$verdict"
        fi
    else
        n_dead=$((n_dead + 1))
        if [ "$in_ledger" = yes ]; then
            printf 'DEBT  %-26s declared, read by nothing (ledgered)\n' "$key"
        else
            dead_new="$dead_new $key"
            printf 'FAIL  %-26s DECLARED AND READ BY NOTHING\n' "$key"
        fi
    fi
done

# VACUITY. A pin that stopped declaring things, or an extractor that broke,
# would sweep clean. The floor is well below today's 35 so an intentional
# removal does not trip it, and well above zero so a collapse does.
if [ "$n_keys" -lt 25 ]; then
    printf '\nFAIL  only %s keys extracted from %s; at least 25 are expected.\n' "$n_keys" "$PIN"
    printf '      Either the declaration collapsed or the extractor broke, and\n'
    printf '      an empty universe passes every check in this file.\n'
    exit 1
fi

if [ -n "$dead_new" ]; then
    printf '\nFAIL  these keys are declared and read by nothing:%s\n' "$dead_new"
    printf '      A pin field that no code consumes is a setting that does not\n'
    printf '      exist. Either join it to the invocation that should honour it,\n'
    printf '      or delete it -- it may not sit in the file looking enforceable.\n'
    printf '      (%s is for debt that was argued for, not for new debt.)\n' "$LEDGER"
    rc=1
fi
if [ -n "$ledger_stale" ]; then
    printf '\nFAIL  these ledger entries are stale -- the key IS now consumed:%s\n' "$ledger_stale"
    printf '      Delete the line. A ledger that keeps entries after they are\n'
    printf '      fixed rots into a list of things that are secretly fine.\n'
    rc=1
fi

printf '\n%s keys declared, %s consumed, %s ledgered as unconsumed debt\n' \
    "$n_keys" "$((n_keys - n_dead))" "$n_dead"
if [ "$rc" -eq 0 ]; then
    printf 'PASS  every declared pin field is read by something, or is recorded\n'
    printf '      debt that may not grow.\n'
else
    printf 'FAIL  see rows above (#2743).\n'
fi
exit "$rc"
