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
# #3569 — THE COUNTERS ARE MEASURED AT BOTH ENDS, NEVER STORED. They used to be
# typed into contracts/lint-baseline.json by `--write` and compared against the
# working tree. That made the comparand a file the pull request itself edits:
# restamp in the same commit and every direction check compared the branch with
# itself. Every PR that moved a counter also had to restamp, and two such PRs
# conflicted on the same lines. Now `--check` measures the comparand tree
# (scripts/lib/resolve_base.sh: merge-base with origin/main on a branch, the
# FIRST PARENT on a push to main — where the merge-base is HEAD itself and HEAD
# judged against HEAD is a vacuous pass — and a refusal, never the tree against
# itself, when no base can be named),
# extracted with `git archive`, and the working tree, with ONE instrument, and
# holds each direction between those two measurements. lint-baseline.json keeps
# only DECISIONS: armed_gates, armed_shapes (both reviewable, and read by `pv
# lint`) and the two Rust gates' own shrink-only numbers.
#
#   bash scripts/check_ont_ratchet.sh            # comparand tree -> working tree
#   bash scripts/check_ont_ratchet.sh --print    # the measurement of the working tree
#   bash scripts/check_ont_ratchet.sh --write    # normalise lint-baseline.json to decisions only
#   bash scripts/check_ont_ratchet.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO_ROOT/contracts/lint-baseline.json"
PROG=check_ont_ratchet
# shellcheck source=scripts/lib/resolve_base.sh
. "$REPO_ROOT/scripts/lib/resolve_base.sh" || exit 2

# ── the consumer probe ───────────────────────────────────────────────────────
# Derived from the binary's own surface, never from a list here. `pv census` is
# ONT-1; until it exists, `entity:` is a key nothing reads.
#
# WHICH pv (#3679). This probe used to take a bare `pv` off PATH. On intel the
# fleet pin was clobbered to 0.65.2 (paiml-implement#315), which has no census,
# so the probe answered "no consumer" about the RUNNER, not the tree, and the
# guard went RED on `contracts_anchored rose 3 -> 6`. It now resolves pv the way
# check_fleet_pv_shapes_gate.sh does (#3633): the fleet paths, the pin, and the
# version proved against the pin. A pv that cannot answer is UNMEASURED, and it
# names its version: a binary too old to judge is not evidence of absence.
#
# Two outcomes, printed on one line:
#   true                        the pinned pv lists `census`
#   unmeasured <reason> <what>  no-pin | no-binary | pin-mismatch | incapable
# There is no `false` outcome. The one binary that could say "absent" is a pinned
# pv without census, and that says only that the pin predates ONT-1.
FLEET_PV_CANDIDATES="${FLEET_PV_BIN:-"/opt/fleet-bin/bin/pv:$HOME/.cargo/bin/pv"}"
FLEET_PV_PIN_FILE="${FLEET_PV_PIN:-"$HOME/.config/fleet/pv.pin"}"
# resolve_fleet_pv -> the first candidate that is executable; rc 1 if none (same order as #3633)
resolve_fleet_pv() {
    local c IFS=:
    for c in $FLEET_PV_CANDIDATES; do
        if [ -x "$c" ]; then printf '%s\n' "$c"; return 0; fi
    done
    return 1
}
ont_consumer_probe() {
    local pin pvbin ver help_out
    [ -r "$FLEET_PV_PIN_FILE" ] || { printf 'unmeasured no-pin pin_file=%s\n' "$FLEET_PV_PIN_FILE"; return 0; }
    pin="$(<"$FLEET_PV_PIN_FILE")"; pin="${pin//[[:space:]]/}"
    pvbin="$(resolve_fleet_pv)" || { printf 'unmeasured no-binary pin=%s candidates=%s\n' "$pin" "$FLEET_PV_CANDIDATES"; return 0; }
    ver="$("$pvbin" --version 2>/dev/null || true)"; ver="${ver%%$'\n'*}"; ver="$(awk '{print $2}' <<<"$ver")"
    [ "$ver" = "$pin" ] || { printf 'unmeasured pin-mismatch pin=%s pv=%s version=%s\n' "$pin" "$pvbin" "${ver:-?}"; return 0; }
    help_out="$("$pvbin" --help 2>&1 || true)"
    # HERE-STRING, never a pipe into a quiet grep. Under pipefail the quiet grep
    # exits on the first match, the producer takes SIGPIPE and returns 141, and the
    # PIPELINE is 141 -- so the probe would read "no census" precisely when census
    # EXISTS. check_no_pipe_into_grep_q.sh caught this line; it then caught the
    # COMMENT that replaced it, because the scanner reads text and a warning that
    # spells the banned construct IS the banned construct as far as it can tell.
    if grep -qE '^[[:space:]]+census[[:space:]]' <<<"$help_out"; then printf 'true\n'; return 0; fi
    printf 'unmeasured incapable pv=%s version=%s (no census subcommand)\n' "$pvbin" "$ver"
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

# Σ IS THE REGISTRY, so Σ is what these two count (ONT-4b2, 2026-09-19). They used to grep the RUST for
# `EntityType::` and `impl Extractor` — two forms this codebase has never used: the entity types are Σ's
# `entity_types:` list and the extractors are free functions in `ontology/extract/`, declared in Σ's
# `extractors:` with `implemented:`. So both counters measured 0 while the baseline carried 1 and 1 from an
# older tree, and every `--write` since would have "ratcheted" a number nobody had measured. The names mean
# what Σ declares; that is now what they read. An absent Σ is 0 registered, not an error — 0 is the honest
# reading, and the rows in --self-test pin both directions.
count_entity_types() {
    local f="$REPO_ROOT/contracts/ontology.yaml"
    [ -f "$f" ] || { printf '0\n'; return 0; }
    { sed -n '/^entity_types:/,/^[a-z_]*:/p' "$f" | grep -cE '^[[:space:]]*-[[:space:]]*\{name:' || true; } | tr -d ' '
}
count_extractors() {
    local f="$REPO_ROOT/contracts/ontology.yaml"
    [ -f "$f" ] || { printf '0\n'; return 0; }
    { sed -n '/^extractors:/,/^[a-z_]*:/p' "$f" | grep -cE 'implemented:[[:space:]]*true' || true; } | tr -d ' '
}

# Keys under `ont` that OTHER gates own and this script does not measure: `formal_prose` (the sigma gate's
# prose-debt ratchet) and `legacy_unresolved_depends_on` (the relations gate's, PV-ONT-010). Both are read
# from this file by `lint/{sigma,relations}_gate.rs` and neither is computed here — so `--write` used to
# DELETE them, disarming two shrink-only ratchets in the act of updating a third. They ride through verbatim,
# the same rule `armed_gates` and `armed_shapes` already follow: what this script does not measure, it does
# not get to drop.
foreign_ont_keys() { # foreign_ont_keys FILE -> `    "k": v,` lines, in file order
    [ -f "$1" ] || return 0
    local key
    for key in formal_prose legacy_unresolved_depends_on; do
        { grep -E "\"$key\"[[:space:]]*:" "$1" || true; } | head -1 | sed 's/^[[:space:]]*/    /; s/,\{0,1\}[[:space:]]*$/,/'
    done
}

# ONT-7 (PMAT-4076): `contracts_without_valid_under` is a TOP-LEVEL key (the row's probe reads it there), owned
# and enforced by `lint/valid_under_gate.rs` (PV-ONT-016, shrink-only). This script does not measure it, so by
# the rule above it rides through `--write` verbatim; without this, `make ont-ratchet` would delete it and
# disarm the ratchet. Prints nothing when the key is absent, so measure() can omit it.
# PVL-001 EV-11 (PMAT-4166): the same holds for `command` and the two `pv lint` ratchets
# (`unpaired_theorem_modules`, `contracts_without_depends_on`), owned by `lint/ratchet_gates.rs` and moved only by
# `make lint-ratchet`. A top-level key is one at EXACTLY two spaces of indent (the layout this script and
# lint_ratchet.sh write); a nested key of the same name sits deeper and is never carried.
FOREIGN_TOP_KEYS="contracts_without_valid_under command unpaired_theorem_modules contracts_without_depends_on"
foreign_top_keys() { # foreign_top_keys FILE -> `  "k": v,` lines, in FOREIGN_TOP_KEYS order
    [ -f "$1" ] || return 0
    local key
    for key in $FOREIGN_TOP_KEYS; do
        { grep -E "^  \"$key\"[[:space:]]*:" "$1" || true; } | head -1 \
            | sed 's/^[[:space:]]*/  /; s/,\{0,1\}[[:space:]]*$/,/'
    done
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
    # ONE awk over every file, not three greps per file: the comparand is now
    # measured too (#3569), and 3 forks x ~1900 files x 2 trees was a minute.
    # Same four per-file predicates as the grep form it replaced.
    { find "$REPO_ROOT/contracts" -name '*.yaml' -type f -print0 2>/dev/null || true; } \
        | xargs -0 -r awk '
            function flush() { if (seen && !ent && ((ker && bind) || fil)) n++ }
            FNR == 1 { flush(); seen = 1; ent = ker = bind = fil = 0 }
            /^entity:/                                 { ent = 1 }
            /^kind:[[:space:]]*Kernel/                 { ker = 1 }
            /^[[:space:]]*binding:/                    { bind = 1 }
            /^[[:space:]]*(file|path|source_file):/    { fil = 1 }
            END { flush(); print n + 0 }' \
        | awk '{ t += $1 } END { print t + 0 }'
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
    { find "$REPO_ROOT/contracts" -name '*.yaml' -type f -print0 2>/dev/null || true; } \
        | xargs -0 -r awk 'FNR == 1 { inl = 0 } /^shape:/ { c++ } /^shapes:/ { inl = 1; next } inl && /^[^ ]/ { inl = 0 } inl && /^  - id:/ { c++ } END { print c + 0 }' \
        | awk '{ t += $1 } END { print t + 0 }'
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
# ONT-4c (v4.14): `readme` and `claude_md` carry the MEASURED claim sets `pv lint` checks — F-33 (the committed
# `verified_commands[]` must equal the live extraction) and F-34 (`withdrawn(HEAD) \ withdrawn(merge-base) ==
# set(merge-base) \ set(current)`). They are computed here, never typed: `--write` asks `pv lint --gate shapes`
# for the live sets and sets `withdrawn` = set(merge-base) \ set(current), both sorted, each key on ONE line.
# `--check` and the self-test carry the committed lines through untouched (they write nothing a gate reads).
# A nonzero `pv lint` exit is tolerated — a stale set IS a failing F-33, and the restamp exists to clear it —
# but output that is not a JSON object is refused: a set read from nothing would be ∅ and would pass.
measured_set_lines() { # measured_set_lines BASELINE_FILE -> `  "readme": {...},` lines (or nothing)
    local key
    if [ "${_ONT_MEASURE_SETS:-0}" != 1 ]; then
        [ -f "$1" ] || return 0
        for key in readme claude_md; do
            { grep -E "^  \"$key\"[[:space:]]*:" "$1" || true; } | head -1 | sed 's/,\{0,1\}[[:space:]]*$/,/'
        done
        return 0
    fi
    local pvbin out base cmp
    pvbin="${PV:-$(command -v pv 2>/dev/null || true)}"
    [ -n "$pvbin" ] || { printf 'NO-GO: no pv on PATH (set PV=) — the measured sets cannot be computed\n' >&2; return 2; }
    out="$(mktemp)"
    "$pvbin" lint "$REPO_ROOT/contracts" --gate shapes --format json >"$out" 2>/dev/null || true
    if ! jq -e 'type == "object" and (.readme.verified_commands | type) == "array" and (.claude_md.verified_commands | type) == "array"' "$out" >/dev/null 2>&1; then
        printf 'NO-GO: %s lint --gate shapes printed no readme/claude_md sets\n' "$pvbin" >&2
        rm -f "$out"; return 2
    fi
    cmp="$(mktemp)"
    base="$(git -C "$REPO_ROOT" merge-base HEAD origin/main 2>/dev/null || git -C "$REPO_ROOT" rev-parse --verify --quiet 'origin/main^{commit}' 2>/dev/null || true)"
    if [ -n "$base" ] && git -C "$REPO_ROOT" cat-file -e "$base:contracts/lint-baseline.json" 2>/dev/null; then
        git -C "$REPO_ROOT" show "$base:contracts/lint-baseline.json" >"$cmp"
    else
        printf '{}\n' >"$cmp"
    fi
    for key in readme claude_md; do
        printf '  "%s": %s,\n' "$key" "$(jq -c --slurpfile c "$cmp" --arg k "$key" '
            (.[$k].verified_commands | unique) as $cur
            | ((($c | first)[$k].verified_commands // []) | unique) as $was
            | {verified_commands: $cur, withdrawn: ($was - $cur)}' "$out")"
    done
    rm -f "$out" "$cmp"
}

# What `--write` keeps: the declarations and the foreign keys. No counter — a
# stored counter is a comparand the PR under test can rewrite (#3569).
decisions() { # prints the lint-baseline.json document
    local armed armed_shapes foreign sets
    armed="$(armed_gates_of "$BASELINE")" || return 2
    armed_shapes="$(armed_shapes_of "$BASELINE")" || return 2
    # ONT-4c (v4.14) F-33/F-34: readme/claude_md carry MEASURED claim sets, not decisions —
    # but they still belong in the baseline document decisions() writes.
    sets="$(measured_set_lines "$BASELINE")" || return 2
    foreign="$(foreign_ont_keys "$BASELINE" | sed '$ s/,$//')"
    printf '{\n  "_spec": "APR-RELEASE-001 §11.2 counters are MEASURED comparand->head by scripts/check_ont_ratchet.sh (#3569); this file holds decisions only",\n'
    printf '  "armed_gates": %s,\n' "$armed"
    [ -z "$armed_shapes" ] || printf '  "armed_shapes": %s,\n' "$armed_shapes"
    [ -z "$sets" ] || printf '%s\n' "$sets"
    # ONT-7 / EV-11 top-level ratchets are the Rust gates' foreign keys too: --write must not drop them.
    foreign_top_keys "$BASELINE"
    if [ -n "$foreign" ]; then printf '  "ont": {\n%s\n  }\n' "$foreign"; else printf '  "ont": {}\n'; fi
    printf '}\n'
}

# The comparand's measurement: the SAME measure(), run over the comparand's
# contracts/ extracted from the object store — never over a number on disk.
measure_comparand() { # measure_comparand SCRATCH_DIR -> JSON on stdout; rc 1 when unmeasurable
    local tree="$1" ref
    BASE_REF="" BASE_HOW=""
    if ! resolve_base HEAD || [ -z "$BASE_REF" ]; then
        printf 'FAIL  no comparand could be named for HEAD: the counters are UNMEASURED at the base, and that is not "unchanged".\n' >&2
        printf '      In CI: git fetch --no-tags --depth=2 origin +refs/heads/main:refs/remotes/origin/main\n' >&2
        return 1
    fi
    ref="$BASE_REF"
    git -C "$REPO_ROOT" archive --format=tar "$ref" -- contracts | tar -xf - -C "$tree" || {
        printf 'FAIL  could not extract contracts/ at %s\n' "$ref" >&2; return 1; }
    [ -d "$tree/contracts" ] || { printf 'FAIL  %s carries no contracts/\n' "$ref" >&2; return 1; }
    printf '  comparand   %s (%s)\n' "$(git -C "$REPO_ROOT" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")" "$BASE_HOW" >&2
    REPO_ROOT="$tree" BASELINE="$tree/contracts/lint-baseline.json" measure
}

measure() { # prints the JSON document
    local anchored shaped types extractors bindable consumer total armed armed_shapes shapes_line unarmed sets top_line
    armed="$(armed_gates_of "$BASELINE")" || return 2
    armed_shapes="$(armed_shapes_of "$BASELINE")" || return 2
    shapes_line=""
    [ -z "$armed_shapes" ] || shapes_line="$(printf '  "armed_shapes": %s,\n' "$armed_shapes")
"
    unarmed="$(count_shapes_unarmed "$BASELINE")" || return 2
    sets="$(measured_set_lines "$BASELINE")" || return 2
    [ -z "$sets" ] || sets="$sets
"
    top_line="$(foreign_top_keys "$BASELINE")"
    [ -z "$top_line" ] || top_line="$top_line
"
    anchored="$(count_anchored)"; shaped="$(count_shaped)"
    types="$(count_entity_types)"; extractors="$(count_extractors)"
    bindable="$(count_unanchored_bindable)"
    total="$({ find "$REPO_ROOT/contracts" -name '*.yaml' -type f 2>/dev/null || true; } | wc -l | tr -d ' ')"
    # `true`, or the JSON STRING "unmeasured" -- never `false` from a runner that cannot judge (#3679).
    # compare_against() skips the consumer rule on it, visibly; --write refuses to stamp it.
    [ -n "${ONT_PROBE:-}" ] || ONT_PROBE="$(ont_consumer_probe)"
    case "$ONT_PROBE" in true) consumer=true ;; *) consumer='"unmeasured"' ;; esac
    cat <<JSON
{
  "_spec": "APR-RELEASE-001 §11.2 — moves only through \`make ont-ratchet\` (ONT R-6)",
  "armed_gates": $armed,
${shapes_line}${sets}${top_line}  "ont": {
    "consumer_present": $consumer,
    "contracts_total": $total,
    "entity_types_registered": $types,
    "extractors_implemented": $extractors,
    "contracts_anchored": $anchored,
    "contracts_shaped": $shaped,
    "unanchored_but_bindable": $bindable,
$(foreign_ont_keys "$BASELINE")
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
        | sed -E "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"?//" \
        | sed -E 's/[^A-Za-z0-9._-].*$//'
}

self_test() {
    local t pass=0 fail=0
    # the self-test never asks pv: --write below carries the measured-set lines through (ONT-4c)
    local _ONT_MEASURE_SETS=0
    t="$(mktemp -d)"
    case "$t" in /tmp/*|/var/tmp/*|"${TMPDIR:-/nonexistent}"/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT WANT
        if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok    %-44s %s\n' "$1" "$3"
        else fail=$((fail+1)); printf '  FAIL  %-44s want=%s got=%s\n' "$1" "$3" "$2"; fi
    }
    printf 'check_ont_ratchet self-test\n'
    # The --write rows below test what --write PRESERVES, so they run against a MEASURED consumer. Left to
    # the host's real probe, an unconverged box makes --write refuse and every "preserved" row passes on an
    # untouched file (#3679). The probe's own rows set ONT_PROBE themselves.
    local ONT_PROBE=true
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
    row "--write stores NO measured counter (#3569)" "$(grep -cE '"(shapes_unarmed|contracts_anchored|contracts_shaped|unanchored_but_bindable|contracts_total|consumer_present)"' "$t/ws.json" || true)" "0"
    row "--write does not invent armed_shapes"     "$(grep -c '"armed_shapes"' "$t/w.json")" "0"
    printf '{\n  "armed_gates": ["validate"],\n  "readme": {"verified_commands":["a"],"withdrawn":[]},\n  "claude_md": {"verified_commands":[],"withdrawn":["b"]},\n  "ont": {}\n}\n' > "$t/sets.json"
    set +e
    BASELINE="$t/sets.json" main --write >/dev/null 2>&1
    set -e
    row "--write keeps the measured-set lines (ONT-4c)" "$(grep -cE '^  "(readme|claude_md)": \{' "$t/sets.json")" "2"
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
    # ONT-4b2: the two counters read Σ, and a key another gate owns survives --write.
    printf 'schema: ont-sigma-v1\nentity_types:\n  - {name: a, extractor: a, implemented: true}\n  - {name: b, extractor: b, implemented: false}\nextractors:\n  - {name: a, reader: x.rs, implemented: true}\n  - {name: b, reader: y.rs, implemented: false}\nreaders:\n  concepts: x\n' > "$t/sigma.yaml"
    mkdir -p "$t/repo/contracts"
    cp "$t/sigma.yaml" "$t/repo/contracts/ontology.yaml"
    row "entity types counted from Σ, not from a Rust form nobody writes" "$(REPO_ROOT="$t/repo" count_entity_types)" 2
    row "extractors counted from Σ's implemented: true" "$(REPO_ROOT="$t/repo" count_extractors)" 1
    printf '{\n  "armed_gates": ["validate"],\n  "ont": {\n    "formal_prose": 1464,\n    "legacy_unresolved_depends_on": 8\n  }\n}\n' > "$t/foreign.json"
    BASELINE="$t/foreign.json" measure > "$t/f.json"
    row "measure() keeps formal_prose (the sigma gate reads it)" "$(grep -c '"formal_prose": 1464' "$t/f.json")" 1
    row "measure() keeps legacy_unresolved_depends_on (the relations gate reads it)" "$(grep -c '"legacy_unresolved_depends_on": 8' "$t/f.json")" 1
    cp "$t/foreign.json" "$t/fw.json"
    set +e
    BASELINE="$t/fw.json" main --write >/dev/null 2>&1
    set -e
    row "--write keeps both foreign keys in place" "$(grep -cE '"formal_prose"|"legacy_unresolved_depends_on"' "$t/fw.json")" 2
    # ONT-7: the valid-under gate's top-level ratchet survives measure() and --write, and absence stays absent.
    printf '{\n  "armed_gates": ["validate"],\n  "contracts_without_valid_under": 386,\n  "ont": {\n    "formal_prose": 1\n  }\n}\n' > "$t/vu.json"
    BASELINE="$t/vu.json" measure > "$t/vum.json"
    row "measure() keeps contracts_without_valid_under (the valid-under gate reads it)" "$(grep -c '"contracts_without_valid_under": 386' "$t/vum.json")" 1
    set +e
    BASELINE="$t/vu.json" main --write >/dev/null 2>&1
    set -e
    row "--write keeps contracts_without_valid_under in place" "$(grep -c '"contracts_without_valid_under": 386' "$t/vu.json")" 1
    row "--write does not invent contracts_without_valid_under" "$(grep -c '"contracts_without_valid_under"' "$t/fw.json")" 0
    # EV-11 (PMAT-4166): `make lint-ratchet`'s three keys ride through --write too; a NESTED key of the same name does not.
    printf '{\n  "armed_gates": ["validate"],\n  "contracts_without_valid_under": 386,\n  "command": "make lint-ratchet",\n  "unpaired_theorem_modules": 130,\n  "contracts_without_depends_on": 278,\n  "ont": {\n    "formal_prose": 1\n  }\n}\n' > "$t/lr.json"
    set +e
    BASELINE="$t/lr.json" main --write >/dev/null 2>&1
    set -e
    row "--write keeps command + both lint ratchets in place" "$(grep -cE '^  "(command": "make lint-ratchet"|unpaired_theorem_modules": 130|contracts_without_depends_on": 278),$' "$t/lr.json")" 3
    printf '{\n  "armed_gates": ["validate"],\n  "ont": {\n    "command": "nested",\n    "formal_prose": 1\n  }\n}\n' > "$t/nest.json"
    row "a nested \"command\" is not carried to the top level" "$(foreign_top_keys "$t/nest.json" | grep -c '"command"')" 0
    row "a top-level \"command\" beside a nested one is carried once" "$(printf '{\n  "command": "x",\n  "ont": {\n    "command": "y"\n  }\n}\n' > "$t/both.json"; foreign_top_keys "$t/both.json" | tr '\n' '|')" '  "command": "x",|'
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import json;json.load(open('$t/vum.json'))" >/dev/null 2>&1 \
            && row "measure() with the valid-under key is valid JSON" ok ok || row "measure() with the valid-under key is valid JSON" bad ok
    fi
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import json,sys;json.load(open('$t/f.json'))" >/dev/null 2>&1 \
            && row "measure() with foreign keys is valid JSON" ok ok || row "measure() with foreign keys is valid JSON" bad ok
    fi
    # #3569 AC3 — the never-down check runs comparand -> head, in a scratch repo
    # whose origin/main is real. Both directions of every counter are proven.
    local r="$t/dir" base_sha
    mkdir -p "$r/contracts"
    git -C "$r" init -q -b main
    git -C "$r" config user.email t@t; git -C "$r" config user.name t
    git -C "$r" config commit.gpgsign false; git -C "$r" config core.hooksPath /dev/null
    printf '{\n  "armed_gates": ["validate"],\n  "ont": {}\n}\n' > "$r/contracts/lint-baseline.json"
    printf 'entity: kernel\n' > "$r/contracts/a1.yaml"
    printf 'entity: kernel\nshape: x\n' > "$r/contracts/a2.yaml"
    printf 'kind: Kernel\nbinding: x\n' > "$r/contracts/b1.yaml"
    printf 'file: x.rs\n' > "$r/contracts/b2.yaml"
    git -C "$r" add -A; git -C "$r" commit -qm base
    base_sha=$(git -C "$r" rev-parse HEAD)
    git -C "$r" update-ref refs/remotes/origin/main "$base_sha"
    git -C "$r" checkout -qb feat
    git -C "$r" commit -q --allow-empty -m "feat: a branch commit, so HEAD is not the origin/main tip (push shape)"
    dir_row() { # dir_row NAME WANT_RC — measures $r's working tree against origin/main
        local rc=0
        REPO_ROOT="$r" BASELINE="$r/contracts/lint-baseline.json" \
            ONT_PROBE=true main --check >"$t/dir.out" 2>&1 || rc=$?
        row "$1" "$rc" "$2"
        git -C "$r" checkout -q -- . ; git -C "$r" clean -qfd
    }
    dir_row "base -> head unchanged: GREEN" 0
    printf 'entity: kernel\n' > "$r/contracts/a3.yaml";                 dir_row "contracts_anchored RISES (consumer present): GREEN" 0
    rm "$r/contracts/a1.yaml";                                           dir_row "contracts_anchored FALLS: RED" 1
    printf 'shape: y\n' > "$r/contracts/s2.yaml";                        dir_row "contracts_shaped RISES: GREEN" 0
    printf 'entity: kernel\n' > "$r/contracts/a2.yaml"; printf 'entity: kernel\n' > "$r/contracts/a4.yaml"
                                                                         dir_row "contracts_shaped FALLS (anchored held): RED" 1
    printf 'file: y.rs\n' > "$r/contracts/b3.yaml";                      dir_row "unanchored_but_bindable RISES: RED" 1
    printf 'entity: kernel\nfile: x.rs\n' > "$r/contracts/b2.yaml";      dir_row "unanchored_but_bindable FALLS (anchored it): GREEN" 0
    # a committed restamp cannot move the comparand: the base is measured, not read
    rm "$r/contracts/a1.yaml"
    printf '{\n  "armed_gates": ["validate"],\n  "ont": {"contracts_anchored": 0}\n}\n' > "$r/contracts/lint-baseline.json"
    git -C "$r" commit -qam "drop an anchor and restamp the counter in the same commit"
    local rc=0
    REPO_ROOT="$r" BASELINE="$r/contracts/lint-baseline.json" \
        ONT_PROBE=true main --check >"$t/dir.out" 2>&1 || rc=$?
    row "RED: a fall committed WITH a restamped counter" "$rc" 1
    # PUSH SHAPE: HEAD is the origin/main tip, so merge-base(origin/main, HEAD) is HEAD
    # itself and a merge-base comparand would pass the fall vacuously (pvl-a, 19:55Z)
    git -C "$r" update-ref refs/remotes/origin/main HEAD
    rc=0
    REPO_ROOT="$r" BASELINE="$r/contracts/lint-baseline.json" \
        ONT_PROBE=true main --check >"$t/dir.out" 2>&1 || rc=$?
    row "RED: a push to main that drops an anchor is judged against its FIRST PARENT" "$rc" 1
    grep -q 'first parent of HEAD' "$t/dir.out"; row "push shape names its comparand as the first parent" "$?" 0
    printf 'entity: kernel\n' > "$r/contracts/a1.yaml"; git -C "$r" add -A; git -C "$r" commit -qm "restore the anchor"
    git -C "$r" update-ref refs/remotes/origin/main HEAD
    rc=0
    REPO_ROOT="$r" BASELINE="$r/contracts/lint-baseline.json" \
        ONT_PROBE=true main --check >"$t/dir.out" 2>&1 || rc=$?
    row "GREEN: a push to main that restores the anchor (control)" "$rc" 0
    git -C "$r" update-ref -d refs/remotes/origin/main
    rc=0
    REPO_ROOT="$r" BASELINE="$r/contracts/lint-baseline.json" \
        ONT_PROBE=true main --check >"$t/dir.out" 2>&1 || rc=$?
    row "RED: no comparand is UNMEASURED, not unchanged" "$rc" 1

    # THE DECIDING ROW: with no consumer, a rise in contracts_anchored is refused.
    printf '{"ont":{"consumer_present": false,"contracts_anchored": 0}}\n' > "$t/base.json"
    local rc
    set +e
    BASELINE="$t/base.json" _ONT_FORCE_ANCHORED=5 _ONT_FORCE_CONSUMER=false compare_against "$t/base.json" >/dev/null 2>&1
    rc=$?
    set -e
    row "anchored rises with NO consumer -> refused" "$rc" "1"
    # #3679: WHICH pv the consumer probe asks. Stubs only -- a real pv on this box would make the rows
    # pass here and vacuously everywhere else. `stale` has no census and sits FIRST on PATH (the intel
    # 0.65.2 condition); `fleet` has census and is the pinned binary. The PATH-probe this replaced reads
    # `stale` and says false, so reverting the resolution turns the first row RED.
    mkdir -p "$t/stale" "$t/fleet"
    printf '#!/bin/sh\ncase "$1" in --version) echo "pv 0.65.2 (stub)";; --help) printf "Commands:\\n  validate  V\\n";; esac\n' > "$t/stale/pv"
    printf '#!/bin/sh\ncase "$1" in --version) echo "pv 9.9.9 (stub)";; --help) printf "Commands:\\n  validate  V\\n  census    C\\n";; esac\n' > "$t/fleet/pv"
    chmod +x "$t/stale/pv" "$t/fleet/pv"
    printf '9.9.9\n' > "$t/pin.ok"; printf '0.65.2\n' > "$t/pin.old"; printf '1.0.0\n' > "$t/pin.other"
    PATH="$t/stale:$PATH" FLEET_PV_CANDIDATES="$t/fleet/pv" FLEET_PV_PIN_FILE="$t/pin.ok" ONT_PROBE='' \
        BASELINE="$t/armed.json" measure > "$t/pinned.json"
    row "stale PATH pv + pinned census pv -> consumer true" "$(field "$t/pinned.json" consumer_present)" "true"
    row "pinned pv without census -> unmeasured, names version" \
        "$(FLEET_PV_CANDIDATES="$t/stale/pv" FLEET_PV_PIN_FILE="$t/pin.old" ont_consumer_probe | awk '{print $1, $2, $4}')" \
        "unmeasured incapable version=0.65.2"
    row "no pin -> unmeasured no-pin" \
        "$(FLEET_PV_CANDIDATES="$t/fleet/pv" FLEET_PV_PIN_FILE="$t/nope" ont_consumer_probe | awk '{print $1, $2}')" "unmeasured no-pin"
    row "pin != binary version -> unmeasured pin-mismatch" \
        "$(FLEET_PV_CANDIDATES="$t/fleet/pv" FLEET_PV_PIN_FILE="$t/pin.other" ont_consumer_probe | awk '{print $1, $2}')" "unmeasured pin-mismatch"
    row "no fleet binary -> unmeasured no-binary" \
        "$(FLEET_PV_CANDIDATES="$t/none/pv" FLEET_PV_PIN_FILE="$t/pin.ok" ont_consumer_probe | awk '{print $1, $2}')" "unmeasured no-binary"
    PATH="$t/stale:$PATH" FLEET_PV_CANDIDATES="$t/stale/pv" FLEET_PV_PIN_FILE="$t/pin.old" ONT_PROBE='' \
        BASELINE="$t/armed.json" measure > "$t/unm.json"
    row "an unmeasured consumer is never written as false" "$(field "$t/unm.json" consumer_present)" "unmeasured"
    set +e
    BASELINE="$t/base.json" ONT_PROBE='unmeasured no-pin stub' _ONT_FORCE_ANCHORED=5 _ONT_FORCE_CONSUMER='"unmeasured"' \
        compare_against "$t/base.json" > "$t/cmp.out" 2>&1
    rc=$?
    set -e
    row "anchored rises, consumer UNMEASURED -> not judged (rc 0)" "$rc" "0"
    row "...and says UNMEASURED, never silent" "$(grep -c '^UNMEASURED consumer probe' "$t/cmp.out")" "1"
    # #3569: --write stores DECISIONS only, so an unmeasured probe has nothing to stamp and is no reason
    # to refuse. It writes, and consumer_present stays out of the file (#3679's worry cannot arise).
    cp "$t/armed.json" "$t/wu.json"
    set +e
    BASELINE="$t/wu.json" ONT_PROBE='unmeasured no-pin stub' main --write >/dev/null 2>&1
    rc=$?
    set -e
    row "--write under an UNMEASURED probe writes decisions (rc 0)" "$rc" "0"
    row "...and stamps no consumer_present" "$(grep -c '"consumer_present"' "$t/wu.json" || true)" "0"
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

    # UNMEASURED is fleet state, not a verdict (#3679, the #3633 convention): the runner's pv cannot
    # say whether the consumer exists, so the consumer rule is not judged here -- and says so. The
    # ratchet directions below still are.
    if [ "$consumer_now" = "unmeasured" ]; then
        printf 'UNMEASURED consumer probe: %s -- the anchored-rise rule is not judged on this runner; fleet state, not a pass\n' "${ONT_PROBE#unmeasured }"
    fi
    # THE GATE. An anchor nothing reads is decoration, and a counter that rises on
    # decoration is the theater §11 exists to stop. Refuse the rise, name the fix.
    if [ "$consumer_now" = "false" ] && [ "${anchored_now:-0}" -gt "${anchored_was:-0}" ]; then
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
        --print) measure; return $? ;;
        --write)
            mkdir -p "$(dirname "$BASELINE")"
            # Never `decisions > "$BASELINE"`: the shell truncates the file BEFORE
            # decisions() reads armed_gates out of it, so the declaration read back
            # empty and was rewritten as [].
            local tmp
            tmp="$(mktemp "$BASELINE.XXXXXX")"
            _ONT_MEASURE_SETS="${_ONT_MEASURE_SETS:-1}" decisions > "$tmp" || { rm -f "$tmp"; return 2; }
            mv "$tmp" "$BASELINE"
            printf 'wrote %s (decisions only; the counters are measured, #3569)\n' "${BASELINE#"$REPO_ROOT"/}"
            return 0 ;;
        --check|"") ;;
        *) printf 'usage: %s [--check|--print|--write|--self-test]\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    printf '== ONT ratchet (APR-RELEASE-001 §11.2), comparand tree -> working tree (#3569) ==\n'
    if [ ! -f "$BASELINE" ]; then
        printf 'NO-GO: %s does not exist. It declares armed_gates; without it\n' "${BASELINE#"$REPO_ROOT"/}" >&2
        printf 'the measurement has no arming to read and this guard would pass vacuously.\n' >&2
        return 2
    fi
    local scratch base_json rc=0
    rmscratch() { case "${1:-}" in "${TMPDIR:-/tmp}"/tmp.*|/tmp/tmp.*) rm -rf -- "$1" ;; *) : ;; esac; }
    scratch="$(mktemp -d)"; base_json="$scratch/base.json"
    mkdir -p "$scratch/tree"
    if ! measure_comparand "$scratch/tree" > "$base_json"; then
        rmscratch "$scratch"
        printf 'FAILED: §11.2 — the comparand could not be measured.\n'
        return 1
    fi
    printf '  base: %s\n' "$(sed -n '/"ont"/,/^  }/p' "$base_json" | tr -d ' \n')"
    printf '  head: %s\n' "$(measure | sed -n '/"ont"/,/^  }/p' | tr -d ' \n')"
    compare_against "$base_json" || rc=1
    rmscratch "$scratch"
    if [ "$rc" = 0 ]; then printf 'PASS\n'; return 0; fi
    printf 'FAILED: §11.2 — the ratchet turns one way, measured comparand -> head.\n'
    return 1
}

main "$@"
