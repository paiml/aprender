#!/usr/bin/env bash
# check_dogfood_shim.sh — one runner, one prose, and the user-scope copy stays a
# shim.
#
# WHY THIS EXISTS
# ---------------
# #2640 merged two 1170-line copies of the dogfood protocol into one. The
# user-scope copy at ~/.claude/skills/dogfood/ became a ~50-line shim that execs
# the repo runner. But a shim is still a file at user scope that can drift — it
# is the shadow with extra steps unless shim-ness is GATED. #2361 is what that
# looks like: ~/.claude/skills/dogfood/ shadowed the repo's release-certifying
# skill, so hardening the repo edited a file that never ran, and nothing warned.
#
# The asymmetry that makes this sound: the CAP lives here, in the repo, on
# protected main, where changing it needs a PR and review. The FILE it caps lives
# at user scope, where it does not. "Any state the author writes and the gate
# reads can be moved in the same commit" — so the state is deliberately not
# co-located with the thing it constrains.
#
# WHAT IT ASSERTS
#   1  exactly one dogfood.sh in this repo, tracked by git            (AC-1/AC-7)
#   2  the canonical prose is in the repo and carries an EXPLICIT name:
#      without it the skill takes its name from its directory, collides with a
#      user-scope skill, and never appears in the session listing (#2332)
#   3  a user-scope dogfood.sh, if present, is a SHIM:
#        3a  <= SHIM_MAX_LINES
#        3b  invokes NO gate of the protocol, and mentions no gate name
#        3c  FAILS CLOSED when the aprender checkout is absent — never silently
#            falls back, never degrades to a local copy
#   4  no user-scope SKILL.md shadows the repo's dogfood skill
#
#   bash scripts/check_dogfood_shim.sh              # check
#   bash scripts/check_dogfood_shim.sh --self-test  # case table
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# THE CAP. 70, and the number is chosen rather than guessed: the shim as written
# is 49 lines, of which 32 are the rationale and the revisit trigger. 70 leaves
# ~20 lines of headroom so prose can be improved without touching this file —
# and no more. The smallest gate in the protocol is ~12 lines with its rationale
# plus a call site, so re-inlining even ONE gate spends the whole headroom, and
# raising the cap to make room is a diff in this repo, in a PR, which is the
# point. Against the 1170-line protocol the cap is ~17x smaller.
SHIM_MAX_LINES=70

USER_SHIM="${DOGFOOD_USER_SHIM:-$HOME/.claude/skills/dogfood/dogfood.sh}"
USER_SKILL="${DOGFOOD_USER_SKILL:-$HOME/.claude/skills/dogfood/SKILL.md}"
CANON_SKILL="$REPO_ROOT/.claude/skills/dogfood/SKILL.md"
CANON_RUNNER="$REPO_ROOT/scripts/dogfood.sh"

# ---------------------------------------------------------------------------
# Gate names, DERIVED from the canonical runner. A hardcoded list would be the
# same defect one level down: the runner grows a gate, the list does not, and
# the shim may then quietly acquire the new one.
gate_names() {
    grep -oE '^[[:space:]]*(mark|gate) [a-z0-9][a-z0-9-]*' "$1" \
        | awk '{print $2}' | LC_ALL=C sort -u
}

# Does this file INVOKE a gate? Structural, and the primary check.
shim_invokes_gate() {
    gate_names "$1" | tr '\n' ' '
}

# Does this file MENTION a gate name as literal text? Secondary, and restricted
# to the HYPHENATED names on purpose. The single-word names (fmt, test, clippy,
# coverage, contracts, renacer, reachability) are ordinary English and appear in
# any honest description of what the runner does — demanding their absence would
# forbid the shim from explaining itself and would be a false alarm, not a gate.
# The hyphenated names are unambiguous protocol identifiers.
shim_mentions_gate() {
    local f="$1" n hits=""
    while read -r n; do
        case "$n" in
            *-*) ;;
            *) continue ;;
        esac
        if grep -qF -- "$n" "$f" 2>/dev/null; then hits="$hits $n"; fi
    done < <(gate_names "$CANON_RUNNER")
    printf '%s' "$hits"
}

# ---------------------------------------------------------------------------
check_shim_file() {
    # $1 = path to the candidate shim. Prints rows; returns 1 on any failure.
    local f="$1" rc=0 n invoked mentioned out srrc
    n=$(wc -l < "$f")
    if [ "$n" -le "$SHIM_MAX_LINES" ]; then
        printf 'ok    3a shim is %s lines (cap %s)\n' "$n" "$SHIM_MAX_LINES"
    else
        printf 'FAIL  3a shim is %s lines, cap is %s — this is a copy re-forming.\n' "$n" "$SHIM_MAX_LINES"
        printf '         The protocol lives in scripts/dogfood.sh. Nothing else may hold it.\n'
        rc=1
    fi

    invoked=$(shim_invokes_gate "$f")
    if [ -z "${invoked// /}" ]; then
        printf 'ok    3b shim invokes no gate of the protocol\n'
    else
        printf 'FAIL  3b shim INVOKES protocol gate(s):%s\n' " $invoked"
        printf '         A shim that runs a gate is a second runner with a smaller diff.\n'
        rc=1
    fi

    mentioned=$(shim_mentions_gate "$f")
    if [ -z "${mentioned// /}" ]; then
        printf 'ok    3b shim names no protocol gate\n'
    else
        printf 'FAIL  3b shim names protocol gate(s):%s\n' "$mentioned"
        rc=1
    fi

    # 3c FAIL CLOSED. Behavioural, not textual: point it at a checkout that does
    # not exist and require a loud non-zero exit that names the revisit trigger.
    out=$(DOGFOOD_CANON_ROOT="$f.no-such-checkout" bash "$f" --version 2>&1); srrc=$?
    if [ "$srrc" -eq 0 ]; then
        printf 'FAIL  3c shim EXITED 0 with the aprender checkout absent — it fell back to\n'
        printf '         something. A shim that degrades instead of failing is the shadow.\n'
        rc=1
    elif ! printf '%s' "$out" | grep -qi 'REVISIT TRIGGER'; then
        printf 'FAIL  3c shim failed (rc=%s) but did not name the REVISIT TRIGGER. It must\n' "$srrc"
        printf '         say WHAT the reader is looking at, or the next person patches around it.\n'
        rc=1
    else
        printf 'ok    3c shim fails closed (rc=%s) and names the revisit trigger\n' "$srrc"
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
self_test() {
    local td fails=0 out
    td=$(mktemp -d) || return 2

    # A good shim: short, gateless, fails closed.
    cp "$REPO_ROOT/scripts/.dogfood-shim-reference.sh" "$td/good.sh" 2>/dev/null \
        || { printf 'FAIL  self-test needs scripts/.dogfood-shim-reference.sh\n'; rm -rf "${td:?}"; return 1; }

    if check_shim_file "$td/good.sh" >/dev/null 2>&1; then
        printf 'ok    row 1 the reference shim passes every shim assertion\n'
    else
        printf 'FAIL  row 1 the reference shim does NOT pass:\n'
        check_shim_file "$td/good.sh" | sed 's/^/        /'
        fails=1
    fi

    # Mutation A: a gate name is added.
    cp "$td/good.sh" "$td/gate.sh"
    printf 'mark pv-contracts PASS "ported"\n' >> "$td/gate.sh"
    out=$(check_shim_file "$td/gate.sh" 2>&1)
    if printf '%s' "$out" | grep -q 'FAIL  3b'; then
        printf 'ok    row 2 a shim that invokes a gate is REJECTED\n'
    else
        printf 'FAIL  row 2 a shim invoking `mark pv-contracts` was accepted\n'; fails=1
    fi

    # Mutation B: the shim grows past the cap.
    cp "$td/good.sh" "$td/long.sh"
    for _ in $(seq 1 "$((SHIM_MAX_LINES + 5))"); do printf '# padding\n' >> "$td/long.sh"; done
    out=$(check_shim_file "$td/long.sh" 2>&1)
    if printf '%s' "$out" | grep -q 'FAIL  3a'; then
        printf 'ok    row 3 a shim over the line cap is REJECTED\n'
    else
        printf 'FAIL  row 3 an over-cap shim was accepted\n'; fails=1
    fi

    # Mutation C: the shim degrades instead of failing closed.
    printf '#!/usr/bin/env bash\nexit 0\n' > "$td/soft.sh"
    out=$(check_shim_file "$td/soft.sh" 2>&1)
    if printf '%s' "$out" | grep -q 'FAIL  3c'; then
        printf 'ok    row 4 a shim that exits 0 without a checkout is REJECTED\n'
    else
        printf 'FAIL  row 4 a silently-succeeding shim was accepted\n'; fails=1
    fi

    # Mutation D: it fails, but says nothing useful.
    printf '#!/usr/bin/env bash\nexit 2\n' > "$td/mute.sh"
    out=$(check_shim_file "$td/mute.sh" 2>&1)
    if printf '%s' "$out" | grep -q 'FAIL  3c'; then
        printf 'ok    row 5 a shim that fails MUTELY is REJECTED\n'
    else
        printf 'FAIL  row 5 a mute failing shim was accepted\n'; fails=1
    fi

    rm -rf "${td:?}"
    return "$fails"
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    --self-test) self_test; exit $? ;;
esac

cd "$REPO_ROOT" || exit 2
rc=0
printf -- '--- one dogfood runner (#2640) --------------------------------------\n'

printf 'case table\n'
if self_test; then :; else rc=1; fi
printf '\n'

# 1 — exactly one dogfood.sh, tracked.
RUNNERS=$(git ls-files | grep -E '(^|/)dogfood\.sh$' | LC_ALL=C sort)
N_RUN=$(printf '%s\n' "$RUNNERS" | grep -c . )
if [ "$N_RUN" -eq 1 ] && [ "$RUNNERS" = "scripts/dogfood.sh" ]; then
    printf 'ok    1  exactly one tracked dogfood.sh: %s\n' "$RUNNERS"
else
    printf 'FAIL  1  expected exactly one tracked dogfood.sh at scripts/dogfood.sh, found %s:\n' "$N_RUN"
    printf '%s\n' "$RUNNERS" | sed 's/^/         /'
    rc=1
fi

# 2 — canonical prose, in the repo, with an explicit name.
if [ ! -f "$CANON_SKILL" ]; then
    printf 'FAIL  2  no canonical prose at .claude/skills/dogfood/SKILL.md — the protocol\n'
    printf '         description would live only at user scope, where it is not diffable.\n'
    rc=1
elif ! grep -qE '^name:[[:space:]]*dogfood[[:space:]]*$' "$CANON_SKILL"; then
    printf 'FAIL  2  .claude/skills/dogfood/SKILL.md has no explicit `name: dogfood`.\n'
    printf '         Without it the skill is named after its directory, collides with a\n'
    printf '         user-scope skill, and never appears in the session listing (#2332).\n'
    rc=1
else
    printf 'ok    2  canonical prose at .claude/skills/dogfood/SKILL.md, name: dogfood\n'
fi

# 3 — the user-scope copy.
if [ ! -e "$USER_SHIM" ]; then
    printf 'ok    3  no user-scope dogfood.sh on this host (the ideal state)\n'
else
    printf 'note  3  user-scope copy present at %s\n' "$USER_SHIM"
    if check_shim_file "$USER_SHIM"; then :; else rc=1; fi
    # 3d — and it is the reviewed shim, byte for byte. The shim's TEXT is
    # tracked here as scripts/.dogfood-shim-reference.sh, so even the pointer is
    # diffable. Without this row the previous three only bound the shim's SHAPE,
    # and a rewritten-but-still-short shim would pass.
    if cmp -s "$USER_SHIM" "$REPO_ROOT/scripts/.dogfood-shim-reference.sh"; then
        printf 'ok    3d user-scope shim is byte-identical to the reviewed reference\n'
    else
        printf 'FAIL  3d user-scope shim DIFFERS from scripts/.dogfood-shim-reference.sh:\n'
        diff -u "$REPO_ROOT/scripts/.dogfood-shim-reference.sh" "$USER_SHIM" \
            | head -20 | sed 's/^/         /'
        printf '         Redeploy it:  cp scripts/.dogfood-shim-reference.sh %s\n' "$USER_SHIM"
        rc=1
    fi
fi

# 4 — nothing at user scope may shadow the repo skill.
if [ -e "$USER_SKILL" ]; then
    printf 'FAIL  4  a user-scope SKILL.md still exists at %s.\n' "$USER_SKILL"
    printf '         A user-scope skill WINS over the repo one, so the reviewed copy would\n'
    printf '         never appear in the session listing. Delete it or rename the file.\n'
    rc=1
else
    printf 'ok    4  no user-scope SKILL.md shadowing the repo dogfood skill\n'
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  one runner, one prose, and the user-scope copy is a gated shim.\n'
else
    printf 'FAIL  see rows above (#2640).\n'
fi
exit "$rc"
