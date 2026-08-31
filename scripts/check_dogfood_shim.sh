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
# Gate invocations, DERIVED from the canonical runner. A hardcoded list would be
# the same defect one level down: the runner grows a gate, the list does not,
# and the shim may then quietly acquire the new one.
#
# NOT LINE-ANCHORED, and that is a fix rather than a preference. The first
# version matched `^[[:space:]]*(mark|gate) …`, which omitted every invocation
# that follows a `;`, a `&&`, or a `then` on the same line — 11 of them in the
# runner — so both 3b rows below could be evaded by writing the gate mid-line.
#
# It also has to see DYNAMIC call sites. The runner's declared-gate section
# calls `mark "$dg_name" …` three times; a name-only extraction drops those, and
# a shim that re-implemented that section would have invoked a gate while
# reporting none. So there are two extractions:
#
#   gate_calls  every command-position `mark`/`gate` invocation, literal or
#               dynamic — this is what "does the shim invoke a gate" means.
#   gate_names  the subset whose argument is a literal gate name — this is the
#               vocabulary the textual 3b row greps for.
#
# The leading alternation is what keeps a MENTION from reading as a call: in
# `# mark pv-contracts PASS` and in `"a shim that runs a gate"` the character
# before the keyword is neither the line start nor a shell separator.
# Comments are removed FIRST, quote-aware. Without that the `\bthen\b`
# alternative matches English: the runner's own comment "then gate on CB-200
# specifically" produced a phantom gate named `on`, and a phantom name is a
# grep the shim can never satisfy.
gate_calls() {
    python3 - "$1" <<'PY'
import re, sys

CALL = re.compile(r"(?:^|[;&|(){}]|\bthen\b|\belse\b|\bdo\b)\s*(?:mark|gate)\s+(\S+)")


def strip_comment(line):
    out, i, n, q = [], 0, len(line), ""
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            out.append(line[i:i + 2]); i += 2; continue
        if q:
            if c == q:
                q = ""
            out.append(c); i += 1; continue
        if c in ("'", '"'):
            q = c; out.append(c); i += 1; continue
        if c == "#":
            prev = out[-1] if out else ""
            if not out or prev.isspace() or prev in "(;&|":
                break
            out.append(c); i += 1; continue
        out.append(c); i += 1
    return "".join(out)


for raw in open(sys.argv[1], encoding="utf-8", errors="replace"):
    for m in CALL.finditer(strip_comment(raw.rstrip("\n"))):
        print(m.group(1))
PY
}

gate_names() {
    gate_calls "$1" | grep -E '^[a-z0-9][a-z0-9-]*$' | LC_ALL=C sort -u
}

# Does this file INVOKE a gate? Structural, and the primary check. Dynamic call
# sites are reported as the literal text of their argument, so `mark "$dg_name"`
# is visible as `"$dg_name"` rather than vanishing.
shim_invokes_gate() {
    gate_calls "$1" | LC_ALL=C sort -u | tr '\n' ' '
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
    elif ! grep -qi 'REVISIT TRIGGER' <<< "$out" ; then
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
    if grep -q 'FAIL  3b' <<< "$out" ; then
        printf 'ok    row 2 a shim that invokes a gate is REJECTED\n'
    else
        printf 'FAIL  row 2 a shim invoking `mark pv-contracts` was accepted\n'; fails=1
    fi

    # Mutation B: the shim grows past the cap.
    cp "$td/good.sh" "$td/long.sh"
    for _ in $(seq 1 "$((SHIM_MAX_LINES + 5))"); do printf '# padding\n' >> "$td/long.sh"; done
    out=$(check_shim_file "$td/long.sh" 2>&1)
    if grep -q 'FAIL  3a' <<< "$out" ; then
        printf 'ok    row 3 a shim over the line cap is REJECTED\n'
    else
        printf 'FAIL  row 3 an over-cap shim was accepted\n'; fails=1
    fi

    # Mutation C: the shim degrades instead of failing closed.
    printf '#!/usr/bin/env bash\nexit 0\n' > "$td/soft.sh"
    out=$(check_shim_file "$td/soft.sh" 2>&1)
    if grep -q 'FAIL  3c' <<< "$out" ; then
        printf 'ok    row 4 a shim that exits 0 without a checkout is REJECTED\n'
    else
        printf 'FAIL  row 4 a silently-succeeding shim was accepted\n'; fails=1
    fi

    # Mutation D: it fails, but says nothing useful.
    printf '#!/usr/bin/env bash\nexit 2\n' > "$td/mute.sh"
    out=$(check_shim_file "$td/mute.sh" 2>&1)
    if grep -q 'FAIL  3c' <<< "$out" ; then
        printf 'ok    row 5 a shim that fails MUTELY is REJECTED\n'
    else
        printf 'FAIL  row 5 a mute failing shim was accepted\n'; fails=1
    fi

    # Mutation E: the gate is invoked MID-LINE. The line-anchored extractor this
    # replaces saw nothing here, so both 3b rows were evadable by writing `;`.
    cp "$td/good.sh" "$td/midline.sh"
    printf 'true; mark pv-contracts PASS "ported"\n' >> "$td/midline.sh"
    out=$(check_shim_file "$td/midline.sh" 2>&1)
    if grep -q 'FAIL  3b' <<< "$out" ; then
        printf 'ok    row 6 a MID-LINE `; mark pv-contracts` is REJECTED\n'
    else
        printf 'FAIL  row 6 a mid-line gate invocation was accepted\n'; fails=1
    fi

    # Mutation F: the gate name is a VARIABLE. The runner itself does this three
    # times (`mark "$dg_name" …`), so a name-only extraction is blind to the one
    # section a re-implementing shim would most plausibly copy.
    cp "$td/good.sh" "$td/dynamic.sh"
    printf 'mark "$dg_name" FAIL "declared but absent"\n' >> "$td/dynamic.sh"
    out=$(check_shim_file "$td/dynamic.sh" 2>&1)
    if grep -q 'FAIL  3b' <<< "$out" ; then
        printf 'ok    row 7 a DYNAMIC `mark "$name"` invocation is REJECTED\n'
    else
        printf 'FAIL  row 7 a dynamic gate invocation was accepted\n'; fails=1
    fi

    # Row 8 — and the widened regex must still let the shim describe itself. A
    # keyword in prose or in a comment is not a call site; demanding its absence
    # would forbid the shim from explaining what it is, which is a false alarm,
    # not a gate.
    cp "$td/good.sh" "$td/prose.sh"
    {
        printf '# mark tests PASS -- this is prose about what the runner does\n'
        printf '# it invokes no gate and does not gate anything itself\n'
        # The live false positive: `\bthen\b` matching English inside a comment.
        printf '# build the index first, then gate on CB-200 specifically\n'
    } >> "$td/prose.sh"
    if check_shim_file "$td/prose.sh" >/dev/null 2>&1; then
        printf 'ok    row 8 prose naming `mark`/`gate` is NOT a call site\n'
    else
        printf 'FAIL  row 8 a shim describing itself was rejected:\n'
        check_shim_file "$td/prose.sh" | sed 's/^/        /'
        fails=1
    fi

    # Row 9 — VACUITY. An extraction that yields nothing makes both 3b rows
    # pass on any shim at all. The number is MEASURED and printed rather than
    # asserted: an earlier handoff quoted "61 gate names" from a regex run over
    # both forked copies, and the real extraction is a different number.
    local n_names n_calls
    n_names=$(gate_names "$CANON_RUNNER" | grep -c .)
    n_calls=$(gate_calls "$CANON_RUNNER" | grep -c .)
    if [ "$n_names" -ge 20 ] && [ "$n_calls" -ge "$n_names" ]; then
        printf 'ok    row 9 canon yields %s distinct gate names over %s call sites\n' \
            "$n_names" "$n_calls"
    else
        printf 'FAIL  row 9 extraction yielded %s names / %s call sites — a 3b row that\n' \
            "$n_names" "$n_calls"
        printf '         greps an empty vocabulary accepts every shim.\n'
        fails=1
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

# 1b — no SECOND fleet/install sweep. (PARITY-011, aprender#2678)
#
# Row 1 stops a second `dogfood.sh`. It does NOT stop the same methodology
# reappearing under a different NAME, which is how the 0.64.0 four-host sweep
# came to be hand-rolled: no tracked procedure existed, so one was invented
# outside the protocol — in a protocol whose own rule is "Never dogfood by hand".
#
# The construct, not the filename: a script that installs the published crate
# AND probes the result is a fleet sweep whatever it is called. The runner and
# the release skill are the only places that may do both.
#
# UNIVERSE INCLUDES UNTRACKED. A `git ls-files`-only scan lets a brand-new copy
# pass until the moment it is committed — SHIM-2644-03's exact shape, where an
# untracked copy passed a check whose whole purpose was catching a second copy.
# The signature is narrow ON PURPOSE. A first, broader version flagged
# cascade-publish.sh (which installs to VERIFY A PUBLISH — a different job),
# check_facade_compat.sh, and this file itself (its own pattern string matched).
# Three false positives on the first run, which is what a construct ban costs
# when the construct is described too loosely.
#
# What actually distinguishes a FLEET SWEEP from a legitimate install: it
# writes a per-host RECEIPT. That is the duplicated methodology — not
# installing, and not probing, but producing the evidence artifact that
# scripts/dogfood.sh and Gate 12 already own.
SWEEP_ALLOWED="scripts/dogfood.sh scripts/check_multiplatform_dogfood.sh scripts/check_dogfood_shim.sh"
sweep_hits=""
while IFS= read -r f; do
    [ -f "$f" ] || continue
    case " $SWEEP_ALLOWED " in *" $f "*) continue ;; esac
    # COMMENTS ARE NOT INVOCATIONS. scripts/bench_host_receipt.sh documents
    # "RUN THIS ON THE HOST, AFTER `cargo install aprender`" in its header and
    # installs nothing — it was flagged as a second sweep on the strength of
    # that sentence. Found when PARITY-003 and PARITY-011 met in the cumulative
    # stack head: each branch was green alone.
    grep -q 'cargo install aprender' <<< "$(grep -vE '^[[:space:]]*#' "$f" 2>/dev/null)" || continue
    # ...and writes a per-host receipt. Both halves, or it is not a sweep.
    grep -qE 'evidence/dogfood|install_rc' <<< "$(grep -vE '^[[:space:]]*#' "$f" 2>/dev/null)" || continue
    sweep_hits="$sweep_hits $f"
done < <(
    { git ls-files 'scripts/*.sh' 'crates/*/scripts/*.sh' 2>/dev/null
      find scripts -maxdepth 2 -type f -name '*.sh' 2>/dev/null
    } | LC_ALL=C sort -u
)
if [ -n "$sweep_hits" ]; then
    printf 'FAIL  1b a SECOND fleet/install sweep exists:%s\n' "$sweep_hits"
    printf '         One methodology, one runner. Add the step to scripts/dogfood.sh\n'
    printf '         or to the release skill; do not grow a parallel sweep (#2678).\n'
    rc=1
else
    printf 'ok    1b no second fleet/install sweep\n'
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
