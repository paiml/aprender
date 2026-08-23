#!/usr/bin/env bash
# check_verifier_pinning.sh — the verifier-pinning rule, ENFORCED.
#
# THE RULE IS NOT RESTATED HERE. It is stated once, in scripts/verifier_pin.sh,
# under the heading "THE RULE" — read it there. A rule written down twice is two
# rules that can disagree, which is the thesis of the ticket this gate closes
# (#2640): one protocol, two copies, nine silent divergences. This file is the
# rule's ENFORCEMENT, not a second copy of its text.
#
# WHY A GATE AND NOT A PARAGRAPH
# ------------------------------
# The rule already had FIVE independent ad-hoc rediscoveries before anyone
# noticed it was one rule (PMAT_BIN, scripts/pv_bin.sh, scripts/apr_bin.sh,
# aprender#2384, APR-BENCH-RFC-001). Five is the evidence that a rule merely
# stated is documentation. #2640 merged the two dogfood runners into one; this
# is what stops the sixth tool re-discovering the rule in a sixth copy.
#
# TWO PARTS, because presence is not behaviour:
#
#   PART 1 (static)      no bare pv/pmat/apr in command position in the runner
#                        or its pin library. Mutation: reintroduce a bare `pmat`
#                        call -> RED.
#   PART 2 (behavioural) the two pins are EXERCISED and must select something
#                        other than what PATH offers. Mutations: delete the
#                        PMAT_BIN self-referential branch -> RED; bypass
#                        pv_bin.sh to a PATH pv -> RED. Part 1 alone would pass
#                        on a pin that resolves to the wrong binary.
#
# THE UNIVERSE, and why it is these three tokens
# ----------------------------------------------
# VERIFIERS = pv, pmat, apr — exactly the tools for which THIS REPO ships a pin
# (scripts/pv_bin.sh, verifier_pin_pmat, scripts/apr_bin.sh). bashrs and
# probador are verifiers too and are NOT listed: no pin exists for them, and the
# rule's second clause is "where the repo does not pin, report" — which the
# runner does. Adding them here would demand a pin that does not exist and make
# the gate unfixable. The day one ships, add the token here.
#
# What counts as an invocation: the token in COMMAND POSITION after comments and
# quoted strings are removed. A bare mention in a comment or a string is NOT an
# invocation — the runner's own prose says "`pv lint <DIR>` is a real gate" and
# marks a row "pv is not pinned in this repo", and both must stay legal. The
# distinction is the whole difficulty, so it ships a case table (--self-test)
# rather than a reviewed regex: the apr-invocation patterns in this repo were
# wrong FIVE times and every one was caught by a table, none by review.
#
#   bash scripts/check_verifier_pinning.sh              # check
#   bash scripts/check_verifier_pinning.sh --self-test  # the case table only
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The files that decide a release and are therefore in scope.
SCOPE="scripts/dogfood.sh scripts/verifier_pin.sh"

# ---------------------------------------------------------------------------
# The scanner. Reports "<file>:<line>: <token>" for each bare invocation.
#
# It tokenises rather than regexing the raw line, because a regex over raw text
# gets `pmat-verify` wrong: \bpmat\b MATCHES inside `pmat-verify`, since `-` is a
# non-word character. Token equality does not. Quoted spans collapse to a single
# @Q placeholder rather than being deleted, so the ARITY of wrappers such as
# `run_to <log> <cmd...>` is preserved and `run_to "$LOG" pmat query` is still
# seen as pmat in command position.
scan() {
    python3 - "$@" <<'PY'
import re, sys

VERIFIERS = {"pv", "pmat", "apr"}
# Tokens after which the next word is a command.
OPENERS = {"", ";", "&&", "||", "|", "|&", "(", "$(", "((", "{", "}", ")",
           "if", "then", "else", "elif", "while", "until", "do", "!",
           "env", "exec", "nohup", "time", "sudo", "xargs", "&"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

def strip_line(line):
    """Remove comments; replace quoted spans with @Q. Returns the stripped line."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            out.append("X"); i += 2; continue
        if c in ("'", '"'):
            q = c; i += 1
            while i < n:
                if line[i] == "\\" and q == '"':
                    i += 2; continue
                if line[i] == q:
                    i += 1; break
                i += 1
            out.append(" @Q "); continue
        if c == "#":
            prev = out[-1] if out else ""
            # A `#` starts a comment only at the start of a word.
            if not out or prev.isspace() or prev in "(;&|":
                break
            out.append(c); i += 1; continue
        out.append(c); i += 1
    return "".join(out)

SPLIT = re.compile(r"(\$\(|[();|&{}]|&&|\|\|)")

def tokens(s):
    return [t for t in SPLIT.sub(r" \1 ", s).split() if t]

def command_positions(toks):
    """Yield indices of tokens that sit in command position."""
    prev = ""
    i = 0
    while i < len(toks):
        t = toks[i]
        if prev in OPENERS:
            # skip leading VAR=value assignment prefixes
            j = i
            while j < len(toks) and ASSIGN.match(toks[j]) and "=" in toks[j]:
                j += 1
            if j < len(toks):
                yield j
            prev = toks[i]
            i += 1
            continue
        prev = t
        i += 1

def wrapper_positions(toks):
    """Command position created by this repo's own runner wrappers."""
    for i, t in enumerate(toks):
        if t == "run_to" and i + 2 < len(toks):
            yield i + 2
        elif t == "run_split" and i + 3 < len(toks):
            yield i + 3
        elif t == "gate" and i + 2 < len(toks):
            yield i + 2
        elif t == "timeout" and i + 2 < len(toks):
            yield i + 2

def findings(path, text):
    out = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        s = strip_line(raw)
        if not s.strip():
            continue
        toks = tokens(s)
        hits = set(command_positions(toks)) | set(wrapper_positions(toks))
        for idx in sorted(hits):
            if idx < len(toks) and toks[idx] in VERIFIERS:
                out.append("%s:%d: %s" % (path, lineno, toks[idx]))
    return out

rc = 0
for p in sys.argv[1:]:
    try:
        text = open(p, encoding="utf-8", errors="replace").read()
    except OSError as e:
        print("SCANERROR %s %s" % (p, e)); rc = 2; continue
    for f in findings(p, text):
        print(f); rc = 1
sys.exit(rc)
PY
}

# ---------------------------------------------------------------------------
# PART 1 case table. LEFT column must be FLAGGED, right column must NOT.
# Re-run this rather than re-reading the tokeniser.
self_test() {
    local td fails=0 got want
    td=$(mktemp -d) || return 2

    # MUST FLAG. The nine violating lines are ASSEMBLED from $M/$P/$A rather
    # than written out, so this file contains no literal bare invocation of its
    # own. Otherwise the sibling guard scripts/check_apr_bin_pinned.sh — which
    # scans every script for exactly this construct — reports the case table as
    # five real violations. Fixtures that trip a neighbouring guard get that
    # guard an exemption entry, and an exemption is how a guard stops guarding.
    local M=pmat P=pv A=apr
    {
        printf 'gate %s-verify %s verify --format json\n' "$M" "$M"
        printf 'run_to "$WORKLOG/x.log" timeout 900 %s query "x" --limit 1\n' "$M"
        printf 'run_split "$W/a.json" "$W/b.err" timeout 900 %s comply check\n' "$M"
        printf '%s validate "$c" >/dev/null 2>&1\n' "$P"
        printf '%s qa model.apr\n' "$A"
        printf 'foo && %s comply check\n' "$M"
        printf 'OUT=$(%s lint contracts)\n' "$P"
        printf 'PATH=/stale:$PATH %s verify\n' "$M"
        printf 'if %s validate x; then :; fi\n' "$P"
    } > "$td/bad.sh"
    # MUST NOT FLAG
    cat > "$td/good.sh" <<'EOF'
# `pv lint <DIR>` is a real gate and is run separately below.
mark pv-contracts REPORT "pv is not pinned in this repo -- contracts NOT validated."
run_to "$WORKLOG/pv-pc.log" "$PV" validate "$WORKLOG/bogus-contract.yaml"
gate pmat-verify "$PMAT_BIN" verify --format json
run_to "$WORKLOG/pmat-index.log" timeout 900 "$PMAT_BIN" query "x" --limit 1
for t in bashrs pmat probador; do
PMAT_BIN=pmat
echo "pmat"
. scripts/apr_bin.sh || exit 1
#   run_to "$LOG" pmat query "x"
mark pmat-verify SKIP "package has no lib target"
EOF

    got=$(scan "$td/bad.sh" | awk -F: '{print $2}' | tr '\n' ' ')
    want="1 2 3 4 5 6 7 8 9 "
    if [ "$got" = "$want" ]; then
        printf 'ok    MUST-FLAG    all 9 bare invocations reported\n'
    else
        printf 'FAIL  MUST-FLAG    got lines [%s], want [%s]\n' "$got" "$want"; fails=1
    fi

    got=$(scan "$td/good.sh" | tr '\n' ' ')
    if [ -z "$got" ]; then
        printf 'ok    MUST-NOT-FLAG all 10 pinned/comment/string forms accepted\n'
    else
        printf 'FAIL  MUST-NOT-FLAG false positives: %s\n' "$got"; fails=1
    fi

    rm -rf "${td:?}"
    return "$fails"
}

# ---------------------------------------------------------------------------
# PART 2. The pins must BEHAVE. Presence of `PMAT_BIN` proves nothing about
# which binary the gate ends up running, which is the only thing that matters.
behaviour_test() {
    local td fails=0 built stale
    td=$(mktemp -d) || return 2

    # shellcheck source=/dev/null
    if ! . "$REPO_ROOT/scripts/verifier_pin.sh"; then
        printf 'FAIL  pins       scripts/verifier_pin.sh could not be sourced\n'
        rm -rf "${td:?}"; return 1
    fi

    # A STALE pmat, first on PATH. This is the binary the gate must NOT pick.
    mkdir -p "$td/stalebin"
    stale="$td/stalebin/pmat"
    printf '#!/bin/sh\necho stale-pmat\n' > "$stale"
    chmod +x "$stale"
    built="$td/target-release-pmat"
    printf '#!/bin/sh\necho built-pmat\n' > "$built"
    chmod +x "$built"

    # Row 1 — the self-referential case: releasing pmat itself.
    PMAT_BIN=""
    PATH="$td/stalebin:$PATH" verifier_pin_pmat "pmat" "$built"
    if [ "$PMAT_BIN" = "$built" ]; then
        printf 'ok    pmat-pin   releasing pmat selects the BUILT artifact, not the PATH copy\n'
    else
        printf 'FAIL  pmat-pin   releasing pmat resolved to [%s]; the stale PATH pmat at %s\n' "$PMAT_BIN" "$stale"
        printf '                 would have measured a different build than the one shipping.\n'
        fails=1
    fi

    # Row 2 — every OTHER crate: PATH is correct there and must stay the answer.
    PMAT_BIN=""
    verifier_pin_pmat "aprender" "$built"
    if [ "$PMAT_BIN" = "pmat" ]; then
        printf 'ok    pmat-pin   a non-pmat crate still uses the fleet pmat\n'
    else
        printf 'FAIL  pmat-pin   non-pmat crate resolved to [%s], expected the PATH pmat\n' "$PMAT_BIN"; fails=1
    fi

    # Row 3 — a crate named pmat with NO built artifact must not invent one.
    PMAT_BIN=""
    verifier_pin_pmat "pmat" ""
    if [ "$PMAT_BIN" = "pmat" ]; then
        printf 'ok    pmat-pin   no built artifact -> no fabricated pin\n'
    else
        printf 'FAIL  pmat-pin   empty BINPATH resolved to [%s]\n' "$PMAT_BIN"; fails=1
    fi

    # Row 4 — pv. The pin must not be whatever PATH offers. A decoy `pv` goes
    # first on PATH; the resolved PV must differ from it.
    mkdir -p "$td/pvbin"
    printf '#!/bin/sh\necho "pv 0.0.0-decoy"\n' > "$td/pvbin/pv"
    chmod +x "$td/pvbin/pv"
    (
        cd "$REPO_ROOT" || exit 2
        PATH="$td/pvbin:$PATH"
        export PATH
        PV=""
        verifier_pin_pv
        pv_rc=$?
        # The decoy is named directly, not via `command -v`: this file must not
        # itself contain a PATH resolution of a verifier, and naming the path we
        # planted is strictly more precise than asking PATH what it found.
        decoy="$td/pvbin/pv"
        if [ "$pv_rc" -eq 2 ]; then
            printf 'FAIL  pv-pin     this repo SHIPS scripts/pv_bin.sh but the pin reported "unpinned"\n'
            exit 1
        fi
        if [ "$pv_rc" -ne 0 ]; then
            printf 'FAIL  pv-pin     the pin failed to resolve pv (rc=%s) — a release cannot be\n' "$pv_rc"
            printf '                 decided by a verifier that did not build.\n'
            exit 1
        fi
        if [ "$PV" = "$decoy" ]; then
            printf 'FAIL  pv-pin     resolved pv IS the PATH decoy (%s) — the pin was bypassed\n' "$decoy"
            exit 1
        fi
        printf 'ok    pv-pin     resolved pv is %s, NOT the PATH decoy %s\n' "$PV" "$decoy"
    ) || fails=1

    rm -rf "${td:?}"
    return "$fails"
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    --self-test)
        self_test; exit $?
        ;;
esac

rc=0
printf -- '--- verifier pinning ------------------------------------------------\n'

printf 'case table (the tokeniser must be right before its verdict means anything)\n'
if self_test; then :; else rc=1; fi

printf '\nPART 1 — static: no bare verifier in command position\n'
cd "$REPO_ROOT" || exit 2
missing=""
for f in $SCOPE; do
    [ -f "$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
    # A scope entry that does not exist is a gate scanning nothing. The runner
    # could be renamed out from under this guard and it would sweep an empty
    # universe and report clean — the exact failure this repo keeps finding.
    printf 'FAIL  in-scope file(s) missing:%s — the guard scanned an empty universe\n' "$missing"
    rc=1
else
    # shellcheck disable=SC2086
    hits=$(scan $SCOPE)
    scan_rc=$?
    if [ "$scan_rc" -eq 2 ]; then
        printf 'FAIL  the scanner errored:\n%s\n' "$hits"; rc=1
    elif [ -n "$hits" ]; then
        printf 'FAIL  bare verifier invocation(s) — these resolve through PATH:\n'
        printf '%s\n' "$hits" | sed 's/^/      /'
        printf '      Use the pin: "$PV" / "$PMAT_BIN" / "$APR". See scripts/verifier_pin.sh.\n'
        rc=1
    else
        printf 'ok    %s: no bare pv/pmat/apr in command position\n' "$SCOPE"
    fi
fi

printf '\nPART 2 — behavioural: the pins select something other than PATH\n'
if behaviour_test; then :; else rc=1; fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the release runner resolves every pinned verifier through its pin.\n'
else
    printf 'FAIL  see rows above. A gate measured with an unknown binary is not a gate.\n'
fi
exit "$rc"
