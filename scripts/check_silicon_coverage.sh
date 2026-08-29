#!/usr/bin/env bash
# Every silicon axis we claim to cover must have a runner that can serve it —
# and every axis we have DEFERRED must still be deferred for a reason.
#
# WHY THIS EXISTS (paiml/infra#361).
#
# aprender's correctness is silicon-dependent: SIMD paths, CUDA kernels, arch
# intrinsics, float behaviour. Today two axes are exercised nightly — x86_64-CPU
# and aarch64+Blackwell — and the gap just widened on purpose: paiml/aprender#2740
# makes cuda-nightly gx10-only because its x86_64 GPU leg ran on lambda-labs,
# which must never be a CI host (paiml/infra#359). sm_89 was the STABLE reference
# and sm_121 the higher-risk target with a history of JIT bugs. We now gate only
# the risky one.
#
# That is a defensible trade, but an UNRECORDED one decays into "we test on
# whatever is plugged in". So the axes live in .github/silicon-coverage.txt and
# this asks GitHub which of them a runner could actually serve.
#
# TWO DIRECTIONS, and the second is the one that keeps the ledger honest:
#
#   required   -> FAIL if no runner can serve the selector. A lane that silently
#                 stops covering an architecture looks exactly like one that
#                 never covered it.
#   pending:N  -> FAIL if a runner CAN now serve it. At that moment the only
#                 thing between us and the coverage is a line in a policy file,
#                 and a "pending" that survives its own blocker is how a debt
#                 ledger becomes a blanket exemption.
#
# SELECTOR SEMANTICS, because getting this backwards is the fleet's recorded
# mistake (paiml/infra#352): a `runs-on` list matches a runner when EVERY named
# label is present on that runner. Extra runner labels never exclude. So "can
# this axis run?" is "does some online runner's label set CONTAIN all of ours?".
#
# INSTRUMENT PROBES, both mandatory:
#   1. --self-test runs committed fixtures through the matcher and demands each
#      verdict. A matcher that cannot tell a subset from a superset exits 2.
#   2. A POSITIVE CONTROL on the live API: the runner listing must be non-empty.
#      Zero runners looks identical whether the fleet is down or the token lost
#      its scope, and blind is a NO-GO.
#
# AUTH: the runner's ambient org-scoped gh auth. GPU runners are REPO-scoped and
# invisible in the org listing, so BOTH lists are read and merged — that
# invisibility is exactly why yoga-gpu's absence went unnoticed for months.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${SILICON_POLICY:-$REPO_ROOT/.github/silicon-coverage.txt}"
ORG="${ORG:-paiml}"
REPO="${GITHUB_REPOSITORY:-paiml/aprender}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

# labels_contain <runner-label-csv> <required-label-csv>
# 0 when every required label is present in the runner's set. Case-insensitive:
# GitHub's built-in labels are `Linux`/`X64`/`ARM64` and a selector may spell
# them either way.
# All locals are `_lc_`-prefixed. The first version used `_have`/`_want`, which
# are also the loop variables in selftest() — POSIX sh has no function scope, so
# the call CLOBBERED the caller's expectation and every case compared a verdict
# against a label string. The self-test caught it on the first run, which is the
# only reason to have one.
labels_contain() {
    _lc_have=",$(printf '%s' "$1" | tr 'A-Z' 'a-z'),"
    _lc_want="$(printf '%s' "$2" | tr 'A-Z' 'a-z')"
    _lc_ifs="$IFS"; IFS=','
    for _lc_l in $_lc_want; do
        [ -n "$_lc_l" ] || continue
        case "$_lc_have" in
            *",$_lc_l,"*) ;;
            *) IFS="$_lc_ifs"; return 1 ;;
        esac
    done
    IFS="$_lc_ifs"
    return 0
}

selftest() {
    _broken=0
    # runner labels                    | selector                | want (0=serves)
    for _case in \
        "self-hosted,linux,x64,clean-room,intel|self-hosted,clean-room,intel|0" \
        "self-hosted,linux,arm64,gpu,gx10,cuda,blackwell,gb10|self-hosted,gpu,gx10,cuda,blackwell|0" \
        "self-hosted,linux,x64,clean-room,intel|self-hosted,gpu,gx10|1" \
        "self-hosted,linux,arm64,gpu,gx10,cuda|self-hosted,gpu,gx10,cuda,blackwell|1" \
        "self-hosted,Linux,ARM64,gpu,gx10,cuda,blackwell|self-hosted,gpu,gx10,cuda,blackwell|0" \
        "self-hosted,linux,x64,cuda,gpu,lambda-4090|self-hosted,gpu,yoga,cuda,ada|1" \
        ; do
        _have="${_case%%|*}"; _rest="${_case#*|}"
        _want_sel="${_rest%%|*}"; _want="${_rest#*|}"
        if labels_contain "$_have" "$_want_sel"; then _got=0; else _got=1; fi
        if [ "$_got" != "$_want" ]; then
            printf '  [%s] vs [%s]: expected %s (0=serves), got %s\n' \
                "$_have" "$_want_sel" "$_want" "$_got"
            _broken=$((_broken + 1))
        fi
    done
    if [ "$_broken" -gt 0 ]; then
        printf 'INSTRUMENT BROKEN — the label matcher cannot classify its own cases.\n'
        printf 'Refusing to report on the fleet with a matcher that failed its own tests.\n'
        return 2
    fi
    printf 'self-test: 6 label cases, matcher classifies all of them as specified\n'
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    printf -- '-- instrument self-test --\n'
    selftest; exit $?
fi

printf '== silicon coverage ==\n'
printf 'policy: %s\n\n' "$POLICY"
[ -f "$POLICY" ] || { printf 'NO-GO: %s does not exist.\n' "$POLICY"; exit 2; }

printf -- '-- instrument self-test --\n'
selftest || exit 2

# ── the runners: org-scoped AND repo-scoped ─────────────────────────────────
RUNNERS="$(mktemp)"; trap 'rm -f "$RUNNERS"' EXIT
: > "$RUNNERS"
for src in "orgs/$ORG/actions/runners" "repos/$REPO/actions/runners"; do
    gh api --paginate "$src?per_page=100" \
        --jq '.runners[] | select(.status=="online") | (.name) + "\t" + ([.labels[].name] | join(","))' \
        2>/dev/null >> "$RUNNERS" || true
done
sort -u -o "$RUNNERS" "$RUNNERS"
n_runners="$(grep -cve '^[[:space:]]*$' "$RUNNERS" || true)"

printf -- '\n-- runners --\n'
printf 'online runners visible (org + this repo): %s\n' "${n_runners:-0}"
# Probe 2: the positive control.
if [ "${n_runners:-0}" -eq 0 ]; then
    printf 'NO-GO: zero online runners. That looks identical whether the fleet is\n'
    printf 'down or this token lost its scope, and GPU runners are REPO-scoped and\n'
    printf 'invisible in the org listing. Blind is a NO-GO, not a pass.\n'
    exit 2
fi

# ── the axes ────────────────────────────────────────────────────────────────
printf -- '\n-- axes --\n'
axes=0; required=0; covered=0; missing=0; deferred=0; promotable=0
fail=0
while IFS= read -r line; do
    line="${line%%#*}"
    case "$line" in ''|[[:space:]]*[[:space:]]) ;; esac
    set -- $line
    [ $# -ge 3 ] || continue
    axis="$1"; status="$2"; selector="$3"
    axes=$((axes + 1))

    server=""
    while IFS="$(printf '\t')" read -r rname rlabels; do
        [ -n "$rname" ] || continue
        if labels_contain "$rlabels" "$selector"; then server="$rname"; break; fi
    done < "$RUNNERS"

    case "$status" in
        required)
            required=$((required + 1))
            if [ -n "$server" ]; then
                covered=$((covered + 1))
                printf '  ok        %-20s served by %s\n' "$axis" "$server"
            else
                missing=$((missing + 1)); fail=1
                printf '  MISSING   %-20s REQUIRED, but no online runner carries [%s]\n' "$axis" "$selector"
            fi
            ;;
        pending:*)
            deferred=$((deferred + 1))
            if [ -n "$server" ]; then
                promotable=$((promotable + 1)); fail=1
                printf '  PROMOTE   %-20s marked %s, but %s can serve it NOW\n' "$axis" "$status" "$server"
            else
                printf '  deferred  %-20s %s — no runner yet\n' "$axis" "$status"
            fi
            ;;
        *)
            printf '  BAD       %-20s unknown status "%s"\n' "$axis" "$status"
            fail=1
            ;;
    esac
done < "$POLICY"

printf -- '\n-- denominators --\n'
printf 'axes declared: %s  (required %s: covered %s, MISSING %s; deferred %s: PROMOTABLE %s)\n' \
    "$axes" "$required" "$covered" "$missing" "$deferred" "$promotable"

# THE DENOMINATOR. A policy that parsed nothing must not read as clean.
if [ "$axes" -eq 0 ]; then
    printf '\nNO-GO: 0 axes parsed from %s. Nothing was measured.\n' "$POLICY"; exit 2
fi

{
    printf '\n### Silicon coverage\n\n'
    printf -- '- axes: **%s** (required %s, deferred %s)\n' "$axes" "$required" "$deferred"
    printf -- '- required covered: **%s of %s**\n' "$covered" "$required"
    # No parentheses in this string, and an `if` rather than `[ ] &&`: bashrs
    # lexes a `(` inside a printf argument on the same line as a `[ ]` test as an
    # unescaped test paren (SC1028), and a `[ ] &&` tail returns non-zero when the
    # test is false.
    if [ "$promotable" -gt 0 ]; then
        printf -- '- **%s deferred axis/axes now have a runner and must be promoted**\n' "$promotable"
    fi
    printf -- '- runners inspected: %s\n' "$n_runners"
} >> "$SUMMARY"

if [ "$fail" -ne 0 ]; then
    printf '\nFAIL: silicon coverage does not match the policy.\n'
    printf 'A MISSING required axis means an architecture stopped being tested and\n'
    printf 'nothing said so. A PROMOTABLE deferred axis means the blocker is gone and\n'
    printf 'the only thing left is the line in .github/silicon-coverage.txt.\n'
    exit 1
fi
printf '\nOK: every required axis has a runner, and every deferred axis is still blocked.\n'
