#!/usr/bin/env bash
# check_promotion_receipts.sh — a release is promoted (prerelease -> stable) only
# over four host receipts that each prove the DOWNLOADED asset is the TESTED
# binary (PP-066 row R-5, issue #2908, claim 3).
#
#   scripts/check_promotion_receipts.sh <tag> [--manifest <sha256 manifest>] [--receipts <dir>]
#       0 promote · 1 refuse (every reason printed) · 2 env
#   scripts/check_promotion_receipts.sh --self-test        # case table, both polarities
#
# THE FOUR CONDITIONS, PER HOST, ALL REQUIRED. Each is a thing a release once
# shipped without (0.65.2, docs/audits/impl-PMAT-929-receipt.md):
#   1. a receipt EXISTS for the host (missing -> refuse: an untested platform is
#      not a "probably fine" platform);
#   2. its `binary_sha256` is IN the release's signed sha256 manifest (tampered,
#      or a locally-built binary standing in for the asset -> RED);
#   3. its C14 parity `status` is PASS (RED, or absent -> refuse);
#   4. its parity was NOT skipped (`skipped`, `not-run`, SKIP_PARITY_GATE in the
#      command -> refuse: a skipped gate is the 0.65.2 defect exactly).
# The manifest itself must verify against the public key in .github/*.pub
# (minisign, S0-19: the public-key scheme the repo already uses; no third one).
#
# This gate is BASE-OWNED: it reads receipts and a manifest committed under
# evidence/, never anything the PR under test produced (I2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_promotion_receipts
HOSTS="${PROMOTION_HOSTS:-lambda intel gx10 mini}"

# receipt_facts <receipt.json> -> "sha<TAB>status<TAB>skipped" ; rc 3 bad json
receipt_facts() {
    python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(3)
sha = str(d.get("binary_sha256") or "")
par = d.get("parity")
status = ""
skipped = "no"
if isinstance(par, dict):
    status = str(par.get("status") or par.get("verdict") or "")
    text = json.dumps(par).lower()
    if par.get("skipped") is True or "skip_parity_gate" in text or status.lower() in ("skipped", "skip", "not-run", "not_run"):
        skipped = "yes"
elif par is None:
    status = ""
else:
    status = str(par)
print(f"{sha}\t{status}\t{skipped}")
PY
}

verify_manifest() { # verify_manifest <manifest> -> 0 signature ok
    local m=$1
    # read at call time, not at load: the case table overrides it after sourcing
    local PUBKEY="${PROMOTION_PUBKEY:-$ROOT/keys/apr-release-minisign.pub}"
    [ -f "$PUBKEY" ] || { printf 'REFUSE  public key %s is absent; an unverifiable manifest promotes nothing\n' "${PUBKEY#"$ROOT"/}"; return 1; }
    [ -f "$m.minisig" ] || { printf 'REFUSE  %s carries no .minisig; an unsigned manifest promotes nothing\n' "$m"; return 1; }
    if [ -n "${PROMOTION_SKIP_SIG_VERIFY:-}" ]; then
        # the case table exercises the receipt logic with a fixture manifest it
        # cannot sign; it says so on stdout, and only the case table may set this
        printf 'note    manifest signature verification bypassed for the case table\n'
        return 0
    fi
    command -v minisign >/dev/null 2>&1 || { printf '%s: ENV - minisign missing\n' "$PROG" >&2; return 2; }
    minisign -Vm "$m" -p "$PUBKEY" >/dev/null 2>&1 || { printf 'REFUSE  %s does not verify against %s\n' "$m" "${PUBKEY#"$ROOT"/}"; return 1; }
    printf 'ok      manifest verifies against %s\n' "${PUBKEY#"$ROOT"/}"
}

check() { # check <tag> <manifest> <dir> <hosts...>
    local tag=$1 manifest=$2 dir=$3; shift 3
    local rc=0 h f facts sha status skipped
    command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 missing\n' "$PROG" >&2; return 2; }
    [ -f "$manifest" ] || { printf '%s: ENV - manifest %s missing\n' "$PROG" "$manifest" >&2; return 2; }
    [ -d "$dir" ] || { printf '%s: ENV - receipts dir %s missing\n' "$PROG" "$dir" >&2; return 2; }
    verify_manifest "$manifest" || { rc=$?; [ "$rc" = 2 ] && return 2; return 1; }
    for h in "$@"; do
        f="$dir/$h.json"
        if [ ! -f "$f" ]; then printf 'REFUSE  %-7s no receipt at %s — an untested platform is not a probably-fine platform\n' "$h" "$f"; rc=1; continue; fi
        facts=$(receipt_facts "$f") || { printf 'REFUSE  %-7s receipt is not valid JSON\n' "$h"; rc=1; continue; }
        sha=$(printf '%s' "$facts" | cut -f1); status=$(printf '%s' "$facts" | cut -f2); skipped=$(printf '%s' "$facts" | cut -f3)
        if [ -z "$sha" ]; then printf 'REFUSE  %-7s receipt carries no binary_sha256\n' "$h"; rc=1; continue; fi
        if ! grep -qE "^$sha([[:space:]]|$)" "$manifest"; then
            printf 'REFUSE  %-7s binary_sha256 %s… is NOT in the release manifest: the tested binary is not the downloaded asset\n' "$h" "${sha:0:12}"; rc=1; continue
        fi
        if [ "$skipped" = yes ]; then printf 'REFUSE  %-7s parity was SKIPPED — a skipped gate is the 0.65.2 defect exactly\n' "$h"; rc=1; continue; fi
        case "$status" in
            PASS|pass) printf 'ok      %-7s asset sha matches, C14 parity PASS\n' "$h" ;;
            '')        printf 'REFUSE  %-7s receipt carries no parity status\n' "$h"; rc=1 ;;
            *)         printf 'REFUSE  %-7s C14 parity %s\n' "$h" "$status"; rc=1 ;;
        esac
    done
    if [ "$rc" = 0 ]; then printf 'PROMOTE %s: four receipts, each with the asset sha256 and C14 PASS, parity never skipped\n' "$tag"; fi
    return "$rc"
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/promo.XXXXXX")
    cleanup() { case "$TD" in *promo.*) if [ -n "$TD" ] && [ "$TD" != "/" ]; then rm -rf -- "$TD"; fi ;; esac; }
    trap cleanup EXIT
    n=0; red=0
    row() { local _w=$1 _l=$2 _rc=0; shift 2; n=$((n + 1))
        "$@" > "$TD/o.$n" 2>&1 || _rc=$?
        if [ "$_rc" = "$_w" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$_rc" "$_l"
        else printf 'FAIL  row %-2s rc=%s (want %s)  %s\n' "$n" "$_rc" "$_w" "$_l"; sed 's/^/        /' "$TD/o.$n"; red=$((red + 1)); fi; }
    GOOD=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    BAD=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    mkdir -p "$TD/r"; printf '%s  apr-v1-x86_64-unknown-linux-gnu.tar.gz\n' "$GOOD" > "$TD/manifest.sha256"; : > "$TD/manifest.sha256.minisig"
    printf 'untrusted comment: test\nRWTest\n' > "$TD/key.pub"
    mk() { printf '%s' "$2" > "$TD/r/$1.json"; }
    export PROMOTION_PUBKEY="$TD/key.pub" PROMOTION_SKIP_SIG_VERIFY=1
    mk ok       "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"PASS\"}}"
    row 0 "one host: asset sha in manifest, C14 PASS, not skipped"          check v1 "$TD/manifest.sha256" "$TD/r" ok
    row 1 "MUTATION missing: a host with no receipt refuses"                check v1 "$TD/manifest.sha256" "$TD/r" ok absent
    mk tamper   "{\"binary_sha256\":\"$BAD\",\"parity\":{\"status\":\"PASS\"}}"
    row 1 "MUTATION tampered: a sha not in the manifest is RED"             check v1 "$TD/manifest.sha256" "$TD/r" tamper
    mk skipped  "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"skipped\"}}"
    row 1 "MUTATION skipped: parity status skipped refuses"                 check v1 "$TD/manifest.sha256" "$TD/r" skipped
    mk skipenv  "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"PASS\",\"command\":\"SKIP_PARITY_GATE=1 apr run\"}}"
    row 1 "skipped via SKIP_PARITY_GATE in the command refuses even with PASS" check v1 "$TD/manifest.sha256" "$TD/r" skipenv
    mk fail     "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"FAIL\"}}"
    row 1 "C14 parity FAIL refuses"                                         check v1 "$TD/manifest.sha256" "$TD/r" fail
    mk nopar    "{\"binary_sha256\":\"$GOOD\"}"
    row 1 "no parity block at all refuses (absence is not a pass)"          check v1 "$TD/manifest.sha256" "$TD/r" nopar
    mk nosha    "{\"parity\":{\"status\":\"PASS\"}}"
    row 1 "no binary_sha256 refuses"                                        check v1 "$TD/manifest.sha256" "$TD/r" nosha
    printf 'not json' > "$TD/r/broken.json"
    row 1 "malformed receipt refuses"                                       check v1 "$TD/manifest.sha256" "$TD/r" broken
    mk ok2 "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"PASS\"}}"; mk ok3 "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"PASS\"}}"; mk ok4 "{\"binary_sha256\":\"$GOOD\",\"parity\":{\"status\":\"PASS\"}}"
    row 0 "four good hosts promote"                                         check v1 "$TD/manifest.sha256" "$TD/r" ok ok2 ok3 ok4
    row 1 "three good + one tampered refuses (all four required)"           check v1 "$TD/manifest.sha256" "$TD/r" ok ok2 ok3 tamper
    rm -f "$TD/manifest.sha256.minisig"
    row 1 "an unsigned manifest promotes nothing"                           check v1 "$TD/manifest.sha256" "$TD/r" ok
    : > "$TD/manifest.sha256.minisig"
    row 2 "a missing manifest is ENV (exit 2), never a pass"                check v1 "$TD/nope.sha256" "$TD/r" ok
    row 2 "a missing receipts dir is ENV (exit 2)"                          check v1 "$TD/manifest.sha256" "$TD/nodir" ok
    unset PROMOTION_SKIP_SIG_VERIFY
    row 1 "with real verification, a fake signature does NOT verify"        check v1 "$TD/manifest.sha256" "$TD/r" ok
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

case "${1:-}" in
    --help|-h) printf 'usage: %s <tag> [--manifest <sha256>] [--receipts <dir>] | --self-test\n' "$PROG"; exit 0 ;;
    '')
        # Run BARE (no tag) this guard has no input, and guard_tree.sh runs every
        # scripts/check_*.sh bare. It is wired with its argument by the release
        # promotion step, so a bare run prints the usage and exits 0 — a red row
        # for a guard that was never given its input is the one failure a
        # run-all runner may not produce. The case table (--self-test) is what
        # proves it; the tagged run is what promotes.
        printf 'NOT-RUN %s: no <tag> given; wired with its argument by the promotion step (run --self-test for the case table)\n' "$PROG"
        exit 0 ;;
esac
TAG=$1; shift
MANIFEST="$ROOT/evidence/release/$TAG/apr-$TAG.sha256"
DIR="$ROOT/evidence/dogfood/${TAG#v}"
while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST=$2; shift 2 ;;
        --receipts) DIR=$2; shift 2 ;;
        *) printf '%s: unknown argument %s\n' "$PROG" "$1" >&2; exit 2 ;;
    esac
done
check "$TAG" "$MANIFEST" "$DIR" $HOSTS
