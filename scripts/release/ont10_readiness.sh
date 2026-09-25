#!/usr/bin/env bash
# ont10_readiness.sh <version> [--run-oracle] -- ONT-001 row ONT-10: is aprender-contracts-cli <version>
#   ready for the operator's publish? READ-ONLY toward crates.io: it never publishes, never holds a
#   token, and ends where the row ends, at STOP(PUBLISH: aprender-contracts-cli <version>, dry-run
#   receipt at <path>) -- the publish is the operator's (RP-001), run through publish_strict.sh.
#
# What it measures, each a line of the receipt $AP/ont10-readiness.json:
#   previous_pin   the version crates.io serves today (the spec's release receipt records it)
#   pkgid          `cargo pkgid -p aprender-contracts-cli` at HEAD; must equal <version>
#   pin_below      previous_pin sorts strictly below <version> (a re-publish is refused by crates.io)
#   in_order       publish-order.txt lists the crate, so the cascade reaches it
#   dryrun_receipt $AP/dryrun-receipt-commit (T-4) -- the path the STOP line names
#   cleanroom      $AP/cleanroom-run-id (rule 7: clean-room-aprender first)
#   oracle         $AP/oracle-check-commit == HEAD; --run-oracle runs `make oracle-check` and writes it
#
# Exit: 0 READY (prints the STOP line) | 10 NOT READY (names every missing item) | 3 NOT MEASURED
#   (crates.io unreachable or cargo failed -- unmeasured is never ready, and never a code a crash
#   could also produce) | 2 usage. The ONT row merge status is not re-derived here: it is #3559.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || { echo "cannot resolve the repo root from $0" >&2; exit 2; }
# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
release_params "${1:-}" "$REPO_ROOT" || { echo "usage: ont10_readiness.sh <version> [--run-oracle]" >&2; exit 2; }
shift
RUN_ORACLE=0
case "${1:-}" in
    "") ;;
    --run-oracle) RUN_ORACLE=1 ;;
    *) echo "usage: ont10_readiness.sh <version> [--run-oracle]" >&2; exit 2 ;;
esac
unset CARGO_REGISTRY_TOKEN
CRATE=aprender-contracts-cli
cd "$REPO_ROOT" || exit 2
HEAD_SHA=$(git rev-parse HEAD) || { echo "NOT MEASURED: git rev-parse HEAD failed" >&2; exit 3; }

# version_lt A B -> 0 when A sorts strictly below B (sort -V; equal is not below)
version_lt() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$1" ]
}

api=$(curl -fsS --max-time 20 -H "User-Agent: paiml-aprender-ont10-readiness" \
    "https://crates.io/api/v1/crates/$CRATE") || { echo "NOT MEASURED: crates.io did not answer for $CRATE" >&2; exit 3; }
previous_pin=$(printf '%s' "$api" | python3 -c 'import json,sys; print(json.load(sys.stdin)["crate"]["max_version"])') \
    || { echo "NOT MEASURED: crates.io answer for $CRATE has no crate.max_version" >&2; exit 3; }
pkgid=$(cargo pkgid -p "$CRATE" 2>/dev/null) || { echo "NOT MEASURED: cargo pkgid -p $CRATE failed" >&2; exit 3; }
pkgid_version=${pkgid##*[@#]}

if [ "$RUN_ORACLE" = 1 ]; then
    mkdir -p "$AP" || exit 2
    if make --no-print-directory oracle-check; then
        printf '%s\n' "$HEAD_SHA" > "$AP/oracle-check-commit"
    else
        rm -f "$AP/oracle-check-commit"
    fi
fi

missing=()
[ "$pkgid_version" = "$V" ] || missing+=("pkgid: HEAD builds $CRATE $pkgid_version, not $V")
version_lt "$previous_pin" "$V" || missing+=("pin_below: crates.io already serves $previous_pin, not below $V")
grep -qx "$CRATE" scripts/release/publish-order.txt || missing+=("in_order: $CRATE is not in scripts/release/publish-order.txt")
[ -s "$AP/dryrun-receipt-commit" ] || missing+=("dryrun_receipt: no $AP/dryrun-receipt-commit (T-4)")
[ -s "$AP/cleanroom-run-id" ] || missing+=("cleanroom: no $AP/cleanroom-run-id (rule 7)")
# RP-001 no-publish-in-ci (ONT-10: "RP-001 no-publish-in-ci green"): measured on this tree, every run
rp001=$(bash scripts/release-policy.sh --only no-publish-in-ci 2>&1); rp001_rc=$?
# Green is rc 0 AND the gate's own PASS line: an empty or truncated release-policy.sh exits 0 too.
case "$rp001_rc:$rp001" in
    "0:PASS no-publish-in-ci"*) ;;
    *) missing+=("no_publish_in_ci: rc $rp001_rc, ${rp001:-release-policy.sh printed nothing}") ;;
esac
oracle_commit=$(cat "$AP/oracle-check-commit" 2>/dev/null || true)
[ "$oracle_commit" = "$HEAD_SHA" ] || missing+=("oracle: make oracle-check not recorded green at $HEAD_SHA (--run-oracle)")

ready=false
[ "${#missing[@]}" = 0 ] && ready=true
if mkdir -p "$AP" 2>/dev/null; then
    python3 - "$AP/ont10-readiness.json" "$V" "$HEAD_SHA" "$previous_pin" "$pkgid_version" "$ready" "${missing[@]}" <<'PY'
import json, sys
out, version, head, pin, pkgid, ready, *missing = sys.argv[1:]
# every item, by name: true only when it was measured green (its absence from `missing`)
ITEMS = ("pkgid", "pin_below", "in_order", "dryrun_receipt", "cleanroom", "no_publish_in_ci", "oracle")
failed = {m.split(":", 1)[0] for m in missing}
unknown = failed - set(ITEMS)
if unknown:
    sys.exit("readiness item(s) not in ITEMS: %s" % sorted(unknown))
json.dump({"row": "ONT-10", "crate": "aprender-contracts-cli", "version": version, "head": head,
           "previous_pin": pin, "pkgid": pkgid, "ready": ready == "true",
           "items": {i: i not in failed for i in ITEMS}, "missing": missing},
          open(out, "w"), indent=2)
PY
    [ "$?" = 0 ] || { echo "NOT MEASURED: could not write $AP/ont10-readiness.json" >&2; exit 3; }
else
    echo "NOT MEASURED: cannot create $AP for the readiness receipt" >&2; exit 3
fi

echo "ONT-10 $CRATE $V at ${HEAD_SHA:0:9}: previous_pin=$previous_pin pkgid=$pkgid_version"
if [ "$ready" = true ]; then
    echo "STOP(PUBLISH: $CRATE $V, dry-run receipt at $AP/dryrun-receipt-commit)"
    exit 0
fi
printf 'NOT READY: %s\n' "${missing[@]}"
exit 10
