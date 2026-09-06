#!/usr/bin/env bash
# fleet_verify.sh — `make fleet-verify ROW=<row>`: run the row's `.pr/<row>/accept.sh`
# on the hosts its DAG card names and collect one receipt per host (G-11b, #3018).
#
# The sanctioned path to the fleet is THIS target (forjar's verb surface has no
# exec verb — `forjar verb list` 2026-09-06 — so the transport is ssh, driven only
# from here, never ad hoc): the host aliases are ~/.ssh/config's (lambda -> lambda-labs,
# gx10, intel, mini); the remote repo is $FLEET_REPO_DIR (default ~/src/aprender);
# the remote checkout is fetched to the SHA being verified, so every receipt names
# the commit it ran.
#
#   bash scripts/fleet_verify.sh --row <id> [--sha <commit>] [--hosts "lambda gx10"] [--dry-run]
#   bash scripts/fleet_verify.sh --self-test
# Receipts: evidence/fleet/<row>/<host>.json {row, host, alias, sha, rc, started, ended, log_sha256}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=fleet_verify
ROW=""; SHA=""; HOSTS=""; DRY=0
alias_of() { case "$1" in lambda|lambda-labs) echo lambda-labs ;; gx10) echo gx10 ;; intel|mac-server) echo intel ;; mini) echo mini ;; *) return 1 ;; esac; }
hosts_of_row() { # from the DAG card's host field: the known host names it mentions, in order
    python3 - "$ROOT/docs/specifications/pp-066-dag.yaml" "$1" <<'PY'
import sys, yaml, re
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
row = next((r for r in d["rows"] if r["id"] == sys.argv[2]), None)
if row is None: sys.exit(3)
h = str(row.get("host") or "").lower()
names = [n for n in ("lambda", "gx10", "intel", "mini") if re.search(rf"\b{n}\b", h)]
if "all four" in h or "four hosts" in h: names = ["lambda", "gx10", "intel", "mini"]
print(" ".join(names))
PY
}
if [ "${1:-}" = "--self-test" ]; then
    n=0; red=0
    t() { local want=$1 label=$2 got=$3; n=$((n + 1)); if [ "$got" = "$want" ]; then printf 'ok    row %-2s %s\n' "$n" "$label"; else printf 'FAIL  row %-2s got=%s wanted=%s  %s\n' "$n" "$got" "$want" "$label"; red=1; fi; }
    t lambda-labs "alias: lambda -> lambda-labs" "$(alias_of lambda)"
    t gx10 "alias: gx10 -> gx10" "$(alias_of gx10)"
    t "" "alias: an unknown host is refused" "$(alias_of nope 2>/dev/null || true)"
    TD=$(mktemp -d "${TMPDIR:-/tmp}/fleetv.XXXXXX"); mkdir -p "$TD/docs/specifications"
    printf 'rows:\n- {id: X, host: "lambda, gx10 (fleet-verify)"}\n- {id: Y, host: "all four (dogfood)"}\n- {id: Z, host: any}\n' > "$TD/docs/specifications/pp-066-dag.yaml"
    t "lambda gx10" "hosts from the card: lambda, gx10" "$(ROOT=$TD hosts_of_row X)"
    t "lambda gx10 intel mini" "hosts from the card: all four" "$(ROOT=$TD hosts_of_row Y)"
    t "" "hosts from the card: any -> none (run locally)" "$(ROOT=$TD hosts_of_row Z)"
    rm -rf "${TD:?}"
    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi
while [ $# -gt 0 ]; do case "$1" in --row) ROW=$2; shift 2 ;; --sha) SHA=$2; shift 2 ;; --hosts) HOSTS=$2; shift 2 ;; --dry-run) DRY=1; shift ;; *) printf 'usage: %s --row <id> [--sha <commit>] [--hosts "..."] [--dry-run] | --self-test\n' "$PROG" >&2; exit 2 ;; esac; done
[ -n "$ROW" ] || { printf '%s: --row is required\n' "$PROG" >&2; exit 2; }
[ -x "$ROOT/.pr/$ROW/accept.sh" ] || { printf '%s: %s/.pr/%s/accept.sh is missing or not executable (P1 writes it)\n' "$PROG" "$ROOT" "$ROW" >&2; exit 2; }
SHA=${SHA:-$(git -C "$ROOT" rev-parse HEAD)}
[ -n "$HOSTS" ] || HOSTS=$(hosts_of_row "$ROW") || { printf '%s: row %s is not in the DAG\n' "$PROG" "$ROW" >&2; exit 2; }
[ -n "$HOSTS" ] || { printf '%s: row %s names no fleet host (host: any) — run .pr/%s/accept.sh locally\n' "$PROG" "$ROW" "$ROW"; exit 0; }
REPO="${FLEET_REPO_DIR:-src/aprender}"; OUT="$ROOT/evidence/fleet/$ROW"; mkdir -p "$OUT"; rc_all=0
for h in $HOSTS; do
    a=$(alias_of "$h") || { printf '%s: unknown host %s\n' "$PROG" "$h" >&2; rc_all=1; continue; }
    started=$(date -u +%FT%TZ); log="$OUT/$h.log"
    if [ "$DRY" = 1 ]; then printf 'DRY  %s (%s): ssh %s "cd %s && git fetch -q origin && git checkout -q %s && bash .pr/%s/accept.sh"\n' "$h" "$a" "$a" "$REPO" "$SHA" "$ROW"; continue; fi
    rc=0; ssh -o BatchMode=yes -o ConnectTimeout=20 "$a" "cd $REPO && git fetch -q origin && git checkout -q $SHA 2>/dev/null && ls .pr/$ROW/accept.sh >/dev/null && bash .pr/$ROW/accept.sh" > "$log" 2>&1 || rc=$?
    ended=$(date -u +%FT%TZ)
    python3 - "$OUT/$h.json" "$ROW" "$h" "$a" "$SHA" "$rc" "$started" "$ended" "$log" <<'PY'
import json, sys, hashlib
out, row, host, alias, sha, rc, started, ended, log = sys.argv[1:10]
json.dump({"row": row, "host": host, "alias": alias, "sha": sha, "rc": int(rc), "started": started, "ended": ended, "log": log.split("/evidence/")[-1], "log_sha256": hashlib.sha256(open(log, "rb").read()).hexdigest()}, open(out, "w"), indent=1)
PY
    printf '%s  %s (%s) rc=%s -> evidence/fleet/%s/%s.json\n' "$([ "$rc" = 0 ] && echo PASS || echo FAIL)" "$h" "$a" "$rc" "$ROW" "$h"; [ "$rc" = 0 ] || rc_all=1
done
exit "$rc_all"
