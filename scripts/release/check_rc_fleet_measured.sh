#!/usr/bin/env bash
# check_rc_fleet_measured.sh — an rc tag is not a release until it is DEPLOYED to the
# whole fleet and MEASURED there (APR-RELEASE-001 §3 rule 18, #4434; operator 2026-09-25:
# "no, only release candidate and deployed to fleet and measured going forward all releases").
#
#   check_rc_fleet_measured.sh <tag> <install-receipt.md> [perf-ledger.jsonl]
#   check_rc_fleet_measured.sh --self-test
#
# GREEN only when BOTH hold for every host in FLEET (lambda-labs gx10 yoga intel mini):
#   1 install receipt — a markdown row `| <host> | ... apr <X.Y.Z-rc.N> (<sha9>) ... | ok |`
#     (the table fire.sh / the rc-cut writes after reading PATH `apr --version` on the host);
#   2 perf ledger (paiml/infra#1057, default ~/.local/state/arbiter/perf/ledger.jsonl) —
#     at least one row with host=<host>, binary=apr, version=<X.Y.Z-rc.N>, and a sha that,
#     when present, matches the tag's commit.
# Fails closed: a missing receipt or ledger, or a tag that does not resolve, is RED (exit 1).
# Exit: 0 GREEN, 1 RED, 2 usage / harness broken.
set -euo pipefail

FLEET=${FLEET:-"lambda-labs gx10 yoga intel mini"}

check() { # <want-version> <sha9> <receipt> <ledger>  -> prints one line per failure, returns 0/1
  python3 - "$1" "$2" "$3" "$4" "$FLEET" <<'PY'
import json, os, re, sys
want, sha9, receipt, ledger, fleet = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split()
bad = []
if not os.path.isfile(receipt):
    bad.append(f"install receipt missing: {receipt}")
    rows = []
else:
    rows = [l for l in open(receipt) if l.startswith("|")]
for h in fleet:
    hit = [r for r in rows if [c.strip() for c in r.strip().strip("|").split("|")][:1] == [h]]
    if not hit:
        bad.append(f"install: no receipt row for {h}")
    elif not any(f"apr {want} ({sha9})" in r and r.rstrip().rstrip("|").rstrip().endswith("ok") for r in hit):
        bad.append(f"install: {h} row does not read 'apr {want} ({sha9})' with verdict ok")
if not os.path.isfile(ledger):
    bad.append(f"perf ledger missing: {ledger}")
    recs = []
else:
    recs = []
    for n, l in enumerate(open(ledger), 1):
        l = l.strip()
        if not l:
            continue
        try:
            recs.append(json.loads(l))
        except ValueError:
            bad.append(f"ledger line {n} is not JSON")
for h in fleet:
    ok = [r for r in recs if r.get("host") == h and r.get("binary") == "apr" and r.get("version") == want
          and (not r.get("sha") or str(r["sha"]).startswith(sha9) or sha9.startswith(str(r["sha"])))]
    if not ok:
        bad.append(f"ledger: no apr {want} row for {h}")
for b in bad:
    print("RED  " + b)
sys.exit(1 if bad else 0)
PY
}

self_test() {
  local d; d=$(mktemp -d); trap 'rm -rf "${d:?}"' RETURN
  local W=0.1.0-rc.1 S=abcdef123 pass=0 n=0
  good_receipt() { printf '| host | arch | apr | --version | verdict |\n|---|---|---|---|---|\n'
    for h in $FLEET; do printf '| %s | x | (path) | apr %s (%s) | ok |\n' "$h" "$W" "$S"; done; }
  good_ledger() { for h in $FLEET; do printf '{"host":"%s","binary":"apr","version":"%s","sha":"%s"}\n' "$h" "$W" "$S"; done; }
  # name @ expected rc @ receipt transform @ ledger transform  ('@', because transforms contain '|')
  local cases=(
    "all-green@0@cat@cat"
    "host-missing-from-receipt@1@grep -v '| mini |'@cat"
    "receipt-wrong-sha@1@sed 's/($S)/(0000000ff)/'@cat"
    "receipt-verdict-RED@1@sed '/| gx10 |/s/ok |\$/RED |/'@cat"
    "receipt-older-version@1@sed 's/apr $W/apr 0.1.0-rc.0/'@cat"
    "ledger-empty@1@cat@true"
    "ledger-host-missing@1@cat@grep -v yoga"
    "ledger-other-version@1@cat@sed 's/$W/0.0.9/'"
    "ledger-wrong-sha@1@cat@sed 's/$S/0000000ff/'"
    "ledger-null-sha-ok@0@cat@sed 's/\"$S\"/null/'"
    "ledger-other-binary@1@cat@sed 's/\"apr\"/\"pv\"/'"
    "receipt-file-absent@1@ABSENT@cat"
    "ledger-file-absent@1@cat@ABSENT"
  )
  local c name want rt lt rc
  for c in "${cases[@]}"; do
    IFS='@' read -r name want rt lt <<<"$c"; n=$((n + 1))
    rm -f "$d/r.md" "$d/l.jsonl"
    # a transform that errors would make its row pass vacuously: that is a broken harness
    [ "$rt" = ABSENT ] || good_receipt | bash -c "$rt" >"$d/r.md" || { echo "HARNESS $name: receipt transform failed"; return 2; }
    [ "$lt" = ABSENT ] || good_ledger | bash -c "$lt" >"$d/l.jsonl" || { echo "HARNESS $name: ledger transform failed"; return 2; }
    rc=0; check "$W" "$S" "$d/r.md" "$d/l.jsonl" >/dev/null || rc=$?
    if [ "$rc" = "$want" ]; then pass=$((pass + 1)); else echo "SELF-TEST FAIL $name: rc=$rc want=$want"; fi
  done
  echo "self-test: $pass/$n"
  [ "$pass" = "$n" ] || return 2
}

case "${1:-}" in
  --self-test) self_test ;;
  "" | -h | --help) sed -n '2,15p' "$0"; exit 2 ;;
  *)
    TAG=$1; RECEIPT=${2:?install receipt path}; LEDGER=${3:-$HOME/.local/state/arbiter/perf/ledger.jsonl}
    WANT=${TAG#v}
    case "$WANT" in *-rc.*) ;; *) echo "RED  $TAG is not an rc tag (vX.Y.Z-rc.N)"; exit 1 ;; esac
    SHA=$(git rev-parse --short=9 "$TAG^{commit}" 2>/dev/null) || { echo "RED  tag $TAG does not resolve"; exit 1; }
    if check "$WANT" "$SHA" "$RECEIPT" "$LEDGER"; then
      echo "GREEN $TAG ($SHA): installed + measured on: $FLEET"
    else
      echo "RED  $TAG ($SHA): not deployed+measured on the whole fleet"; exit 1
    fi ;;
esac
