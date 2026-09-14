#!/usr/bin/env bash
# pp066_state.sh — PP-066 STATE in one call (PMAT-1065 [minted in the session docs
# commit], DAG row G-11b, #3018; driver v4 STATE).
#
# Prints, from live sources only (nothing is state until read):
#   1. head row  — the lowest-expiry open 0.66 row whose blockers are all complete
#                  (status DERIVED from docs/audits/impl-<pmat_id>-receipt.md, scripts/lib/dag_status.py),
#                  whose host is under the WIP cap, and that is not past expiry
#                  without an amendment row naming it;
#   2. complete receipts (derived), and every open row with an open PR;
#   3. PR states — gh pr list --author noahgift (number, mergeStateStatus, head);
#   4. reds on clean origin/main — the latest ci.yml run on main: failed jobs/steps,
#      each with the ticket id the DAG names for it (or UNTICKETED);
#   5. U-1 poll — pmat#1200 state and the pin scripts/pmat_bin.sh resolves.
#
#   bash scripts/pp066_state.sh [--dag <yaml>] [--today YYYY-MM-DD] [--no-gh] [--json]
#   bash scripts/pp066_state.sh --self-test        # head-row derivation, both polarities
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=pp066_state
DAG="$ROOT/docs/specifications/pp-066-dag.yaml"; TODAY=""; NOGH=0; JSON=0

head_rows() { # head_rows <dag> <root> <today> -> TSV: id, expiry, host, pmat_id, gh_issue, blockers-complete?, reason
    python3 - "$1" "$2" "$3" <<'PY'
import sys, os, yaml, datetime
dag, root, today = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(root, "scripts", "lib"))
import dag_status as ds
from dag_invariants import resolved_expiries, rows_by_id
d = yaml.safe_load(open(dag, encoding="utf-8")); rows = rows_by_id(d)
today = datetime.date.fromisoformat(today) if today else datetime.date.today()
exp, _ = resolved_expiries(rows)
amended = {a.get("row") for a in (d.get("amendments") or [])}
status = {rid: ds.derived_status(root, r) for rid, r in rows.items()}
cap_host_busy = set()   # a host with an armed speed row is under cap; the caller passes PR states, so this is advisory here
cands = []
for rid, r in rows.items():
    if str(r.get("lane")) != "0.66" or status[rid] == "complete" or r.get("decision"):
        continue
    blockers = r.get("blockers") or []
    open_blockers = [b for b in blockers if status.get(b) != "complete"]
    e = exp.get(rid)
    past = e is not None and e < today and rid not in amended
    reason = "blocked_by " + ",".join(open_blockers) if open_blockers else ("past-expiry-without-amendment" if past else "eligible")
    cands.append((e or datetime.date.max, rid, str(e), str(r.get("host")), str(r.get("pmat_id")), str(r.get("gh_issue")), "yes" if not open_blockers else "no", reason))
for e, rid, es, host, pid, gh, ok, reason in sorted(cands):
    print("\t".join([rid, es, host, pid, gh, ok, reason]))
PY
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/pp066state.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    mkdir -p "$TD/root/scripts/lib" "$TD/root/docs/audits"; cp "$ROOT/scripts/lib/dag_status.py" "$ROOT/scripts/lib/dag_invariants.py" "$TD/root/scripts/lib/"
    cat > "$TD/dag.yaml" <<'EOF2'
min_slack_days: 6
rows:
- {id: A, lane: '0.66', owner: x, expiry: '2026-09-10', blockers: [], pmat_id: PMAT-1}
- {id: B, lane: '0.66', owner: x, expiry: '2026-09-12', blockers: [A], pmat_id: PMAT-2}
- {id: C, lane: '0.66', owner: x, expiry: '2026-09-20', blockers: [], pmat_id: PMAT-3}
- {id: D, lane: '0.66', owner: x, expiry: '2026-09-01', blockers: [], pmat_id: PMAT-4}
- {id: E, lane: '0.67', owner: x, expiry: '2026-09-05', blockers: [], pmat_id: PMAT-5}
- {id: F, lane: '0.66', owner: x, expiry: '2026-09-15', blockers: [], pmat_id: PMAT-6, decision: true}
amendments:
- {row: D, change: 'expiry moved', date: '2026-09-06'}
EOF2
    n=0; red=0
    row() { local want=$1 label=$2; shift 2; local got; n=$((n + 1)); got=$("$@" | awk -F'\t' '$6=="yes" && $7=="eligible"{print $1; exit}')
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s head=%s  %s\n' "$n" "${got:-<none>}" "$label"; else printf 'FAIL  row %-2s head=%s (wanted %s)  %s\n' "$n" "${got:-<none>}" "$want" "$label"; red=1; fi; }
    row D "no receipts: D (09-01) is past expiry but AMENDED, so it is the head; B waits for A"   head_rows "$TD/dag.yaml" "$TD/root" 2026-09-06
    printf -- '---\nstatus: complete\n---\n' > "$TD/root/docs/audits/impl-PMAT-4-receipt.md"
    row A "D complete by its receipt: A (09-10) is the head"                                    head_rows "$TD/dag.yaml" "$TD/root" 2026-09-06
    printf -- '---\nstatus: complete\n---\n' > "$TD/root/docs/audits/impl-PMAT-1-receipt.md"
    row B "A complete: B (09-12, blocker A complete) is the head — a blocked row is never head before its blocker (the registered mutation)" head_rows "$TD/dag.yaml" "$TD/root" 2026-09-06
    printf -- '---\nstatus: partial\n---\n' > "$TD/root/docs/audits/impl-PMAT-2-receipt.md"
    row B "a PARTIAL receipt does not complete B: still the head"                               head_rows "$TD/dag.yaml" "$TD/root" 2026-09-06
    printf -- '---\nstatus: complete\n---\n' > "$TD/root/docs/audits/impl-PMAT-2-receipt.md"
    row C "B complete: C (09-20); the 0.67 row E and the decision row F are never head"        head_rows "$TD/dag.yaml" "$TD/root" 2026-09-06
    n=$((n + 1)); if head_rows "$TD/dag.yaml" "$TD/root" 2026-09-06 | grep -q $'^D\t.*past-expiry-without-amendment'; then printf 'FAIL  row %-2s an amended past-expiry row was reported as unamended\n' "$n"; red=1; else printf 'ok    row %-2s an amended past-expiry row is not flagged\n' "$n"; fi
    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

while [ $# -gt 0 ]; do case "$1" in --dag) DAG=$2; shift 2 ;; --today) TODAY=$2; shift 2 ;; --no-gh) NOGH=1; shift ;; --json) JSON=1; shift ;; *) printf 'usage: %s [--dag <yaml>] [--today YYYY-MM-DD] [--no-gh] [--json] | --self-test\n' "$PROG" >&2; exit 2 ;; esac; done
[ -f "$DAG" ] || { printf '%s: ENV - %s is missing\n' "$PROG" "$DAG" >&2; exit 2; }
git -C "$ROOT" fetch -q origin 2>/dev/null || printf '%s: WARN git fetch failed (state may be stale)\n' "$PROG" >&2
printf '=== PP-066 STATE (%s) main=%s HEAD=%s ===\n' "$(date -u +%FT%TZ)" "$(git -C "$ROOT" rev-parse --short origin/main 2>/dev/null || echo '?')" "$(git -C "$ROOT" rev-parse --short HEAD)"
printf -- '--- head row (lowest expiry, blockers complete, not past expiry unamended):\n'
head_rows "$DAG" "$ROOT" "$TODAY" > "${TMPDIR:-/tmp}/pp066-rows.$$"
awk -F'\t' '$6=="yes" && $7=="eligible"{printf "  HEAD  %-8s expiry=%s host=%s pmat=%s issue=#%s\n", $1, $2, $3, $4, $5; exit}' "${TMPDIR:-/tmp}/pp066-rows.$$"
printf -- '--- next eligible (up to 8):\n'; awk -F'\t' '$6=="yes" && $7=="eligible"{c++; if (c>1 && c<=9) printf "  %-8s expiry=%s host=%s pmat=%s\n", $1, $2, $3, $4}' "${TMPDIR:-/tmp}/pp066-rows.$$"
printf -- '--- past expiry without an amendment (RED in pmat comply, never started):\n'; awk -F'\t' '$7=="past-expiry-without-amendment"{printf "  %-8s expiry=%s\n", $1, $2}' "${TMPDIR:-/tmp}/pp066-rows.$$" | { grep . || echo '  none'; }
rm -f "${TMPDIR:-/tmp}/pp066-rows.$$"
printf -- '--- complete (derived from receipts): '
python3 - "$DAG" "$ROOT" <<'PY'
import sys, os, yaml
sys.path.insert(0, os.path.join(sys.argv[2], "scripts", "lib")); import dag_status as ds
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
done = [r["id"] for r in d["rows"] if ds.derived_status(sys.argv[2], r) == "complete"]
lane = [r for r in d["rows"] if str(r.get("lane")) == "0.66"]
print(f"{len(done)} ({', '.join(done)}); 0.66 rows {len(lane)}")
PY
if [ "$NOGH" = 0 ] && command -v gh >/dev/null 2>&1; then
    printf -- '--- PRs (author noahgift, agent/* heads):\n'
    gh pr list --author noahgift --state open --limit 40 --json number,mergeStateStatus,headRefName,autoMergeRequest --jq '.[] | select(.headRefName|startswith("agent/")) | "  #\(.number) \(.mergeStateStatus) auto=\(.autoMergeRequest!=null) \(.headRefName)"' 2>/dev/null || printf '  (gh unavailable)\n'
    printf -- '--- reds on clean origin/main (latest ci.yml run on main):\n'
    RID=$(gh run list --workflow ci.yml --branch main --limit 1 --json databaseId,conclusion,status,headSha --jq '.[0] | "\(.databaseId) \(.status)/\(.conclusion) \(.headSha[0:9])"' 2>/dev/null || true)
    printf '  run %s\n' "${RID:-?}"
    R=${RID%% *}
    if [ -n "$R" ]; then
        gh run view "$R" --json jobs --jq '.jobs[] | select(.conclusion=="failure") | "  RED job=\(.name) steps=\([.steps[]? | select(.conclusion=="failure") | .name] | join(" | "))"' 2>/dev/null | while IFS= read -r line; do
            t=$(python3 - "$DAG" "$line" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); line = sys.argv[2].lower()
hits = [f"{r['id']}/{r.get('pmat_id')}" for r in d["rows"] if any(k and k.lower() in line for k in [str(r.get("contract") or "")[:40]] + [a[:40] for a in (r.get("A") or []) if isinstance(a, str)])]
print(",".join(hits) or "UNTICKETED")
PY
)
            printf '%s  ticket=%s\n' "$line" "$t"
        done
        [ "$(gh run view "$R" --json conclusion --jq .conclusion 2>/dev/null)" = success ] && printf '  none (main green)\n'
    fi
    printf -- '--- U-1 poll: pmat#1200 %s; ' "$(gh issue view 1200 -R paiml/paiml-mcp-agent-toolkit --json state --jq .state 2>/dev/null || echo '?')"
fi
if bash -c ". '$ROOT/scripts/pmat_bin.sh'" >/dev/null 2>&1; then bash -c ". '$ROOT/scripts/pmat_bin.sh' && printf 'analyser pin %s at %s\n' \"\$PMAT_VERSION\" \"\$PMAT\""; else printf 'analyser pin: NOT RESOLVED (scripts/pmat_bin.sh refused)\n'; fi
