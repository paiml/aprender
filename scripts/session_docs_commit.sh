#!/usr/bin/env bash
# session_docs_commit.sh — the ONE batched docs commit per session (DAG row G-11b,
# #3018; driver v4 SESSION END, I13: shared files are written once per session).
#
# On the orchestrator branch (agent/pp-066-*), in order, then one commit:
#   1. pmat work add   for DAG rows whose pmat_id is null and gh_issue is set (rows created this session)
#   2. pmat work complete for rows whose receipt says status: complete (derived) while the roadmap
#      entry is not `completed` — proof: the merged PR and the receipt path
#   3. README counts regenerated to the measured values (scripts/check_readme_claims.sh --regen)
#      and verified with --exact
#   4. docs/audits/pp-066-status-<date>.md refreshed from scripts/pp066_state.sh
#   5. one kaizen line per prompt defect into docs/audits/driver-kaizen.md (from --kaizen "<line>")
# Every write goes through the pinned analyser (scripts/pmat_bin.sh) where pmat is used.
#
#   bash scripts/session_docs_commit.sh --dry-run [--kaizen "<line>"]...   # list the edits, write nothing
#   bash scripts/session_docs_commit.sh [--kaizen "<line>"]... [--arm]     # write, commit; --arm opens the PR with auto-merge
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=session_docs_commit
DRY=0; ARM=0; KAIZEN=()
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1; shift ;; --arm) ARM=1; shift ;; --kaizen) KAIZEN+=("$2"); shift 2 ;; *) printf 'usage: %s [--dry-run] [--arm] [--kaizen "<line>"]...\n' "$PROG" >&2; exit 2 ;; esac; done
BR=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
case "$BR" in agent/pp-066-*|agent/pr-triage*) ;; *) printf '%s: refused — %s is not an orchestrator branch (agent/pp-066-*); shared files are written there only (G-11)\n' "$PROG" "$BR" >&2; exit 1 ;; esac
DAG="$ROOT/docs/specifications/pp-066-dag.yaml"; RM="$ROOT/docs/roadmaps/roadmap.yaml"; README="$ROOT/README.md"
DATE=$(date -u +%F)
say() { printf '%s\n' "$*"; }

# 1+2: what the DAG and the receipts say vs the roadmap
python3 - "$DAG" "$RM" "$ROOT" > "${TMPDIR:-/tmp}/sdc-plan.$$" <<'PY'
import sys, os, yaml
dag, rm, root = sys.argv[1:4]
sys.path.insert(0, os.path.join(root, "scripts", "lib")); import dag_status as ds
d = yaml.safe_load(open(dag, encoding="utf-8")); r = yaml.safe_load(open(rm, encoding="utf-8"))
entries = {e["id"]: e for e in (r.get("roadmap") or [])}
for row in d["rows"]:
    pid, gh = row.get("pmat_id"), row.get("gh_issue")
    if not pid and gh:
        print(f"ADD\t{row['id']}\t#{gh}\t{(row.get('title') or '')[:100]}")
    elif pid and ds.derived_status(root, row) == "complete" and entries.get(pid, {}).get("status") != "completed":
        print(f"COMPLETE\t{row['id']}\t{pid}\tdocs/audits/impl-{pid}-receipt.md")
    elif pid and pid not in entries:
        # the DAG pre-assigned the id (the orchestrator is the only minter; pmat work add mints colliding ids, pmat#1169): mint BY HAND with that id
        print(f"MINT\t{row['id']}\t{pid}\t#{gh}\t{(row.get('title') or '')[:140]}")
PY
say "=== $PROG on $BR ($( [ "$DRY" = 1 ] && echo DRY-RUN || echo WRITE )) ==="
say "--- 1. pmat work add (rows with an issue and no ticket):"; grep -c '^ADD' "${TMPDIR:-/tmp}/sdc-plan.$$" | sed 's/^/  count: /'; grep '^ADD' "${TMPDIR:-/tmp}/sdc-plan.$$" | sed 's/^/  /' || true
say "--- 1b. mint by hand (the DAG pre-assigned the id):"; grep '^MINT' "${TMPDIR:-/tmp}/sdc-plan.$$" | sed 's/^/  /' || say '  none'
say "--- 2. pmat work complete (receipt complete, roadmap not):"; grep '^COMPLETE' "${TMPDIR:-/tmp}/sdc-plan.$$" | sed 's/^/  /' || say '  none'
say "--- 3. README counts (measured):"; bash "$ROOT/scripts/check_readme_claims.sh" --regen 2>/dev/null | sed 's/^/  /'
say "--- 4. status doc: docs/audits/pp-066-status-$DATE.md (from scripts/pp066_state.sh)"
say "--- 5. kaizen lines: ${#KAIZEN[@]}"; for k in ${KAIZEN[@]+"${KAIZEN[@]}"}; do say "  - $k"; done
if [ "$DRY" = 1 ]; then rm -f "${TMPDIR:-/tmp}/sdc-plan.$$"; exit 0; fi

# ---- writes ----
. "$ROOT/scripts/pmat_bin.sh" || { printf '%s: ENV - no analyser at the pin; refusing to mint\n' "$PROG" >&2; exit 2; }
while IFS=$'\t' read -r kind rid a b c; do
    case "$kind" in
        ADD) "$PMAT" work add "$rid: $b (PP-066 #2873, issue $a)" >/dev/null 2>&1 || { printf '%s: pmat work add failed for %s — mint by hand (pmat#1169)\n' "$PROG" "$rid" >&2; } ;;
        MINT) python3 - "$RM" "$a" "$b" "$rid" "$c" <<'PY'
import sys, datetime
rm, pid, gh, rid, title = sys.argv[1:6]
s = open(rm, encoding="utf-8").read()
if f"- id: {pid}\n" in s: sys.exit(0)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
t = title.replace("'", "''")
entry = f"- id: {pid}\n  github_issue: {gh.lstrip('#') or 'null'}\n  item_type: task\n  title: '{rid}: {t}'\n  status: planned\n  priority: medium\n  assigned_to: null\n  created: {now}\n  updated: {now}\n  spec: null\n  acceptance_criteria:\n  - 'see issue {gh} (A + mutation)'\n  phases: []\n  subtasks: []\n  estimated_effort: null\n  labels:\n  - kind:code\n  - pp-066\n  notes: null\n"
open(rm, "w", encoding="utf-8").write(s.rstrip("\n") + "\n" + entry)
PY
        ;;
        COMPLETE) python3 - "$RM" "$a" "$b" <<'PY'
import re, sys
rm, pid, receipt = sys.argv[1:4]
s = open(rm, encoding="utf-8").read()
m = re.search(rf"^- id: {pid}\n(?:  .*\n)+", s, re.M)
if not m: sys.exit(0)
b = m.group(0)
nb = re.sub(r"^  status: .*$", "  status: completed", b, count=1, flags=re.M)
note = f"complete: proof:{receipt}"
nb = re.sub(r"^  notes: null$", f"  notes: '{note}'", nb, count=1, flags=re.M) if re.search(r"^  notes: null$", nb, re.M) else (re.sub(r"^  notes: '(.*)'$", lambda mm: f"  notes: '{mm.group(1)} | {note}'", nb, count=1, flags=re.M) if re.search(r"^  notes: '", nb, re.M) else nb + f"  notes: '{note}'\n")
open(rm, "w", encoding="utf-8").write(s.replace(b, nb, 1))
PY
        ;;
    esac
done < "${TMPDIR:-/tmp}/sdc-plan.$$"
rm -f "${TMPDIR:-/tmp}/sdc-plan.$$"
bash "$ROOT/scripts/check_roadmap_diff_additive.sh" >/dev/null || { printf '%s: the roadmap diff is not additive; refusing\n' "$PROG" >&2; exit 1; }
# 3. README counts: regenerate the three claims-table numbers from the measurement, then verify exactly
crates=$(bash "$ROOT/scripts/check_readme_claims.sh" --regen 2>/dev/null | awk '/workspace members:/{print $3}')
contracts=$(bash "$ROOT/scripts/check_readme_claims.sh" --regen 2>/dev/null | awk '/contracts\/ \*\.yaml:/{print $3}')
[ -n "$crates" ] && sed -i -E "s/\*\*[0-9]+\*\* workspace crates/**${crates}** workspace crates/" "$README"
[ -n "$contracts" ] && sed -i -E "s/\*\*[0-9]+\*\* provable contracts/**${contracts}** provable contracts/; s/^([0-9]+) contracts across/${contracts} contracts across/" "$README"
README_EXACT=1 bash "$ROOT/scripts/check_readme_claims.sh" --claim crate_count >/dev/null && README_EXACT=1 bash "$ROOT/scripts/check_readme_claims.sh" --claim contract_count >/dev/null || { printf '%s: README counts not exact after regeneration\n' "$PROG" >&2; exit 1; }
# 4. status doc
bash "$ROOT/scripts/pp066_state.sh" > "$ROOT/docs/audits/pp-066-status-$DATE.md.tmp" 2>&1 || true
{ printf '# PP-066 status — %s (scripts/pp066_state.sh, one call)\n\n```\n' "$DATE"; cat "$ROOT/docs/audits/pp-066-status-$DATE.md.tmp"; printf '```\n'; } > "$ROOT/docs/audits/pp-066-status-$DATE.md.new"
mv "$ROOT/docs/audits/pp-066-status-$DATE.md.new" "$ROOT/docs/audits/pp-066-status-$DATE.md"; rm -f "$ROOT/docs/audits/pp-066-status-$DATE.md.tmp"
# 5. kaizen
if [ "${#KAIZEN[@]}" -gt 0 ]; then
    if [ ! -f "$ROOT/docs/audits/driver-kaizen.md" ]; then
        printf '# Driver kaizen — one line per prompt defect met\n\n' > "$ROOT/docs/audits/driver-kaizen.md"
    fi
    for k in "${KAIZEN[@]}"; do
        printf -- '- %s: %s\n' "$DATE" "$k" >> "$ROOT/docs/audits/driver-kaizen.md"
    done
fi
# re-render and commit
python3 "$ROOT/scripts/render_dag.py" render > "${TMPDIR:-/tmp}/sdc-block.$$"
python3 - "$ROOT/docs/specifications/PP-066-release-spec.md" "${TMPDIR:-/tmp}/sdc-block.$$" <<'PY'
import sys
p, blk = sys.argv[1:3]; s = open(p, encoding="utf-8").read(); b = open(blk, encoding="utf-8").read()
B = "<!-- dag:table:begin (rendered by scripts/render_dag.py; do not edit by hand) -->"; E = "<!-- dag:table:end -->"
i = s.index(B); j = s.index(E, i) + len(E) + 1; open(p, "w", encoding="utf-8").write(s[:i] + b + s[j:])
PY
rm -f "${TMPDIR:-/tmp}/sdc-block.$$"
git -C "$ROOT" add docs/roadmaps/roadmap.yaml README.md "docs/audits/pp-066-status-$DATE.md" docs/specifications/PP-066-release-spec.md docs/audits/driver-kaizen.md 2>/dev/null || true
git -C "$ROOT" commit -q -m "docs(PP-066): session docs commit $DATE — tickets minted/completed from the DAG and the receipts, README counts exact, status doc, kaizen" -m "Pmat-Ticket: PMAT-966" && say "committed $(git -C "$ROOT" rev-parse --short HEAD)"
if [ "$ARM" = 1 ]; then git -C "$ROOT" push -q -u origin "$BR" && gh pr create --fill --base main --head "$BR" >/dev/null 2>&1 && gh pr merge --squash --auto "$BR" >/dev/null 2>&1 && say "armed"; fi
