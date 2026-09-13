#!/usr/bin/env bash
# check_kind_labels_derived.sh — every PP-066 ticket's `kind:` label is DERIVED from its DAG
# row, never typed (PMAT-1093; blocker found by paiml-implement's kind-gate, AUTO-IMPL-SKILL-001 T-1).
#
# WHY THIS EXISTS
# ---------------
# `paiml-implement` refuses at Phase 0 with `kind missing on <ticket>` unless the roadmap
# entry carries `kind:<code|triage|docs|measurement>`. Measured 2026-09-08:
#
#     PP-066 tickets in docs/roadmaps/roadmap.yaml : 105
#     carrying a kind: label                       :  21
#     NOT carrying one                             :  84
#
# So the skill was refused on 80% of the epic. Hand-typing 84 labels would fix today and rot
# tomorrow the way every hand-maintained list in this repo has: the label would be a second
# statement of something the DAG already says.
#
# THE RULE (one place, both directions)
# --------------------------------------
# For a DAG row that carries a `pmat_id`:
#   * id starts SPEC- / DEC- / TAG-                    -> docs   (a spec, a decision, the cut)
#   * any acceptance `A_i` is COMMAND-SHAPED           -> code
#   * else the row carries a `contract:`               -> code
#   * else                                             -> UNDECIDABLE, and this guard says so
#
# "Command-shaped" is: the line's first token is a bare command name or a path
# (`^[.]?[a-z0-9_./-]+`), and the line does not open like prose (`one `, `the `, `a `,
# `every `, `no `, `two `, `three `, `all `). That test is deliberately crude and its
# residue is REPORTED rather than guessed: on the 2026-09-08 tree it leaves exactly two
# rows undecidable, and both are findings in their own right —
#   G-2  "one line in spec §0 with decided_by and date"        -> a docs row, correctly prose
#   R-8  "the workflow is green once on all four hosts"        -> a PROSE acceptance on a code
#        row, which is the "a prose test: never runs" defect this repo has met before.
# An undecidable row is listed with its A_i so it can be fixed in the DAG, never labelled by
# this script.
#
#   bash scripts/check_kind_labels_derived.sh              # verdict
#   bash scripts/check_kind_labels_derived.sh --update     # write the derived labels
#   bash scripts/check_kind_labels_derived.sh --self-test  # case table, both polarities
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_kind_labels_derived
DAG="${KIND_DAG:-$ROOT/docs/specifications/pp-066-dag.yaml}"
RM="${KIND_ROADMAP:-$ROOT/docs/roadmaps/roadmap.yaml}"

engine() { # engine <dag> <roadmap> <mode: check|update|list-undecidable>
python3 - "$1" "$2" "$3" <<'PY'
import sys, yaml, re
dag_p, rm_p, mode = sys.argv[1:4]
CMD   = re.compile(r'^[.]?[a-z0-9_./-]+(\s|$)')
PROSE = re.compile(r'^(one |the |a |every |no |two |three |all )', re.I)

def execish(a):
    a = a.strip()
    return bool(CMD.match(a)) and not PROSE.match(a)

def derive(r):
    rid = str(r['id'])
    if rid.startswith(('SPEC-', 'DEC-', 'TAG-')):
        return 'docs'
    if any(execish(str(x)) for x in (r.get('A') or [])):
        return 'code'
    if r.get('contract'):
        return 'code'
    return None

dag = yaml.safe_load(open(dag_p, encoding='utf-8'))
want, undecidable = {}, []
for r in dag.get('rows') or []:
    pid = r.get('pmat_id')
    if not pid:
        continue
    k = derive(r)
    if k is None:
        undecidable.append((r['id'], pid, [str(x)[:70] for x in (r.get('A') or ['-'])][:2]))
    else:
        want[pid] = k

if mode == 'list-undecidable':
    for rid, pid, A in undecidable:
        print(f"UNDECIDABLE {rid} ({pid}): {A}")
    sys.exit(0)

# The roadmap is PARSED, never regexed, for the verdict. An earlier draft of this script
# read it by line regex in both directions; its --update then corrupted the file (the 284
# entries whose labels are the INLINE `labels: []` have no `  - ` block to scan, so the
# insert landed after a flow sequence and the YAML stopped loading) and --check reported
# `missing=0 wrong=0` on the wreckage, because a regex reader cannot see a parse error.
# A writer that can break the file its own checker reads is the defect this guard exists
# to prevent, one level up.
try:
    rm_doc = yaml.safe_load(open(rm_p, encoding='utf-8'))
except yaml.YAMLError as e:
    print(f'ENV   {rm_p} does not parse: {e}')
    sys.exit(2)
by_id = {e.get('id'): e for e in (rm_doc.get('roadmap') or []) if isinstance(e, dict)}
lines = open(rm_p, encoding='utf-8').read().split('\n')
entries, cur = {}, None
for i, l in enumerate(lines):
    m = re.match(r'^- id: (\S+)\s*$', l)
    if m:
        cur = m.group(1); entries[cur] = {'start': i, 'labels': None, 'labels_end': None, 'inline': False}
    elif cur and re.match(r'^  labels:', l):
        entries[cur]['labels'] = i
        if re.match(r'^  labels:\s*\[\s*\]\s*$', l):
            entries[cur]['inline'] = True
            entries[cur]['labels_end'] = i + 1
        else:
            j = i + 1
            while j < len(lines) and re.match(r'^  - ', lines[j]):
                j += 1
            entries[cur]['labels_end'] = j

wrong, missing = [], []
for pid, k in sorted(want.items()):
    e = entries.get(pid)
    if e is None:
        continue                      # a DAG pmat_id with no roadmap entry is another guard's job
    have = [str(x) for x in (by_id.get(pid, {}).get('labels') or [])]
    kinds = [h for h in have if h.startswith('kind:')]
    if not kinds:
        missing.append((pid, k))
    elif kinds[0] != f'kind:{k}':
        wrong.append((pid, kinds[0], f'kind:{k}'))

if mode == 'check':
    for pid, k in missing:
        print(f'MISSING  {pid}: derived kind:{k}, roadmap carries none')
    for pid, got, k in wrong:
        print(f'WRONG    {pid}: roadmap says {got}, the DAG derives {k}')
    for rid, pid, A in undecidable:
        print(f'REPORT   {rid} ({pid}) undecidable — no command-shaped A_i: {A}')
    print(f'kind-labels: derived={len(want)} missing={len(missing)} wrong={len(wrong)} undecidable={len(undecidable)}')
    sys.exit(1 if (missing or wrong) else 0)

# update: insert or correct the kind: label, touching nothing else
out, i = [], 0
changed = 0
while i < len(lines):
    l = lines[i]
    m = re.match(r'^- id: (\S+)\s*$', l)
    out.append(l)
    if m and m.group(1) in want:
        pid = m.group(1); k = f"kind:{want[pid]}"
        e = entries[pid]
        if e['labels'] is None:
            i += 1
            continue
        # copy through to the labels line
        while i + 1 <= e['labels']:
            i += 1
            out.append(lines[i])
        if e['inline']:
            out[-1] = '  labels:'          # `labels: []` becomes a block, then the one label
            out.append(f'  - {k}')
        else:
            body = [lines[x] for x in range(e['labels'] + 1, e['labels_end'])]
            body = [b for b in body if not b[4:].strip().startswith('kind:')]
            body.insert(0, f'  - {k}')
            out.extend(body)
        changed += 1
        i = e['labels_end']
        continue
    i += 1
new_text = '\n'.join(out)
try:
    after = yaml.safe_load(new_text)
    assert isinstance(after, dict) and len(after.get('roadmap') or []) == len(rm_doc.get('roadmap') or [])
except Exception as e:
    print(f'REFUSED: the rewrite does not parse or lost entries ({e}); {rm_p} left untouched')
    sys.exit(1)
open(rm_p, 'w', encoding='utf-8').write(new_text)
print(f'kind-labels: wrote {changed} entr(ies)')
PY
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/kindlbl.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    n=0; red=0
    row() { local want=$1 label=$2; shift 2; local rc=0; n=$((n+1)); "$@" > "$TD/o.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s %s\n' "$n" "$label"
        else printf 'FAIL  row %-2s %s (rc=%s want %s)\n        %s\n' "$n" "$label" "$rc" "$want" "$(tail -2 "$TD/o.$n")"; red=1; fi; }
    cat > "$TD/dag.yaml" <<'EOF2'
rows:
- {id: X-1, pmat_id: PMAT-1, A: ['bash scripts/x.sh exits 0']}
- {id: SPEC-9, pmat_id: PMAT-2, A: ['one line in the spec']}
- {id: X-2, pmat_id: PMAT-3, A: ['the workflow is green once'], contract: contracts/c.yaml}
- {id: X-3, pmat_id: PMAT-4, A: ['the workflow is green once']}
EOF2
    mk()  { printf -- '- id: %s\n  labels:\n  - pp-066\n' "$1"; }
    mki() { printf -- '- id: %s\n  labels: []\n' "$1"; }   # the INLINE form: 284 of the real entries
    { printf 'roadmap:\n'; mk PMAT-1; mki PMAT-2; mk PMAT-3; mki PMAT-4; } | sed 's/^-/-/' > "$TD/rm.yaml"
    row 1 "an unlabelled roadmap is MISSING, not a pass"        engine "$TD/dag.yaml" "$TD/rm.yaml" check
    row 0 "--update writes the derived labels"                  engine "$TD/dag.yaml" "$TD/rm.yaml" update
    # An UNDECIDABLE row is a REPORT, not a failure: this guard refuses to guess a label, and
    # refusing to guess is not the same as refusing the tree. It must still be PRINTED, or the
    # residue disappears — which is the whole reason the crude test is allowed to be crude.
    row 0 "an undecidable row does NOT fail the guard (it refuses to guess, it does not refuse the tree)" engine "$TD/dag.yaml" "$TD/rm.yaml" check
    if engine "$TD/dag.yaml" "$TD/rm.yaml" check 2>&1 | grep -q "REPORT   X-3"; then
        printf 'ok    row %-2s the undecidable row is REPORTED by name, never dropped\n' "$((n+1))"
    else
        printf 'FAIL  row %-2s the undecidable row vanished from the output\n' "$((n+1))"; red=1
    fi; n=$((n+1))
    grep -q 'kind:code' "$TD/rm.yaml" && printf 'ok    row 4  a command-shaped A_i derived kind:code\n' || { printf 'FAIL  row 4\n'; red=1; }; n=$((n+1))
    grep -q 'kind:docs' "$TD/rm.yaml" && printf 'ok    row 5  a SPEC- row derived kind:docs\n' || { printf 'FAIL  row 5\n'; red=1; }; n=$((n+1))
    # a WRONG label is refused, not silently kept
    sed -i 's/  - kind:code/  - kind:docs/' "$TD/rm.yaml"
    row 1 "a label that disagrees with the DAG is WRONG"        engine "$TD/dag.yaml" "$TD/rm.yaml" check
    printf '%s/%s rows\n' "$((n-red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

case "${1:-}" in
    --update) engine "$DAG" "$RM" update ;;
    --list-undecidable) engine "$DAG" "$RM" list-undecidable ;;
    ''|--check) engine "$DAG" "$RM" check ;;
    *) printf 'usage: %s [--check|--update|--list-undecidable|--self-test]\n' "$PROG" >&2; exit 2 ;;
esac
