#!/usr/bin/env bash
# check_arch_constraints_contract_covers_fallback.sh — the in-tree contract
# contracts/arch-constraints-v1.yaml must name every architecture that
# crates/aprender-serve/src/gguf/arch_constraints_fallback.rs names, with the
# same fields.
#
# WHY. build.rs generates the ArchConstraints table from the YAML when it can
# find it and from the fallback when it cannot. Until 2026-09-18 it looked for
# the YAML at ../../../provable-contracts/… — a PRE-MONOREPO sibling checkout —
# so CI (clean clone) shipped the fallback, while every dev box that still had
# the archived repo built from a stale April YAML with no MoE entry. Four
# apr-cli parity_refusal tests failed only on dev boxes, which is exactly where
# the T-2 coverage row runs (EPIC #3477). One source is not enough when two
# exist and nothing compares them; this guard is the comparison.
#
# Usage: bash scripts/check_arch_constraints_contract_covers_fallback.sh [--self-test]
# Exit: 0 covered · 1 a fallback arm is missing or a field disagrees · 2 unreadable.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 2
YAML="contracts/arch-constraints-v1.yaml"
FB="crates/aprender-serve/src/gguf/arch_constraints_fallback.rs"
case "${1:-}" in -h|--help) sed -n '2,19p' "$0"; exit 0 ;; esac

compare() { # compare <yaml> <fallback.rs> → prints rows, exit 0/1/2
  python3 - "$1" "$2" <<'PY'
import re, sys, yaml
yp, fp = sys.argv[1], sys.argv[2]
try:
    fb = open(fp).read(); Y = yaml.safe_load(open(yp))["architectures"]
except Exception as e:
    print(f"decline: unreadable: {e}"); sys.exit(2)
arms = re.findall(r'((?:"[^"]+"\s*\|\s*)*"[^"]+")\s*=>\s*ArchConstraints\s*\{(.*?)\}', fb, re.S)
if not arms: print("decline: no ArchConstraints arms found in the fallback — the regex matched nothing"); sys.exit(2)
ykeys = {}
for name, v in Y.items():
    ykeys[name] = (name, v)
    for a in (v.get("aliases") or []): ykeys[a] = (name, v)
rc = 0; n = 0
for keys_s, body in arms:
    keys = re.findall(r'"([^"]+)"', keys_s)
    f = dict(re.findall(r'(\w+):\s*([^,]+),', body))
    for k in keys:
        n += 1
        if k not in ykeys:
            print(f"FAIL  fallback names '{k}' and the contract does not (add it, with aliases)"); rc = 1; continue
        name, v = ykeys[k]
        for fld in ("has_bias", "tied_embeddings", "has_qk_norm"):
            if str(v.get(fld)).lower() != f.get(fld):
                print(f"FAIL  {k} ({name}): {fld} contract={v.get(fld)} fallback={f.get(fld)}"); rc = 1
        fb_moe = f.get("is_moe") == "true"
        if bool(v.get("is_moe", False)) != fb_moe:
            print(f"FAIL  {k} ({name}): is_moe contract={bool(v.get('is_moe', False))} fallback={fb_moe}"); rc = 1
if n == 0: print("decline: zero fallback keys compared"); sys.exit(2)
print(f"{'ok' if rc == 0 else 'RED'}    {n} fallback keys checked against {len(Y)} contract architectures ({len(ykeys)} keys incl. aliases)")
sys.exit(rc)
PY
}

if [ "${1:-}" = "--self-test" ]; then
  bad=0; t=$(mktemp -d)
  # 1. the real pair is green
  compare "$YAML" "$FB" >/dev/null; rc=$?; [ $rc -eq 0 ] && echo "ok    real pair: covered (rc=0)" || { echo "FAIL  real pair rc=$rc"; bad=1; }
  # 2. mutation: drop the MoE entry from a copy → RED naming qwen3moe
  python3 - "$YAML" "$t/no-moe.yaml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); d["architectures"].pop("qwen3_moe", None); yaml.safe_dump(d, open(sys.argv[2], "w"))
PY
  out=$(compare "$t/no-moe.yaml" "$FB"); rc=$?
  if [ $rc -eq 1 ] && grep -q "qwen3moe" <<< "$out"; then echo "ok    mutation dropped qwen3_moe: RED names qwen3moe (rc=1)"; else echo "FAIL  mutation dropped qwen3_moe: rc=$rc"; printf '%s\n' "$out" | sed 's/^/        /'; bad=1; fi
  # 3. mutation: flip is_moe off → RED on the field
  python3 - "$YAML" "$t/moe-off.yaml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); d["architectures"]["qwen3_moe"]["is_moe"] = False; yaml.safe_dump(d, open(sys.argv[2], "w"))
PY
  out=$(compare "$t/moe-off.yaml" "$FB"); rc=$?
  if [ $rc -eq 1 ] && grep -q "is_moe contract=False fallback=True" <<< "$out"; then echo "ok    mutation is_moe=false: RED on the field (rc=1)"; else echo "FAIL  mutation is_moe=false: rc=$rc"; bad=1; fi
  # 4. an empty fallback is a decline, never a pass
  : > "$t/empty.rs"; compare "$YAML" "$t/empty.rs" >/dev/null; rc=$?
  [ $rc -eq 2 ] && echo "ok    empty fallback: decline (rc=2)" || { echo "FAIL  empty fallback rc=$rc (want 2)"; bad=1; }
  case "$t" in /tmp/?*|/var/folders/?*|/mnt/?*) if [ -n "$t" ] && [ "$t" != "/" ]; then rm -rf -- "$t" || :; fi ;; *) : ;; esac
  echo "self-test: $([ $bad -eq 0 ] && echo PASS || echo FAIL)"; exit $bad
fi
compare "$YAML" "$FB"
