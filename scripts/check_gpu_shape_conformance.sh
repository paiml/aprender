#!/usr/bin/env bash
# check_gpu_shape_conformance.sh — no GPU-whitelisted quant type is used at a HELD shape that the device
# harness has not proven on EVERY host (#3968, the test half of #3945).
#
# WHY. The whitelist admits a TYPE; a kernel runs at a SHAPE. IQ4_XS was admitted on one model's shape
# while a held model used eight others (#3951). This joins three things none of which is hand-listed:
#   - the WHITELIST, parsed from gpu_unsupported_quant_qtype in crates/aprender-serve/src/gguf/dtype.rs
#   - the CENSUS, evidence/gpu-shape-census/census-<host>.json (scripts/lib/gguf_census.py)
#   - the RECEIPTS, evidence/gpu-shape-conformance/<host>.json (scripts/gpu_shape_conformance.sh)
# and fails when any held whitelisted (type, k, n) has no passing row in a host's receipt, a receipt is
# missing, stale (it tested a different census), failed, or has a type whose negative control never went
# RED. A parse that finds no whitelisted type, or a census with no rows, is a refusal, never agreement.
#
# --self-test runs the case table against fixtures: the must-RED rows (#3968's own: a held shape of a
# whitelisted type absent from the receipt) and the must-GREEN row.
#
# Exit: 0 every held shape proven on every host · 1 a gap or a broken receipt · 2 could not check.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_gpu_shape_conformance
command -v python3 >/dev/null 2>&1 || { echo "$PROG: ENV - python3 is missing" >&2; exit 2; }

read -r -d '' CHECK <<'PY'
import hashlib, json, re, sys
from pathlib import Path

def whitelist(dtype_rs):
    src = Path(dtype_rs).read_text(encoding="utf-8")
    m = re.search(r"fn gpu_unsupported_quant_qtype\([^)]*\)[^{]*\{\s*!matches!\(\s*qtype\s*,([^)]*)\)", src)
    types = {int(t) for t in re.findall(r"\d+", m.group(1))} if m else set()
    return types

def klass(k, q):
    per = 1 if q in (0, 1, 30) else (32 if q in (2, 3, 6, 7, 8, 20) else 256)
    spb = k // per
    return "one" if spb == 1 else ("pow2" if spb & (spb - 1) == 0 else "nonpow2")

def check(root):
    root = Path(root)
    errs = []
    wl = whitelist(root / "crates/aprender-serve/src/gguf/dtype.rs")
    if not wl:
        return ["refused: no whitelisted type parsed from gpu_unsupported_quant_qtype — this check no longer knows what it guards"]
    censuses = sorted((root / "evidence/gpu-shape-census").glob("*.json"))
    if not censuses:
        return ["refused: no census under evidence/gpu-shape-census"]
    need, hosts, shas = set(), [], {}
    for c in censuses:
        doc = json.loads(c.read_text(encoding="utf-8"))
        hosts.append(doc.get("host") or c.stem.replace("census-", ""))
        shas[c.name] = hashlib.sha256(c.read_bytes()).hexdigest()
        for f in doc.get("files", []):
            for s in f.get("shapes", []):
                if s["qtype"] in wl:
                    need.add((s["qtype"], s["k"], s["n"]))
    if not need:
        return ["refused: the census holds no whitelisted 2-D tensor"]
    for host in hosts:
        rp = root / "evidence/gpu-shape-conformance" / f"{host}.json"
        if not rp.exists():
            errs.append(f"{host}: no receipt ({rp.relative_to(root)}) — run scripts/gpu_shape_conformance.sh {host}")
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        if r.get("schema") != "gpu-shape-conformance/v1":
            errs.append(f"{host}: receipt schema {r.get('schema')!r}"); continue
        if r.get("census") != shas:
            errs.append(f"{host}: STALE receipt — it tested a different census than the tree holds")
        if r.get("harness_rc") != 0 or r.get("failed") != 0:
            errs.append(f"{host}: harness rc {r.get('harness_rc')}, {r.get('failed')} failed rows")
        if r.get("blind_types"):
            errs.append(f"{host}: negative control stayed GREEN for ggml types {r['blind_types']} — their greens license nothing")
        covered = {(x["qtype"], x["k"], x["n_held"]) for x in r.get("rows", []) if x.get("pass")}
        controlled = {int(q) for q, v in (r.get("negative_controls") or {}).items() if v.get("red")}
        for q, k, n in sorted(need - covered):
            errs.append(f"{host}: ggml {q} at k={k} n={n} ({klass(k, q)}) is held and whitelisted but not proven on the device")
        for q in sorted({q for q, _, _ in need} - controlled):
            errs.append(f"{host}: ggml {q} has no RED negative control")
    return errs

if __name__ == "__main__":
    errs = check(sys.argv[1])
    for e in errs:
        print("  " + e)
    sys.exit(2 if errs and errs[0].startswith("refused") else (1 if errs else 0))
PY

run_check() { python3 -c "$CHECK" "$1"; }

if [ "${1:-}" = "--self-test" ]; then
  TMP=$(mktemp -d) || exit 2
  trap 'rm -rf "$TMP"' EXIT
  fails=0
  mk() { # mk <dir> : a minimal green tree — whitelist {12, 23}, one host, every held shape proven
    local d="$1"
    mkdir -p "$d/crates/aprender-serve/src/gguf" "$d/evidence/gpu-shape-census" "$d/evidence/gpu-shape-conformance"
    printf 'pub(crate) fn gpu_unsupported_quant_qtype(qtype: u32) -> bool {\n    !matches!(\n        qtype,\n        12 | 23\n    )\n}\n' \
      > "$d/crates/aprender-serve/src/gguf/dtype.rs"
    python3 - "$d" <<'FIX'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
census = {"schema": "gguf-census/v1", "host": "h1", "files": [{"file": "m.gguf", "shapes": [
    {"qtype": 12, "k": 2560, "n": 9216}, {"qtype": 23, "k": 3584, "n": 1024}, {"qtype": 11, "k": 512, "n": 512}]}]}
cp = d / "evidence/gpu-shape-census/census-h1.json"
cp.write_text(json.dumps(census))
receipt = {"schema": "gpu-shape-conformance/v1", "harness_rc": 0, "failed": 0, "blind_types": [],
           "census": {cp.name: hashlib.sha256(cp.read_bytes()).hexdigest()},
           "rows": [{"qtype": 12, "k": 2560, "n_held": 9216, "pass": True}, {"qtype": 23, "k": 3584, "n_held": 1024, "pass": True}],
           "negative_controls": {"12": {"red": True}, "23": {"red": True}}}
(d / "evidence/gpu-shape-conformance/h1.json").write_text(json.dumps(receipt))
FIX
  }
  edit() { python3 - "$1" "$2" <<'ED'
import hashlib, json, sys
from pathlib import Path
d, what = Path(sys.argv[1]), sys.argv[2]
cp, rp = d / "evidence/gpu-shape-census/census-h1.json", d / "evidence/gpu-shape-conformance/h1.json"
c, r = json.loads(cp.read_text()), json.loads(rp.read_text())
if what == "new-held-shape":      # #3968's must-RED: a held model with an uncovered shape of a whitelisted type
    c["files"].append({"file": "new.gguf", "shapes": [{"qtype": 23, "k": 2560, "n": 9216}]})
    cp.write_text(json.dumps(c)); r["census"] = {cp.name: hashlib.sha256(cp.read_bytes()).hexdigest()}
elif what == "blind":   r["blind_types"] = [23]; r["negative_controls"]["23"]["red"] = False
elif what == "failed":  r["failed"] = 1; r["rows"][0]["pass"] = False
elif what == "stale":   c["files"][0]["shapes"].append({"qtype": 11, "k": 256, "n": 256}); cp.write_text(json.dumps(c))
elif what == "no-receipt": rp.unlink(); rp = None
elif what == "no-whitelist": (d / "crates/aprender-serve/src/gguf/dtype.rs").write_text("fn other() {}\n")
elif what == "unlisted-type-held": pass  # ggml 11 is held but not whitelisted: must NOT be required
if rp is not None and rp.exists(): rp.write_text(json.dumps(r))
ED
  }
  row() { # row <label> <edit> <want-rc>
    local d="$TMP/$1" rc out
    mk "$d"; [ "$2" = none ] || edit "$d" "$2"
    out=$(run_check "$d"); rc=$?
    if [ "$rc" = "$3" ]; then echo "  ok    $1 (rc $rc)"; else echo "  BROKE $1: rc $rc, want $3"; printf '%s\n' "$out"; fails=1; fi
  }
  row green-every-held-shape-proven      none               0
  row green-unlisted-type-not-required   unlisted-type-held 0
  row RED-new-held-shape-unproven        new-held-shape     1
  row RED-blind-negative-control         blind              1
  row RED-failed-row                     failed             1
  row RED-stale-receipt                  stale              1
  row RED-missing-host-receipt           no-receipt         1
  row REFUSED-no-whitelist-parsed        no-whitelist       2
  [ "$fails" -eq 0 ] && { echo "$PROG --self-test: PASS"; exit 0; }
  echo "$PROG --self-test: FAIL"; exit 1
fi

out=$(run_check "$ROOT"); rc=$?
printf '%s\n' "$out"
case $rc in
  0) echo "$PROG: PASS — every held whitelisted (type, k, n) is proven on every host" ;;
  1) echo "$PROG: FAIL" ;;
  *) echo "$PROG: could not check" ;;
esac
exit "$rc"
