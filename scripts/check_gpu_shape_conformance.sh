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
# PER-CAPABILITY EXCLUSIONS (#4096). GPU_QTYPES_UNLOADABLE_AT_CC lists whitelisted types refused on devices of
# compute-capability major >= GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR (read from the same dtype.rs). On such a host
# those types are not required to pass, but they must still be EXERCISED (the harness runs them, marked
# `excluded`), and an exclusion is STALE — RED — once every held shape of the type passes there with a RED
# negative control: the fix has landed and the whitelist is refusing a type that works. An exclusion cannot
# outlive its fix. The host's capability comes from the receipt (`cc_major`, else its `gpu` line).
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
    m = re.search(r"fn gpu_unsupported_quant_qtype_on\([^)]*\)[^{]*\{\s*!matches!\(\s*qtype\s*,([^)]*)\)", src)
    types = {int(t) for t in re.findall(r"\d+", m.group(1))} if m else set()
    return types

def exclusions(dtype_rs):
    # -> (set of excluded types, min cc major) ; (set(), None) when the file declares none; raises when it
    # names the constants but they do not parse (a half-read exclusion must refuse, never read as "none").
    src = Path(dtype_rs).read_text(encoding="utf-8")
    if "GPU_QTYPES_UNLOADABLE_AT_CC" not in src:
        return set(), None
    t = re.search(r"const GPU_QTYPES_UNLOADABLE_AT_CC: \[u32; \d+\] = \[([\d,\s]*)\];", src)
    c = re.search(r"const GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR: i32 = (\d+);", src)
    if not t or not c:
        raise ValueError("GPU_QTYPES_UNLOADABLE_AT_CC / GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR are named but do not parse")
    return {int(x) for x in re.findall(r"\d+", t.group(1))}, int(c.group(1))

def host_cc(r):
    if isinstance(r.get("cc_major"), int):
        return r["cc_major"]
    m = re.search(r",\s*(\d+)\.\d+\s*$", r.get("gpu") or "")
    return int(m.group(1)) if m else None

def klass(k, q):
    per = 1 if q in (0, 1, 30) else (32 if q in (2, 3, 6, 7, 8, 20) else 256)
    spb = k // per
    return "one" if spb == 1 else ("pow2" if spb & (spb - 1) == 0 else "nonpow2")

def check(root):
    root = Path(root)
    errs = []
    wl = whitelist(root / "crates/aprender-serve/src/gguf/dtype.rs")
    if not wl:
        return ["refused: no whitelisted type parsed from gpu_unsupported_quant_qtype_on — this check no longer knows what it guards"]
    try:
        excl, excl_cc = exclusions(root / "crates/aprender-serve/src/gguf/dtype.rs")
    except ValueError as e:
        return [f"refused: {e}"]
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
        tested = {(x["qtype"], x["k"], x["n_held"]) for x in r.get("rows", [])}
        controlled = {int(q) for q, v in (r.get("negative_controls") or {}).items() if v.get("red")}
        here = set()
        if excl:
            cc = host_cc(r)
            if cc is None:
                errs.append(f"{host}: the receipt names no compute capability, so exclusions {sorted(excl)} cannot be applied")
                continue
            here = excl if cc >= excl_cc else set()
        need_here = {x for x in need if x[0] not in here}
        for q, k, n in sorted(need_here - covered):
            errs.append(f"{host}: ggml {q} at k={k} n={n} ({klass(k, q)}) is held and whitelisted but not proven on the device")
        for q in sorted({q for q, _, _ in need_here} - controlled):
            errs.append(f"{host}: ggml {q} has no RED negative control")
        for q in sorted(here & {q for q, _, _ in need}):
            held_q = {x for x in need if x[0] == q}
            if not held_q <= tested:
                errs.append(f"{host}: ggml {q} is EXCLUDED here (cc>={excl_cc}) but the receipt does not exercise it — a stale exclusion would be invisible")
            elif held_q <= covered and q in controlled:
                errs.append(f"{host}: STALE exclusion — ggml {q} is excluded at cc>={excl_cc} but every held shape PASSES here with a RED negative control; remove it from GPU_QTYPES_UNLOADABLE_AT_CC (and the contract's gpu_excluded_from_cc_major)")
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
    printf 'pub(crate) fn gpu_unsupported_quant_qtype_on(qtype: u32, cc_major: Option<i32>) -> bool {\n    !matches!(\n        qtype,\n        12 | 23\n    ) || gpu_qtype_excluded_on(qtype, cc_major)\n}\n' \
      > "$d/crates/aprender-serve/src/gguf/dtype.rs"
    python3 - "$d" <<'FIX'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
census = {"schema": "gguf-census/v1", "host": "h1", "files": [{"file": "m.gguf", "shapes": [
    {"qtype": 12, "k": 2560, "n": 9216}, {"qtype": 23, "k": 3584, "n": 1024}, {"qtype": 11, "k": 512, "n": 512}]}]}
cp = d / "evidence/gpu-shape-census/census-h1.json"
cp.write_text(json.dumps(census))
receipt = {"schema": "gpu-shape-conformance/v1", "harness_rc": 0, "failed": 0, "blind_types": [], "cc_major": 8,
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
elif what.startswith("excl-"):          # #4096: ggml 23 excluded at cc>=12
    rs = d / "crates/aprender-serve/src/gguf/dtype.rs"
    rs.write_text(rs.read_text() + "pub(crate) const GPU_QTYPES_UNLOADABLE_AT_CC: [u32; 1] = [23];\n"
                  "pub(crate) const GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR: i32 = 12;\n")
    row23 = r["rows"][1]
    if what == "excl-below-threshold": pass                       # cc 8: 23 still required, and proven
    elif what == "excl-red-on-cc12":   # the capability is RED there: the row fails, its control cannot fire
        r["cc_major"] = 12; row23["pass"] = False; row23["excluded"] = True; r["negative_controls"]["23"]["red"] = False
    elif what == "excl-stale":         # #4096 fixed: 23 passes on cc 12 with a RED control, still excluded
        r["cc_major"] = 12; row23["excluded"] = True
    elif what == "excl-unexercised":   # the harness skipped the excluded type: staleness would be invisible
        r["cc_major"] = 12; r["rows"].pop(1); del r["negative_controls"]["23"]
    elif what == "excl-no-cc":
        del r["cc_major"]
    elif what == "excl-half-parsed":
        rs.write_text(rs.read_text().replace("i32 = 12", "i32 = twelve"))
if rp is not None and rp.exists(): rp.write_text(json.dumps(r))
ED
  }
  row() { # row <label> <edit> <want-rc> [<want-substring>] — the rc AND, when given, the reason (an rc 1 for
    local d="$TMP/$1" rc out   # an unrelated reason would otherwise pass a RED row)
    mk "$d"; [ "$2" = none ] || edit "$d" "$2"
    out=$(run_check "$d"); rc=$?
    if [ "$rc" = "$3" ] && { [ -z "${4:-}" ] || grep -qF -- "$4" <<< "$out"; }; then echo "  ok    $1 (rc $rc)"
    else echo "  BROKE $1: rc $rc, want $3${4:+ with \"$4\"}"; printf '%s\n' "$out"; fails=1; fi
  }
  row green-every-held-shape-proven      none               0
  row green-unlisted-type-not-required   unlisted-type-held 0
  row RED-new-held-shape-unproven        new-held-shape     1
  row RED-blind-negative-control         blind              1
  row RED-failed-row                     failed             1
  row RED-stale-receipt                  stale              1
  row RED-missing-host-receipt           no-receipt         1
  row REFUSED-no-whitelist-parsed        no-whitelist       2
  row green-exclusion-below-threshold    excl-below-threshold 0
  row green-excluded-type-red-on-cc12    excl-red-on-cc12   0
  row RED-stale-exclusion                excl-stale         1 "STALE exclusion — ggml 23"
  row RED-excluded-type-not-exercised    excl-unexercised   1 "ggml 23 is EXCLUDED here (cc>=12) but the receipt does not exercise it"
  row RED-exclusion-host-cc-unknown      excl-no-cc         1 "names no compute capability"
  row REFUSED-exclusion-half-parsed      excl-half-parsed   2 "do not parse"
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
