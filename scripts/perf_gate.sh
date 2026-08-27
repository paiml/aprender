#!/usr/bin/env bash
# perf_gate.sh — APR-PERF-GATE-001 v2.2 §4, the release performance gate.
#
#   scripts/perf_gate.sh --host <name> --phase {merge|release} \
#                        --workload {W1|W2} --receipt <path>
#   scripts/perf_gate.sh --selftest
#
# WHY THIS EXISTS. On 2026-08-25 apr measured 0.097x llama.cpp aggregate at
# c=16 while per-user decode read 1.554x. A gate reading only decode scores
# that a comfortable PASS. Arms B1 and B2 are therefore BOTH required and
# neither substitutes for the other.
#
# WHAT THIS DOES NOT DO. It does not re-implement receipt validation: that is
# scripts/lib/bench_receipt.py, which is the single schema authority. This
# script computes the ARMS and renders the verdict.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="$ROOT/scripts/perf-matrix.yaml"

die() { printf 'perf_gate: %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------- arms -----
# Every arm returns 0 (pass) or 1 (fail) and prints one PASS/FAIL line. The
# verdict is the MIN over arms (§4.8: exactly one verdict function, H11).

arm_c_integrity() {
  local receipt="$1" rc=0
  python3 "$ROOT/scripts/lib/bench_receipt.py" "$receipt" >/dev/null 2>&1 \
    || { echo "FAIL ArmC schema: bench_receipt.py rejected the receipt"; rc=1; }
  python3 - "$receipt" <<'PY' || rc=1
import json,sys
r=json.load(open(sys.argv[1]))
bad=[]
req,comp=r.get("requested"),r.get("completed")
if req is None or comp is None or req!=comp:
    bad.append(f"completed({comp}) != requested({req})")
if r.get("timeouts",0)!=0:
    bad.append(f"timeouts={r.get('timeouts')} (fatal to this host's ratio)")
if not (r.get("tokenization") or {}).get("method"):
    bad.append("tokenization.method absent")
if r.get("drain_ms") is None:
    bad.append("drain_ms absent")
for b in (r.get("bands") or []):
    if b.get("tokens_total",1)==0:
        bad.append(f"band {b.get('concurrency')}: zero-token response is a failure, not a fast request")
for m in bad: print(f"FAIL ArmC {m}")
sys.exit(1 if bad else 0)
PY
  [ "$rc" = 0 ] && echo "PASS ArmC integrity"
  return "$rc"
}

arm_a_scaling() {
  # scaling_efficiency(c) = (agg(c)/agg(1))/c, ratchet-up-only per host+band.
  local receipt="$1" host="$2" workload="$3"
  python3 - "$receipt" "$MATRIX" "$host" "$workload" <<'PY'
import json,sys,yaml
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2]))
host,wl=sys.argv[3],sys.argv[4]
import os,datetime
# Injectable so the selftest can prove BOTH sides of the expiry boundary without
# waiting for a date to arrive. Defaults to the real clock.
TODAY=os.environ.get("PERF_GATE_TODAY") or datetime.date.today().isoformat()
bands={b["concurrency"]:b for b in (r.get("bands") or [])}
if 1 not in bands:
    print("FAIL ArmA band c=1 absent — scaling_efficiency is undefined without it"); sys.exit(1)
base=bands[1].get("aggregate_tok_per_sec")
if not base:
    print("FAIL ArmA agg(1) missing or zero"); sys.exit(1)
bl=((m.get("baselines") or {}).get(host) or {}).get(wl) or {}
fail=False
for c in sorted(bands):
    if c==1: continue
    agg=bands[c].get("aggregate_tok_per_sec")
    if agg is None: print(f"FAIL ArmA band {c}: aggregate_tok_per_sec absent"); fail=True; continue
    eff=(agg/base)/c
    if bl.get("status")=="UNMEASURED":
        # §4.7.3: an UNMEASURED cell degrades to REPORT only UNTIL its expiry.
        # `expires` was declared on every cell in perf-matrix.yaml and read by
        # ZERO lines of code, so the whole matrix would have sat in REPORT
        # forever and 2026-09-25 would have passed in silence. An expiry nothing
        # evaluates is a promise, not a deadline.
        #
        # A cell with no `expires` at all is a FAIL, not a pass: the absent
        # field is exactly how an UNMEASURED cell would otherwise become
        # permanent, and defaulting it to "never" rewards omitting it.
        exp = bl.get("expires")
        if not exp:
            print(f"FAIL ArmA c={c}: baseline UNMEASURED with no `expires` — an "
                  f"UNMEASURED cell without a deadline never expires")
            fail=True
        elif str(exp) < TODAY:
            print(f"FAIL ArmA c={c}: baseline UNMEASURED and EXPIRED {exp} "
                  f"(today {TODAY}, owner={bl.get('owner')}) — measure it or "
                  f"re-decide the cell; do not extend the date to stay green")
            fail=True
        else:
            print(f"REPORT ArmA c={c} scaling_efficiency={eff:.4f} "
                  f"(baseline UNMEASURED until {exp}, owner={bl.get('owner')})")
    else:
        floor=bl.get(f"c{c}")
        if floor is None: print(f"FAIL ArmA c={c}: no committed baseline and status is not UNMEASURED"); fail=True
        elif eff<floor: print(f"FAIL ArmA c={c} scaling_efficiency={eff:.4f} < baseline {floor}"); fail=True
        else: print(f"PASS ArmA c={c} scaling_efficiency={eff:.4f} >= {floor}")
sys.exit(1 if fail else 0)
PY
}

arm_b_adoption() {
  # B1 aggregate floor (policy 0.80) and B2 decode floor (inherited 1.00),
  # EVERY band, both required.
  local receipt="$1"
  python3 - "$receipt" "$MATRIX" <<'PY'
import json,sys,yaml
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2]))
b1=m["arms"]["B1"]["floor"]; b2=m["arms"]["B2"]["floor"]
bands=r.get("bands") or []
if not bands: print("FAIL ArmB no bands present"); sys.exit(1)
fail=False
for b in bands:
    c=b.get("concurrency")
    st=b.get("comparator_status")
    if st in ("NOT_APPLICABLE","UNMEASURED"):
        print(f"REPORT ArmB c={c} {st} (Arm A still gates this cell)"); continue
    ag,de=b.get("agg_ratio"),b.get("decode_ratio")
    if ag is None or de is None:
        print(f"FAIL ArmB c={c}: agg_ratio/decode_ratio absent and cell is not marked"); fail=True; continue
    if ag<b1: print(f"FAIL ArmB1 c={c} agg_ratio={ag:.3f} < {b1}"); fail=True
    else:     print(f"PASS ArmB1 c={c} agg_ratio={ag:.3f} >= {b1}")
    if de<b2: print(f"FAIL ArmB2 c={c} decode_ratio={de:.3f} < {b2}"); fail=True
    else:     print(f"PASS ArmB2 c={c} decode_ratio={de:.3f} >= {b2}")
    if de>=1.0 and ag<b1:
        print(f"NOTE  c={c} rising decode beside falling aggregate is the SERIALIZATION signature, not a win")
sys.exit(1 if fail else 0)
PY
}

arm_d_memory() {
  # REPORTING at v2.2, blocking with PERF-001. "Reporting" governs the
  # THRESHOLD, not the FIELD: a reporting arm whose metric may silently vanish
  # instruments nothing, so at release the fields must be PRESENT even though
  # no bound is applied to them yet.
  local receipt="$1" phase="$2"
  python3 - "$receipt" "$phase" <<'PY_D'
import json,sys
r=json.load(open(sys.argv[1])); phase=sys.argv[2]
kv=r.get("kv") or {}
used,resv=kv.get("bytes_used"),kv.get("bytes_reserved")
missing=[]
if used is None or resv is None: missing.append("kv.bytes_used/bytes_reserved")
if kv.get("admission_rejected") is None: missing.append("kv.admission_rejected")
if kv.get("preempted_swap") is None: missing.append("kv.preempted_swap")
if missing:
    lvl="FAIL" if phase=="release" else "REPORT"
    print(f"{lvl} ArmD instrumentation absent: {', '.join(missing)}")
    sys.exit(1 if phase=="release" else 0)
util=used/resv if resv else 0.0
print(f"REPORT ArmD kv_utilization={util:.4f} admission_rejected={kv['admission_rejected']} "
      f"preempted_swap={kv['preempted_swap']} (no bound applied until PERF-001)")
if util<0.5 and kv["admission_rejected"]>0:
    print("NOTE  ArmD refusing work while memory sits reserved-and-empty is the "
          "contiguous-allocation signature this arm exists to catch")
sys.exit(0)
PY_D
}

arm_e_interference() {
  # W2 ONLY (§4.3.2). Arm E is what chunked prefill exists to move; without it a
  # batching implementation that blocks the GPU on an 8192-token prefill scores
  # as a win on Arm A.
  local receipt="$1" phase="$2" workload="$3"
  if [ "$workload" != "W2" ]; then
    echo "SKIP  ArmE measured on W2 only (workload=$workload)"
    return 0
  fi
  python3 - "$receipt" "$phase" <<'PY_E'
import json,sys
r=json.load(open(sys.argv[1])); phase=sys.argv[2]
itl=r.get("itl") or {}
inj=r.get("injector") or {}
missing=[]
if itl.get("p95_w2_ms") is None or itl.get("p95_w1_ms") is None:
    missing.append("itl.p95_w2_ms/p95_w1_ms")
if inj.get("stall_p95_ms") is None: missing.append("injector.stall_p95_ms")
if inj.get("arrival_index") is None: missing.append("injector.arrival_index")
if missing:
    lvl="FAIL" if phase=="release" else "REPORT"
    print(f"{lvl} ArmE instrumentation absent: {', '.join(missing)}")
    sys.exit(1 if phase=="release" else 0)
w1=itl["p95_w1_ms"]
if not w1:
    print("FAIL ArmE p95_itl(W1) is zero or missing — the ratio is undefined")
    sys.exit(1)
ratio=itl["p95_w2_ms"]/w1
print(f"REPORT ArmE itl_p95_ratio={ratio:.4f} injector_stall_p95_ms={inj['stall_p95_ms']} "
      f"arrival_index={inj['arrival_index']} (no bound applied until PERF-001)")
sys.exit(0)
PY_E
}

cell_completeness() {
  # release only: every host x band in the matrix must be present.
  local receipt="$1" host="$2"
  python3 - "$receipt" "$MATRIX" "$host" <<'PY'
import json,sys,yaml
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2])); host=sys.argv[3]
want=set(m["bands"]); have={b.get("concurrency") for b in (r.get("bands") or [])}
missing=sorted(want-have)
if missing:
    print(f"FAIL cells host={host} missing bands {missing} — a missing cell is not a passing cell")
    sys.exit(1)
print(f"PASS cells host={host} all bands {sorted(want)} present")
PY
}

run_gate() {
  local host="$1" phase="$2" workload="$3" receipt="$4" rc=0
  [ -f "$receipt" ] || die "receipt not found: $receipt"
  arm_c_integrity "$receipt" || rc=1
  arm_a_scaling  "$receipt" "$host" "$workload" || rc=1
  arm_b_adoption "$receipt" || rc=1
  arm_d_memory "$receipt" "$phase" || rc=1
  arm_e_interference "$receipt" "$phase" "$workload" || rc=1
  if [ "$phase" = release ]; then
    cell_completeness "$receipt" "$host" || rc=1
  fi
  if [ "$rc" = 0 ]; then echo "VERDICT PASS host=$host phase=$phase workload=$workload"
  else echo "VERDICT FAIL host=$host phase=$phase workload=$workload"; fi
  return "$rc"
}

# ------------------------------------------------------------ selftest -----
# A guard is admissible only if a mutation of the thing it guards turns it RED.
# Each row states what it mutates and which arm must reject it.
selftest() {
  local tmp pass=0 fail=0
  tmp="$(mktemp -d)"
  # Everything below ends in `rm -rf "$tmp"`; bashrs SEC011 is right that an
  # unvalidated one is a loaded gun. Same guard shape as
  # scripts/check_apr_bin_resolution.sh, which is the repo's accepted form.
  case "$tmp" in
    /tmp/*|/var/folders/*) : ;;
    *) die "mktemp -d gave ${tmp:-<empty>}, refusing to rm -rf it" ;;
  esac
  _mk() { printf '%s' "$2" > "$tmp/$1.json"; }
  local OK='{"requested":16,"completed":16,"timeouts":0,"drain_ms":12,
   "provenance":{"binary_path":"/opt/pinned-binary","binary_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
                 "resolution":"apr_bin.sh","compute_class":"cuda","host":"lambda","accelerator":"rtx-4090",
                 "model":"qwen2.5-coder-1.5b-instruct","quantization":"Q4_K_M"},
   "tokenization":{"method":"hf-tokenizers"},
   "samples_ms":[10.0,11.0,10.5,10.2,10.8],
   "bands":[{"concurrency":1,"aggregate_tok_per_sec":100,"tokens_total":900,"agg_ratio":0.9,"decode_ratio":1.1},
            {"concurrency":4,"aggregate_tok_per_sec":360,"tokens_total":900,"agg_ratio":0.9,"decode_ratio":1.1}]}'
  _case() { # name, json, expect(pass|fail)
    _mk "$1" "$2"
    if run_gate lambda merge W1 "$tmp/$1.json" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$3" ]; then
      printf '  ok    %-34s expect=%s\n' "$1" "$3"
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s expected %s got %s\n' "$1" "$3" "$got"
      fail=$((fail + 1))
    fi
  }
  _casepw() { # name, json, phase, workload, expect
    _mk "$1" "$2"
    if run_gate lambda "$3" "$4" "$tmp/$1.json" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$5" ]; then
      printf '  ok    %-34s expect=%s\n' "$1" "$5"
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s expected %s got %s\n' "$1" "$5" "$got"
      fail=$((fail + 1))
    fi
  }
  # Arms D and E are REPORTING: no bound is applied, but the FIELDS must exist
  # at release, or the arm instruments nothing while reading green.
  local FULLBANDS ARMDE
  FULLBANDS='"bands":[{"concurrency":1,"aggregate_tok_per_sec":100,"tokens_total":9,"agg_ratio":0.9,"decode_ratio":1.1},
            {"concurrency":4,"aggregate_tok_per_sec":360,"tokens_total":9,"agg_ratio":0.9,"decode_ratio":1.1},
            {"concurrency":8,"aggregate_tok_per_sec":720,"tokens_total":9,"agg_ratio":0.9,"decode_ratio":1.1},
            {"concurrency":16,"aggregate_tok_per_sec":1440,"tokens_total":9,"agg_ratio":0.9,"decode_ratio":1.1}]}'
  ARMDE='"kv":{"bytes_used":50,"bytes_reserved":100,"admission_rejected":0,"preempted_swap":0},
   "itl":{"p95_w1_ms":10.0,"p95_w2_ms":14.0},"injector":{"stall_p95_ms":42.0,"arrival_index":7},'
  local REL_NO_DE REL_DE
  REL_NO_DE="${OK%\"bands\"*}$FULLBANDS"
  REL_DE="${OK%\"bands\"*}$ARMDE$FULLBANDS"
  _casepw armDE_absent_is_reporting_at_merge   "$REL_NO_DE" merge   W2 pass
  _casepw armDE_absent_is_fatal_at_release     "$REL_NO_DE" release W2 fail
  _casepw armDE_present_passes_release         "$REL_DE"    release W2 pass
  local REL_D_ONLY
  REL_D_ONLY="${OK%\"bands\"*}\"kv\":{\"bytes_used\":50,\"bytes_reserved\":100,\"admission_rejected\":0,\"preempted_swap\":0},$FULLBANDS"
  _casepw armE_skipped_on_w1_with_d_present    "$REL_D_ONLY" release W1 pass
  _casepw armE_absent_is_fatal_on_w2           "$REL_D_ONLY" release W2 fail
  _case baseline_healthy               "$OK" pass
  _case completed_lt_requested         "${OK/\"completed\":16/\"completed\":15}" fail
  _case a_timeout_is_fatal             "${OK/\"timeouts\":0/\"timeouts\":1}" fail
  _case tokenization_absent            "${OK/\"method\":\"hf-tokenizers\"/\"method\":\"\"}" fail
  _case drain_ms_absent                "${OK/\"drain_ms\":12/\"drain_ms\":null}" fail
  _case zero_token_response            "${OK/\"tokens_total\":900,\"agg_ratio\":0.9,\"decode_ratio\":1.1}]}/\"tokens_total\":0,\"agg_ratio\":0.9,\"decode_ratio\":1.1}]}}" fail
  _case b1_aggregate_below_floor       "${OK//\"agg_ratio\":0.9/\"agg_ratio\":0.79}" fail
  _case b1_aggregate_at_floor          "${OK//\"agg_ratio\":0.9/\"agg_ratio\":0.80}" pass
  _case b2_decode_below_floor          "${OK//\"decode_ratio\":1.1/\"decode_ratio\":0.99}" fail
  _case b2_decode_at_floor             "${OK//\"decode_ratio\":1.1/\"decode_ratio\":1.00}" pass
  # THE 2026-08-25 SHAPE: decode soaring while aggregate collapses.
  _case serialization_shape_rejected   "$(printf '%s' "$OK" | sed 's/"agg_ratio":0.9/"agg_ratio":0.097/g; s/"decode_ratio":1.1/"decode_ratio":1.554/g')" fail
  _case band_c1_absent                 "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["bands"]=[b for b in r["bands"] if b["concurrency"]!=1];print(json.dumps(r))')" fail
  printf '  %d passed, %d broken\n' "$pass" "$fail"
  rm -rf "${tmp:?refusing to rm an empty path}"
  [ "$fail" = 0 ]
}

main() {
  local host="" phase="" workload="" receipt=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --selftest) selftest; return $? ;;
      --host) host="$2"; shift 2 ;;
      --phase) phase="$2"; shift 2 ;;
      --workload) workload="$2"; shift 2 ;;
      --receipt) receipt="$2"; shift 2 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  [ -n "$host" ] && [ -n "$phase" ] && [ -n "$workload" ] && [ -n "$receipt" ] \
    || die "usage: perf_gate.sh --host H --phase {merge|release} --workload {W1|W2} --receipt PATH"
  case "$phase" in merge|release) ;; *) die "phase must be merge or release" ;; esac
  run_gate "$host" "$phase" "$workload" "$receipt"
}
main "$@"
