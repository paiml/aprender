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
# THE SCHEMA MAP (PERF-004). Every field below is classified there as PRODUCED,
# DERIVED, POLICY or UNMEASURED, and the UNMEASURED entries carry an owning
# ticket. Arm C reads it so an absent field is reported as the instrumentation
# gap it is -- "drain_ms absent: the drain rule is not implemented in any
# client (§4.4.7, owner PERF-004)" -- rather than as a schema error, which
# names nothing and leaves the reader to guess whether the fix is a converter
# or a measurement.
FIELDS="$ROOT/scripts/perf-receipt-fields.yaml"

die() { printf 'perf_gate: %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------- arms -----
# Every arm returns 0 (pass) or 1 (fail) and prints one PASS/FAIL line. The
# verdict is the MIN over arms (§4.8: exactly one verdict function, H11).

arm_c_integrity() {
  local receipt="$1" phase="$2" rc=0
  # The delegate's errors are PRINTED, not swallowed. They used to go to
  # /dev/null behind "bench_receipt.py rejected the receipt", so the one line a
  # reader got named neither the field nor the rule -- and every real artifact
  # in the tree fails here, so that line was the whole diagnosis.
  local schema_out
  if ! schema_out="$(python3 "$ROOT/scripts/lib/bench_receipt.py" "$receipt" 2>&1)"; then
    printf 'FAIL ArmC schema: %s\n' "$schema_out"
    rc=1
  fi
  python3 - "$receipt" "$FIELDS" "$phase" <<'PY' || rc=1
import json,re,sys,yaml
r=json.load(open(sys.argv[1]))
fielddoc=yaml.safe_load(open(sys.argv[2])) or {}
ledger=fielddoc.get("fields") or {}
phase=sys.argv[3]

def why(field):
    """The instrumentation gap behind an absent field, with its owner.

    A gate that says 'absent' has told the reader the shape of the receipt. A
    gate that says WHAT WOULD HAVE TO BE BUILT and WHO OWNS IT has told them
    the shape of the work. The five fields that made every real artifact fail
    here split two ways -- two derivable, three unmeasured -- and only this
    lookup makes the difference visible at the point of failure.
    """
    e=ledger.get(field) or {}
    if e.get("class")!="UNMEASURED":
        return ""
    needs=" ".join((e.get("needs") or "").split())
    return f" -- {needs} ({e.get('spec')}, owner {e.get('owner')})"

bad=[]
req,comp=r.get("requested"),r.get("completed")
if req is None or comp is None or req!=comp:
    bad.append(f"completed({comp}) != requested({req}){why('completed')}")
if r.get("timeouts") is None:
    bad.append(f"timeouts absent{why('timeouts')}")
elif r.get("timeouts")!=0:
    bad.append(f"timeouts={r.get('timeouts')} (fatal to this host's ratio)")

# ---------------------------------------------------------------- 4.4.6 -----
# PERF-048 / #2754. Every streaming receipt this repo could produce declared
# `server_usage` while the tokens were client-counted: apr serve's SSE stream
# carries no usage object, the harness silently took its fallback arm, and
# NOTHING cross-checked the assertion against the observation. A field no
# observation can contradict is decoration.
#
# So the receipt now carries the observation the method was derived FROM, and
# this re-derives the method here. A producer edited to write the REQUESTED
# method into `method` does not thereby make the two agree at the gate: the
# counts still say what happened.
tok=r.get("tokenization") or {}
method=tok.get("method")
chunk_counted=False
if not method:
    bad.append(f"tokenization.method absent{why('tokenization.method')}")
else:
    requested=tok.get("method_requested")
    seen,counted=tok.get("responses_with_server_usage"),tok.get("responses_counted")
    by_client=tok.get("responses_counted_by_client_tokenizer")
    if requested is None:
        bad.append("tokenization.method_requested absent -- the receipt names one counting "
                   "method and cannot say whether it is the one that was asked for (#2754)")
    if seen is None or counted is None or by_client is None:
        bad.append("tokenization.responses_with_server_usage/"
                   "responses_counted_by_client_tokenizer/responses_counted absent -- without "
                   "the observation, tokenization.method is an unfalsifiable assertion (#2754). "
                   "AN ABSENT COUNTER IS NOT A ZERO COUNTER.")
    elif counted == 0:
        bad.append("tokenization.responses_counted=0 -- no response was counted, so no counting "
                   "method was used and naming one asserts what the run did not do")
    elif seen > counted or by_client > counted:
        bad.append(f"tokenization: {seen} server-counted and {by_client} client-tokenizer-counted "
                   f"responses against {counted} counted -- a counter cannot exceed its own "
                   f"denominator")
    else:
        # THE RE-DERIVATION. `method` is a pure function of these three counters
        # in the producer; computing it again here is what makes a producer
        # edited to write the REQUESTED method into `method` go RED. A method is
        # admissible only when EVERY counted response was counted that way: a
        # mixture is the fallback class, not either pure class.
        if seen == counted:
            expected="server_usage"
        elif by_client == counted:
            expected="client_tokenizer"
        else:
            expected="client_chunk_count"
        if method != expected:
            bad.append(f"tokenization.method={method!r} contradicts its own observation: of "
                       f"{counted} completed responses, {seen} carried a server usage object and "
                       f"{by_client} were counted by a client tokenizer, so the method USED was "
                       f"{expected!r} (#2754)")
    if method=="client_tokenizer" and not re.fullmatch(r"[0-9a-f]{64}", str(tok.get("tokenizer_sha256") or "")):
        bad.append("tokenization.method=client_tokenizer with no 64-hex tokenizer_sha256 -- "
                   "4.4.6 requires the digest of the tokenizer that did the counting")
    if requested is not None and requested != method:
        # The downgrade is admissible; a SILENT downgrade is not. The reason is
        # the field that stops this from becoming the next fallback arm nobody
        # can see.
        if not (tok.get("downgrade_reason") or "").strip():
            bad.append(f"tokenization.method={method!r} != method_requested={requested!r} with "
                       f"no downgrade_reason -- a silent downgrade is the defect (#2754)")
        else:
            print(f"REPORT ArmC tokenization DOWNGRADED requested={requested} used={method}: "
                  f"{tok['downgrade_reason']}")
    chunk_counted = method=="client_chunk_count"

# ------------------------------------------------------------- 4.4.2 N ------
# PERF-048 / #2755. `--replicates 1` produced a receipt byte-indistinguishable
# from one replicate of a spec N=3 cell, and it self-reported conformant. The
# warning went to stdout, which is not retained.
rep=r.get("replicates")
if not isinstance(rep,dict) or rep.get("effective") is None or rep.get("required") is None:
    bad.append("replicates absent -- an N=1 run is otherwise byte-indistinguishable from one "
               "replicate of a spec N=3 cell and reports itself conformant (#2755)")
elif rep["effective"] < rep["required"]:
    lvl="FAIL" if phase=="release" else "REPORT"
    print(f"{lvl} ArmC replicates={rep['effective']} < 4.4.2 N={rep['required']} -- the cell is "
          f"under-replicated and its bootstrap CI is correspondingly weak")
    if phase=="release": bad.append("under-replicated cell at release phase")

# --------------------------------------------------------------- 4.3 --------
# PERF-048 / #2756. `--workload` was free text: a one-prompt-repeated-30-times
# run was recorded as W1, the very degenerate case the committed W1 corpus's
# `_meta.distinctness_rationale` documents as invalidating Arm A.
wc=r.get("workload_corpus")
if not isinstance(wc,dict) or not wc.get("sha256"):
    bad.append("workload_corpus absent -- `workload` is then a free-text label with no "
               "connection to the prompts actually sent (#2756)")
else:
    cs=[b.get("concurrency") for b in (r.get("bands") or []) if isinstance(b.get("concurrency"),int)]
    distinct=wc.get("distinct_prompts")
    if cs and distinct is not None:
        floor=max(30, 8*min(cs))
        if distinct < floor:
            bad.append(f"workload {r.get('workload')}: the sent set holds {distinct} distinct "
                       f"prompt(s) against a {floor}-sample narrowest band, so the same prompt "
                       f"is served repeatedly inside one band and prefix caching -- not the "
                       f"scheduler -- drives Arm A's scaling_efficiency (#2756)")

# Departures the run STATED. Reported at merge, fatal at release: conformant,
# silent and indistinguishable is the worst of the three.
sv=r.get("stated_violations")
if not isinstance(sv,list):
    bad.append("stated_violations absent -- a run that departed from 4.4 then reads exactly "
               "like one that did not (#2755)")
else:
    for v in sv:
        print(f"{'FAIL' if phase=='release' else 'REPORT'} ArmC stated violation: {v}")
    if sv and phase=="release":
        bad.append(f"{len(sv)} stated 4.4 violation(s) at release phase")

if r.get("drain_ms") is None:
    bad.append(f"drain_ms absent{why('drain_ms')}")
for b in (r.get("bands") or []):
    if b.get("tokens_total",1)==0:
        bad.append(f"band {b.get('concurrency')}: zero-token response is a failure, not a fast request")
    # THE OTHER SIDE OF THE RATIO. Arm C asserts completed == requested on the
    # SUBJECT and never looked at the comparator, whose tok/s counts tokens
    # from successful requests only over the same wall clock -- so every
    # request the comparator loses lowers the denominator and flatters us.
    # Reported, not failed: the comparator's loss rate has no committed
    # threshold, and inventing one here is the defect this gate exists to stop.
    # A ratio over chunk counts compares two servers' CHUNKING POLICIES, not
    # their throughput. 4.4.6 exists so a ratio is only computed when both
    # sides count the same way; this is the consequence that makes the
    # downgrade above matter rather than merely appear.
    if chunk_counted and (b.get("agg_ratio") is not None or b.get("decode_ratio") is not None):
        bad.append(f"band {b.get('concurrency')}: a comparator ratio was computed over "
                   f"client_chunk_count tokens -- that is a count of SSE deltas, equal to a "
                   f"token count only for a server emitting one token per chunk (4.4.6, #2754)")
    cr,cc=b.get("comparator_requested"),b.get("comparator_completed")
    if cr is not None and cc is not None and cc!=cr:
        print(f"REPORT ArmC c={b.get('concurrency')} comparator completed {cc}/{cr} "
              f"-- its lost requests lower the denominator of agg_ratio")
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
  arm_c_integrity "$receipt" "$phase" || rc=1
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
  # A FIXTURE THAT IS NOT VALID JSON FAILS THE GATE FOR THE WRONG REASON.
  #
  # `${OK/"replicates":{...},/}` looks like a deletion and is not: bash ends the
  # parameter expansion at the first `}` inside the pattern, so that row wrote
  # `{"a":1,},"b":2},/}` to disk. Garbage fails, so the row read `ok` while
  # proving nothing about the rule it names -- and it stayed `ok` when that rule
  # was deleted, which is the only reason it was ever found.
  #
  # So every fixture is parsed before it is judged. A row that cannot build its
  # own input is BROKE, never a pass.
  _mk() {
    printf '%s' "$2" > "$tmp/$1.json"
    if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$tmp/$1.json" 2>/dev/null; then
      printf '  BROKE %-34s fixture is not valid JSON -- the row cannot be testing its rule\n' "$1"
      fail=$((fail + 1))
      return 1
    fi
  }
  local OK='{"requested":16,"completed":16,"timeouts":0,"drain_ms":12,
   "provenance":{"binary_path":"/opt/pinned-binary","binary_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
                 "resolution":"apr_bin.sh","compute_class":"cuda","host":"lambda","accelerator":"rtx-4090",
                 "model":"qwen2.5-coder-1.5b-instruct","quantization":"Q4_K_M"},
   "tokenization":{"method":"server_usage","method_requested":"server_usage","downgraded":false,
                   "counts_special_tokens":true,"counts_prompt_echo":false,
                   "responses_with_server_usage":30,"responses_counted_by_client_tokenizer":0,
                   "responses_counted":30},
   "replicates":{"index":1,"effective":3,"required":3,"below_spec":false},
   "workload_corpus":{"prompts":256,"distinct_prompts":256,
                      "sha256":"1111111111111111111111111111111111111111111111111111111111111111",
                      "source":"file prompts-w1.jsonl"},
   "stated_violations":[],
   "samples_ms":[10.0,11.0,10.5,10.2,10.8],
   "bands":[{"concurrency":1,"aggregate_tok_per_sec":100,"tokens_total":900,"agg_ratio":0.9,"decode_ratio":1.1},
            {"concurrency":4,"aggregate_tok_per_sec":360,"tokens_total":900,"agg_ratio":0.9,"decode_ratio":1.1}]}'
  _case() { # name, json, expect(pass|fail)
    _mk "$1" "$2" || return 0
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
    _mk "$1" "$2" || return 0
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
  # A below-spec N and any stated 4.4 departure REPORT at merge and are FATAL at
  # release. "Conformant, silent and indistinguishable" is the worst of three
  # (#2755); reporting-only at release would be the second worst.
  _casepw replicates_below_spec_is_fatal_at_release \
          "${REL_DE/\"effective\":3/\"effective\":1}" release W1 fail
  _casepw a_stated_violation_is_fatal_at_release \
          "$(printf '%s' "$REL_DE" | python3 -c 'import sys,json;r=json.load(sys.stdin);r.update(stated_violations=["4.3 W2: 99 distinct prompts against 128 sampled requests"]);print(json.dumps(r))')" release W1 fail
  _case baseline_healthy               "$OK" pass
  _case completed_lt_requested         "${OK/\"completed\":16/\"completed\":15}" fail
  _case a_timeout_is_fatal             "${OK/\"timeouts\":0/\"timeouts\":1}" fail
  # AN ABSENT COUNTER IS NOT A ZERO COUNTER. This read `r.get("timeouts",0)`,
  # so a receipt that never counted timeouts was indistinguishable from one
  # that counted none -- and no producer in this tree counts them at all
  # (loadtest.rs collapses every failure to one `failed` tally), so the default
  # was doing all the work on every real artifact. That is the same shape as a
  # missing measurement reading as a passing one, which is what this gate is
  # for.
  _case timeouts_absent_is_not_zero    "${OK/\"timeouts\":0,/}" fail
  _case tokenization_absent            "${OK/\"method\":\"server_usage\"/\"method\":\"\"}" fail
  # ---- 4.4.6, PERF-048 / #2754 --------------------------------------------
  # THE PERF-045 RECEIPT, EXACTLY: method says server_usage, the observation
  # says no response carried a usage object. This is the row that was missing.
  _case tok_method_contradicts_its_own_observation \
        "${OK/\"responses_with_server_usage\":30/\"responses_with_server_usage\":0}" fail
  # A mixture is the fallback class, not the server class.
  _case tok_partial_server_usage_is_not_server_usage \
        "${OK/\"responses_with_server_usage\":30/\"responses_with_server_usage\":29}" fail
  # An absent observation is not a satisfied one.
  _case tok_observation_absent         "${OK/\"responses_with_server_usage\":30,/}" fail
  _case tok_counted_zero               "${OK/\"responses_counted\":30/\"responses_counted\":0}" fail
  # The requested method must be recorded beside the used one.
  _case tok_method_requested_absent    "${OK/\"method_requested\":\"server_usage\",/}" fail
  # A downgrade is admissible. A SILENT downgrade is not.
  # ONE RULE PER ROW. The downgraded base has its comparator ratios removed --
  # a chunk-counted receipt that still carried them would fail the ratio rule
  # below, and the row would then be green for a reason it does not name. A
  # case that fails for the wrong reason cannot detect the removal of the rule
  # it is nominally about; this exact row did, and only the mutation run found
  # it.
  local TOK_DOWN TOK_SILENT TOK_LOUD TOK_LOUD_RATIO TOK_CLIENT
  TOK_DOWN="$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["tokenization"].update(method="client_chunk_count",downgraded=True,responses_with_server_usage=0);[(b.pop("agg_ratio",None),b.pop("decode_ratio",None),b.update(comparator_status="UNMEASURED")) for b in r["bands"]];print(json.dumps(r))')"
  TOK_SILENT="$TOK_DOWN"
  _case tok_downgrade_without_a_reason "$TOK_SILENT" fail
  TOK_LOUD="$(printf '%s' "$TOK_DOWN" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["tokenization"].update(downgrade_reason="apr serve emits no usage object; 0 of 30 responses carried one");print(json.dumps(r))')"
  _case tok_downgrade_with_a_reason_is_reported "$TOK_LOUD" pass
  # ...and the downgrade has a CONSEQUENCE: no ratio over chunk counts. Same
  # receipt as the row above, with the ratios put back.
  TOK_LOUD_RATIO="$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["tokenization"].update(method="client_chunk_count",downgraded=True,responses_with_server_usage=0,downgrade_reason="apr serve emits no usage object");print(json.dumps(r))')"
  _case tok_chunk_counted_ratio_is_refused "$TOK_LOUD_RATIO" fail
  # 4.4.6 requires the digest of the tokenizer that counted.
  TOK_CLIENT="$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["tokenization"].update(method="client_tokenizer",method_requested="client_tokenizer",responses_with_server_usage=0,responses_counted_by_client_tokenizer=30);print(json.dumps(r))')"
  _case tok_client_tokenizer_without_a_digest "$TOK_CLIENT" fail
  _case tok_client_tokenizer_partial_is_chunk_counted \
        "$(printf '%s' "$TOK_CLIENT" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["tokenization"].update(responses_counted_by_client_tokenizer=29,tokenizer_sha256="'"$(printf 'b%.0s' $(seq 64))"'");print(json.dumps(r))')" fail
  _case tok_counter_exceeds_its_denominator \
        "${OK/\"responses_with_server_usage\":30/\"responses_with_server_usage\":31}" fail
  _case tok_client_tokenizer_with_a_digest \
        "$(printf '%s' "$TOK_CLIENT" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["tokenization"].update(tokenizer_sha256="'"$(printf 'b%.0s' $(seq 64))"'");print(json.dumps(r))')" pass
  # ---- 4.4.2 N, PERF-048 / #2755 ------------------------------------------
  _case replicates_absent              "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r.pop("replicates");print(json.dumps(r))')" fail
  _case replicates_below_spec_reports_at_merge \
        "${OK/\"effective\":3/\"effective\":1}" pass
  # ---- 4.3, PERF-048 / #2756 ----------------------------------------------
  _case workload_corpus_absent         "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r.pop("workload_corpus");print(json.dumps(r))')" fail
  # A one-prompt corpus labelled W1 is the #2756 receipt exactly.
  _case single_prompt_corpus_is_not_w1 "${OK/\"distinct_prompts\":256/\"distinct_prompts\":1}" fail
  _case corpus_at_the_narrowest_band_floor "${OK/\"distinct_prompts\":256/\"distinct_prompts\":30}" pass
  _case corpus_one_below_the_floor     "${OK/\"distinct_prompts\":256/\"distinct_prompts\":29}" fail
  _case stated_violations_absent       "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r.pop("stated_violations");print(json.dumps(r))')" fail
  _case a_stated_violation_reports_at_merge \
        "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r.update(stated_violations=["4.4.2 replicates=1 < N=3"]);print(json.dumps(r))')" pass
  _case drain_ms_absent                "${OK/\"drain_ms\":12/\"drain_ms\":null}" fail
  # PRE-EXISTING, found the day _mk started parsing its own fixtures: this row
  # substituted a pattern ending in `}]}` inside `${OK/.../...}`, which bash
  # terminated at the first `}`. The row had read `ok` since it was written,
  # over a fixture that was not JSON at all.
  _case zero_token_response            "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["bands"][-1].update(tokens_total=0);print(json.dumps(r))')" fail
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
