#!/usr/bin/env bash
# perf_gate.sh — PP-LLAMA-001-MASTER.md §7, the performance gate.
#
#   scripts/perf_gate.sh --host <name> --phase {merge|release} \
#                        --workload {W1|W2} --receipt <path> \
#                        [--commit <commit-under-test>]   # REQUIRED at release
#   scripts/perf_gate.sh --selftest
#   scripts/perf_gate.sh --list-selftests
#
# WHY THIS EXISTS. A gate reading only per-user decode scores a comfortable PASS
# on the serialization signature -- decode RISING while aggregate FALLS, because
# each request owns the whole device in turn. The measurement that named that
# trap is recorded in evidence/parity/LEDGER.md (the 2026-08-25 row for commit
# 53062e7f3, whose validity_by_band marks c>1 INVALID-BUILD). The digits are
# deliberately not quoted here: PP-12 forbids publishing a c>1 aggregate from
# that run, and this header used to quote two of them. Arm L3 therefore gates a
# band-class-specific SET of ratios and reports the rest; neither substitutes
# for the other.
#
# WHAT THIS DOES NOT DO. It does not re-implement receipt validation: that is
# scripts/lib/bench_receipt.py, which is the single schema authority. This
# script computes the ARMS and renders the verdict.
#
# EVERY NUMBER IT COMPARES AGAINST LIVES IN scripts/perf-matrix.yaml (PP-33),
# and every arm runs at the PHASE that file declares for it (PP-6). There are no
# threshold literals below; scripts/check_thresholds_in_matrix.sh refuses one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Injectable for the same reason PERF_GATE_TODAY is: the expiry rules below
# have FAIL branches that no well-formed committed matrix can reach, and a
# branch no case can enter is a branch nothing proves.
MATRIX="${PERF_GATE_MATRIX:-$ROOT/scripts/perf-matrix.yaml}"
# THE SCHEMA MAP (PERF-004). Every field below is classified there as PRODUCED,
# DERIVED, POLICY or UNMEASURED, and the UNMEASURED entries carry an owning
# ticket. Arm C reads it so an absent field is reported as the instrumentation
# gap it is -- "drain_ms absent: the drain rule is not implemented in any
# client (§4.4.7, owner PERF-004)" -- rather than as a schema error, which
# names nothing and leaves the reader to guess whether the fix is a converter
# or a measurement.
FIELDS="$ROOT/scripts/perf-receipt-fields.yaml"

die() { printf 'perf_gate: %s\n' "$*" >&2; exit 2; }

# ------------------------------------------------------------- the phases ---
# PP-6. Each arm declares `phase:` in perf-matrix.yaml. The vocabulary there is
# {merge, release, both, reporting}; the CLI's is {merge, release}. The mapping:
#
#   both       run, and its verdict counts, at either CLI phase
#   reporting  run, and its verdict counts, at either CLI phase -- "reporting"
#              governs the THRESHOLD (no bound is applied to the metric), not
#              the FIELD (its presence rule is still enforced), which is why
#              Arms D and E receive $phase and decide for themselves
#   merge      run and count at --phase merge only
#   release    run and count at --phase release only
#
# OUTSIDE ITS PHASE AN ARM STILL RUNS, and its FAIL is demoted to REPORT. It is
# not skipped. A skipped arm prints nothing, so the reader of a merge verdict
# learns nothing about the release arms; a demoted one prints the same finding
# with the same words and does not decide the merge. The demotion line is
# printed ONLY when something was actually demoted, so an arm that had nothing
# to say at merge leaves the merge verdict byte-identical.
arm_phase() { # arm-key -> the phase perf-matrix.yaml declares for it
  python3 - "$MATRIX" "$1" <<'PY_PHASE'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1])) or {}
mx_arm = ((m.get("arms") or {}).get(sys.argv[2])) or {}
print(mx_arm.get("phase") or "both")
PY_PHASE
}

phase_counts() { # declared, cli-phase -> 0 when the arm's verdict counts
  case "$1" in
    both|reporting) return 0 ;;
    merge|release) [ "$1" = "$2" ] ;;
    *) die "perf-matrix.yaml declares phase=$1, which is not one of merge, release, both, reporting" ;;
  esac
}

run_phased() { # arm-key, cli-phase, command...
  local arm="$1" phase="$2"
  shift 2
  local declared out rc=0
  declared="$(arm_phase "$arm")"
  if phase_counts "$declared" "$phase"; then
    "$@"
    return $?
  fi
  # No pipe reads a status here: `out=$(...)` is captured, then examined.
  out="$("$@" 2>&1)" || rc=1
  if [ "$rc" = 0 ]; then
    if [ -n "$out" ]; then printf '%s\n' "$out"; fi
    return 0
  fi
  if [ -n "$out" ]; then printf '%s\n' "$out" | sed 's/^FAIL /REPORT /'; fi
  printf 'REPORT Arm%s declares phase=%s in scripts/perf-matrix.yaml; at --phase %s its FAIL is demoted to REPORT (PP-6)\n' \
    "$arm" "$declared" "$phase"
  return 0
}

cell_status() { # host, workload -> the matrix baseline status (MEASURED | UNMEASURED | NA | ABSENT)
  python3 - "$MATRIX" "$1" "$2" <<'PY_CELL'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1])) or {}
bl = ((m.get("baselines") or {}).get(sys.argv[2]) or {}).get(sys.argv[3]) or {}
print(bl.get("status") or "ABSENT")
PY_CELL
}

receipt_is_historical() { # receipt -> yes when schema_version < 3 (the pre-v3 wire), else no
  python3 - "$1" <<'PY_HIST'
import json, sys
r = json.load(open(sys.argv[1]))
sv = r.get("schema_version")
print("no" if (isinstance(sv, int) and not isinstance(sv, bool) and sv >= 3) else "yes")
PY_HIST
}

# THE HISTORICAL-RECEIPT RULE FOR THE RELEASE-ONLY ARMS (§7.2, PP-1). A release
# is legal while a cell is UNMEASURED{owner, expires} and unexpired; the only
# receipt on disk for such a cell is the pre-v3 one the matrix has already
# marked SPENT. ArmL1 reads that receipt as historical and says so. ArmC-sig
# and ArmD did not: they failed it for being unsigned and for lacking the kv
# block -- two rules that apply FROM the first conformant receipt (PP-21,
# PP-2) -- so `--phase release` could not PASS in the exact state §7.2 permits
# a release in, and the 0.65.0 cut was the first to run it (2026-09-03). The
# demotion is narrow on purpose: BOTH conditions, or the FAIL stands. A v3
# receipt is never demoted (unsigned v3 = FAIL), and a historical receipt
# cited for a MEASURED cell is a defect in the citation, not a release state.
historical_for_unmeasured() { # cell-status, historical(yes|no) -> 0 when the demotion applies
  [ "$2" = yes ] && [ "$1" = UNMEASURED ]
}

# ---------------------------------------------------------------- arms -----
# Every arm returns 0 (pass) or 1 (fail) and prints one PASS/FAIL line. The
# verdict is the MIN over arms (exactly one verdict function, H11).

arm_c_integrity() {
  local receipt="$1" rc=0
  # The delegate's errors are PRINTED, not swallowed. They used to go to
  # /dev/null behind "bench_receipt.py rejected the receipt", so the one line a
  # reader got named neither the field nor the rule -- and every real artifact
  # in the tree fails here, so that line was the whole diagnosis.
  local schema_out
  if ! schema_out="$(python3 "$ROOT/scripts/lib/bench_receipt.py" "$receipt" 2>&1)"; then
    printf 'FAIL ArmC schema: %s\n' "$schema_out"
    rc=1
  fi
  python3 - "$receipt" "$FIELDS" <<'PY' || rc=1
import json,sys,yaml
r=json.load(open(sys.argv[1]))
fielddoc=yaml.safe_load(open(sys.argv[2])) or {}
ledger=fielddoc.get("fields") or {}

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
if not (r.get("tokenization") or {}).get("method"):
    bad.append(f"tokenization.method absent{why('tokenization.method')}")
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

arm_l1_schema() {
  # THE STATIC v3 RULES (L1). Every rule here is decidable from the receipt and
  # the matrix alone -- no timing assertion, nothing that needs a host -- which
  # is why perf-matrix.yaml declares `phase: both` for this arm and why it is
  # admissible in a required merge check (PP-6, check_no_timing_in_required.sh).
  #
  # SCOPE. `schema_version >= 3` is the wire generation, not a date. A receipt
  # without it is HISTORICAL: the v3 keys did not exist when it was produced, so
  # demanding them would fail every artifact in evidence/ for not having been
  # written in the future. Historical receipts are REPORTED as such, once.
  local receipt="$1" host="$2"
  python3 - "$receipt" "$MATRIX" "$host" <<'PY_L1'
import json,sys,yaml
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2])) or {}; host=sys.argv[3]
fail=[]; report=[]

sv = r.get("schema_version")
V3 = isinstance(sv, int) and not isinstance(sv, bool) and sv >= 3

mx_hosts = m.get("hosts") or {}
mx_host  = mx_hosts.get(host) or {}
mx_stream = m.get("stream") or {}
mx_witness = m.get("witness") or {}
mx_protocol = m.get("protocol") or {}

prov = r.get("provenance") or {}

# ---- PP-16 host class: the class must be the one the matrix declares, and a
# build in this tree must be able to reach it. `metal` passed every validator
# for months while no build could take that path.
klass = prov.get("compute_class")
if mx_host.get("status") == "NA":
    report.append("class host=%s is NA (%s; decided_by=%s) -- no cell on this host is gated"
                  % (host, " ".join(str(mx_host.get("reason") or "").split()), mx_host.get("decided_by")))
else:
    want = mx_host.get("compute_class")
    reach = mx_host.get("reachable_by") or []
    if want is not None and klass != want:
        fail.append("class receipt compute_class=%r but perf-matrix.yaml hosts.%s.compute_class=%r"
                    % (klass, host, want))
    elif klass is not None and reach and klass not in reach:
        fail.append("class %r is not in hosts.%s.reachable_by=%s -- a class no build reaches is a "
                    "declared claim, not a measurement" % (klass, host, reach))

if not V3:
    report.append("schema_version=%r: this receipt predates the v3 wire, so the L1 rules below are "
                  "not applied to it (historical)" % (sv,))
    for line in report: print("REPORT ArmL1 " + line)
    for line in fail:   print("FAIL ArmL1 " + line)
    sys.exit(1 if fail else 0)

# ---- PP-30 the clock ------------------------------------------------------
if not prov.get("started_utc"):
    fail.append("provenance.started_utc absent -- a receipt with no start time cannot be ordered "
                "against a comparator pin's expiry (PP-20) or against another run")
if not prov.get("clock_source"):
    fail.append("provenance.clock_source absent -- which clock produced started_utc is part of the "
                "measurement, not a detail")

# ---- PP-2 / PP-13 the server's own configuration --------------------------
cfg = prov.get("server_config")
if not isinstance(cfg, dict):
    fail.append("provenance.server_config is %s -- every server fact in this receipt is then "
                "HARNESS-DECLARED, which is the shape PP-13 refuses (GET /v1/effective-config)"
                % ("absent" if cfg is None else "not an object"))
else:
    sc_class = prov["server_config"].get("compute_class")
    if sc_class is not None and klass is not None and sc_class != klass:
        fail.append("provenance.compute_class=%r disagrees with server_config.compute_class=%r -- "
                    "the server's own answer wins; a harness flag is not evidence of a dispatch path"
                    % (klass, sc_class))
    # PP-14: an autofit that silently overrode an explicit operator argument
    # measured a configuration nobody asked for.
    off = prov["server_config"].get("offload")
    if isinstance(off, dict):
        applied = prov["server_config"]["offload"].get("autofit_applied")
        explicit = prov["server_config"]["offload"].get("explicit_args") or []
        if applied and explicit:
            fail.append("server_config.offload.autofit_applied is true while explicit_args=%s were "
                        "given -- autofit overrode an operator decision, so the receipt does not "
                        "describe the requested configuration" % (explicit,))

# ---- PP-24 admission --------------------------------------------------------
ladder = r.get("ladder")
if not isinstance(ladder, dict):
    fail.append("ladder absent -- the band set was REQUESTED, and without slots_admitted nothing "
                "records what either server actually granted")
    slots_apr = slots_llama = None
else:
    slots = r["ladder"].get("slots_admitted")
    slots_apr = r["ladder"]["slots_admitted"].get("apr") if isinstance(slots, dict) else None
    slots_llama = r["ladder"]["slots_admitted"].get("llama") if isinstance(slots, dict) else None
    if isinstance(slots, dict) and (slots_apr is not None or slots_llama is not None) and not isinstance(cfg, dict):
        fail.append("ladder.slots_admitted is present while provenance.server_config is absent -- an "
                    "admission figure with no server configuration behind it is INFERRED, and PP-13 "
                    "refuses an inferred field wearing a reported one's name")
    # PP-24: `derived` is not a free field. It follows from `declared` and the
    # server-reported admissions, and the cells arm reads it to EXCUSE a missing
    # band -- so a hand-written `derived` would let a receipt drop the bands the
    # servers admitted. Recompute it and refuse a disagreement.
    declared = r["ladder"].get("declared")
    derived = r["ladder"].get("derived")
    known = [s for s in (slots_apr, slots_llama) if isinstance(s, int)]
    if isinstance(declared, list) and known:
        want_derived = [c for c in declared if isinstance(c, int) and c <= min(known)]
        if derived != want_derived:
            fail.append("ladder.derived=%s does not follow from declared=%s and slots_admitted=%s "
                        "(expected %s) -- PP-24 derives the ladder from what the servers granted; a "
                        "receipt may not excuse a band by writing its own" % (derived, declared, slots, want_derived))

bands = r.get("bands") or []

# ---- PP-28 the sampler ------------------------------------------------------
proto = r.get("protocol")
if not isinstance(proto, dict):
    fail.append("protocol absent -- window, replicates and the sampler are join-key components "
                "(PP-22), not defaults")
else:
    samp = r["protocol"].get("sampler")
    if not isinstance(samp, dict):
        fail.append("protocol.sampler absent -- an unpinned sampler is an unpinned measurement")
    else:
        pinned = mx_protocol.get("sampler") or {}
        for key, got in (("temperature", r["protocol"]["sampler"].get("temperature")),
                         ("seed", r["protocol"]["sampler"].get("seed")),
                         ("ignore_eos", r["protocol"]["sampler"].get("ignore_eos"))):
            if key in pinned and got != pinned[key]:
                fail.append("protocol.sampler.%s=%r but perf-matrix.yaml protocol.sampler.%s=%r -- "
                            "the sampler is pinned by the matrix, not by the run" % (key, got, key, pinned[key]))
short = r.get("short_of_n_predict")
if short is None:
    fail.append("short_of_n_predict absent -- `truncated` counts drain-abandoned requests only, so "
                "nothing else in this receipt witnesses a completion that stopped early (PP-28)")
elif short:
    fail.append("short_of_n_predict=%d -- requests completed below n_predict, so the sampler did not "
                "hold and the throughput denominators are not comparable" % (short,))

# ---- per band ---------------------------------------------------------------
THROUGHPUT = ("aggregate_tok_per_sec", "decode_tok_per_sec", "prefill_tok_per_sec")
ttft_max = mx_stream.get("live_ttft_over_e2e_max")
min_agree = mx_witness.get("min_agree_tokens")
max_run = mx_witness.get("max_constant_run")
client_sha = prov["client"].get("sha256") if isinstance(prov.get("client"), dict) else None
if not isinstance(prov.get("client"), dict):
    fail.append("provenance.client absent -- PP-25 requires the SAME client binary on both lanes and "
                "nothing here records which one drove this run")
if not isinstance(prov.get("subject"), dict):
    fail.append("provenance.subject absent -- binary_* describe the CLIENT; the served binary is "
                "unidentified")

for b in bands:
    c = b.get("concurrency")
    tag = "c=%s" % (c,)
    status = b.get("status")
    invalid = status == "INVALID-CORRECTNESS"

    # ---- PP-26 batch invariance --------------------------------------------
    wit = b.get("witness")
    if c is not None and c > 1:
        ok = isinstance(wit, dict) and b["witness"].get("batch_invariance") == "PASS"
        if not ok and not invalid:
            fail.append("%s batch-invariance witness is %s and the band is not marked "
                        "INVALID-CORRECTNESS -- a batch that is not invariant across its own "
                        "slots, or froze on one token, is measuring garbage at full speed (PP-26)"
                        % (tag, "absent" if not isinstance(wit, dict)
                           else repr(b["witness"].get("batch_invariance"))))
        if isinstance(wit, dict) and min_agree is not None:
            formed = b["witness"].get("m_formed")
            declared = b["witness"].get("declared_min")
            intra = b["witness"].get("intra_agree_to")
            if declared is not None and declared < min_agree:
                fail.append("%s witness.declared_min=%s is below perf-matrix.yaml "
                            "witness.min_agree_tokens=%s -- a witness that agrees on fewer tokens "
                            "than that agrees on nothing" % (tag, declared, min_agree))
            # PP-26 v3.1 (a), re-checked from the recorded number so a PASS token
            # cannot outrun the agreement it claims to rest on.
            if ok and intra is not None and intra < min_agree and not invalid:
                fail.append("%s witness says PASS but intra_agree_to=%s is below "
                            "witness.min_agree_tokens=%s -- the slots of one batch part before "
                            "the declared point, so the verdict token is not the measurement "
                            "(PP-26)" % (tag, intra, min_agree))
            # PP-26 v3.1 (b), the same way: a PASS over a recorded frozen run is
            # #2753 wearing a green token.
            run = b["witness"].get("max_constant_run")
            if ok and run is not None and max_run is not None and run >= max_run and not invalid:
                fail.append("%s witness says PASS but max_constant_run=%s reaches "
                            "witness.max_constant_run=%s -- a slot repeated one token id that "
                            "long, which is #2753's signature, so the verdict token is not the "
                            "measurement (PP-26)" % (tag, run, max_run))
            if formed is not None and formed < 2:
                report.append("%s witness.m_formed=%s: the batch never formed, so the witness "
                              "observed no batched path" % (tag, formed))

    # ---- an INVALID-CORRECTNESS band emits NO throughput --------------------
    present = [k for k in THROUGHPUT if b.get(k) is not None]
    if invalid and present:
        fail.append("%s is INVALID-CORRECTNESS and still carries %s -- a band whose tokens are wrong "
                    "has no throughput to report" % (tag, ", ".join(present)))
    if invalid:
        continue

    # ---- PP-4 every band carries agg, dec and prefill -----------------------
    for key in THROUGHPUT:
        if b.get(key) is None:
            fail.append("%s %s absent -- at schema_version>=3 an absent band metric is fatal; a "
                        "receipt that predates v3 is cited as historical instead" % (tag, key))
    if b.get("prefill_tok_per_sec") is not None and b.get("prefill_source") != "server":
        fail.append("%s prefill_source=%r -- a client-derived prefill is a different quantity and may "
                    "not be reported under the server-timed name" % (tag, b.get("prefill_source")))

    # ---- PP-27 the stream ---------------------------------------------------
    # The rule is the producer's (drain.rs stream_witness, PP-27): a server that
    # DECLARES `replayed` is replayed; a server that declares nothing (upstream
    # llama-server never will) is judged by the CLIENT witness alone, and is
    # conformant iff that witness says live; a server that declares `live` is
    # still overruled by a client witness that says replayed. The boundary is
    # INCLUSIVE on the live side, exactly as the producer computes it: a median
    # AT perf-matrix.yaml stream.live_ttft_over_e2e_max is live, above it is not.
    mode = b.get("stream_mode")
    verdict = b["stream_witness"].get("verdict") if isinstance(b.get("stream_witness"), dict) else None
    if mode == "replayed":
        fail.append("%s stream_mode='replayed' -- ttft, itl and decode are undefined without a live "
                    "stream (PP-27)" % (tag,))
    elif mode is None and verdict != "live":
        fail.append("%s stream_mode undeclared and the client witness did not say live (verdict=%r) -- "
                    "an undeclared stream is conformant only on the client's own evidence (PP-27)"
                    % (tag, verdict))
    elif mode not in (None, "live"):
        fail.append("%s stream_mode=%r is not a PP-27 token (live|replayed|null)" % (tag, mode))
    if not isinstance(b.get("stream_witness"), dict):
        fail.append("%s stream_witness absent -- the server's own `live` declaration is the only "
                    "witness, and a replayed stream declares itself live" % (tag,))
    else:
        if verdict == "replayed":
            fail.append("%s the client-side stream witness says REPLAYED" % (tag,))
        ratio = b["stream_witness"].get("client_ttft_over_e2e_median")
        if ratio is not None and ttft_max is not None and ratio > ttft_max:
            fail.append("%s client_ttft/e2e median %.4f > perf-matrix.yaml stream."
                        "live_ttft_over_e2e_max %s -- every token arrived at the end, which is a "
                        "replayed stream however it was declared" % (tag, ratio, ttft_max))

    # ---- PP-7 / PP-10 the raw rows -----------------------------------------
    rows = b.get("samples")
    if not isinstance(rows, list) or not rows:
        fail.append("%s samples[] absent or empty -- summary statistics cannot be resampled, so a "
                    "band without its raw rows forecloses every later re-derivation (PP-7)" % (tag,))
    else:
        win = b.get("window_ms")
        late = 0
        for i in range(len(rows)):
            issued = b["samples"][i].get("issued_ms")
            if issued is not None and win is not None and issued >= win:
                late += 1
        if late:
            fail.append("%s %d request(s) were ISSUED at or after window_ms -- work admitted after the "
                        "window closed is drain, and counting it inflates the window's throughput "
                        "(PP-10)" % (tag, late))
    if b.get("drain_ms") is None:
        fail.append("%s drain_ms absent -- the overshoot past the window is unrecorded (PP-10)" % (tag,))

    # ---- PP-23 the roofline -------------------------------------------------
    roof = b.get("roofline_tok_per_sec")
    dec = b.get("decode_tok_per_sec")
    if roof is not None and dec is not None and dec > roof:
        fail.append("%s decode %.4f tok/s exceeds roofline_tok_per_sec %.4f -- a single stream cannot "
                    "beat the memory bandwidth its own weights require, so the measurement or the "
                    "model_file size is wrong (PP-23)" % (tag, dec, roof))
    agg = b.get("aggregate_tok_per_sec")
    if roof is not None and agg is not None and agg > roof:
        report.append("%s aggregate %.4f tok/s is above the single-stream roofline %.4f, which is "
                      "expected under batching and is NOT a violation" % (tag, agg, roof))

    # ---- PP-24 a band above admission must be NA ---------------------------
    if slots_apr is not None and slots_llama is not None and slots_apr != slots_llama:
        ceiling = min(slots_apr, slots_llama)
        if c is not None and c > ceiling and status != "NA":
            fail.append("%s the two lanes admitted UNEQUAL slots (apr=%s, llama=%s) and this band is "
                        "above the derived ceiling %s while marked %r -- comparing them measures the "
                        "admission difference, not the servers (PP-24)"
                        % (tag, slots_apr, slots_llama, ceiling, status))

    # ---- PP-22 / PP-25 the join --------------------------------------------
    base = b.get("baseline")
    if isinstance(base, dict):
        bjk = b["baseline"].get("join_key")
        sjk = b.get("join_key")
        if not isinstance(sjk, dict) or not isinstance(bjk, dict):
            fail.append("%s carries a baseline with no join_key on one side -- an unkeyed pair is not "
                        "a pair (PP-22)" % (tag,))
        else:
            diff = sorted(k for k in set(sjk) | set(bjk) if sjk.get(k) != bjk.get(k))
            if diff:
                fail.append("%s join_key differs from its baseline on %s -- the two bands were not "
                            "measured under the same protocol and their quotient is not a ratio "
                            "(PP-22)" % (tag, diff))
            if b["join_key"].get("n_batch") == 1:
                fail.append("%s join_key.n_batch=1 -- a numeric batch size of one switches the "
                            "comparator's batching OFF; that is a crippled comparator, not a pinned "
                            "one (PP-15/PP-22)" % (tag,))
        bc = b["baseline"].get("client")
        base_sha = b["baseline"]["client"].get("sha256") if isinstance(bc, dict) else None
        if base_sha is None:
            fail.append("%s the baseline names no client -- PP-25 requires ONE client driving both "
                        "lanes and this receipt cannot show it" % (tag,))
        elif client_sha is not None and base_sha != client_sha:
            fail.append("%s the baseline was driven by client sha256=%s while the subject was driven "
                        "by %s -- two clients is two experiments (PP-25)"
                        % (tag, base_sha[:12], client_sha[:12]))

for line in report: print("REPORT ArmL1 " + line)
for line in fail:   print("FAIL ArmL1 " + line)
if not fail:
    print("PASS ArmL1 v3 schema rules")
sys.exit(1 if fail else 0)
PY_L1
}

arm_expiry_clock() {
  # THE UNMEASURED-CELL CLOCK. Split out of Arm A so that demoting Arm A to a
  # release arm (PP-6) does not silently move eleven armed rules to release with
  # it. Nothing here reads the receipt: it is a statement about the MATRIX, and
  # a merge can and should see a cell that has sat UNMEASURED past its deadline.
  local host="$1" workload="$2"
  python3 - "$MATRIX" "$host" "$workload" <<'PY_EXP'
import sys,yaml,os,datetime
m=yaml.safe_load(open(sys.argv[1])) or {}
host,wl=sys.argv[2],sys.argv[3]
# Injectable so the selftest can prove BOTH sides of the expiry boundary without
# waiting for a date to arrive. Defaults to the real clock.
TODAY=os.environ.get("PERF_GATE_TODAY") or datetime.date.today().isoformat()
bl=((m.get("baselines") or {}).get(host) or {}).get(wl) or {}

def resolve_expiry(bl, m):
    """The deadline -- which is not always a calendar date (PERF-056, #2777).

    All four W2 cells carried a hardcoded `expires: '2026-09-25'`. W2's expiry
    is dated from an EVENT instead -- "PERF-001 merge + 30 days" -- because a
    serialising server gives W2 nothing to measure and a blocking W2 would be
    permanently red. Under perf-matrix.yaml's own GROUNDING RULE that 2026-09-25
    was none of policy / inherited / ratchet: an invented continuous threshold,
    in the one file that says there are none.

    Returns (kind, detail):
      ("date",    "YYYY-MM-DD")  a fixed deadline; compare it against TODAY
      ("unarmed", "text")        an event-dated deadline whose event has not
                                 happened. The cell REPORTS and the text names
                                 the event, its status and its owner, so the
                                 wait is a printed line and not an assumption
      ("bad",     "reason")      the cell cannot be evaluated at all -> FAIL
    """
    fixed, cond = bl.get("expires"), bl.get("expires_after")
    if fixed and cond:
        return ("bad", "declares BOTH `expires` and `expires_after` -- two "
                       "clocks is no clock; pick the one this cell is given")
    if fixed:
        return ("date", str(fixed))
    if not cond:
        # UNCHANGED RULE, restated: defaulting an absent deadline to "never" is
        # exactly how an UNMEASURED cell becomes permanent.
        return ("bad", "UNMEASURED with no `expires` and no `expires_after` -- an "
                       "UNMEASURED cell without a deadline never expires")
    if not isinstance(cond, dict):
        return ("bad", "`expires_after` must be a mapping {anchor: TICKET, days: N}, "
                       "got %r" % (cond,))
    name, days = cond.get("anchor"), cond.get("days")
    if not name:
        return ("bad", "`expires_after` names no `anchor`")
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        return ("bad", "`expires_after.days` = %r is not a non-negative integer" % (days,))
    anchors = m.get("expiry_anchors") or {}
    if name not in anchors:
        return ("bad", "`expires_after.anchor` = %s is not declared under "
                       "`expiry_anchors:` -- an expiry hanging off an event this "
                       "file never defines is the absent-`expires` hole wearing a "
                       "different field name" % (name,))
    a = anchors[name] or {}
    merged, owner = a.get("merged_on"), a.get("owner") or "<no owner>"
    if merged:
        try:
            d = datetime.date.fromisoformat(str(merged))
        except ValueError:
            return ("bad", "anchor %s has merged_on=%r, which is not an ISO date"
                           % (name, merged))
        return ("date", (d + datetime.timedelta(days=days)).isoformat())
    if a.get("status") == "merged":
        return ("bad", "anchor %s says status: merged but records no `merged_on`; "
                       "a +%d-day clock cannot start from null" % (name, days))
    return ("unarmed", "expiry is %s merge + %d days; %s has NOT merged "
                       "(status=%r, owner=%s), so the clock has not started"
                       % (name, days, name, a.get("status"), owner))

status = bl.get("status")
if status == "NA":
    print("PASS ArmExpiry %s/%s is NA (%s; decided_by=%s) -- an NA cell has no clock"
          % (host, wl, " ".join(str(bl.get("reason") or "").split()), bl.get("decided_by")))
    sys.exit(0)
if status != "UNMEASURED":
    print("PASS ArmExpiry %s/%s status=%r -- no UNMEASURED clock to evaluate" % (host, wl, status))
    sys.exit(0)
kind, detail = resolve_expiry(bl, m)
if kind == "bad":
    print("FAIL ArmExpiry %s/%s baseline %s" % (host, wl, detail))
    sys.exit(1)
if kind == "unarmed":
    print("REPORT ArmExpiry %s/%s baseline UNMEASURED, %s, owner=%s"
          % (host, wl, detail, bl.get("owner")))
    sys.exit(0)
if detail < TODAY:
    print("FAIL ArmExpiry %s/%s baseline UNMEASURED and EXPIRED %s (today %s, owner=%s) — "
          "measure it or re-decide the cell; do not extend the date to stay green"
          % (host, wl, detail, TODAY, bl.get("owner")))
    sys.exit(1)
print("REPORT ArmExpiry %s/%s baseline UNMEASURED until %s, owner=%s"
      % (host, wl, detail, bl.get("owner")))
sys.exit(0)
PY_EXP
}

arm_a_self_regression() {
  # PP-31. SELF-REGRESSION, not scaling efficiency. The old arm ratcheted
  # (agg(c)/agg(1))/c UP-ONLY, a quantity that FALLS when agg(1) improves: a
  # faster single-client path made the gate redder, so the ratchet punished the
  # work it existed to protect. What is ratcheted now is the QUANTITY per band
  # -- agg(c), dec(c), prefill(1) -- against the value the last MEASURED receipt
  # on protected main achieved, and only the lower confidence bound at n >= n_min
  # can FAIL. scaling_efficiency and overhead_share are REPORTED, never gated.
  local receipt="$1" host="$2" workload="$3"
  python3 - "$receipt" "$MATRIX" "$host" "$workload" <<'PY_A'
import json,sys,yaml,math
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2]) ) or {}
host,wl=sys.argv[3],sys.argv[4]
mx_arm=((m.get("arms") or {}).get("A")) or {}
metrics=mx_arm.get("metrics") or {}
n_min=mx_arm.get("n_min")
bl=((m.get("baselines") or {}).get(host) or {}).get(wl) or {}

# One-sided t lower bound at 95%, df = n-1. The table is the spec's; >30 is the
# normal quantile. A ratchet that used the mean would fire on noise, and one
# that used the minimum would never fire at all.
T = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
     8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761,
     15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
     21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706,
     27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697}

def t_lower(values):
    """mean - t(df) * s / sqrt(n); None when n < 2."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= 0:
        return mean
    return mean - T.get(n - 1, 1.645) * math.sqrt(var) / math.sqrt(n)

WIRE = {"agg": "aggregate_tok_per_sec", "dec": "decode_tok_per_sec",
        "prefill": "prefill_tok_per_sec"}

# THE ARM MUST SAY WHEN IT MEASURED NOTHING (#2830). c=1 is the denominator, so
# a receipt whose only band is c=1 walks every print below and fires none of
# them: no c>1 band to report scaling on, and (when the baseline itself seeds
# nothing the configured metrics can match) no seeded cell to ratchet either.
# Silence used to read as "nothing failed" and the gate said VERDICT PASS. Every
# line this arm emits now goes through `say`, so a genuinely empty run is
# detectable and is reported -- never passed -- instead of vanishing.
emitted = False


def say(line):
    global emitted
    emitted = True
    print(line)


bands = r.get("bands") or []
base = next((b.get("aggregate_tok_per_sec") for b in bands
             if b.get("concurrency") == 1 and b.get("aggregate_tok_per_sec")), None)

samples = {}
for b in bands:
    c = b.get("concurrency")
    if c is None or b.get("status") in ("NA", "INVALID-CORRECTNESS"):
        continue
    for name in ("agg", "dec", "prefill"):
        v = b.get(WIRE[name])
        if v is not None:
            samples.setdefault((name, c), []).append(v)
    # REPORTED, NEVER GATED. scaling_efficiency falls when agg(1) rises; that is
    # information, not a verdict. A v3 producer reports it; for a historical
    # receipt that carries only the aggregates the gate derives it, so the line
    # a reader has always seen does not vanish with the ratchet that misused it.
    se = b.get("scaling_efficiency")
    if se is None and base and c > 1 and b.get("aggregate_tok_per_sec") is not None:
        se = (b["aggregate_tok_per_sec"] / base) / c
    if se is not None:
        say("REPORT ArmA c=%s scaling_efficiency=%.4f (reported, never ratcheted)" % (c, se))
    if b.get("overhead_share") is not None:
        say("REPORT ArmA c=%s overhead_share=%.4f (reported, never ratcheted)" % (c, b["overhead_share"]))

status = bl.get("status")
if status != "MEASURED":
    say("REPORT ArmA %s/%s baseline is %r -- a ratchet needs a measurement to ratchet FROM; "
        "nothing is gated until a conformant receipt seeds this cell" % (host, wl, status))
    sys.exit(0)
seed_bands = bl.get("bands") or {}
fail = False
for name in sorted(metrics):
    for c in metrics[name]:
        seed_cell = seed_bands.get("c%d" % c) or {}
        seed = seed_cell.get(name)
        got = samples.get((name, c))
        if seed is None:
            continue
        if not got:
            say("FAIL ArmA c=%d %s absent while the baseline seeds it at %s" % (c, name, seed))
            fail = True
            continue
        n = len(got)
        lcb = t_lower(got)
        if n_min is not None and n < n_min:
            say("REPORT ArmA c=%d %s n=%d < n_min=%s: reporting only (point %.4f vs seed %s)"
                % (c, name, n, n_min, sum(got) / n, seed))
            continue
        if lcb is None:
            say("REPORT ArmA c=%d %s n=%d: no lower bound is computable" % (c, name, n))
            continue
        if lcb < seed:
            say("FAIL ArmA c=%d %s lcb95=%.4f < baseline %s (n=%d) -- a ratchet moves one way"
                % (c, name, lcb, seed, n))
            fail = True
        else:
            say("PASS ArmA c=%d %s lcb95=%.4f >= baseline %s (n=%d)" % (c, name, lcb, seed, n))
if not emitted:
    # Every branch above ran and none of them had anything to say -- the
    # commonest cause is a c=1-only receipt (no c>1 band to compute scaling
    # from) paired with a baseline that measured no cell the matrix's metrics
    # config names. An arm with nothing to report is a REPORT, and a REPORT is
    # never spent as a PASS: fail closed so the gate's overall VERDICT can only
    # read FAIL, never PASS, on a run this arm never actually measured.
    print("REPORT ArmA scaling: c=1 only, no scaling measured")
    sys.exit(1)
sys.exit(1 if fail else 0)
PY_A
}

arm_l3_parity() {
  # PP-3 / PP-17 / §7.5. Non-inferiority against the comparator, with an
  # ASYMMETRIC gated set: at c=1 the aggregate IS the decode, so dec_ratio and
  # prefill_ratio carry the verdict; at c>1 the aggregate is the only number a
  # deployment feels, so agg_ratio carries it and decode is REPORTED. B1's
  # authorless 0.80 and B2's floor inherited from a document that never existed
  # in any ref are gone; the bound is `lcb95 >= 1 - delta` with delta in the
  # matrix. An (cell, band, metric) absent from `armed_by` is REPORTING: a gate
  # arms when a measurement arms it, never on a date.
  local receipt="$1" host="$2" workload="$3"
  python3 - "$receipt" "$MATRIX" "$host" "$workload" <<'PY_L3'
import json,sys,yaml
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2])) or {}
host,wl=sys.argv[3],sys.argv[4]
mx_arm=((m.get("arms") or {}).get("L3")) or {}
gated=mx_arm.get("gated") or {}
delta=mx_arm.get("delta") or {}
armed=(((mx_arm.get("armed_by") or {}).get(host) or {}).get(wl)) or {}
RATIOKEY={"agg_ratio":"agg","dec_ratio":"dec","prefill_ratio":"prefill"}
bl=((m.get("baselines") or {}).get(host) or {}).get(wl) or {}
if bl.get("status") == "NA":
    print("REPORT ArmL3 %s/%s is NA (%s; decided_by=%s) -- an NA cell has no comparator to be "
          "non-inferior to" % (host, wl, " ".join(str(bl.get("reason") or "").split()),
                               bl.get("decided_by")))
    sys.exit(0)
sv=r.get("schema_version")
V3 = isinstance(sv, int) and not isinstance(sv, bool) and sv >= 3
bands=r.get("bands") or []
if not bands:
    print("FAIL ArmL3 no bands present"); sys.exit(1)
fail=False
for b in bands:
    c=b.get("concurrency")
    st=b.get("comparator_status")
    if st in ("NOT_APPLICABLE","NA","UNMEASURED"):
        print("REPORT ArmL3 c=%s %s (Arm A still gates this cell)" % (c, st)); continue
    if not V3:
        # A v2 band carries the ratio as a bare scalar. It is READ -- the
        # artifact predates the paired form and refusing it would fail every
        # receipt in evidence/ for not having been written later -- and it is
        # never GATED, because a scalar records the quotient and discards the
        # comparator run, the concurrency and the protocol that made it valid.
        ag, de = b.get("agg_ratio"), b.get("decode_ratio")
        if ag is None and de is None:
            print("FAIL ArmL3 c=%s: comparator_status=%r and the band carries no ratio at all"
                  % (c, st))
            fail=True
        else:
            print("REPORT ArmL3 c=%s historical agg_ratio=%s decode_ratio=%s (schema_version<3: a "
                  "bare scalar ratio is read, never gated)" % (c, ag, de))
        continue
    ratios=b.get("ratios")
    if not isinstance(ratios,dict):
        print("FAIL ArmL3 c=%s: comparator_status=%r but no `ratios` object -- a ratio is "
              "representable ONLY inside a band that carries its own paired baseline (PP-3)"
              % (c, st))
        fail=True; continue
    want = gated.get("c1") if c == 1 else gated.get("c_gt_1")
    want = list(want or [])
    armed_cell = (armed.get("c%s" % c) or {})
    for metric in want:
        rk = RATIOKEY[metric]
        if not isinstance(ratios.get(rk), dict):
            print("FAIL ArmL3 c=%s %s absent from `ratios` and this band class gates it" % (c, metric))
            fail=True; continue
        lcb = b["ratios"][rk].get("lcb95")
        bound = 1 - (delta.get(metric) or 0)
        if metric not in armed_cell:
            print("REPORT ArmL3 c=%s %s lcb95=%s (bound %.4f) -- (host=%s, %s, c=%s, %s) is not in "
                  "arms.L3.armed_by, so this metric REPORTS until a measurement arms it"
                  % (c, metric, lcb, bound, host, wl, c, metric))
            continue
        if lcb is None:
            print("FAIL ArmL3 c=%s %s is armed and carries no lcb95 -- an armed metric with no "
                  "interval is a point estimate wearing a bound" % (c, metric))
            fail=True; continue
        if lcb < bound:
            print("FAIL ArmL3 c=%s %s lcb95=%.4f < %.4f" % (c, metric, lcb, bound)); fail=True
        else:
            print("PASS ArmL3 c=%s %s lcb95=%.4f >= %.4f" % (c, metric, lcb, bound))
    for metric in sorted(RATIOKEY):
        rk = RATIOKEY[metric]
        if metric in want or not isinstance(ratios.get(rk), dict):
            continue
        print("REPORT ArmL3 c=%s %s point=%s lcb95=%s (reported at this band class, not gated)"
              % (c, metric, b["ratios"][rk].get("point"), b["ratios"][rk].get("lcb95")))
sys.exit(1 if fail else 0)
PY_L3
}

arm_d_memory() {
  # REPORTING: "reporting" governs the THRESHOLD, not the FIELD. A reporting arm
  # whose metric may silently vanish instruments nothing, so at release the
  # fields must be PRESENT even though no bound is applied to them yet.
  local receipt="$1" phase="$2" cell="${3:-}" hist="${4:-no}"
  if [ "$phase" = release ] && historical_for_unmeasured "$cell" "$hist"; then
    echo "REPORT ArmD historical receipt (schema_version<3) cited for an UNMEASURED cell: the kv block exists from the first conformant receipt (PP-2); nothing is instrumented here because nothing here is measured"
    return 0
  fi
  python3 - "$receipt" "$phase" "$MATRIX" <<'PY_D'
import json,sys,yaml
r=json.load(open(sys.argv[1])); phase=sys.argv[2]
m=yaml.safe_load(open(sys.argv[3])) or {}
mx_arm=((m.get("arms") or {}).get("D")) or {}
note_below=mx_arm.get("note_kv_utilization_below")
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
if note_below is not None and util<note_below and kv["admission_rejected"]>0:
    print("NOTE  ArmD refusing work while memory sits reserved-and-empty is the "
          "contiguous-allocation signature this arm exists to catch")
sys.exit(0)
PY_D
}

arm_e_interference() {
  # W2 ONLY. Arm E is what chunked prefill exists to move; without it a batching
  # implementation that blocks the GPU on an 8192-token prefill scores as a win
  # on Arm A.
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

arm_c_signature() {
  # THE STALENESS ARM (PP-21) AND THE ANCESTRY ARM (PP-18).
  #
  # Two hosts are not CI runners and the fully-comparated one is do-not-revive,
  # so the gate cannot run ON the host that measures. What arrives here is a
  # FILE. Unsigned, that file binds to no host and no commit -- anyone can write
  # one, and this gate would read it as evidence.
  #
  # The crypto and the containment test live in scripts/lib/receipt_sig.py,
  # which scripts/perf_receipt_sign.sh also calls, so the signed payload cannot
  # drift between producer and verifier. This function owns the PHASE rule, the
  # andon line and PP-18's ancestry test; it owns no key handling of its own.
  local receipt="$1" host="$2" phase="$3" commit="$4" cell="${5:-}" hist="${6:-no}"
  if [ "$phase" != release ]; then
    # The only legal skip in this arm, and it is spec-mandated: the rule is
    # scoped to release. No PR can supply a host receipt, so wiring it at merge
    # would be a required check that can never PASS -- the mirror of one that
    # can never fail.
    echo "SKIP  ArmC-sig signature+freshness is a RELEASE-phase rule (phase=$phase)"
    return 0
  fi
  if [ -z "$commit" ]; then
    # main() rejects this as a usage error; run_gate can also be called
    # directly. An arm handed no input stands down silently unless told not to,
    # and that is the cannot-fail shape.
    echo "FAIL ArmC-sig NO-COMMIT-UNDER-TEST: release phase with no commit-under-test."
    echo "      The staleness arm has nothing to compare, and an arm with no input is not a passing arm."
    return 1
  fi
  if historical_for_unmeasured "$cell" "$hist"; then
    echo "REPORT ArmC-sig historical receipt (schema_version<3) cited for an UNMEASURED cell: no conformant receipt exists to bind to host=$host and commit-under-test=$commit; PP-21 applies from the first conformant receipt, and the cell's owner and expiry are ArmExpiry's to report"
    return 0
  fi
  python3 - "$receipt" "$ROOT" "$host" "$commit" <<'PY_SIG'
import json,os,subprocess,sys
sys.path.insert(0, os.path.join(sys.argv[2], "scripts", "lib"))
import receipt_sig
r=json.load(open(sys.argv[1])); host=sys.argv[3]; cut=sys.argv[4]
# The andon line: what this receipt CLAIMS, printed before anything is
# believed. `<absent>` here is the shape of the defect this arm closes.
sig=r.get("signature") or {}
prov=r.get("provenance") or {}
print("REPORT ArmC-sig receipt claims commit=%s host=%s key_id=%s "
      "(commit-under-test=%s)"
      % (r.get("commit") or "<absent>", prov.get("host") or "<absent>",
         sig.get("key_id") or "<absent>", cut))
# Injectable so the selftest can build a two-commit repo and prove BOTH
# polarities of containment. Defaults to this checkout.
git_dir = os.environ.get("PERF_GATE_GIT_DIR") or sys.argv[2]
fails, report = receipt_sig.verify_receipt(
    r, host, cut, os.environ.get("APR_PERF_RECEIPT_KEYRING"), git_dir)
for line in report:
    print(line)

# PP-18. `receipt.commit` says which tree the RUN was labelled with. The SERVED
# binary and the CLIENT binary each carry their own commit, and neither is
# implied by the label: a receipt can name a commit under test while the binary
# that answered came from an unrelated branch. Both must be ancestors of the
# commit under test, or the receipt describes code the release does not contain.
def ancestor(candidate):
    if not candidate:
        return "ABSENT"
    proc = subprocess.run(["git", "-C", git_dir, "merge-base", "--is-ancestor",
                           str(candidate), cut],
                          capture_output=True, text=True, check=False)
    return "OK" if proc.returncode == 0 else "NO"

declared = []
if isinstance(prov.get("subject"), dict):
    declared.append(("SUBJECT", prov["subject"].get("commit")))
if isinstance(prov.get("client"), dict):
    declared.append(("CLIENT", prov["client"].get("commit")))
for label, got in declared:
    verdict = ancestor(got)
    if verdict == "ABSENT":
        fails.append(("%s-NO-COMMIT" % label,
                      "provenance.%s carries no commit; the binary that %s is unidentified"
                      % (label.lower(), "served" if label == "SUBJECT" else "drove the load")))
    elif verdict == "NO":
        fails.append(("%s-NOT-ANCESTOR" % label,
                      "provenance.%s.commit=%s is not an ancestor of the commit under test %s -- "
                      "the receipt measures code this release does not contain"
                      % (label.lower(), got, cut)))
    else:
        print("REPORT ArmC-sig %s commit %s is an ancestor of %s" % (label.lower(), got, cut))

for code, message in fails:
    print("FAIL ArmC-sig %s: %s" % (code, message))
sys.exit(1 if fails else 0)
PY_SIG
}

cell_completeness() {
  # PP-1. The expected cell set is enumerated in perf-matrix.yaml, and an NA
  # cell PASSES WITH NO RECEIPT -- loudly, naming its decider. An NA host with
  # no measurable class had no way to be expressed at all before, so its cells
  # were UNMEASURED forever with an expiry nothing could clear.
  local receipt="$1" host="$2" workload="$3"
  python3 - "$receipt" "$MATRIX" "$host" "$workload" <<'PY_CELLS'
import json,sys,yaml
r=json.load(open(sys.argv[1])); m=yaml.safe_load(open(sys.argv[2])) or {}
host,wl=sys.argv[3],sys.argv[4]
bl=((m.get("baselines") or {}).get(host) or {}).get(wl) or {}
if bl.get("status") == "NA":
    print("PASS cells host=%s workload=%s is NA (%s; decided_by=%s) -- an NA cell is excluded from "
          "the denominator and needs no receipt"
          % (host, wl, " ".join(str(bl.get("reason") or "").split()), bl.get("decided_by")))
    sys.exit(0)
want=set((m.get("ladder") or {}).get("declared") or [])
have={b.get("concurrency") for b in (r.get("bands") or [])}
# PP-24: a band the servers did not admit is not a missing cell. `ladder.derived`
# is the server-reported ceiling, and a band outside it is excused BY NAME.
derived=(r.get("ladder") or {}).get("derived")
if isinstance(derived, list) and derived:
    excused=sorted(want - set(derived))
    if excused:
        print("REPORT cells host=%s bands %s are outside ladder.derived=%s (server-reported "
              "admission) and are not counted missing" % (host, excused, sorted(derived)))
    want = want & set(derived)
missing=sorted(want-have)
if missing:
    print(f"FAIL cells host={host} missing bands {missing} — a missing cell is not a passing cell")
    sys.exit(1)
print(f"PASS cells host={host} all bands {sorted(want)} present")
PY_CELLS
}

run_gate() {
  local host="$1" phase="$2" workload="$3" receipt="$4" commit="${5:-}" rc=0
  [ -f "$receipt" ] || die "receipt not found: $receipt"
  local cell hist
  cell="$(cell_status "$host" "$workload")"
  hist="$(receipt_is_historical "$receipt")"
  run_phased C     "$phase" arm_c_integrity     "$receipt" || rc=1
  run_phased L1    "$phase" arm_l1_schema       "$receipt" "$host" || rc=1
  run_phased C_sig "$phase" arm_c_signature     "$receipt" "$host" "$phase" "$commit" "$cell" "$hist" || rc=1
  run_phased expiry "$phase" arm_expiry_clock   "$host" "$workload" || rc=1
  run_phased A     "$phase" arm_a_self_regression "$receipt" "$host" "$workload" || rc=1
  run_phased L3    "$phase" arm_l3_parity       "$receipt" "$host" "$workload" || rc=1
  run_phased D     "$phase" arm_d_memory        "$receipt" "$phase" "$cell" "$hist" || rc=1
  run_phased E     "$phase" arm_e_interference  "$receipt" "$phase" "$workload" || rc=1
  run_phased cells "$phase" cell_completeness   "$receipt" "$host" "$workload" || rc=1
  if [ "$rc" = 0 ]; then echo "VERDICT PASS host=$host phase=$phase workload=$workload"
  else echo "VERDICT FAIL host=$host phase=$phase workload=$workload"; fi
  return "$rc"
}

# ------------------------------------------------------- selftest fixtures --
# THE OK FIXTURE IS A v3 RECEIPT and carries every key the L1 rules read, so a
# rule that stops firing shows up as a row that stops discriminating rather than
# as a key nothing exercises. The v2 fixture beside it is the HISTORICAL shape:
# the same gate must still read an artifact written before the v3 wire existed,
# and must say so out loud rather than failing it for not having been written in
# the future.
#
# THEY ARE FILES, not string literals inside this script. A gate whose only
# passing input is a string literal inside itself measures the string literal
# (PERF-004); a committed fixture can be diffed, re-read by another tool, and
# handed to the Rust producer as the shape it must emit.
FIXTURES="$ROOT/tests/fixtures/perf-gate"

_write_fixtures() { # tmpdir
  cp "$FIXTURES/receipt-v3-ok.json" "$1/ok3.json"
  cp "$FIXTURES/receipt-v2-historical.json" "$1/ok2.json"
}

# ------------------------------------------------------------ selftest -----
# A guard is admissible only if a mutation of the thing it guards turns it RED.
# Each row states what it mutates and which arm must reject it.
#
# THE NAMES ARE THE SPEC'S. PP-LLAMA-001-MASTER.md §6 gives every ARMED rule a
# must-fire and a must-not-fire case name, and scripts/spec_conformance.sh JOINS
# that table to `--list-selftests` below. A row renamed here without renaming it
# there turns that guard RED, which is the point: the table and the tree cannot
# drift apart in silence.
SELFTEST_NAMES=()
LIST_ONLY="${SELFTEST_LIST_ONLY:-0}"

_reg() { # register a case name; returns 1 in list mode so the case body is skipped
  SELFTEST_NAMES+=("$1")
  if [ "$LIST_ONLY" = 1 ]; then return 1; fi
  return 0
}

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

  _verdict() { # name, expect, got, output, needle
    local name="$1" expect="$2" got="$3" out="$4" needle="${5:-}"
    if [ "$got" != "$expect" ]; then
      printf '  BROKE %-34s expected %s got %s\n' "$name" "$expect" "$got"
      fail=$((fail + 1))
      return 0
    fi
    # POLARITY IS NOT ENOUGH. A row that only checks pass/fail is green when
    # the gate fails for an unrelated reason, which is how a wrong diagnosis
    # ships looking tested.
    if [ -n "$needle" ]; then
      case "$out" in
        *"$needle"*) : ;;
        *) printf '  BROKE %-34s %s but never said %s\n' "$name" "$expect" "$needle"
           fail=$((fail + 1))
           return 0 ;;
      esac
    fi
    printf '  ok    %-34s expect=%s\n' "$name" "$expect"
    pass=$((pass + 1))
  }

  _mut() { # dest-name, source-path, python body operating on `r`
    if [ "$LIST_ONLY" = 1 ]; then return 0; fi
    python3 - "$2" "$tmp/$1.json" "$3" <<'PY_MUT'
import copy, json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    r = json.load(fh)
# Mutations edit `r` IN PLACE. Rebinding it in the body would be invisible here,
# and a fixture that silently did not change is a case that proves nothing.
exec(sys.argv[3], {"r": r, "json": json, "copy": copy})
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(r, fh)
PY_MUT
    printf '%s\n' "$tmp/$1.json"
  }

  _sigprep() { # receipt-path, host -- stamp the commit/host/ancestry, then sign
    python3 -c 'import json,sys
with open(sys.argv[1]) as fh:
    r = json.load(fh)
r["commit"] = sys.argv[2]
r["provenance"]["host"] = sys.argv[3]
if isinstance(r["provenance"].get("subject"), dict):
    r["provenance"]["subject"]["commit"] = sys.argv[4]
if isinstance(r["provenance"].get("client"), dict):
    r["provenance"]["client"]["commit"] = sys.argv[4]
with open(sys.argv[1], "w") as fh:
    json.dump(r, fh)' "$1" "$sigc1" "$2" "$sigc0"
    python3 "$ROOT/scripts/lib/receipt_sig.py" --sign --in "$1" \
      --out "$1" --key-id "$2-selftest" --keyring "$sigkr" \
      --signed-at 2026-08-29T00:00:00Z >/dev/null
  }

  _row() { # name, fixture, phase, workload, host, matrix(empty=committed), expect, needle, today, raw
    _reg "$1" || return 0
    local name="$1" f="$2" ph="$3" wl="$4" h="$5" mx="$6" expect="$7" needle="${8:-}" today="${9:-}" raw="${10:-}"
    local saved="$MATRIX" out rc=0 got
    if [ -n "$mx" ]; then MATRIX="$mx"; fi
    if [ "$ph" = release ]; then
      # `raw` = the receipt goes in exactly as committed: unstamped, UNSIGNED.
      # That is the production shape of the pre-v3 lambda receipt, and the
      # historical rows below are about that shape; _sigprep would erase it.
      if [ -z "$raw" ]; then _sigprep "$f" "$h"; fi
      out="$( ( export APR_PERF_RECEIPT_KEYRING="$sigkr"; export PERF_GATE_GIT_DIR="$sigrepo"; \
                if [ -n "$today" ]; then export PERF_GATE_TODAY="$today"; fi; \
                run_gate "$h" release "$wl" "$f" "$sigc1" ) 2>&1 )" || rc=1
    else
      out="$( ( if [ -n "$today" ]; then export PERF_GATE_TODAY="$today"; fi; \
                run_gate "$h" "$ph" "$wl" "$f" ) 2>&1 )" || rc=1
    fi
    MATRIX="$saved"
    got=pass
    [ "$rc" = 0 ] || got=fail
    _verdict "$name" "$expect" "$got" "$out" "$needle"
  }

  # A MATRIX NO WELL-FORMED COMMITTED ONE CAN BE. Several rules below have FAIL
  # branches the shipped file cannot reach, and a branch no case can enter is a
  # branch nothing proves. Each edit ASSERTS its own anchor text, so a matrix
  # that changes shape breaks the row loudly instead of silently ceasing to test
  # anything.
  _mx() { # variant-name, python-edit-body -> prints the variant matrix path
    if [ "$LIST_ONLY" = 1 ]; then printf '%s\n' "$tmp/matrix-$1.yaml"; return 0; fi
    python3 - "$ROOT/scripts/perf-matrix.yaml" "$tmp/matrix-$1.yaml" "$2" <<'PY_MX'
import sys
src, dst, edit = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding="utf-8") as fh:
    s = fh.read()
for pair in edit.split("\x1e"):
    if not pair:
        continue
    old, new = pair.split("\x1f")
    assert old in s, (
        "selftest matrix edit did not match %r. The committed matrix no longer has\n"
        "the shape these rows were written against -- most likely a cell lost its\n"
        "`expires_after:` event-dated expiry, an `expiry_anchors:` entry moved, or\n"
        "an arm's `phase:`/`armed_by:` block was reshaped. Fix the matrix or the\n"
        "rows; do not delete the rows." % old[:60])
    s = s.replace(old, new)
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(s)
print(dst)
PY_MX
  }

  # ---- the signing repository, keyring and the two fixtures ----------------
  local sigrepo sigc0 sigc1 sigkr sigkeya sigkeyb sigkeyc
  sigkeya=a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1
  sigkeyb=b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2
  sigkeyc=c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3
  sigrepo="$tmp/sigrepo"
  sigkr="$tmp/keyring"
  sigc0=0000000000000000000000000000000000000000
  sigc1=1111111111111111111111111111111111111111
  local OK3 OK2
  OK3="$tmp/ok3.json"
  OK2="$tmp/ok2.json"
  if [ "$LIST_ONLY" != 1 ]; then
    mkdir -p "$sigrepo"
    git -C "$sigrepo" init -q --template= >/dev/null 2>&1
    git -C "$sigrepo" config user.email selftest@example.invalid
    git -C "$sigrepo" config user.name 'perf-gate selftest'
    git -C "$sigrepo" config commit.gpgsign false
    git -C "$sigrepo" config core.hooksPath "$tmp/nohooks"
    printf '0\n' > "$sigrepo/f0"
    git -C "$sigrepo" add -A
    git -C "$sigrepo" commit -q -m c0
    sigc0="$(git -C "$sigrepo" rev-parse HEAD)"
    printf '1\n' > "$sigrepo/f1"
    git -C "$sigrepo" add -A
    git -C "$sigrepo" commit -q -m c1
    sigc1="$(git -C "$sigrepo" rev-parse HEAD)"
    {
      printf 'lambda-selftest %s\n' "$sigkeya"
      printf 'gx10-selftest %s\n' "$sigkeyb"
      printf 'mini-selftest %s\n' "$sigkeyc"
    } > "$sigkr"
    _write_fixtures "$tmp"
  fi

  local F
  # ---- Arm C integrity ------------------------------------------------------
  F="$(_mut healthy "$OK3" 'pass')"
  _row baseline_healthy            "$F" merge W1 lambda "" pass "PASS ArmC integrity"
  F="$(_mut completed_lt "$OK3" 'r["completed"]=15')"
  _row completed_lt_requested      "$F" merge W1 lambda "" fail "completed(15) != requested(16)"
  # PP-5. Renamed from a_timeout_is_fatal to the §6 name; the fixture and the
  # rule are unchanged.
  F="$(_mut timeout1 "$OK3" 'r["timeouts"]=1')"
  _row timeout_fatal               "$F" merge W1 lambda "" fail "timeouts=1"
  # AN ABSENT COUNTER IS NOT A ZERO COUNTER. This read `r.get("timeouts",0)`, so
  # a receipt that never counted timeouts was indistinguishable from one that
  # counted none -- and no producer in this tree counts them at all, so the
  # default was doing all the work on every real artifact.
  F="$(_mut timeoutabs "$OK3" 'r.pop("timeouts")')"
  _row timeouts_absent_is_not_zero "$F" merge W1 lambda "" fail "timeouts absent"
  F="$(_mut tokabs "$OK3" 'r["tokenization"]["method"]=""')"
  _row tokenization_absent         "$F" merge W1 lambda "" fail "tokenization.method absent"
  F="$(_mut tokok "$OK3" 'pass')"
  _row tokenization_ok             "$F" merge W1 lambda "" pass "PASS ArmC integrity"
  F="$(_mut drainabs "$OK3" 'r["drain_ms"]=None')"
  _row drain_ms_absent             "$F" merge W1 lambda "" fail "drain_ms absent"
  # PP-5's must-not-fire twin: the drain IS recorded, top level and per band.
  F="$(_mut drainok "$OK3" 'pass')"
  _row drain_ok                    "$F" merge W1 lambda "" pass "PASS ArmL1"
  # THIS ROW WAS ONCE GREEN FOR THE WRONG REASON (PERF-047): the substitution
  # appended a stray `}` and the fixture failed to parse, so neutering the
  # zero-token rule left the table intact. It changes ONE field now.
  F="$(_mut zerotok "$OK3" '[b.update(tokens_total=0) for b in r["bands"]]')"
  _row zero_token_response         "$F" merge W1 lambda "" fail "zero-token response is a failure"

  # ---- PP-2 / PP-13: the server answers for itself --------------------------
  F="$(_mut cfgmiss "$OK3" 'r["provenance"]["server_config"]=None')"
  _row config_missing              "$F" merge W1 lambda "" fail "provenance.server_config is absent"
  F="$(_mut cfgok "$OK3" 'pass')"
  _row config_present              "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut inferred "$OK3" 'r["provenance"]["server_config"]["compute_class"]="cpu"')"
  _row inferred_field              "$F" merge W1 lambda "" fail "disagrees with server_config.compute_class"
  F="$(_mut reported "$OK3" 'pass')"
  _row reported_field              "$F" merge W1 lambda "" pass "PASS ArmL1"
  # The other half of PP-13: an admission figure with no server configuration
  # behind it is INFERRED, whatever field it sits in.
  F="$(_mut inferred2 "$OK3" 'r["provenance"]["server_config"]=None')"
  _row inferred_slots_without_config "$F" merge W1 lambda "" fail "ladder.slots_admitted is present while"
  # PP-24's other half: `derived` is RECOMPUTED from declared + admissions. A
  # receipt that writes its own (here: dropping c=8 and c=16 while both lanes
  # admitted 16) is excusing bands the servers granted.
  F="$(_mut ladderforged "$OK3" 'r["ladder"]["derived"]=[1,4]')"
  _row ladder_derived_forged         "$F" merge W1 lambda "" fail "ladder.derived=[1, 4] does not follow"
  F="$(_mut ladderok "$OK3" 'r["ladder"]["slots_admitted"]["llama"]=8; r["ladder"]["derived"]=[1,4,8]; r["bands"]=[b for b in r["bands"] if b["concurrency"]<=8]')"
  _row ladder_derived_ok             "$F" merge W1 lambda "" pass "PASS ArmL1"

  # ---- PP-14: autofit may not overrule an explicit argument -----------------
  F="$(_mut autofitbad "$OK3" 'r["provenance"]["server_config"]["offload"]={"autofit_applied":True,"explicit_args":["--gpu-layers","28"]}')"
  _row autofit_override            "$F" merge W1 lambda "" fail "autofit overrode an operator decision"
  F="$(_mut autofitok "$OK3" 'r["provenance"]["server_config"]["offload"]={"autofit_applied":True,"explicit_args":[]}')"
  _row autofit_ok                  "$F" merge W1 lambda "" pass "PASS ArmL1"

  # ---- PP-30: the clock -----------------------------------------------------
  F="$(_mut nots "$OK3" 'r["provenance"].pop("started_utc")')"
  _row timestamp_absent            "$F" merge W1 lambda "" fail "provenance.started_utc absent"
  F="$(_mut tsok "$OK3" 'pass')"
  _row timestamp_ok                "$F" merge W1 lambda "" pass "PASS ArmL1"

  # ---- PP-24: unequal admission is not a comparison -------------------------
  F="$(_mut admunequal "$OK3" 'r["ladder"]["slots_admitted"]={"apr":4,"llama":16}')"
  _row admission_unequal           "$F" merge W1 lambda "" fail "admitted UNEQUAL slots"
  # PP-24: the ladder DERIVES from the smaller admission (4), so `derived`
  # shrinks with it; the bands above it are NA with a decided_by, not missing.
  F="$(_mut admna "$OK3" 'r["ladder"]["slots_admitted"]={"apr":4,"llama":16}
r["ladder"]["derived"]=[c for c in r["ladder"]["declared"] if c<=4]
for b in r["bands"]:
    if b["concurrency"] > 4:
        b["status"]="NA"
        b["comparator_status"]="NOT_APPLICABLE"')"
  _row admission_na                "$F" merge W1 lambda "" pass "PASS ArmL1"

  # ---- PP-27: a replayed stream is not a stream -----------------------------
  F="$(_mut streplay "$OK3" '[b.update(stream_mode="replayed") for b in r["bands"]]')"
  _row stream_replayed             "$F" merge W1 lambda "" fail "stream_mode='replayed'"
  F="$(_mut stlive "$OK3" 'pass')"
  _row stream_live                 "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut stabs "$OK3" '[b.update(stream_mode=None) or b.pop("stream_witness") for b in r["bands"]]')"
  _row stream_absent               "$F" merge W1 lambda "" fail "stream_witness absent"
  # An UNDECLARED stream (upstream llama-server never declares) is judged by the
  # client witness alone: live is conformant, replayed is not.
  F="$(_mut stundlive "$OK3" '[b.update(stream_mode=None) for b in r["bands"]]')"
  _row stream_undeclared_client_live   "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut stundrep "$OK3" '[b.update(stream_mode=None) or b["stream_witness"].update(verdict="replayed") for b in r["bands"]]')"
  _row stream_undeclared_client_replayed "$F" merge W1 lambda "" fail "client witness did not say live"
  # THE WITNESS, NOT THE DECLARATION. A replayed stream declares itself live;
  # only the client's own ttft/e2e ratio can contradict it.
  F="$(_mut stwit "$OK3" '[b["stream_witness"].update(client_ttft_over_e2e_median=0.99) for b in r["bands"]]')"
  _row stream_witness_contradicts  "$F" merge W1 lambda "" fail "which is a replayed stream however it was declared"

  # ---- PP-28: the sampler ---------------------------------------------------
  F="$(_mut sampunpin "$OK3" 'r["short_of_n_predict"]=7')"
  _row sampler_unpinned            "$F" merge W1 lambda "" fail "short_of_n_predict=7"
  F="$(_mut samppin "$OK3" 'pass')"
  _row sampler_pinned              "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut samptemp "$OK3" 'r["protocol"]["sampler"]["temperature"]=0.7')"
  _row sampler_temperature_unpinned "$F" merge W1 lambda "" fail "the sampler is pinned by the matrix"

  # ---- PP-26: batch invariance ---------------------------------------------
  F="$(_mut bifail "$OK3" '[b.update(witness=None) for b in r["bands"] if b["concurrency"]>1]')"
  _row batch_invariance_fail       "$F" merge W1 lambda "" fail "batch-invariance witness is absent"
  # The must-not-fire is NOT "the witness passed": it is a band HONESTLY marked
  # INVALID-CORRECTNESS, carrying no throughput at all. A gate that only accepted
  # the green witness would refuse the only correct way to report a broken batch.
  F="$(_mut biok "$OK3" 'for b in r["bands"]:
    if b["concurrency"] == 4:
        b["status"]="INVALID-CORRECTNESS"
        b["witness"]={"batch_invariance":"FAIL","divergence_at":3,"declared_min":128,"m_formed":4,"source":"server"}
        for k in ("aggregate_tok_per_sec","decode_tok_per_sec","prefill_tok_per_sec","prefill_source"):
            b.pop(k, None)')"
  _row batch_invariance_ok         "$F" merge W1 lambda "" pass "PASS ArmL1"
  # PP-26 v3.1 (a): the verdict token says PASS while the recorded intra-batch
  # agreement is below the declared point. The token is not the measurement.
  F="$(_mut biintra "$OK3" 'for b in r["bands"]:
    if b["concurrency"] == 4:
        b["witness"]={"batch_invariance":"PASS","divergence_at":3,"intra_agree_to":3,"max_constant_run":2,"declared_min":128,"m_formed":4,"source":"server"}')"
  _row witness_intra_below_declared "$F" merge W1 lambda "" fail "intra_agree_to=3 is below"
  # PP-26 v3.1 (b): PASS token over a recorded 116-step constant run (#2753).
  F="$(_mut bifrozen "$OK3" 'for b in r["bands"]:
    if b["concurrency"] == 4:
        b["witness"]={"batch_invariance":"PASS","divergence_at":0,"intra_agree_to":128,"max_constant_run":116,"declared_min":128,"m_formed":4,"source":"server"}')"
  _row witness_frozen_run_under_pass "$F" merge W1 lambda "" fail "max_constant_run=116 reaches"
  F="$(_mut biloud "$OK3" 'for b in r["bands"]:
    if b["concurrency"] == 4:
        b["status"]="INVALID-CORRECTNESS"')"
  _row invalid_band_keeps_throughput "$F" merge W1 lambda "" fail "has no throughput to report"

  # ---- PP-4: every band carries agg, dec and prefill ------------------------
  F="$(_mut nometric "$OK3" '[b.pop("prefill_tok_per_sec") for b in r["bands"] if b["concurrency"]==4]')"
  _row band_metric_absent          "$F" merge W1 lambda "" fail "prefill_tok_per_sec absent"
  # The v2 artifact carries no prefill at all and must still be READ, once, as
  # historical -- not failed for having been written before the field existed.
  F="$(_mut hist "$OK2" 'pass')"
  _row historical_cited            "$F" merge W1 lambda "" pass "historical"

  # ---- PP-7 / PP-10: the raw rows and the drain ----------------------------
  F="$(_mut nosamples "$OK3" '[b.update(samples=[]) for b in r["bands"]]')"
  _row samples_stripped            "$F" merge W1 lambda "" fail "samples[] absent or empty"
  F="$(_mut samplesok "$OK3" 'pass')"
  _row samples_ok                  "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut postclose "$OK3" 'r["bands"][0]["samples"][1]["issued_ms"]=60001.0')"
  _row post_close_request          "$F" merge W1 lambda "" fail "were ISSUED at or after window_ms"
  F="$(_mut drainrec "$OK3" 'pass')"
  _row drain_recorded              "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut nobanddrain "$OK3" '[b.update(drain_ms=None) for b in r["bands"]]')"
  _row band_drain_absent           "$F" merge W1 lambda "" fail "drain_ms absent -- the overshoot"

  # ---- PP-16: the compute class a build can actually reach ------------------
  F="$(_mut unreach "$OK3" 'r["provenance"]["compute_class"]="wgpu"
r["provenance"]["feature_set"]="wgpu".split()
r["provenance"]["server_config"]["compute_class"]="wgpu"')"
  _row class_unreachable           "$F" merge W1 lambda "" fail "hosts.lambda.compute_class"
  # `mini` is NA: no build in this tree has a Metal inference path, so its cells
  # are excluded from the denominator by decision rather than by omission.
  F="$(_mut mini "$OK3" 'r["provenance"]["host"]="mini"
r["provenance"]["compute_class"]="cpu"
r["provenance"]["accelerator"]="m4"
r["provenance"]["feature_set"]="cpu".split()
r["provenance"]["server_config"]["compute_class"]="cpu"')"
  _row class_na                    "$F" merge W1 mini   "" pass "class host=mini is NA"

  # ---- PP-23: the roofline --------------------------------------------------
  F="$(_mut roofbad "$OK3" '[b.update(decode_tok_per_sec=5000.0) for b in r["bands"] if b["concurrency"]==1]')"
  _row roofline_exceeded           "$F" merge W1 lambda "" fail "exceeds roofline_tok_per_sec"
  # An AGGREGATE above the single-stream roofline is what batching is for. The
  # committed fixture's c=16 band is already above it, so this row would be
  # vacuous if the rule did not distinguish the two.
  F="$(_mut roofagg "$OK3" 'pass')"
  _row roofline_aggregate_ok       "$F" merge W1 lambda "" pass "is above the single-stream roofline"

  # ---- PP-22 / PP-25: the join and the client ------------------------------
  local PAIRED
  PAIRED='import copy
for b in r["bands"]:
    if b["concurrency"] != 4:
        continue
    b["comparator_status"]="MEASURED"
    b["baseline"]={"run_id": r["run_id"], "concurrency": 4,
                   "aggregate_tok_per_sec": 400.0, "decode_tok_per_sec": 100.0,
                   "prefill_tok_per_sec": 3000.0,
                   "join_key": copy.deepcopy(b["join_key"]),
                   "client": {"path": "/opt/pinned-binary", "sha256": r["provenance"]["client"]["sha256"],
                              "commit": r["provenance"]["client"]["commit"]}}
    b["ratios"]={"agg": {"point": 0.90, "lcb95": 1.02, "method": "replicate_t_lower", "n": 5},
                 "dec": {"point": 1.12, "lcb95": 1.05, "method": "paired_percentile_bootstrap", "n": 5},
                 "prefill": {"point": 1.07, "lcb95": 1.01, "method": "replicate_t_lower", "n": 5}}
'
  F="$(_mut joinok "$OK3" "$PAIRED")"
  _row join_ok                     "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut joinbad "$OK3" "$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["baseline"]["join_key"]["window_ms"]=30000
')"
  _row join_mismatch               "$F" merge W1 lambda "" fail "join_key differs from its baseline"
  F="$(_mut joinb1 "$OK3" "$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["join_key"]["n_batch"]=1
        b["baseline"]["join_key"]["n_batch"]=1
')"
  _row join_batch_one_refused      "$F" merge W1 lambda "" fail "switches the comparator"
  F="$(_mut clientok "$OK3" "$PAIRED")"
  _row client_ok                   "$F" merge W1 lambda "" pass "PASS ArmL1"
  F="$(_mut clientbad "$OK3" "$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["baseline"]["client"]["sha256"]="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
')"
  _row client_mismatch             "$F" merge W1 lambda "" fail "two clients is two experiments"

  # ---- PP-3 / PP-17: a ratio is representable only inside its own band -----
  F="$(_mut ratiobare "$OK3" 'for b in r["bands"]:
    b["agg_ratio"]=0.9
    b["decode_ratio"]=1.1')"
  _row ratio_bare                  "$F" merge W1 lambda "" fail "bare scalar ratio"
  F="$(_mut ratiopaired "$OK3" "$PAIRED")"
  _row ratio_paired                "$F" merge W1 lambda "" pass "PASS ArmC integrity"
  F="$(_mut claimbandless "$OK3" "$PAIRED"'
r["ratios"]={"agg": {"point": 0.9, "lcb95": 0.85, "method": "replicate_t_lower", "n": 5}}
')"
  _row claim_bandless              "$F" merge W1 lambda "" fail "must sit inside a band"
  F="$(_mut claimnamed "$OK3" "$PAIRED")"
  _row claim_named                 "$F" merge W1 lambda "" pass "PASS ArmC integrity"
  F="$(_mut ratioorphan "$OK3" "$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["baseline"]=None
')"
  _row ratio_without_baseline      "$F" merge W1 lambda "" fail "The comparator band the ratio was taken against"

  # ---- the matrix variants the phase, ratchet and expiry rows need ---------
  local MX_SEEDED MX_AMERGE MX_ARMED MX_W1FIXED MX_C1SILENT
  local MX_MERGED MX_NOANCHOR MX_BOTH MX_MERGEDNULL MX_NEITHER MX_BADDAYS
  local A_W1_OLD A_W1_NEW A_PHASE_OLD A_PHASE_NEW ARMED_OLD ARMED_NEW
  local A_W1_MEASURED_NOBANDS
  A_W1_OLD='    W1:
      status: UNMEASURED
      owner: perf-gate
      reason: >-
        Blocked on PP-LLAMA-001 row 18 (the reference measurement). The
        745fa8588 lambda run is SPENT and its ratios are withdrawn.
      expires_after: {anchor: PP-LLAMA-001-row-18, days: 0}'
  A_W1_NEW='    W1:
      status: MEASURED
      receipt: evidence/perf-gate-001-w1-lambda/receipt.r1.json
      commit: 745fa8588
      n: 5
      interleaved: true
      bands:
        c1: {agg: 100.0, dec: 110.0, prefill: 900.0}'
  # #2830. A baseline can flip MEASURED without ever seeding a band the
  # configured metrics name -- the arm then has no seeded cell to ratchet and,
  # paired with a c=1-only receipt, no c>1 band to report scaling on either.
  A_W1_MEASURED_NOBANDS='    W1:
      status: MEASURED
      receipt: evidence/perf-gate-001-w1-lambda/receipt.r1.json
      commit: 745fa8588
      n: 5
      interleaved: true'
  A_PHASE_OLD='    name: self-regression
    phase: release'
  A_PHASE_NEW='    name: self-regression
    phase: merge'
  ARMED_OLD='    armed_by: {}'
  ARMED_NEW='    armed_by:
      lambda:
        W1:
          c1: {dec_ratio: {receipt: evidence/perf-gate-001-w1-lambda/receipt.r1.json, commit: 745fa8588},
               prefill_ratio: {receipt: evidence/perf-gate-001-w1-lambda/receipt.r1.json, commit: 745fa8588}}
          c4: {agg_ratio: {receipt: evidence/perf-gate-001-w1-lambda/receipt.r1.json, commit: 745fa8588}}'
  MX_SEEDED="$(_mx seeded "$A_W1_OLD"$'\x1f'"$A_W1_NEW")"
  MX_C1SILENT="$(_mx c1silent "$A_W1_OLD"$'\x1f'"$A_W1_MEASURED_NOBANDS")"
  MX_AMERGE="$(_mx amerge "$A_W1_OLD"$'\x1f'"$A_W1_NEW"$'\x1e'"$A_PHASE_OLD"$'\x1f'"$A_PHASE_NEW")"
  MX_ARMED="$(_mx armed "$ARMED_OLD"$'\x1f'"$ARMED_NEW")"
  MX_W1FIXED="$(_mx w1fixed "expires_after: {anchor: PP-LLAMA-001-row-18, days: 0}"$'\x1f'"expires: '2026-09-25'")"
  MX_MERGED="$(_mx merged "merged_on: null"$'\x1f'"merged_on: '2026-09-01'")"
  MX_NOANCHOR="$(_mx noanchor "{anchor: PERF-001, days: 30}"$'\x1f'"{anchor: PERF-999, days: 30}")"
  MX_BOTH="$(_mx both "expires_after: {anchor: PERF-001, days: 30}"$'\x1f'"expires_after: {anchor: PERF-001, days: 30}
      expires: '2027-01-01'")"
  MX_MERGEDNULL="$(_mx mergednull "status: in_progress
    # Set by the PR"$'\x1f'"status: merged
    # Set by the PR")"
  MX_NEITHER="$(_mx neither "expires_after: {anchor: PERF-001, days: 30}"$'\x1f'"absent_on_purpose: true")"
  MX_BADDAYS="$(_mx baddays "{anchor: PERF-001, days: 30}"$'\x1f'"{anchor: PERF-001, days: '30'}")"

  # ---- §7.2 / PP-1: release phase over the pre-v3 receipt of an UNMEASURED cell
  # The committed matrix carries lambda/W1 UNMEASURED and the only receipt for
  # it is the pre-v3 one. Both release-only arms REPORT (the release rests on
  # the cell's UNMEASURED status, which ArmExpiry judges), and the verdict is
  # PASS. The two must-not-fire rows are the narrowness: the same receipt cited
  # for a MEASURED cell is refused, and an unsigned v3 receipt is refused even
  # for an UNMEASURED cell.
  F="$(_mut histrel "$OK2" 'pass')"
  _row historical_unmeasured_release_reports "$F" release W1 lambda "" pass "REPORT ArmC-sig historical receipt" "" raw
  F="$(_mut histreld "$OK2" 'pass')"
  _row historical_unmeasured_armd_reports    "$F" release W1 lambda "" pass "REPORT ArmD historical receipt" "" raw
  F="$(_mut histrelm "$OK2" 'pass')"
  _row historical_measured_release_fails     "$F" release W1 lambda "$MX_SEEDED" fail "FAIL ArmC-sig UNSIGNED" "" raw
  F="$(_mut v3raw "$OK3" 'pass')"
  _row v3_unsigned_unmeasured_release_fails  "$F" release W1 lambda "" fail "FAIL ArmC-sig UNSIGNED" "" raw

  # ---- PP-6: an arm runs at the phase perf-matrix.yaml declares for it ------
  local L3_BELOW
  L3_BELOW="$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["ratios"]["agg"]["lcb95"]=0.5
'
  F="$(_mut l3below "$OK3" "$L3_BELOW")"
  _row phase_guard_b_merge         "$F" merge   W1 lambda "$MX_ARMED" pass "REPORT ArmL3 declares phase=release"
  F="$(_mut l3below2 "$OK3" "$L3_BELOW")"
  _row phase_guard_b_release       "$F" release W1 lambda "$MX_ARMED" fail "FAIL ArmL3 c=4 agg_ratio"
  # THE 2026-08-25 SHAPE: decode soaring while aggregate collapses. B2 alone
  # would have scored it a comfortable PASS; the gated set at c>1 is the
  # aggregate, so it cannot.
  F="$(_mut serialshape "$OK3" "$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["ratios"]["agg"]["lcb95"]=0.097
        b["ratios"]["agg"]["point"]=0.097
        b["ratios"]["dec"]["lcb95"]=1.554
        b["ratios"]["dec"]["point"]=1.554
')"
  _row serialization_shape_rejected "$F" release W1 lambda "$MX_ARMED" fail "FAIL ArmL3 c=4 agg_ratio"
  F="$(_mut l3at "$OK3" "$PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 4:
        b["ratios"]["agg"]["lcb95"]=1.0
')"
  _row l3_agg_lcb_at_delta_c4      "$F" release W1 lambda "$MX_ARMED" pass "PASS ArmL3 c=4 agg_ratio"
  F="$(_mut l3below3 "$OK3" "$L3_BELOW")"
  _row l3_agg_lcb_below_delta_c4   "$F" release W1 lambda "$MX_ARMED" fail "FAIL ArmL3 c=4 agg_ratio lcb95=0.5000"
  # At c=1 the aggregate IS the decode, so the gated set is the per-request
  # pair. A gate that used one set at every band would test the wrong metric at
  # one end of the ladder.
  local C1_PAIRED
  C1_PAIRED='import copy
for b in r["bands"]:
    if b["concurrency"] != 1:
        continue
    b["comparator_status"]="MEASURED"
    b["baseline"]={"run_id": r["run_id"], "concurrency": 1,
                   "aggregate_tok_per_sec": 111.0, "decode_tok_per_sec": 120.0,
                   "prefill_tok_per_sec": 800.0,
                   "join_key": copy.deepcopy(b["join_key"]),
                   "client": {"path": "/opt/pinned-binary", "sha256": r["provenance"]["client"]["sha256"],
                              "commit": r["provenance"]["client"]["commit"]}}
    b["ratios"]={"agg": {"point": 0.90, "lcb95": 0.88, "method": "replicate_t_lower", "n": 5},
                 "dec": {"point": 0.50, "lcb95": 0.45, "method": "paired_percentile_bootstrap", "n": 5},
                 "prefill": {"point": 1.12, "lcb95": 1.05, "method": "replicate_t_lower", "n": 5}}
'
  F="$(_mut l3c1 "$OK3" "$C1_PAIRED")"
  _row l3_dec_gated_at_c1          "$F" release W1 lambda "$MX_ARMED" fail "FAIL ArmL3 c=1 dec_ratio"
  # ... and the aggregate at c=1 is REPORTED, never gated: the same number that
  # FAILS at c=4 is a printed line at c=1, because at c=1 the aggregate IS the
  # decode and gating both would gate one quantity twice.
  F="$(_mut l3c1b "$OK3" "$C1_PAIRED"'
for b in r["bands"]:
    if b["concurrency"] == 1:
        b["ratios"]["dec"]["lcb95"]=1.05
        b["ratios"]["agg"]["lcb95"]=0.5
')"
  _row l3_agg_reported_at_c1       "$F" release W1 lambda "$MX_ARMED" pass "REPORT ArmL3 c=1 agg_ratio point="
  # AN UNARMED CELL REPORTS. `armed_by` is empty in the committed matrix, so the
  # same failing ratio is a printed line and not a verdict until a measurement
  # arms it -- a gate arms when a measurement arms it, never on a date.
  F="$(_mut l3unarmed "$OK3" "$L3_BELOW")"
  _row l3_unarmed_is_reporting     "$F" release W1 lambda "" pass "is not in arms.L3.armed_by"

  # ---- PP-31: self-regression ----------------------------------------------
  local REPS
  REPS='import copy
base=[b for b in r["bands"] if b["concurrency"]==1][0]
rest=[b for b in r["bands"] if b["concurrency"]!=1]
reps=[]
for k in range(5):
    b=copy.deepcopy(base)
    b["replicate"]=k+1
    b["aggregate_tok_per_sec"]=AGG+k*0.5
    b["decode_tok_per_sec"]=DEC+k*0.5
    b["prefill_tok_per_sec"]=PRE+k*0.5
    reps.append(b)
r["bands"]=reps+rest
'
  F="$(_mut regfail "$OK3" 'AGG,DEC,PRE=80.0,90.0,700.0
'"$REPS")"
  _row self_regress_fail           "$F" release W1 lambda "$MX_SEEDED" fail "FAIL ArmA c=1 agg lcb95"
  # RAISING agg(1) MUST NOT FAIL ANYTHING. The old arm ratcheted
  # scaling_efficiency, which FALLS when agg(1) rises, so this exact improvement
  # turned the gate red at c=4, c=8 and c=16.
  F="$(_mut regok "$OK3" 'AGG,DEC,PRE=120.0,115.0,950.0
'"$REPS")"
  _row agg1_improve_ok             "$F" release W1 lambda "$MX_SEEDED" pass "PASS ArmA c=1 agg lcb95"
  # n < n_min IS NOT A PASS AND NOT A FAIL. Four replicates of a regression
  # report the point estimate and decline a verdict; a ratchet that fired on
  # n=1 would fire on noise, and one that stayed silent forever would never fire
  # at all.
  local REPS3
  REPS3="${REPS/range(5)/range(3)}"
  F="$(_mut regsmall "$OK3" 'AGG,DEC,PRE=80.0,90.0,700.0
'"$REPS3")"
  _row self_regress_n_below_n_min  "$F" release W1 lambda "$MX_SEEDED" pass "reporting only"
  F="$(_mut amerge "$OK3" 'AGG,DEC,PRE=80.0,90.0,700.0
'"$REPS")"
  _row phase_guard_a_merge         "$F" merge W1 lambda "$MX_AMERGE" fail "FAIL ArmA c=1 agg lcb95"

  # #2830. A receipt whose only band is c=1 has no c>1 band to report scaling
  # from; paired with a baseline that flipped MEASURED without ever seeding a
  # band (MX_C1SILENT), the arm used to walk every branch, print nothing at
  # all, and exit 0 -- so the gate's own VERDICT line said PASS on a run this
  # arm never actually measured. It must now say so and the run must not read
  # as PASS. (overhead_share is popped too: it is the c=1 band's own
  # self-descriptive figure, a v3 producer writes it even on a c=1-only
  # receipt, and it would otherwise emit a line and mask the exact silence
  # this row exists to catch.)
  F="$(_mut c1only "$OK3" 'r["bands"]=[b for b in r["bands"] if b["concurrency"] == 1]
for b in r["bands"]:
    b.pop("overhead_share", None)
    b.pop("scaling_efficiency", None)
r["ladder"]["slots_admitted"]={"apr": 1, "llama": 1}
r["ladder"]["derived"]=[1]')"
  _row arm_a_c1_only_not_pass      "$F" release W1 lambda "$MX_C1SILENT" fail "REPORT ArmA scaling: c=1 only, no scaling measured"
  # The must-not-fire twin: a receipt with a real c>1 band alongside c=1 still
  # gets the ordinary scaling REPORT line, and the collapse text above never
  # appears.
  F="$(_mut c1and8 "$OK3" 'r["bands"]=[b for b in r["bands"] if b["concurrency"] in (1, 8)]
r["ladder"]["declared"]=[1, 8]
r["ladder"]["slots_admitted"]={"apr": 8, "llama": 8}
r["ladder"]["derived"]=[1, 8]')"
  _row arm_a_multi_band_ok         "$F" release W1 lambda "" pass "REPORT ArmA c=8 scaling_efficiency="

  # ---- PP-1: the expected cell set -----------------------------------------
  F="$(_mut cellsfull "$OK3" 'pass')"
  _row cells_complete_at_release   "$F" release W2 lambda "" pass "all bands"
  F="$(_mut cellsshort "$OK3" 'r["bands"]=[b for b in r["bands"] if b["concurrency"] in (1,4)]
r["ladder"]["derived"]=[1,4,8,16]')"
  _row cellset_missing             "$F" release W2 lambda "" fail "missing bands [8, 16]"
  # AN NA CELL PASSES WITH NO RECEIPT BANDS AT ALL, loudly, naming its decider.
  # `mini` has no Metal inference path, so its cells are excluded from the
  # denominator by decision (#2841) rather than by omission.
  F="$(_mut minina "$OK3" 'r["provenance"]["host"]="mini"
r["provenance"]["compute_class"]="cpu"
r["provenance"]["accelerator"]="m4"
r["provenance"]["feature_set"]="cpu".split()
r["provenance"]["server_config"]["compute_class"]="cpu"
r["bands"]=[]
r["kv"]={"bytes_used":50,"bytes_reserved":100,"admission_rejected":0,"preempted_swap":0}')"
  _row cellset_na_ok               "$F" release W1 mini   "" pass "is NA"
  F="$(_mut noc1 "$OK3" 'r["bands"]=[b for b in r["bands"] if b["concurrency"]!=1]')"
  _row band_c1_absent              "$F" release W2 lambda "" fail "missing bands [1]"
  # PP-24 again, at the cell level: a band the servers never admitted is not a
  # missing cell, and the derived ladder is the only thing that may excuse it.
  F="$(_mut derivedshort "$OK3" 'r["bands"]=[b for b in r["bands"] if b["concurrency"] in (1,4)]
r["ladder"]["derived"]=[1,4]
r["ladder"]["slots_admitted"]={"apr":4,"llama":4}')"
  _row cellset_derived_ladder_ok   "$F" release W2 lambda "" pass "are outside ladder.derived"

  # ---- §4.9.1 / PP-21 / PP-18: signature, staleness, ancestry --------------
  # Registered mutation: "Staleness arm | verdict job | receipt one commit stale
  # | fresh receipt green". Both polarities below, plus the four ways a receipt
  # can be about SOMETHING ELSE rather than merely old. Each row asserts its own
  # FAILURE CODE, not just the polarity: a stale receipt reported as WRONG-HOST
  # sends the reader to the wrong fix.
  _sigmk() { # dest-name, src-path, receipt-commit, host, sign(yes|no), subject-commit
    if [ "$LIST_ONLY" = 1 ]; then return 0; fi
    cp "$2" "$tmp/$1.json"
    python3 -c 'import json,sys
with open(sys.argv[1]) as fh:
    r = json.load(fh)
r["commit"] = sys.argv[2]
r["provenance"]["host"] = sys.argv[3]
if isinstance(r["provenance"].get("subject"), dict):
    r["provenance"]["subject"]["commit"] = sys.argv[4]
if isinstance(r["provenance"].get("client"), dict):
    r["provenance"]["client"]["commit"] = sys.argv[4]
with open(sys.argv[1], "w") as fh:
    json.dump(r, fh)' "$tmp/$1.json" "$3" "$4" "$6"
    if [ "$5" = yes ]; then
      python3 "$ROOT/scripts/lib/receipt_sig.py" --sign --in "$tmp/$1.json" \
        --out "$tmp/$1.json" --key-id "$4-selftest" --keyring "$sigkr" \
        --signed-at 2026-08-29T00:00:00Z >/dev/null
    fi
  }
  _sigrow() { # name, fixture-name, host, commit-under-test, keyring, expect(pass|CODE), needle
    _reg "$1" || return 0
    local out rc=0
    out="$( ( export APR_PERF_RECEIPT_KEYRING="$5"; export PERF_GATE_GIT_DIR="$sigrepo"; \
              run_gate "$3" release W2 "$tmp/$2.json" "$4" ) 2>&1 )" || rc=$?
    if [ "$6" = pass ]; then
      local got=pass
      [ "$rc" = 0 ] || got=fail
      _verdict "$1" pass "$got" "$out" "${7:-}"
      return 0
    fi
    # Herestring-free glob match: `producer | grep -q X` returns 141 under
    # pipefail though grep MATCHED, because grep exits early and printf takes
    # SIGPIPE. There is no pipe here at all.
    case "$out" in
      *"FAIL ArmC-sig $6"*)
        if [ "$rc" = 0 ]; then
          printf '  BROKE %-34s named %s and still exited 0\n' "$1" "$6"
          fail=$((fail + 1))
        else
          printf '  ok    %-34s expect=%s\n' "$1" "$6"
          pass=$((pass + 1))
        fi
        ;;
      *)
        printf '  BROKE %-34s expected %s, gate said: %s\n' "$1" "$6" "$out"
        fail=$((fail + 1))
        ;;
    esac
  }
  _sigmk sig_fresh   "$OK3" "$sigc1" lambda yes "$sigc0"
  _sigrow sig_ok                    sig_fresh lambda "$sigc1" "$sigkr" pass "PASS ArmC integrity"
  # ... and it also covers an ANCESTOR of what it measured: `contains`, not `==`.
  _sigrow sig_covers_an_ancestor    sig_fresh lambda "$sigc0" "$sigkr" pass
  # PP-18: the SERVED binary's commit must also be contained. `receipt.commit`
  # is a label; the subject binary can have been built from anywhere.
  _sigrow ancestor_ok               sig_fresh lambda "$sigc1" "$sigkr" pass "is an ancestor of"
  _sigmk sig_stray "$OK3" "$sigc1" lambda yes 9999999999999999999999999999999999999999
  _sigrow ancestor_fail             sig_stray lambda "$sigc1" "$sigkr" SUBJECT-NOT-ANCESTOR

  # RED SIDE 1 -- FRESHNESS. A receipt one commit stale. Signature perfectly
  # valid; the evidence is about older code. Remedy: re-measure.
  _sigmk sig_stale "$OK3" "$sigc0" lambda yes "$sigc0"
  _sigrow sig_one_commit_stale      sig_stale lambda "$sigc1" "$sigkr" STALE

  # RED SIDE 2 -- IDENTITY. Different failures, different remedies. None of
  # these is fixed by re-measuring.
  _sigmk sig_unsigned "$OK3" "$sigc1" lambda no "$sigc0"
  _sigrow sig_missing               sig_unsigned lambda "$sigc1" "$sigkr" UNSIGNED
  _sigrow sig_no_keyring_is_red     sig_fresh lambda "$sigc1" "" NO-KEYRING
  # An arm handed no input must FAIL, not stand down. run_gate is reachable
  # without main()'s usage check -- this is the row that keeps that honest.
  _sigrow sig_no_commit_under_test  sig_fresh lambda "" "$sigkr" NO-COMMIT-UNDER-TEST
  _sigmk sig_other_host "$OK3" "$sigc1" gx10 yes "$sigc0"
  _sigrow sig_receipt_from_other_host sig_other_host lambda "$sigc1" "$sigkr" WRONG-HOST

  # RED SIDE 3 -- FORGERY. Sign a real receipt, then edit one number in the
  # body. This is the "evidence/ is a file anyone can write" case made concrete.
  if [ "$LIST_ONLY" != 1 ]; then
    cp "$tmp/sig_fresh.json" "$tmp/sig_forged.json"
    python3 -c 'import json,sys
with open(sys.argv[1]) as fh:
    r = json.load(fh)
r["bands"][0]["aggregate_tok_per_sec"] = 9999.0
with open(sys.argv[1], "w") as fh:
    json.dump(r, fh)' "$tmp/sig_forged.json"
  fi
  _sigrow sig_body_edited_after_signing sig_forged lambda "$sigc1" "$sigkr" FORGED

  # DISCRIMINATION. The arm is release-scoped, so an unsigned receipt at MERGE
  # phase must stay green -- otherwise every PR reds on a rule no PR can satisfy.
  if _reg sig_unsigned_ok_at_merge; then
    if run_gate lambda merge W2 "$tmp/sig_unsigned.json" >/dev/null 2>&1; then
      printf '  ok    %-34s expect=pass\n' sig_unsigned_ok_at_merge
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s merge phase reddened on a release-only arm\n' sig_unsigned_ok_at_merge
      fail=$((fail + 1))
    fi
  fi

  # ---- PP-8: the client concurrency each side actually drove ---------------
  # scripts/lib/parity_block.py carries it per band and
  # bench_receipt.validate_parity refuses a pair whose two sides were driven at
  # different concurrencies -- the shape where a "c=4 ratio" is a c=4 subject
  # over a c=1 comparator, which is a different experiment wearing one label.
  _concrow() { # name, comparator-concurrency, expect(pass|fail)
    _reg "$1" || return 0
    local out rc=0 got
    out="$(python3 - "$ROOT" "$2" <<'PY_CONC'
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "scripts", "lib"))
import bench_receipt

def prov():
    return {"binary_path": "/opt/x", "binary_sha256": "0" * 64,
            "resolution": "scripts/apr_bin.sh", "compute_class": "cpu",
            "host": "lambda", "accelerator": "cpu",
            "model": "qwen2.5-coder-7b-instruct", "quantization": "q4_k_m"}

def side(conc, install=None, build=None):
    side = {"provenance": prov(), "decode_tok_per_sec": [100.0, 101.0],
            "aggregate_tok_per_sec": [100.0, 101.0], "client_concurrency": conc}
    if install:
        side["install_source"] = install
    if build:
        side["build_commit"] = build
    return side

band = {"concurrency": 4, "subject": side(4, install="crates.io"),
        "comparator": side(int(sys.argv[2]), build="39173bcac"), "verdict": "PASS"}
lane = {"lane": "cpu", "subject": side(4, install="crates.io"),
        "comparator": side(int(sys.argv[2]), build="39173bcac"),
        "ratio_decode": 1.0, "verdict": "PASS", "floor": 1.0, "ceiling": 1.5,
        "declared_bands": [4], "bands": [band]}
errors = bench_receipt.validate_parity(
    {"instrument": "apr test llm bench", "protocol_ref": "scripts/llama_pin.toml#protocol.http",
     "model": "q.gguf", "lanes": [lane]})
for e in errors:
    print(e)
sys.exit(1 if errors else 0)
PY_CONC
)" || rc=1
    got=pass
    [ "$rc" = 0 ] || got=fail
    _verdict "$1" "$3" "$got" "$out" ""
  }
  _concrow client_conc_ok       4 pass
  _concrow client_conc_mismatch 1 fail

  # ---- the expiry clock (PERF-056, #2777) ----------------------------------
  # `PERF_GATE_TODAY` was added "so the selftest can prove BOTH sides of the
  # expiry boundary without waiting for a date to arrive", and then NO ROW EVER
  # USED IT: the clock deciding whether every UNMEASURED cell in the matrix may
  # still REPORT was itself untested.
  #
  # The W1 cells are now anchored to §12 rows rather than to a calendar date, so
  # the two fixed-date rows run against an `_mx` variant that RE-INSERTS a date.
  # The must-fire is kept alive rather than deleted with the date it tested.
  #
  #  matrix / workload                       | today      | must | names
  #  ----------------------------------------|------------|------|-------------
  #  fixed date re-inserted, W1              | 2026-09-25 | pass | the date
  #  fixed date re-inserted, W1              | 2026-09-26 | FAIL | EXPIRED
  #  committed, W1 (anchored to row 18)      | 2099-01-01 | pass | the ANCHOR
  #  committed, W2 (event-dated)             | 2099-01-01 | pass | the ANCHOR
  #  committed, W2 (event-dated)             | 2026-08-29 | pass | why unarmed
  #  anchor merged 2026-09-01, +30d          | 2026-10-01 | pass | 2026-10-01
  #  anchor merged 2026-09-01, +30d          | 2026-10-02 | FAIL | EXPIRED
  #  anchor not declared in expiry_anchors   | any        | FAIL | undeclared
  #  cell declares BOTH clocks               | any        | FAIL | two clocks
  #  anchor status: merged, merged_on null   | any        | FAIL | null
  #  cell declares NEITHER clock             | any        | FAIL | never expires
  #  days is a string, not an integer        | any        | FAIL | integer
  local EXP
  EXP="$(_mut expiry "$OK3" 'pass')"
  _row expiry_w1_fixed_date_still_reports  "$EXP" merge W1 lambda "$MX_W1FIXED" pass "UNMEASURED until 2026-09-25" 2026-09-25
  _row expiry_w1_fixed_date_expires        "$EXP" merge W1 lambda "$MX_W1FIXED" fail "EXPIRED 2026-09-25" 2026-09-26
  _row expiry_w1_is_event_dated            "$EXP" merge W1 lambda ""            pass "PP-LLAMA-001-row-18 merge + 0 days" 2099-01-01
  _row expiry_w2_is_event_dated            "$EXP" merge W2 lambda ""            pass "PERF-001 merge + 30 days" 2099-01-01
  _row expiry_w2_says_why_it_is_unarmed    "$EXP" merge W2 lambda ""            pass "has NOT merged" 2026-08-29
  _row expiry_anchor_merge_arms_the_clock  "$EXP" merge W2 lambda "$MX_MERGED"  pass "UNMEASURED until 2026-10-01" 2026-10-01
  _row expiry_anchor_merge_then_expires    "$EXP" merge W2 lambda "$MX_MERGED"  fail "EXPIRED 2026-10-01" 2026-10-02
  _row expiry_undeclared_anchor_is_fatal   "$EXP" merge W2 lambda "$MX_NOANCHOR" fail "not declared under" 2026-08-29
  _row expiry_two_clocks_is_fatal          "$EXP" merge W2 lambda "$MX_BOTH"    fail "two clocks is no clock" 2026-08-29
  _row expiry_merged_without_date_is_fatal "$EXP" merge W2 lambda "$MX_MERGEDNULL" fail "cannot start from null" 2026-08-29
  _row expiry_no_deadline_at_all_is_fatal  "$EXP" merge W2 lambda "$MX_NEITHER" fail "never expires" 2026-08-29
  _row expiry_days_must_be_an_integer      "$EXP" merge W2 lambda "$MX_BADDAYS" fail "non-negative integer" 2026-08-29

  # main() must refuse `--phase release` with no `--commit` as a USAGE error
  # (exit 2), in a subshell because `die` exits. A gate that silently accepts a
  # release invocation missing the arm that makes it a gate is the whole defect.
  if _reg main_requires_commit_at_release; then
    if ( main --host lambda --phase release --workload W1 --receipt "$tmp/sig_fresh.json" ) >/dev/null 2>&1; then
      printf '  BROKE %-34s release without --commit was accepted\n' main_requires_commit_at_release
      fail=$((fail + 1))
    else
      printf '  ok    %-34s expect=usage-error\n' main_requires_commit_at_release
      pass=$((pass + 1))
    fi
  fi

  # The fine-grained crypto/containment table and the host-side signer both
  # carry their own case tables. Running them from HERE is what wires them:
  # ci.yml already invokes `perf_gate.sh --selftest`, so neither needs a new
  # workflow line, and a guard nothing invokes is this epic's most common
  # finding.
  if _reg receipt_sig_case_table; then
    if python3 "$ROOT/scripts/lib/receipt_sig.py" --selftest >/dev/null 2>&1; then
      printf '  ok    %-34s expect=pass\n' receipt_sig_case_table
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s scripts/lib/receipt_sig.py --selftest failed\n' receipt_sig_case_table
      fail=$((fail + 1))
    fi
  fi
  if _reg receipt_signer_case_table; then
    if bash "$ROOT/scripts/perf_receipt_sign.sh" --selftest >/dev/null 2>&1; then
      printf '  ok    %-34s expect=pass\n' receipt_signer_case_table
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s scripts/perf_receipt_sign.sh --selftest failed\n' receipt_signer_case_table
      fail=$((fail + 1))
    fi
  fi
  # PP-26's probe-side case table (scripts/perf041_batched_parity_probe.py
  # --selftest): witness_constant_token_m3 / witness_identical_128_ok and the
  # vacuity rows ride into ci / gate through this row, the same way the two
  # signature tables above do.
  if _reg perf041_case_table; then
    if python3 "$ROOT/scripts/perf041_batched_parity_probe.py" --selftest >/dev/null 2>&1; then
      printf '  ok    %-34s expect=pass\n' perf041_case_table
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s scripts/perf041_batched_parity_probe.py --selftest failed\n' perf041_case_table
      fail=$((fail + 1))
    fi
  fi

  if [ "$LIST_ONLY" = 1 ]; then
    rm -rf "${tmp:?refusing to rm an empty path}"
    return 0
  fi
  printf '  %d passed, %d broken\n' "$pass" "$fail"
  rm -rf "${tmp:?refusing to rm an empty path}"
  [ "$fail" = 0 ]
}

main() {
  local host="" phase="" workload="" receipt="" commit=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --selftest) selftest; return $? ;;
      # THE JOIN SURFACE (PP-29). scripts/spec_conformance.sh reads this list and
      # asserts that every ARMED row of the master's §6 table names two cases
      # that exist HERE, by name. Parsing --selftest stdout would work too and
      # would cost a full case-table run; a list mode registers the names without
      # executing anything, so the join is cheap enough to be a merge check.
      --list-selftests)
        SELFTEST_LIST_ONLY=1 LIST_ONLY=1 selftest >/dev/null 2>&1 || true
        printf '%s\n' "${SELFTEST_NAMES[@]}"
        return 0 ;;
      --host) host="$2"; shift 2 ;;
      --phase) phase="$2"; shift 2 ;;
      --workload) workload="$2"; shift 2 ;;
      --receipt) receipt="$2"; shift 2 ;;
      --commit) commit="$2"; shift 2 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  [ -n "$host" ] && [ -n "$phase" ] && [ -n "$workload" ] && [ -n "$receipt" ] \
    || die "usage: perf_gate.sh --host H --phase {merge|release} --workload {W1|W2} --receipt PATH [--commit SHA]"
  case "$phase" in merge|release) ;; *) die "phase must be merge or release" ;; esac
  # A missing --commit at release is a USAGE error, never a skipped arm. An arm
  # that quietly stands down when its input is absent is the cannot-fail shape
  # this epic exists to remove.
  if [ "$phase" = release ] && [ -z "$commit" ]; then
    die "--commit <commit-under-test> is required at --phase release: the staleness arm has nothing to compare without it, and it is the arm that makes this a gate"
  fi
  run_gate "$host" "$phase" "$workload" "$receipt" "$commit"
}
main "$@"
