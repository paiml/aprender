#!/usr/bin/env bash
# perf_gate.sh — APR-PERF-GATE-001 v2.2 §4, the release performance gate.
#
#   scripts/perf_gate.sh --host <name> --phase {merge|release} \
#                        --workload {W1|W2} --receipt <path> \
#                        [--commit <commit-under-test>]   # REQUIRED at release
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

# ---------------------------------------------------------------- arms -----
# Every arm returns 0 (pass) or 1 (fail) and prints one PASS/FAIL line. The
# verdict is the MIN over arms (§4.8: exactly one verdict function, H11).

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

arm_c_tokenization() {
  # §4.4.6 / I-11 / I-13 — the FATAL tokenization-mismatch rule. Specified in
  # v2.1, and until this function existed, implemented nowhere.
  #
  # WHAT WAS HERE. `arm_c_integrity` tested that `tokenization.method` was a
  # non-empty STRING, and nothing else in the file read the block at all. The
  # gate's own OK fixture carried `"method":"hf-tokenizers"` — which is not one
  # of the two values §4.4.6 declares — so the single input proving the rule
  # worked was itself illegal input. `counts_special_tokens` and
  # `counts_prompt_echo` were read by ZERO lines. And there was no
  # comparator-vs-measured comparison anywhere in the gate, which is the whole
  # of I-11: "any comparator ratio refuses to execute on join-key mismatch —
  # including a `tokenization` block mismatch".
  #
  # WHY THE BLOCK §4.4.6 SPECIFIES IS NOT ENOUGH — MEASURED, TWICE.
  # On 2026-08-29 apr 0.64.0 (745fa8588, CPU) and llama-server 7746 (39173bcac,
  # CUDA, 29/29 layers) were driven over the same W1 corpus against the same
  # qwen2.5-coder-7b-instruct-q4_k_m.gguf. Both harnesses count tokens from the
  # server's own `usage` object, so both lanes would have declared, HONESTLY:
  #
  #     method: server_usage      counts_prompt_echo: false
  #
  # The two blocks would have been IDENTICAL. The prompts the two servers
  # actually prefilled were not: apr 513 tokens, llama 534, every run, min ==
  # median == max on both sides. Constant delta −21, ratio 0.9607 — 4.09% more
  # prefill work on the comparator, charged to the ratio as though it were
  # throughput. A block that is identical across a 4.09% divergence is blind to
  # the case it exists to catch.
  #
  # IT IS NOT THE VOCABULARY. On raw text all three tokenizations agree
  # exactly: apr 505, llama 505, client 505. It is the CHAT TEMPLATE. llama
  # applies the jinja template embedded in the GGUF, which injects Qwen's
  # default system message; apr applies a hardcoded ChatML builder
  # (crates/aprender-serve/src/api/realize_handlers.rs, `format_chat_messages`)
  # with no system message and a constant 8-token wrapper. Reconstructing each
  # side's wrapper with the canonical tokenizer reproduces both numbers exactly.
  # And the divergence is not correctable: short prompt 25 vs 43 (−18), W1 513
  # vs 534 (−21), special-token text 76 vs 57 (+19, ratio 1.333). It CHANGES
  # SIGN with the input, so there is no constant offset and no constant ratio.
  #
  # EVERY FIELD §4.4.6 NAMES IS A DECLARATION. A declaration cannot be
  # wrong-and-detected — it can only be false, and nothing in this gate can
  # check the honesty of a boolean a producer wrote about itself. So this arm
  # compares two fields that are OUTCOMES of the measurement rather than
  # assertions about it:
  #
  #   chat_template_sha256          the RESOLVED wrapper each lane applied
  #   corpus_prefill_tokens_median  the ids each lane ACTUALLY PREFILLED
  #
  # The second is the only field in the block that a mis-declaration cannot
  # survive, and it is the one that reddens the 2026-08-29 pair. §4.3.1 already
  # asks for it in words — "the receipt must declare which side of that
  # boundary its count was taken on" — and no field existed to carry the answer.
  #
  # EQUALITY IS EXACT, and that is not a threshold. Two lanes that applied the
  # same template and the same tokenizer to the same corpus produce the same
  # count, not a nearby one. Any tolerance here would be an invented continuous
  # threshold, which perf-matrix.yaml's GROUNDING RULE forbids outright; exact
  # equality is the one rule that is not invented. It also has a consequence
  # worth stating: under `server_usage` the two numbers come from two servers'
  # opinions and will not agree, so this rule is what makes §4.4.6's canonical
  # `client_tokenizer` the only method a real receipt can pass with.
  local receipt="$1"
  python3 - "$receipt" <<'PY_TOK'
import json, re, sys

r = json.load(open(sys.argv[1]))

# §4.4.6: two legal values, NO DEFAULT (I-13). The absence of the field is
# ALSO caught by arm_c_integrity, which names the owning ticket; that overlap
# is deliberate defence in depth and the reason the illegal-VALUE row below is
# the one that belongs to this arm alone.
LEGAL_METHODS = ("server_usage", "client_tokenizer")
CANONICAL = "client_tokenizer"
# Required in every receipt whatever the method. `tokenizer_sha256` is NOT
# here: §4.4.6 requires it only under `client_tokenizer`, and that conditional
# is applied below.
REQUIRED = ("method", "counts_special_tokens", "counts_prompt_echo",
            "chat_template_sha256", "corpus_prefill_tokens_median")
BOOLEANS = ("counts_special_tokens", "counts_prompt_echo")
DIGESTS = ("tokenizer_sha256", "chat_template_sha256")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# Field NAMES as constants. The block's schema is spelled LITERALLY exactly
# once, in SCHEMA below, and the checks that run over BOTH lanes address fields
# through these — so scripts/check_perf_receipt_fields_have_producers.sh reads
# one declaration of this block rather than three copies of it drifting apart.
NAME_METHOD = "method"
NAME_TOKENIZER_SHA = "tokenizer_sha256"
NAME_PREFILL = "corpus_prefill_tokens_median"

subject_block = r.get("tokenization") or {}
comparator_block = r.get("comparator_tokenization")

# THE SCHEMA, SPELLED LITERALLY, ONCE.
# scripts/check_perf_receipt_fields_have_producers.sh builds its universe from
# LITERAL key reads, so a field addressed only through a variable leaves that
# guard's universe without saying so — the "guard universe from the wrong side"
# shape this epic has hit three times. Every field this arm requires is
# therefore read by name here, and the shared checks below run over these
# values rather than re-reading the block.
SCHEMA = (
    ("method",
     (r.get("tokenization") or {}).get("method")),
    ("tokenizer_sha256",
     (r.get("tokenization") or {}).get("tokenizer_sha256")),
    ("counts_special_tokens",
     (r.get("tokenization") or {}).get("counts_special_tokens")),
    ("counts_prompt_echo",
     (r.get("tokenization") or {}).get("counts_prompt_echo")),
    ("chat_template_sha256",
     (r.get("tokenization") or {}).get("chat_template_sha256")),
    ("corpus_prefill_tokens_median",
     (r.get("tokenization") or {}).get("corpus_prefill_tokens_median")),
)

fails = []
notes = []
agreed = 0


def shape(label, blk):
    """Every rule that reads ONE lane's block."""
    method = blk.get(NAME_METHOD)
    for name in REQUIRED:
        if blk.get(name) is None:
            fails.append("%s.%s absent — §4.4.6 requires it in every receipt "
                         "and gives it no default" % (label, name))
    if method is not None and method not in LEGAL_METHODS:
        fails.append(
            "%s.method=%r is not one of %s. §4.4.6 declares two legal values "
            "and no default; this gate tested only that the field was a "
            "non-empty string, and its own OK fixture carried \"hf-tokenizers\", "
            "so the input proving the rule worked was itself illegal"
            % (label, method, list(LEGAL_METHODS)))
    for name in BOOLEANS:
        value = blk.get(name)
        if value is not None and not isinstance(value, bool):
            fails.append("%s.%s=%r is not a boolean" % (label, name, value))
    if method == CANONICAL and blk.get(NAME_TOKENIZER_SHA) is None:
        fails.append(
            "%s.method=client_tokenizer with no tokenizer_sha256 — the "
            "canonical method names a tokenizer the receipt then does not "
            "identify. Measured 2026-08-29: TWO tokenizer.json files for this "
            "model are in circulation on the fleet (7,031,645 B and "
            "11,421,892 B, different digests, same vocabulary), and two of the "
            "four hosts carry only the second. Which one ran is not inferable "
            "from anything else in the receipt"
            % (label,))
    for name in DIGESTS:
        value = blk.get(name)
        if value is not None and not HEX64.match(str(value)):
            fails.append("%s.%s=%r is not a 64-character lowercase hex digest"
                         % (label, name, value))
    prefill = blk.get(NAME_PREFILL)
    if prefill is not None and (isinstance(prefill, bool)
                                or not isinstance(prefill, int)
                                or prefill <= 0):
        fails.append(
            "%s.corpus_prefill_tokens_median=%r is not a positive integer. It "
            "is a COUNT OF IDS — the median, over the whole workload corpus, of "
            "the id sequence the lane actually prefills after its own chat "
            "template. The raw-text count (505 on W1, and identical on both "
            "lanes) answers a different question" % (label, prefill))


shape("tokenization", dict(SCHEMA))
if isinstance(comparator_block, dict):
    shape("comparator_tokenization", comparator_block)

# IS THERE A RATIO TO REFUSE? §4.4.6 makes a block mismatch fatal to THE
# RATIO. A band whose comparator side is NOT_APPLICABLE or UNMEASURED computes
# none, so requiring a comparator block there would red a receipt that claims
# nothing. Every other band is gated by Arm B, and for those the comparator
# block is required: I-11's comparison cannot run against an absent operand,
# and an arm that stands down when its input is missing is the cannot-fail
# shape this gate exists to remove.
armed = [b.get("concurrency") for b in (r.get("bands") or [])
         if b.get("comparator_status") not in ("NOT_APPLICABLE", "UNMEASURED")]
# `None` and an int do not sort against each other, and a band with no
# `concurrency` is a receipt Arm A rejects one arm later — this one must not
# die with a traceback on the way there.
armed = sorted(armed, key=lambda c: (c is None, c or 0))
if comparator_block is None:
    if armed:
        fails.append(
            "comparator_tokenization absent while bands %s are gated by Arm B. "
            "I-11 requires the measured lane's tokenization block to be "
            "compared against its comparator baseline's, and a comparison with "
            "one operand is not a comparison" % (armed,))
    else:
        notes.append("REPORT ArmC-tok no band is comparator-gated "
                     "(comparator_status NOT_APPLICABLE/UNMEASURED on all of "
                     "them), so there is no ratio and the I-11 comparison is "
                     "not armed")
elif not isinstance(comparator_block, dict):
    # Nothing else in this gate reads the comparator block, so a scalar here
    # reaches this arm. Named rather than raised: an AttributeError is
    # fail-closed and diagnoses nothing.
    fails.append(
        "comparator_tokenization=%r is not an object — §4.4.6 declares a "
        "block, and a scalar in its place has no fields to compare"
        % (comparator_block,))
else:
    # THE UNION, not the subject's keys. A key present on one side only is a
    # mismatch, not a field to skip: two blocks that do not even have the same
    # shape have already failed to agree.
    keys = sorted(set(subject_block) | set(comparator_block)
                  | set(name for name, _ in SCHEMA))
    diffs = [(k, subject_block.get(k), comparator_block.get(k))
             for k in keys if subject_block.get(k) != comparator_block.get(k)]
    if diffs:
        fails.append(
            "tokenization block MISMATCH against the comparator baseline — "
            "§4.4.6: the ratio is REFUSED, not annotated (I-11)")
        for name, mine, theirs in diffs:
            fails.append("  %s: measured=%r comparator=%r" % (name, mine, theirs))
            if name == "corpus_prefill_tokens_median" and isinstance(mine, int) \
                    and isinstance(theirs, int) and theirs:
                fails.append(
                    "  the two lanes prefilled different prompts: %d vs %d "
                    "(delta %+d, ratio %.4f). Every other field in this block "
                    "can agree while this one does not — that is the measured "
                    "2026-08-29 case, and it is why this field is compared"
                    % (mine, theirs, mine - theirs, float(mine) / theirs))
    else:
        agreed = len(keys)

# THE ANDON LINE. `server_usage` is legal and is NOT canonical: §4.4.6 says
# server-reported usage fields are two implementations' opinions. Saying so on
# every receipt that uses it is what keeps the exactly-equal rule below from
# reading as pedantry when it reddens.
if subject_block.get(NAME_METHOD) == "server_usage":
    notes.append(
        "REPORT ArmC-tok method=server_usage is legal and NOT canonical "
        "(§4.4.6). Both counts are then a server's opinion of its own work; "
        "the only thing separating this receipt from the 4.09% prefill "
        "divergence measured on 2026-08-29 is that "
        "corpus_prefill_tokens_median is compared")

# WHAT THIS ARM STILL CANNOT SEE — stated because a rule whose blind spots are
# unwritten gets read as covering them:
#
#  1. CORPUS IDENTITY. Two lanes could agree on a median computed over
#     DIFFERENT prompt files. That is a join-key property (§4.4.8, "same
#     workload file") and belongs beside host/accelerator/model/quantization in
#     bench_receipt.py's JOIN_KEY_REQUIRED, not in this block. Uncaught here by
#     design; owner PERF-019.
#  2. BOTH LANES WRONG THE SAME WAY. Two lanes agreeing exactly on a prefill
#     length that is outside §4.3.1's declared W1 band (512 ± 8) pass. The band
#     is not represented in perf-matrix.yaml, and writing it in here would be
#     an invented continuous threshold in the one file that says there are
#     none.
#  3. HONEST-LOOKING LIES. counts_special_tokens and counts_prompt_echo are
#     still declarations. Two lanes that declare the same false value agree,
#     and this arm sees agreement. Only the two outcome fields are proof.
#  4. WHERE THE NUMBER CAME FROM. §4.5 requires STREAMING, and in streaming
#     mode NEITHER server reports `usage` at all — apr emits none even with
#     stream_options.include_usage, and `git grep stream_options` finds no
#     caller. So corpus_prefill_tokens_median must come from a separate
#     non-streaming replay of the corpus, and nothing here can tell such a
#     replay from a number that was typed in. The signature arm (§4.9.1) binds
#     the file to a host and a commit; it does not bind a field to an
#     instrument.
#  5. TOKENIZER CORRECTNESS. apr emits 7 ids where the reference emits 3 for
#     `<|im_end|>` preceded by sentence punctuation — 4 junk ids instead of the
#     turnstile. W1 prompts end on a word, so the W1 median does not move and
#     this arm stays green on a real correctness defect. That needs its own
#     falsifier against the reference tokenizer, not a receipt field.
for line in notes:
    print(line)
for line in fails:
    print(("FAIL ArmC-tok " + line) if not line.startswith("  ") else line)
if not fails:
    print("PASS  ArmC-tok %s"
          % ("blocks agree on %d field(s) incl. the resolved chat template and "
             "the measured prefill length" % agreed if agreed
             else "measured lane declares a legal §4.4.6 block; no comparator "
                  "ratio is armed"))
sys.exit(1 if fails else 0)
PY_TOK
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

def resolve_expiry(bl, m):
    """§4.7.3's deadline -- which is not always a calendar date (PERF-056, #2777).

    All four W2 cells carried a hardcoded `expires: '2026-09-25'`. §4.3.2 dates
    W2's expiry from an EVENT instead -- "PERF-001 merge + 30 days" -- because a
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
                       "clocks is no clock; pick the one §4.3 gives this cell")
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
    return ("unarmed", "expiry is %s merge + %d days (§4.3.2); %s has NOT merged "
                       "(status=%r, owner=%s), so the clock has not started"
                       % (name, days, name, a.get("status"), owner))

EXPIRY_KIND, EXPIRY_DETAIL = resolve_expiry(bl, m)
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
        #
        # §4.3.2 gives W2 an EVENT-dated expiry rather than a calendar one, so
        # the deadline is resolved by `resolve_expiry` above and only the
        # verdict is decided here.
        if EXPIRY_KIND == "bad":
            print(f"FAIL ArmA c={c}: baseline {EXPIRY_DETAIL}")
            fail=True
        elif EXPIRY_KIND == "unarmed":
            print(f"REPORT ArmA c={c} scaling_efficiency={eff:.4f} "
                  f"(baseline UNMEASURED, {EXPIRY_DETAIL}, owner={bl.get('owner')})")
        elif EXPIRY_DETAIL < TODAY:
            print(f"FAIL ArmA c={c}: baseline UNMEASURED and EXPIRED {EXPIRY_DETAIL} "
                  f"(today {TODAY}, owner={bl.get('owner')}) — measure it or "
                  f"re-decide the cell; do not extend the date to stay green")
            fail=True
        else:
            print(f"REPORT ArmA c={c} scaling_efficiency={eff:.4f} "
                  f"(baseline UNMEASURED until {EXPIRY_DETAIL}, owner={bl.get('owner')})")
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

arm_c_signature() {
  # SECTION 4.5's Arm C table, release row:
  #     | Receipt signature valid; `receipt.commit` contains
  #       `commit-under-test` | release |
  # and section 4.9.1: "The staleness arm is what makes it a gate. Without
  # receipt.commit contains commit-under-test, evidence/ is a declared-state
  # artifact."
  #
  # Two hosts are not CI runners and the fully-comparated one is
  # do-not-revive, so the gate cannot run ON the host that measures. What
  # arrives here is a FILE. Unsigned, that file binds to no host and no commit
  # -- anyone can write one, and this gate would read it as evidence.
  #
  # The crypto and the ancestry test live in scripts/lib/receipt_sig.py, which
  # scripts/perf_receipt_sign.sh also calls, so the signed payload cannot
  # drift between producer and verifier. This function owns the PHASE rule and
  # the andon line; it owns no key handling of its own.
  local receipt="$1" host="$2" phase="$3" commit="$4"
  if [ "$phase" != release ]; then
    # The only legal skip in this arm, and it is spec-mandated: section 4.5
    # scopes the rule to release. No PR can supply a host receipt, so wiring it
    # at merge would be a required check that can never PASS -- the mirror of
    # one that can never fail.
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
  python3 - "$receipt" "$ROOT" "$host" "$commit" <<'PY_SIG'
import json,os,sys
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
for code, message in fails:
    print("FAIL ArmC-sig %s: %s" % (code, message))
sys.exit(1 if fails else 0)
PY_SIG
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
  local host="$1" phase="$2" workload="$3" receipt="$4" commit="${5:-}" rc=0
  [ -f "$receipt" ] || die "receipt not found: $receipt"
  arm_c_integrity "$receipt" || rc=1
  arm_c_tokenization "$receipt" || rc=1
  arm_c_signature "$receipt" "$host" "$phase" "$commit" || rc=1
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
  # §4.4.6's block, in the CANONICAL shape, on BOTH lanes. The fixture this
  # replaces carried `"method":"hf-tokenizers"` — not one of the two values
  # §4.4.6 declares — so the OK case that proved the tokenization rule worked
  # was itself illegal input, and every other tokenizer stack's name would have
  # passed the same way.
  #
  # `corpus_prefill_tokens_median` LEADS each block on purpose: `"tokenization":{`
  # and `"comparator_tokenization":{` are then distinct anchors, so a row below
  # can mutate ONE lane without a positional trick. (`"tokenization":{"corpus`
  # cannot match inside `comparator_tokenization` — the character before
  # `tokenization` there is `r`, not a quote.)
  local TOKFIELDS COMPTOK
  TOKFIELDS='"method":"client_tokenizer","tokenizer_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","counts_special_tokens":true,"counts_prompt_echo":false,"chat_template_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"'
  COMPTOK='"comparator_tokenization":{"corpus_prefill_tokens_median":513,'"$TOKFIELDS"'},'
  local OK='{"requested":16,"completed":16,"timeouts":0,"drain_ms":12,
   "provenance":{"binary_path":"/opt/pinned-binary","binary_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
                 "resolution":"apr_bin.sh","compute_class":"cuda","host":"lambda","accelerator":"rtx-4090",
                 "model":"qwen2.5-coder-1.5b-instruct","quantization":"Q4_K_M"},
   "tokenization":{"corpus_prefill_tokens_median":513,'"$TOKFIELDS"'},
   '"$COMPTOK"'
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
    # A release run must satisfy the section 4.9.1 arm too. Signing here keeps
    # each row testing the arm it is NAMED for; without it every release row
    # below would go red on an unsigned receipt and prove nothing about Arm D
    # or Arm E.
    if [ "$3" = release ]; then
      _sigstamp "$1" "$sigc1" lambda
      _sigsign "$1" lambda-selftest
      if ( export APR_PERF_RECEIPT_KEYRING="$sigkr"; export PERF_GATE_GIT_DIR="$sigrepo"; run_gate lambda release "$4" "$tmp/$1.json" "$sigc1" ) >/dev/null 2>&1
      then got=pass; else got=fail; fi
    elif run_gate lambda "$3" "$4" "$tmp/$1.json" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$5" ]; then
      printf '  ok    %-34s expect=%s\n' "$1" "$5"
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s expected %s got %s\n' "$1" "$5" "$got"
      fail=$((fail + 1))
    fi
  }
  # ---- section 4.9.1 / I-10: signature + staleness (PERF-007) ---------------
  # Registered mutation, section 5: "Staleness arm | verdict job | receipt one
  # commit stale | fresh receipt green". Both polarities below, plus the three
  # ways a receipt can be about SOMETHING ELSE rather than merely old. Each row
  # asserts its own FAILURE CODE, not just the polarity: a stale receipt
  # reported as WRONG-HOST sends the reader to the wrong fix, which is the
  # apr_bin.sh STALE-vs-WRONG-TREE defect in a new place.
  local sigrepo sigc0 sigc1 sigkr sigkeya sigkeyb
  sigkeya=a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1
  sigkeyb=b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2
  sigrepo="$tmp/sigrepo"
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
  sigkr="$tmp/keyring"
  {
    printf 'lambda-selftest %s\n' "$sigkeya"
    printf 'gx10-selftest %s\n' "$sigkeyb"
  } > "$sigkr"

  _sigstamp() { # name, commit, host -- edits $tmp/<name>.json in place
    python3 -c 'import json,sys
with open(sys.argv[1]) as fh:
    r = json.load(fh)
r["commit"] = sys.argv[2]
r["provenance"]["host"] = sys.argv[3]
with open(sys.argv[1], "w") as fh:
    json.dump(r, fh)' "$tmp/$1.json" "$2" "$3"
  }
  _sigfix() { # name, base-json, commit, host  -> $tmp/<name>.json
    _mk "$1" "$2"
    _sigstamp "$1" "$3" "$4"
  }
  _sigsign() { # name, key_id
    python3 "$ROOT/scripts/lib/receipt_sig.py" --sign --in "$tmp/$1.json" \
      --out "$tmp/$1.json" --key-id "$2" --keyring "$sigkr" \
      --signed-at 2026-08-29T00:00:00Z >/dev/null
  }
  _casesig() { # name, fixture-name, host, commit-under-test, keyring, expect(pass|CODE)
    local out rc=0
    out="$( ( export APR_PERF_RECEIPT_KEYRING="$5"; export PERF_GATE_GIT_DIR="$sigrepo"; run_gate "$3" release W2 "$tmp/$2.json" "$4" ) 2>&1 )" || rc=$?
    if [ "$6" = pass ]; then
      if [ "$rc" = 0 ]; then
        printf '  ok    %-34s expect=pass\n' "$1"
        pass=$((pass + 1))
      else
        printf '  BROKE %-34s expected pass, gate said: %s\n' "$1" "$out"
        fail=$((fail + 1))
      fi
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
  # AN ABSENT COUNTER IS NOT A ZERO COUNTER. This read `r.get("timeouts",0)`,
  # so a receipt that never counted timeouts was indistinguishable from one
  # that counted none -- and no producer in this tree counts them at all
  # (loadtest.rs collapses every failure to one `failed` tally), so the default
  # was doing all the work on every real artifact. That is the same shape as a
  # missing measurement reading as a passing one, which is what this gate is
  # for.
  _case timeouts_absent_is_not_zero    "${OK/\"timeouts\":0,/}" fail
  _case tokenization_absent            "${OK/\"method\":\"client_tokenizer\"/\"method\":\"\"}" fail
  _case drain_ms_absent                "${OK/\"drain_ms\":12/\"drain_ms\":null}" fail
  # THIS ROW WAS GREEN FOR THE WRONG REASON (PERF-047). The old substitution
  # appended a stray `}`, so the fixture was not valid JSON and the case went
  # RED on a parse error: neutering the zero-token rule itself left the table at
  # 17/17. Measured, then fixed. The replacement changes ONE field and parses.
  _case zero_token_response            "${OK//\"tokens_total\":900/\"tokens_total\":0}" fail
  _case b1_aggregate_below_floor       "${OK//\"agg_ratio\":0.9/\"agg_ratio\":0.79}" fail
  _case b1_aggregate_at_floor          "${OK//\"agg_ratio\":0.9/\"agg_ratio\":0.80}" pass
  _case b2_decode_below_floor          "${OK//\"decode_ratio\":1.1/\"decode_ratio\":0.99}" fail
  _case b2_decode_at_floor             "${OK//\"decode_ratio\":1.1/\"decode_ratio\":1.00}" pass
  # THE 2026-08-25 SHAPE: decode soaring while aggregate collapses.
  _case serialization_shape_rejected   "$(printf '%s' "$OK" | sed 's/"agg_ratio":0.9/"agg_ratio":0.097/g; s/"decode_ratio":1.1/"decode_ratio":1.554/g')" fail
  # CELL COMPLETENESS had no row at all: every fixture above carries the full
  # band set, so mutating `sys.exit(1)` -> `sys.exit(0)` in cell_completeness
  # left the table at 17/17. A release-only arm that nothing exercises is the
  # §5 shape this table exists to refuse.
  local REL_SHORT
  REL_SHORT="${OK%\"bands\"*}$ARMDE"'"bands":[{"concurrency":1,"aggregate_tok_per_sec":100,"tokens_total":9,"agg_ratio":0.9,"decode_ratio":1.1},
            {"concurrency":4,"aggregate_tok_per_sec":360,"tokens_total":9,"agg_ratio":0.9,"decode_ratio":1.1}]}'
  _casepw cells_missing_bands_at_release       "$REL_SHORT" release W2 fail
  _casepw cells_complete_at_release            "$REL_DE"    release W2 pass
  _case band_c1_absent                 "$(printf '%s' "$OK" | python3 -c 'import sys,json;r=json.load(sys.stdin);r["bands"]=[b for b in r["bands"] if b["concurrency"]!=1];print(json.dumps(r))')" fail

  # ---- §4.4.6 / I-11: the tokenization block (PERF-057) --------------------
  # THE RULE WAS SPECIFIED AND ABSENT. `perf_gate.sh:79` tested that
  # `tokenization.method` was a non-empty STRING; nothing read the other three
  # fields; and no line in the file compared the measured lane against its
  # comparator. Every row below enters a branch that did not exist, and each
  # asserts the SENTENCE the gate prints, not merely its colour — a row that
  # checks polarity alone is green when the gate fails for an unrelated reason.
  #
  # The last row is the one that motivated the work: two blocks that are
  # IDENTICAL field for field, and lanes that prefilled 513 vs 534 tokens
  # (measured 2026-08-29, apr 0.64.0 vs llama-server 39173bcac, W1, min ==
  # median == max on both sides). Under §4.4.6 as written that receipt PASSES.
  local OK_NOCOMP OK_UNARMED
  OK_NOCOMP="${OK/$COMPTOK/}"
  # No band is comparator-gated, so no ratio exists and there is nothing to
  # refuse: the comparison must say it is unarmed rather than demand an operand.
  OK_UNARMED="${OK_NOCOMP//\"aggregate_tok_per_sec\"/\"comparator_status\":\"UNMEASURED\",\"aggregate_tok_per_sec\"}"
  _tokcase() { # name, json, expect(pass|fail), needle
    _mk "$1" "$2"
    local out rc=0
    out="$(run_gate lambda merge W1 "$tmp/$1.json" 2>&1)" || rc=$?
    local got=pass
    [ "$rc" = 0 ] || got=fail
    if [ "$got" != "$3" ]; then
      printf '  BROKE %-34s expected %s got %s\n' "$1" "$3" "$got"
      fail=$((fail + 1))
      return
    fi
    case "$out" in
      *"$4"*)
        printf '  ok    %-34s expect=%s\n' "$1" "$3"
        pass=$((pass + 1)) ;;
      *)
        printf '  BROKE %-34s %s but never said %s\n' "$1" "$3" "$4"
        fail=$((fail + 1)) ;;
    esac
  }
  # --- one lane's block, on its own -----------------------------------------
  _tokcase tok_blocks_match_passes "$OK" pass "blocks agree on"
  # The value rule the old string test could not make: "hf-tokenizers" is a
  # perfectly good non-empty string and is not one of the two legal values.
  _tokcase tok_method_illegal_value \
    "$(printf '%s' "$OK" | sed 's/"method":"client_tokenizer"/"method":"hf-tokenizers"/g')" \
    fail "is not one of"
  # ... and the other side of that rule: `server_usage` IS legal. A check that
  # only ever accepts the canonical spelling is a different rule than §4.4.6's.
  _tokcase tok_method_server_usage_is_legal \
    "$(printf '%s' "$OK" | sed 's/"method":"client_tokenizer"/"method":"server_usage"/g')" \
    pass "NOT canonical"
  _tokcase tok_client_tokenizer_needs_digest \
    "$(printf '%s' "$OK" | sed 's/"tokenizer_sha256":"c*",//g')" \
    fail "with no tokenizer_sha256"
  # ABSENCE-OR-CONSISTENCY. Under `server_usage` the server counted, so the
  # receipt is not obliged to name a tokenizer; a digest it DOES carry is still
  # checked for shape and still compared across the lanes.
  _tokcase tok_server_usage_digest_optional \
    "$(printf '%s' "$OK" | sed 's/"method":"client_tokenizer"/"method":"server_usage"/g; s/"tokenizer_sha256":"c*",//g')" \
    pass "PASS  ArmC-tok"
  _tokcase tok_digest_must_be_lowercase_hex \
    "$(printf '%s' "$OK" | sed 's/"tokenizer_sha256":"c*"/"tokenizer_sha256":"CAFE"/g')" \
    fail "lowercase hex"
  _tokcase tok_counts_flag_absent_is_fatal \
    "$(printf '%s' "$OK" | sed 's/"counts_special_tokens":true,//g')" \
    fail "counts_special_tokens absent"
  _tokcase tok_chat_template_absent_is_fatal \
    "$(printf '%s' "$OK" | sed 's/,"chat_template_sha256":"d*"//g')" \
    fail "chat_template_sha256 absent"
  _tokcase tok_prefill_median_absent_is_fatal \
    "$(printf '%s' "$OK" | sed 's/"corpus_prefill_tokens_median":513,//g')" \
    fail "corpus_prefill_tokens_median absent"
  _tokcase tok_prefill_median_must_be_positive "${OK//513/0}" fail "not a positive integer"
  # --- I-11: the two lanes, compared ----------------------------------------
  _tokcase tok_method_mismatch_is_fatal \
    "${OK/\"method\":\"client_tokenizer\"/\"method\":\"server_usage\"}" \
    fail "block MISMATCH"
  # R2, measured: llama-server counts the stop token and apr does not, exactly
  # +1 on every naturally-terminated request. That is this axis, with the two
  # servers on opposite sides of it.
  _tokcase tok_counts_special_mismatch_is_fatal \
    "${OK/\"counts_special_tokens\":true/\"counts_special_tokens\":false}" \
    fail "counts_special_tokens: measured="
  _tokcase tok_chat_template_mismatch_is_fatal \
    "${OK/dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee}" \
    fail "chat_template_sha256: measured="
  _tokcase tok_comparator_block_absent_is_fatal "$OK_NOCOMP" fail "comparator_tokenization absent"
  _tokcase tok_unarmed_comparison_reports "$OK_UNARMED" pass "not armed"
  # Nothing else in the gate reads this block, so a scalar here reaches this
  # arm and nothing else. Unguarded it is an AttributeError: fail-closed, and
  # a diagnosis of nothing.
  _tokcase tok_comparator_block_not_an_object \
    "${OK/$COMPTOK/\"comparator_tokenization\":\"server_usage\",}" \
    fail "is not an object"
  # THE ROW THIS WORK EXISTS FOR. Every field §4.4.6 names agrees. The lanes
  # prefilled different prompts. Deleting `corpus_prefill_tokens_median` from
  # the compared set turns THIS row — and only this row — green.
  _tokcase tok_identical_blocks_divergent_prefill \
    "${OK/\"comparator_tokenization\":{\"corpus_prefill_tokens_median\":513/\"comparator_tokenization\":{\"corpus_prefill_tokens_median\":534}" \
    fail "513 vs 534 (delta -21, ratio 0.9607)"

  # GREEN SIDE. A signed receipt whose commit contains the commit under test.
  _sigfix sig_fresh "$REL_DE" "$sigc1" lambda
  _sigsign sig_fresh lambda-selftest
  _casesig sig_signed_and_fresh_passes  sig_fresh lambda "$sigc1" "$sigkr" pass
  # ... and it also covers an ANCESTOR of what it measured: `contains`, not `==`.
  _casesig sig_covers_an_ancestor       sig_fresh lambda "$sigc0" "$sigkr" pass

  # RED SIDE 1 -- FRESHNESS. Section 5's registered mutation, verbatim: a
  # receipt one commit stale. Signature perfectly valid; the evidence is about
  # older code. Remedy: re-measure.
  _sigfix sig_stale "$REL_DE" "$sigc0" lambda
  _sigsign sig_stale lambda-selftest
  _casesig sig_one_commit_stale         sig_stale lambda "$sigc1" "$sigkr" STALE

  # RED SIDE 2 -- IDENTITY. Different failures, different remedies. None of
  # these is fixed by re-measuring.
  _sigfix sig_unsigned "$REL_DE" "$sigc1" lambda
  _casesig sig_unsigned_is_red          sig_unsigned lambda "$sigc1" "$sigkr" UNSIGNED
  _casesig sig_no_keyring_is_red        sig_fresh lambda "$sigc1" "" NO-KEYRING
  # An arm handed no input must FAIL, not stand down. run_gate is reachable
  # without main()'s usage check -- this is the row that keeps that honest.
  _casesig sig_no_commit_under_test     sig_fresh lambda "" "$sigkr" NO-COMMIT-UNDER-TEST
  _sigfix sig_other_host "$REL_DE" "$sigc1" gx10
  _sigsign sig_other_host gx10-selftest
  _casesig sig_receipt_from_other_host  sig_other_host lambda "$sigc1" "$sigkr" WRONG-HOST

  # RED SIDE 3 -- FORGERY. Sign a real receipt, then edit one number in the
  # body. This is the "evidence/ is a file anyone can write" case made concrete.
  cp "$tmp/sig_fresh.json" "$tmp/sig_forged.json"
  python3 -c 'import json,sys
with open(sys.argv[1]) as fh:
    r = json.load(fh)
r["bands"][0]["aggregate_tok_per_sec"] = 9999.0
with open(sys.argv[1], "w") as fh:
    json.dump(r, fh)' "$tmp/sig_forged.json"
  _casesig sig_body_edited_after_signing sig_forged lambda "$sigc1" "$sigkr" FORGED

  # DISCRIMINATION. The arm is release-scoped by section 4.5, so an unsigned
  # receipt at MERGE phase must stay green -- otherwise every PR reds on a rule
  # no PR can satisfy.
  if run_gate lambda merge W2 "$tmp/sig_unsigned.json" >/dev/null 2>&1; then
    printf '  ok    %-34s expect=pass\n' sig_unsigned_ok_at_merge
    pass=$((pass + 1))
  else
    printf '  BROKE %-34s merge phase reddened on a release-only arm\n' sig_unsigned_ok_at_merge
    fail=$((fail + 1))
  fi

  # ---- §4.7.3 THE EXPIRY CLOCK (PERF-056, #2777) ---------------------------
  # `PERF_GATE_TODAY` was added "so the selftest can prove BOTH sides of the
  # expiry boundary without waiting for a date to arrive", and then NO ROW EVER
  # USED IT: on 9d45b927d `grep -n PERF_GATE_TODAY scripts/perf_gate.sh`
  # returned exactly one line, its own definition. The clock deciding whether
  # every UNMEASURED cell in the matrix may still REPORT was itself untested,
  # which is the injectable-seam-with-no-case shape §5 catalogues.
  #
  # Rows 1-4 run against the COMMITTED matrix, so they break if the shipped
  # file changes shape. Rows 5-11 need a matrix no well-formed committed one
  # can be, and use $PERF_GATE_MATRIX for exactly that.
  #
  #  matrix / workload                       | today      | must | names
  #  ----------------------------------------|------------|------|-------------
  #  committed, W1 (fixed date)              | 2026-09-25 | pass | the date
  #  committed, W1 (fixed date)              | 2026-09-26 | FAIL | EXPIRED
  #  committed, W2 (event-dated)             | 2099-01-01 | pass | the ANCHOR
  #  committed, W2 (event-dated)             | 2026-08-29 | pass | why unarmed
  #  anchor merged 2026-09-01, +30d          | 2026-10-01 | pass | 2026-10-01
  #  anchor merged 2026-09-01, +30d          | 2026-10-02 | FAIL | EXPIRED
  #  anchor not declared in expiry_anchors   | any        | FAIL | undeclared
  #  cell declares BOTH clocks               | any        | FAIL | two clocks
  #  anchor status: merged, merged_on null   | any        | FAIL | null
  #  cell declares NEITHER clock             | any        | FAIL | never expires
  #  days is a string, not an integer        | any        | FAIL | integer
  _mk expiry "$OK"
  _mx() { # variant-name, python-edit-body -> prints the variant matrix path
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
        "the shape these expiry rows were written against -- most likely a W2 cell\n"
        "lost its `expires_after:` event-dated expiry (PERF-056, #2777) or an\n"
        "`expiry_anchors:` entry moved. Fix the matrix or the rows; do not delete\n"
        "the rows." % old[:60])
    s = s.replace(old, new)
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(s)
print(dst)
PY_MX
  }
  _expiry() { # name, matrix(empty=committed), workload, today, expect, needle
    local name="$1" mx="$2" wl="$3" today="$4" expect="$5" needle="$6"
    local saved="$MATRIX" out rc=0
    [ -n "$mx" ] && MATRIX="$mx"
    out="$(PERF_GATE_TODAY="$today" run_gate lambda merge "$wl" "$tmp/expiry.json" 2>&1)" || rc=$?
    MATRIX="$saved"
    local got=pass
    [ "$rc" = 0 ] || got=fail
    if [ "$got" != "$expect" ]; then
      printf '  BROKE %-34s expected %s got %s\n' "$name" "$expect" "$got"
      fail=$((fail + 1))
      return
    fi
    # POLARITY IS NOT ENOUGH. A row that only checks pass/fail is green when
    # the gate fails for an unrelated reason, which is how a wrong diagnosis
    # ships looking tested.
    case "$out" in
      *"$needle"*)
        printf '  ok    %-34s expect=%s\n' "$name" "$expect"
        pass=$((pass + 1)) ;;
      *)
        printf '  BROKE %-34s %s but never said %s\n' "$name" "$expect" "$needle"
        fail=$((fail + 1)) ;;
    esac
  }
  local MX_MERGED MX_NOANCHOR MX_BOTH MX_MERGEDNULL MX_NEITHER MX_BADDAYS
  MX_MERGED="$(_mx merged "merged_on: null"$'\x1f'"merged_on: '2026-09-01'")"
  MX_NOANCHOR="$(_mx noanchor "{anchor: PERF-001, days: 30}"$'\x1f'"{anchor: PERF-999, days: 30}")"
  MX_BOTH="$(_mx both "expires_after: {anchor: PERF-001, days: 30}"$'\x1f'"expires_after: {anchor: PERF-001, days: 30}
      expires: '2027-01-01'")"
  MX_MERGEDNULL="$(_mx mergednull "status: in_progress
    # Set by the PR"$'\x1f'"status: merged
    # Set by the PR")"
  MX_NEITHER="$(_mx neither "expires_after: {anchor: PERF-001, days: 30}"$'\x1f'"absent_on_purpose: true")"
  MX_BADDAYS="$(_mx baddays "{anchor: PERF-001, days: 30}"$'\x1f'"{anchor: PERF-001, days: '30'}")"

  _expiry expiry_w1_fixed_date_still_reports  ""              W1 2026-09-25 pass "UNMEASURED until 2026-09-25"
  _expiry expiry_w1_fixed_date_expires        ""              W1 2026-09-26 fail "EXPIRED 2026-09-25"
  _expiry expiry_w2_is_event_dated            ""              W2 2099-01-01 pass "PERF-001 merge + 30 days"
  _expiry expiry_w2_says_why_it_is_unarmed    ""              W2 2026-08-29 pass "has NOT merged"
  _expiry expiry_anchor_merge_arms_the_clock  "$MX_MERGED"    W2 2026-10-01 pass "UNMEASURED until 2026-10-01"
  _expiry expiry_anchor_merge_then_expires    "$MX_MERGED"    W2 2026-10-02 fail "EXPIRED 2026-10-01"
  _expiry expiry_undeclared_anchor_is_fatal   "$MX_NOANCHOR"  W2 2026-08-29 fail "not declared under"
  _expiry expiry_two_clocks_is_fatal          "$MX_BOTH"      W2 2026-08-29 fail "two clocks is no clock"
  _expiry expiry_merged_without_date_is_fatal "$MX_MERGEDNULL" W2 2026-08-29 fail "cannot start from null"
  _expiry expiry_no_deadline_at_all_is_fatal  "$MX_NEITHER"   W2 2026-08-29 fail "never expires"
  _expiry expiry_days_must_be_an_integer      "$MX_BADDAYS"   W2 2026-08-29 fail "non-negative integer"

  # main() must refuse `--phase release` with no `--commit` as a USAGE error
  # (exit 2), in a subshell because `die` exits. A gate that silently accepts a
  # release invocation missing the arm that makes it a gate is the whole defect.
  if ( main --host lambda --phase release --workload W1 --receipt "$tmp/sig_fresh.json" ) >/dev/null 2>&1; then
    printf '  BROKE %-34s release without --commit was accepted\n' main_requires_commit_at_release
    fail=$((fail + 1))
  else
    printf '  ok    %-34s expect=usage-error\n' main_requires_commit_at_release
    pass=$((pass + 1))
  fi

  # The fine-grained crypto/containment table and the host-side signer both
  # carry their own case tables. Running them from HERE is what wires them:
  # ci.yml already invokes `perf_gate.sh --selftest`, so neither needs a new
  # workflow line, and a guard nothing invokes is this epic's most common
  # finding.
  if python3 "$ROOT/scripts/lib/receipt_sig.py" --selftest >/dev/null 2>&1; then
    printf '  ok    %-34s expect=pass\n' receipt_sig_case_table
    pass=$((pass + 1))
  else
    printf '  BROKE %-34s scripts/lib/receipt_sig.py --selftest failed\n' receipt_sig_case_table
    fail=$((fail + 1))
  fi
  if bash "$ROOT/scripts/perf_receipt_sign.sh" --selftest >/dev/null 2>&1; then
    printf '  ok    %-34s expect=pass\n' receipt_signer_case_table
    pass=$((pass + 1))
  else
    printf '  BROKE %-34s scripts/perf_receipt_sign.sh --selftest failed\n' receipt_signer_case_table
    fail=$((fail + 1))
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
    die "--commit <commit-under-test> is required at --phase release: the staleness arm (section 4.9.1) has nothing to compare without it, and it is the arm that makes this a gate"
  fi
  run_gate "$host" "$phase" "$workload" "$receipt" "$commit"
}
main "$@"
