#!/usr/bin/env bash
# check_perf_receipt_fields_have_producers.sh — PERF-004.
#
# WHY THIS EXISTS
# ---------------
# scripts/perf_gate.sh read a receipt schema that NOTHING IN THIS TREE WROTE.
# Four real artifacts were tried against it and all four returned rc=1; the only
# input that passed was the gate's own synthetic fixture, a string literal
# inside the gate. Nobody noticed for the length of an epic, because the gate
# ran green against itself and every real input failed with one line that said
# "schema" and named nothing.
#
# The dangerous repair is the obvious one: teach the gate to fill in the missing
# fields. `requested`, `completed`, `timeouts`, `drain_ms` and
# `tokenization.method` all look like the same kind of gap and they are not.
# Two are derivable from counts the harness already records. Three are
# UNMEASURED -- nothing in this tree measures them -- and defaulting those three
# would hand-assign three numbers no instrument produced, wearing the shape of
# measurements. That is the defect APR-PERF-GATE-001 exists to remove, arriving
# as its own fix.
#
# So scripts/perf-receipt-fields.yaml classifies every field the gate reads, and
# this guard holds the classification to the tree:
#
#   1. every field the gate reads is classified            (no silent new gap)
#   2. every classified field is actually read              (no rotting entry)
#   3. PRODUCED/DERIVED name a producer that still exists   (no rotting claim)
#   4. UNMEASURED names an owning ticket and a spec section (no orphan gap)
#   5. every receiver in the reader scripts is accounted    (no silent miss)
#
# Rule 5 is the one that keeps the other four honest. The extractor INCLUDES a
# receiver by default and fails on one it does not recognise, so a new read
# through an unfamiliar variable is a loud false alarm rather than a field that
# quietly leaves the universe. A guard whose universe can shrink without saying
# so is the shape that made three earlier guards in this epic blind.
#
#   bash scripts/check_perf_receipt_fields_have_producers.sh              # gate
#   bash scripts/check_perf_receipt_fields_have_producers.sh --self-test  # case table
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- the check --
run_check() {
  local root="$1"
  python3 - "$root" <<'PY'
import os, re, sys, yaml

root = sys.argv[1]
LEDGER = os.path.join(root, "scripts", "perf-receipt-fields.yaml")
GATE = os.path.join(root, "scripts", "perf_gate.sh")
DELEGATE = os.path.join(root, "scripts", "lib", "bench_receipt.py")

# Receiver -> the path prefix a read through it addresses. A receiver absent
# from BOTH this map and the ledger's non_receipt_receivers list is an error,
# never a skip.
PREFIX = {
    # PERF-048 added `tok`, `rep` and `wc`: Arm C's 4.4.6/4.4.2/4.3 checks read
    # several sub-fields each, and the chained `(r.get("x") or {}).get("y")`
    # form the gate used for a single read does not survive that. They are
    # NAMED receivers with a path prefix rather than skipped ones, so their
    # fields stay inside this guard's universe -- a receiver added to
    # non_receipt_receivers would have removed seven required fields from it.
    GATE: {"r": "", "b": "bands[].", "bands": "bands[].",
           "kv": "kv.", "itl": "itl.", "inj": "injector.",
           "tok": "tokenization.", "rep": "replicates.",
           "wc": "workload_corpus."},
    DELEGATE: {"receipt": "", "prov": "provenance.",
               "subj_prov": "provenance.", "comp_prov": "provenance."},
}

# EVERY literal key access, both quote styles. The extractor is TOTAL: each
# match must be attributed to a receiver, and one it cannot attribute is an
# error rather than a skip. The first version matched double quotes only and
# silently dropped `kv['admission_rejected']` inside an f-string -- a field that
# happened to be classified anyway through its other spelling, which is exactly
# how a parser's blind spot survives review. A guard's own parser is the part
# most likely to be wrong and least likely to be read.
TOKEN = re.compile(r"""\.get\(\s*(['"])(\w+)\1|\[\s*(['"])(\w+)\3\s*\]""")
# Text immediately before a token, when the receiver is a plain name possibly
# followed by subscripts: `r`, `bands[c]`, `m["arms"]["B1"]`.
RECV = re.compile(r'([A-Za-z_]\w*)(?:\[[^\]]*\])*\s*$')
# ... and when it is `(<name>.get("outer") or {})`, the one chained form in use.
CHAIN = re.compile(r'([A-Za-z_]\w*)\.get\(\s*[\'"](\w+)[\'"]\s*\)\s*or\s*\{\}\s*\)\s*$')
# ... and `(<name> or {})`, the null-coalescing form, where the receiver is the
# name and the key is not qualified any further.
COALESCE = re.compile(r'([A-Za-z_]\w*)\s*or\s*(?:\{\}|\[\])\s*\)\s*$')
TUPLE = re.compile(r'^(PROVENANCE_REQUIRED|JOIN_KEY_REQUIRED)\s*=\s*\(([^)]*)\)', re.M)

errors = []
with open(LEDGER, encoding="utf-8") as fh:
    ledger = yaml.safe_load(fh)
fields = ledger.get("fields") or {}
skip_recv = ledger.get("non_receipt_receivers") or {}
dispatch = set(ledger.get("dispatch_keys") or [])


def read_set(path):
    """Every receipt field this file reads, as a qualified path."""
    base = os.path.basename(path)
    allowed = skip_recv.get(base) or {}
    prefix = PREFIX[path]
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    found = set()
    for match in TOKEN.finditer(src):
        name = match.group(2) or match.group(4)
        before = src[:match.start()]
        chain = CHAIN.search(before)
        coalesce = COALESCE.search(before)
        if chain:
            recv, outer = chain.group(1), chain.group(2)
            qualified = outer + "." + name
        elif coalesce:
            recv, qualified = coalesce.group(1), name
        else:
            plain = RECV.search(before)
            if not plain:
                line = src.count("\n", 0, match.start()) + 1
                errors.append(
                    "%s:%d has a literal key access %r the extractor cannot "
                    "attribute to a receiver. Give it a named receiver, or "
                    "teach the extractor the shape -- an unattributed read "
                    "leaves this guard's universe without saying so."
                    % (base, line, name))
                continue
            recv, qualified = plain.group(1), name
        if recv in allowed:
            continue
        if recv not in prefix:
            errors.append(
                "%s reads `%s.%s` through receiver %r, which is neither a "
                "known receipt receiver nor listed under "
                "non_receipt_receivers in the ledger. Classify it: a receiver "
                "the extractor does not recognise silently removes its fields "
                "from this guard's universe." % (base, recv, name, recv))
            continue
        if qualified in dispatch and not prefix[recv]:
            continue
        found.add(prefix[recv] + qualified)
    for _tuple_name, body in TUPLE.findall(src):
        for item in re.findall(r'"(\w+)"', body):
            found.add("provenance." + item)
    return found


universe = set()
for path in (GATE, DELEGATE):
    universe |= read_set(path)

for name in sorted(universe - set(fields)):
    errors.append(
        "field %r is READ by the gate and absent from the ledger. Classify it "
        "PRODUCED, DERIVED, POLICY or UNMEASURED -- an unclassified required "
        "field is how the gate came to read a schema nothing writes." % name)

for name in sorted(set(fields) - universe):
    errors.append(
        "field %r is classified in the ledger and read by nothing. A stale "
        "entry makes the map look more complete than the gate is." % name)

for name in sorted(set(fields) & universe):
    spec = fields[name] or {}
    klass = spec.get("class")
    if klass in ("PRODUCED", "DERIVED"):
        prod = spec.get("producer") or {}
        rel, symbol = prod.get("file"), prod.get("symbol")
        if not rel or not symbol:
            errors.append("%s is %s and names no producer file+symbol" % (name, klass))
            continue
        target = os.path.join(root, rel)
        if not os.path.exists(target):
            errors.append("%s names producer %s, which does not exist" % (name, rel))
            continue
        with open(target, encoding="utf-8", errors="replace") as fh:
            if symbol not in fh.read():
                errors.append(
                    "%s claims %s produces it via %r, and that text is no "
                    "longer in the file. A producer claim that nothing checks "
                    "rots into a fiction." % (name, rel, symbol))
        if klass == "DERIVED" and not spec.get("formula"):
            errors.append("%s is DERIVED and states no formula" % name)
    elif klass == "POLICY":
        rel = spec.get("policy")
        if not rel or not os.path.exists(os.path.join(root, rel)):
            errors.append("%s is POLICY and names no existing decision file" % name)
    elif klass == "UNMEASURED":
        if not re.match(r'^(PERF|BENCH)-\d{3}$', str(spec.get("owner") or "")):
            errors.append(
                "%s is UNMEASURED with owner=%r. Unmeasured is UNMEASURED WITH "
                "AN OWNER; without one the gap has no route to being closed."
                % (name, spec.get("owner")))
        if "§" not in str(spec.get("spec") or ""):
            errors.append("%s is UNMEASURED and cites no spec section" % name)
        if not (spec.get("needs") or "").strip():
            errors.append(
                "%s is UNMEASURED and says nothing about what would have to be "
                "instrumented. 'absent' is a schema error; 'the drain rule is "
                "not implemented in any client' is a finding." % name)
    else:
        errors.append("%s has class=%r, which is not one of PRODUCED, DERIVED, "
                      "POLICY, UNMEASURED" % (name, klass))

# VACUITY. A sweep over an empty universe is clean and means nothing; the gate
# reads well over a dozen receipt fields and always will.
if len(universe) < 12:
    errors.append("the extracted universe collapsed to %d field(s). A clean "
                  "sweep over nothing is not a pass." % len(universe))

if errors:
    print("FAIL  perf receipt schema map")
    for e in errors:
        print("      " + e)
    sys.exit(1)
counts = {}
for name in universe:
    counts[(fields[name] or {}).get("class")] = counts.get((fields[name] or {}).get("class"), 0) + 1
print("ok    %d field(s) read by the gate, all classified: %s"
      % (len(universe), ", ".join("%s=%d" % kv for kv in sorted(counts.items()))))
PY
}

# ------------------------------------------------------------- case table ---
# Every mutation below is applied to a COPY of the tree, the guard is re-run
# against that copy, and the expected colour is asserted. A guard seen only
# passing is not evidence; these rows are the evidence, and they are re-run by
# CI rather than recorded in a commit message.
self_test() {
  local pass=0 fail=0
  # NOT `local`: the EXIT trap below runs after this function has returned, and
  # a local would be out of scope by then -- `${td:?}` would fire on every run
  # and print a failure after a clean pass.
  td="$(mktemp -d)" || exit 2
  case "$td" in
    /tmp/*|/var/folders/*) : ;;
    *) printf 'refusing to rm -rf %s\n' "${td:-<empty>}" >&2; exit 2 ;;
  esac
  trap 'rm -rf "${td:?}"' EXIT

  # Every producer file the ledger names, READ FROM THE LEDGER.
  #
  # This used to be two hardcoded `cp` lines. PERF-048 added four producer files
  # and `clean_tree` went red for a reason having nothing to do with the tree:
  # a fixture universe that cannot follow the thing it is a fixture for is the
  # same defect this guard exists to catch, one level up.
  _producer_files() {
    python3 - "$ROOT/scripts/perf-receipt-fields.yaml" <<'LEDGER'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for spec in (doc.get("fields") or {}).values():
    rel = ((spec or {}).get("producer") or {}).get("file")
    if rel:
        print(rel)
LEDGER
  }

  _fixture() { # name -> a fresh copy of the tree at $td/$1
    rm -rf "${td:?}/$1"
    mkdir -p "$td/$1/scripts/lib" "$td/$1/crates" "$td/$1/evidence"
    cp "$ROOT/scripts/perf_gate.sh" "$td/$1/scripts/"
    cp "$ROOT/scripts/perf-receipt-fields.yaml" "$td/$1/scripts/"
    cp "$ROOT/scripts/perf-matrix.yaml" "$td/$1/scripts/"
    cp "$ROOT/scripts/lib/bench_receipt.py" "$td/$1/scripts/lib/"
    cp "$ROOT/scripts/lib/parity_block.py" "$td/$1/scripts/lib/"
    mkdir -p "$td/$1/crates/aprender-test-lib/src/llm" "$td/$1/crates/apr-cli/src/commands"
    local rel
    while IFS= read -r rel; do
      [ -n "$rel" ] || continue
      mkdir -p "$td/$1/$(dirname "$rel")"
      cp "$ROOT/$rel" "$td/$1/$rel"
    done < <(_producer_files)
  }

  _expect() { # name, expected(pass|fail)
    local got
    if run_check "$td/$1" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$2" ]; then
      printf '  ok    %-38s expect=%s\n' "$1" "$2"; pass=$((pass + 1))
    else
      printf '  BROKE %-38s expected %s got %s\n' "$1" "$2" "$got"; fail=$((fail + 1))
    fi
  }

  # The tree as it stands. If this is not green the rest proves nothing.
  _fixture clean_tree
  _expect clean_tree pass

  # MUTATION 1 -- the defect that produced this ticket, in miniature: the gate
  # grows a requirement nobody classified.
  _fixture read_without_ledger_entry
  cat >> "$td/read_without_ledger_entry/scripts/perf_gate.sh" <<'MUT'
arm_f_prefill() { python3 -c 'r={}; print(r.get("prefill_efficiency"))'; }
MUT
  _expect read_without_ledger_entry fail

  # MUTATION 2 -- the map claims more coverage than the gate has.
  _fixture ledger_entry_nothing_reads
  printf '  invented_field:\n    class: DERIVED\n    required: always\n    producer: {file: scripts/perf_gate.sh, symbol: "arm_a_scaling"}\n    formula: nothing reads this\n' \
    >> "$td/ledger_entry_nothing_reads/scripts/perf-receipt-fields.yaml"
  _expect ledger_entry_nothing_reads fail

  # MUTATION 3 -- an unmeasured gap loses its owner and becomes permanent.
  _fixture unmeasured_without_owner
  sed -i 's/^    owner: PERF-004$//' \
    "$td/unmeasured_without_owner/scripts/perf-receipt-fields.yaml"
  _expect unmeasured_without_owner fail

  # MUTATION 4 -- THE FORBIDDEN REPAIR. An unmeasured field is relabelled as
  # produced, which is precisely how five invented numbers would enter the
  # receipt wearing the shape of measurements. The producer claim is checked
  # against the file, so the relabel cannot stand on its own.
  _fixture unmeasured_laundered_to_produced
  python3 - "$td/unmeasured_laundered_to_produced/scripts/perf-receipt-fields.yaml" <<'MUT'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = s.replace(
    "  drain_ms:\n    class: UNMEASURED",
    '  drain_ms:\n    producer: {file: crates/aprender-test-lib/src/llm/loadtest.rs, '
    'symbol: "let drain_ms"}\n    formula: recorded by the client\n    class: PRODUCED')
open(p, "w", encoding="utf-8").write(s)
MUT
  _expect unmeasured_laundered_to_produced fail

  # MUTATION 5 -- a read through a receiver the extractor does not know. It
  # must be loud, not skipped: a receiver that silently drops out of the map
  # takes its fields with it.
  _fixture unknown_receiver
  cat >> "$td/unknown_receiver/scripts/perf_gate.sh" <<'MUT'
arm_g_mystery() { python3 -c 'zz={}; print(zz.get("mystery"))'; }
MUT
  _expect unknown_receiver fail

  # DISCRIMINATION -- a SECOND read of a field the ledger already classifies is
  # not a violation. Without this row the guard could be keying on the size of
  # the file, or on the count of reads, and every mutation above would still
  # look correct.
  _fixture second_read_of_ledgered_field
  printf 'x=$(python3 -c "r={}; r.get(\\"drain_ms\\")")\n' \
    >> "$td/second_read_of_ledgered_field/scripts/perf_gate.sh"
  _expect second_read_of_ledgered_field pass

  # MUTATION 6 -- PERF-048 widened this guard's scope with three new receipt
  # receivers (`tok`, `rep`, `wc`), so the old proof does not transfer: these
  # rows are the RED and the GREEN re-run INSIDE the new scope. A read through
  # `tok` that nobody classified must be caught, and caught as an unclassified
  # FIELD rather than as an unknown receiver -- the latter would mean `tok`'s
  # seven fields had left this guard's universe entirely.
  _fixture unclassified_read_through_a_new_receiver
  cat >> "$td/unclassified_read_through_a_new_receiver/scripts/perf_gate.sh" <<'MUT'
arm_h_tok() { python3 -c 'tok={}; print(tok.get("brand_new_subfield"))'; }
MUT
  _expect unclassified_read_through_a_new_receiver fail

  _fixture unclassified_read_through_the_corpus_receiver
  cat >> "$td/unclassified_read_through_the_corpus_receiver/scripts/perf_gate.sh" <<'MUT'
arm_h_wc() { python3 -c 'wc={}; print(wc.get("brand_new_subfield"))'; }
MUT
  _expect unclassified_read_through_the_corpus_receiver fail

  # DISCRIMINATION -- the same read WITH its ledger entry passes, under the
  # `tokenization.` prefix. This is the row that proves `tok` is a recognised
  # RECEIPT receiver rather than a skipped one: a skipped receiver would leave
  # the new entry classified-and-read-by-nothing, and a wrong prefix would
  # leave it unmatched. Both of those fail here; only the correct mapping passes.
  _fixture classified_read_through_a_new_receiver
  cat >> "$td/classified_read_through_a_new_receiver/scripts/perf_gate.sh" <<'MUT'
arm_h_tok() { python3 -c 'tok={}; print(tok.get("brand_new_subfield"))'; }
MUT
  printf '  tokenization.brand_new_subfield:\n    class: PRODUCED\n    required: always\n    producer: {file: scripts/perf_gate.sh, symbol: "arm_c_integrity"}\n' \
    >> "$td/classified_read_through_a_new_receiver/scripts/perf-receipt-fields.yaml"
  _expect classified_read_through_a_new_receiver pass

  # DISCRIMINATION -- editing a producer file without touching the cited symbol
  # leaves the map intact. A guard that reddened on any edit to loadtest.rs
  # would be trained away within a week.
  _fixture unrelated_producer_edit
  printf '\n// an unrelated comment\n' \
    >> "$td/unrelated_producer_edit/crates/aprender-test-lib/src/llm/loadtest.rs"
  _expect unrelated_producer_edit pass

  printf '  %d passed, %d broken\n' "$pass" "$fail"
  [ "$fail" = 0 ]
}

case "${1:-}" in
  --self-test|--selftest) self_test ;;
  "") run_check "$ROOT" ;;
  *) printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
