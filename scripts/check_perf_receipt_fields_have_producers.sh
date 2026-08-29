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
    # `prov` and `sig` are the two intermediates arm_c_signature binds before
    # reading through them (`prov=r.get("provenance") or {}`). They address the
    # same sub-objects `r["provenance"]` and `r["signature"]` do, so they are
    # mapped rather than exempted: a receiver listed here still has EVERY key
    # read through it checked against the ledger. `prov` already carried this
    # meaning under DELEGATE.
    GATE: {"r": "", "b": "bands[].", "bands": "bands[].",
           "kv": "kv.", "itl": "itl.", "inj": "injector.",
           "prov": "provenance.", "sig": "signature."},
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
# followed by subscripts: `r`, `bands[c]`, `m["arms"]["B1"]`. The subscript
# chain is CAPTURED rather than discarded -- see qualify().
RECV = re.compile(r'([A-Za-z_]\w*)((?:\[[^\]]*\])*)\s*$')
SUBSCRIPT = re.compile(r'\[([^\]]*)\]')
LITERAL_KEY = re.compile(r'^\s*[\'"](\w+)[\'"]\s*$')
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


def qualify(chain):
    """The path a receiver's own subscript chain addresses, as a prefix.

    `r["provenance"]["host"]` addresses provenance.host and
    `r["bands"][0][...]` addresses bands[]..., but RECV used to swallow the
    whole chain into the receiver name and report the bare leaf. That renamed
    two real fields into `host` and `aggregate_tok_per_sec`, which are receipt
    fields under NO spelling, while their true spellings sat classified in the
    ledger the whole time. A map is keyed by a field's name, so a parser that
    renames a field cannot be checked against it -- the guard demanded entries
    for two fields that must never exist, and the obliging repair would have
    invented them.

    A literal key extends the path; an index INTO the last literal key makes it
    a list, matching the `bands[].` spelling the ledger and PREFIX already use.
    """
    out = ""
    for raw in SUBSCRIPT.findall(chain):
        literal = LITERAL_KEY.match(raw)
        if literal:
            out += literal.group(1) + "."
        elif out:
            out = out[:-1] + "[]."
    return out


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
            recv = plain.group(1)
            qualified = qualify(plain.group(2)) + name
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

  _fixture() { # name -> a fresh copy of the tree at $td/$1
    rm -rf "${td:?}/$1"
    mkdir -p "$td/$1/scripts/lib" "$td/$1/crates" "$td/$1/evidence"
    cp "$ROOT/scripts/perf_gate.sh" "$td/$1/scripts/"
    cp "$ROOT/scripts/perf-receipt-fields.yaml" "$td/$1/scripts/"
    cp "$ROOT/scripts/perf-matrix.yaml" "$td/$1/scripts/"
    cp "$ROOT/scripts/lib/bench_receipt.py" "$td/$1/scripts/lib/"
    cp "$ROOT/scripts/lib/parity_block.py" "$td/$1/scripts/lib/"
    # P4, the signer. Named as producer by `signature` and `signature.key_id`,
    # so a fixture without it fails the producer-existence check on EVERY row
    # -- including the pass-expected ones, which is how this omission announced
    # itself rather than quietly weakening the table.
    cp "$ROOT/scripts/lib/receipt_sig.py" "$td/$1/scripts/lib/"
    mkdir -p "$td/$1/crates/aprender-test-lib/src/llm" "$td/$1/crates/apr-cli/src/commands"
    cp "$ROOT/crates/aprender-test-lib/src/llm/loadtest.rs" "$td/$1/crates/aprender-test-lib/src/llm/"
    cp "$ROOT/crates/aprender-test-lib/src/llm/benchmark.rs" "$td/$1/crates/aprender-test-lib/src/llm/"
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

  # DISCRIMINATION -- editing a producer file without touching the cited symbol
  # leaves the map intact. A guard that reddened on any edit to loadtest.rs
  # would be trained away within a week.
  _fixture unrelated_producer_edit
  printf '\n// an unrelated comment\n' \
    >> "$td/unrelated_producer_edit/crates/aprender-test-lib/src/llm/loadtest.rs"
  _expect unrelated_producer_edit pass

  # DISCRIMINATION -- a nested literal chain must QUALIFY, not collapse to its
  # leaf. `r["bands"][0]["tokens_total"]` addresses bands[].tokens_total, which
  # the ledger classifies. RECV used to swallow the whole chain into the
  # receiver, so this would have reported a bare `tokens_total` -- a receipt
  # field under no spelling -- and reddened. That is verbatim the defect that
  # made the merge queue demand ledger entries for `host` and
  # `aggregate_tok_per_sec`: the obliging repair invents the two fields the
  # parser named, and the map then documents a schema nothing has.
  _fixture nested_chain_qualifies
  cat >> "$td/nested_chain_qualifies/scripts/perf_gate.sh" <<'MUT'
arm_h_nested() { python3 -c 'r={"bands":[{}]}; print(r["bands"][0]["tokens_total"])'; }
MUT
  _expect nested_chain_qualifies pass

  # MUTATION 6 -- a nested chain whose LEAF is unclassified is still caught.
  # Without this row the one above could be passing because nested reads left
  # the universe altogether, which is the failure mode it exists to exclude.
  _fixture nested_leaf_unclassified
  cat >> "$td/nested_leaf_unclassified/scripts/perf_gate.sh" <<'MUT'
arm_i_nested() { python3 -c 'r={"provenance":{}}; print(r["provenance"]["invented_leaf"])'; }
MUT
  _expect nested_leaf_unclassified fail

  # MUTATION 7 -- mapping a receiver must CLASSIFY its keys, never exempt them.
  # `sig` and `prov` joined PREFIX to resolve arm_c_signature. The cheap wrong
  # fix was to list them under non_receipt_receivers instead, which reads as
  # the same one-line repair and would have dropped every signature field out
  # of the universe in silence. `signature.alg` is real and unclassified
  # because nothing reads it; a read of it must redden.
  _fixture mapped_receiver_still_checked
  cat >> "$td/mapped_receiver_still_checked/scripts/perf_gate.sh" <<'MUT'
arm_j_sig() { python3 -c 'sig={}; print(sig.get("alg"))'; }
MUT
  _expect mapped_receiver_still_checked fail

  printf '  %d passed, %d broken\n' "$pass" "$fail"
  [ "$fail" = 0 ]
}

case "${1:-}" in
  --self-test|--selftest) self_test ;;
  "") run_check "$ROOT" ;;
  *) printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
