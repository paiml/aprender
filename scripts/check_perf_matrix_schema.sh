#!/usr/bin/env bash
# check_perf_matrix_schema.sh — the single source of truth needs a truth check.
#
# WHY THIS EXISTS (PP-33, PP-1, PP-16)
# ------------------------------------
# scripts/perf-matrix.yaml is the file every threshold, phase and cell status
# lives in, and NOTHING validated it. Measured on 2026-09-02, all four of these
# passed CI:
#
#   * `arms.B1.floor: 0.80` with `threshold_class: policy` and NO author, in a
#     file whose own header says a policy number needs one.
#   * `arms.B2.inherited_from: docs/specifications/perf-parity-spec.md` -- a path
#     with no history in ANY ref (`git log --all` empty). A precedence claim
#     pointing at a document that never existed.
#   * `hosts.mini.compute_class: metal`, a dispatch path no build in this tree
#     can reach, so every receipt that host could produce would declare a class
#     its binary never took.
#   * a `cell_exceptions` entry naming comparator `vllm`, which no host declares,
#     so it could not match any receipt and had never been read.
#
# Each is a claim the matrix makes about itself. This checks them.
#
#   R1  schema_version is one this reader understands
#   R2  hosts.*.compute_class is a known class, or null on a host marked NA
#   R3  every host declares `reachable_by` -- which cargo features can take it
#   R4  every NUMBER on the policy surface is covered by a threshold_class and
#       an author, inherited from the nearest ancestor that declares them
#   R5  `inherited_from` resolves to a path in the tree
#   R6  every baseline cell has a legal status and that status's required fields
#   R7  every `expires_after.anchor` is declared under `expiry_anchors:`
#   R8  the `protocol:` and `ladder:` blocks exist
#   R9  a cell_exceptions comparator is one some host declares
#
#   bash scripts/check_perf_matrix_schema.sh              # check
#   bash scripts/check_perf_matrix_schema.sh --selftest   # case table
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_check() { # run_check <root> [matrix-path]
  python3 - "$1" "${2:-}" <<'PY'
import os, sys, yaml

root = sys.argv[1]
matrix = sys.argv[2] or os.path.join(root, "scripts", "perf-matrix.yaml")
errors = []

try:
    with open(matrix, encoding="utf-8") as fh:
        m = yaml.safe_load(fh) or {}
except (OSError, yaml.YAMLError) as exc:
    print("FAIL  perf-matrix.yaml: %s" % exc)
    sys.exit(1)

KNOWN_SCHEMA = (1, 2)
CLASSES = ("cpu", "cuda", "metal", "wgpu")
THRESHOLD_CLASSES = ("policy", "inherited", "ratchet")
CELL_STATUSES = ("MEASURED", "UNMEASURED", "NA")
# THE POLICY SURFACE. R4 applies here and nowhere else: these are the blocks a
# GATE reads a number out of. `hosts.*.github_issue` and `cell_exceptions[].band`
# are numbers too and are not thresholds, and demanding an author for them would
# train the rule away.
POLICY_ROOTS = ("protocol", "ladder", "witness", "stream", "comparator_template",
                "derivation", "arms")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---- R1 --------------------------------------------------------------------
if m.get("schema_version") not in KNOWN_SCHEMA:
    errors.append("schema_version=%r is not one of %s. A reader that does not "
                  "know the generation it is reading cannot know which rules "
                  "apply." % (m.get("schema_version"), list(KNOWN_SCHEMA)))

# ---- R2, R3 ----------------------------------------------------------------
hosts = m.get("hosts") or {}
if not hosts:
    errors.append("`hosts:` is empty -- the expected cell set has no rows")
declared_comparators = set()
for name, host in sorted(hosts.items()):
    host = host or {}
    declared_comparators.add(host.get("comparator"))
    klass = host.get("compute_class")
    if klass is None:
        if host.get("status") != "NA":
            errors.append("hosts.%s.compute_class is null and the host is not "
                          "marked `status: NA`. A host with no class and no "
                          "decision is a cell nothing can ever fill." % name)
        else:
            for field in ("reason", "decided_by", "date"):
                if not host.get(field):
                    errors.append("hosts.%s is NA and names no `%s`. NA is a "
                                  "DECISION; without its author and its date it "
                                  "is an omission wearing a status." % (name, field))
    elif klass not in CLASSES:
        errors.append("hosts.%s.compute_class=%r is not one of %s"
                      % (name, klass, list(CLASSES)))
    reach = host.get("reachable_by")
    if not isinstance(reach, list) or not reach:
        errors.append("hosts.%s declares no `reachable_by:` -- the list of cargo "
                      "features that can take its class. `metal` sat here for "
                      "months while no build in this tree had a Metal path, and "
                      "nothing could say so." % name)
    elif klass is not None and klass not in reach:
        errors.append("hosts.%s.compute_class=%s is not in its own "
                      "reachable_by=%s" % (name, klass, reach))
    if "ci_runner" not in host:
        errors.append("hosts.%s declares no `ci_runner:` (a label, or `none`). "
                      "Whether a host is reachable from CI decides which phase "
                      "its cells can be measured in." % name)

# ---- R4, R5 ----------------------------------------------------------------
def walk(node, path, covered):
    """Numbers must be covered by a threshold_class + author, inherited."""
    if isinstance(node, dict):
        here = covered
        klass, author = node.get("threshold_class"), node.get("author")
        if klass is not None or author is not None:
            if klass not in THRESHOLD_CLASSES:
                errors.append("%s.threshold_class=%r is not one of %s"
                              % (path, klass, list(THRESHOLD_CLASSES)))
            if not author:
                errors.append("%s declares threshold_class=%r and NO author. The "
                              "file's own grounding rule says a policy number is "
                              "a deliberate product decision; a decision with no "
                              "decider is a number somebody typed." % (path, klass))
            here = bool(klass in THRESHOLD_CLASSES and author)
        inherited = node.get("inherited_from")
        if inherited and not os.path.exists(os.path.join(root, str(inherited))):
            errors.append("%s.inherited_from=%r does not exist in the tree. A "
                          "precedence claim pointing at a document nobody can "
                          "open is not precedence." % (path, inherited))
        authors = node.get("authors") or {}
        for key, value in node.items():
            if key in ("threshold_class", "author", "authors", "inherited_from"):
                continue
            child = "%s.%s" % (path, key)
            # A per-key author counts only when it NAMES someone (a non-empty
            # string) and the node declares a legal threshold_class: an empty
            # entry, or a map of authors under an undeclared class, is the
            # unauthored number wearing an `authors:` key.
            entry = authors.get(key) if isinstance(authors, dict) else None
            authored = (isinstance(entry, str) and entry.strip() != ""
                        and klass in THRESHOLD_CLASSES)
            if is_number(value) and not here and not authored:
                errors.append("%s = %r is a number on the policy surface with no "
                              "threshold_class/author above it (PP-33)" % (child, value))
            walk(value, child, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            if is_number(value) and not covered:
                errors.append("%s[%d] = %r is a number on the policy surface with "
                              "no threshold_class/author above it (PP-33)"
                              % (path, i, value))
            walk(value, "%s[%d]" % (path, i), covered)


for key in POLICY_ROOTS:
    if key in m:
        walk(m[key], key, False)

# ---- R8 --------------------------------------------------------------------
for key in ("protocol", "ladder"):
    if not isinstance(m.get(key), dict):
        errors.append("`%s:` block is absent. The gate and both producers read "
                      "it; without it each falls back to its own copy, which is "
                      "how one ladder came to be written in five files." % key)
if not isinstance((m.get("ladder") or {}).get("declared"), list):
    errors.append("ladder.declared is not a list")

# ---- R6, R7 ----------------------------------------------------------------
anchors = m.get("expiry_anchors") or {}
for host, cells in sorted((m.get("baselines") or {}).items()):
    for workload, cell in sorted((cells or {}).items()):
        where = "baselines.%s.%s" % (host, workload)
        cell = cell or {}
        status = cell.get("status")
        if status not in CELL_STATUSES:
            errors.append("%s.status=%r is not one of %s"
                          % (where, status, list(CELL_STATUSES)))
            continue
        if status == "MEASURED":
            for field in ("receipt", "commit", "n", "interleaved", "bands"):
                if cell.get(field) is None:
                    errors.append("%s is MEASURED and names no `%s`. A ratchet "
                                  "seed with no receipt behind it is the "
                                  "fabricated-baseline class this file exists to "
                                  "remove." % (where, field))
            receipt = cell.get("receipt")
            if receipt and not os.path.exists(os.path.join(root, str(receipt))):
                errors.append("%s.receipt=%r is not in the tree" % (where, receipt))
        elif status == "NA":
            for field in ("reason", "decided_by", "date"):
                if not cell.get(field):
                    errors.append("%s is NA and names no `%s`. NA is permanent "
                                  "and excluded from the denominator, so it needs "
                                  "an author the way a policy number does."
                                  % (where, field))
        else:
            if not cell.get("owner"):
                errors.append("%s is UNMEASURED and names no owner" % where)
            fixed, cond = cell.get("expires"), cell.get("expires_after")
            if bool(fixed) == bool(cond):
                errors.append("%s declares %s clock(s). An UNMEASURED cell needs "
                              "EXACTLY one deadline: none never expires, two is "
                              "no clock at all." % (where, "two" if fixed else "no"))
            if isinstance(cond, dict):
                anchor = cond.get("anchor")
                if anchor not in anchors:
                    errors.append("%s.expires_after.anchor=%r is not declared "
                                  "under `expiry_anchors:`" % (where, anchor))
                days = cond.get("days")
                if not isinstance(days, int) or isinstance(days, bool) or days < 0:
                    errors.append("%s.expires_after.days=%r is not a non-negative "
                                  "integer" % (where, days))

for name, anchor in sorted(anchors.items()):
    anchor = anchor or {}
    if not anchor.get("owner"):
        errors.append("expiry_anchors.%s names no owner" % name)
    if anchor.get("status") == "merged" and not anchor.get("merged_on"):
        errors.append("expiry_anchors.%s says merged and records no merged_on; a "
                      "clock cannot start from null" % name)

# ---- R9 --------------------------------------------------------------------
for i, entry in enumerate(m.get("cell_exceptions") or []):
    entry = entry or {}
    comparator = entry.get("comparator")
    if comparator is not None and comparator not in declared_comparators:
        errors.append("cell_exceptions[%d].comparator=%r is declared by no host, "
                      "so this exception can never match a receipt and has never "
                      "been read" % (i, comparator))
    if entry.get("status") == "NOT_APPLICABLE" and not entry.get("decided_by"):
        errors.append("cell_exceptions[%d] is NOT_APPLICABLE and names no "
                      "decided_by" % i)

if errors:
    print("FAIL  perf-matrix.yaml schema")
    for e in errors:
        print("      " + e)
    sys.exit(1)
print("ok    perf-matrix.yaml: %d host(s), %d arm(s), %d baseline cell(s), "
      "%d anchor(s)" % (len(hosts), len(m.get("arms") or {}),
                        sum(len(c or {}) for c in (m.get("baselines") or {}).values()),
                        len(anchors)))
PY
}

# ------------------------------------------------------------- case table ---
selftest() {
  local pass=0 fail=0 td
  td="$(mktemp -d)" || return 2
  case "$td" in
    /tmp/*|/var/folders/*) : ;;
    *) printf 'refusing to rm -rf %s\n' "${td:-<empty>}" >&2; return 2 ;;
  esac
  trap 'rm -rf "${td:?}"' RETURN

  _variant() { # name, python-edit-body -> path to the mutated matrix
    python3 - "$ROOT/scripts/perf-matrix.yaml" "$td/$1.yaml" "$2" <<'PY_MX'
import sys
src, dst, edit = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding="utf-8") as fh:
    s = fh.read()
old, new = edit.split("\x1f")
assert old in s, ("the matrix no longer has the shape this row was written "
                  "against: %r" % old[:60])
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(s.replace(old, new, 1))
print(dst)
PY_MX
  }

  _expect() { # name, matrix(empty=committed), expected(pass|fail)
    local got
    if run_check "$ROOT" "$2" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$3" ]; then
      printf '  ok    %-38s expect=%s\n' "$1" "$3"; pass=$((pass + 1))
    else
      printf '  BROKE %-38s expected %s got %s\n' "$1" "$3" "$got"; fail=$((fail + 1))
    fi
  }

  # The matrix as it stands. If this is not green the rest proves nothing.
  _expect matrix_clean_tree "" pass

  # MUTATION 1 -- a policy number loses its author. This is B1's 0.80 verbatim:
  # `threshold_class: policy` beside no author, in a file whose own header says
  # a policy number is a deliberate decision.
  _expect matrix_policy_without_author \
    "$(_variant policy_no_author "$(printf 'live_ttft_over_e2e_max: 0.95\n  threshold_class: policy\n  author: spec-owner')"$'\x1f'"$(printf 'live_ttft_over_e2e_max: 0.95\n  threshold_class: policy')")" fail

  # MUTATION 2 -- a precedence claim pointing at a document that never existed.
  # B2 carried `inherited_from: docs/specifications/perf-parity-spec.md`, a path
  # with no history in any ref.
  # MUTATION 1b -- the `authors:` map names a key with an EMPTY author. Listing
  # the key used to exempt the number from both requirements at once.
  _expect matrix_authors_entry_empty \
    "$(_variant authors_empty "$(printf '  author: spec-owner\n  authors: {min_agree_tokens: spec-owner, max_constant_run: spec-owner, max_age_days: perf-gate}')"$'\x1f'"$(printf "  authors: {min_agree_tokens: '', max_constant_run: spec-owner, max_age_days: perf-gate}")")" fail

  _expect matrix_inherited_from_missing \
    "$(_variant inherited_missing "inherited_from: docs/archive/perf-2026-09-01/APR-PERF-GATE-001-v2.2.md"$'\x1f'"inherited_from: docs/specifications/perf-parity-spec.md")" fail

  # MUTATION 3 -- a compute class outside the vocabulary. The live instance was
  # `metal`, which IS in the vocabulary and which no build reaches; the R3
  # `reachable_by` rule is what catches that one, and this row keeps the
  # vocabulary itself honest.
  _expect matrix_unknown_class \
    "$(_variant unknown_class "$(printf 'accelerator: rtx-4090\n    compute_class: cuda')"$'\x1f'"$(printf 'accelerator: rtx-4090\n    compute_class: rocm')")" fail

  # MUTATION 4 -- NA without a decider. NA is permanent and excluded from the
  # denominator, so an anonymous one is the cheapest way to delete a cell.
  _expect matrix_na_without_decided_by \
    "$(_variant na_anon "W1: {status: NA, reason: 'no Metal inference path (#2841)', decided_by: spec-owner, date: '2026-09-02'}"$'\x1f'"W1: {status: NA, reason: 'no Metal inference path (#2841)', date: '2026-09-02'}")" fail

  # MUTATION 5 -- a MEASURED cell whose receipt is not in the tree. A ratchet
  # seeded from a file nobody can open is a number somebody typed.
  _expect matrix_measured_receipt_missing \
    "$(_variant measured_ghost "$(printf 'W1:\n      status: UNMEASURED\n      owner: perf-gate')"$'\x1f'"$(printf 'W1:\n      status: MEASURED\n      receipt: evidence/nope/receipt.r1.json\n      commit: deadbeef\n      n: 5\n      interleaved: true\n      bands: {c1: {agg: 1.0}}\n      owner: perf-gate')")" fail

  # MUTATION 6 -- a host loses `reachable_by`, which is the rule that would have
  # caught `metal`.
  _expect matrix_host_without_reachable_by \
    "$(_variant no_reach "$(printf 'compute_class: cuda\n    comparator: llamacpp\n    reachable_by: [cuda]\n    ci_runner: none')"$'\x1f'"$(printf 'compute_class: cuda\n    comparator: llamacpp\n    ci_runner: none')")" fail

  # MUTATION 7 -- an exception naming a comparator no host declares. The live
  # instance was `vllm`, and it had never been read.
  _expect matrix_exception_unknown_comparator \
    "$(_variant ghost_cmp "$(printf '  comparator: llamacpp\n  arm: L3')"$'\x1f'"$(printf '  comparator: vllm\n  arm: L3')")" fail

  printf '  %d passed, %d broken\n' "$pass" "$fail"
  [ "$fail" = 0 ]
}

case "${1:-}" in
  --selftest|--self-test) selftest ;;
  "") run_check "$ROOT" ;;
  *) printf 'usage: %s [--selftest]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
