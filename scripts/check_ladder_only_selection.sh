#!/usr/bin/env bash
# check_ladder_only_selection.sh — `--only` must refuse an id that names nothing (#3936).
#
# WHY. `--only <rung-id>` exists to settle a flake by re-running ONE rung: #3936 spent
# four GPU measurements and a human establishing that one red serve cell was transient,
# and a single-rung re-run makes that one command. Which means the failure mode that
# matters is not "it measured the wrong rung" — it is:
#
#     a targeted run that matches NOTHING, measures zero rungs, and exits 0.
#
# That is the empty-universe vacuity, and it is the worst possible outcome for this
# flag specifically: the whole point is to read the result as a verdict about one
# rung, and a green that means "nothing ran" looks exactly like a green that means
# "the rung is fine". So the refusal is the property under test, not the selection.
#
# WHAT THIS CHECKS, IN TWO LAYERS.
#
#   1. THE DECISION. `only_universe` and `only_selected` are LIFTED FROM
#      model_ladder.sh and run, so this exercises the shipped code rather than a copy
#      of its rules. The table pins exact-match semantics: an id that is a PREFIX of a
#      real id, or a SUBSTRING of one, must not select — because `grep -qxF` and
#      `grep -qF` differ by one letter and the second silently selects the wrong rung.
#
#   2. THE WIRING. A correct decision that nothing calls is the third vacuity shape.
#      So: the refusal must exist and must sit BEFORE the measurement loops, both
#      loops must carry the skip, and the receipt path must be RECEIPT_BASE — a
#      targeted one-row receipt overwriting the sweep receipt would destroy the
#      evidence it was run to produce.
#
# Exit: 0 all cases as expected · 1 a case landed wrong · 2 could not check.
#       --self-test: 0 when each planted mutation turns this RED, 1 otherwise.
set -euo pipefail

SCRIPT="scripts/model_ladder.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_only_selection: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# ── layer 1: the DECISION ────────────────────────────────────────────────────
load_selection() { # -> the two functions, lifted
  local src="$1" fn body out=""
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  for fn in only_universe only_selected; do
    body=$(awk -v F="^$fn\\\\(\\\\) \\\\{" '$0 ~ F {f=1} f{print} f && /^\}$/{exit}' "$src")
    [ -n "$body" ] || { echo "  $src defines no $fn() — --only's decision is gone" >&2; return 2; }
    out="$out$body
"
  done
  printf '%s' "$out"
}

# A universe with the shapes that matter: two ids where one is a PREFIX of the other,
# and an inventory file whose name contains a rung id as a substring.
RUNGS_FIXTURE='qwen35-27b-q4km|Qwen3.5-27B-Q4_K_M.gguf|aa|cpu,cuda|1|
qwen35-2b-q4km|Qwen3.5-2B-Q4_K_M.gguf|bb|cpu,cuda|1|'
INV_FIXTURE='qwen2.5-coder-7b-instruct-q4_k_m.gguf|/m/qwen2.5-coder-7b-instruct-q4_k_m.gguf
qwen35-27b-q4km-extra.gguf|/m/qwen35-27b-q4km-extra.gguf'

selected() { # selected <src> <id> -> "yes" | "no"
  local src="$1" tmp rc
  tmp=$(mktemp) || return 2
  load_selection "$src" > "$tmp" || { rm -f "$tmp"; return 2; }
  cat >> "$tmp" <<'DRV'
if only_selected "$1" "$2" "$3"; then echo yes; else echo no; fi
DRV
  bash "$tmp" "$2" "$RUNGS_FIXTURE" "$INV_FIXTURE"; rc=$?
  rm -f "$tmp"
  return $rc
}

# id|expect
selection_cases() {
cat <<'CASES'
a-ladder-rung-id|qwen35-27b-q4km|yes
the-other-rung|qwen35-2b-q4km|yes
an-inventory-id|inv:qwen2.5-coder-7b-instruct-q4_k_m.gguf|yes
unknown-id-refuses|no-such-rung|no
a-PREFIX-of-a-real-id|qwen35-27b|no
a-real-id-without-the-inv-prefix|qwen2.5-coder-7b-instruct-q4_k_m.gguf|no
an-id-that-is-a-SUBSTRING-of-another|q4km|no
the-empty-string|__EMPTY__|no
CASES
}

run_selection_table() { # -> 0 all as expected
  local src="$1" rc=0 name id want got
  while IFS='|' read -r name id want; do
    [ -n "$name" ] || continue
    [ "$id" = "__EMPTY__" ] && id=""
    got=$(selected "$src" "$id") || return 2
    if [ "$got" = "$want" ]; then
      printf '  ok    %-40s %s\n' "$name" "$got"
    else
      printf '  FAIL  %-40s %s, expected %s\n' "$name" "$got" "$want"; rc=1
    fi
  done < <(selection_cases)
  return $rc
}

# ── layer 2: the WIRING ──────────────────────────────────────────────────────
# name|expect|grep-args...   (expect: present = must match, absent = must not)
check_wiring() { # -> 0 ok
  local src="$1" rc=0 n

  n=$(grep -c 'only_selected "$ONLY"' "$src" || true)
  if [ "${n:-0}" -ge 1 ]; then printf '  ok    %-40s %s\n' "refusal-calls-the-decision" "$n"
  else printf '  FAIL  %-40s the refusal does not call only_selected — the decision is dead code\n' "refusal-calls-the-decision"; rc=1; fi

  # the refusal must come BEFORE the measurement loops, or it refuses after spending
  local ref loop
  ref=$(grep -n "decline: --only" "$src" | head -1 | cut -d: -f1 || true)
  loop=$(grep -n "^# ---- 1. the ladder's rungs" "$src" | head -1 | cut -d: -f1 || true)
  if [ -n "$ref" ] && [ -n "$loop" ] && [ "$ref" -lt "$loop" ]; then
    printf '  ok    %-40s line %s < %s\n' "refusal-precedes-measurement" "$ref" "$loop"
  else
    printf '  FAIL  %-40s refusal at %s, first loop at %s\n' "refusal-precedes-measurement" "${ref:-none}" "${loop:-none}"; rc=1
  fi

  # Count the SPECIFIC skips, not the pattern: a bare count of `[ -z "$ONLY" ] ||`
  # is 3 in the shipped script, because the RECEIPT_BASE line matches it too. So a
  # `>= 2` check passes with one loop skip missing and the receipt line standing in
  # for it — this check's own first version had exactly that hole.
  local rung_skip=0 inv_skip=0
  grep -qF '[ "$rid" = "$ONLY" ] || continue' "$src" && rung_skip=1
  grep -qF '[ "inv:$ifile" = "$ONLY" ] || continue' "$src" && inv_skip=1
  if [ "$rung_skip" = 1 ] && [ "$inv_skip" = 1 ]; then
    printf '  ok    %-40s rung+inventory\n' "both-loops-carry-the-skip"
  else
    printf '  FAIL  %-40s rung=%s inventory=%s — the loop without it measures the whole universe\n' \
      "both-loops-carry-the-skip" "$rung_skip" "$inv_skip"; rc=1
  fi

  # A targeted one-row receipt must never land on the sweep receipt's path.
  if grep -q 'OUT_DIR/\$RECEIPT_BASE\.json' "$src" && grep -q 'RECEIPT_BASE="\$HOST.only-' "$src"; then
    printf '  ok    %-40s separate path\n' "targeted-receipt-is-a-separate-file"
  else
    printf '  FAIL  %-40s a targeted run would overwrite the sweep receipt\n' "targeted-receipt-is-a-separate-file"; rc=1
  fi
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  m1=$(mktemp); m2=$(mktemp); m3=$(mktemp); m4=$(mktemp)
  trap 'rm -f "$m1" "$m2" "$m3" "$m4"' EXIT

  echo "self-test: the shipped script"
  run_selection_table "$SCRIPT" > /dev/null && check_wiring "$SCRIPT" > /dev/null \
    || { echo "SELF-TEST FAILED: the shipped script is already red" >&2; exit 1; }
  echo "  GREEN (expected)"

  # Mutant 1: exact match becomes substring match — one letter, and `--only q4km`
  # would then select a rung nobody asked for while looking like it worked.
  sed 's/grep -qxF -- "\$1"/grep -qF -- "$1"/' "$SCRIPT" > "$m1"
  cmp -s "$SCRIPT" "$m1" && { echo "SELF-TEST INCONCLUSIVE: mutant 1 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 1 (exact match becomes substring match)"
  if run_selection_table "$m1" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 1 passed — a substring now selects a rung" >&2; exit 1
  fi
  echo "  RED (expected)"

  # Mutant 2: the refusal is removed. An unmatched id then measures zero rungs and
  # exits 0 — the empty-universe vacuity this guard exists for.
  sed 's/^  if ! only_selected "\$ONLY" "\$RUNGS" "\$INVENTORY"; then$/  if false; then/' "$SCRIPT" > "$m2"
  cmp -s "$SCRIPT" "$m2" && { echo "SELF-TEST INCONCLUSIVE: mutant 2 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 2 (the refusal never fires)"
  if check_wiring "$m2" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 2 passed — an id matching nothing would measure nothing and exit 0" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 3: the targeted receipt lands on the sweep receipt's path, destroying the
  # evidence the re-run exists to check against.
  sed 's|OUT_DIR/\$RECEIPT_BASE\.json|OUT_DIR/$HOST.json|' "$SCRIPT" > "$m3"
  cmp -s "$SCRIPT" "$m3" && { echo "SELF-TEST INCONCLUSIVE: mutant 3 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 3 (targeted receipt overwrites the sweep receipt)"
  if check_wiring "$m3" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 3 passed — a one-row receipt could replace a sweep" >&2; exit 1
  fi
  echo "  RED (expected)"

  # Mutant 4: ONE loop loses its skip. The first version of the wiring check counted
  # occurrences of the pattern and would have passed this, because the RECEIPT_BASE
  # line matches the same pattern and kept the count at 2.
  sed '/\[ "inv:\$ifile" = "\$ONLY" \] || continue/d' "$SCRIPT" > "$m4"
  cmp -s "$SCRIPT" "$m4" && { echo "SELF-TEST INCONCLUSIVE: mutant 4 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 4 (the inventory loop loses its --only skip)"
  if check_wiring "$m4" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 4 passed — a targeted run would measure every inventory model" >&2
    exit 1
  fi
  echo "  RED (expected)"

  echo "self-test: PASS — red when a substring selects, when the refusal cannot fire, when a targeted receipt overwrites the sweep, and when a loop loses its skip"
  exit 0
fi

echo "ladder --only selection: an id that names nothing must be refused, not measured as zero ($SCRIPT)"
rc=0
run_selection_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
check_wiring "$SCRIPT" || rc=1
if [ "$rc" = 0 ]; then
  echo "OK: --only selects exactly, refuses before measuring, and writes its own receipt"
  exit 0
fi
echo "FAIL: --only could measure nothing and report success (#3936)"
exit 1
