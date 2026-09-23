#!/usr/bin/env bash
# check_ladder_write_errors.sh — model_ladder.sh must DECLINE (rc 2, no receipt) when it
# cannot write, never produce a receipt with holes.
#
# WHY. gx10, 0.69.1 final sweep: the root fs filled mid-run. qwen35-27b-q4km's artifacts,
# its row append to $ROWS, and an inventory record's append to $INV_ROWS all failed with
# "No space left on device", and the ladder CONTINUED. The trace was stderr only; the
# receipt it was heading for would have looked complete while missing a rung and a held
# model. The judge reads that receipt.
#
# WHAT THIS CHECKS, AND WHY IT IS NOT A GREP. It lifts the SHIPPED helpers and runs them
# against real failing writes (a file behind /dev/full is a true ENOSPC; a read-only dir is
# a true EACCES), and it runs the SHIPPED model_ladder.sh end to end with a fake `apr`:
#   append-enospc     ladder_append to a /dev/full-backed file   -> exit 2, names the file
#   append-readonly   ladder_append into a read-only dir         -> exit 2
#   append-ok         ladder_append to a normal file             -> exit 0, record present
#   probe-readonly    ladder_disk_probe on a read-only dir       -> fails, says so
#   probe-floor       ladder_disk_probe under an unmeetable free floor -> fails, names it
#   e2e-readonly-out  model_ladder.sh --out <read-only>          -> rc 2, NO receipt
#   e2e-readonly-tmp  model_ladder.sh with TMPDIR read-only      -> rc 2, NO receipt
#   e2e-full-disk     model_ladder.sh under an unmeetable floor  -> rc 2, NO receipt
#   work-tmpfs        ladder_work_preflight on a tmpfs dir (/dev/shm) -> fails, names tmpfs (#4053)
#   work-footprint    ladder_work_preflight, largest held model > free -> fails, names the model (#4053)
#   work-ok           ladder_work_preflight on a disk dir, small model -> passes (#4053)
#   e2e-work-tmpfs    model_ladder.sh with APR_LADDER_WORK_ROOT on tmpfs -> rc 2, NO receipt (#4053)
#   e2e-apr-tmpdir    every apr the ladder runs gets TMPDIR inside the WORK root, never /tmp (#4053)
# --self-test plants the old behaviour (an append that ignores its failure; a probe that
# always passes; a WORK preflight that always passes) and requires each to turn this RED.
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SCRIPT="scripts/model_ladder.sh"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -e /dev/full ] || { echo "  cannot check: no /dev/full" >&2; exit 2; }
[ "$(id -u)" != 0 ] || { echo "  cannot check as root: a read-only dir does not refuse root" >&2; exit 2; }

lift() { # the three helpers, from the shipped script
  local src="$1" out="" fn body
  for fn in ladder_write_decline ladder_disk_probe ladder_append ladder_work_preflight; do
    body=$(awk -v F="^$fn\\\\(\\\\) \\\\{" '$0 ~ F {f=1} f{print} f && /^\}$/{exit}' "$src")
    [ -n "$body" ] || { echo "  $src defines no $fn()" >&2; return 2; }
    out="$out$body"$'\n'
  done
  printf '%s' "$out"
}

T=$(mktemp -d) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
case "$T" in /tmp/?*) ;; *) echo "  cannot check: expected a temp dir under /tmp, got '$T'" >&2; exit 2 ;; esac
cleanup() {
  case "${T:-}" in
    /tmp/?*) [ -d "$T" ] && { chmod -R u+w -- "$T" 2>/dev/null; rm -rf -- "$T"; } ;;
  esac
}
trap cleanup EXIT
mkdir -p "$T/ro" "$T/ok" && chmod 555 "$T/ro"
ln -s /dev/full "$T/ok/full.jsonl"

helper() { # <bodies> <snippet> -> rc of the snippet with the helpers defined
  local bodies="$1" snip="$2"
  bash -c "LADDER_MIN_FREE_MB=\${LADDER_MIN_FREE_MB:-1}; ROWS=$T/ok/rows.jsonl; INV_ROWS=$T/ok/inv.jsonl; LADDER_APPENDS_ROWS=0; LADDER_APPENDS_INV=0
$bodies
$snip" > "$T/h.out" 2>&1
}

run_cases() { # <bodies> <ladder-script> -> 0 when every case lands
  local b="$1" ladder="$2" rc=0 r
  ok() { printf '  ok    %s\n' "$1"; }
  bad() { printf '  FAIL  %s: %s\n' "$1" "$2"; rc=1; }

  helper "$b" "ladder_append $T/ok/full.jsonl x; echo reached" ; r=$?
  { [ "$r" = 2 ] && ! grep -q reached "$T/h.out" && grep -q "full.jsonl" "$T/h.out"; } && ok append-enospc || bad append-enospc "rc=$r $(tr '\n' ' ' < "$T/h.out" | cut -c1-80)"
  helper "$b" "ladder_append $T/ro/rows.jsonl x; echo reached" ; r=$?
  { [ "$r" = 2 ] && ! grep -q reached "$T/h.out"; } && ok append-readonly || bad append-readonly "rc=$r"
  : > "$T/ok/rows.jsonl"
  helper "$b" "ladder_append \$ROWS '{\"id\":1}'; echo n=\$LADDER_APPENDS_ROWS" ; r=$?
  { [ "$r" = 0 ] && grep -q "n=1" "$T/h.out" && grep -q '"id":1' "$T/ok/rows.jsonl"; } && ok append-ok || bad append-ok "rc=$r"
  helper "$b" "ladder_disk_probe $T/ro" ; r=$?
  { [ "$r" != 0 ] && grep -q "probe write" "$T/h.out"; } && ok probe-readonly || bad probe-readonly "rc=$r"
  LADDER_MIN_FREE_MB=999999999 helper "$b" "ladder_disk_probe $T/ok" ; r=$?
  { [ "$r" != 0 ] && grep -q "floor" "$T/h.out"; } && ok probe-floor || bad probe-floor "rc=$r"
  # #4053: the WORK preflight -- RAM-backed refused by name; free space sized to the largest cell
  if [ "$(stat -f -c %T /dev/shm 2>/dev/null)" = tmpfs ]; then
    helper "$b" "ladder_work_preflight /dev/shm ''" ; r=$?
    { [ "$r" != 0 ] && grep -q "is on tmpfs" "$T/h.out"; } && ok work-tmpfs || bad work-tmpfs "rc=$r $(cut -c1-80 "$T/h.out")"
  else bad work-tmpfs "no tmpfs /dev/shm to test on"; fi
  # a SPARSE 10 TiB "model" (ext4's max file is 16 TiB): no disk used, but larger than any free space.
  # The fixture is asserted, not assumed -- a silent fallback to an empty file made this row vacuous once.
  truncate -s 10T "$T/ok/huge.gguf" 2>/dev/null
  hs=$(stat -c %s "$T/ok/huge.gguf" 2>/dev/null) || hs=0
  [ "$hs" = 10995116277760 ] || bad work-footprint "could not make the 10 TiB sparse fixture"
  helper "$b" "ladder_work_preflight $T/ok 'huge.gguf|$T/ok/huge.gguf'" ; r=$?
  { [ "$r" != 0 ] && grep -q "largest cell needs" "$T/h.out" && grep -q "huge.gguf" "$T/h.out"; } && ok work-footprint || bad work-footprint "rc=$r $(cut -c1-100 "$T/h.out")"
  printf 'x' > "$T/ok/small.gguf"
  helper "$b" "ladder_work_preflight $T/ok 'small.gguf|$T/ok/small.gguf'" ; r=$?
  [ "$r" = 0 ] && ok work-ok || bad work-ok "rc=$r $(cut -c1-100 "$T/h.out")"

  # end to end: the shipped ladder, a fake apr that only answers --version
  [ -n "$ladder" ] || { echo "  (function-level only)"; return "$rc"; }
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" "${TMPDIR:-unset}" >> "%s/apr-tmpdir.log"\ncase "$1" in --version) echo "apr 0.0.0 (fixture)";; *) exit 0;; esac\n' "$T" > "$T/apr"; chmod +x "$T/apr"
  e2e() { # <name> <env...> -- <out-dir>
    local name="$1"; shift; local out="${@: -1}"; set -- "${@:1:$#-1}"
    env APR_LADDER_WORK_ROOT="$T/ok/work" "$@" DOGFOOD_ALLOW_UNPINNED=1 APR="$T/apr" timeout 120 bash "$ladder" --host lambda --out "$out" > "$T/e.out" 2>&1; r=$?
    if [ "$r" = 2 ] && ! ls "$out"/*.json >/dev/null 2>&1 && grep -q "^decline:" "$T/e.out"; then ok "$name"
    else bad "$name" "rc=$r receipt=$(ls "$out"/*.json 2>/dev/null | head -1) $(grep -m1 decline "$T/e.out" | cut -c1-60)"; fi
  }
  e2e e2e-readonly-out X=1 "$T/ro/receipts"
  mkdir -p "$T/ro-tmp" && chmod 555 "$T/ro-tmp"
  e2e e2e-readonly-tmp APR_LADDER_WORK_ROOT="$T/ro-tmp" "$T/ok/r2"
  e2e e2e-full-disk LADDER_MIN_FREE_MB=999999999 "$T/ok/r3"
  # #4053: every apr a MEASURING ladder runs gets TMPDIR inside the WORK root. One fixture model via
  # the inventory seam, a private GPU lock (never the fleet's /tmp/apr-gpu.lock), and a bound: only
  # the recorded TMPDIRs are judged, not the (red, fake-apr) cell.
  mkdir -p "$T/inv"; printf 'x' > "$T/inv/tiny-fixture-q4_k_m.gguf"; : > "$T/apr-tmpdir.log"
  env MODEL_LADDER_INVENTORY_DIRS="$T/inv" MODEL_LADDER_GPU_LOCK="$T/gpu.lock" APR_LADDER_WORK_ROOT="$T/ok/work" \
      DOGFOOD_ALLOW_UNPINNED=1 APR="$T/apr" timeout 120 bash "$ladder" --host lambda --out "$T/ok/r5" \
      --only inv:tiny-fixture-q4_k_m.gguf > "$T/e.out" 2>&1
  if grep -q . "$T/apr-tmpdir.log" && ! grep -v "^$T/ok/work/" "$T/apr-tmpdir.log" | grep -q .; then ok e2e-apr-tmpdir
  else bad e2e-apr-tmpdir "apr calls=$(grep -c . "$T/apr-tmpdir.log"); outside the work root: $(grep -v "^$T/ok/work/" "$T/apr-tmpdir.log" | cut -d' ' -f1 | sort -u | head -2 | tr '\n' ' ')"; fi
  if [ "$(stat -f -c %T /dev/shm 2>/dev/null)" = tmpfs ]; then
    e2e e2e-work-tmpfs APR_LADDER_WORK_ROOT="/dev/shm/ladder-4053-$$" "$T/ok/r4"
    grep -q "is on tmpfs" "$T/e.out" || bad e2e-work-tmpfs-names-it "$(grep -m1 decline "$T/e.out" | cut -c1-80)"
    rm -rf -- "/dev/shm/ladder-4053-$$"
  fi
  return "$rc"
}

bodies=$(lift "$SCRIPT") || exit 2
if [ "$SELF_TEST" = 1 ]; then
  p1=$(sed 's/ 2>\/dev\/null || ladder_write_decline "could not append a record to \$1"/ 2>\/dev\/null || true  # PLANTED/' <<< "$bodies")
  grep -q PLANTED <<< "$p1" || { echo "  self-test: could not plant the append regression" >&2; exit 2; }
  bash -n <(printf '%s\n' "$p1") || { echo "  self-test: plant 1 does not parse" >&2; exit 2; }
  o1=$(run_cases "$p1" "" 2>&1) || true
  p2=$(awk '/^ladder_disk_probe\(\) \{/{print; print "    return 0  # PLANTED"; next} {print}' <<< "$bodies")
  bash -n <(printf '%s\n' "$p2") || { echo "  self-test: plant 2 does not parse" >&2; exit 2; }
  o2=$(run_cases "$p2" "" 2>&1) || true
  p3=$(awk '/^ladder_work_preflight\(\) \{/{print; print "    return 0  # PLANTED"; next} {print}' <<< "$bodies")
  bash -n <(printf '%s\n' "$p3") || { echo "  self-test: plant 3 does not parse" >&2; exit 2; }
  o3=$(run_cases "$p3" "" 2>&1) || true
  printf '%s\n%s\n%s\n' "$o1" "$o2" "$o3"
  if grep -q "FAIL  append-enospc" <<< "$o1" && grep -q "FAIL  append-readonly" <<< "$o1" \
     && grep -q "FAIL  probe-readonly" <<< "$o2" && grep -q "FAIL  probe-floor" <<< "$o2" \
     && grep -q "FAIL  work-tmpfs" <<< "$o3" && grep -q "FAIL  work-footprint" <<< "$o3"; then
    echo "SELF-TEST OK: an append that ignores its failure, a probe that always passes, and a WORK preflight that always passes each turn this RED"; exit 0
  fi
  echo "SELF-TEST FAIL: a planted regression was not caught"; exit 1
fi
echo "ladder write errors: a write the ladder cannot make is a decline, never a hole ($SCRIPT)"
if run_cases "$bodies" "$SCRIPT"; then echo "PASS"; exit 0; fi
echo "FAIL"; exit 1
