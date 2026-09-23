#!/usr/bin/env bash
# check_ladder_short_circuit.sh — a cell apr REFUSES BY NAME is short-circuited, never passed (#4052).
#
# WHY. The refused MoE rung (Qwen3.5-35B-A3B-UD-IQ4_XS, qwen35moe, #3977) took 10.1 min on lambda at
# fc942f6be although `apr run --gpu` refuses it by name at once; its other verbs then failed one by
# one. model_ladder.sh now runs `apr run` first and, where F10's refusal proof holds, records the
# backend's remaining verbs (and `code`) as `not_run: "refused-by-name"`, ran:false.
#
# WHAT THIS CHECKS. The SHIPPED model_ladder.sh, end to end, with a fake `apr` (a private GPU lock,
# a private work root, one fixture model through the inventory seam — nothing touches the fleet):
#   refused-by-name   run --gpu refuses qwen35moe BY NAME, generates nothing -> cuda chat/code/serve
#                     recorded not_run + ran:false, and apr is NEVER asked to run them on the GPU
#   generic-failure   run --gpu fails WITHOUT a by-name refusal -> NO short-circuit (the verbs run)
#   wrong-arch        the refusal names ANOTHER architecture than the file's header -> NO short-circuit
# --self-test plants the two regressions #4052 names and requires each to turn this RED:
#   the refusal-proof precondition dropped (any rc != 0 short-circuits), and not-run mapped to ran:true.
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
[ -f "$SCRIPT" ] || { echo "  cannot check: no $SCRIPT" >&2; exit 2; }
# the repository the SHIPPED ladder belongs to: planted copies live in a temp dir, so the contract
# root is passed explicitly (MODEL_LADDER_ROOT), never inferred from the copy's location
REPO=$(cd "$(dirname "$SCRIPT")/.." && pwd) || { echo "  cannot check: no repo root for $SCRIPT" >&2; exit 2; }

T=$(mktemp -d) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
case "$T" in /tmp/?*) ;; *) echo "  cannot check: expected a temp dir under /tmp, got '$T'" >&2; exit 2 ;; esac
cleanup() {
  case "${T:-}" in
    /tmp/?*) [ -d "$T" ] && rm -rf -- "$T" ;;
  esac
}
trap cleanup EXIT

REFUSAL="error: Not implemented: this build has no CUDA forward for architecture 'qwen35moe': fixture. This is a refusal, not a fallback: nothing was loaded and nothing was generated."
# fake apr <mode>: mode refused | generic | wrongarch. Logs every call's argv to $T/<mode>/apr.log.
make_apr() { # <mode>
  local d="$T/$1"; mkdir -p "$d/inv"; printf 'x' > "$d/inv/tiny-fixture-q4_k_m.gguf"
  cat > "$d/apr" <<APR
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$d/apr.log"
case " \$* " in *" --help "*) echo "  --gpu     use the GPU"; echo "  --no-gpu  CPU only"; exit 0 ;; esac  # flag_for probes these
case "\$1" in
  --version) echo "apr 0.0.0 (fixture)" ;;
  inspect) echo '{"architecture": "qwen35moe"}' ;;
  qa) echo '{"gates": []}' ;;
  run)
    case " \$* " in
      *" --gpu "*)
        echo "verbose: fixture preamble"
        case "$1" in
          refused)   echo "$REFUSAL" >&2; exit 12 ;;
          wrongarch) echo "${REFUSAL//qwen35moe/llama}" >&2; exit 12 ;;
          *)         echo "error: CUDA out of memory" >&2; exit 1 ;;
        esac ;;
      *) echo "Paris" ;;
    esac ;;
  *) exit 1 ;;
esac
APR
  chmod +x "$d/apr"
}

ladder() { # <ladder-script> <mode> -> runs it; the receipt path on stdout
  local d="$T/$2"
  env MODEL_LADDER_ROOT="$REPO" MODEL_LADDER_INVENTORY_DIRS="$d/inv" MODEL_LADDER_GPU_LOCK="$d/gpu.lock" APR_LADDER_WORK_ROOT="$d/work" \
      DOGFOOD_ALLOW_UNPINNED=1 APR="$d/apr" timeout 300 bash "$1" --host lambda --out "$d/out" \
      --only inv:tiny-fixture-q4_k_m.gguf > "$d/ladder.out" 2>&1
  ls "$d"/out/*.json 2>/dev/null | head -1
}

cuda_verbs() { # <receipt> -> "chat=<not_run>/<ran> code=... serve=<not_run>/<probed>" for the cuda backend
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = [r for k in ("rungs", "inventory") for r in d.get(k, []) if isinstance(r, dict) and r.get("backends")]
cu = (rows[0]["backends"].get("cuda") or rows[0]["backends"].get("gpu") or {}) if rows else {}
vb = cu.get("verbs") or {}
f = lambda v, k: "%s/%s" % ((vb.get(v) or {}).get("not_run"), (vb.get(v) or {}).get(k))
print("chat=%s code=%s serve=%s" % (f("chat", "ran"), f("code", "ran"), f("serve", "probed")))
PY
}

run_cases() { # <ladder-script> -> 0 when every case lands
  local l="$1" rc=0 r v
  ok() { printf '  ok    %s\n' "$1"; }
  bad() { printf '  FAIL  %s: %s\n' "$1" "$2"; rc=1; }
  for m in refused generic wrongarch; do rm -rf "${T:?}/$m"; make_apr "$m"; done

  r=$(ladder "$l" refused)
  if [ -z "$r" ]; then bad refused-by-name "no receipt ($(tail -1 "$T/refused/ladder.out"))"
  else
    v=$(cuda_verbs "$r")
    if [ "$v" = "chat=refused-by-name/False code=refused-by-name/False serve=refused-by-name/False" ] \
       && ! grep -qE '^(chat|serve run) .*--gpu|^code ' "$T/refused/apr.log"; then ok refused-by-name
    else bad refused-by-name "verbs [$v]; gpu verb calls: $(grep -cE '^(chat|serve run) .*--gpu|^code ' "$T/refused/apr.log")"; fi
  fi

  for m in generic wrongarch; do
    r=$(ladder "$l" "$m")
    if [ -z "$r" ]; then bad "$m" "no receipt ($(tail -1 "$T/$m/ladder.out"))"; continue; fi
    v=$(cuda_verbs "$r")
    if ! grep -q "refused-by-name" <<< "$v" && grep -q '^code ' "$T/$m/apr.log"; then ok "$m-no-short-circuit"
    else bad "$m-no-short-circuit" "verbs [$v]; code calls: $(grep -c '^code ' "$T/$m/apr.log")"; fi
  done
  return "$rc"
}

if [ "$SELF_TEST" = 1 ]; then
  p1="$T/plant-no-proof.sh"; p2="$T/plant-notrun-is-pass.sh"
  sed 's/^  case "\$1" in cuda|gpu) ;; \*) return 1 ;; esac$/  [ "$2" != 0 ] \&\& return 0  # PLANTED: any failure short-circuits/' "$SCRIPT" > "$p1"
  sed "s/^      chat_ran=false; chat_rc=null; chat_bad=\"\"; chat_notrun=/      chat_ran=true; chat_rc=null; chat_bad=\"\"; chat_notrun=/" "$SCRIPT" > "$p2"
  cmp -s "$SCRIPT" "$p1" && { echo "  self-test: could not plant the no-proof regression" >&2; exit 2; }
  cmp -s "$SCRIPT" "$p2" && { echo "  self-test: could not plant the not-run-is-pass regression" >&2; exit 2; }
  o1=$(run_cases "$p1" 2>&1) || true
  o2=$(run_cases "$p2" 2>&1) || true
  printf '%s\n%s\n' "$o1" "$o2"
  if grep -q "FAIL  generic-no-short-circuit" <<< "$o1" && grep -q "FAIL  refused-by-name" <<< "$o2"; then
    echo "SELF-TEST OK: a short-circuit without the proof, and a not-run recorded as run, each turn this RED"; exit 0
  fi
  echo "SELF-TEST FAIL: a planted regression was not caught"; exit 1
fi
echo "ladder short-circuit: a cell refused by name is not-run, never pass ($SCRIPT)"
if run_cases "$SCRIPT"; then echo "PASS"; exit 0; fi
echo "FAIL"; exit 1
