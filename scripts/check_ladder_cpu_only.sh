#!/usr/bin/env bash
# check_ladder_cpu_only.sh — `model_ladder.sh --cpu-only` runs apr with NO GPU visible and WITHOUT the GPU lock (#4520).
#
# WHY. 2026-09-27 23:3xZ: gx10's resident :8091 shadow serve held /tmp/apr-gpu.lock for 4h38m+, so the
# ladder's first apr call waited 1800 s and exited 75 -- no cell on gx10 could run at all, CPU cells
# included. Cop ruling: run the CPU cells without the lock. Skipping a lock is only safe if the call
# CANNOT touch the card, so the proof is the environment apr actually sees, not the flag we passed.
#
# CASES (the SHIPPED apr_locked, lifted from model_ladder.sh, against a lock a planted holder keeps):
#   1. --cpu-only: returns the fake apr's rc 0 within the lock wait, and apr saw CUDA_VISIBLE_DEVICES="".
#   2. control, default mode: the same call is refused with LOCK_BUSY (75) -- the planted lock is real.
#   3. wiring: measure() forces rbackends=cpu under --cpu-only, the flag parses, the receipt says cpu_only.
# --self-test plants an apr_locked whose cpu-only branch never fires and requires this check to go RED.
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SELF_TEST=0
case "${1:-}" in --self-test) SELF_TEST=1 ;; "") ;; *) echo "unknown argument '$1'" >&2; exit 2 ;; esac
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "  cannot check: not in a repository" >&2; exit 2; }
cd "$ROOT" || exit 2
command -v flock >/dev/null && command -v choom >/dev/null || { echo "  cannot check: flock/choom absent" >&2; exit 2; }

T=$(mktemp -d) || exit 2
HOLDER=""
trap '[ -n "$HOLDER" ] && kill "$HOLDER" 2>/dev/null; rm -rf -- "${T:?}"' EXIT
fails=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

line=$(sed -n '/^apr_locked() {/p' scripts/model_ladder.sh)
[ -n "$line" ] || { echo "  cannot check: apr_locked not found in model_ladder.sh" >&2; exit 2; }
if [ "$SELF_TEST" = 1 ]; then   # the mutant: the cpu-only branch never fires
  line=${line//'"${CPU_ONLY:-0}" = 1'/'"${CPU_ONLY:-0}" = never'}
  grep -q 'never' <<< "$line" || { echo "  cannot check: self-test mutation did not apply" >&2; exit 2; }
fi
eval "$line"
declare -F apr_locked >/dev/null || { echo "  cannot check: apr_locked did not load" >&2; exit 2; }

cat > "$T/apr" <<'SH'
#!/usr/bin/env bash
printf 'CVD=[%s]\n' "${CUDA_VISIBLE_DEVICES-unset}"
SH
chmod +x "$T/apr"; APR="$T/apr"
GPU_LOCK="$T/gpu.lock"; LOCK_WAIT=3; LOCK_BUSY=75
unset LADDER_METER CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0   # the caller's env makes a GPU visible: --cpu-only must take it away
flock "$GPU_LOCK" sleep 60 & HOLDER=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do flock -n "$GPU_LOCK" true 2>/dev/null || break; sleep 0.2; done
flock -n "$GPU_LOCK" true 2>/dev/null && { echo "  cannot check: the planted holder never took the lock" >&2; exit 2; }

CPU_ONLY=1; s=$SECONDS; out=$(apr_locked run model.gguf 2>/dev/null); rc=$?; dt=$((SECONDS - s))
if [ "$rc" = 0 ] && [ "$out" = "CVD=[]" ] && [ "$dt" -lt "$LOCK_WAIT" ]; then ok "--cpu-only: ran past a held GPU lock in ${dt}s with no GPU visible ($out)"
else bad "--cpu-only: rc=$rc out='$out' in ${dt}s (want rc 0, CVD=[], under ${LOCK_WAIT}s)"; fi

CPU_ONLY=0; out=$(apr_locked run model.gguf 2>/dev/null); rc=$?
[ "$rc" = "$LOCK_BUSY" ] && ok "control: default mode waits on the held lock and is refused ($rc)" \
                         || bad "control: default mode returned rc=$rc out='$out' -- the planted lock proves nothing"

grep -qE '^\s+\[ "\$\{CPU_ONLY:-0\}" = 1 \] && rbackends=cpu' <(sed -n '/^measure() {/,/^}/p' scripts/model_ladder.sh) \
  && ok "wiring: measure() forces rbackends=cpu under --cpu-only" || bad "wiring: measure() does not force the cpu backend"
grep -q -- '--cpu-only) CPU_ONLY=1' scripts/model_ladder.sh && ok "wiring: --cpu-only parses" || bad "wiring: --cpu-only not parsed"
grep -q '"cpu_only": os.environ.get("LADDER_CPU_ONLY") == "1"' scripts/model_ladder.sh && grep -q 'LADDER_CPU_ONLY="$CPU_ONLY"' scripts/model_ladder.sh \
  && ok "wiring: the receipt records cpu_only" || bad "wiring: the receipt does not say it was cpu-only"

if [ "$SELF_TEST" = 1 ]; then
  [ "$fails" -gt 0 ] && { echo "self-test: the planted lock-taking cpu-only turned this RED ($fails case(s)) -- good"; exit 0; }
  echo "self-test: FAIL -- an apr_locked that ignores --cpu-only passed every case"; exit 1
fi
[ "$fails" = 0 ] && { echo "ladder cpu-only: all cases as expected"; exit 0; }
echo "ladder cpu-only: $fails case(s) wrong"; exit 1
