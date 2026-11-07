#!/usr/bin/env bash
# falsify_run_t2_lambda.sh -- self-test for scripts/release/run_t2_lambda.sh (never touches lambda; every run is --check-only
# or --test-run against dummy processes on a scratch RAID_ROOT).
#   falsify_run_t2_lambda.sh [WRAPPER]    WRAPPER defaults to the sibling run_t2_lambda.sh. Pointing it at the old
#                                         hardcoded 0.72.4 host-local copy must FAIL (the RED case).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:-$HERE/run_t2_lambda.sh}"
BASE="${T2_SELFTEST_BASE:-/mnt/nvme-raid0/tmp}"   # RAID_ROOT must not sit on the root filesystem
[ -d "$BASE" ] || { echo "SKIP: scratch base $BASE missing (set T2_SELFTEST_BASE to a non-root filesystem)"; exit 0; }
T="$(mktemp -d "$BASE/falsify-t2-lambda.XXXXXX")"
trap 'rm -rf -- "${T:?}"' EXIT
PASS=0; BAD=0
ok()  { echo "ok   - $1"; PASS=$((PASS+1)); }
nok() { echo "FAIL - $1"; BAD=$((BAD+1)); }

V="$(printf '%d.%d.%d' 9 8 7)"; C="$(printf '%040d' 0 | tr 0 a)"   # synthetic identity, built not literal
export RAID_ROOT="$T/raid"; mkdir -p "$RAID_ROOT" "$T/raid-root-fs-probe"
export T2_LOAD1=1 T2_MEMAVAIL_KB=$((128*1024*1024)) T2_NVSMI_CMD=true FLOOR_GB_TEST=1
# real swap use drifts between the wrapper two samples on a live host; a negative offset absorbs that noise in the baseline
export T2_SWAP_GROWTH_KB=$((-1024*1024))

# 1. parameterised, not hardcoded: no release literal and no 40-hex sha in the wrapper source
if grep -qE '0\.72\.[0-9]+|[0-9a-f]{40}' "$W"; then nok "wrapper hardcodes a version or sha"; else ok "no hardcoded version or sha"; fi
if grep -qE 'EPIC="?[0-9]{3,}' "$W"; then nok "wrapper hardcodes an epic number"; else ok "no hardcoded epic"; fi
grep -qE '^FLOOR_GB=210;' "$W" && ok "disk floor 210G kept" || nok "disk floor 210G missing"
grep -qE 'MEM_MIN_KB=\$\(\(64\*1024\*1024\)\)' "$W" && ok "MemAvailable 64GiB gate kept" || nok "64GiB gate missing"
grep -qE '^set -euo pipefail' "$W" && ok "set -euo pipefail" || nok "set -euo pipefail missing"
if grep -qiE -- '--(force|override|skip[a-z-]*|no-check|ignore[a-z-]*)\)' "$W"; then nok "override-style flag present"; else ok "no override flag"; fi

# 2. arguments: version and commit are required and validated
out="$("$W" --check-only 2>&1)" && nok "no args accepted" || ok "no version/commit refused"
"$W" "$V" notasha --check-only >/dev/null 2>&1 && nok "bad sha accepted" || ok "bad sha refused"
"$W" bad.version "$C" --check-only >/dev/null 2>&1 && nok "bad version accepted" || ok "bad version refused"

# 3. check-only: healthy host passes and creates nothing; each gate violation is caught
out="$("$W" "$V" "$C" --check-only 2>&1)" && ok "baseline check-only passes" || nok "baseline should pass: $out"
[ -z "$(ls -A "$RAID_ROOT")" ] && ok "check-only creates nothing" || nok "check-only created files"
expect_fail() {
  local name="$1" pat="$2"; shift 2
  local out rc=0
  out="$(env "$@" "$W" "$V" "$C" --check-only 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ] && grep -q -- "$pat" <<<"$out"; then ok "violation caught: $name"
  else nok "violation NOT caught: $name (rc=$rc) :: $out"; fi
}
expect_fail "free below 210G floor" "< floor" FLOOR_GB_TEST=99999999
expect_fail "MemAvailable below 64GiB" "MemAvailable" T2_MEMAVAIL_KB=$((60*1024*1024))
expect_fail "MemAvailable exactly 64GiB" "MemAvailable" T2_MEMAVAIL_KB=$((64*1024*1024))
expect_fail "load1 high" "load1" T2_LOAD1=45
expect_fail "GPU busy" "GPU busy" T2_NVSMI_CMD="echo 4242"
expect_fail "nvidia-smi errors" "nvidia-smi failed" T2_NVSMI_CMD="exit 9"
expect_fail "swap growth" "swap use growing" T2_SWAP_GROWTH_KB=$((16*1024*1024))
expect_fail "TMPDIR off raid" "not on raid" T2_TMPDIR=/nonexistent/t2-planted
expect_fail "RAID_ROOT missing" "RAID_ROOT .* missing" RAID_ROOT="$T/nope"
expect_fail "wrong REPO_ROOT" "REPO_ROOT" REPO_ROOT="$T/other"
expect_fail "POLL_S > 60" "POLL_S" POLL_S_TEST=120
# the real floors cannot be lowered in --run (overrides inert): a 1G floor override must not rescue a failing host
out="$(env FLOOR_GB_TEST=1 T2_MEMAVAIL_KB=1 RELEASE_EPIC=1 "$W" "$V" "$C" --run 2>&1)" && nok "--run honoured override" || ok "--run ignores test overrides"
grep -q 'MemAvailable 1kB' <<<"$out" && nok "--run used T2_MEMAVAIL_KB" || ok "--run reads real MemAvailable"

# 4. --test-run: version and commit reach the child; group is cleaned up
EPIC_T="$(printf '%d' 7)"
export RECEIPT_DIR="$T/receipts" OUT="$T/out"
printf '#!/usr/bin/env bash\necho "ver=%s sha=%s epic=%s" > "$OUT"\n' "$V" "$C" "$EPIC_T" > "$T/child.sh"; chmod +x "$T/child.sh"
RELEASE_EPIC="$EPIC_T" "$W" "$V" "$C" --test-run -- "$T/child.sh" >/dev/null 2>&1 && ok "test-run exits 0" || nok "test-run failed"
grep -q "^ver=$V sha=$C epic=$EPIC_T" "$OUT" && ok "child ran with version, commit, epic" || nok "child env wrong: $(cat "$OUT" 2>&1)"
ls "$RECEIPT_DIR"/t2-"$V"-*.verdict >/dev/null 2>&1 && ok "receipt names the version" || nok "receipt not versioned"
[ -z "$(find "$RAID_ROOT" -maxdepth 1 -name 't2-*' ! -name 't2-lambda-receipts')" ] && ok "run dirs cleaned" || nok "run dirs left"

echo "== $PASS passed, $BAD failed =="
[ "$BAD" -eq 0 ]
