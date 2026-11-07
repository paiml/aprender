#!/usr/bin/env bash
# Falsifier for merge_host_receipts.sh (REL1 fallback). Must-GO baseline plus must-RED cases:
# sha mismatch, assertion FAIL on one host, allowlist miss, missing x86_64 cuda C14, red on every host,
# declared-count mismatch, missing header sha. Each RED case is a mutation of the baseline.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd) || exit 2
M="$here/merge_host_receipts.sh"
d=$(mktemp -d) || exit 2
cleanup() { case "${d:-}" in ''|/) return 0 ;; *) [ -d "$d" ] && rm -rf -- "${d:?}" ;; esac; }
trap cleanup EXIT
S=$(printf 'a%.0s' $(seq 1 40)); T=$(printf 'b%.0s' $(seq 1 40))
bad=0
ok()  { printf 'ok    %s\n' "$*"; }
nok() { printf 'FAIL  %s\n' "$*"; bad=$((bad + 1)); }

# mk <file> <sha> <row...>: header, declared-count row, rows, VERDICT last (dogfood's real shapes).
mk() {
    local f=$1 sha=$2 r; shift 2
    { printf '# receipt-commit: %s\n' "$sha"
      printf '  [ OK ] dogfood-gates              3 declared gate(s) discovered and all green\n'
      printf '  [ OK ] declared:g1  scripts/check_g1.sh\n'
      printf '  [ OK ] declared:g2  scripts/check_g2.sh\n'
      for r in "$@"; do printf '%s\n' "$r"; done
      printf 'VERDICT: GO (phase pre-publish)\n'; } > "$f"
}
CUDA_OK='  [ OK ] model-parity  C14: 3 manifest model(s) measured over >= 64 positions on x86 with --features cuda'
CUDA_ENV='  [FAIL] model-parity  C14: Feature not enabled: cuda'
run() { bash "$M" --commit "$S" --out "$d/v.out" "$@" > "$d/stdout" 2>&1; echo $?; }
expect() { # name want-rc got-rc
    if [ "$3" = "$2" ]; then ok "$1 -> rc $3"; else nok "$1: rc $3, wanted $2 ($(head -n 1 "$d/stdout"))"; fi
}

# x86: C14 green with cuda, g3 env-red; arm: C14 env-red, g3 green. Each red is covered by the other host.
mk "$d/x86.log" "$S" "$CUDA_OK" '  [FAIL] declared:g3 scripts/check_g3.sh -- model X is not on this host under /m'
mk "$d/arm.log" "$S" "$CUDA_ENV" '  [ OK ] declared:g3  scripts/check_g3.sh'
r=$(run --log x86="$d/x86.log" --log arm="$d/arm.log" --arch x86=x86_64)
expect "baseline: env-only reds covered by the other host" 0 "$r"
if [ "$r" = 0 ]; then
    if grep -q 'receipt=' "$d/v.out" && grep -q 'host=arm' "$d/v.out"; then ok "verdict names receipt path and host per row"; else nok "verdict lacks receipt/host per row"; fi
fi

# must-RED: sha mismatch
mk "$d/arm_b.log" "$T" "$CUDA_ENV" '  [ OK ] declared:g3  scripts/check_g3.sh'
r=$(run --log x86="$d/x86.log" --log arm="$d/arm_b.log" --arch x86=x86_64); expect "sha mismatch" 1 "$r"
if [ ! -e "$d/v.out" ]; then ok "no verdict file on NO-GO"; else nok "verdict file exists after NO-GO"; fi

# must-RED: assertion FAIL on one host (green on the other)
mk "$d/arm_a.log" "$S" "$CUDA_ENV" '  [FAIL] declared:g3  scripts/check_g3.sh exit=1 capability_match FAIL'
mk "$d/x86_ok.log" "$S" "$CUDA_OK" '  [ OK ] declared:g3  scripts/check_g3.sh'
r=$(run --log x86="$d/x86_ok.log" --log arm="$d/arm_a.log" --arch x86=x86_64); expect "assertion FAIL on one host" 1 "$r"

# must-RED: allowlist miss
mk "$d/x86_m.log" "$S" "$CUDA_OK" '  [FAIL] declared:g3 scripts/check_g3.sh exit=1 -- weights differ'
r=$(run --log x86="$d/x86_m.log" --log arm="$d/arm.log" --arch x86=x86_64); expect "allowlist miss" 1 "$r"

# must-RED: red on every host, even with allowlisted text
mk "$d/x86_b.log" "$S" "$CUDA_OK" '  [FAIL] declared:g3 scripts/check_g3.sh -- X is not on this host under /m'
mk "$d/arm_r.log" "$S" "$CUDA_ENV" '  [FAIL] declared:g3 scripts/check_g3.sh -- X is not on this host under /m'
r=$(run --log x86="$d/x86_b.log" --log arm="$d/arm_r.log" --arch x86=x86_64); expect "red on every host" 1 "$r"

# must-RED: missing x86_64 cuda C14 -> refused (2); a ruling naming the commit lifts it, another commit's does not
r=$(run --log x86="$d/x86.log" --log arm="$d/arm.log"); expect "no --arch x86_64 host" 2 "$r"
mk "$d/x86_nc.log" "$S" '  [ OK ] model-parity  C14 on x86 without the flag' '  [ OK ] declared:g3  scripts/check_g3.sh'
r=$(run --log x86="$d/x86_nc.log" --log arm="$d/arm.log" --arch x86=x86_64); expect "x86 model-parity green but not --features cuda" 2 "$r"
printf 'quorum receipt for %s\n' "$S" > "$d/ruling.txt"; printf 'quorum receipt for %s\n' "$T" > "$d/ruling_bad.txt"
r=$(run --log x86="$d/x86_nc.log" --log arm="$d/arm.log" --arch x86=x86_64 --x86-cuda-ruling "$d/ruling.txt"); expect "ruling naming the commit" 0 "$r"
r=$(run --log x86="$d/x86_nc.log" --log arm="$d/arm.log" --arch x86=x86_64 --x86-cuda-ruling "$d/ruling_bad.txt"); expect "ruling naming another commit" 2 "$r"

# must-RED: declared gate count differs
sed 's/3 declared gate/4 declared gate/' "$d/arm.log" > "$d/arm_d.log"
r=$(run --log x86="$d/x86.log" --log arm="$d/arm_d.log" --arch x86=x86_64); expect "declared count differs" 1 "$r"

# no header sha at all -> UNKNOWN, never guessed
grep -v '^# receipt-commit' "$d/arm.log" > "$d/arm_h.log"
r=$(run --log x86="$d/x86.log" --log arm="$d/arm_h.log" --arch x86=x86_64); expect "missing header sha" 2 "$r"

if [ "$bad" -eq 0 ]; then echo "falsify_merge_host_receipts: PASS"; else echo "falsify_merge_host_receipts: $bad FAIL"; exit 1; fi
