#!/usr/bin/env bash
# cuda_rust_fleet_check.sh — is NVIDIA's CUDA Rust stack (cuda-core / cutile) working on
# THIS GPU host, and does aprender-gpu's --features cuda suite pass here?  PMAT-1095.
#
# Emits ONE receipt per host, evidence/cuda-rust-fleet/<host>.json, with a verdict that
# has three states on purpose:
#   PASS        every probe PASSed
#   INCOMPLETE  no probe FAILed, but at least one was SKIPped for a host PREREQUISITE
#               (driver below R580, toolkit below 13.1, no clang resource dir, no checkout).
#               A prerequisite is recorded, never counted as a pass: "flawless on three
#               hosts" must not be reachable by skipping the probes that would fail.
#   FAIL        any probe FAILed, OR zero probes ran (vacuity), OR a probe reported a
#               status outside {PASS,FAIL,SKIP} (a blacklist is fail-open on its
#               complement; the verdict requires membership, not absence of "FAIL").
#
# Usage:
#   scripts/cuda_rust_fleet_check.sh --local [--repo DIR] [--out DIR]
#   scripts/cuda_rust_fleet_check.sh --host gx10 [--repo-remote DIR] [--out DIR]
#   scripts/cuda_rust_fleet_check.sh --self-test
#
# Exit status is the verdict: 0 PASS, 2 INCOMPLETE, 1 FAIL. Every `$?` is captured into a
# variable on the line it is produced — a later command substitution on the same line
# would consume it (measured: a check printed OK on a failing test that way, 2026-09-09).
set -euo pipefail

MODE=""; HOST=""; REPO="${REPO_ROOT:-}"; REPO_REMOTE=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --local) MODE=local;; --host) MODE=host; HOST="$2"; shift;; --self-test) MODE=selftest;;
    --repo) REPO="$2"; shift;; --repo-remote) REPO_REMOTE="$2"; shift;; --out) OUT="$2"; shift;;
    *) echo "unknown arg: $1" >&2; exit 64;;
  esac; shift
done
[ -n "$MODE" ] || { echo "usage: --local | --host H | --self-test" >&2; exit 64; }

# ---------------------------------------------------------------- verdict (pure) ----
# stdin: one status per line, each "PASS" | "FAIL" | "SKIP" (a SKIP may carry ":reason").
verdict() {
  local n=0 pass=0 fail=0 skip=0 bad=0 s
  while IFS= read -r s; do
    [ -n "$s" ] || continue; n=$((n+1))
    case "${s%%:*}" in PASS) pass=$((pass+1));; FAIL) fail=$((fail+1));; SKIP) skip=$((skip+1));; *) bad=$((bad+1));; esac
  done
  if [ "$n" -eq 0 ] || [ "$bad" -gt 0 ] || [ "$fail" -gt 0 ]; then echo FAIL; return; fi
  if [ "$skip" -gt 0 ]; then echo INCOMPLETE; return; fi
  echo PASS
}

write_receipt() {
python3 - "$OUT/$HOSTN.json" "$HOSTN" "$NOW" "$gpu" "$cc" "$driver" "${tk:-}" "$V" "$(sha256sum "$0" | cut -c1-16)" "$memavail_gb" "${IDS[@]}" "${STATUS[@]}" "${REASON[@]}" "${CMDS[@]}" <<'PY'
import json,sys
a=sys.argv[1:]; path,host,now,gpu,cc,drv,tk,verdict,sha,mem=a[:10]; rest=a[10:]; n=len(rest)//4
ids,st,rs,cm=rest[:n],rest[n:2*n],rest[2*n:3*n],rest[3*n:]
probes=[{"id":i,"status":s.split(":")[0],"reason":r,"cmd":c} for i,s,r,c in zip(ids,st,rs,cm)]
blocked=[p["id"]+": "+p["reason"] for p in probes if p["status"]=="SKIP" and not p["reason"].startswith("prerequisite:")]  # ROOT causes only; a probe skipped BECAUSE of another is not a second blocker
json.dump({"host":host,"utc":now,"gpu":gpu,"compute_cap":cc,"driver":drv,"toolkit_max":tk,"script_sha256_16":sha,"mem_available_gib":int(mem or 0),"blocked_on":blocked,
           "probes":probes,"verdict":verdict},open(path,"w"),indent=2)
PY
}

# stdin-free, pure: is this MemAvailable (GiB) enough to run a one-crate build + one test process?
# Empty or non-numeric is NOT enough — a guard that cannot read its input must refuse, not fall through.
mem_floor_ok() {
  if ! printf '%s' "${1:-}" | grep -qE '^[0-9]+$'; then return 1; fi
  [ "$1" -ge 12 ]
}

# ------------------------------------------------------------------- self-test -----
if [ "$MODE" = selftest ]; then
  fail=0
  case_() { local want="$1" got; got=$(printf '%s\n' "${@:2}" | verdict); if [ "$got" = "$want" ]; then echo "OK   $want <- [${*:2}]"; else echo "FAIL want=$want got=$got <- [${*:2}]"; fail=1; fi; }
  case_ PASS       PASS PASS PASS
  case_ FAIL       PASS FAIL PASS
  case_ INCOMPLETE PASS "SKIP:driver 570 < R580" PASS
  case_ FAIL                                   # zero probes: vacuity is RED
  case_ FAIL       PASS pass PASS              # lowercase is NOT a status; fail-open guard
  # blank lines are ignored by design, so [PASS '' PASS] is two PASSes:
  got=$(printf 'PASS\n\nPASS\n' | verdict); [ "$got" = PASS ] && echo "OK   PASS <- [PASS '' PASS] (blank lines ignored)" || { echo "FAIL blank-line handling: $got"; fail=1; }
  case_ FAIL       "SKIP:x" FAIL               # SKIP never masks a FAIL
  # the receipt writer must refuse an unknown status too
  if printf 'PASS\nMAYBE\n' | verdict | grep -qx FAIL; then echo "OK   FAIL <- [PASS MAYBE] (unknown status is RED)"; else echo "FAIL unknown status not RED"; fail=1; fi
  # the RECEIPT WRITER, not just the verdict function: synthetic probes through the real python
  # arrays cannot ride a command's env prefix (they arrive as the literal string "(a b)" —
  # measured by this very test), so set them as real assignments inside a subshell
  tmpd=$(mktemp -d)
  ( OUT="$tmpd"; HOSTN=selftest; NOW=now; gpu=g; cc=8.9; driver=595; tk=13.3; memavail_gb=7; V=INCOMPLETE
    IDS=(a b); STATUS=(PASS "SKIP:driver 570 < R580"); REASON=("ok" "driver 570 < R580"); CMDS=(x y)
    write_receipt )
  if python3 - "$tmpd/selftest.json" <<'PY2'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["verdict"]=="INCOMPLETE", d
assert [p["status"] for p in d["probes"]]==["PASS","SKIP"], d
assert d["blocked_on"]==["b: driver 570 < R580"], d
assert d["mem_available_gib"]==7 and isinstance(d["mem_available_gib"],int), d
PY2
  then echo "OK   receipt writer: INCOMPLETE + blocked_on + typed mem field"; else echo "FAIL receipt writer"; fail=1; fi
  [ -n "$tmpd" ] && [ "$tmpd" != / ] && [ -d "$tmpd" ] && rm -rf "$tmpd"
  for c in "" "x" "0" "5" "11"; do
    if mem_floor_ok "$c"; then echo "FAIL mem_floor_ok('$c') must REFUSE"; fail=1; else echo "OK   refuse mem_floor_ok('$c')"; fi
  done
  for c in "12" "20" "115"; do
    if mem_floor_ok "$c"; then echo "OK   allow  mem_floor_ok('$c')"; else echo "FAIL mem_floor_ok('$c') must allow"; fail=1; fi
  done
  [ "$fail" -eq 0 ] && { echo "SELF-TEST PASS"; exit 0; } || { echo "SELF-TEST FAIL"; exit 1; }
fi

# ---------------------------------------------------------------- remote mode ------
HERE=$(cd "$(dirname "$0")/.." && pwd)
if [ "$MODE" = host ]; then
  OUT="${OUT:-$HERE/evidence/cuda-rust-fleet}"; mkdir -p "$OUT"
  # Keepalive: a cold `cargo test --features cuda` on the far side runs for minutes with no
  # output, and a silent ssh session was measured dropping mid-run ("client_loop: send
  # disconnect: Broken pipe") — losing the verdict. The remote also tees to a log so a
  # dropped session cannot lose what was printed.
  SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
  $SSH "$HOST" 'mkdir -p ~/.cuda-rust-fleet/scripts ~/.cuda-rust-fleet/experiments ~/.cuda-rust-fleet/out'
  rsync -a --delete --exclude target --exclude Cargo.lock "$HERE/experiments/cuda-rust-probes" "$HOST:~/.cuda-rust-fleet/experiments/"
  rsync -a "$HERE/scripts/cuda_rust_fleet_check.sh" "$HOST:~/.cuda-rust-fleet/scripts/"
  rc=0; $SSH "$HOST" "REPO_ROOT='$REPO_REMOTE' bash ~/.cuda-rust-fleet/scripts/cuda_rust_fleet_check.sh --local --out ~/.cuda-rust-fleet/out 2>&1 | tee ~/.cuda-rust-fleet/out/run.log; exit \${PIPESTATUS[0]}" || rc=$?
  rsync -a "$HOST:~/.cuda-rust-fleet/out/" "$OUT/"
  echo "receipt: $OUT/$(ssh -o BatchMode=yes "$HOST" hostname).json  exit=$rc"
  exit "$rc"
fi

# ----------------------------------------------------------------- local mode ------
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
OUT="${OUT:-$HERE/evidence/cuda-rust-fleet}"; mkdir -p "$OUT"
HOSTN=$(hostname); NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PROBES=$(cd "$HERE/experiments/cuda-rust-probes" 2>/dev/null && pwd || true)
[ -n "$PROBES" ] || PROBES="$HOME/.cuda-rust-fleet/experiments/cuda-rust-probes"
declare -a IDS STATUS REASON CMDS
add() { IDS+=("$1"); STATUS+=("$2"); REASON+=("$3"); CMDS+=("$4"); printf '  %-22s %-10s %s\n' "$1" "$2" "$3"; }

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
tk=""; for d in /usr/local/cuda-13.*; do [ -x "$d/bin/nvcc" ] && v=$("$d/bin/nvcc" --version | grep -oE 'release [0-9.]+' | awk '{print $2}') && tk=$(printf '%s\n%s\n' "$tk" "$v" | sort -V | tail -1); done
echo "host=$HOSTN gpu='$gpu' cc=$cc driver=$driver toolkit_max=${tk:-none}"

# P1 driver floor (cuda-bindings refuses < R580 at run time; measured: "built against 13.0 but runtime is 12.8")
if [ -z "$driver" ]; then add driver_floor SKIP "no nvidia-smi / no driver" "nvidia-smi"
elif [ "${driver%%.*}" -ge 580 ]; then add driver_floor PASS "driver $driver >= R580" "nvidia-smi"
else add driver_floor SKIP "driver $driver < R580 (cuda-bindings runtime floor); needs an R580+ driver package AND a reboot to activate it -- installing one live replaces the userspace libs under the running module and breaks CUDA until then (measured 2026-09-09)" "nvidia-smi"; fi
# P2 toolkit floor (cutile: 13.1+ for sm_100+, 13.2+ for sm_8x)
need=13.1; case "$cc" in 8.*) need=13.2;; esac
if [ -z "$tk" ]; then add toolkit_floor SKIP "no CUDA 13.x toolkit under /usr/local" "ls /usr/local/cuda-13.*"
elif [ "$(printf '%s\n%s\n' "$need" "$tk" | sort -V | head -1)" = "$need" ]; then add toolkit_floor PASS "toolkit $tk >= $need for sm_${cc/./}" "nvcc --version"
else add toolkit_floor SKIP "toolkit $tk < $need for sm_${cc/./} (cutile floor)" "nvcc --version"; fi
# P3 clang resource dir (bindgen needs clang's own stddef.h; a bare libclang is not enough — measured
# on yoga). Discover like cuda-oxide does: bare `clang`, then versioned clang-N on PATH and under
# /usr/lib/llvm-N/bin (gx10 has only clang-21, and a non-interactive ssh PATH has no llvm bin dir).
rd=""; CL=""
for c in clang clang-23 clang-22 clang-21 clang-20 clang-19 clang-18 clang-17 clang-16 clang-15 clang-14; do
  for b in "$c" /usr/lib/llvm-*/bin/"$c"; do
    if command -v "$b" >/dev/null 2>&1; then r=$("$b" -print-resource-dir 2>/dev/null || true); if [ -n "$r" ] && [ -f "$r/include/stddef.h" ]; then rd="$r"; CL="$b"; break 2; fi; fi
  done
done
if [ -n "$rd" ]; then
  add clang_resource_dir PASS "$CL -> $rd" "$CL -print-resource-dir"
  export PATH="$(dirname "$(command -v "$CL")"):$PATH"   # so bindgen/clang-sys see the same clang
else add clang_resource_dir SKIP "no clang resource dir (bindgen: 'stddef.h' file not found); apt install clang" "clang -print-resource-dir"; fi
p1=${STATUS[0]}; p2=${STATUS[1]}; p3=${STATUS[2]}

run_probe() { # id dir prereq-ok(0/1) reason-if-skipped
  local id="$1" dir="$2" ok="$3" why="$4" rc log
  if [ "$ok" != 1 ]; then add "$id" SKIP "$why" "-"; return; fi
  log="$OUT/$HOSTN.$id.log"; rc=0
  ( cd "$dir" && CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cuda-rust-fleet/target}" nice -n 19 timeout 1500 cargo run --release ) > "$log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ] && grep -qE 'Could not resolve host|failed to download|spurious network error|Connection refused' "$log"; then
    # ENV death, not a CUDA Rust failure: try the registry cache, then classify as SKIP:env
    # (INCOMPLETE, never PASS) — a guard that names a code cause for a broken box is theater.
    rc=0; ( cd "$dir" && CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cuda-rust-fleet/target}" nice -n 19 timeout 1500 cargo run --release --offline ) > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ] && grep -qE 'Could not resolve host|failed to download|--offline|no matching package' "$log"; then add "$id" SKIP "env: crates.io unreachable and no registry cache (see $HOSTN.$id.log)" "cargo run --release [--offline] ($dir)"; return; fi
  fi
  if [ "$rc" -eq 0 ] && grep -q '^VERDICT: PASS' "$log"; then add "$id" PASS "$(grep -E '^(compute_cap|cutile:)' "$log" | head -1 | tr -s ' ')" "cargo run --release ($dir)"
  else add "$id" FAIL "exit=$rc $(grep -E 'panicked|Error|error(\[|:)|VERDICT|mismatch' "$log" | grep -v '^warning' | tail -1 | cut -c1-140) (see $HOSTN.$id.log)" "cargo run --release ($dir)"; fi
}
ok=0; [ "$p1" = PASS ] && [ "$p3" = PASS ] && ok=1
run_probe cudacore_probe "$PROBES/cudacore" "$ok" "prerequisite: driver_floor=$p1 clang_resource_dir=$p3"
ok=0; [ "$p1" = PASS ] && [ "$p2" = PASS ] && [ "$p3" = PASS ] && ok=1
run_probe cutile_probe "$PROBES/cutile" "$ok" "prerequisite: driver_floor=$p1 toolkit_floor=$p2 clang_resource_dir=$p3"

# P6 aprender-gpu --features cuda: the tests the CUDA Rust work added/repaired.
# RESOURCE GUARD. On 2026-09-09 a cold build+test of this crate on gx10 (the only Blackwell
# CI host) coincided with a global OOM and a reboot. So: never while a CI job runs, never
# below 12 GiB available (a one-crate build plus one test process; 32 was first tried and
# is unreachable on yoga, a laptop with ~31 GiB total), always nice'd with capped build
# jobs, and the TEST PROCESS runs in
# a memory cgroup when systemd-run is available so a regression can kill only itself.
memavail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || true)
memavail_gb=${memavail_gb:-0}   # an unreadable MemAvailable must fail CLOSED: awk prints nothing and exits 0, so `|| echo 0` never fires (measured)
if [ -z "$REPO" ] || [ ! -f "$REPO/crates/aprender-gpu/Cargo.toml" ]; then add aprender_gpu_cuda SKIP "no aprender checkout at '${REPO:-<unset>}' (REPO_ROOT / --repo / --repo-remote)" "-"
elif pgrep -f '[R]unner.Worker' >/dev/null 2>&1; then add aprender_gpu_cuda SKIP "env: a GitHub Actions job is running on this host; not competing with CI" "pgrep Runner.Worker"
elif ! mem_floor_ok "$memavail_gb"; then add aprender_gpu_cuda SKIP "env: MemAvailable '${memavail_gb}' GiB below the 12 GiB floor, or unreadable" "/proc/meminfo"
else
  log="$OUT/$HOSTN.aprender_gpu_cuda.log"; rc=0
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
  bcap=$(( memavail_gb * 60 / 100 )); [ "$bcap" -gt 48 ] && bcap=48; [ "$bcap" -lt 4 ] && bcap=4
  brun=(nice -n 19); command -v systemd-run >/dev/null 2>&1 && brun=(systemd-run --user --scope -p "MemoryMax=${bcap}G" -p MemorySwapMax=0 --quiet -- nice -n 19)
  ( cd "$REPO" && "${brun[@]}" timeout 2400 cargo test -p aprender-gpu --features cuda --lib --no-run ) > "$log" 2>&1 || rc=$?
  # a CI job may have started during the build (TOCTOU on the earlier pgrep): check again before running
  if [ "$rc" -eq 0 ] && pgrep -f '[R]unner.Worker' >/dev/null 2>&1; then rc=0; add aprender_gpu_cuda SKIP "env: a GitHub Actions job started during the build; not competing with CI" "pgrep Runner.Worker"; fi
  if [ "$rc" -eq 0 ] && [ "${STATUS[${#STATUS[@]}-1]}" = SKIP ] && [ "${IDS[${#IDS[@]}-1]}" = aprender_gpu_cuda ]; then :; else
  if [ "$rc" -ne 0 ]; then add aprender_gpu_cuda FAIL "build exit=$rc $(grep -E '^error' "$log" | head -1 | cut -c1-120) (see $HOSTN.aprender_gpu_cuda.log)" "cargo test --no-run"; else
    # cargo prints the executable path RELATIVE to the manifest dir when CARGO_TARGET_DIR is
    # unset (measured on gx10: `target/debug/deps/trueno_gpu-…`) and absolute when it is set
    # (lambda-vector, which is why this bug hid there). Resolve against $REPO either way.
    bin=$(grep -oE 'Executable unittests src/lib.rs \(([^)]+)\)' "$log" | sed -E 's/.*\((.*)\)/\1/' | head -1 || true)   # unmatched => empty, NOT a set -e abort
    case "$bin" in /*) ;; *) bin="$REPO/$bin";; esac
    # MemoryMax = min(48 GiB, 60% of MemAvailable): a cap above physical memory protects nothing
    # (yoga has ~31 GiB total), and 40% headroom is what the host keeps whatever the test does.
    capg=$(( memavail_gb * 60 / 100 )); [ "$capg" -gt 48 ] && capg=48; [ "$capg" -lt 4 ] && capg=4
    runner=(nice -n 19); command -v systemd-run >/dev/null 2>&1 && runner=(systemd-run --user --scope -p "MemoryMax=${capg}G" -p MemorySwapMax=0 --quiet -- nice -n 19)
    if [ ! -x "$bin" ]; then rc=127; echo "test binary not found: '$bin'" >> "$log"; else
    ( cd "$REPO" && "${runner[@]}" "$bin" launch_budget determinism test_alloc_oversize_100gb test_cublas_gemm_f16_training_shape ) >> "$log" 2>&1 || rc=$?
    fi
    line=$(grep -E '^test result:' "$log" | tail -1 || true); passed=$(printf '%s' "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
    if [ "$rc" -eq 0 ] && [ "${passed:-0}" -ge 1 ]; then add aprender_gpu_cuda PASS "$line (MemAvailable ${memavail_gb} GiB, cgroup=$( [ "${runner[0]}" = systemd-run ] && echo "${capg}G" || echo none))" "cargo test -p aprender-gpu --features cuda --lib -- <filters>"
    else add aprender_gpu_cuda FAIL "exit=$rc passed=${passed:-0} ${line:-no test result line} (see $HOSTN.aprender_gpu_cuda.log)" "cargo test -p aprender-gpu --features cuda --lib -- <filters>"; fi
  fi
  fi
fi

V=$(printf '%s\n' "${STATUS[@]}" | verdict)
write_receipt
echo "verdict=$V receipt=$OUT/$HOSTN.json"
case "$V" in PASS) exit 0;; INCOMPLETE) exit 2;; *) exit 1;; esac
