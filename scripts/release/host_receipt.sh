#!/usr/bin/env bash
# host_receipt.sh -- one host's post-publish receipt, produced BY THE TRAIN (#3731; #3544 item 2).
#
# WHY. check_multiplatform_dogfood.sh reads evidence/dogfood/<version>/<host>.json for every host
# in its matrix, and no train had produced one since 0.65.2: the receipts that exist were made by
# hand, so the post-publish dogfood could never discharge its pre-publish DEFER. This produces the
# receipt ON THE HOST, from what a user gets: `cargo install aprender --version =<v> --locked`
# from crates.io into a scratch root, never a PATH `apr` (a bare `apr` once resolved to a 49-day-old
# copy), then the existing producers for the bench block (bench_host_receipt.sh) and the parity
# block (parity_host_receipt.sh, against the pinned llama.cpp at its canonical location).
#
# IT RUNS FROM A KIT, not from the host's checkout. autopilot.sh ships the release commit's Cargo.toml
# and its whole scripts/ tree to a scratch dir on the host (gx10's own checkout was 24 days stale; the
# producers source helpers beside them, so a hand-kept file list drifts). It is bash-3.2
# safe, because mini runs macOS /bin/bash: no associative arrays, no mapfile, no `${x,,}`; the
# clock, the hashes and the JSON are shell + jq (jq 1.8.2 on all four hosts, measured 2026-09-25;
# python3 left the release path in #4352): the clock falls back to perl because BSD date has no %N,
# and macOS has no /proc and no timeout(1).
#
# THE GPU RULE (cop, 2026-09-21, rule rev 5): on lambda and gx10 every apr/llama measurement runs
# through the ORDERED queue `gpu-q --prio N --`, which serves (priority, arrival) and then runs
# `flock /tmp/apr-gpu.lock choom -n 1000 --` itself (flock alone is not FIFO: a P0 row starved
# 40+ min behind ~15 waiters). --gpu-prio picks N (default 5; the train passes 1, since its receipts
# are release-blocking). A host without gpu-q falls back to the bare flock+choom, which excludes
# correctly but may jump the queue, and unmeasured[] says so. The build runs under neither.
# --gpu-wait (default 3600 s) bounds every queue wait (GPUQ_WAIT). The PARITY run is not wrapped:
# its cpu bands must never hold the lock, so parity_host_receipt.sh takes it PER accel band
# (scripts/lib/gpu_band_lock.sh, <= 20 min a band) with the rule passed in as GPU_BAND_Q. Wrapped
# whole, one parity run held lambda's lock 54+ min at 1% GPU (2026-09-21).
#
# NOTHING IS OMITTED SILENTLY. Every item this host could not prove is named in `unmeasured`, which
# is never empty (the crates.io build carries no `cuda` feature, so the CUDA-lane line is always
# there), and a producer that refuses leaves its reason in `<block>_attempt` rather than a missing
# key. What apr itself reports is kept verbatim: on the published 0.68.2 the default `apr run` chose
# wgpu, rejected it (cosine 0.955 vs CPU) and fell back to CPU on intel, gx10 and mini, and that
# line is in the receipt (generate.fallback_line_verbatim) and in unmeasured[].
#
# usage: host_receipt.sh <version> <host> [--model <gguf>] [--no-bench] [--no-parity] [--gpu-prio 0-9]
#                       [--gpu-wait <seconds>]
#   run from the kit root (or anywhere: it cds to its own ../..)
# writes evidence/dogfood/<version>/<host>.json under the kit root and prints it between
#   ---RECEIPT <host>--- / ---END RECEIPT--- on stdout
# exit: 0 a receipt was written (its content says what passed and what did not)
#       2 usage/ENV: no receipt (no cargo, no jq/curl/iconv/sha256, a bad argument)
set -u
cd "$(dirname "$0")/../.." || exit 2
KIT=$(pwd)

ver=${1:-}; host=${2:-}; shift 2 2>/dev/null
model="${HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"; do_bench=1; do_parity=1; gpu_prio=5; gpu_wait=3600
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model=${2:-}; shift 2 ;;
    --gpu-prio) gpu_prio=${2:-}; shift 2 ;;
    --gpu-wait) gpu_wait=${2:-}; shift 2 ;;
    --no-bench) do_bench=0; shift ;;
    --no-parity) do_parity=0; shift ;;
    *) echo "host_receipt: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
# Both end up in a path, so both are validated as plain tokens (no '/', no '..').
[[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: host_receipt.sh <version> <host> [--model <gguf>] [--no-bench] [--no-parity]" >&2; exit 2; }
[[ $host =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "host_receipt: host '$host' is not a plain host name" >&2; exit 2; }
[[ $gpu_prio =~ ^[0-9]$ ]] || { echo "host_receipt: --gpu-prio must be 0-9, not '$gpu_prio'" >&2; exit 2; }
[[ $gpu_wait =~ ^[0-9]+$ ]] || { echo "host_receipt: --gpu-wait must be whole seconds, not '$gpu_wait'" >&2; exit 2; }
export GPUQ_WAIT="$gpu_wait"
# ---- the clock, the hashes and the JSON: shell + jq (#4352: python3 is out of the release path) ----
# Each replaces a python3 block of the same name, case for case (parity table in #4352). bash-3.2
# safe: no EPOCHREALTIME without a fallback, no mapfile, no associative arrays.
# HR_JQ: python's str() of a scalar, str.splitlines() and str.strip() -- the receipt's text fields
# are built from lines python split and stripped, so the twin splits and strips the same way.
HR_JQ='def pystr: if . == null then "None" elif . == true then "True" elif . == false then "False"
    elif type == "string" then . else tojson end;
def pylines: [splits("\r\n|[\n\r\u000b\u000c\u001c\u001d\u001e\u0085  ]")]
    | if length > 0 and .[-1] == "" then .[:-1] else . end;
def strip: sub("^\\s+"; "") | sub("\\s+$"; "");
def nonblank: [pylines[] | select(strip != "")];
def one: input as $d | if ([inputs] | length) > 0 then error("extra data") else $d end;'
# hr_now_ms -> wall-clock milliseconds: bash 5's EPOCHREALTIME, GNU date's %N, else perl (macOS
# /bin/bash is 3.2 and BSD date has no %N).
hr_now_ms() {
  local t
  if [ -n "${EPOCHREALTIME:-}" ]; then t=${EPOCHREALTIME//[.,]/}; echo $(( 10#$t / 1000 )); return 0; fi
  t=$(date +%s%N 2> /dev/null)
  case $t in
    '' | *[!0-9]*) perl -MTime::HiRes=time -e 'printf "%d\n", time * 1000' ;;
    *) echo $(( t / 1000000 )) ;;
  esac
}
# hr_timeout SECS CMD... -> CMD bounded like subprocess.run(timeout=): timeout(1), else perl's alarm
# (macOS has no timeout(1)).
hr_timeout() {
  local s=$1; shift
  if command -v timeout > /dev/null 2>&1; then timeout "$s" "$@"
  else perl -e '$s = shift; alarm $s; exec { $ARGV[0] } @ARGV or exit 127' "$s" "$@"; fi
}
# hr_strip TEXT -> TEXT without leading/trailing whitespace (str.strip()).
hr_strip() {
  local x=$1
  x="${x#"${x%%[![:space:]]*}"}"; x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "$x"
}
# hr_sh CMD... -> stripped stdout only when CMD is present and SUCCEEDED: nvidia-smi on a host whose
# driver is gone prints its failure text on stdout and exits 9, and that text became intel's
# "accelerator" (measured, 0.68.2 first-green run, 2026-09-21).
hr_sh() {
  local out
  command -v "$1" > /dev/null 2>&1 || return 0
  out=$(hr_timeout 120 "$@" 2> /dev/null) || return 0
  hr_strip "$out"
}
# hr_sh_fail CMD... -> `<rc> <first non-blank output line, 160 chars>` for a command that is present
# and failed; nothing when it is absent or green.
hr_sh_fail() {
  local rc
  command -v "$1" > /dev/null 2>&1 || return 0
  hr_timeout 120 "$@" > "$W/.sh_fail.out" 2> "$W/.sh_fail.err"; rc=$?
  [ "$rc" -ne 0 ] || return 0
  printf '%s ' "$rc"
  cat "$W/.sh_fail.out" "$W/.sh_fail.err" | jq -Rrs "$HR_JQ"'(nonblank | first // "") | strip | .[:160]'
}
# hr_sha256 FILE -> hex digest
hr_sha256() {
  local h
  if command -v sha256sum > /dev/null 2>&1; then h=$(sha256sum < "$1") || return 1
  else h=$(shasum -a 256 < "$1") || return 1; fi
  printf '%s\n' "${h%% *}"
}
# hr_rewrite FILTER [jq args...] -> the receipt, rewritten through FILTER (json.dump indent=2, ASCII)
hr_rewrite() {
  jq -a "${@:2}" "$HR_JQ$1" "$RECEIPT" > "$RECEIPT.tmp" && mv -- "$RECEIPT.tmp" "$RECEIPT"
}
# ---- end shell + jq ----
for t in jq curl iconv; do
  command -v "$t" > /dev/null 2>&1 || { echo "host_receipt: ENV no $t on $host" >&2; exit 2; }
done
command -v sha256sum > /dev/null 2>&1 || command -v shasum > /dev/null 2>&1 \
  || { echo "host_receipt: ENV no sha256sum or shasum on $host" >&2; exit 2; }
case $(hr_now_ms) in '' | *[!0-9]*) echo "host_receipt: ENV no millisecond clock on $host (no bash 5, GNU date or perl)" >&2; exit 2 ;; esac
CARGO="${CARGO_HOME:-$HOME/.cargo}/bin/cargo"
[ -x "$CARGO" ] || { echo "host_receipt: ENV no cargo at $CARGO on $host" >&2; exit 2; }

W="$KIT/.receipt-work"; mkdir -p "$W" || exit 2
: > "$W/unmeasured.txt"
unmeasured() { printf '%s\n' "$*" >> "$W/unmeasured.txt"; }

# the measurement wrapper: the fleet GPU rule on the two CUDA hosts, nothing elsewhere
WRAP=""
case "$host" in
  lambda|gx10)
    if command -v gpu-q > /dev/null 2>&1; then
      WRAP="gpu-q --prio $gpu_prio --"
    elif command -v flock > /dev/null 2>&1 && command -v choom > /dev/null 2>&1; then
      WRAP="flock -w $gpu_wait /tmp/apr-gpu.lock choom -n 1000 --"
      unmeasured "gpu-rule: no gpu-q on $host; the bare flock+choom excluded correctly but may have jumped the ordered queue (rule rev 5)"
    else
      unmeasured "gpu-rule: neither gpu-q nor flock/choom on $host, measurements ran without the fleet GPU lock"
    fi ;;
esac

# timed NAME CMD... -> runs CMD with stdout+stderr to $W/<name>.log; sets T_RC and T_MS (hr_now_ms).
# With TIMED_SPLIT=1, stderr goes to $W/<name>.err instead, so a JSON stdout stays parseable.
timed() {
  local name=$1 t0 rc; shift
  t0=$(hr_now_ms)
  if [ "${TIMED_SPLIT:-0}" = 1 ]; then
    ( "$@" ) > "$W/$name.log" 2> "$W/$name.err"; rc=$?
  else
    ( "$@" ) > "$W/$name.log" 2>&1; rc=$?
  fi
  printf '%s %s\n' "$rc" "$(( $(hr_now_ms) - t0 ))" > "$W/$name.rc"
  read -r T_RC T_MS < "$W/$name.rc"
}

# ---- 1. install, as a user would: crates.io, pinned version, locked, a scratch root -----------
ROOT="$W/root"
# SEC011: validated before rm -rf -- an empty or '/' value must never reach it.
if [ -n "$ROOT" ] && [ "$ROOT" != "/" ]; then rm -rf -- "$ROOT"; fi
timed install "$CARGO" install aprender --version "=$ver" --locked --root "$ROOT"
install_rc=$T_RC; install_ms=$T_MS
APR="$ROOT/bin/apr"
[ "$install_rc" -eq 0 ] && [ -x "$APR" ] || { unmeasured "install: cargo install aprender --version =$ver --locked exited $install_rc"; APR=""; }

# ---- 2. assemble the base receipt (shell + jq: portable, and the JSON is built, not printed) ----
RECEIPT_DIR="$KIT/evidence/dogfood/$ver"
case "$RECEIPT_DIR" in *..*) echo "host_receipt: ENV bad receipt dir" >&2; exit 2 ;; esac
mkdir -p "$RECEIPT_DIR" || exit 2
RECEIPT="$RECEIPT_DIR/$host.json"
# The OS facts come from uname(1), not from python's platform module: one source for the shell and
# the JSON, and the Darwin branch is drivable by the case table (a stubbed uname), which
# os.uname() is not.
os_name=$(uname -s); os_rel=$(uname -r); os_arch=$(uname -m)
# The crates.io sparse-index record for aprender; the case table points it at a file:// fixture.
INDEX_URL="${HOST_RECEIPT_INDEX_URL:-https://index.crates.io/ap/re/aprender}"
# The base receipt, built in shell + jq; the JSON is built by jq, never printed by hand.
hr_base_receipt() {
  local cpu nproc nv name cc drv nvf accelerator driver ms secs rem compiled idx crc pub why cache f
  local measured bin_sha bin_ver dv u="$W/.base.unmeasured"
  # unmeasured.txt's non-empty lines, then this step's own
  jq -Rrs "$HR_JQ"'pylines[] | select(. != "")' < "$W/unmeasured.txt" > "$u" || return 1
  if [ "$os_name" = Darwin ]; then
    cpu=$(hr_sh sysctl -n machdep.cpu.brand_string); nproc=$(hr_sh sysctl -n hw.ncpu)
  else
    cpu=$(hr_sh lscpu | awk '/^Model name/ { sub(/^[^:]*:/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); print; exit }')
    nproc=$(hr_sh nproc)
  fi
  nv=$(hr_sh nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | awk -F, 'NF == 3 { print; exit }')
  nvf=$(hr_sh_fail nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader)
  [ -z "$nvf" ] || printf 'accelerator: nvidia-smi is present but failed (rc %s): %s\n' "${nvf%% *}" "${nvf#* }" >> "$u"
  if [ -n "$nv" ]; then
    IFS=, read -r name cc drv <<< "$nv"
    name=$(hr_strip "$name"); cc=$(hr_strip "$cc"); drv=$(hr_strip "$drv")
    accelerator="$name sm_${cc//./}"; driver=$drv
  elif [ "$os_name" = Darwin ]; then
    accelerator="$cpu (Metal)"; driver="n/a (macOS)"
  else
    accelerator=none; driver=n/a
  fi
  case $nproc in '' | *[!0-9]*) nproc=null ;; *) nproc=$(( 10#$nproc )) ;; esac
  # round(ms / 1000), as python rounds: half to even
  ms=$install_ms; secs=$(( ms / 1000 )); rem=$(( ms % 1000 ))
  if [ "$rem" -gt 500 ] || { [ "$rem" -eq 500 ] && [ $(( secs % 2 )) -eq 1 ]; }; then secs=$(( secs + 1 )); fi
  compiled=$(jq -Rrs "$HR_JQ"'[pylines[] | select(strip | startswith("Compiling "))] | length' < "$W/install.log") || return 1
  # sha256_published: crates.io's own cksum for the .crate (the sparse index record);
  # sha256_measured: the .crate cargo actually downloaded, hashed here. Two fields, never merged.
  idx=$(curl -fsSL --max-time 30 -A aprender-release-train -- "$INDEX_URL" 2> /dev/null); crc=$?
  if [ "$crc" -ne 0 ]; then
    case $crc in 22) why=HTTPError ;; 28) why=TimeoutError ;; *) why=URLError ;; esac
    pub='{"ok":false,"why":"'$why'"}'
  elif ! printf '%s' "$idx" | iconv -f UTF-8 -t UTF-8 > /dev/null 2>&1; then
    pub='{"ok":false,"why":"UnicodeDecodeError"}'
  else
    # python parses every line first, then walks them lazily to the first vers match
    pub=$(printf '%s' "$idx" | jq -Rcs --arg v "$ver" "$HR_JQ"'
        try ([nonblank[] | try fromjson catch error("JSONDecodeError")]
             | reduce .[] as $x ({done: false, c: null};
                 if .done then .
                 elif ($x | type) != "object" then error("AttributeError")
                 elif $x.vers == $v then
                   (if $x | has("cksum") then {done: true, c: $x.cksum} else error("KeyError") end)
                 else . end)
             | {ok: true, c})
        catch {ok: false, why: .}') || return 1
  fi
  jq -rn --argjson p "$pub" '$p | select(.ok | not) | "sha256_published: the crates.io index was not read (\(.why))"' >> "$u"
  if [ "$(jq -rn --argjson p "$pub" '$p.ok and $p.c == null')" = true ] && ! grep -q '^sha256_published' "$u"; then
    printf 'sha256_published: aprender %s is not in the crates.io index\n' "$ver" >> "$u"
  fi
  cache=""
  for f in "${CARGO_HOME-$HOME/.cargo}"/registry/cache/*/"aprender-$ver.crate"; do
    [ -e "$f" ] && cache="$cache$f"$'\n'
  done
  cache=$(printf '%s' "$cache" | LC_ALL=C sort | head -n 1)
  measured=null
  if [ -n "$cache" ]; then measured="\"$(hr_sha256 "$cache")\"" || return 1
  else printf 'sha256_measured: no downloaded aprender-%s.crate in the cargo registry cache\n' "$ver" >> "$u"; fi
  bin_sha=""; bin_ver=""
  printf 'null\n' > "$W/.devices.json"
  if [ -n "$APR" ]; then
    bin_sha=$(hr_sha256 "$APR") || return 1
    bin_ver=$(hr_sh "$APR" --version); bin_ver=${bin_ver%%$'\n'*}
    dv=$(hr_sh "$APR" devices --json)
    if ! printf '%s' "$dv" | jq -an "$HR_JQ"'one' > "$W/.devices.json" 2> /dev/null; then
      printf 'null\n' > "$W/.devices.json"
      printf 'devices: `apr devices --json` did not print JSON\n' >> "$u"
    fi
  fi
  printf '%s\n' "accel lane: the crates.io build carries no \`cuda\` feature (apr-cli default features), so no CUDA lane is measured in this receipt; the CUDA binary is the release asset, which the train's hosts step runs" >> "$u"
  jq -an --arg host "$host" --arg arch "$os_arch" --arg os "$os_name $os_rel" --arg cpu "$cpu" \
      --argjson nproc "$nproc" --arg accelerator "$accelerator" --arg driver "$driver" \
      --arg ver "$ver" --arg date "$(date -u +%Y-%m-%d)" --argjson install_rc "$install_rc" \
      --argjson secs "$secs" --argjson compiled "$compiled" --argjson pub "$pub" \
      --argjson measured "$measured" --arg apr "$APR" --arg bin_sha "$bin_sha" --arg bin_ver "$bin_ver" \
      --slurpfile devices "$W/.devices.json" --rawfile unmeasured "$u" "$HR_JQ"'
    {host: $host, arch: $arch, os: $os,
     cpu: (if $cpu == "" then "unknown" else $cpu end), nproc: $nproc,
     accelerator: $accelerator, driver: $driver,
     version_tested: $ver, date: $date,
     produced_by: "scripts/release/host_receipt.sh (the release train, #3731)",
     provenance: "crates.io", asset: "aprender-\($ver).crate",
     install_command: "cargo install aprender --version =\($ver) --locked --root <scratch>",
     install_rc: $install_rc, install_wall_seconds: $secs, crates_compiled: $compiled,
     sha256_published: (if $pub.ok then $pub.c else null end), sha256_measured: $measured}
    + (if $apr != "" then {binary_path: $apr, binary_sha256: $bin_sha, binary_version_output: $bin_ver,
                           devices: ($devices | .[0])}
       else {binary_path: null, devices: null} end)
    + {unmeasured: ($unmeasured | pylines)}' > "$RECEIPT.tmp" && mv -- "$RECEIPT.tmp" "$RECEIPT"
}
hr_base_receipt
[ -s "$RECEIPT" ] || { echo "host_receipt: ENV the base receipt was not written" >&2; exit 2; }

# receipt_set KEY JSON-FILE|null -> sets receipt[KEY] from a JSON file (or null)
receipt_set() {
  if [ -n "${2:-}" ]; then
    jq -n "$HR_JQ"'one' < "$2" > "$W/.set.json" 2> /dev/null || return 1
    hr_rewrite '.[$k] = ($v | .[0])' --arg k "$1" --slurpfile v "$W/.set.json"
  else
    hr_rewrite '.[$k] = null' --arg k "$1"
  fi
}
note() { printf '%s\n' "$*" >> "$W/unmeasured.txt"; hr_rewrite '.unmeasured = ((.unmeasured // []) + [$u])' --arg u "$*"; }
# attempt NAME RC LOG -> receipt[NAME_attempt] = {status: refused, rc, reason: last lines}
attempt() {
  local tail='[]'
  if [ -r "$3" ]; then tail=$(jq -Rcs "$HR_JQ"'nonblank | .[-3:]' < "$3") || tail='[]'; fi
  hr_rewrite '.[$n + "_attempt"] = {status: "refused", rc: ($rc | tonumber), reason: ($tail | join(" | ") | .[:400])}' \
    --arg n "$1" --arg rc "$2" --argjson tail "$tail"
}

# ---- 3. generate: one short, checkable answer from the installed binary -----------------------
if [ -n "$APR" ] && [ -f "$model" ]; then
  # shellcheck disable=SC2086
  # --format json: the answer, the token count and whether the GPU was USED come from apr's own
  # record on stdout; the human lines (Backend:, a fallback warning) come from stderr, verbatim.
  # (The text output prints no token count: 0.68.2's first-green run recorded tokens=null.)
  # shellcheck disable=SC2086
  TIMED_SPLIT=1 timed generate $WRAP "$APR" run "$model" --prompt "What is 2+2? Answer with just the number." --max-tokens 16 --format json
  model_sha=$(hr_sha256 "$model") || model_sha=""
  # the record is ONE JSON document on stdout, valid UTF-8 (python's json.load refused anything else)
  if iconv -f UTF-8 -t UTF-8 < "$W/generate.log" > /dev/null 2>&1 \
      && jq -n "$HR_JQ"'one' < "$W/generate.log" > "$W/.generate.json" 2> /dev/null; then :
  else printf 'null\n' > "$W/.generate.json"; fi
  hr_rewrite '
    ($errs | nonblank | map(strip)) as $e | (($doc | .[0])) as $out
    | if ($rc | tonumber) != 0 or ($out | type) != "object" then
        .generate = null
        | .unmeasured = ((.unmeasured // []) + ["generate: `apr run --format json` \(if ($rc | tonumber) != 0 then "exited \($rc)" else "printed no JSON record on stdout" end): \((($e | .[-1]) // "")[:160])"])
      else
        (first(($e | .[]) | select(test("fallback|rejected"; "i"))) // null) as $fb
        | .generate = {model: ($model | split("/") | .[-1]), model_sha256: $sha,
                       backend_line_verbatim: ((first(($e | .[]) | select(startswith("Backend:"))) // "")[:300]),
                       fallback_line_verbatim: (if $fb then ($fb | .[:300]) else null end),
                       used_gpu: $out.used_gpu,
                       tokens: $out.tokens_generated, wall_ms: ($ms | tonumber),
                       output_sane: (($out | if has("text") then .text else "" end) | pystr | test("\\b4\\b")),
                       prompt: "What is 2+2? Answer with just the number."}
        | if .generate.output_sane then . else .unmeasured = ((.unmeasured // []) + ["generate: the answer to 2+2 did not contain 4"]) end
        | if $fb then .unmeasured = ((.unmeasured // []) + ["generate: apr'"'"'s accelerated path was not used: \(($fb | .[:200]))"]) else . end
      end' --arg model "$model" --arg sha "$model_sha" --arg rc "$T_RC" --arg ms "$T_MS" \
      --rawfile errs "$W/generate.err" --slurpfile doc "$W/.generate.json"
else
  receipt_set generate
  note "generate: no installed apr or no model at $model"
fi

# ---- 4. bench block: the existing producer, on the installed binary --------------------------
if [ "$do_bench" -eq 1 ] && [ -n "$APR" ] && [ -f "$model" ]; then
  # shellcheck disable=SC2086
  timed bench env APR_UNDER_TEST="$APR" $WRAP bash scripts/bench_host_receipt.sh "$host" "$model"
  [ "$T_RC" -eq 0 ] || { attempt bench "$T_RC" "$W/bench.log"; note "bench: bench_host_receipt.sh refused (rc $T_RC)"; }
else
  note "bench: not run (disabled, or no installed apr/model)"
fi

# ---- 5. parity block: the existing producer, against the pinned llama.cpp ----------------------
if [ "$do_parity" -eq 1 ] && [ -n "$APR" ] && [ -f "$model" ]; then
  pin=$(sed -n 's/^[[:space:]]*build_commit[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' scripts/llama_pin.toml | head -n 1)
  LLAMA_BENCH_PATH="${LLAMA_BENCH_PATH:-$HOME/src/llama.cpp-$pin/build/bin/llama-bench}"
  # shellcheck disable=SC2086
  # NOT wrapped: the GPU rule goes IN, as GPU_BAND_Q, and is applied per accel band (see the header)
  timed parity env LLAMA_BENCH_PATH="$LLAMA_BENCH_PATH" LLAMA_PIN_HOST="$host" GPU_BAND_Q="$WRAP" GPU_BAND_TIMEOUT_S=1200 \
      bash scripts/parity_host_receipt.sh --apr "$APR" --model "$model" --out "$W/parity.json"
  # parity_block.py writes {"parity": <block>}: the BLOCK is what the receipt holds. Stored whole, it
  # nested as parity.parity and the gate read a block with no instrument and no lanes (measured on the
  # 0.68.2 first-green run, lambda). Any other shape is a refusal, never a block.
  if [ "$T_RC" -eq 0 ] && [ -s "$W/parity.json" ] && jq -n "$HR_JQ"'
      def pytype: if type == "array" then "list" elif type == "string" then "str" elif type == "boolean" then "bool"
        elif type == "null" then "NoneType" elif (tostring | test("[.eEn]")) then "float" else "int" end;
      one
      | if type == "object" and keys_unsorted == ["parity"] and (.parity | type) == "object" then .parity
        else "parity_host_receipt.sh --out is not {\"parity\": <block>}: keys \(if type == "object" then "[" + (keys | map("'"'"'" + . + "'"'"'") | join(", ")) + "]" else pytype end)\n" | halt_error(1)
        end' < "$W/parity.json" > "$W/parity.block.json" 2>> "$W/parity.log"
  then
    receipt_set parity "$W/parity.block.json"
  else
    attempt parity "$T_RC" "$W/parity.log"; note "parity: parity_host_receipt.sh refused (rc $T_RC)"
  fi
else
  note "parity: not run (disabled, or no installed apr/model)"
fi

echo "---RECEIPT $host---"
cat "$RECEIPT"
echo "---END RECEIPT---"
exit 0
