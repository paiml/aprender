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
# clock, the hashes and the JSON are python3 (present on all four hosts, measured 2026-09-21),
# because BSD date has no %N and macOS has no /proc.
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
#       2 usage/ENV: no receipt (no cargo, no python3, a bad argument)
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
command -v python3 > /dev/null 2>&1 || { echo "host_receipt: ENV no python3 on $host" >&2; exit 2; }
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

# timed NAME CMD... -> runs CMD with stdout+stderr to $W/<name>.log; sets T_RC and T_MS (python3 clock).
# With TIMED_SPLIT=1, stderr goes to $W/<name>.err instead, so a JSON stdout stays parseable.
timed() {
  local name=$1; shift
  python3 - "$W/$name.log" "$W/$name.rc" "${TIMED_SPLIT:-0}" "$W/$name.err" "$@" <<'PY'
import subprocess, sys, time
log, rcf, split, errf, cmd = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4], sys.argv[5:]
t0 = time.monotonic()
err = open(errf, "wb") if split else subprocess.STDOUT
with open(log, "wb") as out:
    try:
        rc = subprocess.call(cmd, stdout=out, stderr=err)
    except OSError as e:
        out.write(f"exec failed: {e}\n".encode()); rc = 127
open(rcf, "w").write(f"{rc} {int((time.monotonic() - t0) * 1000)}\n")
PY
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

# ---- 2. assemble the base receipt (python3: portable, and the JSON is built, not printed) ------
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
python3 - "$RECEIPT" "$ver" "$host" "$install_rc" "$install_ms" "$APR" "$ROOT" "$W" "$os_name" "$os_rel" "$os_arch" "$INDEX_URL" <<'PY'
import datetime, glob, hashlib, json, os, subprocess, sys, urllib.request
receipt, ver, host, install_rc, install_ms, apr, root, work, os_name, os_rel, os_arch, index_url = sys.argv[1:13]
def sh(cmd):
    # stdout only when the command SUCCEEDED: nvidia-smi on a host whose driver is gone prints its
    # failure text on stdout and exits 9, and that text became intel's "accelerator" (measured,
    # 0.68.2 first-green run, 2026-09-21)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""
def sh_fail(cmd):
    # (rc, first output line) for a command that is present and failed; None when absent or green
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if p.returncode == 0: return None
    first = next((l.strip() for l in (p.stdout + p.stderr).splitlines() if l.strip()), "")
    return p.returncode, first
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
unmeasured = [l for l in open(os.path.join(work, "unmeasured.txt")).read().splitlines() if l]
darwin = os_name == "Darwin"
if darwin:
    cpu = sh(["sysctl", "-n", "machdep.cpu.brand_string"]); nproc = sh(["sysctl", "-n", "hw.ncpu"])
else:
    cpu = next((l.split(":", 1)[1].strip() for l in sh(["lscpu"]).splitlines() if l.startswith("Model name")), "")
    nproc = sh(["nproc"])
NVQ = ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
nv = [l for l in sh(NVQ).splitlines() if len(l.split(",")) == 3]
nv_failed = sh_fail(NVQ)
if nv_failed:
    unmeasured.append(f"accelerator: nvidia-smi is present but failed (rc {nv_failed[0]}): {nv_failed[1][:160]}")
if nv:
    name, cc, drv = (x.strip() for x in (nv[0].split(",") + ["", "", ""])[:3])
    accelerator, driver = f"{name} sm_{cc.replace('.', '')}", drv
elif darwin:
    accelerator, driver = f"{cpu} (Metal)", "n/a (macOS)"
else:
    accelerator, driver = "none", "n/a"
log = open(os.path.join(work, "install.log"), errors="replace").read()
r = {
    "host": host, "arch": os_arch, "os": f"{os_name} {os_rel}",
    "cpu": cpu or "unknown", "nproc": int(nproc) if nproc.isdigit() else None,
    "accelerator": accelerator, "driver": driver,
    "version_tested": ver, "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "produced_by": "scripts/release/host_receipt.sh (the release train, #3731)",
    "provenance": "crates.io", "asset": f"aprender-{ver}.crate",
    "install_command": f"cargo install aprender --version ={ver} --locked --root <scratch>",
    "install_rc": int(install_rc), "install_wall_seconds": round(int(install_ms) / 1000),
    "crates_compiled": sum(1 for l in log.splitlines() if l.strip().startswith("Compiling ")),
}
# sha256_published: crates.io's own cksum for the .crate (the sparse index record);
# sha256_measured: the .crate cargo actually downloaded, hashed here. Two fields, never merged.
try:
    req = urllib.request.Request(index_url, headers={"User-Agent": "aprender-release-train"})
    rec = [json.loads(l) for l in urllib.request.urlopen(req, timeout=30).read().decode().splitlines() if l.strip()]
    r["sha256_published"] = next((x["cksum"] for x in rec if x.get("vers") == ver), None)
except Exception as e:
    r["sha256_published"] = None; unmeasured.append(f"sha256_published: the crates.io index was not read ({type(e).__name__})")
if r["sha256_published"] is None and not any(u.startswith("sha256_published") for u in unmeasured):
    unmeasured.append(f"sha256_published: aprender {ver} is not in the crates.io index")
cache = sorted(glob.glob(os.path.join(os.environ.get("CARGO_HOME", os.path.expanduser("~/.cargo")), "registry", "cache", "*", f"aprender-{ver}.crate")))
r["sha256_measured"] = sha256(cache[0]) if cache else None
if not cache: unmeasured.append(f"sha256_measured: no downloaded aprender-{ver}.crate in the cargo registry cache")
if apr:
    r["binary_path"] = apr; r["binary_sha256"] = sha256(apr)
    r["binary_version_output"] = sh([apr, "--version"]).splitlines()[0] if sh([apr, "--version"]) else ""
    dv = sh([apr, "devices", "--json"])
    try: r["devices"] = json.loads(dv)
    except Exception:
        r["devices"] = None; unmeasured.append("devices: `apr devices --json` did not print JSON")
else:
    r["binary_path"] = None; r["devices"] = None
unmeasured.append("accel lane: the crates.io build carries no `cuda` feature (apr-cli default features), so no CUDA lane is measured in this receipt; the CUDA binary is the release asset, which the train's hosts step runs")
r["unmeasured"] = unmeasured
json.dump(r, open(receipt, "w"), indent=2); open(receipt, "a").write("\n")
PY
[ -s "$RECEIPT" ] || { echo "host_receipt: ENV the base receipt was not written" >&2; exit 2; }

# receipt_set KEY JSON-FILE|null -> sets receipt[KEY] from a JSON file (or null)
receipt_set() {
  python3 - "$RECEIPT" "$1" "${2:-}" <<'PY'
import json, sys
p, key, src = sys.argv[1:4]
r = json.load(open(p))
r[key] = json.load(open(src)) if src else None
json.dump(r, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
}
note() { printf '%s\n' "$*" >> "$W/unmeasured.txt"; python3 - "$RECEIPT" "$*" <<'PY'
import json, sys
p, u = sys.argv[1:3]
r = json.load(open(p)); r.setdefault("unmeasured", []).append(u)
json.dump(r, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
}
# attempt NAME RC LOG -> receipt[NAME_attempt] = {status: refused, rc, reason: last lines}
attempt() {
  python3 - "$RECEIPT" "$1" "$2" "$3" <<'PY'
import json, sys
p, name, rc, log = sys.argv[1:5]
try: tail = [l for l in open(log, errors="replace").read().splitlines() if l.strip()][-3:]
except OSError: tail = []
r = json.load(open(p)); r[name + "_attempt"] = {"status": "refused", "rc": int(rc), "reason": " | ".join(tail)[:400]}
json.dump(r, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
}

# ---- 3. generate: one short, checkable answer from the installed binary -----------------------
if [ -n "$APR" ] && [ -f "$model" ]; then
  # shellcheck disable=SC2086
  # --format json: the answer, the token count and whether the GPU was USED come from apr's own
  # record on stdout; the human lines (Backend:, a fallback warning) come from stderr, verbatim.
  # (The text output prints no token count: 0.68.2's first-green run recorded tokens=null.)
  # shellcheck disable=SC2086
  TIMED_SPLIT=1 timed generate $WRAP "$APR" run "$model" --prompt "What is 2+2? Answer with just the number." --max-tokens 16 --format json
  python3 - "$RECEIPT" "$model" "$W/generate.log" "$W/generate.err" "$T_RC" "$T_MS" <<'PY'
import hashlib, json, re, sys
p, model, log, errf, rc, ms = sys.argv[1:7]
err = [l.strip() for l in open(errf, errors="replace").read().splitlines() if l.strip()]
h = hashlib.sha256()
with open(model, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
r = json.load(open(p))
try:
    out = json.load(open(log))
except ValueError:
    out = None
if int(rc) != 0 or not isinstance(out, dict):
    r["generate"] = None
    why = f"exited {rc}" if int(rc) != 0 else "printed no JSON record on stdout"
    r.setdefault("unmeasured", []).append(f"generate: `apr run --format json` {why}: {(err[-1] if err else '')[:160]}")
else:
    fallback = next((l for l in err if re.search(r"fallback|rejected", l, re.I)), None)
    r["generate"] = {"model": model.rsplit("/", 1)[-1], "model_sha256": h.hexdigest(),
                     "backend_line_verbatim": next((l for l in err if l.startswith("Backend:")), "")[:300],
                     "fallback_line_verbatim": fallback[:300] if fallback else None,
                     "used_gpu": out.get("used_gpu"),
                     "tokens": out.get("tokens_generated"), "wall_ms": int(ms),
                     "output_sane": bool(re.search(r"\b4\b", str(out.get("text", "")))),
                     "prompt": "What is 2+2? Answer with just the number."}
    if not r["generate"]["output_sane"]:
        r.setdefault("unmeasured", []).append("generate: the answer to 2+2 did not contain 4")
    if fallback:
        r.setdefault("unmeasured", []).append(f"generate: apr's accelerated path was not used: {fallback[:200]}")
json.dump(r, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
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
  if [ "$T_RC" -eq 0 ] && [ -s "$W/parity.json" ] && python3 - "$W/parity.json" "$W/parity.block.json" >> "$W/parity.log" 2>&1 <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
if not (isinstance(doc, dict) and list(doc) == ["parity"] and isinstance(doc["parity"], dict)):
    print("parity_host_receipt.sh --out is not {\"parity\": <block>}: keys %s" % (sorted(doc) if isinstance(doc, dict) else type(doc).__name__))
    sys.exit(1)
json.dump(doc["parity"], open(sys.argv[2], "w"))
PY
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
