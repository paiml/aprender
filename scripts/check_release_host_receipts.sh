#!/usr/bin/env bash
# check_release_host_receipts.sh -- the release train produces every host's post-publish receipt in
# the schema the multi-platform gate reads, judges them AFTER producing them, fails closed, and never
# reports infrastructure as a host verdict (#3731; #3544 items 1, 2, 4 and 5).
#
# THE DEFECTS, measured on the v0.68.2 train (2026-09-20). (1) No train had produced a host receipt
# since 0.65.2: evidence/dogfood/<version>/<host>.json existed only when made by hand, so the
# post-publish dogfood could never discharge the pre-publish DEFER of check_multiplatform_dogfood.sh.
# (2) autopilot's `install` step ran that dogfood BEFORE `hosts` ran, and only recorded its rc. (3)
# With no $AP/receipts dir, every `> $AP/receipts/...` failed, and four green hosts were reported as
# four failed receipts.
#
# WHAT THIS RUNS
#   part A  the REAL scripts/release/host_receipt.sh in a fixture kit, with stubs for the crates.io
#           install, the installed binary, uname/lscpu/nproc/sysctl/nvidia-smi, flock/choom, the
#           crates.io index (a file:// record) and the two existing block producers.
#   part B  the REAL scripts/release/autopilot.sh through its own from/to steps (`hosts postpub`,
#           `install install`, `postpub postpub`) in a throwaway release repo whose kit IS part A's
#           real host_receipt.sh, an ssh stub that runs each host's command locally with that host's
#           uname/GPU, and a dogfood stub that runs the REAL check_multiplatform_dogfood.sh over the
#           receipts the train just wrote -- so the schema claim is judged by the gate, not by us.
#
# THE ROWS
#   A1 gpu-linux      lambda: every schema field; generate sane; bench and parity blocks; the GPU
#                     rule (rev 5) queues generate and bench through gpu-q at the default priority 5
#                     (flock+choom, 2x each), NOT the install, and NOT the parity run: parity is
#                     handed the rule (GPU_BAND_Q, GPUQ_WAIT 3600, 1200 s a band) and runs unlocked.
#   A2 darwin         mini (uname Darwin): cpu from sysctl, Metal, driver n/a, no GPU wrap.
#   A3 install-fails  the install exits 101: a receipt IS written, install_rc 101, generate null, and
#                     unmeasured names the install, generate, bench and parity.
#   A4 parity-refuses the parity producer refuses: parity_attempt carries its reason; unmeasured names it.
#   A5 insane-answer  2+2 answered 5: output_sane false and unmeasured says so.
#   A6 two-hashes     the index cksum differs from the downloaded .crate: both are recorded, unequal.
#   A7 bad-version    a version that is not x.y.z ("not-a-version"): rc 2 and no receipt.
#   A8 nvsmi-fails    nvidia-smi is present but FAILS, printing its error on stdout (intel, measured
#                     on the 0.68.2 first-green run): accelerator "none", and unmeasured names it.
#   A10 legacy-wrap   a host with no gpu-q: the bare `flock -w 3600` + choom wraps generate and bench,
#                     parity is handed that rule, and unmeasured[] says it may have jumped the queue.
#   A11 bad-prio      --gpu-prio 12: rc 2, no receipt.
#   A9 gpu-fallback   apr's accelerated path is rejected and it falls back (0.68.2 on intel and mini:
#                     "GPU (wgpu) path rejected ... cosine vs CPU = 0.955"): the line is kept verbatim,
#                     used_gpu is apr's own false, and unmeasured names it.
#   B1 train-green    hosts -> postpub: four HOST-RECEIPT lines BEFORE the post-publish dogfood, which
#                     reads $AP/receipts/dogfood (4 receipts); the REAL gate says "ok <host> 9.9.9
#                     verified" AND "ok <host> generate sane, .crate sha256 published = measured" for
#                     all four; mini's receipt is Darwin; "DOGFOOD post-publish GO".
#   B2 dir-removed    $AP/receipts removed BEFORE the step: the step creates it and goes green.
#   B3 dir-vanishes   $AP/receipts removed mid-step (#3544 item 4's falsifier): STOP naming the
#                     missing dir as INFRA -- no "host/installer receipt(s) failed" count.
#   B4 dir-blocked    a FILE where the dir goes: STOP, INFRA "cannot create the receipt dir".
#   B5 unreachable    gx10's ssh exits 255: INFRA lines, "0 host/installer receipt(s) failed besides".
#   B6 env-no-cargo   intel has no Rust toolchain (host_receipt rc 2): INFRA "returned no receipt".
#   B7 postpub-nogo   the post-publish dogfood says NO-GO: STOP.
#   B8 postpub-defer  it exits 0 but its receipt still DEFERS a row: STOP ("still DEFERS").
#   B9 postpub-noreceipt it exits 0 and writes no receipt: STOP.
#   B10 install-only  `install install` never runs the dogfood.
#   B11 postpub-alone `postpub` with no host receipts: STOP, INFRA "the hosts step has not run".
#   L1 ledger-pr      hosts -> ledger: the four receipts and the train record are committed on
#                     `ledger/9.9.9` in origin (evidence/dogfood/9.9.9/<host>.json,
#                     docs/build-ledger/<day>/<sha9>-lambda-vector-train.json) on top of main, and the
#                     PR is opened against main UNARMED (no merge, no --auto).
#   L2 ledger-rerun   `ledger` again: no second PR, and the branch moves by FAST-FORWARD (no force).
#   L3 ledger-empty   `ledger` with no host receipts: STOP, INFRA "nothing to ledger".
#   L4 ledger-rejected origin refuses the push: STOP "the receipts exist only in $AP", before close.
#   B12 report-notes  every bench producer refuses and a fixture REPORT_ONLY lists (9.9.9, <host>,
#                     bench) for all four, so the REAL gate REPORTs "no bench block" with the issue and
#                     the producer's own refusal: those rows are appended to the release notes --
#                     once, even when postpub runs twice.
#   G1 report-listed  the gate, with a fixture REPORT list naming (9.9.9, intel, parity): intel's
#                     absent parity block is a REPORT naming the issue; lambda's (unlisted) FAILs.
#   G2 report-other-v an entry for 9.9.8 is inert at 9.9.9: intel's absent block FAILs.
#   G3 report-present a listed host whose block is PRESENT but invalid still FAILs (INVALID).
#   G4 report-shape   the REAL list: every entry VERSION:HOST:BLOCK:#ISSUE, HOST in the gate's
#                     matrix, BLOCK bench or parity.
#   G13 report-floor  a listed host whose parity block is VALID but a lane is below its floor (the
#                     committed 0.65.2 lambda block: cpu 0.59, cuda 0.69): REPORT; unlisted: FAIL.
#   G14 report-lanes  a listed sm_* host whose valid block lacks the cuda lane (the crates.io build has
#                     no cuda feature, #3805): REPORT naming the missing lane.
#   G10 bench-required an absent bench block FAILs from the train on, naming the recorded refusal.
#   G11 bench-listed  ... unless the list names (9.9.9, intel, bench): then a REPORT with the issue.
#   G12 bench-grandfathered a hand-made pre-#3731 version still REPORTs an absent bench block.
#   G5 gen-null       the gate FAILs a receipt whose generate is null, quoting its recorded reason.
#   G6 gen-insane     ... whose generate.output_sane is false.
#   G7 hash-differs   ... whose downloaded .crate hash is not the published one.
#   G8 unmeasured-empty ... whose unmeasured[] is empty.
#   G9 grandfathered  a hand-made pre-#3731 version (0.65.2) is REPORTed for those blocks, not FAILed.
#   P1 band-lock      scripts/lib/gpu_band_lock.sh on a real flock(1): acquire HOLDS the lock, release
#                     frees it and leaves no holder process.
#   P2 band-bound     a band past GPU_BAND_TIMEOUT_S: the holder kills its registered server, marks it
#                     expired, and the lock is free.
#   P3 band-refused   the queue gives up (gpu-q exit 75): acquire returns 1, the lock was never held.
#   P4 band-no-rule   no GPU rule on the host (GPU_BAND_Q empty): a no-op, the lock untouched.
#   R1 cpu-band       parity_host_receipt.sh's real run_band(): a cpu band runs with the lock FREE.
#   R2 accel-band     an accel band runs with the lock HELD, and it is free again after.
#   R3 accel-refused  the queue gives up: the band FAILS and its body never runs.
#   R4 accel-bound    an accel band past its bound: its server is killed and the band FAILS.
#   T1 twin-equal     scripts/lib/yaml_twin.py: with PyYAML BLOCKED, the real readers (bench_receipt's
#                     matrix, perf_receipt's ledger) load the JSON twins and get the SAME object as the
#                     YAML path, byte-equal after a canonical dump -- mini's python3 has no PyYAML.
#   T2 twin-missing   PyYAML blocked and no twin: an ImportError that names the missing twin.
#   S1 no-jq-e        neither autopilot.sh nor host_receipt.sh carries `jq -e` (#3554).
# THE MUTANTS -- one per claim; each must turn its row RED
#   hr-drop-accel-line   the always-present accel-lane unmeasured line deleted     -> A1
#   hr-parity-nested     the producer's {"parity": block} stored whole             -> A1
#   hr-no-gpu-wrap       WRAP never set                                            -> A1
#   hr-os-literal        the OS read as a literal Linux, not uname                 -> A2
#   hr-drop-attempt      a refused parity leaves no parity_attempt                 -> A4
#   hr-sane-always       output_sane hard-wired true                               -> A5
#   hr-merge-hashes      sha256_measured copied from sha256_published              -> A6
#   hr-no-version-check  the x.y.z validation deleted                              -> A7
#   hr-gpuq-not-preferred gpu-q present but the bare flock used                    -> A1
#   ap-receipt-prio      the train's receipts queued at the default, not prio 1    -> B1
#   hr-nvsmi-any-rc      a failed nvidia-smi's stdout read as a GPU                -> A8
#   hr-drop-fallback     a fallback leaves no unmeasured line                      -> A9
#   hr-version-key       the receipt says "version", not "version_tested"          -> B1
#   ap-steps-reordered   postpub listed before hosts in STEPS                      -> B1
#   ap-no-receipts-env   the dogfood run without DOGFOOD_RECEIPTS_DIR              -> B1
#   ap-hosts-literal     the matrix re-listed without mini, not read from the gate -> B1
#   ap-drop-mkdir        the receipt dir never created                             -> B2
#   ap-no-receipt-to     receipt_to a no-op (the pre-#3731 redirect)               -> B3
#   ap-255-is-a-host     ssh rc 255 counted as a host failure                      -> B5
#   ap-drop-nogo-stop    the post-publish NO-GO `die` deleted                      -> B7
#   ap-drop-receipt-read the post-publish receipt never read                       -> B8
#   ap-install-dogfood   the pre-#3731 dogfood line restored in `install`          -> B10
#   ap-drop-notes        the release-notes edit deleted                            -> B12
#   ap-ledger-no-push    the ledger push deleted                                   -> L1
#   ap-ledger-armed      the ledger PR armed for auto-merge                        -> L1
#   ap-ledger-from-main  a re-run rebuilds the branch from main (needs a force)    -> L2
#   ap-ledger-no-stop    a refused push does not STOP                              -> L4
#   hr-parity-wrapped    the whole parity run wrapped in the GPU rule again     -> A1
#   lock-release-noop    gpu_band_release frees nothing                           -> P1
#   lock-bound-kills-none the holder kills no server at the bound                 -> P2
#   band-lock-any-class  run_band takes the lock for every band, cpu included     -> R1
#   ap-no-twins          the kit built without its JSON twins                     -> B1
#   reader-yaml-only     bench_receipt reads the matrix with PyYAML only           -> T1
#   twin-writes-wrong    the twin writer drops the document                        -> T1
#   gate-any-version     a REPORT entry matched by host alone, any version         -> G2
#   gate-bench-absent-report an absent bench block REPORTed, not FAILed (pre-#3731) -> G10
#   gate-report-present  a listed host's PRESENT block reported, not judged        -> G3
#   gate-floor-any       a listed host's below-floor lane FAILs (entry ignored)   -> G13
#   gate-gen-any         output_sane never read                                    -> G6
#   gate-hash-any        published != measured never compared                      -> G7
#   gate-unmeasured-any  an empty unmeasured[] accepted                            -> G8
#
# Exit 0 = every row green and every mutant killed. 1 = a row RED or a mutant survived.
# 2 = ENV: a subject or a mutation anchor is missing -- the table judged nothing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
AUTOPILOT="$ROOT/scripts/release/autopilot.sh"
HOST_RECEIPT="$ROOT/scripts/release/host_receipt.sh"
PARAMS="$ROOT/scripts/release/lib_release_params.sh"
GATE="$ROOT/scripts/check_multiplatform_dogfood.sh"
VALIDATOR="$ROOT/scripts/lib/bench_receipt.py"
PIN="$ROOT/scripts/llama_pin.toml"
LEDGER="$ROOT/scripts/release/ledger.py"
BANDLOCK="$ROOT/scripts/lib/gpu_band_lock.sh"
PARITY="$ROOT/scripts/parity_host_receipt.sh"
TWIN="$ROOT/scripts/lib/yaml_twin.py"

env_die() { printf 'ENV   %s -- the table judged nothing, not a pass\n' "$*" >&2; exit 2; }
for f in "$AUTOPILOT" "$HOST_RECEIPT" "$PARAMS" "$GATE" "$VALIDATOR" "$PIN" "$LEDGER" "$BANDLOCK" "$PARITY" "$TWIN"; do [ -r "$f" ] || env_die "no $f"; done
for t in git python3 tar; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done
REAL_FLOCK=$(PATH=/usr/bin:/bin:/usr/sbin:/sbin command -v flock) || env_die "no flock(1): the band-lock rows need a real one"
# Rows point HOME at a scratch dir, which would hide a PyYAML installed in the user site-packages
# (lambda's is ~/.local/lib/python3.13): pin the user base to the REAL one first.
PYTHONUSERBASE="${PYTHONUSERBASE:-$(python3 -m site --user-base)}"; export PYTHONUSERBASE

TMP=$(mktemp -d) || exit 2
# SEC011: validate before rm -rf. An empty or '/' value must never reach it.
cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
trap cleanup EXIT

# Hermetic git: no global hooks, signing or identity from the operator's config.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid
export GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid

# ---- mutants: each is a python3 replace that must match its anchor EXACTLY once -----------------
mutate() { # mutate SRC DST OLD NEW -> 0, or ENV naming the anchor
    python3 - "$1" "$2" "$3" "$4" <<'PY' || env_die "mutation anchor not found exactly once in $(basename "$1"): $3"
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src).read()
if s.count(old) != 1: sys.exit(1)
open(dst, "w").write(s.replace(old, new))
PY
}
M="$TMP/mutants"; mkdir -p "$M"
mutate "$HOST_RECEIPT" "$M/hr-drop-accel-line.sh" \
    'unmeasured.append("accel lane: the crates.io build carries no `cuda` feature (apr-cli default features), so no CUDA lane is measured in this receipt; the CUDA binary is the release asset, which the train'"'"'s hosts step runs")' 'pass'
mutate "$HOST_RECEIPT" "$M/hr-parity-nested.sh" \
    '    receipt_set parity "$W/parity.block.json"' '    receipt_set parity "$W/parity.json"'
mutate "$HOST_RECEIPT" "$M/hr-nvsmi-any-rc.sh" \
    '    return p.stdout.strip() if p.returncode == 0 else ""' '    return p.stdout.strip()'
mutate "$HOST_RECEIPT" "$M/hr-drop-fallback.sh" \
    '        r.setdefault("unmeasured", []).append(f"generate: apr'"'"'s accelerated path was not used: {fallback[:200]}")' '        pass'
mutate "$HOST_RECEIPT" "$M/hr-no-gpu-wrap.sh" \
    '      WRAP="gpu-q --prio $gpu_prio --"' '      WRAP=""'
mutate "$HOST_RECEIPT" "$M/hr-gpuq-not-preferred.sh" \
    '    if command -v gpu-q > /dev/null 2>&1; then' '    if false; then'
mutate "$HOST_RECEIPT" "$M/hr-os-literal.sh" \
    'os_name=$(uname -s);' 'os_name=Linux;'
mutate "$HOST_RECEIPT" "$M/hr-drop-attempt.sh" \
    'attempt parity "$T_RC" "$W/parity.log"; ' ''
mutate "$HOST_RECEIPT" "$M/hr-sane-always.sh" \
    '"output_sane": bool(re.search(r"\b4\b", str(out.get("text", "")))),' '"output_sane": True,'
mutate "$HOST_RECEIPT" "$M/hr-merge-hashes.sh" \
    'r["sha256_measured"] = sha256(cache[0]) if cache else None' 'r["sha256_measured"] = r["sha256_published"]'
mutate "$HOST_RECEIPT" "$M/hr-no-version-check.sh" \
    '[[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {' 'true || {'
mutate "$HOST_RECEIPT" "$M/hr-version-key.sh" \
    '"version_tested": ver,' '"version": ver,'
mutate "$AUTOPILOT" "$M/ap-steps-reordered.sh" \
    'install hosts postpub ledger close)' 'install postpub hosts ledger close)'
mutate "$AUTOPILOT" "$M/ap-no-receipts-env.sh" \
    '  DOGFOOD_RECEIPTS_DIR="$AP/receipts/dogfood" bash scripts/dogfood.sh --phase post-publish' '  bash scripts/dogfood.sh --phase post-publish'
mutate "$AUTOPILOT" "$M/ap-hosts-literal.sh" \
    '  rhosts=$(matrix_hosts)' '  rhosts="lambda intel gx10"'
mutate "$AUTOPILOT" "$M/ap-drop-mkdir.sh" \
    '  mkdir -p "$RDIR/dogfood" || die' '  true || die'
mutate "$AUTOPILOT" "$M/ap-no-receipt-to.sh" \
    '  receipt_to() { : > "$1" 2> /dev/null || die' '  receipt_to() { return 0; : > "$1" 2> /dev/null || die'
mutate "$AUTOPILOT" "$M/ap-255-is-a-host.sh" \
    '    if [ "$rc" -eq "$SSH_FAILED" ]; then infra="$infra $h(ssh-255)"; say "INFRA $h unreachable: ssh rc=255 -- not a host verdict"; continue; fi
    say "HOST $h rc=$rc' '    say "HOST $h rc=$rc'
mutate "$AUTOPILOT" "$M/ap-drop-nogo-stop.sh" \
    '  [ $rc -eq 0 ] || die "post-publish dogfood NO-GO rc=$rc' '  true || die "post-publish dogfood NO-GO rc=$rc'
mutate "$AUTOPILOT" "$M/ap-drop-receipt-read.sh" \
    '  ) || die "post-publish dogfood refused on its receipt: $line"' '  ) || true'
mutate "$AUTOPILOT" "$M/ap-drop-notes.sh" \
    '      gh release edit "$T" --repo $REPO --notes-file "$AP/release-notes.md" >> "$LOG" 2>&1' '      true'
mutate "$GATE" "$M/gate-any-version.sh" \
    'case "$e" in "$VERSION:$h:$1:"*) printf '"'"'%s'"'"' "${e##*:}"; return 0 ;; esac' 'case "$e" in *":$h:$1:"*) printf '"'"'%s'"'"' "${e##*:}"; return 0 ;; esac'
mutate "$GATE" "$M/gate-report-present.sh" \
    'if [ -n "$report_only" ] && ! python3 "$REPO_BENCH_VALIDATOR" --has-parity "$f" >/dev/null 2>&1; then' 'if [ -n "$report_only" ]; then'
mutate "$GATE" "$M/gate-bench-absent-report.sh" \
    'printf '"'"'FAIL  %-7s no bench block: %s\n'"'"'' 'printf '"'"'REPORT %-6s no bench block: %s\n'"'"''
mutate "$GATE" "$M/gate-floor-any.sh" \
    '2>/dev/null)" && [ -n "$report_only" ]; then' '2>/dev/null)" && false; then'
mutate "$GATE" "$M/gate-gen-any.sh" \
    'elif g.get("output_sane") is not True:' 'elif False:'
mutate "$GATE" "$M/gate-hash-any.sh" \
    'elif pub != mea:' 'elif False:'
mutate "$GATE" "$M/gate-unmeasured-any.sh" \
    'if not (isinstance(u, list) and u and all(isinstance(x, str) and x for x in u)):' 'if False:'
mutate "$AUTOPILOT" "$M/ap-ledger-no-push.sh" \
    '  git -C "$lw" push -q origin "$lb" >> "$LOG" 2>&1 || die "cannot push $lb: the receipts exist only in $AP"' '  true'
mutate "$AUTOPILOT" "$M/ap-ledger-armed.sh" \
    '  say "LEDGER PR $lpr ($lb)' '  gh pr merge "$lpr" --repo $REPO --auto --squash; say "LEDGER PR $lpr ($lb)'
mutate "$AUTOPILOT" "$M/ap-ledger-from-main.sh" \
    '    lbase="origin/$lb"' '    lbase=origin/main'
mutate "$AUTOPILOT" "$M/ap-ledger-no-stop.sh" \
    '|| die "cannot push $lb: the receipts exist only in $AP"' '|| true'
mutate "$AUTOPILOT" "$M/ap-receipt-prio.sh" \
    '$h"'"'"' --gpu-prio 1; rc=$?' '$h"'"'"'; rc=$?'
mutate "$HOST_RECEIPT" "$M/hr-parity-wrapped.sh" \
    'GPU_BAND_Q="$WRAP" GPU_BAND_TIMEOUT_S=1200' 'GPU_BAND_Q="$WRAP" GPU_BAND_TIMEOUT_S=1200 $WRAP'
mutate "$BANDLOCK" "$M/lock-release-noop.sh" \
    '    [ -z "$holder_pid" ] || kill "$holder_pid" 2> /dev/null || true
    wait "$GPU_BAND_HOLDER" 2> /dev/null || true' '    :'
mutate "$BANDLOCK" "$M/lock-bound-kills-none.sh" \
    '        while read -r p; do [ -n "$p" ] && kill "$p" 2> /dev/null; done < "$3"' '        :'
mutate "$PARITY" "$M/band-lock-any-class.sh" \
    '    if [ "$klass" = accel ] && [ "$DRY_RUN" -eq 0 ]; then
        if ! gpu_band_acquire' '    if [ "$DRY_RUN" -eq 0 ]; then
        if ! gpu_band_acquire'
mutate "$AUTOPILOT" "$M/ap-no-twins.sh" \
    '    && ( cd "$ks" && python3 scripts/lib/yaml_twin.py write scripts/perf-matrix.yaml scripts/perf-receipt-fields.yaml ) >> "$LOG" 2>&1 \' '    \'
mkdir -p "$M/lib-reader" "$M/lib-writer" && cp -- "$ROOT"/scripts/lib/*.py "$M/lib-reader/" && cp -- "$ROOT"/scripts/lib/*.py "$M/lib-writer/" || env_die "cannot copy scripts/lib for the twin mutants"
mutate "$VALIDATOR" "$M/lib-reader/bench_receipt.py" \
    '        _MATRIX_CACHE[path] = yaml_twin.load(path) or {}' '        import yaml; _MATRIX_CACHE[path] = yaml.safe_load(open(path, encoding="utf-8")) or {}'
mutate "$TWIN" "$M/lib-writer/yaml_twin.py" \
    '            json.dump(doc, handle, sort_keys=True, indent=1)' '            json.dump({}, handle, sort_keys=True, indent=1)'
mutate "$AUTOPILOT" "$M/ap-install-dogfood.sh" \
    '  [ $rc -eq 0 ] || die "post-publish install failed"
fi' '  [ $rc -eq 0 ] || die "post-publish install failed"
  bash scripts/dogfood.sh --phase post-publish > "$AP/dogfood-post-publish.log" 2>&1
fi'

# ---- stubs ---------------------------------------------------------------------------------------
B="$TMP/bin"; mkdir -p "$B"
# the Rust package manager: `metadata` answers the manifest's version; `install --root R` builds a
# fake apr into R/bin and a fake .crate into the registry cache (a row picks the rc)
cat > "$TMP/cargo-stub" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  metadata)
    m=""; prev=""
    for a in "$@"; do [ "$prev" = --manifest-path ] && m=$a; prev=$a; done
    [ -n "$m" ] || m="$PWD/Cargo.toml"
    v=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$m" | head -n 1)
    printf '{"packages":[{"id":"fx","name":"fx","version":"%s","manifest_path":"%s","dependencies":[],"targets":[]}],"workspace_members":["fx"],"resolve":null}\n' "$v" "$m" ;;
  install)
    printf 'install host=%s under_lock=%s %s\n' "${FX_HOST:-local}" "${FX_UNDER_LOCK:-0}" "$*" >> "$FX_STATE/pkg.log"
    root=""; prev=""; for a in "$@"; do [ "$prev" = --root ] && root=$a; prev=$a; done
    [ -n "$root" ] || exit 0
    printf '   Compiling fixture-dep v1.0.0\n   Compiling aprender v9.9.9\n'
    rc=${FX_INSTALL_RC:-0}
    [ "$rc" = 0 ] || { printf 'error: failed to compile `aprender v9.9.9`\n'; exit "$rc"; }
    mkdir -p "$root/bin" "$CARGO_HOME/registry/cache/index.crates.io-fixture" || exit 1
    cp "$FX_APR_STUB" "$root/bin/apr" && chmod +x "$root/bin/apr" || exit 1
    printf 'fixture crate bytes\n' > "$CARGO_HOME/registry/cache/index.crates.io-fixture/aprender-9.9.9.crate"
    printf '  Installed package `aprender v9.9.9` (executable `apr`)\n' ;;
  *) exit 0 ;;
esac
STUB
cat > "$TMP/apr-stub" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  --version) printf 'apr 9.9.9\n' ;;
  devices) printf '{"devices":[{"kind":"cpu","name":"fixture"}]}\n' ;;
  run)
    printf 'Backend: %s\n' "${FX_BACKEND:-CPU (fixture)}" >&2
    [ "${FX_FALLBACK:-0}" = 1 ] && printf 'warning: GPU (wgpu) path rejected, attempting fallback: cosine vs CPU = 0.955376 (< 0.99) at step 2/3\n' >&2
    printf '{"text": "%s", "tokens": [19], "tokens_generated": 16, "used_gpu": %s, "cached": true}\n' "${FX_ANSWER:-4}" "${FX_USED_GPU:-false}" ;;
  *) exit 2 ;;
esac
STUB
cat > "$B/uname" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  -r) printf '%s\n' "${FX_UNAME_R:-6.8.0-fixture}" ;;
  -m) printf '%s\n' "${FX_UNAME_M:-x86_64}" ;;
  *) printf '%s\n' "${FX_UNAME_S:-Linux}" ;;
esac
STUB
cat > "$B/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
# the failure text nvidia-smi prints ON STDOUT, with commas, so only the exit status tells it apart
[ "${FX_NVSMI_BROKEN:-0}" = 1 ] && { printf 'Failed to initialize NVML: Driver/library version mismatch, NVML library version: 580.95, kernel: 575.64\n'; exit 9; }
[ -n "${FX_GPU:-}" ] || exit 9
printf '%s\n' "$FX_GPU"
STUB
printf '#!/usr/bin/env bash\nprintf "Architecture: x86_64\\nModel name:            Fixture CPU\\n"\n' > "$B/lscpu"
printf '#!/usr/bin/env bash\nprintf "8\\n"\n' > "$B/nproc"
printf '#!/usr/bin/env bash\ncase "${2:-}" in machdep.cpu.brand_string) echo "Apple M4" ;; hw.ncpu) echo 10 ;; *) exit 1 ;; esac\n' > "$B/sysctl"
cat > "$B/flock" <<'STUB'
#!/usr/bin/env bash
w=""; [ "${1:-}" = -w ] && { w=$2; shift 2; }
printf 'flock %s host=%s\n' "$1" "${FX_HOST:-local}" >> "$FX_STATE/wrap.log"; shift
[ -z "$w" ] || printf 'flock-wait %s host=%s\n' "$w" "${FX_HOST:-local}" >> "$FX_STATE/wrap.log"
export FX_UNDER_LOCK=1; exec "$@"
STUB
# gpu-q: the ordered queue in front of the same flock (rule rev 5); it ends by running flock+choom
cat > "$B/gpu-q" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = --prio ] && [ "${3:-}" = -- ] || exit 2
printf 'gpu-q prio %s host=%s\n' "$2" "${FX_HOST:-local}" >> "$FX_STATE/wrap.log"; shift 3
exec flock /tmp/apr-gpu.lock choom -n 1000 -- "$@"
STUB
cat > "$B/choom" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = -n ] && [ "${3:-}" = -- ] || exit 2
printf 'choom %s host=%s\n' "$2" "${FX_HOST:-local}" >> "$FX_STATE/wrap.log"; shift 3; exec "$@"
STUB
# ssh: runs the host's command HERE, as that host (uname, GPU); `bash -s` bodies are the asset and
# installer checks, answered green. A row can make hosts unreachable (255), toolchain-less, or have
# the first call remove $AP/receipts (the mid-step falsifier).
cat > "$B/ssh" <<'STUB'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
h=$1; shift; cmd="$*"
printf 'ssh %s\n' "$h" >> "$FX_STATE/ssh.log"
if [ "${FX_RM_RECEIPTS_ON_SSH:-0}" = 1 ] && [ ! -e "$FX_STATE/rm.done" ]; then
  : > "$FX_STATE/rm.done"; r="${RELEASE_AP:+$RELEASE_AP/receipts}"
  if [ -n "$r" ] && [ "$r" != "/" ]; then rm -rf -- "$r"; fi
fi
case " ${FX_UNREACHABLE:-} " in *" $h "*) printf 'ssh: connect to host %s port 22: No route to host\n' "$h" >&2; exit 255 ;; esac
export FX_HOST=$h FX_GPU="" FX_UNAME_S=Linux FX_UNAME_M=x86_64
case "$h" in
  gx10) FX_GPU="NVIDIA GB10, 12.1, 580.95.05"; FX_UNAME_M=aarch64 ;;
  mini) FX_UNAME_S=Darwin; FX_UNAME_M=arm64 ;;
esac
case " ${FX_NOCARGO_HOSTS:-} " in *" $h "*) export CARGO_HOME="$FX_STATE/no-toolchain" ;; esac
if [ "$cmd" = "bash -s" ]; then
  body=$(cat)
  case "$body" in
    *releases/download*) printf 'version: apr 9.9.9\ndevices:\n[]\n' ;;
    *install.sh*) printf 'installer: apr 9.9.9\n' ;;
    *) exit 3 ;;
  esac
  exit 0
fi
exec bash -c "$cmd"
STUB
# gh: milestone 7 titled 9.9.9, PR 99 merged as $FX_MC, a release URL; anything else fails loudly
cat > "$B/gh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FX_STATE/gh.log"
case "${1:-}" in
  api) case "${2:-}" in *'milestones?state=all'*) printf '7\n' ;; *) exit 1 ;; esac ;;
  pr)
    case "${2:-}" in
      view) printf '%s\n' "$FX_MC" ;;
      list) [ -f "$FX_STATE/pr.created" ] && printf '4242\n'; exit 0 ;;
      create) : > "$FX_STATE/pr.created"; printf 'https://github.com/paiml/aprender/pull/4242\n' ;;
      *) exit 1 ;;
    esac ;;
  release)
    case "${2:-}" in
      view) case "$*" in
              *'--json body'*) if [ -f "$FX_STATE/notes.md" ]; then cat "$FX_STATE/notes.md"; else printf 'fixture notes\n'; fi ;;
              *) printf 'https://github.com/paiml/aprender/releases/tag/v9.9.9\n' ;;
            esac ;;
      edit) nf=""; prev=""; for a in "$@"; do [ "$prev" = --notes-file ] && nf=$a; prev=$a; done
            cp -- "$nf" "$FX_STATE/notes.md" ;;
      *) exit 1 ;;
    esac ;;
  *) exit 1 ;;
esac
STUB
chmod +x "$TMP/cargo-stub" "$TMP/apr-stub" "$B"/*
# the same stubs without gpu-q: a host on which only the legacy flock+choom exists
BNQ="$TMP/bin-no-gpuq"; mkdir -p "$BNQ"
for f in "$B"/*; do [ "${f##*/}" = gpu-q ] || ln -s "$f" "$BNQ/${f##*/}"; done

# the crates.io index record the receipt reads sha256_published from (two: matching, and not)
CRATE_SHA=$(printf 'fixture crate bytes\n' | python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')
printf '{"name":"aprender","vers":"9.9.8","cksum":"%s"}\n{"name":"aprender","vers":"9.9.9","cksum":"%s"}\n' "$(printf '0%.0s' $(seq 64))" "$CRATE_SHA" > "$TMP/index-ok"
FF64=$(printf 'f%.0s' $(seq 64))
printf '{"name":"aprender","vers":"9.9.9","cksum":"%s"}\n' "$FF64" > "$TMP/index-other"
mkdir -p "$TMP/home/models" && printf 'fixture gguf\n' > "$TMP/home/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

# kit_files DIR HOST_RECEIPT -> the files host_receipt.sh needs beside it: the manifest, the pin and
# the two block producers (fixtures: bench adds a block, parity writes --out or refuses)
kit_files() {
    local k=$1
    mkdir -p "$k/scripts/release" "$k/scripts/lib" || return 2
    cp -- "$2" "$k/scripts/release/host_receipt.sh" && cp -- "$PIN" "$k/scripts/llama_pin.toml" || return 2
    printf '[package]\nname = "fx"\nversion = "9.9.9"\nedition = "2021"\n' > "$k/Cargo.toml"
    cat > "$k/scripts/bench_host_receipt.sh" <<'STUB'
#!/usr/bin/env bash
[ "${FX_BENCH_RC:-0}" = 0 ] || { printf 'bench: REFUSED (fixture)\n'; exit "$FX_BENCH_RC"; }
v=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
python3 - "evidence/dogfood/$v/$1.json" "$APR_UNDER_TEST" <<'PY'
import json, sys
p, apr = sys.argv[1:3]
r = json.load(open(p)); r["bench"] = {"fixture": True, "apr": apr}
json.dump(r, open(p, "w"), indent=2)
PY
STUB
    cat > "$k/scripts/parity_host_receipt.sh" <<'STUB'
#!/usr/bin/env bash
out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=${2:-}; shift; done
printf 'parity: host=%s llama=%s\n' "${LLAMA_PIN_HOST:-}" "${LLAMA_BENCH_PATH:-}"
printf 'parity-env GPU_BAND_Q=[%s] GPUQ_WAIT=%s GPU_BAND_TIMEOUT_S=%s under_lock=%s host=%s\n' "${GPU_BAND_Q:-}" "${GPUQ_WAIT:-}" \
    "${GPU_BAND_TIMEOUT_S:-}" "${FX_UNDER_LOCK:-0}" "${LLAMA_PIN_HOST:-}" >> "$FX_STATE/parity-env.log"
[ "${FX_PARITY_RC:-0}" = 0 ] || { printf 'REFUSED: no pinned llama-bench at %s\n' "${LLAMA_BENCH_PATH:-}"; exit "$FX_PARITY_RC"; }
# the REAL producer's shape: parity_block.py writes {"parity": <block>} (a stub that wrote the bare block
# hid host_receipt.sh storing it as parity.parity until the 0.68.2 first-green run)
printf '{"parity": {"fixture": true, "lanes": []}}\n' > "$out"
STUB
    printf '#!/usr/bin/env bash\nreturn 0 2>/dev/null || exit 0\n' > "$k/scripts/llama_bin.sh"
    printf '# fixture\n' > "$k/scripts/lib/parity_block.py"; printf '# fixture\n' > "$k/scripts/lib/perf_receipt.py"
    cp -- "$VALIDATOR" "$k/scripts/lib/bench_receipt.py" && cp -- "$TWIN" "$k/scripts/lib/yaml_twin.py" \
        && cp -- "$ROOT/scripts/perf-matrix.yaml" "$ROOT/scripts/perf-receipt-fields.yaml" "$k/scripts/" || return 2
}

fails=0; rows=0
row() { # row NAME RC MESSAGE
    rows=$((rows + 1))
    if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"
    elif [ "$2" = 2 ]; then env_die "row $1 could not build its fixture"
    else printf 'FAIL  %s: %s\n' "$1" "$3" >&2; fails=$((fails + 1)); fi
}
# rcheck FILE EXPR MSG -> 0 when the python expression over the receipt `r` holds (python, never jq)
rcheck() {
    python3 - "$1" "$2" <<'PY' > /dev/null 2>&1 || { printf '%s\n' "$3"; return 1; }
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if eval(sys.argv[2], {"r": r}) else 1)
PY
}

# ============ part A: host_receipt.sh in a kit ====================================================
# hr NAME SUBJECT VERSION HOST [ENV=VAL ...] -> rc in $TMP/NAME/rc; receipt under $TMP/NAME/kit
hr() {
    local d="$TMP/$1" subject=$2 ver=$3 host=$4; shift 4
    mkdir -p "$d/state" "$d/pkg/bin" || return 2
    kit_files "$d/kit" "$subject" || return 2
    cp -- "$TMP/cargo-stub" "$d/pkg/bin/cargo" || return 2
    : > "$d/state/wrap.log"; : > "$d/state/pkg.log"
    ( export PATH="${FX_BIN:-$B}:$PATH" CARGO_HOME="$d/pkg" HOME="$TMP/home" FX_STATE="$d/state" FX_APR_STUB="$TMP/apr-stub" \
          HOST_RECEIPT_INDEX_URL="file://$TMP/index-ok" FX_HOST="$host"
      for kv in "$@"; do export "${kv?}"; done
      [ "${FX_BIN:-$B}" = "$B" ] || PATH="$FX_BIN:/usr/bin:/bin"
      bash "$d/kit/scripts/release/host_receipt.sh" "$ver" "$host" ${FX_ARGS:-} ) > "$d/out.log" 2>&1
    printf '%s\n' "$?" > "$d/rc"
}
GPU_LAMBDA="NVIDIA GeForce RTX 4090, 8.9, 580.95.05"

a1_gpu_linux() { # SUBJECT -> 0 green
    local n="a1-$1" d f; d="$TMP/a1-$1"
    hr "$n" "$2" 9.9.9 lambda "FX_GPU=$GPU_LAMBDA" || return 2
    [ "$(cat "$d/rc")" = 0 ] || { printf 'host_receipt exited %s: %s\n' "$(cat "$d/rc")" "$(tail -1 "$d/out.log")"; return 1; }
    f="$d/kit/evidence/dogfood/9.9.9/lambda.json"; [ -s "$f" ] || { printf 'no receipt at %s\n' "$f"; return 1; }
    grep -qx -- '---RECEIPT lambda---' "$d/out.log" || { printf 'the receipt was not printed between its markers\n'; return 1; }
    rcheck "$f" 'r["host"] == "lambda" and r["version_tested"] == "9.9.9" and r["provenance"] == "crates.io" and r["asset"] == "aprender-9.9.9.crate"' 'identity/provenance fields wrong' || return 1
    rcheck "$f" 'r["install_rc"] == 0 and r["crates_compiled"] == 2 and isinstance(r["install_wall_seconds"], int)' 'install fields wrong' || return 1
    rcheck "$f" 'r["sha256_published"] and r["sha256_published"] == r["sha256_measured"]' 'sha256_published/sha256_measured wrong' || return 1
    rcheck "$f" 'r["devices"] == {"devices": [{"kind": "cpu", "name": "fixture"}]}' 'devices not verbatim' || return 1
    rcheck "$f" 'r["generate"]["output_sane"] is True and r["generate"]["tokens"] == 16 and len(r["generate"]["model_sha256"]) == 64 and r["generate"]["backend_line_verbatim"] == "Backend: CPU (fixture)" and r["generate"]["used_gpu"] is False and r["generate"]["fallback_line_verbatim"] is None and isinstance(r["generate"]["wall_ms"], int)' 'generate block wrong' || return 1
    rcheck "$f" 'r["driver"] == "580.95.05" and "sm_89" in r["accelerator"] and r["nproc"] == 8 and r["cpu"] == "Fixture CPU" and r["os"].startswith("Linux ")' 'host facts wrong' || return 1
    rcheck "$f" 'r["bench"]["fixture"] and r["parity"]["fixture"] and "parity" not in r["parity"]' 'bench/parity blocks missing, or the parity block is nested as parity.parity' || return 1
    rcheck "$f" 'any(u.startswith("accel lane:") for u in r["unmeasured"])' 'unmeasured[] lacks the always-present accel-lane line' || return 1
    [ "$(grep -c '^gpu-q prio 5 host=lambda$' "$d/state/wrap.log")" = 2 ] && [ "$(grep -c '^flock /tmp/apr-gpu.lock host=lambda$' "$d/state/wrap.log")" = 2 ] \
        && [ "$(grep -c '^choom 1000 host=lambda$' "$d/state/wrap.log")" = 2 ] \
        || { printf 'the GPU rule (gpu-q, then flock+choom) did not wrap exactly generate and bench (wrap.log: %s)\n' "$(tr '\n' ';' < "$d/state/wrap.log")"; return 1; }
    grep -qx 'parity-env GPU_BAND_Q=\[gpu-q --prio 5 --\] GPUQ_WAIT=3600 GPU_BAND_TIMEOUT_S=1200 under_lock=0 host=lambda' "$d/state/parity-env.log" \
        || { printf 'parity was not handed the GPU rule to apply per band, or ran under the lock (%s)\n' "$(cat "$d/state/parity-env.log" 2> /dev/null)"; return 1; }
    grep -q 'under_lock=0' "$d/state/pkg.log" || { printf 'the build ran under the GPU lock\n'; return 1; }
    return 0
}
a2_darwin() {
    local n="a2-$1" d f; d="$TMP/a2-$1"
    hr "$n" "$2" 9.9.9 mini FX_UNAME_S=Darwin FX_UNAME_M=arm64 || return 2
    [ "$(cat "$d/rc")" = 0 ] || { printf 'host_receipt exited %s\n' "$(cat "$d/rc")"; return 1; }
    f="$d/kit/evidence/dogfood/9.9.9/mini.json"
    rcheck "$f" 'r["os"].startswith("Darwin ") and r["arch"] == "arm64" and r["cpu"] == "Apple M4" and r["nproc"] == 10 and r["accelerator"] == "Apple M4 (Metal)" and r["driver"] == "n/a (macOS)"' 'Darwin facts not read from sysctl/uname' || return 1
    [ ! -s "$d/state/wrap.log" ] || { printf 'mini ran under the GPU wrap\n'; return 1; }
    grep -qx 'parity-env GPU_BAND_Q=\[\] GPUQ_WAIT=3600 GPU_BAND_TIMEOUT_S=1200 under_lock=0 host=mini' "$d/state/parity-env.log" \
        || { printf 'mini'"'"'s parity was handed a GPU rule (%s)\n' "$(cat "$d/state/parity-env.log" 2> /dev/null)"; return 1; }
    return 0
}
a3_install_fails() {
    local n="a3-$1" d f; d="$TMP/a3-$1"
    hr "$n" "$2" 9.9.9 intel FX_INSTALL_RC=101 || return 2
    [ "$(cat "$d/rc")" = 0 ] || { printf 'host_receipt exited %s: a failed install must still leave a receipt\n' "$(cat "$d/rc")"; return 1; }
    f="$d/kit/evidence/dogfood/9.9.9/intel.json"
    rcheck "$f" 'r["install_rc"] == 101 and r["generate"] is None and r["binary_path"] is None' 'install failure not recorded' || return 1
    rcheck "$f" 'all(any(u.startswith(k) for u in r["unmeasured"]) for k in ("install:", "generate:", "bench:", "parity:"))' 'unmeasured[] does not name install, generate, bench and parity' || return 1
    return 0
}
a4_parity_refuses() {
    local n="a4-$1" d f; d="$TMP/a4-$1"
    hr "$n" "$2" 9.9.9 intel FX_PARITY_RC=1 || return 2
    f="$d/kit/evidence/dogfood/9.9.9/intel.json"
    rcheck "$f" '"parity" not in r and r["parity_attempt"]["status"] == "refused" and r["parity_attempt"]["rc"] == 1 and "REFUSED" in r["parity_attempt"]["reason"]' 'a refused parity left no parity_attempt with its reason' || return 1
    rcheck "$f" 'any(u.startswith("parity:") for u in r["unmeasured"])' 'unmeasured[] does not name the parity refusal' || return 1
    return 0
}
a5_insane() {
    local n="a5-$1" d f; d="$TMP/a5-$1"
    hr "$n" "$2" 9.9.9 intel FX_ANSWER=5 || return 2
    f="$d/kit/evidence/dogfood/9.9.9/intel.json"
    rcheck "$f" 'r["generate"]["output_sane"] is False and any("did not contain 4" in u for u in r["unmeasured"])' 'a wrong answer was scored sane' || return 1
    return 0
}
a6_two_hashes() {
    local n="a6-$1" d f; d="$TMP/a6-$1"
    hr "$n" "$2" 9.9.9 intel "HOST_RECEIPT_INDEX_URL=file://$TMP/index-other" || return 2
    f="$d/kit/evidence/dogfood/9.9.9/intel.json"
    rcheck "$f" "r['sha256_published'] == '$FF64' and r['sha256_measured'] and r['sha256_measured'] != r['sha256_published']" 'published and measured hashes were not kept as two fields' || return 1
    return 0
}
a7_bad_version() {
    local n="a7-$1" d; d="$TMP/a7-$1"
    hr "$n" "$2" not-a-version intel || return 2
    [ "$(cat "$d/rc")" = 2 ] || { printf 'a version "not-a-version" exited %s, not 2\n' "$(cat "$d/rc")"; return 1; }
    [ ! -e "$d/kit/evidence" ] || { printf 'a receipt dir was created for a bad version\n'; return 1; }
    return 0
}
a8_nvsmi_fails() {
    local n="a8-$1" d f; d="$TMP/a8-$1"
    hr "$n" "$2" 9.9.9 intel FX_NVSMI_BROKEN=1 || return 2
    f="$d/kit/evidence/dogfood/9.9.9/intel.json"
    rcheck "$f" 'r["accelerator"] == "none" and r["driver"] == "n/a"' 'a FAILED nvidia-smi was read as an accelerator' || return 1
    rcheck "$f" 'any(u.startswith("accelerator: nvidia-smi is present but failed (rc 9): Failed to initialize NVML") for u in r["unmeasured"])' 'unmeasured[] does not name the failed nvidia-smi' || return 1
    return 0
}
a9_gpu_fallback() {
    local n="a9-$1" d f; d="$TMP/a9-$1"
    hr "$n" "$2" 9.9.9 mini FX_UNAME_S=Darwin FX_UNAME_M=arm64 "FX_BACKEND=wgpu (Vulkan)" FX_FALLBACK=1 || return 2
    f="$d/kit/evidence/dogfood/9.9.9/mini.json"
    rcheck "$f" 'r["generate"]["backend_line_verbatim"] == "Backend: wgpu (Vulkan)" and "path rejected" in r["generate"]["fallback_line_verbatim"] and r["generate"]["used_gpu"] is False' 'the fallback was not recorded verbatim' || return 1
    rcheck "$f" 'any(u.startswith("generate: apr'"'"'s accelerated path was not used: warning: GPU (wgpu) path rejected") for u in r["unmeasured"])' 'unmeasured[] does not name the fallback' || return 1
    return 0
}
a10_legacy_wrap() {
    local n="a10-$1" d f; d="$TMP/a10-$1"
    hr "$n" "$2" 9.9.9 lambda "FX_GPU=$GPU_LAMBDA" "FX_BIN=$BNQ" || return 2
    f="$d/kit/evidence/dogfood/9.9.9/lambda.json"
    [ "$(cat "$d/rc")" = 0 ] || { printf 'host_receipt exited %s\n' "$(cat "$d/rc")"; return 1; }
    [ "$(grep -c '^gpu-q ' "$d/state/wrap.log")" = 0 ] && [ "$(grep -c '^flock /tmp/apr-gpu.lock host=lambda$' "$d/state/wrap.log")" = 2 ] \
        && [ "$(grep -c '^flock-wait 3600 host=lambda$' "$d/state/wrap.log")" = 2 ] \
        || { printf 'without gpu-q the bare flock -w 3600 + choom did not wrap generate and bench (wrap.log: %s)\n' "$(tr '\n' ';' < "$d/state/wrap.log")"; return 1; }
    grep -qx 'parity-env GPU_BAND_Q=\[flock -w 3600 /tmp/apr-gpu.lock choom -n 1000 --\] GPUQ_WAIT=3600 GPU_BAND_TIMEOUT_S=1200 under_lock=0 host=lambda' "$d/state/parity-env.log" \
        || { printf 'parity was not handed the legacy rule (%s)\n' "$(cat "$d/state/parity-env.log" 2> /dev/null)"; return 1; }
    rcheck "$f" 'any(u.startswith("gpu-rule: no gpu-q on lambda") for u in r["unmeasured"])' 'unmeasured[] does not say the ordered queue was bypassed' || return 1
    return 0
}
a11_bad_prio() {
    local n="a11-$1" d; d="$TMP/a11-$1"
    fx_args=$(printf -- '--gpu-prio %s' 12)
    hr "$n" "$2" 9.9.9 lambda "FX_ARGS=$fx_args" || return 2
    [ "$(cat "$d/rc")" = 2 ] || { printf -- '--gpu-prio 12 exited %s, not 2\n' "$(cat "$d/rc")"; return 1; }
    [ ! -e "$d/kit/evidence" ] || { printf 'a receipt dir was created for a bad priority\n'; return 1; }
    return 0
}

# ============ part B: the autopilot's hosts / postpub / install steps =============================
# fixture NAME AUTOPILOT HOST_RECEIPT [REPORT_ONLY] -> $TMP/NAME: origin + checkout at 9.9.9 carrying
# the kit files; the gate copy's REPORT_ONLY is the fixture list given, else the real one
fixture() {
    local d="$TMP/$1" r; r="$TMP/$1/repo"
    mkdir -p "$d/ap" "$d/state" "$d/pkg/bin" || return 2
    kit_files "$r" "$3" || return 2
    cp -- "$2" "$r/scripts/release/autopilot.sh" && cp -- "$PARAMS" "$r/scripts/release/lib_release_params.sh" \
        && cp -- "$GATE" "$r/scripts/check_multiplatform_dogfood.sh" && cp -- "$TMP/cargo-stub" "$d/pkg/bin/cargo" \
        && cp -- "$LEDGER" "$r/scripts/release/ledger.py" || return 2
    if [ -n "${4:-}" ]; then
        python3 - "$r/scripts/check_multiplatform_dogfood.sh" "$4" <<'PY' || return 2
import re, sys
p, lst = sys.argv[1:3]
s, n = re.subn(r'(?m)^REPORT_ONLY=".*"$', f'REPORT_ONLY="{lst}"', open(p).read())
if n != 1: sys.exit(1)
open(p, "w").write(s)
PY
    fi
    printf '#!/usr/bin/env bash\nexit 0\n' > "$r/scripts/bump-version.sh"
    # the post-publish dogfood: runs the REAL gate over the receipts it is pointed at, then writes the
    # receipt a real run writes, with the verdict/deferred the row picks
    cat > "$r/scripts/dogfood.sh" <<'STUB'
#!/usr/bin/env bash
dir=${DOGFOOD_RECEIPTS_DIR:-<unset>}; n=0
for f in "$dir"/*.json; do [ -f "$f" ] && n=$((n + 1)); done
printf 'ran %s dir=%s n=%s\n' "$*" "$dir" "$n" >> "$FX_STATE/dogfood.log"
bash scripts/check_multiplatform_dogfood.sh > "$FX_STATE/gate.log" 2>&1
v=${FX_DOGFOOD_VERDICT:-GO}
mkdir -p .dogfood
[ "${FX_NO_RECEIPT:-0}" = 1 ] || printf '{"crate":"fx","version":"9.9.9","timestamp":"20260921T000000Z","commit":"%s","gates":[],"phase":"post-publish","deferred":%s,"verdict":"%s"}\n' \
    "$(git rev-parse HEAD)" "${FX_DEFERRED:-[]}" "$v" > .dogfood/receipt-20260921T000000Z.json
printf 'VERDICT: %s\n' "$v"
[ "$v" = GO ]
STUB
    printf '.dogfood/\ntarget/\n' > "$r/.gitignore"
    git init -q --bare -b main "$d/origin.git" && git -C "$r" init -q -b main && git -C "$r" add -A \
        && git -C "$r" commit -q -m 'release: 9.9.9' && git -C "$r" remote add origin "$d/origin.git" \
        && git -C "$r" push -q origin main && git -C "$r" fetch -q origin || return 2
    git -C "$r" rev-parse HEAD > "$d/mc"
    : > "$d/state/dogfood.log"; : > "$d/state/ssh.log"; : > "$d/state/gh.log"; : > "$d/state/wrap.log"; : > "$d/state/pkg.log"
}
# autopilot NAME FROM TO [ENV=VAL ...] -> rc in $TMP/NAME/rc; lambda (the train host) runs locally
autopilot() {
    local d="$TMP/$1" from=$2 to=$3; shift 3
    ( export RELEASE_AP="$d/ap" RELEASE_EPIC=9002 CARGO_HOME="$d/pkg" PATH="$B:$PATH" HOME="$TMP/home" \
          FX_STATE="$d/state" FX_MC="$(cat "$d/mc")" FX_APR_STUB="$TMP/apr-stub" \
          HOST_RECEIPT_INDEX_URL="file://$TMP/index-ok" FX_HOST=lambda FX_GPU="$GPU_LAMBDA" RELEASE_INDEX_SETTLE_S=0
      for kv in "$@"; do export "${kv?}"; done
      bash "$d/repo/scripts/release/autopilot.sh" 9.9.9 99 "$from" "$to" ) > "$d/out.log" 2>&1
    printf '%s\n' "$?" > "$d/rc"
}
last_status() { tail -n 1 "$1/ap/STATUS" 2> /dev/null; }

b1_train_green() { # TAG AUTOPILOT HOST_RECEIPT
    local n="b1-$1" d h; d="$TMP/b1-$1"
    fixture "$n" "$2" "$3" || return 2
    autopilot "$n" hosts postpub
    [ "$(cat "$d/rc")" = 0 ] || { printf 'autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    for h in lambda intel gx10 mini; do
        grep -q "HOST-RECEIPT $h install_rc=0 " "$d/ap/STATUS" || { printf 'no HOST-RECEIPT line for %s\n' "$h"; return 1; }
        grep -Eq "^ok +$h +9\.9\.9 verified" "$d/state/gate.log" || { printf 'the REAL gate did not accept %s'"'"'s receipt: %s\n' "$h" "$(grep -m1 " $h " "$d/state/gate.log")"; return 1; }
        grep -Eq "^ok +$h +generate sane, \.crate sha256 published = measured, unmeasured\[\] named$" "$d/state/gate.log" \
            || { printf 'the REAL gate did not accept %s'"'"'s generate/sha256/unmeasured blocks: %s\n' "$h" "$(grep -E "^(FAIL|REPORT) +$h " "$d/state/gate.log" | tr '\n' ';')"; return 1; }
    done
    grep -qx "ran --phase post-publish dir=$d/ap/receipts/dogfood n=4" "$d/state/dogfood.log" \
        || { printf 'the post-publish dogfood did not read the 4 train receipts: %s\n' "$(cat "$d/state/dogfood.log")"; return 1; }
    python3 - "$d/ap/STATUS" <<'PY' || { printf 'the post-publish dogfood did not run AFTER the host receipts\n'; return 1; }
import sys
s = open(sys.argv[1]).read().splitlines()
last_receipt = max(i for i, l in enumerate(s) if " HOST-RECEIPT " in l)
dog = [i for i, l in enumerate(s) if " DOGFOOD post-publish GO at " in l]
sys.exit(0 if dog and dog[0] > last_receipt else 1)
PY
    rcheck "$d/ap/receipts/dogfood/mini.json" 'r["os"].startswith("Darwin ")' 'mini'"'"'s receipt is not Darwin' || return 1
    kit_list=$(tar -tf "$d/ap/receipts/kit.tar" 2> /dev/null)
    for tw in ./scripts/perf-matrix.json ./scripts/perf-receipt-fields.json; do
        grep -qx -- "$tw" <<< "$kit_list" || { printf 'the host-receipt kit carries no JSON twin %s\n' "$tw"; return 1; }
    done
    for h in lambda gx10; do
        [ "$(grep -c "^gpu-q prio 1 host=$h\$" "$d/state/wrap.log")" = 2 ] || { printf 'the train did not queue %s'"'"'s measurements at release priority 1 (wrap.log: %s)\n' "$h" "$(grep " host=$h" "$d/state/wrap.log" | tr '\n' ';')"; return 1; }
    done
    return 0
}
b2_dir_removed() {
    local n="b2-$1" d; d="$TMP/b2-$1"
    fixture "$n" "$2" "$3" || return 2
    [ ! -e "$d/ap/receipts" ] || return 2
    autopilot "$n" hosts hosts
    [ "$(cat "$d/rc")" = 0 ] || { printf 'with no receipts dir before the step, autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    [ -s "$d/ap/receipts/dogfood/intel.json" ] || { printf 'no receipt written\n'; return 1; }
    return 0
}
b3_dir_vanishes() {
    local n="b3-$1" d s; d="$TMP/b3-$1"
    fixture "$n" "$2" "$3" || return 2
    autopilot "$n" hosts hosts FX_RM_RECEIPTS_ON_SSH=1
    s=$(last_status "$d")
    [ "$(cat "$d/rc")" = 1 ] || { printf 'autopilot exited %s, not a STOP\n' "$(cat "$d/rc")"; return 1; }
    case "$s" in *"STOP INFRA the receipt dir $d/ap/receipts is missing or unwritable"*) ;; *) printf 'the STOP does not name the missing dir as INFRA: %s\n' "$s"; return 1 ;; esac
    ! grep -q 'receipt(s) failed' "$d/ap/STATUS" || { printf 'a missing dir was counted as failed hosts\n'; return 1; }
    return 0
}
b4_dir_blocked() {
    local n="b4-$1" d s; d="$TMP/b4-$1"
    fixture "$n" "$2" "$3" || return 2
    printf 'not a dir\n' > "$d/ap/receipts"
    autopilot "$n" hosts hosts
    s=$(last_status "$d")
    case "$s" in *"STOP INFRA cannot create the receipt dir $d/ap/receipts/dogfood"*) ;; *) printf 'the STOP does not name the uncreatable dir: %s\n' "$s"; return 1 ;; esac
    return 0
}
b5_unreachable() {
    local n="b5-$1" d s; d="$TMP/b5-$1"
    fixture "$n" "$2" "$3" || return 2
    autopilot "$n" hosts hosts FX_UNREACHABLE=gx10
    s=$(last_status "$d")
    [ "$(cat "$d/rc")" = 1 ] || { printf 'an unreachable host did not STOP the train\n'; return 1; }
    case "$s" in *"STOP INFRA fault(s): gx10(ssh-255) install-gx10(ssh-255) receipt-gx10(ssh-255) -- infrastructure, not host verdicts; 0 host/installer receipt(s) failed besides"*) ;;
        *) printf 'the STOP counts an unreachable host as a verdict: %s\n' "$s"; return 1 ;; esac
    ! grep -q 'HOST gx10 rc=' "$d/ap/STATUS" || { printf 'gx10 got a HOST verdict line\n'; return 1; }
    return 0
}
b6_env_no_toolchain() {
    local n="b6-$1" d; d="$TMP/b6-$1"
    fixture "$n" "$2" "$3" || return 2
    autopilot "$n" hosts hosts FX_NOCARGO_HOSTS=intel
    grep -q 'INFRA intel returned no receipt (host_receipt rc=2)' "$d/ap/STATUS" || { printf 'no INFRA line for a host with no toolchain: %s\n' "$(last_status "$d")"; return 1; }
    case "$(last_status "$d")" in *"STOP INFRA fault(s): receipt-intel --"*) ;; *) printf 'wrong STOP: %s\n' "$(last_status "$d")"; return 1 ;; esac
    return 0
}
postpub_row() { # TAG AUTOPILOT WANT-IN-STOP ENV... -> the post-publish dogfood STOPs with WANT
    local n="$1" ap=$2 want=$3 d; shift 3; d="$TMP/$n"
    fixture "$n" "$ap" "$HOST_RECEIPT" || return 2
    autopilot "$n" hosts postpub "$@"
    [ "$(cat "$d/rc")" = 1 ] || { printf 'autopilot exited %s, not a STOP: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    case "$(last_status "$d")" in *"STOP $want"*) ;; *) printf 'expected STOP "%s", got: %s\n' "$want" "$(last_status "$d")"; return 1 ;; esac
    ! grep -q 'DOGFOOD post-publish GO' "$d/ap/STATUS" || { printf 'a GO was recorded\n'; return 1; }
    return 0
}
b10_install_only() {
    local n="b10-$1" d; d="$TMP/b10-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" || return 2
    autopilot "$n" install install
    [ "$(cat "$d/rc")" = 0 ] || { printf 'install exited %s: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    [ ! -s "$d/state/dogfood.log" ] || { printf 'install ran the post-publish dogfood before any host receipt existed\n'; return 1; }
    return 0
}
b11_postpub_alone() {
    local n="b11-$1" d; d="$TMP/b11-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" || return 2
    autopilot "$n" postpub postpub
    case "$(last_status "$d")" in *"STOP INFRA no $d/ap/receipts/dogfood -- the hosts step has not run"*) ;; *) printf 'postpub without receipts: %s\n' "$(last_status "$d")"; return 1 ;; esac
    [ ! -s "$d/state/dogfood.log" ] || { printf 'the dogfood ran with nothing to judge\n'; return 1; }
    return 0
}
b12_report_notes() {
    local n="b12-$1" d h; d="$TMP/b12-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" "9.9.9:lambda:bench:#3758 9.9.9:intel:bench:#3758 9.9.9:gx10:bench:#3758 9.9.9:mini:bench:#3758" || return 2
    autopilot "$n" hosts postpub FX_BENCH_RC=1
    [ "$(cat "$d/rc")" = 0 ] || { printf 'autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    [ -f "$d/state/notes.md" ] || { printf 'the release notes were never edited (REPORT rows: %s)\n' "$(grep -c '^REPORT ' "$d/ap/multiplatform.log")"; return 1; }
    grep -qx '## Post-publish host receipts: rows that ran as REPORT' "$d/state/notes.md" || { printf 'no REPORT section in the notes\n'; return 1; }
    for h in lambda intel gx10 mini; do
        grep -Eq "^- $h +no bench block: runs as REPORT for 9\.9\.9, owed by #3758; the producer refused on this host: bench: REFUSED \(fixture\)" "$d/state/notes.md" || { printf 'the notes do not name %s'"'"'s REPORT row with its issue and refusal\n' "$h"; return 1; }
    done
    autopilot "$n" postpub postpub FX_BENCH_RC=1
    [ "$(grep -c '^release edit' "$d/state/gh.log")" = 1 ] || { printf 'a second postpub edited the notes again\n'; return 1; }
    [ "$(grep -c '^## Post-publish host receipts' "$d/state/notes.md")" = 1 ] || { printf 'the REPORT section was appended twice\n'; return 1; }
    return 0
}

# ============ part G: the gate's per-version parity REPORT list ===================================
# gate_run NAME GATE LIST [INTEL-PATCH-JSON [VERSION]] -> the gate's output in $TMP/NAME/gate.log. The
# gate's REPORT_ONLY is replaced by LIST (the mechanism is judged here; G4 judges the real
# list), over four receipts shaped as host_receipt.sh writes them, with no parity block; intel's is
# patched by the JSON object given.
gate_run() {
    local d="$TMP/$1" r h; r="$TMP/$1/repo"
    mkdir -p "$r/scripts/lib" "$d/receipts" "$d/pkg/bin" || return 2
    python3 - "$2" "$r/scripts/check_multiplatform_dogfood.sh" "$3" <<'PY' || return 2
import re, sys
src, dst, lst = sys.argv[1:4]
s, n = re.subn(r'(?m)^REPORT_ONLY=".*"$', f'REPORT_ONLY="{lst}"', open(src).read())
if n != 1: sys.exit(1)
open(dst, "w").write(s)
PY
    # the validator's whole library and the two YAML files it reads: a parity block cannot be judged
    # valid without them, and a gate that reads every block as INVALID proves nothing about validity
    cp -- "$ROOT"/scripts/lib/*.py "$r/scripts/lib/" && cp -- "$ROOT/scripts/perf-matrix.yaml" "$ROOT/scripts/perf-receipt-fields.yaml" "$r/scripts/" \
        && cp -- "$TMP/cargo-stub" "$d/pkg/bin/cargo" || return 2
    printf '[package]\nname = "fx"\nversion = "%s"\nedition = "2021"\n' "${5:-9.9.9}" > "$r/Cargo.toml"
    for h in lambda intel gx10 mini; do
        python3 - "$d/receipts/$h.json" "$h" "${4:-}" "${5:-9.9.9}" <<'PY' || return 2
import json, sys
p, h, patch, v = sys.argv[1:5]
r = {"host": h, "version_tested": v, "install_rc": 0, "date": "2026-09-21", "accelerator": "none",
     "sha256_published": "a" * 64, "sha256_measured": "a" * 64,
     "generate": {"output_sane": True, "tokens": 1}, "unmeasured": ["accel lane: fixture"]}
if h == "intel" and patch:
    r.update(json.load(open(patch[1:])) if patch.startswith("@") else json.loads(patch))
json.dump(r, open(p, "w"))
PY
    done
    ( cd "$r" && DOGFOOD_RECEIPTS_DIR="$d/receipts" PATH="$d/pkg/bin:$PATH" bash scripts/check_multiplatform_dogfood.sh ) > "$d/gate.log" 2>&1
    return 0
}
g1_report_listed() {
    local d="$TMP/g1-$1"
    gate_run "g1-$1" "$2" "9.9.9:intel:parity:#1" || return 2
    grep -Eq '^REPORT intel +no parity block: runs as REPORT for 9\.9\.9, owed by #1 ' "$d/gate.log" || { printf 'no REPORT row for the listed host: %s\n' "$(grep -m1 ' intel ' "$d/gate.log")"; return 1; }
    ! grep -Eq '^FAIL +intel +no parity block' "$d/gate.log" || { printf 'the listed host still FAILed its absent parity block\n'; return 1; }
    grep -Eq '^FAIL +lambda +no parity block' "$d/gate.log" || { printf 'an UNLISTED host did not FAIL for its absent block\n'; return 1; }
    [ "$(grep -Ec '^ok +(lambda|intel|gx10|mini) +generate sane, \.crate sha256 published = measured, unmeasured\[\] named$' "$d/gate.log")" = 4 ] \
        || { printf 'the train blocks were not judged green on all four well-formed receipts\n'; return 1; }
    return 0
}
g2_report_other_version() {
    local d="$TMP/g2-$1"
    gate_run "g2-$1" "$2" "9.9.8:intel:parity:#1" || return 2
    grep -Eq '^FAIL +intel +no parity block' "$d/gate.log" || { printf 'an entry for 9.9.8 exempted 9.9.9: %s\n' "$(grep -m1 ' intel ' "$d/gate.log")"; return 1; }
    return 0
}
g3_report_present() {
    local d="$TMP/g3-$1"
    gate_run "g3-$1" "$2" "9.9.9:intel:parity:#1" '{"parity": {"fixture": true}}' || return 2
    grep -Eq '^FAIL +intel +parity block present but INVALID' "$d/gate.log" || { printf 'a PRESENT invalid block on a listed host was not judged: %s\n' "$(grep -m1 ' intel ' "$d/gate.log")"; return 1; }
    return 0
}
gate_block_row() { # TAG GATE PATCH WANT-REGEX [VERSION [LIST]] -> intel's line matches WANT
    local d="$TMP/$1"
    gate_run "$1" "$2" "${6:-}" "$3" "${5:-}" || return 2
    grep -Eq "$4" "$d/gate.log" || { printf 'no line matching /%s/; intel said: %s\n' "$4" "$(grep -E '^(ok|FAIL|REPORT) +intel ' "$d/gate.log" | tr '\n' ';')"; return 1; }
    return 0
}
g4_report_shape() {
    python3 - "$GATE" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
hosts = re.search(r'(?m)^HOSTS="(.*)"$', s).group(1).split()
m = re.findall(r'(?m)^REPORT_ONLY="(.*)"$', s)
if len(m) != 1: print(f"{len(m)} REPORT_ONLY lines"); sys.exit(1)
for e in m[0].split():
    v = re.fullmatch(r'(\d+\.\d+\.\d+):([a-z0-9-]+):(bench|parity):#(\d+)', e)
    if not v or v.group(2) not in hosts: print(f"bad entry {e!r} (want VERSION:HOST:BLOCK:#ISSUE, HOST in {hosts}, BLOCK bench|parity)"); sys.exit(1)
PY
}
ledger_files() { # ORIGIN-GIT -> the files on ledger/9.9.9, one per line
    git --git-dir="$1" ls-tree -r --name-only refs/heads/ledger/9.9.9 2> /dev/null
}
l1_ledger_pr() {
    local n="l1-$1" d h files mc; d="$TMP/l1-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" || return 2
    autopilot "$n" hosts ledger
    [ "$(cat "$d/rc")" = 0 ] || { printf 'autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    files=$(ledger_files "$d/origin.git"); mc=$(cut -c1-9 "$d/mc")
    [ -n "$files" ] || { printf 'no ledger/9.9.9 branch in origin\n'; return 1; }
    for h in lambda intel gx10 mini; do
        grep -qx "evidence/dogfood/9.9.9/$h.json" <<< "$files" || { printf 'the ledger branch lacks %s'"'"'s receipt\n' "$h"; return 1; }
    done
    grep -Eqx "docs/build-ledger/[0-9]{4}-[0-9]{2}-[0-9]{2}/$mc-lambda-vector-train\.json" <<< "$files" || { printf 'the ledger branch lacks the train record\n'; return 1; }
    [ "$(git --git-dir="$d/origin.git" rev-parse ledger/9.9.9^)" = "$(git --git-dir="$d/origin.git" rev-parse main)" ] || { printf 'the ledger commit is not on top of main\n'; return 1; }
    grep -q '^pr create --repo paiml/aprender --base main --head ledger/9.9.9 ' "$d/state/gh.log" || { printf 'no ledger PR was opened\n'; return 1; }
    ! grep -Eq '^pr merge|--auto' "$d/state/gh.log" || { printf 'the ledger PR was armed\n'; return 1; }
    grep -q 'LEDGER PR https://github.com/paiml/aprender/pull/4242 (ledger/9.9.9)' "$d/ap/STATUS" || { printf 'no LEDGER PR line\n'; return 1; }
    return 0
}
l2_ledger_rerun() {
    local n="l2-$1" d t1 t2; d="$TMP/l2-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" || return 2
    autopilot "$n" hosts ledger
    [ "$(cat "$d/rc")" = 0 ] || return 2
    t1=$(git --git-dir="$d/origin.git" rev-parse ledger/9.9.9)
    autopilot "$n" ledger ledger
    [ "$(cat "$d/rc")" = 0 ] || { printf 'the re-run exited %s: %s\n' "$(cat "$d/rc")" "$(last_status "$d")"; return 1; }
    t2=$(git --git-dir="$d/origin.git" rev-parse ledger/9.9.9)
    git --git-dir="$d/origin.git" merge-base --is-ancestor "$t1" "$t2" || { printf 'the re-run rewrote the branch instead of building on it\n'; return 1; }
    [ "$(grep -c '^pr create' "$d/state/gh.log")" = 1 ] || { printf 'the re-run opened a second PR\n'; return 1; }
    return 0
}
l3_ledger_empty() {
    local n="l3-$1" d; d="$TMP/l3-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" || return 2
    autopilot "$n" ledger ledger
    case "$(last_status "$d")" in *"STOP INFRA no host receipts under $d/ap/receipts/dogfood"*) ;; *) printf 'ledger without receipts: %s\n' "$(last_status "$d")"; return 1 ;; esac
    [ -z "$(ledger_files "$d/origin.git")" ] || { printf 'a ledger branch was pushed with nothing in it\n'; return 1; }
    return 0
}
l4_ledger_rejected() {
    local n="l4-$1" d; d="$TMP/l4-$1"
    fixture "$n" "$2" "$HOST_RECEIPT" || return 2
    printf '#!/bin/sh\nexit 1\n' > "$d/origin.git/hooks/pre-receive" && chmod +x "$d/origin.git/hooks/pre-receive" || return 2
    autopilot "$n" hosts ledger
    [ "$(cat "$d/rc")" = 1 ] || { printf 'a refused push exited %s, not a STOP\n' "$(cat "$d/rc")"; return 1; }
    case "$(last_status "$d")" in *"STOP cannot push ledger/9.9.9: the receipts exist only in $d/ap"*) ;; *) printf 'wrong STOP: %s\n' "$(last_status "$d")"; return 1 ;; esac
    ! grep -q '^pr create' "$d/state/gh.log" || { printf 'a PR was opened for a branch that was never pushed\n'; return 1; }
    return 0
}

# ============ part P/R: the per-band GPU lock (scripts/lib/gpu_band_lock.sh, #3731) ===============
# A real flock(1) on a scratch lock; the gpu-q stub queues nothing and runs `flock <lock> <cmd>`, as
# gpu-q does after its queue (or exits 75, as gpu-q does when GPUQ_WAIT runs out).
# Every row has its OWN lock and directory, and reaps whatever holder it leaves: a mutant that leaks
# a holder (a no-op release) then fails its own row instead of blocking every row after it.
PQ="$TMP/pq-bin"; mkdir -p "$PQ"
cat > "$PQ/gpu-q" <<STUB
#!/usr/bin/env bash
[ "\${FX_Q_REFUSE:-0}" = 1 ] && { echo "gpu-q: ENV waited (fixture)" >&2; exit 75; }
shift 3; exec "$REAL_FLOCK" "\${PQ_LOCK:?}" "\$@"
STUB
chmod +x "$PQ/gpu-q"
lock_state() { if "$REAL_FLOCK" -n "$PQ_LOCK" true; then echo free; else echo held; fi; }
# mktemp, not a counter: rows run inside $( ), where a counter's increment never reaches the next row
band_dir() { BD=$(mktemp -d "$TMP/band-XXXXXX") || return 2; export PQ_LOCK="$BD/gpu.lock"; : > "$PQ_LOCK"; }
band_reap() { local f p; for f in "$1"/gpu-held-*; do [ -f "$f" ] || continue; p=""; read -r p < "$f"; [ -z "$p" ] || kill "$p" 2> /dev/null; done; return 0; }
# band_env LIB -> a subshell prologue: the scratch PATH and the library under test
p1_hold_release() {
    band_dir
    ( trap 'band_reap "$BD"' EXIT; export PATH="$PQ:/usr/bin:/bin"; . "$1"; GPU_BAND_Q="gpu-q --prio 8 --"
      gpu_band_acquire accel-c1 "$BD" || { echo "acquire failed"; exit 1; }
      [ "$(lock_state)" = held ] || { echo "the lock was not held after acquire"; exit 1; }
      h=$GPU_BAND_HOLDER; gpu_band_release
      [ "$(lock_state)" = free ] || { echo "the lock was still held after release"; exit 1; }
      ! kill -0 "$h" 2> /dev/null || { echo "the holder process survived release"; exit 1; } )
}
p2_bound() {
    band_dir
    ( trap 'band_reap "$BD"' EXIT; export PATH="$PQ:/usr/bin:/bin"; . "$1"; GPU_BAND_Q="gpu-q --prio 8 --"; GPU_BAND_TIMEOUT_S=2
      gpu_band_acquire accel-c4 "$BD" || { echo "acquire failed"; exit 1; }
      sleep 60 & srv=$!; gpu_band_track "$srv"
      i=0; while kill -0 "$srv" 2> /dev/null && [ "$i" -lt 15 ]; do sleep 1; i=$((i + 1)); done
      ! kill -0 "$srv" 2> /dev/null || { kill "$srv"; echo "the band's server outlived its bound"; exit 1; }
      gpu_band_expired || { echo "the band was not marked expired"; exit 1; }
      gpu_band_release; [ "$(lock_state)" = free ] || { echo "the lock was held after the bound"; exit 1; } )
}
p3_refused() {
    band_dir
    ( trap 'band_reap "$BD"' EXIT; export PATH="$PQ:/usr/bin:/bin" FX_Q_REFUSE=1; . "$1"; GPU_BAND_Q="gpu-q --prio 8 --"
      if gpu_band_acquire accel-c8 "$BD" 2> /dev/null; then echo "acquire succeeded past a refusing queue"; exit 1; fi
      [ "$(lock_state)" = free ] || { echo "the lock is held"; exit 1; } )
}
p4_no_rule() {
    band_dir
    ( trap 'band_reap "$BD"' EXIT; export PATH="$PQ:/usr/bin:/bin"; . "$1"; GPU_BAND_Q=""
      gpu_band_acquire cpu-c1 "$BD" || { echo "a no-rule acquire failed"; exit 1; }
      [ "$(lock_state)" = free ] && [ -z "$GPU_BAND_HOLDER" ] || { echo "no rule, yet something holds the lock"; exit 1; }
      gpu_band_release )
}
# run_band_row PARITY-SCRIPT KLASS WANT-RC WANT-BODY [ENV=VAL ...]: the REAL run_band() wrapper, extracted,
# around a stub body that records whether the lock was held while it ran (or runs a server past the bound)
run_band_row() {
    local src=$1 klass=$2 want_rc=$3 want_body=$4 d rc body; shift 4
    band_dir; d=$BD
    awk '/^run_band\(\) \{/ {f = 1} f {print} f && /^}/ {exit}' "$src" > "$d/run_band.sh"
    grep -q '^run_band() {' "$d/run_band.sh" || { echo "no run_band() in $src"; return 2; }
    ( trap 'band_reap "$d"' EXIT; export PATH="$PQ:/usr/bin:/bin"; for kv in "$@"; do export "${kv?}"; done
      . "$BANDLOCK"; . "$d/run_band.sh"; GPU_BAND_Q="gpu-q --prio 8 --"; DRY_RUN=0; WORK=$d
      run_band_body() {
          lock_state > "$d/body"
          if [ "${FX_BODY_OVERRUN:-0}" = 1 ]; then sleep 60 & p=$!; gpu_band_track "$p"; wait "$p" 2> /dev/null || true; fi
          return 0
      }
      run_band "$klass" 0 0 1 2> "$d/err"; echo "$?" > "$d/rc" )
    rc=$(cat "$d/rc" 2> /dev/null)
    [ "$rc" = "$want_rc" ] || { echo "run_band $klass exited ${rc:-?} (want $want_rc): $(cat "$d/err")"; return 1; }
    body=$(cat "$d/body" 2> /dev/null) || body=not-run
    [ "$body" = "$want_body" ] || { echo "the $klass band body saw the lock: $body (want $want_body)"; return 1; }
    [ "$(lock_state)" = free ] || { echo "the lock is still held after run_band returned"; return 1; }
    return 0
}

# ============ part T: the kit's JSON twins (scripts/lib/yaml_twin.py) ============================
# twin_row LIBDIR -> the real readers from LIBDIR, run twice on twins LIBDIR's writer made: with PyYAML,
# and with it BLOCKED (sys.modules["yaml"] = None, as on mini); the two canonical dumps must be equal
twin_row() {
    local lib=$1 d
    d=$(mktemp -d "$TMP/twin-XXXXXX") || return 2
    cp -- "$ROOT/scripts/perf-matrix.yaml" "$ROOT/scripts/perf-receipt-fields.yaml" "$d/" || return 2
    python3 "$lib/yaml_twin.py" write "$d/perf-matrix.yaml" "$d/perf-receipt-fields.yaml" || { echo "the twin writer failed"; return 1; }
    python3 - "$lib" "$d" <<'PY'
import subprocess, sys
lib, d = sys.argv[1:3]
prog = r"""
import json, os, sys
if sys.argv[3] == "blocked":
    sys.modules["yaml"] = None
sys.path.insert(0, sys.argv[1])
os.environ["PERF_GATE_MATRIX"] = os.path.join(sys.argv[2], "perf-matrix.yaml")
import bench_receipt, perf_receipt
doc = {"matrix": bench_receipt._matrix(),
       "ledger": perf_receipt.unmeasured_from_ledger(os.path.join(sys.argv[2], "perf-receipt-fields.yaml"))}
print(json.dumps(doc, sort_keys=True))
"""
def run(mode):
    r = subprocess.run([sys.executable, "-c", prog, lib, d, mode], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else [""]
a, b = run("yaml"), run("blocked")
if a[0] != 0:
    print("the YAML path failed:", a[2][0][:300]); sys.exit(1)
if b[0] != 0:
    print("with PyYAML blocked the readers did not load the twin:", b[2][0][:300]); sys.exit(1)
if a[1] != b[1] or len(a[1]) < 100:
    print("the twin does not parse to the object the YAML does"); sys.exit(1)
PY
}
twin_missing_row() {
    local d
    d=$(mktemp -d "$TMP/twin-XXXXXX") || return 2
    cp -- "$ROOT/scripts/perf-matrix.yaml" "$d/" || return 2
    python3 - "$ROOT/scripts/lib" "$d/perf-matrix.yaml" <<'PY'
import sys
sys.modules["yaml"] = None
sys.path.insert(0, sys.argv[1])
import yaml_twin
try:
    yaml_twin.load(sys.argv[2])
except ImportError as e:
    sys.exit(0 if "no JSON twin" in str(e) else (print("wrong refusal:", e) or 1))
print("loaded without PyYAML and without a twin"); sys.exit(1)
PY
}

# ---- the rows ------------------------------------------------------------------------------------
echo "=== the train's host receipts, in the gate's schema, judged after they exist (#3731) ==="
out=$(a1_gpu_linux real "$HOST_RECEIPT"); row "A1 gpu-linux" $? "$out"
out=$(a2_darwin real "$HOST_RECEIPT"); row "A2 darwin" $? "$out"
out=$(a3_install_fails real "$HOST_RECEIPT"); row "A3 install-fails" $? "$out"
out=$(a4_parity_refuses real "$HOST_RECEIPT"); row "A4 parity-refuses" $? "$out"
out=$(a5_insane real "$HOST_RECEIPT"); row "A5 insane-answer" $? "$out"
out=$(a6_two_hashes real "$HOST_RECEIPT"); row "A6 two-hashes" $? "$out"
out=$(a7_bad_version real "$HOST_RECEIPT"); row "A7 bad-version" $? "$out"
out=$(a8_nvsmi_fails real "$HOST_RECEIPT"); row "A8 nvsmi-fails" $? "$out"
out=$(a9_gpu_fallback real "$HOST_RECEIPT"); row "A9 gpu-fallback" $? "$out"
out=$(a10_legacy_wrap real "$HOST_RECEIPT"); row "A10 legacy-wrap" $? "$out"
out=$(a11_bad_prio real "$HOST_RECEIPT"); row "A11 bad-prio" $? "$out"
out=$(b1_train_green real "$AUTOPILOT" "$HOST_RECEIPT"); row "B1 train-green" $? "$out"
out=$(b2_dir_removed real "$AUTOPILOT" "$HOST_RECEIPT"); row "B2 dir-removed" $? "$out"
out=$(b3_dir_vanishes real "$AUTOPILOT" "$HOST_RECEIPT"); row "B3 dir-vanishes" $? "$out"
out=$(b4_dir_blocked real "$AUTOPILOT" "$HOST_RECEIPT"); row "B4 dir-blocked" $? "$out"
out=$(b5_unreachable real "$AUTOPILOT" "$HOST_RECEIPT"); row "B5 unreachable" $? "$out"
out=$(b6_env_no_toolchain real "$AUTOPILOT" "$HOST_RECEIPT"); row "B6 env-no-toolchain" $? "$out"
out=$(postpub_row b7-real "$AUTOPILOT" "post-publish dogfood NO-GO rc=1" FX_DOGFOOD_VERDICT=NO-GO); row "B7 postpub-nogo" $? "$out"
out=$(postpub_row b8-real "$AUTOPILOT" "post-publish dogfood refused on its receipt: receipt-20260921T000000Z.json still DEFERS declared:check_multiplatform_dogfood" 'FX_DEFERRED=["declared:check_multiplatform_dogfood"]'); row "B8 postpub-defer" $? "$out"
out=$(postpub_row b9-real "$AUTOPILOT" "post-publish dogfood refused on its receipt: no post-publish receipt" FX_NO_RECEIPT=1); row "B9 postpub-noreceipt" $? "$out"
out=$(b10_install_only real "$AUTOPILOT"); row "B10 install-only" $? "$out"
out=$(b11_postpub_alone real "$AUTOPILOT"); row "B11 postpub-alone" $? "$out"
out=$(b12_report_notes real "$AUTOPILOT"); row "B12 report-notes" $? "$out"
out=$(l1_ledger_pr real "$AUTOPILOT"); row "L1 ledger-pr" $? "$out"
out=$(l2_ledger_rerun real "$AUTOPILOT"); row "L2 ledger-rerun" $? "$out"
out=$(l3_ledger_empty real "$AUTOPILOT"); row "L3 ledger-empty" $? "$out"
out=$(l4_ledger_rejected real "$AUTOPILOT"); row "L4 ledger-rejected" $? "$out"
out=$(twin_row "$ROOT/scripts/lib"); row "T1 twin-equal" $? "$out"
out=$(twin_missing_row); row "T2 twin-missing" $? "$out"
out=$(p1_hold_release "$BANDLOCK"); row "P1 band-lock" $? "$out"
out=$(p2_bound "$BANDLOCK"); row "P2 band-bound" $? "$out"
out=$(p3_refused "$BANDLOCK"); row "P3 band-refused" $? "$out"
out=$(p4_no_rule "$BANDLOCK"); row "P4 band-no-rule" $? "$out"
out=$(run_band_row "$PARITY" cpu 0 free); row "R1 cpu-band" $? "$out"
out=$(run_band_row "$PARITY" accel 0 held); row "R2 accel-band" $? "$out"
out=$(run_band_row "$PARITY" accel 1 not-run FX_Q_REFUSE=1); row "R3 accel-refused" $? "$out"
out=$(run_band_row "$PARITY" accel 1 held GPU_BAND_TIMEOUT_S=2 FX_BODY_OVERRUN=1); row "R4 accel-bound" $? "$out"
out=$(g1_report_listed real "$GATE"); row "G1 report-listed" $? "$out"
out=$(g2_report_other_version real "$GATE"); row "G2 report-other-version" $? "$out"
out=$(g3_report_present real "$GATE"); row "G3 report-present" $? "$out"
out=$(g4_report_shape); row "G4 report-shape" $? "$out"
G5_PATCH='{"generate": null, "unmeasured": ["generate: `apr run --format json` exited 1: boom"]}'
G5_WANT='^FAIL +intel +generate is null: generate: `apr run --format json` exited 1: boom$'
G6_PATCH='{"generate": {"output_sane": false}}'; G6_WANT='^FAIL +intel +generate\.output_sane is not true'
G7_PATCH='{"sha256_measured": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}'; G7_WANT='^FAIL +intel +the \.crate the host downloaded \(eeeeeeeeeeee\) is not the one crates\.io published'
G8_PATCH='{"unmeasured": []}'; G8_WANT='^FAIL +intel +unmeasured\[\] is empty or absent'
out=$(gate_block_row g5-real "$GATE" "$G5_PATCH" "$G5_WANT"); row "G5 gen-null" $? "$out"
out=$(gate_block_row g6-real "$GATE" "$G6_PATCH" "$G6_WANT"); row "G6 gen-insane" $? "$out"
out=$(gate_block_row g7-real "$GATE" "$G7_PATCH" "$G7_WANT"); row "G7 hash-differs" $? "$out"
out=$(gate_block_row g8-real "$GATE" "$G8_PATCH" "$G8_WANT"); row "G8 unmeasured-empty" $? "$out"
out=$(gate_block_row g9-real "$GATE" "$G6_PATCH" '^REPORT intel +generate/sha256/unmeasured not required for 0\.65\.2' 0.65.2); row "G9 grandfathered" $? "$out"
BENCH_PATCH='{"bench_attempt": {"status": "refused", "rc": 1, "reason": "FAIL  apr bench exited non-zero"}}'
G10_WANT='^FAIL +intel +no bench block: the producer refused on this host: FAIL  apr bench exited non-zero$'
# a VALID parity block with every lane below its floor: the committed 0.65.2 lambda receipt's
python3 - "$ROOT/evidence/dogfood/0.65.2/lambda.json" "$TMP/floor-patch.json" "$TMP/nocuda-patch.json" <<'PY' || env_die "cannot build the parity-floor fixtures from evidence/dogfood/0.65.2/lambda.json"
import json, sys
blk = json.load(open(sys.argv[1]))["parity"]
json.dump({"parity": blk, "accelerator": "NVIDIA GeForce RTX 4090 sm_89"}, open(sys.argv[2], "w"))
nocuda = dict(blk); nocuda["lanes"] = [l for l in blk["lanes"] if l.get("lane") != "cuda"]
assert nocuda["lanes"] and len(nocuda["lanes"]) < len(blk["lanes"])
json.dump({"parity": nocuda, "accelerator": "NVIDIA GeForce RTX 4090 sm_89"}, open(sys.argv[3], "w"))
PY
G13_WANT='^REPORT intel +parity: a lane is below its declared floor -- runs as REPORT for 9\.9\.9, owed by #2844$'
out=$(gate_block_row g13-real "$GATE" "@$TMP/floor-patch.json" "$G13_WANT" 9.9.9 "9.9.9:intel:parity:#2844"); row "G13 report-floor (listed)" $? "$out"
out=$(gate_block_row g13b-real "$GATE" "@$TMP/floor-patch.json" '^FAIL +intel +a parity lane is below its declared floor$'); row "G13b report-floor (unlisted FAILs)" $? "$out"
out=$(gate_block_row g14-real "$GATE" "@$TMP/nocuda-patch.json" 'needs lane\(s\) cuda, receipt has: cpu +-- runs as REPORT for 9\.9\.9, owed by #3805$' 9.9.9 "9.9.9:intel:parity:#3805"); row "G14 report-lanes" $? "$out"
out=$(gate_block_row g10-real "$GATE" "$BENCH_PATCH" "$G10_WANT"); row "G10 bench-required" $? "$out"
out=$(gate_block_row g11-real "$GATE" "$BENCH_PATCH" '^REPORT intel +no bench block: runs as REPORT for 9\.9\.9, owed by #7; the producer refused on this host: FAIL  apr bench exited non-zero$' 9.9.9 "9.9.9:intel:bench:#7"); row "G11 bench-listed" $? "$out"
out=$(gate_block_row g12-real "$GATE" "$BENCH_PATCH" '^REPORT intel +no bench block \(hand-made receipt, pre-#3731\): the producer refused' 0.65.2); row "G12 bench-grandfathered" $? "$out"
jqe="jq"" -e"   # built, so this file does not match its own search
out=$(grep -n -- "$jqe" "$AUTOPILOT" "$HOST_RECEIPT"); [ -z "$out" ]; row "S1 no-jq-e" $? "found: $out"

echo "=== mutants: each must turn its row RED ==="
killed() { # killed NAME ROW-RC
    rows=$((rows + 1))
    if [ "$2" = 1 ]; then printf 'ok    mutant %s killed\n' "$1"
    elif [ "$2" = 2 ]; then env_die "mutant $1 could not build its fixture"
    else printf 'FAIL  mutant %s SURVIVED (its row stayed green)\n' "$1" >&2; fails=$((fails + 1)); fi
}
a1_gpu_linux m1 "$M/hr-drop-accel-line.sh" > /dev/null; killed hr-drop-accel-line $?
a1_gpu_linux m2 "$M/hr-no-gpu-wrap.sh" > /dev/null; killed hr-no-gpu-wrap $?
a1_gpu_linux m2b "$M/hr-gpuq-not-preferred.sh" > /dev/null; killed hr-gpuq-not-preferred $?
a1_gpu_linux m2c "$M/hr-parity-nested.sh" > /dev/null; killed hr-parity-nested $?
a2_darwin m3 "$M/hr-os-literal.sh" > /dev/null; killed hr-os-literal $?
a4_parity_refuses m4 "$M/hr-drop-attempt.sh" > /dev/null; killed hr-drop-attempt $?
a5_insane m5 "$M/hr-sane-always.sh" > /dev/null; killed hr-sane-always $?
a6_two_hashes m6 "$M/hr-merge-hashes.sh" > /dev/null; killed hr-merge-hashes $?
a7_bad_version m7 "$M/hr-no-version-check.sh" > /dev/null; killed hr-no-version-check $?
a8_nvsmi_fails m7b "$M/hr-nvsmi-any-rc.sh" > /dev/null; killed hr-nvsmi-any-rc $?
a9_gpu_fallback m7c "$M/hr-drop-fallback.sh" > /dev/null; killed hr-drop-fallback $?
b1_train_green m8 "$AUTOPILOT" "$M/hr-version-key.sh" > /dev/null; killed hr-version-key $?
b1_train_green m9 "$M/ap-steps-reordered.sh" "$HOST_RECEIPT" > /dev/null; killed ap-steps-reordered $?
b1_train_green m10 "$M/ap-no-receipts-env.sh" "$HOST_RECEIPT" > /dev/null; killed ap-no-receipts-env $?
b1_train_green m11 "$M/ap-hosts-literal.sh" "$HOST_RECEIPT" > /dev/null; killed ap-hosts-literal $?
b1_train_green m11b "$M/ap-receipt-prio.sh" "$HOST_RECEIPT" > /dev/null; killed ap-receipt-prio $?
b2_dir_removed m12 "$M/ap-drop-mkdir.sh" "$HOST_RECEIPT" > /dev/null; killed ap-drop-mkdir $?
b3_dir_vanishes m13 "$M/ap-no-receipt-to.sh" "$HOST_RECEIPT" > /dev/null; killed ap-no-receipt-to $?
b5_unreachable m14 "$M/ap-255-is-a-host.sh" "$HOST_RECEIPT" > /dev/null; killed ap-255-is-a-host $?
postpub_row m15 "$M/ap-drop-nogo-stop.sh" "post-publish dogfood NO-GO rc=1" FX_DOGFOOD_VERDICT=NO-GO > /dev/null; killed ap-drop-nogo-stop $?
postpub_row m16 "$M/ap-drop-receipt-read.sh" "post-publish dogfood refused on its receipt" 'FX_DEFERRED=["declared:check_multiplatform_dogfood"]' > /dev/null; killed ap-drop-receipt-read $?
b10_install_only m17 "$M/ap-install-dogfood.sh" > /dev/null; killed ap-install-dogfood $?
b12_report_notes m18 "$M/ap-drop-notes.sh" > /dev/null; killed ap-drop-notes $?
l1_ledger_pr m24 "$M/ap-ledger-no-push.sh" > /dev/null; killed ap-ledger-no-push $?
l1_ledger_pr m25 "$M/ap-ledger-armed.sh" > /dev/null; killed ap-ledger-armed $?
l2_ledger_rerun m26 "$M/ap-ledger-from-main.sh" > /dev/null; killed ap-ledger-from-main $?
l4_ledger_rejected m27 "$M/ap-ledger-no-stop.sh" > /dev/null; killed ap-ledger-no-stop $?
g2_report_other_version m19 "$M/gate-any-version.sh" > /dev/null; killed gate-any-version $?
a1_gpu_linux m28 "$M/hr-parity-wrapped.sh" > /dev/null; killed hr-parity-wrapped $?
p1_hold_release "$M/lock-release-noop.sh" > /dev/null; killed lock-release-noop $?
p2_bound "$M/lock-bound-kills-none.sh" > /dev/null; killed lock-bound-kills-none $?
run_band_row "$M/band-lock-any-class.sh" cpu 0 free > /dev/null; killed band-lock-any-class $?
b1_train_green m29 "$M/ap-no-twins.sh" "$HOST_RECEIPT" > /dev/null; killed ap-no-twins $?
twin_row "$M/lib-reader" > /dev/null; killed reader-yaml-only $?
twin_row "$M/lib-writer" > /dev/null; killed twin-writes-wrong $?
g3_report_present m20 "$M/gate-report-present.sh" > /dev/null; killed gate-report-present $?
gate_block_row m20b "$M/gate-floor-any.sh" "@$TMP/floor-patch.json" "$G13_WANT" 9.9.9 "9.9.9:intel:parity:#2844" > /dev/null; killed gate-floor-any $?
gate_block_row m21 "$M/gate-gen-any.sh" "$G6_PATCH" "$G6_WANT" > /dev/null; killed gate-gen-any $?
gate_block_row m21b "$M/gate-bench-absent-report.sh" "$BENCH_PATCH" "$G10_WANT" > /dev/null; killed gate-bench-absent-report $?
gate_block_row m22 "$M/gate-hash-any.sh" "$G7_PATCH" "$G7_WANT" > /dev/null; killed gate-hash-any $?
gate_block_row m23 "$M/gate-unmeasured-any.sh" "$G8_PATCH" "$G8_WANT" > /dev/null; killed gate-unmeasured-any $?

if [ "$fails" -eq 0 ]; then
    printf 'PASS  %s row(s) and mutant(s): the train emits, then judges, every host receipt; infra is never a host verdict\n' "$rows"
    exit 0
fi
printf 'FAIL  %s of %s row(s)/mutant(s) RED\n' "$fails" "$rows" >&2
exit 1
