#!/usr/bin/env bash
# cuda_module_key_gate.sh - #3759: the CUDA lib suites under the module-key guard, on every
# GPU host a release claims, before it publishes.
#
# WHAT THE GUARD IS. Kernels bake parameters into the PTX (epsilon, shapes, rope theta) and
# the executor caches each compiled module under a key the call site builds by hand. A key
# that leaves out a baked parameter hands every later request the first request's kernel.
# That shipped twice in one day: the FP8 activation cache (#3727), and RMSNorm keyed by shape
# alone, so a model with epsilon 1e-6 ran at 1e-5 and its special tokens came out of RMSNorm
# at 0.431x (#3759). crates/aprender-serve/src/cuda/executor/module_key_guard.rs proves
# "one key, one PTX" in debug and `cfg(test)` builds. A release build of the library does
# only the lookup, so the proof exists only where the CUDA test suites run.
#
# WHY A RELEASE GATE. PR CI's cuda-unit job runs those suites on yoga (sm_89) and only for PRs
# that touch GPU paths. The PTX is generated per device, and the default prefill path differs
# by compute capability (cc 89 batched, cc >= 120 serial), so a key complete on one card can
# be incomplete on another. The cop's ruling on #3759 (from dogfood section 14, #3768): the
# aprender-serve and aprender-gpu CUDA lib suites run under the guard on lambda AND gx10 as a
# pre-publish dogfood gate, with a mutant (one key, two epsilons) that the gate catches.
#
# TWO MODES.
#   --run --host LABEL   On a GPU host: build and run both suites, then write
#                        evidence/cuda-module-key/LABEL.json. The aprender-serve set is
#                        DERIVED exactly as ci.yml's cuda-unit derives it (the cuda binary's
#                        --list minus the default binary's), so a new cuda-gated test is
#                        selected the day it is written.
#   (no arguments)       RELEASE MODE, declared in Cargo.toml [package.metadata.dogfood].
#                        Reads the committed receipts and is RED when a required host has
#                        none, a receipt is not PASS, the mutant row did not run on the device
#                        and pass, any test skipped for want of a device, or the receipt was
#                        measured on CUDA source that differs from HEAD's.
#
# FRESHNESS IS BY CONTENT, NOT BY AGE. A receipt records the git tree ids of the two source
# trees the guard's subject lives in (KERNEL_TREES). The receipt is current while HEAD's trees
# are identical, and stale the moment either changes, whatever the date. No time window is
# invented here. Committing the receipt itself changes neither tree.
#
# THE MUTANT IS THE SUITE'S OWN ROW. rmsnorm_eps_tests_3759::
# the_module_key_guard_refuses_one_key_for_two_epsilons asks one key for two epsilons and
# asserts the guard refuses. It prints "SKIP #3759 guard row: no CUDA device" and returns when
# there is no device, which libtest counts as a pass. So the gate requires that the row RAN,
# PASSED and did not print its SKIP line. A receipt whose mutant was skipped proves nothing.
#
# A HOST-SIDE GATE, NOT A TREE GUARD. It needs a CUDA device, so it is not named check_*.sh
# (scripts/guard_tree.sh runs those bare, in a required job). The PR-time half, this file's
# case table over the receipt evaluator, is scripts/check_cuda_module_key_gate.sh.
#
# usage:
#   bash scripts/cuda_module_key_gate.sh                     # release mode
#   bash scripts/cuda_module_key_gate.sh --dir DIR           # release mode over a fixture tree
#   bash scripts/cuda_module_key_gate.sh --run --host LABEL  # measure this host
#   bash scripts/cuda_module_key_gate.sh --self-test         # evaluator case table
# exit: 0 green, 1 red, 2 cannot evaluate (bad usage, no git, no python3).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVIDENCE_DIR="${ROOT}/evidence/cuda-module-key"
# The hosts the release claims CUDA on, per the cop's ruling on #3759 (lambda sm_89, gx10 sm_121).
REQUIRED_HOSTS="lambda gx10"
# The guard's subject: the call sites that build module keys, and the kernels that generate PTX.
KERNEL_TREES=(crates/aprender-serve/src/cuda crates/aprender-gpu/src)
MUTANT="cuda::executor::rmsnorm_eps_tests_3759::the_module_key_guard_refuses_one_key_for_two_epsilons"
MUTANT_SKIP_LINE="SKIP #3759 guard row: no CUDA device"
# The host's shared device lock; --run takes it only around the device runs, never the builds.
GPU_LOCK="${APR_GPU_LOCK:-/tmp/apr-gpu.lock}"

command -v python3 >/dev/null 2>&1 || { echo "cuda_module_key_gate: python3 is required" >&2; exit 2; }

# trees_json REV - {"<tree>": "<git tree id>"} for KERNEL_TREES at REV.
trees_json() {
    local rev="$1" t id out="{" sep=""
    for t in "${KERNEL_TREES[@]}"; do
        id=$(git -C "$ROOT" rev-parse "${rev}:${t}" 2>/dev/null) || return 1
        out="${out}${sep}\"${t}\": \"${id}\""
        sep=", "
    done
    printf '%s}\n' "$out"
}

# evaluate DIR HEAD_TREES_JSON HOSTS - one line per required host; 0 green, 1 red.
evaluate() {
    python3 - "$1" "$2" "$3" "$MUTANT" <<'PY'
import json, os, sys

evidence_dir, head_trees, hosts, mutant = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3].split(), sys.argv[4]
failed = 0

def red(msg):
    global failed
    failed += 1
    print("  RED   " + msg)

for host in hosts:
    path = os.path.join(evidence_dir, host + ".json")
    if not os.path.isfile(path):
        red("%s: no receipt at %s; run `bash scripts/cuda_module_key_gate.sh --run --host %s` on it. "
            "A host that was not measured is not green." % (host, os.path.relpath(path), host))
        continue
    try:
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError) as exc:
        red("%s: receipt does not parse (%s)" % (host, exc))
        continue
    if r.get("host") != host:
        red("%s: receipt names host %r; a receipt describes the host in its file name" % (host, r.get("host")))
        continue
    if r.get("status") != "PASS":
        red("%s: status=%s reason=%s" % (host, r.get("status"), r.get("reason")))
        continue
    m = r.get("mutant") or {}
    if m.get("name") != mutant or not m.get("ran") or m.get("skipped") or not m.get("passed"):
        red("%s: the one-key-two-epsilons mutant did not run on the device and pass (%s); "
            "without it the suite cannot show the guard is compiled in" % (host, json.dumps(m, sort_keys=True)))
        continue
    skips = r.get("device_skips")
    if not isinstance(skips, int) or skips != 0:
        red("%s: device_skips=%r; a test that skipped for want of a device did not run under the guard" % (host, skips))
        continue
    for key in ("serve_passed", "gpu_passed"):
        v = r.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            red("%s: %s=%r; a suite that ran nothing is not a pass" % (host, key, v))
            break
    else:
        trees = r.get("trees") or {}
        stale = [t for t in head_trees if trees.get(t) != head_trees[t]]
        if stale:
            red("%s: measured at %s on CUDA source that HEAD has since changed (%s); re-run it"
                % (host, str(r.get("commit"))[:12], ", ".join(stale)))
            continue
        print("  ok    %s: %s (cc %s) at %s, serve %d + gpu %d passed, mutant caught"
              % (host, r.get("gpu"), r.get("compute_cap"), str(r.get("commit"))[:12],
                 r["serve_passed"], r["gpu_passed"]))
sys.exit(1 if failed else 0)
PY
}

release_mode() {
    local dir="$1" head_trees
    head_trees=$(trees_json HEAD) || { echo "cuda_module_key_gate: cannot read HEAD's trees (${KERNEL_TREES[*]})" >&2; exit 2; }
    echo "cuda_module_key_gate: #3759 module-key guard receipts for: $REQUIRED_HOSTS"
    if evaluate "$dir" "$head_trees" "$REQUIRED_HOSTS"; then
        echo "cuda_module_key_gate: GREEN"
        return 0
    fi
    echo "cuda_module_key_gate: RED"
    return 1
}

# summary_count LOG WORD - the WORD count from libtest's last `test result:` line.
summary_count() {
    local n
    n=$(grep -E '^test result: ' "$1" | tail -1 | sed -nE "s/.* ([0-9]+) $2.*/\\1/p")
    printf '%s\n' "${n:-0}"
}

run_mode() {
    local host="$1" t log_serve log_gpu rc_serve rc_gpu started commit trees gpu cc
    command -v nvidia-smi >/dev/null 2>&1 || { echo "cuda_module_key_gate --run: no nvidia-smi on this host" >&2; exit 2; }
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    [ -n "$gpu" ] || { echo "cuda_module_key_gate --run: nvidia-smi lists no GPU" >&2; exit 2; }
    [ -z "$(git -C "$ROOT" status --porcelain -- "${KERNEL_TREES[@]}")" ] \
        || { echo "cuda_module_key_gate --run: uncommitted changes under ${KERNEL_TREES[*]}; a receipt describes a commit" >&2; exit 2; }
    commit=$(git -C "$ROOT" rev-parse HEAD)
    trees=$(trees_json HEAD) || exit 2
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    t=$(mktemp -d "${TMPDIR:-/tmp}/cuda-module-key.XXXXXX")
    export LC_ALL=C
    cd "$ROOT" || exit 2

    # 1. Derive the aprender-serve cuda-only set, as ci.yml's cuda-unit does.
    cargo test -p aprender-serve --lib --release -- --list 2>"$t/cpu-list.err" \
        | sed -n 's/: test$//p' | sort >"$t/cpu-list.txt"
    cargo test -p aprender-serve --features cuda --lib --release -- --list 2>"$t/cuda-list.err" \
        | sed -n 's/: test$//p' | sort >"$t/cuda-list.txt"
    comm -23 "$t/cuda-list.txt" "$t/cpu-list.txt" >"$t/cuda-only.txt"
    local n_only
    n_only=$(wc -l <"$t/cuda-only.txt")
    if [ "$n_only" -eq 0 ] || ! grep -qxF "$MUTANT" "$t/cuda-only.txt"; then
        echo "cuda_module_key_gate --run: derived cuda-only set has $n_only tests and the mutant is $(grep -qxF "$MUTANT" "$t/cuda-only.txt" && echo present || echo ABSENT)" >&2
        tail -5 "$t/cuda-list.err" >&2
        exit 2
    fi

    # 2. Build aprender-gpu's suite too, so the device lock below covers only device time.
    cargo test -p aprender-gpu --features cuda --lib --release --no-run >"$t/gpu-build.log" 2>&1 \
        || { echo "cuda_module_key_gate --run: aprender-gpu cuda lib tests do not build" >&2; tail -5 "$t/gpu-build.log" >&2; exit 2; }

    # 3. Run both on the device, one test at a time (#4043: parallel runs poison the context),
    #    under the host's shared GPU lock.
    log_serve="$t/serve.log"
    # shellcheck disable=SC2046 # test names contain no whitespace
    flock "$GPU_LOCK" cargo test -p aprender-serve --features cuda --lib --release -- \
        --exact --test-threads 1 --nocapture $(cat "$t/cuda-only.txt") >"$log_serve" 2>&1
    rc_serve=$?
    log_gpu="$t/gpu.log"
    flock "$GPU_LOCK" cargo test -p aprender-gpu --features cuda --lib --release -- \
        --test-threads 1 --nocapture >"$log_gpu" 2>&1
    rc_gpu=$?

    # 4. Judge, with ci.yml cuda-unit's anti-vacuity checks. Device skips print and return,
    #    which libtest counts as a pass; the grep is unanchored because under --nocapture the
    #    message lands on the `test X ... ` line. For the same reason the mutant's verdict is
    #    read from libtest's `failures:` list, not from an `ok` on its own line.
    local skips accounted mutant_ran mutant_passed mutant_skipped status reason
    local dev_skip='(CUDA|GPU) (executor |scheduler )?(unavailable|not available)|no CUDA device|CUDA model init failed'
    skips=$(cat "$log_serve" "$log_gpu" | grep -cE "$dev_skip")
    accounted=$(( $(summary_count "$log_serve" passed) + $(summary_count "$log_serve" ignored) ))
    mutant_ran=false; mutant_passed=false; mutant_skipped=false
    grep -qF "test ${MUTANT} ... " "$log_serve" && mutant_ran=true
    if [ "$mutant_ran" = true ] && [ "$rc_serve" -eq 0 ] && ! grep -qxF "    ${MUTANT}" "$log_serve"; then
        mutant_passed=true
    fi
    grep -qF "$MUTANT_SKIP_LINE" "$log_serve" && mutant_skipped=true
    status=PASS; reason=""
    if [ "$rc_serve" -ne 0 ] || [ "$rc_gpu" -ne 0 ]; then
        status=FAIL; reason="suite exit serve=$rc_serve gpu=$rc_gpu"
    elif [ "$accounted" -ne "$n_only" ]; then
        status=FAIL; reason="libtest accounted for $accounted of $n_only derived cuda-only tests"
    elif [ "$skips" -ne 0 ]; then
        status=FAIL; reason="$skips device skip line(s)"
    elif [ "$mutant_ran" != true ] || [ "$mutant_passed" != true ] || [ "$mutant_skipped" = true ]; then
        status=FAIL; reason="mutant ran=$mutant_ran passed=$mutant_passed skipped=$mutant_skipped"
    fi

    mkdir -p "$EVIDENCE_DIR"
    python3 - "$EVIDENCE_DIR/${host}.json" <<PY
import json, sys
r = {
    "schema": "cuda-module-key-v1",
    "issue": "#3759",
    "host": "${host}",
    "gpu": "${gpu}",
    "compute_cap": "${cc}",
    "commit": "${commit}",
    "trees": json.loads('''${trees}'''),
    "started_utc": "${started}",
    "serve_cuda_only": ${n_only},
    "serve_passed": $(summary_count "$log_serve" passed),
    "serve_failed": $(summary_count "$log_serve" failed),
    "serve_ignored": $(summary_count "$log_serve" ignored),
    "gpu_passed": $(summary_count "$log_gpu" passed),
    "gpu_failed": $(summary_count "$log_gpu" failed),
    "device_skips": ${skips},
    "mutant": {"name": "${MUTANT}", "ran": ${mutant_ran^}, "passed": ${mutant_passed^}, "skipped": ${mutant_skipped^}},
    "status": "${status}",
    "reason": "${reason}",
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(r, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
    echo "cuda_module_key_gate --run: $host $status ${reason:+($reason) }-> ${EVIDENCE_DIR#"$ROOT"/}/${host}.json; logs in $t"
    [ "$status" = PASS ]
}

# ---------------------------------------------------------------------------
# Case table over the evaluator: fixtures written to a temp dir, judged against fixed trees.
self_test() {
    local d fails=0 trees
    d=$(mktemp -d "${TMPDIR:-/tmp}/cuda-module-key-selftest.XXXXXX")
    trees='{"crates/aprender-serve/src/cuda": "aaaa", "crates/aprender-gpu/src": "bbbb"}'
    receipt() { # host status mutant_ran mutant_passed mutant_skipped device_skips serve_tree serve_passed
        python3 - "$d/$1.json" "$@" <<'PY'
import json, sys
path, host, status, ran, passed, skipped, skips, tree, served = sys.argv[1:10]
tf = lambda s: s == "true"
json.dump({"host": host, "status": status, "commit": "c0ffee", "gpu": "fixture", "compute_cap": "8.9",
           "trees": {"crates/aprender-serve/src/cuda": tree, "crates/aprender-gpu/src": "bbbb"},
           "serve_passed": int(served), "gpu_passed": 7, "device_skips": int(skips),
           "mutant": {"name": "cuda::executor::rmsnorm_eps_tests_3759::the_module_key_guard_refuses_one_key_for_two_epsilons",
                      "ran": tf(ran), "passed": tf(passed), "skipped": tf(skipped)}},
          open(path, "w"))
PY
    }
    row() { # name want_rc hosts setup...
        local name="$1" want="$2" hosts="$3" got
        shift 3
        rm -f "${d:?}"/*.json
        "$@"
        evaluate "$d" "$trees" "$hosts" >"$d/out" 2>&1
        got=$?
        if [ "$got" -eq "$want" ]; then printf 'ok    %s (rc %s)\n' "$name" "$got"; else
            printf 'FAIL  %s: rc %s, want %s\n' "$name" "$got" "$want"; sed 's/^/        /' "$d/out"; fails=$((fails + 1)); fi
    }
    both_green() { receipt lambda PASS true true false 0 aaaa 1661; receipt gx10 PASS true true false 0 aaaa 1661; }
    one_missing() { receipt lambda PASS true true false 0 aaaa 1661; }
    status_fail() { both_green; receipt gx10 FAIL true true false 0 aaaa 1661; }
    mutant_skipped() { both_green; receipt gx10 PASS true true true 0 aaaa 1661; }
    mutant_absent() { both_green; receipt gx10 PASS false false false 0 aaaa 1661; }
    mutant_failed() { both_green; receipt gx10 PASS true false false 0 aaaa 1661; }
    device_skip() { both_green; receipt gx10 PASS true true false 3 aaaa 1661; }
    stale_tree() { both_green; receipt gx10 PASS true true false 0 zzzz 1661; }
    ran_nothing() { both_green; receipt gx10 PASS true true false 0 aaaa 0; }
    wrong_host() { both_green; receipt gx10 PASS true true false 0 aaaa 1661; sed -i 's/"host": "gx10"/"host": "lambda"/' "$d/gx10.json"; }
    garbage() { both_green; printf '{not json' >"$d/gx10.json"; }
    row both_hosts_green_pass 0 "lambda gx10" both_green
    row a_required_host_missing_red 1 "lambda gx10" one_missing
    row status_fail_red 1 "lambda gx10" status_fail
    row mutant_skipped_red 1 "lambda gx10" mutant_skipped
    row mutant_absent_red 1 "lambda gx10" mutant_absent
    row mutant_failed_red 1 "lambda gx10" mutant_failed
    row device_skip_red 1 "lambda gx10" device_skip
    row stale_kernel_tree_red 1 "lambda gx10" stale_tree
    row suite_ran_nothing_red 1 "lambda gx10" ran_nothing
    row receipt_names_other_host_red 1 "lambda gx10" wrong_host
    row unparseable_receipt_red 1 "lambda gx10" garbage
    row empty_evidence_red 1 "lambda gx10" true
    rm -f "${d:?}"/*.json "${d:?}/out"
    rmdir "$d" 2>/dev/null
    if [ "$fails" -eq 0 ]; then echo "cuda_module_key_gate self-test: 12/12 rows hold"; return 0; fi
    echo "cuda_module_key_gate self-test: $fails row(s) FAILED"
    return 1
}

case "${1:-}" in
    "") release_mode "$EVIDENCE_DIR" ;;
    --dir) [ -n "${2:-}" ] || { echo "usage: --dir DIR" >&2; exit 2; }; release_mode "$2" ;;
    --run)
        [ "${2:-}" = "--host" ] && [ -n "${3:-}" ] || { echo "usage: --run --host LABEL" >&2; exit 2; }
        run_mode "$3" ;;
    --self-test) self_test ;;
    -h | --help) sed -n '2,57p' "$0" ;;
    *) echo "cuda_module_key_gate: unknown argument $1 (see --help)" >&2; exit 2 ;;
esac
