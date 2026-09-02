#!/usr/bin/env bash
# check_perf_concurrency_groups.sh — PP-19 host isolation, statically.
#
# THE RULE (PP-LLAMA-001 §5.4, PP-19)
#   One global CI concurrency group per host, `cancel-in-progress: false`,
#   shared with any job that contends the host.
#
# WHY A STATIC GUARD
#   GitHub concurrency groups are REPO-WIDE, so the same group name in two
#   different workflows serialises them. That property is what makes PP-19
#   implementable at all — and it is invisible unless something reads every
#   workflow at once. At the time this was written, `grep -rn 'group:.*perf'`
#   over .github/workflows/ returned NOTHING, while gx10 was driven by
#   cuda-nightly (01:30 UTC, group `cuda-nightly`) AND silicon-nightly's
#   `aarch64-cuda-sm121` (03:30 UTC, no group at all), both with 90-minute
#   timeouts. Overlap was reachable, and a measurement taken during overlap is
#   not a measurement of the machine.
#
#   The intel perf lanes were worse than ungrouped: `bench-${{ github.ref }}`
#   and `beat-speed-${{ github.ref }}` with `cancel-in-progress: true`. A
#   ref-scoped group serialises a lane against ITSELF and against nothing else,
#   and cancel-in-progress means a queued run KILLS a measurement mid-window.
#
# WHAT IS CHECKED
#   A job is PERF-SENSITIVE when either
#     (a) its `runs-on` names a GPU label — the vocabulary is
#         scripts/check_runner_labels.sh's DISCRIM list, not a new one — or
#     (b) one of its steps runs `--ignored` tests, which is how every
#         wall-clock beat/bench target in this repo is invoked.
#   Such a job must declare, AT JOB LEVEL, `concurrency.group` matching
#   `perf-<host>` for a host the matrix declares, with
#   `cancel-in-progress: false`. Job level, not workflow level: two workflows
#   must be able to share one host's group.
#
#   bash scripts/check_perf_concurrency_groups.sh              # gate
#   bash scripts/check_perf_concurrency_groups.sh --dir DIR    # gate a fixture
#   bash scripts/check_perf_concurrency_groups.sh --selftest   # case table
#   bash scripts/check_perf_concurrency_groups.sh --list-selftests
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DIR="${REPO_ROOT}/.github/workflows"
DEFAULT_MATRIX="${REPO_ROOT}/scripts/perf-matrix.yaml"

SELFTEST_NAMES="isolation_breach isolation_ok cancel_true_is_red ignored_bench_without_group_is_red ref_scoped_group_is_red"

scan() {
    python3 - "$1" "$2" <<'PY'
import glob
import os
import re
import sys

workflow_dir, matrix_path = sys.argv[1], sys.argv[2]

try:
    import yaml
except ImportError:
    print("  RED   PyYAML is not importable; the workflows could not be parsed")
    sys.exit(1)

# The host vocabulary is the matrix's, so a new host does not need this file
# edited (PP-33: the declaration lives in one place).
FALLBACK_HOSTS = ("gx10", "intel", "lambda", "mini")
hosts = None
try:
    with open(matrix_path, encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    declared = matrix.get("hosts")
    if isinstance(declared, dict) and declared:
        hosts = tuple(sorted(str(h) for h in declared))
except (OSError, yaml.YAMLError):
    hosts = None
if hosts is None:
    hosts = FALLBACK_HOSTS
    print("  note  %s hosts unreadable; using the fallback host set %s"
          % (matrix_path, list(FALLBACK_HOSTS)))

GROUP_RE = re.compile(r"^perf-(%s)$" % "|".join(re.escape(h) for h in hosts))

# Same vocabulary as scripts/check_runner_labels.sh DISCRIM, GPU half only.
GPU_LABELS = {"gpu", "gx10", "cuda", "blackwell", "gb10", "ada", "rtx4090"}


def labels_of(runs_on):
    if isinstance(runs_on, str):
        return {part.strip().lower() for part in runs_on.split(",")}
    if isinstance(runs_on, list):
        return {str(part).strip().lower() for part in runs_on}
    if isinstance(runs_on, dict):          # {group: …, labels: […]}
        return labels_of(runs_on.get("labels")) | labels_of(runs_on.get("group"))
    return set()


def runs_ignored_tests(job):
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        body = step.get("run")
        if isinstance(body, str) and re.search(r"(^|\s)--ignored(\s|$)", body):
            return True
    return False


def concurrency_of(job):
    """(group, cancel_in_progress) as declared AT JOB LEVEL, or (None, None)."""
    block = job.get("concurrency")
    if isinstance(block, str):
        return block, False                # the shorthand: cancel defaults false
    if isinstance(block, dict):
        return block.get("group"), block.get("cancel-in-progress", False)
    return None, None


findings = 0
jobs_seen = 0
sensitive_seen = 0
paths = sorted(set(glob.glob(os.path.join(workflow_dir, "*.yml"))
                   + glob.glob(os.path.join(workflow_dir, "*.yaml"))))
for path in paths:
    try:
        with open(path, encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        print("  RED   %s does not parse (%s)" % (os.path.basename(path), exc))
        findings += 1
        continue
    jobs = doc.get("jobs")
    if not isinstance(jobs, dict):
        continue
    for name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        jobs_seen += 1
        by_runner = bool(labels_of(job.get("runs-on")) & GPU_LABELS)
        by_steps = runs_ignored_tests(job)
        if not (by_runner or by_steps):
            continue
        sensitive_seen += 1
        why = "gpu runs-on" if by_runner else "runs --ignored tests"
        if by_runner and by_steps:
            why = "gpu runs-on + --ignored tests"
        where = "%s:%s" % (os.path.basename(path), name)
        group, cancel = concurrency_of(job)
        if group is None:
            print("  RED   %s (%s) declares no job-level concurrency.group; "
                  "PP-19 requires perf-<host> so every consumer of that host "
                  "serialises against every other one, across workflows"
                  % (where, why))
            findings += 1
            continue
        if not GROUP_RE.match(str(group)):
            print("  RED   %s (%s) concurrency.group is %r, not perf-<host> "
                  "for a host the matrix declares %s. A ref-scoped or "
                  "workflow-scoped group serialises the lane against itself "
                  "and against nothing else."
                  % (where, why, group, list(hosts)))
            findings += 1
            continue
        if cancel is not False:
            print("  RED   %s (%s) sets cancel-in-progress: %r. A queued run "
                  "must never kill a measurement mid-window."
                  % (where, why, cancel))
            findings += 1
            continue
        print("  ok    %s (%s) group=%s cancel-in-progress=false"
              % (where, why, group))

# Vacuity. A directory that matched nothing, or a parse that produced no jobs,
# would report zero findings and look exactly like compliance.
if jobs_seen == 0:
    print("  RED   (vacuity) no jobs parsed under %s — the scan is broken, "
          "not the workflows" % workflow_dir)
    sys.exit(1)
print("  %d job(s) scanned, %d perf-sensitive, %d finding(s)"
      % (jobs_seen, sensitive_seen, findings))
sys.exit(1 if findings else 0)
PY
}

gate() {
    printf '=== PP-19 perf host isolation (check_perf_concurrency_groups.sh) ===\n'
    printf 'workflows: %s\n' "$1"
    scan "$1" "$2"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'FAIL: a job that contends a perf host is not serialised against the\n'
        printf '      other consumers of that host. Declare, at JOB level:\n'
        printf '        concurrency:\n'
        printf '          group: perf-<host>\n'
        printf '          cancel-in-progress: false\n'
        return 1
    fi
    printf 'PASS\n'
    return 0
}

# --------------------------------------------------------------------------
SELFTEST_TMP=""
selftest_cleanup() {
    [ -n "$SELFTEST_TMP" ] && rm -rf "${SELFTEST_TMP:?refusing to rm an empty path}"
    return 0
}

_fixture() {  # dir, filename, concurrency-block(may be empty), extra-step
    mkdir -p "$1"
    {
        printf 'name: fixture\non:\n  schedule:\n    - cron: "0 1 * * *"\njobs:\n'
        printf '  measure:\n    runs-on: [self-hosted, gpu, gx10, cuda, blackwell]\n'
        if [ -n "$3" ]; then printf '%s\n' "$3"; fi
        printf '    steps:\n      - run: echo hello\n'
        if [ -n "$4" ]; then printf '%s\n' "$4"; fi
    } > "$1/$2"
}

_fixture_cpu_bench() {  # dir, filename, concurrency-block(may be empty)
    mkdir -p "$1"
    {
        printf 'name: fixture\non:\n  schedule:\n    - cron: "0 1 * * *"\njobs:\n'
        printf '  measure:\n    runs-on: [self-hosted, X64, Linux, clean-room]\n'
        if [ -n "$3" ]; then printf '%s\n' "$3"; fi
        printf '    steps:\n'
        printf '      - run: cargo test -p aprender-compute --lib -- --ignored --nocapture\n'
    } > "$1/$2"
}

selftest() {
    local tmp pass=0 fail=0 out rc got
    tmp="$(mktemp -d)" || return 2
    case "$tmp" in /tmp/*|/var/folders/*) : ;; *) printf 'refusing %s\n' "$tmp"; return 2 ;; esac
    SELFTEST_TMP="$tmp"
    trap selftest_cleanup EXIT

    _row() {  # name, expect(red|green), dir
        out=$(scan "$3" "$DEFAULT_MATRIX" 2>&1); rc=$?
        got=green
        [ "$rc" -eq 0 ] || got=red
        if [ "$got" = "$2" ]; then
            printf '  ok    %-40s expect=%s\n' "$1" "$2"; pass=$((pass + 1))
        else
            printf '  BROKE %-40s expected %s got %s: %s\n' "$1" "$2" "$got" "$out"
            fail=$((fail + 1))
        fi
    }

    # 1. MUST-FIRE: a gx10 job with no group at all. This is the state
    #    silicon-nightly's aarch64-cuda-sm121 job was in.
    _fixture "$tmp/breach" "w.yml" "" ""
    _row isolation_breach red "$tmp/breach"

    # 2. MUST-NOT-FIRE: the same job, correctly declared. A clean tree.
    _fixture "$tmp/ok" "w.yml" \
        "    concurrency:
      group: perf-gx10
      cancel-in-progress: false" ""
    _row isolation_ok green "$tmp/ok"

    # 3. The right group with the wrong cancellation policy. beat-speed-nightly
    #    and nightly-bench were both `cancel-in-progress: true`.
    _fixture "$tmp/cancel" "w.yml" \
        "    concurrency:
      group: perf-gx10
      cancel-in-progress: true" ""
    _row cancel_true_is_red red "$tmp/cancel"

    # 4. The predicate's SECOND half: a CPU clean-room job whose steps run
    #    `--ignored` beat/bench targets is perf-sensitive too. Without this row
    #    the whole `runs_ignored_tests` branch is untested and nightly-bench /
    #    beat-speed-nightly could go ungrouped unnoticed.
    _fixture_cpu_bench "$tmp/ignored" "w.yml" ""
    _row ignored_bench_without_group_is_red red "$tmp/ignored"

    # 5. A ref-scoped group is not a host group: it serialises the lane against
    #    itself and against nothing else. This was the live state of both intel
    #    perf lanes.
    _fixture "$tmp/refscoped" "w.yml" \
        '    concurrency:
      group: bench-${{ github.ref }}
      cancel-in-progress: false' ""
    _row ref_scoped_group_is_red red "$tmp/refscoped"

    printf '  %d passed, %d broken\n' "$pass" "$fail"
    [ "$fail" = 0 ]
}

DIR="$DEFAULT_DIR"
MATRIX="$DEFAULT_MATRIX"
while [ $# -gt 0 ]; do
    case "$1" in
        --selftest) selftest; exit $? ;;
        --list-selftests) printf '%s\n' $SELFTEST_NAMES; exit 0 ;;
        --dir) DIR="$2"; shift 2 ;;
        --matrix) MATRIX="$2"; shift 2 ;;
        *) printf 'usage: %s [--dir DIR] [--matrix PATH] [--selftest]\n' "$0" >&2; exit 2 ;;
    esac
done

gate "$DIR" "$MATRIX"
