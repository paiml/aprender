#!/usr/bin/env bash
# check_workflow_env_defined.sh -- every ${VAR} a workflow `run:` block
# interpolates must resolve to something that job actually defines.
#
# aprender#2627. The model-tests step added by #2627 was written by copying
# workspace-test's docker mount lines, which read:
#
#     -v "/mnt/nvme-raid0/cargo-ci/registry/${PR_OR_REF}:/usr/local/cargo/registry"
#     -v "/mnt/nvme-raid0/targets/aprender-ci/${PR_OR_REF}/run-${GITHUB_RUN_ID}:/workspace/target"
#
# `PR_OR_REF` is JOB-level `env:` on workspace-test. The step went into
# guard-runner-labels, which never defined it. An undefined shell variable is
# the empty string, so the registry mount collapsed to the SHARED PARENT
# `/mnt/nvme-raid0/cargo-ci/registry/` -- bind-mounted as the cargo registry,
# whose children are PR numbers rather than cache/index/src -- and the target
# mount to `.../aprender-ci//run-<id>`, which no other step in the run shares.
#
# Nothing fails. Docker creates missing bind sources, cargo re-downloads, the
# step goes green, and the mechanism its own comment describes never engages.
# That is the "gate that cannot fail" class one level up: the step is real, its
# plumbing is not.
#
# What this checks: for every `run:` block in every workflow, every `$VAR` /
# `${VAR}` reference must be (a) an env key defined at workflow, job or step
# level, (b) assigned earlier in that same job's shell, (c) exported by a file
# the block `source`s, or (d) provided by the runner (GITHUB_*, RUNNER_*, ...).
#
# What it does NOT check: `${{ }}` GitHub expressions (substituted before bash
# ever runs, and a bad one is a workflow parse error, not a silent empty), and
# non-POSIX `shell:` blocks (pwsh/python have their own scoping rules).
#
# Usage:
#   bash scripts/check_workflow_env_defined.sh              # scan the repo
#   bash scripts/check_workflow_env_defined.sh --self-test  # case table

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# The scanner. Emits UNDEFINED rows on stdout; silent means clean.
#   $1 = workflow file, $2 = root to resolve `source`d paths against
# ---------------------------------------------------------------------------
scan_workflow() {
    awk -v src_root="$2" -v wf="$1" \
        -f "${REPO_ROOT}/scripts/lib/workflow_env_scan.awk" "$1"
}

# ---------------------------------------------------------------------------
# Self-test: a case table plus a fixture proving the enumerator can emit a row.
# ---------------------------------------------------------------------------
self_test() {
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' RETURN
    mkdir -p "${tmp}/scripts"
    printf 'export SOURCED_VAR=1\nPLAIN_IN_LIB=2\n' > "${tmp}/scripts/lib.sh"

    fail=0
    n=0

    # $1 = expectation (fail|pass), $2 = label, $3 = yaml body
    probe() {
        n=$((n + 1))
        printf '%s\n' "$3" > "${tmp}/wf.yml"
        scan_workflow "${tmp}/wf.yml" "${tmp}" > "${tmp}/out.txt" 2>&1
        rc=$?
        if [ "$1" = "fail" ] && [ "${rc}" -eq 0 ]; then
            echo "SELF-TEST FAIL [${2}]: expected a finding, got none"; fail=1
        elif [ "$1" = "pass" ] && [ "${rc}" -ne 0 ]; then
            echo "SELF-TEST FAIL [${2}]: expected clean, got:"; sed 's/^/    /' "${tmp}/out.txt"; fail=1
        fi
    }

    # --- must FAIL: the aprender#2627 defect itself, and its neighbours ----
    probe fail "the #2627 defect: mount interpolates a var the job never defines" \
'jobs:
  guard:
    steps:
      - run: |
          docker run -v "/reg/${PR_OR_REF}:/r" img'

    probe fail "bare \$VAR form, no braces" \
'jobs:
  guard:
    steps:
      - run: echo "$PR_OR_REF"'

    probe fail "defined in a DIFFERENT job is not defined here" \
'jobs:
  a:
    env:
      PR_OR_REF: 1
    steps:
      - run: echo "${PR_OR_REF}"
  b:
    steps:
      - run: echo "${PR_OR_REF}"'

    probe fail "assigned in a different job does not carry over" \
'jobs:
  a:
    steps:
      - run: PR_OR_REF=7
  b:
    steps:
      - run: echo "${PR_OR_REF}"'

    probe fail "\${VAR:-default} still needs VAR to exist as a name we know" \
'jobs:
  guard:
    steps:
      - run: echo "${UNKNOWN_THING:-x}"'

    # --- must PASS -------------------------------------------------------
    probe pass "job-level env (the fix)" \
'jobs:
  guard:
    env:
      PR_OR_REF: 1
    steps:
      - run: |
          docker run -v "/reg/${PR_OR_REF}:/r" img'

    probe pass "workflow-level env reaches every job" \
'env:
  GLOBAL_THING: 1
jobs:
  guard:
    steps:
      - run: echo "${GLOBAL_THING}"'

    probe pass "step-level env" \
'jobs:
  guard:
    steps:
      - env:
          STEP_THING: 1
        run: echo "${STEP_THING}"'

    probe pass "plain shell assignment" \
'jobs:
  guard:
    steps:
      - run: |
          THING=1
          echo "${THING}"'

    probe pass "assignment inside a case branch (binary-release.yml STRIP=)" \
'jobs:
  guard:
    steps:
      - run: |
          case "$1" in
            a) STRIP=aarch64-linux-musl-strip ;;
            *) STRIP="" ;;
          esac
          echo "$STRIP"'

    probe pass "while IFS= read -r NAME (book-contracts.yml)" \
'jobs:
  guard:
    steps:
      - run: |
          while IFS= read -r line; do
            echo "$line"
          done < f'

    probe pass "for NAME in" \
'jobs:
  guard:
    steps:
      - run: |
          for f in a b; do echo "$f"; done'

    probe pass "export in a sourced library (qwen-story-daily.yml \$APR)" \
'jobs:
  guard:
    steps:
      - run: |
          . scripts/lib.sh
          echo "$SOURCED_VAR $PLAIN_IN_LIB"'

    probe pass "runner-provided GITHUB_* / RUNNER_*" \
'jobs:
  guard:
    steps:
      - run: echo "${GITHUB_WORKSPACE} ${GITHUB_RUN_ID} ${RUNNER_OS}"'

    probe pass "\${{ }} expressions are not shell variables" \
'jobs:
  guard:
    steps:
      - run: echo "${{ github.event.pull_request.number }}"'

    probe pass "pwsh blocks are out of scope (nightly.yml \$archive)" \
'jobs:
  guard:
    steps:
      - shell: pwsh
        run: |
          $archive = "x"
          Write-Host "$archive"'

    probe pass "positional and special parameters" \
'jobs:
  guard:
    steps:
      - run: |
          echo "$1 $? $@ $# $$ $!"'

    # --- the enumerator itself must be able to produce a row --------------
    n=$((n + 1))
    printf '%s\n' 'jobs:
  guard:
    steps:
      - run: echo "${NOPE}"' > "${tmp}/fixture.yml"
    rows="$(scan_workflow "${tmp}/fixture.yml" "${tmp}" | wc -l)"
    if [ "${rows}" -lt 1 ]; then
        echo "SELF-TEST FAIL [enumerator]: fixture produced 0 rows -- the scan is inert"
        fail=1
    fi

    # --- and it must actually be READING the workflow directory -----------
    n=$((n + 1))
    found=0
    for f in "${REPO_ROOT}"/.github/workflows/*.yml; do
        [ -f "${f}" ] && found=$((found + 1))
    done
    if [ "${found}" -lt 1 ]; then
        echo "SELF-TEST FAIL [universe]: no workflows found under .github/workflows/"
        fail=1
    fi

    if [ "${fail}" -ne 0 ]; then
        echo "check_workflow_env_defined.sh: self-test FAILED"
        return 1
    fi
    echo "check_workflow_env_defined.sh: self-test passed (${n} cases, ${found} workflows in scope)"
    return 0
}

main() {
    if [ "${1:-}" = "--self-test" ]; then
        self_test
        return $?
    fi

    scanned=0
    bad=0
    for f in "${REPO_ROOT}"/.github/workflows/*.yml; do
        [ -f "${f}" ] || continue
        scanned=$((scanned + 1))
        scan_workflow "${f}" "${REPO_ROOT}" || bad=$((bad + 1))
    done

    if [ "${scanned}" -eq 0 ]; then
        echo "ERROR: scanned 0 workflows -- an empty scan is not a pass"
        return 1
    fi
    if [ "${bad}" -ne 0 ]; then
        echo ""
        echo "FAIL: ${bad} of ${scanned} workflow(s) interpolate a variable their job does not define."
        echo "An undefined shell variable is the empty string: the step runs, goes green,"
        echo "and silently stops doing what it says. Define it in the job's \`env:\` block."
        return 1
    fi
    echo "OK: ${scanned} workflows -- every run: block interpolates only names its job defines"
    return 0
}

main "$@"
