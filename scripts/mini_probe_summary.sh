#!/usr/bin/env bash
# mini_probe_summary.sh — render the mini-probe measurement.
#
# WHY THIS EXISTS
# ---------------
# `.github/workflows/mini-probe.yml` is step (1) PROBE of the MINI doctrine
# change (operator ruling 2026-09-16, APR-RELEASE-001): dispatch the PR job set
# on mini-m4 against a sha green on intel and DIFF verdict + duration per job.
# Each probe leg drops one TSV row `job / verdict / duration_s /
# first_error_line` and a per-step detail file; this script is the only consumer.
# It runs in the workflow's `summary` job, which checks out the DISPATCHING
# branch (the tree under test is an older sha and does not contain this file).
#
# The table it prints is the artifact a human reads to decide the `rust-neutral`
# label (aprender#3402). Three columns exist because the verdict alone is not
# enough to make that decision:
#   * steps ran/pass/fail  — a PASS over 51 executed steps and a PASS over 1 are
#                            not the same evidence.
#   * skipped container/tool/other — a job whose steps are all `docker run`
#                            (mutants, workspace-test) is NOT evidence that mini
#                            can run it, and a SKIP(tool:) is paiml/infra#645's
#                            ticket, not a defect in the job.
#   * first error line     — so a FAIL names itself in the table instead of in a
#                            log nobody opens.
#
#   bash scripts/mini_probe_summary.sh <rows-dir> [--tsv <out.tsv>]
#   bash scripts/mini_probe_summary.sh --self-test
set -euo pipefail

# The ci.yml order of the neutral set, so the table reads like the pipeline.
JOB_ORDER="guard-tree guard-cargo vendored-schemas pr-review-shadow pr-review-sign mutants workspace-test gate"

md_cell() {
  # A pipe inside a cell ends the cell; a tab ends nothing but reads as noise.
  printf '%s' "$1" | tr '|\t' '/ ' | cut -c1-160
}

count_class() {
  # count_class <detail-file> <awk-predicate-on-$3>
  local detail="$1" pred="$2"
  if [ ! -f "$detail" ]; then
    printf '0'
    return
  fi
  awk -F'\t' -v pred="$pred" '
    {
      c = $3
      keep = 0
      if (pred == "ran"       && (c == "PASS" || c == "FAIL")) keep = 1
      if (pred == "pass"      && c == "PASS") keep = 1
      if (pred == "fail"      && c == "FAIL") keep = 1
      if (pred == "container" && c == "SKIP(container)") keep = 1
      if (pred == "tool"      && c ~ /^SKIP\(tool:/) keep = 1
      if (pred == "other"     && c ~ /^SKIP\(/ && c != "SKIP(container)" && c !~ /^SKIP\(tool:/) keep = 1
      n += keep
    }
    END { printf "%d", n + 0 }
  ' "$detail"
}

render() {
  local dir="$1" tsv="$2"
  local job row detail verdict dur err ran pass fail cont tool other

  : > "$tsv"
  printf '## mini-probe — the measured PR job set\n\n'
  printf 'Step (1) PROBE, operator ruling 2026-09-16 (APR-RELEASE-001). This table is a\n'
  printf 'MEASUREMENT: nothing here gates anything. `SKIP(container)` means the job is\n'
  printf 'docker-bound and this run is no evidence either way; `SKIP(tool:x)` is\n'
  printf 'paiml/infra#645, not a defect in the job.\n\n'
  printf '| job | verdict | duration_s | steps ran/pass/fail | skipped container/tool/other | first error line |\n'
  printf '| --- | --- | --- | --- | --- | --- |\n'

  for job in $JOB_ORDER; do
    row="${dir}/row-${job}.tsv"
    detail="${dir}/detail-${job}.tsv"
    if [ -f "$row" ]; then
      head -1 "$row" >> "$tsv"
      verdict="$(cut -f2 "$row" | head -1)"
      dur="$(cut -f3 "$row" | head -1)"
      err="$(cut -f4 "$row" | head -1)"
    elif [ -f "${dir}/not-dispatched-${job}.txt" ]; then
      verdict="(not dispatched)"
      dur="-"
      err="$(head -1 "${dir}/not-dispatched-${job}.txt")"
    else
      verdict="(no row)"
      dur="-"
      err="no artifact — the leg uploaded nothing (cancelled, or it died before the runner)"
    fi
    ran="$(count_class "$detail" ran)"
    pass="$(count_class "$detail" pass)"
    fail="$(count_class "$detail" fail)"
    cont="$(count_class "$detail" container)"
    tool="$(count_class "$detail" tool)"
    other="$(count_class "$detail" other)"
    printf '| %s | %s | %s | %s/%s/%s | %s/%s/%s | %s |\n' \
      "$(md_cell "$job")" "$(md_cell "$verdict")" "$(md_cell "$dur")" \
      "$ran" "$pass" "$fail" "$cont" "$tool" "$other" "$(md_cell "$err")"
  done

  printf '\n`%s` carries the rows verbatim (job / verdict / duration_s / first_error_line).\n' "$tsv"
}

self_test() {
  local tmp rc out rows
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf \"${tmp:?}\"" EXIT
  rc=0
  rows=0

  printf 'guard-tree\tPASS\t412\t\n' > "${tmp}/row-guard-tree.tsv"
  printf 'guard-tree\t001\tPASS\t3\tfirst guard\t\n' > "${tmp}/detail-guard-tree.tsv"
  printf 'guard-tree\t002\tSKIP(container)\t0\tdocker step\tdocker\n' >> "${tmp}/detail-guard-tree.tsv"
  printf 'guard-cargo\tFAIL\t900\t012 cargo step: error: linker `cc` not found\n' > "${tmp}/row-guard-cargo.tsv"
  printf 'guard-cargo\t012\tFAIL\t88\tcargo step\terror: linker not found\n' > "${tmp}/detail-guard-cargo.tsv"
  printf 'mutants\tSKIP(container)\t2\t\n' > "${tmp}/row-mutants.tsv"
  printf 'pr-review-sign\tSKIP(tool:gsed)\t9\t\n' > "${tmp}/row-pr-review-sign.tsv"
  printf 'pr-review-sign\t003\tSKIP(tool:gsed)\t1\tsign the receipt\t\n' > "${tmp}/detail-pr-review-sign.tsv"
  echo "include_workspace_test=false — workspace-test was not probed" \
    > "${tmp}/not-dispatched-workspace-test.txt"

  out="$(render "$tmp" "${tmp}/out.tsv")"

  check() {
    rows=$((rows + 1))
    if printf '%s' "$out" | grep -qF "$1"; then
      printf 'ok    %s\n' "$2"
    else
      printf 'FAIL  %s (expected to find: %s)\n' "$2" "$1"
      rc=1
    fi
  }
  refute() {
    rows=$((rows + 1))
    if printf '%s' "$out" | grep -qF "$1"; then
      printf 'FAIL  %s (must NOT contain: %s)\n' "$2" "$1"
      rc=1
    else
      printf 'ok    %s\n' "$2"
    fi
  }

  check '| guard-tree | PASS | 412 | 1/1/0 | 1/0/0 |' 'a PASS row carries its duration and its step counts'
  check '| guard-cargo | FAIL | 900 | 1/0/1 | 0/0/0 |' 'a FAIL row counts the failing step'
  check 'error: linker `cc` not found' 'the first error line reaches the table'
  check '| mutants | SKIP(container) | 2 |' 'a container-bound job is not reported as PASS'
  check '| pr-review-sign | SKIP(tool:gsed) | 9 |' 'a missing GNU tool is a SKIP(tool:), not a FAIL'
  check '| workspace-test | (not dispatched) |' 'an undispatched leg has no verdict at all'
  check '| gate | (no row) |' 'a leg that uploaded nothing is named, never assumed green'
  refute '| gate | PASS |' 'a missing row never renders as PASS'

  # The TSV is the rows verbatim: one line per row file present, four columns.
  local lines cols
  lines="$(wc -l < "${tmp}/out.tsv" | tr -d ' ')"
  rows=$((rows + 2))
  if [ "$lines" = "4" ]; then
    printf 'ok    the TSV carries one line per present row (4)\n'
  else
    printf 'FAIL  the TSV carries %s lines, expected 4\n' "$lines"
    rc=1
  fi
  cols="$(awk -F'\t' 'NR==1{print NF}' "${tmp}/out.tsv")"
  if [ "$cols" = "4" ]; then
    printf 'ok    the TSV row shape is job/verdict/duration_s/first_error_line\n'
  else
    printf 'FAIL  the TSV row has %s columns, expected 4\n' "$cols"
    rc=1
  fi

  # A pipe in an error line must not forge a table column.
  printf 'gate\tFAIL\t1\tboom | forged | column\n' > "${tmp}/row-gate.tsv"
  out="$(render "$tmp" "${tmp}/out.tsv")"
  check 'boom / forged / column' 'a pipe inside an error line cannot forge a column'

  if [ "$rc" -eq 0 ]; then
    printf '✓ mini_probe_summary self-test: %s/%s rows\n' "$rows" "$rows"
  else
    printf '✗ mini_probe_summary self-test FAILED\n' >&2
  fi
  return "$rc"
}

main() {
  local dir="" tsv=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --self-test)
        self_test
        return "$?"
        ;;
      --tsv)
        tsv="${2:?--tsv needs a path}"
        shift 2
        ;;
      -h|--help)
        printf 'usage: %s <rows-dir> [--tsv <out.tsv>] | --self-test\n' "$0"
        return 0
        ;;
      *)
        dir="$1"
        shift
        ;;
    esac
  done
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    printf 'mini_probe_summary: need a directory of mini-probe row artifacts\n' >&2
    return 2
  fi
  if [ -z "$tsv" ]; then
    tsv="${dir}/mini-probe-summary.tsv"
  fi
  render "$dir" "$tsv"
}

main "$@"
