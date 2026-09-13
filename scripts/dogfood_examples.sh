#!/usr/bin/env bash
# scripts/dogfood_examples.sh — every workspace example BUILDS and RUNS.
#
# WHY THIS EXISTS
# ---------------
# CI compiles examples (`--examples`) and has never executed one. An example
# that compiles and then panics, or exits 1 on its first line, is published in
# the crate and in the book as working code. The operator's instruction
# (2026-09-11, #3121) is that running them is part of every tagged release:
#
#   "ensure-you update apr-cookbook and release notes and cargo run --examples
#    as part of pre-release to each tagged release -- dogfood skill"
#
# The gate rows live in .claude/skills/apr-dogfood/SKILL.md (G3.EX, G3.CB,
# G3.RN) and in the release runbook, docs/specifications/PP-LLAMA-001-MASTER.md
# §7.2. This script is the G3.EX body.
#
# THE UNIVERSE IS CARGO'S, NEVER A DIRECTORY LISTING
# --------------------------------------------------
# `cargo metadata --no-deps --format-version 1` is asked which targets have
# kind == ["example"]. A `find crates -path '*/examples/*.rs'` universe is wrong
# in both directions: it counts modules a `[[example]]` never names, and it
# misses an example declared with an explicit `path =`. It also cannot tell you
# the package that owns the target or its `required-features`, and running an
# example in the wrong package, or without its features, fails for a reason that
# has nothing to do with the example.
#
# A FIVE-WAY CLASSIFICATION, AND WHY `pass` IS THE NARROW ONE
# ----------------------------------------------------------
#   pass            rc 0.
#   fail            rc != 0, and not one of the rows below.
#   timeout         killed by the wrapper (rc 124 from timeout, 137 from KILL).
#   needs-args      rc != 0 AND stderr opens a clap usage line. An example that
#                   requires a CLI argument is not broken; it is not runnable
#                   bare. The row cites the line so the claim is checkable.
#   needs-hardware  stderr names a missing CUDA/wgpu device. Counts as a SKIP,
#                   never as a pass, and the row cites the line: a green run on
#                   a driver-less host must not be readable as "the CUDA example
#                   works". These rows are re-run on the CUDA host before the
#                   release verdict (G3.EX).
#
# Exit 1 if any row is `fail` or `timeout`. Exit 2 if the enumeration is EMPTY —
# a gate that finds nothing to run and exits 0 is the vacuity defect this repo
# names most often. No row is ever written without a class.
#
#   bash scripts/dogfood_examples.sh                      # the release run
#   bash scripts/dogfood_examples.sh --filter '^apr-cli'  # one package
#   bash scripts/dogfood_examples.sh --selftest           # hermetic case table
#
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
# Absolute: the selftest re-invokes this file from a scratch directory.
SELF="${SELF_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SELF_DIR}/.." > /dev/null 2>&1 && pwd)"
FIXTURE_DIR="${REPO_ROOT}/tests/fixtures/dogfood_examples"

CARGO_BIN="${CARGO:-cargo}"
TIMEOUT_SECS=120
FILTER=''
OUT=''
MANIFEST=''
SELFTEST=0

# THE TWO CLASSIFIER PATTERNS.
#
# Both are anchored on wording a TOOL emits, never on a bare English phrase.
# needs-args is clap's two openings; needs-hardware is what the CUDA runtime,
# the driver loader and wgpu print when the device is absent. Widening either
# one turns a real defect into a skip, so each addition belongs in the selftest
# table first.
NEEDS_ARGS_RE='^(Usage|error: the following required arguments)'
NEEDS_HW_RE='(CUDA_ERROR_[A-Z_]+|no CUDA-capable device|CUDA driver version is insufficient|cuInit|libcuda\.so|libnvidia-ml|[Nn]o (suitable )?(graphics )?adapter|RequestAdapterError|NoAdapter|wgpu.*(device|adapter) (not|un)|Metal device (not|un))'

# ---------------------------------------------------------------------------
# helpers

die() {
    printf 'dogfood_examples: %s\n' "$1" >&2
    exit "${2:-2}"
}

usage() {
    printf 'usage: dogfood_examples.sh [--timeout-secs N] [--filter REGEX]\n'
    printf '                           [--out TSV] [--manifest-path P] [--selftest]\n'
}

# oneline TEXT -- a TSV cell: no tab, no newline, bounded length.
oneline() {
    printf '%s' "$1" | tr '\t\n\r' '   ' | cut -c1-200
}

require_tools() {
    command -v "${CARGO_BIN}" > /dev/null 2>&1 || die "cargo not found (CARGO=${CARGO_BIN})"
    command -v jq > /dev/null 2>&1 || die 'jq not found; the enumeration parses cargo metadata JSON'
    command -v timeout > /dev/null 2>&1 || die 'timeout(1) not found; it is what classifies a hang'
}

# ---------------------------------------------------------------------------
# enumeration

# cargo_meta -- the metadata document, honouring --manifest-path.
cargo_meta() {
    if [ -n "${MANIFEST}" ]; then
        "${CARGO_BIN}" metadata --no-deps --format-version 1 --manifest-path "${MANIFEST}"
    else
        "${CARGO_BIN}" metadata --no-deps --format-version 1
    fi
}

# enumerate_examples META_FILE -- pkg \t example \t src_path \t required-features
#
# The `{p: .name, t: .targets[]}` product keeps the OWNING PACKAGE attached to
# every target; `required-features` is joined with commas so the row is one
# line and the caller can hand it straight to --features.
enumerate_examples() {
    jq -r '[.packages[] | {p: .name, t: .targets[]}]
           | map(select(.t.kind == ["example"]))
           | .[]
           | [.p, .t.name, .t.src_path, ((.t["required-features"] // []) | join(","))]
           | @tsv' "$1"
}

# workspace_version META_FILE -- the version the evidence directory is named for.
workspace_version() {
    local root ver
    root=$(jq -r '.workspace_root' "$1")
    ver=$(jq -r --arg mp "${root}/Cargo.toml" \
            '[.packages[] | select(.manifest_path == $mp) | .version] | first // ""' "$1")
    if [ -z "${ver}" ] || [ "${ver}" = "null" ]; then
        ver=$(jq -r '[.packages[].version] | first // "unknown"' "$1")
    fi
    printf '%s' "${ver}"
}

# ---------------------------------------------------------------------------
# the timeout wrapper — and the one place the selftest may defeat it

# run_bounded SECS CMD... -- the wrapper whose absence makes `hang` unclassifiable.
#
# --signal=KILL because a decode loop that ignores SIGTERM is a shape we have
# already met (42 hung llama-cli copies, load 157). rc 124 is timeout(1)'s own,
# 137 is 128+9 when the shell reports the KILL.
#
# THE MUTATION ARM. DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT=1 runs the command with
# NO wrapper, so a hang runs forever and can never be reported as `timeout`.
# It is honoured ONLY when DOGFOOD_EXAMPLES_SELFTEST=1 is also set, which only
# the selftest harness sets; set alone it is refused rather than ignored,
# because an ignored kill-switch is indistinguishable from an armed one.
run_bounded() {
    local secs="$1"
    shift
    if [ "${DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT:-0}" = '1' ] \
        && [ "${DOGFOOD_EXAMPLES_SELFTEST:-0}" = '1' ]; then
        "$@"
        return $?
    fi
    timeout --signal=KILL "${secs}" "$@"
    return $?
}

# ---------------------------------------------------------------------------
# classification

# classify RC LOG STAGE -- prints CLASS \t CITE
classify() {
    local rc="$1" log="$2" stage="$3" line
    if [ "${rc}" -eq 0 ]; then
        printf 'pass\t\n'
        return 0
    fi
    if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
        printf 'timeout\t%s killed after %ss\n' "${stage}" "${TIMEOUT_SECS}"
        return 0
    fi
    line=$(grep -m1 -E "${NEEDS_ARGS_RE}" "${log}" 2> /dev/null) || line=''
    if [ -n "${line}" ]; then
        printf 'needs-args\t%s\n' "$(oneline "${line}")"
        return 0
    fi
    line=$(grep -m1 -E "${NEEDS_HW_RE}" "${log}" 2> /dev/null) || line=''
    if [ -n "${line}" ]; then
        printf 'needs-hardware\t%s\n' "$(oneline "${line}")"
        return 0
    fi
    line=$(grep -m1 -E '[^[:space:]]' "${log}" 2> /dev/null) || line=''
    printf 'fail\t%s: %s\n' "${stage}" "$(oneline "${line:-no output}")"
}

# ---------------------------------------------------------------------------
# one example

# build_then_run PKG NAME FEATS LOG -- rc of the first stage that failed,
# printing the stage name on stdout.
#
# The build is NOT wrapped: a build that hangs is cargo's lock or the host, a
# different defect with a different remedy, and wrapping it would let a slow
# cold build be reported as an example that hangs. Only the RUN is bounded.
build_then_run() {
    local pkg="$1" name="$2" feats="$3" log="$4" rc=0
    local -a cargs=()
    if [ -n "${feats}" ]; then
        cargs+=(--features "${feats}")
    fi
    # -p resolves against the workspace cargo DISCOVERS, which is the one in the
    # current directory unless it is told otherwise. Without this the selftest's
    # scratch package was looked up in aprender's workspace and every row came
    # back `fail|101|package ID specification ... did not match any packages` --
    # five green-looking classifications of a defect in the harness.
    if [ -n "${MANIFEST}" ]; then
        cargs+=(--manifest-path "${MANIFEST}")
    fi

    "${CARGO_BIN}" build -q --example "${name}" -p "${pkg}" "${cargs[@]}" \
        > "${log}" 2>&1 || rc=$?
    if [ "${rc}" -ne 0 ]; then
        printf 'build\n'
        return "${rc}"
    fi
    run_bounded "${TIMEOUT_SECS}" \
        "${CARGO_BIN}" run -q --example "${name}" -p "${pkg}" "${cargs[@]}" \
        < /dev/null > "${log}" 2>&1 || rc=$?
    printf 'run\n'
    return "${rc}"
}

# ---------------------------------------------------------------------------
# the run

main_run() {
    local meta_file rows count out ver ws_root
    local td pkg name feats stage rc secs t0 class cite row note
    local n_pass=0 n_fail=0 n_timeout=0 n_args=0 n_hw=0

    require_tools
    td=$(mktemp -d)
    # shellcheck disable=SC2064  # expand td now: the trap must not depend on a later value
    trap "rm -rf \"${td:?}\"" EXIT

    meta_file="${td}/metadata.json"
    cargo_meta > "${meta_file}" || die 'cargo metadata failed; the universe is unknown' 2
    ws_root=$(jq -r '.workspace_root' "${meta_file}")
    ver=$(workspace_version "${meta_file}")

    rows=$(enumerate_examples "${meta_file}")
    if [ -n "${FILTER}" ]; then
        rows=$(printf '%s\n' "${rows}" | awk -F'\t' -v re="${FILTER}" \
                 '$1 "::" $2 ~ re { print }') || rows=''
    fi
    count=$(printf '%s\n' "${rows}" | grep -c . || true)

    if [ "${count}" -eq 0 ]; then
        printf 'FAIL (vacuity): cargo metadata reports 0 example targets'
        if [ -n "${FILTER}" ]; then
            printf ' matching %s' "${FILTER}"
        fi
        printf '.\nA run with nothing to run is not a pass.\n'
        exit 2
    fi

    out="${OUT}"
    [ -n "${out}" ] || out="${ws_root}/evidence/dogfood/${ver}/examples.tsv"
    mkdir -p "$(dirname "${out}")"
    {
        printf '# dogfood_examples.sh version=%s examples=%s timeout_secs=%s\n' \
            "${ver}" "${count}" "${TIMEOUT_SECS}"
        printf '# pkg\texample\tclass\trc\tsecs\tcite\n'
    } > "${out}"

    printf '=== every example builds and runs (dogfood_examples.sh) ===\n'
    printf '%s example target(s), version %s, timeout %ss\n' "${count}" "${ver}" "${TIMEOUT_SECS}"

    while IFS=$'\t' read -r pkg name _src feats; do
        [ -n "${pkg}" ] || continue
        t0=${SECONDS}
        rc=0
        stage=$(build_then_run "${pkg}" "${name}" "${feats}" "${td}/example.log") || rc=$?
        secs=$((SECONDS - t0))
        IFS=$'\t' read -r class cite \
            < <(classify "${rc}" "${td}/example.log" "${stage}") || true
        [ -n "${class}" ] || { class='fail'; cite='unclassified outcome'; }
        row=$(printf '%s\t%s\t%s\t%s\t%s\t%s' \
                "${pkg}" "${name}" "${class}" "${rc}" "${secs}" "${cite}")
        printf '%s\n' "${row}" >> "${out}"
        note=''
        [ -z "${cite}" ] || note=" -- ${cite}"
        printf '%-14s %s::%s%s\n' "${class}" "${pkg}" "${name}" "${note}"
        case "${class}" in
            pass) n_pass=$((n_pass + 1)) ;;
            fail) n_fail=$((n_fail + 1)) ;;
            timeout) n_timeout=$((n_timeout + 1)) ;;
            needs-args) n_args=$((n_args + 1)) ;;
            needs-hardware) n_hw=$((n_hw + 1)) ;;
        esac
    done < <(printf '%s\n' "${rows}")

    printf '# summary pass=%s fail=%s timeout=%s needs-args=%s needs-hardware=%s\n' \
        "${n_pass}" "${n_fail}" "${n_timeout}" "${n_args}" "${n_hw}" >> "${out}"
    printf 'wrote %s\n' "${out}"
    printf 'summary pass=%s fail=%s timeout=%s needs-args=%s needs-hardware=%s\n' \
        "${n_pass}" "${n_fail}" "${n_timeout}" "${n_args}" "${n_hw}"

    if [ "${n_fail}" -gt 0 ] || [ "${n_timeout}" -gt 0 ]; then
        printf 'FAIL: %s example(s) failed, %s timed out.\n' "${n_fail}" "${n_timeout}"
        exit 1
    fi
    printf 'PASS\n'
    exit 0
}

# ---------------------------------------------------------------------------
# selftest

ST_FAILED=0

st_row() {
    if [ "$1" = 'PASS' ]; then
        printf 'PASS  %s\n' "$2"
    else
        ST_FAILED=$((ST_FAILED + 1))
        printf 'FAIL  %s\n' "$2"
        [ -z "${3:-}" ] || printf '      %s\n' "$3"
    fi
}

# st_expect ROW TSV PKG EXAMPLE WANT_CLASS -- one classification assertion.
st_expect() {
    local label="$1" tsv="$2" pkg="$3" name="$4" want="$5" got
    got=$(awk -F'\t' -v p="${pkg}" -v n="${name}" '$1 == p && $2 == n { print $3 "|" $4 "|" $6 }' \
            "${tsv}") || got=''
    case "${got}" in
        "${want}"'|'*) st_row PASS "${label} (${got})" ;;
        *) st_row FAIL "${label}" "want class ${want}, got '${got:-<no row>}'" ;;
    esac
}

# st_make_ws DEST -- the scratch workspace, from the committed fixture.
st_make_ws() {
    local dest="$1"
    [ -d "${FIXTURE_DIR}/examples" ] || return 3
    mkdir -p "${dest}"
    cp -R "${FIXTURE_DIR}/examples" "${FIXTURE_DIR}/src" "${dest}/"
    cp "${FIXTURE_DIR}/Cargo.toml.in" "${dest}/Cargo.toml"
}

selftest() {
    local td rc out ws
    require_tools
    td=$(mktemp -d)
    # shellcheck disable=SC2064  # expand td now, not at trap time
    trap "rm -rf \"${td:?}\"" EXIT
    ws="${td}/ws"

    printf '=== dogfood_examples.sh selftest ===\n'
    printf 'scratch: %s\n' "${td}"

    # Errexit is SUSPENDED inside a tested context, so the fixture is built in a
    # subshell that re-arms it: a cp that dies mid-fixture must not leave every
    # later row asserting against a half-built workspace and still printing PASS.
    local frc=0 detail
    detail=$(set -e; st_make_ws "${ws}" 2>&1) || frc=$?
    if [ "${frc}" -ne 0 ]; then
        st_row FAIL 'fixture: scratch workspace built' "rc=${frc} ${detail}"
        printf '1 row(s) failed\n'
        return 1
    fi
    st_row PASS 'fixture: scratch workspace built from tests/fixtures/dogfood_examples'

    # --- the measured run. Timeout 3 s: the run stage only, the build is not
    # wrapped, so this is not a race with rustc.
    out="${td}/examples.tsv"
    rc=0
    env CARGO_TARGET_DIR="${td}/target" \
        bash "${SELF}" --manifest-path "${ws}/Cargo.toml" --timeout-secs 3 --out "${out}" \
        > "${td}/run.log" 2>&1 || rc=$?

    if [ ! -f "${out}" ]; then
        st_row FAIL 'run: TSV written' "rc=${rc}; log: $(oneline "$(tail -5 "${td}/run.log")")"
        printf '1 row(s) failed\n'
        return 1
    fi
    st_row PASS 'run: TSV written'

    local p='dogfood-examples-fixture'
    st_expect 'class: ok -> pass' "${out}" "${p}" ok pass
    st_expect 'class: bad -> fail' "${out}" "${p}" bad fail
    st_expect 'class: hang -> timeout' "${out}" "${p}" hang timeout
    st_expect 'class: needs_arg -> needs-args' "${out}" "${p}" needs_arg needs-args
    st_expect 'class: nohw -> needs-hardware' "${out}" "${p}" nohw needs-hardware

    # rc of the failing example is recorded, not flattened to 1.
    if awk -F'\t' '$2 == "bad" && $4 == 3 { f = 1 } END { exit !f }' "${out}"; then
        st_row PASS 'rc: bad recorded rc=3'
    else
        st_row FAIL 'rc: bad recorded rc=3' \
            "$(oneline "$(awk -F'\t' '$2 == "bad" { print }' "${out}")")"
    fi

    # Every citing class cites a LINE, not an empty cell.
    if awk -F'\t' '$3 == "needs-args" || $3 == "needs-hardware" { if ($6 == "") bad = 1 } END { exit bad }' \
            "${out}"; then
        st_row PASS 'cite: needs-args and needs-hardware rows cite a line'
    else
        st_row FAIL 'cite: needs-args and needs-hardware rows cite a line'
    fi

    # Trailer counts.
    if grep -qxF '# summary pass=1 fail=1 timeout=1 needs-args=1 needs-hardware=1' "${out}"; then
        st_row PASS 'trailer: summary counts'
    else
        st_row FAIL 'trailer: summary counts' "$(oneline "$(grep '^# summary' "${out}" || true)")"
    fi

    # Exit contract: one fail and one timeout means rc 1.
    if [ "${rc}" -eq 1 ]; then
        st_row PASS 'exit: 1 with a fail and a timeout present'
    else
        st_row FAIL 'exit: 1 with a fail and a timeout present' "rc=${rc}"
    fi

    # No unclassified row.
    if awk -F'\t' '/^#/ { next } { if ($3 !~ /^(pass|fail|timeout|needs-args|needs-hardware)$/) bad = 1 } END { exit bad }' \
            "${out}"; then
        st_row PASS 'rows: every row carries one of the five classes'
    else
        st_row FAIL 'rows: every row carries one of the five classes'
    fi

    st_mutation_row "${td}" "${ws}" "${p}"
    st_vacuity_row "${td}" "${ws}"

    if [ "${ST_FAILED}" -gt 0 ]; then
        printf '%s row(s) failed\n' "${ST_FAILED}"
        return 1
    fi
    printf 'all rows passed\n'
    return 0
}

# st_mutation_row TD WS PKG -- prove the wrapper is what classifies a hang.
#
# With the wrapper defeated, `hang` runs forever: the child is killed by an
# OUTER timeout and the TSV never carries `hang<TAB>timeout`. If this row ever
# passes while the mutation is armed, the timeout is decorative and the class is
# coming from somewhere else.
st_mutation_row() {
    local td="$1" ws="$2" p="$3" mout mrc got
    mout="${td}/mutant.tsv"
    mrc=0
    # timeout(1) re-raises SIGKILL on itself so the death is reported
    # faithfully, and the shell whose DIRECT child dies of a signal prints
    # "Killed" on its own stderr. The subshell below is deliberately more than
    # one command so bash cannot exec-optimise it away: it becomes that shell,
    # its stderr is discarded, and the rc still reaches us intact.
    (
        irc=0
        env CARGO_TARGET_DIR="${td}/target" \
            DOGFOOD_EXAMPLES_SELFTEST=1 DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT=1 \
            timeout --signal=KILL 25 \
            bash "${SELF}" --manifest-path "${ws}/Cargo.toml" --timeout-secs 3 \
            --filter 'hang' --out "${mout}" > "${td}/mutant.log" 2>&1 || irc=$?
        exit "${irc}"
    ) 2> /dev/null || mrc=$?
    got=''
    if [ -f "${mout}" ]; then
        got=$(awk -F'\t' -v p="${p}" '$1 == p && $2 == "hang" { print $3 }' \
                "${mout}") || got=''
    fi
    if [ "${got}" = 'timeout' ]; then
        st_row FAIL 'mutation: no-timeout arm must not classify hang as timeout' \
            "mutant rc=${mrc}, class=${got}"
    elif [ "${mrc}" -ne 124 ] && [ "${mrc}" -ne 137 ]; then
        st_row FAIL 'mutation: unwrapped hang must be killed by the OUTER bound' \
            "mutant rc=${mrc}, class='${got:-<no row>}'"
    else
        st_row PASS "mutation: unwrapped hang never reaches 'timeout' (outer rc=${mrc}, class='${got:-<no row>}')"
    fi

    # The kill-switch is refused without the selftest marker, not ignored.
    local grc=0
    env CARGO_TARGET_DIR="${td}/target" DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT=1 \
        timeout --signal=KILL 60 \
        bash "${SELF}" --manifest-path "${ws}/Cargo.toml" --timeout-secs 3 \
        --filter 'hang' --out "${td}/refused.tsv" > "${td}/refused.log" 2>&1 || grc=$?
    if [ "${grc}" -eq 2 ] && grep -q 'selftest-only mutation arm' "${td}/refused.log"; then
        st_row PASS 'mutation: the arm is refused outside the selftest (rc=2)'
    else
        st_row FAIL 'mutation: the arm is refused outside the selftest (rc=2)' \
            "rc=${grc} $(oneline "$(tail -2 "${td}/refused.log")")"
    fi
}

# st_vacuity_row TD WS -- a package with no examples must exit 2, never 0.
st_vacuity_row() {
    local td="$1" ws="$2" vrc=0 vac="$1/vac"
    mkdir -p "${vac}"
    cp -R "${ws}/src" "${vac}/"
    cp "${ws}/Cargo.toml" "${vac}/Cargo.toml"
    env CARGO_TARGET_DIR="${td}/target" \
        bash "${SELF}" --manifest-path "${vac}/Cargo.toml" --timeout-secs 3 \
        --out "${td}/vac.tsv" > "${td}/vac.log" 2>&1 || vrc=$?
    if [ "${vrc}" -eq 2 ] && grep -q 'FAIL (vacuity)' "${td}/vac.log"; then
        st_row PASS 'vacuity: zero examples exits 2'
    else
        st_row FAIL 'vacuity: zero examples exits 2' \
            "rc=${vrc} $(oneline "$(tail -2 "${td}/vac.log")")"
    fi

    # And a --filter that matches nothing is the same vacuity, not a pass.
    local frc=0
    env CARGO_TARGET_DIR="${td}/target" \
        bash "${SELF}" --manifest-path "${ws}/Cargo.toml" --timeout-secs 3 \
        --filter 'zzz-no-such-example' --out "${td}/nofilter.tsv" \
        > "${td}/nofilter.log" 2>&1 || frc=$?
    if [ "${frc}" -eq 2 ]; then
        st_row PASS 'vacuity: a --filter matching nothing exits 2'
    else
        st_row FAIL 'vacuity: a --filter matching nothing exits 2' "rc=${frc}"
    fi
}

# ---------------------------------------------------------------------------
# arguments

while [ $# -gt 0 ]; do
    case "$1" in
        --timeout-secs)
            TIMEOUT_SECS="${2:-}"
            shift 2
            ;;
        --filter)
            FILTER="${2:-}"
            shift 2
            ;;
        --out)
            OUT="${2:-}"
            shift 2
            ;;
        --manifest-path)
            MANIFEST="${2:-}"
            shift 2
            ;;
        --selftest)
            SELFTEST=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown argument: $1"
            ;;
    esac
done

# THE KILL-SWITCH IS REFUSED AT STARTUP, NOT AT USE.
#
# This check first lived inside run_bounded, which runs inside a command
# substitution: `die` exited that subshell with 2, the caller read 2 as the
# example's own exit code and wrote a `fail` row. A refusal that looks exactly
# like a broken example is not a refusal. Here it precedes every cargo call.
if [ "${DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT:-0}" = '1' ] \
    && [ "${DOGFOOD_EXAMPLES_SELFTEST:-0}" != '1' ]; then
    die 'DOGFOOD_EXAMPLES_MUTATE_NO_TIMEOUT is a selftest-only mutation arm'
fi

case "${TIMEOUT_SECS}" in
    '' | *[!0-9]*) die "--timeout-secs wants a positive integer, got '${TIMEOUT_SECS}'" ;;
esac
[ "${TIMEOUT_SECS}" -gt 0 ] || die '--timeout-secs must be > 0'

if [ "${SELFTEST}" -eq 1 ]; then
    # NOT exported: the refusal row below runs a child WITHOUT the marker, and an
    # exported marker would make that child accept the mutation arm.
    selftest
    exit $?
fi

main_run
