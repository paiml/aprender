#!/usr/bin/env bash
# ci_self_hosted_preflight.sh — the first step of a self-hosted job: fail fast,
# and by NAME, on a tool the job assumes and the box does not have.
# Row 67-C2 (PMAT-1098, issue #3083).
#
# WHY THIS EXISTS
# ---------------
# Run 34448908554 (the 0.66.0 CUDA-asset backfill) resolved the tag, checked out,
# built apr --features cuda on gx10, proved `libcuda.so` was in the bytes, stripped
# it, packaged a 21 MB tarball, wrote its sha256 — and then died on the last line:
#
#     gh: command not found
#
# A GPU build spent on a shell error that costs one second to detect. Self-hosted
# boxes are not GitHub-hosted images: they carry whatever was installed on them,
# and nothing in this repo ever checked that. `check_runner_labels.sh` proves a
# self-hosted selector DISCRIMINATES between pools; it cannot know what the pool
# it selects has installed. That is this file's job.
#
# `gh` IS DELIBERATELY NOT IN THE DEFAULT SET. The CUDA lane was moved to the REST
# API with curl + python3 precisely because fleet boxes have no gh (#3074). A job
# that genuinely wants it says so: `--need gh`. Putting gh in the defaults would
# re-assert the assumption that caused the failure.
#
# INTERFACE
#   bash scripts/ci_self_hosted_preflight.sh [--need TOOL...] [--cuda]
#   bash scripts/ci_self_hosted_preflight.sh --self-test
#
#   defaults : jq curl git python3 tar sha256sum rustup cargo
#   --cuda   : adds nvidia-smi AND asserts `nvidia-smi -L` lists >= 1 device
#              (a driver present with no device visible is the cgroup/permission
#              failure that reads as "CUDA is broken" three hours later)
#   --need   : one or more extra tools, e.g. `--need docker objdump`
#
#   stdout   : identity lines, then one line per tool —
#                ok <tool> <path> <version-or->
#                MISSING <tool>
#   exit 0   : every tool present (and, with --cuda, a device is visible)
#   exit 1   : at least one MISSING — a finding about the BOX
#   exit 2   : usage error — a finding about the STEP that called this
#              (the two must not share a code: one is "fix the runner", the other
#              is "fix the workflow", and a merged code sends the wrong person)
#
#   JSON     : when PREFLIGHT_OUT or RUNNER_TEMP is set, the same facts are written
#              to ${PREFLIGHT_OUT:-$RUNNER_TEMP/preflight.json}. That file IS the
#              fleet-toolset probe's output (.github/workflows/fleet-toolset.yml),
#              so "what does this box have" becomes a file under evidence/fleet/
#              instead of somebody's memory.
#
# DETERMINISM: stdout carries no clock reading. The only timestamp in this script
# is the JSON's `measured_at` field, which is a measurement's date and belongs in
# a record; nothing compares stdout across runs, so nothing can be broken by it.
#
# Refs: PMAT-1098, row 67-C2, #3083, #3074, run 34448908554.

set -uo pipefail

# Pure parameter expansion, no `dirname`/`basename`: the first step of a job on a
# box with a broken PATH is exactly where an external call must not be needed.
_src="${BASH_SOURCE[0]}"
_dir="${_src%/*}"
if [ "$_dir" = "$_src" ]; then
    _dir="."
fi
SELF="$(cd -- "$_dir" && pwd)/${_src##*/}"

DEFAULT_TOOLS="jq curl git python3 tar sha256sum rustup cargo"

usage() {
    printf 'usage: %s [--need TOOL...] [--cuda] [--self-test]\n' "${SELF##*/}"
    printf '  defaults: %s\n' "$DEFAULT_TOOLS"
    printf '  --cuda    also require nvidia-smi and >= 1 visible device\n'
    printf '  --need    require the named extra tool(s), e.g. --need docker objdump\n'
    printf '  exit 0 all present · exit 1 something MISSING · exit 2 usage error\n'
}

# Strip the characters that would break a JSON string, plus control characters.
# Escaping is not attempted: a version banner has no business carrying a quote or
# a backslash, and deleting them keeps the writer a printf instead of a parser.
sanitize() {
    tr -d '\\"[:cntrl:]' 2>/dev/null || printf '%s' ''
}

# First line of a tool's version banner, or '-'. stdin is closed so a tool that
# waits for input cannot hang a job's first step.
tool_version() {
    local t="$1" v=""
    v="$("$t" --version < /dev/null 2>/dev/null | head -n 1 | sanitize)"
    if [ -z "$v" ]; then
        v="$("$t" version < /dev/null 2>/dev/null | head -n 1 | sanitize)"
    fi
    if [ -z "$v" ]; then
        printf '%s' '-'
    else
        printf '%s' "${v:0:120}"
    fi
}

preflight_main() {
    local want_cuda=0
    local extra=""
    local arg

    while [ "$#" -gt 0 ]; do
        arg="$1"
        case "$arg" in
            --cuda)
                want_cuda=1
                shift
                ;;
            --need)
                shift
                if [ "$#" -eq 0 ] || [ -z "${1:-}" ] || [ "${1#-}" != "$1" ]; then
                    printf 'ci_self_hosted_preflight: --need requires at least one tool name\n' >&2
                    usage >&2
                    return 2
                fi
                # `--need docker objdump` — consume every following non-flag word.
                while [ "$#" -gt 0 ] && [ -n "${1:-}" ] && [ "${1#-}" = "$1" ]; do
                    extra="$extra $1"
                    shift
                done
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                printf 'ci_self_hosted_preflight: unknown argument: %s\n' "$arg" >&2
                usage >&2
                return 2
                ;;
        esac
    done

    local tools="$DEFAULT_TOOLS$extra"
    if [ "$want_cuda" -eq 1 ]; then
        tools="$tools nvidia-smi"
    fi

    # ---- where the machine-readable copy goes -----------------------------
    local out=""
    if [ -n "${PREFLIGHT_OUT:-}" ]; then
        out="$PREFLIGHT_OUT"
    elif [ -n "${RUNNER_TEMP:-}" ]; then
        out="$RUNNER_TEMP/preflight.json"
    fi
    if [ -n "$out" ]; then
        # A `..` in a path handed to a step is either a mistake or an escape;
        # neither belongs in a file this script creates.
        case "$out" in
            *..*)
                printf 'ci_self_hosted_preflight: refusing an output path containing "..": %s\n' "$out" >&2
                return 2
                ;;
        esac
        local outdir="${out%/*}"
        if [ "$outdir" = "$out" ]; then
            outdir="."
        fi
        mkdir -p "$outdir" 2>/dev/null || true
    fi

    # ---- identity: which box answered ------------------------------------
    local runner="${RUNNER_NAME:--}"
    local labels="${RUNNER_LABELS:--}"
    local arch glibc
    arch="$(uname -m 2>/dev/null | head -n 1 | sanitize)"
    [ -n "$arch" ] || arch='-'
    glibc="$(ldd --version 2>/dev/null | head -n 1 | sanitize)"
    [ -n "$glibc" ] || glibc='-'

    printf 'runner=%s labels=%s\n' "$runner" "$labels"
    printf 'arch=%s\n' "$arch"
    printf 'glibc=%s\n' "$glibc"

    # ---- the tools --------------------------------------------------------
    local t p v missing="" json_tools=""
    for t in $tools; do
        p="$(command -v "$t" 2>/dev/null)"
        if [ -z "$p" ]; then
            printf 'MISSING %s\n' "$t"
            missing="$missing $t"
            json_tools="$json_tools{\"name\":\"$t\",\"present\":false,\"path\":null,\"version\":null},"
            continue
        fi
        v="$(tool_version "$t")"
        printf 'ok %s %s %s\n' "$t" "$p" "$v"
        json_tools="$json_tools{\"name\":\"$t\",\"present\":true,\"path\":\"$p\",\"version\":\"$v\"},"
    done

    # ---- the device, when asked -------------------------------------------
    local cuda_devices=0 driver='-' listing='' line
    if [ "$want_cuda" -eq 1 ] && command -v nvidia-smi > /dev/null 2>&1; then
        listing="$(nvidia-smi -L 2>/dev/null)"
        while IFS= read -r line; do
            case "$line" in
                GPU\ *) cuda_devices=$((cuda_devices + 1)) ;;
            esac
        done <<< "$listing"
        driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | sanitize)"
        [ -n "$driver" ] || driver='-'
        printf 'cuda_devices=%s driver=%s\n' "$cuda_devices" "$driver"
        if [ "$cuda_devices" -lt 1 ]; then
            # nvidia-smi exists and lists nothing: the driver is installed and the
            # job cannot see a GPU. Naming it here is the whole point — otherwise
            # it surfaces as a CUDA "correctness" failure an hour into the run.
            printf 'MISSING cuda-device\n'
            missing="$missing cuda-device"
        fi
    fi

    # ---- verdict ----------------------------------------------------------
    local rc=0 nmiss=0
    for t in $missing; do
        nmiss=$((nmiss + 1))
    done
    if [ "$nmiss" -gt 0 ]; then
        rc=1
        printf '::error::self-hosted preflight FAILED on %s:%s (%s tool(s)/device(s) missing)\n' \
            "$runner" "$missing" "$nmiss"
    else
        local ntools=0
        for t in $tools; do
            ntools=$((ntools + 1))
        done
        printf 'preflight ok: %s tool(s) present on %s\n' "$ntools" "$runner"
    fi

    # ---- the record -------------------------------------------------------
    if [ -n "$out" ]; then
        local json_missing="" stamp
        for t in $missing; do
            json_missing="$json_missing\"$t\","
        done
        # THE ONLY CLOCK READING IN THIS SCRIPT. It is a field of a measurement
        # record, never part of a compared output. If `date` is absent the field
        # is null rather than a lie.
        stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null | sanitize)"
        if [ -n "$stamp" ]; then
            stamp="\"$stamp\""
        else
            stamp='null'
        fi
        {
            printf '{"measured_at":%s,' "$stamp"
            printf '"runner":"%s","labels":"%s","arch":"%s","glibc":"%s",' \
                "$runner" "$labels" "$arch" "$glibc"
            printf '"cuda_requested":%s,"cuda_devices":%s,"cuda_driver":"%s",' \
                "$([ "$want_cuda" -eq 1 ] && printf 'true' || printf 'false')" \
                "$cuda_devices" "$driver"
            printf '"tools":[%s],' "${json_tools%,}"
            printf '"missing":[%s],' "${json_missing%,}"
            printf '"exit":%s}\n' "$rc"
        } > "$out"
        printf 'preflight json: %s\n' "$out"
    fi

    return "$rc"
}

# ---------------------------------------------------------------------------
# --self-test: the case table. Hermetic — every tool the rows see is a stub in a
# scratch PATH, so the table's verdict does not depend on what this box has, and
# no row touches the network.
#
# THE MUTATION THE TABLE MUST SURVIVE: delete `jq` from the scratch PATH and the
# script must exit 1 naming it. Row 1 (the same PATH, jq present, exit 0) is not
# decoration — without it, row 2 is satisfied by a script that fails on every
# input, which is the shape of the guards this repo keeps finding.
# ---------------------------------------------------------------------------
self_test() {
    local td rows=0 fails=0 out rc bin
    td="$(mktemp -d)" || { printf 'ENV: cannot create a temp dir\n' >&2; return 2; }

    local helper
    for helper in uname tr head date mkdir ldd sh; do
        if ! command -v "$helper" > /dev/null 2>&1; then
            printf 'ENV: this box has no %s — the case table cannot be built here\n' "$helper" >&2
            rm -rf "${td:?}"
            return 2
        fi
    done

    bin="$td/bin"
    mkdir -p "$bin"

    # Stubs for every default tool: the table asserts the SCRIPT's behaviour, not
    # the box's inventory, so a clean-room container with no rustup still runs it.
    local t
    for t in $DEFAULT_TOOLS gh; do
        {
            printf '#!/bin/sh\n'
            printf 'printf "%%s\\n" "%s 0.0-stub"\n' "$t"
        } > "$bin/$t"
        chmod +x "$bin/$t"
    done
    # Real coreutils, because the script under test calls them.
    for helper in uname tr head date mkdir ldd; do
        ln -sf "$(command -v "$helper")" "$bin/$helper"
    done

    # nvidia-smi stubs: one that lists a device, one that lists none. The listing
    # is baked into the stub rather than catted from a file — the child's PATH is
    # the scratch dir, and a stub that needs `cat` would report zero devices for
    # the wrong reason (measured: it did, before this was fixed).
    local which listing
    for which in one none; do
        if [ "$which" = "one" ]; then
            listing='printf "%s\\n" "GPU 0: STUB DEVICE (UUID: GPU-00000000-0000-0000-0000-000000000000)"'
        else
            listing=':'
        fi
        {
            printf '#!/bin/sh\n'
            printf 'if [ "$1" = "-L" ]; then %s; exit 0; fi\n' "$listing"
            printf 'printf "%%s\\n" "999.99-stub"\n'
        } > "$td/nvidia-smi-$which"
        chmod +x "$td/nvidia-smi-$which"
    done

    row() { # row LABEL WANT_RC GOT_RC PATTERN OUTPUT
        rows=$((rows + 1))
        if [ "$3" = "$2" ] && { [ -z "$4" ] || printf '%s\n' "$5" | grep -q -- "$4"; }; then
            printf 'ok  %s\n' "$1"
        else
            printf 'FAIL  %s — rc=%s (want %s), pattern [%s]\n' "$1" "$3" "$2" "$4"
            printf '%s\n' "$5" | sed 's/^/      /'
            fails=$((fails + 1))
        fi
    }

    run() { # run PATHDIR ARGS... — the child sees ONLY PATHDIR
        local d="$1"
        shift
        env -u PREFLIGHT_OUT -u RUNNER_TEMP PATH="$d" "${BASH:-/bin/bash}" "$SELF" "$@" 2>&1
    }

    out="$(run "$bin")"; rc=$?
    row "every default tool present -> exit 0" 0 "$rc" '^ok jq ' "$out"

    out="$(run "$bin" --need gh)"; rc=$?
    row "--need gh with gh present -> exit 0" 0 "$rc" '^ok gh ' "$out"

    out="$(run "$bin" --need bogus-tool-xyz)"; rc=$?
    row "--need bogus-tool-xyz -> exit 1" 1 "$rc" 'MISSING bogus-tool-xyz' "$out"

    cp "$td/nvidia-smi-one" "$bin/nvidia-smi"
    out="$(run "$bin" --cuda)"; rc=$?
    row "--cuda with one visible device -> exit 0" 0 "$rc" 'cuda_devices=1' "$out"

    cp "$td/nvidia-smi-none" "$bin/nvidia-smi"
    out="$(run "$bin" --cuda)"; rc=$?
    row "--cuda with a driver but NO device -> exit 1" 1 "$rc" 'MISSING cuda-device' "$out"

    rm -f "$bin/nvidia-smi"
    out="$(run "$bin" --cuda)"; rc=$?
    row "--cuda with no nvidia-smi at all -> exit 1" 1 "$rc" 'MISSING nvidia-smi' "$out"

    out="$(run "$bin" --nope)"; rc=$?
    row "an unknown flag -> exit 2 (usage, not a box finding)" 2 "$rc" 'unknown argument' "$out"

    out="$(run "$bin" --need)"; rc=$?
    row "--need with no tool name -> exit 2" 2 "$rc" 'requires at least one tool' "$out"

    out="$(env -u RUNNER_TEMP PREFLIGHT_OUT="$td/x/../y.json" PATH="$bin" "${BASH:-/bin/bash}" "$SELF" 2>&1)"; rc=$?
    row 'a PREFLIGHT_OUT containing ".." -> exit 2' 2 "$rc" 'refusing an output path' "$out"

    out="$(env -u RUNNER_TEMP PREFLIGHT_OUT="$td/rec/preflight.json" PATH="$bin" "${BASH:-/bin/bash}" "$SELF" 2>&1)"; rc=$?
    row "PREFLIGHT_OUT is written" 0 "$rc" 'preflight json:' "$out"
    rows=$((rows + 1))
    if [ -s "$td/rec/preflight.json" ] && grep -q '"missing":\[\]' "$td/rec/preflight.json"; then
        printf 'ok  the record exists and reports nothing missing\n'
    else
        printf 'FAIL  the record was not written, or does not report an empty missing[]\n'
        fails=$((fails + 1))
    fi

    # THE MUTATION, last so the table above proves the script works before this
    # proves it can fail: hide jq and nothing else.
    rm -f "$bin/jq"
    out="$(env -u RUNNER_TEMP PREFLIGHT_OUT="$td/rec/mut.json" PATH="$bin" "${BASH:-/bin/bash}" "$SELF" 2>&1)"; rc=$?
    row "MUTATION: jq hidden from PATH -> exit 1 naming it" 1 "$rc" '^MISSING jq$' "$out"
    rows=$((rows + 1))
    if grep -q '"missing":\["jq"\]' "$td/rec/mut.json" 2>/dev/null; then
        printf 'ok  the record names jq, and only jq, as missing\n'
    else
        printf 'FAIL  the record does not name jq as the only missing tool\n'
        printf '      %s\n' "$(cat "$td/rec/mut.json" 2>/dev/null)"
        fails=$((fails + 1))
    fi

    rm -rf "${td:?}"

    printf '\n'
    if [ "$fails" -ne 0 ]; then
        printf 'SELF-TEST FAILED (%s/%s rows red)\n' "$fails" "$rows"
        return 1
    fi
    printf 'SELF-TEST PASSED (%s/%s)\n' "$rows" "$rows"
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

preflight_main "$@"
exit $?
