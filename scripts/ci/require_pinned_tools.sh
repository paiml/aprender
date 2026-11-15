#!/usr/bin/env bash
# require_pinned_tools.sh -- assert the tools pr-light uses are already on this runner; NEVER install one (GATE-JUNK-001 C6).
#
# Provisioning lives where it is pinned (forjar machines/*/forjar.yaml; pre-job.sh asserts rustup/cargo/forjar). A
# missing tool is a RED step naming the resource, not a download at gate time:
#   jq 1.8.x                         guard_tree_test needs 1.8 (it replaces the old fetch of jq-1.8.2)
#   yq v4.52.2                       the pull_request_target guard reads workflow triggers with it (replaces a curl of the binary)
#   the rust-toolchain.toml release  (it replaces the old toolchain install); only an exact release is checked
#   the sovereign-ci image digest    (only when a caller names one; pr-light names none since cargo fmt moved to the host toolchain)
#
#   bash scripts/ci/require_pinned_tools.sh [workspace-dir]
#   bash scripts/ci/require_pinned_tools.sh --self-test
set -euo pipefail

pin_of() { # pin_of <rust-toolchain.toml> -- the exact release, or empty
    sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)".*/\1/p' "$1" | head -n 1
}

jq_ok() { # jq_ok <version string like jq-1.8.2>
    [[ $1 =~ ^jq-1\.([8-9]|[1-9][0-9])(\.[0-9]+)?$ ]]
}

check() {
    local ws=${1:-.} bad=0 v pin img
    command -v jq > /dev/null 2>&1 || { echo "::error::jq is not provisioned on this runner (pinned jq 1.8.x in forjar); pr-light does not download it"; bad=1; }
    if [ "$bad" = 0 ]; then
        v="$(jq --version)"
        jq_ok "$v" || { echo "::error::jq is '$v', pr-light needs 1.8.x from forjar; pr-light does not fetch it"; bad=1; }
    fi
    command -v yq > /dev/null 2>&1 || { echo "::error::yq is not provisioned on this runner (pinned v4.52.2 in forjar); pr-light does not download it"; bad=1; }
    if command -v yq > /dev/null 2>&1; then
        case "$(yq --version 2>&1)" in *" v4.52.2") ;; *) echo "::error::yq is '$(yq --version 2>&1)', pr-light needs v4.52.2 from forjar; pr-light does not fetch it"; bad=1 ;; esac
    fi
    if ! test -f "$ws/rust-toolchain.toml"; then
        echo "::error::no rust-toolchain.toml under '$ws': the toolchain pin cannot be read, so it is not checked (run from the workspace root or pass it)"; bad=1
    else
        pin="$(pin_of "$ws/rust-toolchain.toml")"
        if [ -n "$pin" ]; then
            command -v rustup > /dev/null 2>&1 || { echo "::error::rustup is not provisioned (forjar rust-toolchain)"; bad=1; }
            if [ "$bad" = 0 ] && ! grep -q "^${pin//./\\.}-" <<< "$(rustup toolchain list 2> /dev/null)"; then
                echo "::error::toolchain $pin (rust-toolchain.toml) is not on this runner; provision it in forjar machines/*/forjar.yaml. pr-light does not fetch it"; bad=1
            fi
            if [ "$bad" = 0 ] && ! grep -q '^rustfmt' <<< "$(rustup component list --toolchain "$pin" --installed 2> /dev/null)"; then
                echo "::error::rustfmt for $pin is not on this runner (pr-light's fmt lane runs it with RUSTUP_AUTO_INSTALL=0); pr-light does not fetch it"; bad=1
            fi
        fi
    fi
    img="$(grep -oE 'localhost:5000/sovereign-ci:stable@sha256:[0-9a-f]{64}' "$ws/scripts/ci/pr_light_fast.sh" | head -n 1 || true)"
    if [ -n "$img" ]; then
        docker image inspect "$img" > /dev/null 2>&1 || { echo "::error::image $img is not in this runner's local cache; pr-light does not fetch it"; bad=1; }
    fi
    [ "$bad" = 0 ] && echo "ok    jq, yq and the pinned toolchain are provisioned"
    return "$bad"
}

self_test() {
    local pass=0 fail=0 d
    row() { if [[ $2 == "$3" ]]; then pass=$((pass + 1)); printf '  ok    %s\n' "$1"; else fail=$((fail + 1)); printf '  FAIL  %s want=%s got=%s\n' "$1" "$3" "$2"; fi; }
    row "jq-1.8.2 passes" "$(jq_ok jq-1.8.2 && echo y || echo n)" y
    row "jq-1.7.1 is refused" "$(jq_ok jq-1.7.1 && echo y || echo n)" n
    row "jq-1.7 is refused" "$(jq_ok jq-1.7 && echo y || echo n)" n
    row "jq-1.10 passes" "$(jq_ok jq-1.10 && echo y || echo n)" y
    d="$(mktemp -d "${TMPDIR:-/tmp}/rpt.XXXXXX")"
    trap "rm -rf '${d:?}'" EXIT
    printf '[toolchain]\nchannel = "1.93.0"\ncomponents = ["rustfmt"]\n' > "$d/rust-toolchain.toml"
    row "the exact pin is read" "$(pin_of "$d/rust-toolchain.toml")" 1.93.0
    printf '[toolchain]\nchannel = "stable"\n' > "$d/rust-toolchain.toml"
    row "a floating channel is no pin" "$(pin_of "$d/rust-toolchain.toml")" ""
    row "a prefixed jq string is refused" "$(jq_ok xjq-1.8.2 && echo y || echo n)" n
    # scen <name> <want y|n> <jq-version|-> <yq-version|-> <toolchains|-> <components|-> <image: ok|no|none> <channel>: run check() on stub tools only
    scen() {
        local name=$1 want=$2; shift 2
        rm -rf "${d:?}/bin" "${d:?}/scripts"; mkdir -p "$d/bin" "$d/scripts/ci"
        local t; for t in sed grep head; do ln -sf "$(command -v "$t")" "$d/bin/$t"; done
        if [[ $1 != - ]]; then printf '#!/bin/sh\necho %s\n' "$1" > "$d/bin/jq"; fi
        if [[ $2 != - ]]; then printf '#!/bin/sh\necho "yq version %s"\n' "$2" > "$d/bin/yq"; fi
        if [[ $3 != - ]]; then printf '#!/bin/sh\nif [ "$1" = toolchain ]; then echo "%s"; else echo "%s"; fi\n' "$3" "$4" > "$d/bin/rustup"; fi
        if [[ $5 != none ]]; then printf 'x localhost:5000/sovereign-ci:stable@sha256:%064d\n' 0 > "$d/scripts/ci/pr_light_fast.sh"; fi
        printf '#!/bin/sh\nexit %s\n' "$([[ $5 == no ]] && echo 1 || echo 0)" > "$d/bin/docker"
        printf '[toolchain]\nchannel = "%s"\n' "$6" > "$d/rust-toolchain.toml"
        if [[ $6 == absent ]]; then rm -f "${d:?}/rust-toolchain.toml"; fi
        for t in jq yq rustup docker; do [ ! -f "$d/bin/$t" ] || chmod +x "$d/bin/$t"; done
        local got=n; PATH="$d/bin" check "$d" > /dev/null 2>&1 && got=y; row "$name" "$got" "$want"
    }
    scen "all tools present is green" y jq-1.8.2 v4.52.2 1.93.0-x86_64 rustfmt-x none 1.93.0
    scen "no pin needs no rustup" y jq-1.8.2 v4.52.2 - - none stable
    scen "jq missing is RED" n - v4.52.2 1.93.0-x rustfmt none 1.93.0
    scen "jq 1.7.1 is RED" n jq-1.7.1 v4.52.2 1.93.0-x rustfmt none 1.93.0
    scen "yq missing is RED" n jq-1.8.2 - 1.93.0-x rustfmt none 1.93.0
    scen "yq other version is RED" n jq-1.8.2 v4.52.3 1.93.0-x rustfmt none 1.93.0
    scen "yq v4.52.20 is RED (the pin is exact)" n jq-1.8.2 v4.52.20 1.93.0-x rustfmt none 1.93.0
    scen "a missing rust-toolchain.toml is RED, not skipped" n jq-1.8.2 v4.52.2 1.93.0-x rustfmt none absent
    scen "the pin's dots are literal" n jq-1.8.2 v4.52.2 1x93y0-x rustfmt none 1.93.0
    scen "rustup missing is RED" n jq-1.8.2 v4.52.2 - - none 1.93.0
    scen "a missing toolchain is RED, not fetched" n jq-1.8.2 v4.52.2 1.92.0-x rustfmt none 1.93.0
    scen "a missing rustfmt is RED" n jq-1.8.2 v4.52.2 1.93.0-x clippy none 1.93.0
    scen "a missing image is RED" n jq-1.8.2 v4.52.2 1.93.0-x rustfmt no 1.93.0
    scen "a cached image is green" y jq-1.8.2 v4.52.2 1.93.0-x rustfmt ok 1.93.0
    printf 'require_pinned_tools self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    *) check "${1:-.}" ;;
esac
