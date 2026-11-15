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
        case "$(yq --version 2>&1)" in *" v4.52.2"*) ;; *) echo "::error::yq is '$(yq --version 2>&1)', pr-light needs v4.52.2 from forjar; pr-light does not fetch it"; bad=1 ;; esac
    fi
    if [ -f "$ws/rust-toolchain.toml" ]; then
        pin="$(pin_of "$ws/rust-toolchain.toml")"
        if [ -n "$pin" ]; then
            command -v rustup > /dev/null 2>&1 || { echo "::error::rustup is not provisioned (forjar rust-toolchain)"; bad=1; }
            if [ "$bad" = 0 ] && ! grep -q "^${pin}-" <<< "$(rustup toolchain list 2> /dev/null)"; then
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
    row() { if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    %s\n' "$1"; else fail=$((fail + 1)); printf '  FAIL  %s want=%s got=%s\n' "$1" "$3" "$2"; fi; }
    row "jq-1.8.2 passes" "$(jq_ok jq-1.8.2 && echo y || echo n)" y
    row "jq-1.7.1 is refused" "$(jq_ok jq-1.7.1 && echo y || echo n)" n
    row "jq-1.7 is refused" "$(jq_ok jq-1.7 && echo y || echo n)" n
    row "jq-1.10 passes" "$(jq_ok jq-1.10 && echo y || echo n)" y
    d="$(mktemp -d "${TMPDIR:-/tmp}/rpt.XXXXXX")"
    printf '[toolchain]\nchannel = "1.93.0"\ncomponents = ["rustfmt"]\n' > "$d/rust-toolchain.toml"
    row "the exact pin is read" "$(pin_of "$d/rust-toolchain.toml")" 1.93.0
    printf '[toolchain]\nchannel = "stable"\n' > "$d/rust-toolchain.toml"
    row "a floating channel is no pin" "$(pin_of "$d/rust-toolchain.toml")" ""
    printf '[toolchain]\nchannel = "9.99.0"\n' > "$d/rust-toolchain.toml"
    row "a missing toolchain is RED, not fetched" "$(check "$d" > /dev/null 2>&1 && echo y || echo n)" n
    rm -rf "${d:?}"
    printf 'require_pinned_tools self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    *) check "${1:-.}" ;;
esac
