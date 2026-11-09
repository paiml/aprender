#!/usr/bin/env bash
# install_pinned_toolchain.sh - install the toolchain rust-toolchain.toml pins (#5403).
#
# The job's toolchain step installs `stable`. A cargo run that does not set
# RUSTUP_TOOLCHAIN (the guard-cargo#m60 fixture) reads rust-toolchain.toml, finds the
# pinned release absent and downloads it at test time: one network blip then failed a
# merge group. Installing the pin up front takes that download out of the test.
#
# Only an exact release (1.93.0) is installed; a floating channel is what the toolchain
# step already installs. A failed install FAILS the step (C324: no fail-open state):
# the job would otherwise download the same release at test time and fail there.
#
#   install_pinned_toolchain.sh [workspace-dir]
#   install_pinned_toolchain.sh --self-test
#
# EXIT 0 installed or nothing to install; 1 the pinned install failed (--self-test: 1 on a failed row).

set -euo pipefail

install_pin() {
    local ws=${1:-.} f ch comps args=() c
    f="$ws/rust-toolchain.toml"
    if [ ! -f "$f" ]; then
        echo "install_pinned_toolchain: no $f, nothing to install"
        return 0
    fi
    ch=$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)".*/\1/p' "$f" | head -n 1)
    if [ -z "$ch" ]; then
        echo "install_pinned_toolchain: channel is not an exact release, nothing to install"
        return 0
    fi
    comps=$(sed -n 's/^[[:space:]]*components[[:space:]]*=[[:space:]]*\[\(.*\)\].*/\1/p' "$f" | head -n 1 | tr -d '"' | tr ',' ' ')
    for c in $comps; do args+=(-c "$c"); done
    if rustup toolchain install "$ch" --profile minimal --no-self-update "${args[@]}"; then
        echo "install_pinned_toolchain: $ch installed (${comps:-no components})"
    else
        echo "::error::install_pinned_toolchain: $ch did not install"; return 1
    fi
    return 0
}

self_test() {
    local t fails=0 ws out
    t=$(mktemp -d) || return 2
    mkdir -p "$t/bin" "$t/ws"
    printf '#!/bin/sh\necho "rustup $*" >> "%s/calls"\n[ -z "${STUB_FAIL:-}" ]\n' "$t" >"$t/bin/rustup"
    chmod +x "$t/bin/rustup"
    export PATH="$t/bin:$PATH"
    ws=$t/ws
    chk() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1 (got '$2' want '$3')"; fails=$((fails + 1)); fi; }
    called() { if [ -f "$t/calls" ]; then echo called; else echo none; fi; }
    install_pin "$ws" >/dev/null
    chk "no toml: no rustup call" "$(called)" none
    printf '[toolchain]\nchannel = "1.93.0"\ncomponents = ["rustfmt", "clippy"]\n' >"$ws/rust-toolchain.toml"
    install_pin "$ws" >/dev/null
    chk "exact pin: installed with components" "$(cat "$t/calls")" "rustup toolchain install 1.93.0 --profile minimal --no-self-update -c rustfmt -c clippy"
    rm -f "${t:?}/calls"
    printf '[toolchain]\nchannel = "stable"\n' >"$ws/rust-toolchain.toml"
    install_pin "$ws" >/dev/null
    chk "floating channel: no rustup call" "$(called)" none
    printf '[toolchain]\nchannel = "1.93.0"\n' >"$ws/rust-toolchain.toml"
    out=$(STUB_FAIL=1 install_pin "$ws" && echo rc=0 || echo rc=1)
    chk "failed install: error and rc 1" "$(echo "$out" | grep -c "^::error::")$(echo "$out" | tail -n 1)" "1rc=1"
    rm -rf -- "${t:?}"
    [ "$fails" -eq 0 ] && echo "self-test: all ok" || echo "self-test: $fails FAILED"
    return $((fails > 0))
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi
install_pin "${1:-.}"
