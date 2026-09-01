#!/usr/bin/env bash
# install_pr_review_tools.sh — put the four tools scripts/check_pr_review_receipt.sh
# refuses to run without into a PER-RUN PRIVATE bin directory.
#
#   install_pr_review_tools.sh <bindir>
#
# WHY A PRIVATE ROOT, NOT ~/.local/bin
# ------------------------------------
# mac-server hosts 16 runners under ONE $HOME, so $HOME/.local/bin is shared
# mutable state between concurrently running jobs. That is the same class as
# aprender#2353, where a concurrent `cargo install` into the shared ~/.cargo/bin
# made Coverage Nightly die with ENOENT on an absolute path and report an EMPTY
# coverage figure — a missing measurement dressed as a result.
# check_cargo_install_private_root.sh bans the cargo spelling of it; nothing bans
# the ~/.local/bin spelling, so this script simply does not use it. Everything
# lands under <bindir>, which the caller makes per-run.
#
# WHY THIS IS A SCRIPT AND NOT INLINE YAML
# ----------------------------------------
# It carries pinned versions and their digests, and it is the kind of thing that
# gets edited under pressure. In scripts/ it is lintable by bashrs and readable
# in one place; in a `run: |` block it would be six unreviewable lines that
# nothing checks.
#
# WHY IT REFUSES INSTEAD OF DEGRADING
# -----------------------------------
# The guard's own header: "A tool this guard needs and cannot find is a
# REJECTION, not a skip: a gate that cannot run must not read green." This
# script exits 2 — distinct from the guard's own 1 — when the ENVIRONMENT cannot
# supply a tool, so a broken box is never read as a broken tree. Both are
# non-zero. There is no code path here that reports success without the tool.
#
# EXIT: 0 all four are installed and executable under <bindir>.
#       2 the environment could not supply one; the reason is named.

set -euo pipefail

PROG=${0##*/}

BINDIR=${1:-}
if [ -z "$BINDIR" ]; then
    echo "$PROG: usage: $PROG <bindir>" >&2
    exit 2
fi
mkdir -p "$BINDIR"
BINDIR=$(cd -- "$BINDIR" && pwd)

# Pins. A version without a digest is a download, not a dependency.
MINISIGN_VERSION=0.12
MINISIGN_TARBALL_SHA256=9a599b48ba6eb7b1e80f12f36b94ceca7c00b7a5173c95c3efc88d9822957e73
# bats ships no release binary, so it is pinned by the COMMIT the v1.11.0 tag
# points at rather than by a digest of GitHub's auto-generated source tarball —
# those are regenerated and their bytes have changed under a fixed tag before.
BATS_VERSION=v1.11.0
BATS_COMMIT=5da66876b8b619235aee1eb3e54954eaca88059b
# Same pin the vendored-schemas job already proves works on this fleet.
CHECK_JSONSCHEMA_VERSION=0.38.0

die_env() {
    echo "$PROG: FAIL (environment) - $1" >&2
    echo "  An unmeasured gate is not a passing gate. Exit 2 = the box could not answer;" >&2
    echo "  the guard's own rejections are exit 1. Both are non-zero." >&2
    exit 2
}

for t in curl tar git sha256sum; do
    command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH, and it is needed to install the rest."
done

ARCH=$(uname -m)
case "$ARCH" in
    x86_64|aarch64) ;;
    *) die_env "no pinned minisign build for machine '$ARCH' (x86_64 and aarch64 only)." ;;
esac

WORK=$(mktemp -d) || die_env "cannot create a scratch directory."
cleanup() { rm -rf "${WORK:?}"; }
trap cleanup EXIT

# --- minisign --------------------------------------------------------------
if [ ! -x "$BINDIR/minisign" ]; then
    url="https://github.com/jedisct1/minisign/releases/download/${MINISIGN_VERSION}/minisign-${MINISIGN_VERSION}-linux.tar.gz"
    curl -sfL --retry 3 --max-time 120 -o "$WORK/minisign.tar.gz" "$url" \
        || die_env "cannot download $url"
    # The status is read from sha256sum itself, never from the tail of a pipeline.
    got=$(sha256sum "$WORK/minisign.tar.gz")
    got=${got%% *}
    if [ "$got" != "$MINISIGN_TARBALL_SHA256" ]; then
        die_env "minisign ${MINISIGN_VERSION} tarball digest is $got, pinned $MINISIGN_TARBALL_SHA256"
    fi
    tar -xzf "$WORK/minisign.tar.gz" -C "$WORK" "minisign-linux/${ARCH}/minisign" \
        || die_env "minisign tarball does not contain minisign-linux/${ARCH}/minisign"
    install -m 0755 "$WORK/minisign-linux/${ARCH}/minisign" "$BINDIR/minisign" \
        || die_env "cannot install minisign into $BINDIR"
fi

# --- bats ------------------------------------------------------------------
if [ ! -x "$BINDIR/bats" ]; then
    git clone --quiet --depth 1 --branch "$BATS_VERSION" \
        https://github.com/bats-core/bats-core.git "$WORK/bats-core" \
        || die_env "cannot clone bats-core $BATS_VERSION"
    head=$(git -C "$WORK/bats-core" rev-parse HEAD) || die_env "cannot read the bats-core HEAD"
    if [ "$head" != "$BATS_COMMIT" ]; then
        die_env "bats-core $BATS_VERSION is at $head, pinned $BATS_COMMIT (a moved tag is not a version)"
    fi
    # install.sh takes a PREFIX and creates PREFIX/bin, so hand it the parent.
    "$WORK/bats-core/install.sh" "$(dirname "$BINDIR")/bats-prefix" >/dev/null \
        || die_env "bats-core install.sh failed"
    ln -sf "$(dirname "$BINDIR")/bats-prefix/bin/bats" "$BINDIR/bats" \
        || die_env "cannot link bats into $BINDIR"
fi

# --- check-jsonschema (via uv, into the private root) ----------------------
if [ ! -x "$BINDIR/check-jsonschema" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        UV_INSTALL_DIR="$BINDIR" curl -LsSf --retry 3 --max-time 120 https://astral.sh/uv/install.sh \
            > "$WORK/uv-install.sh" || die_env "cannot download the uv installer"
        UV_INSTALL_DIR="$BINDIR" UV_NO_MODIFY_PATH=1 sh "$WORK/uv-install.sh" >/dev/null \
            || die_env "the uv installer failed"
        export PATH="$BINDIR:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || die_env "uv is still not on PATH after installing it"
    # UV_TOOL_BIN_DIR is what keeps this out of the SHARED ~/.local/bin.
    UV_TOOL_BIN_DIR="$BINDIR" UV_TOOL_DIR="$BINDIR/.uv-tools" \
        uv tool install --quiet "check-jsonschema==${CHECK_JSONSCHEMA_VERSION}" \
        || die_env "uv tool install check-jsonschema==${CHECK_JSONSCHEMA_VERSION} failed"
fi

# --- prove it, rather than assume the installs worked ----------------------
missing=''
for t in minisign bats check-jsonschema; do
    [ -x "$BINDIR/$t" ] || missing="$missing $t"
done
# `[ -n ... ] && die_env ...` would be an AND-list whose left side failing is
# exempt from errexit -- correct today, and the classic set -e footgun. An `if`
# has no such rule to remember.
if [ -n "$missing" ]; then
    die_env "still absent from $BINDIR after installing:$missing"
fi

# jq is a base-image tool everywhere in this repo's CI; it is checked, not installed,
# so an image that loses it fails here by name rather than inside the guard.
command -v jq >/dev/null 2>&1 || die_env "jq is not on PATH"

# The versions are PRINTED, not asserted, and printed WITHOUT a pipe: a
# `cmd | head -1` can return 141 on SIGPIPE despite the command succeeding, and
# this repository shipped four instances of that in one day.
printf '%s: ready in %s\n' "$PROG" "$BINDIR"
"$BINDIR/bats" --version
"$BINDIR/check-jsonschema" --version
printf 'minisign %s (pinned, digest %s)\n' "$MINISIGN_VERSION" "$MINISIGN_TARBALL_SHA256"
exit 0
