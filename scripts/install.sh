#!/bin/sh
# apr installer — downloads the apr binary from a tagged aprender release (or
# the nightly prerelease) and installs it onto PATH. No Rust toolchain needed.
#
# Usage:
#   curl -LsSf https://paiml.com/apr/install.sh | sh
#   curl -LsSf https://paiml.com/apr/install.sh | sh -s -- --version v0.67.0
#   curl -LsSf https://paiml.com/apr/install.sh | sh -s -- --nightly
#
# Env vars (flags below take precedence over these):
#   INSTALL_DIR   Where to place the binary (default: $HOME/.local/bin)
#   APR_VARIANT   "cpu" or "cuda" (default: auto-detect via nvidia-smi)
#
# Refs aprender#2869 (pre-built binaries), aprender#2571 (nightly cross-build
# reused as the build matrix), aprender#2873 (parent tracking issue).
#
# Only Linux x86_64/aarch64 (gnu) have binaries today — the nightly.yml matrix
# once cross-built macOS/Windows too, but no self-hosted runner exists for
# either anymore (see nightly.yml's own header comment), and binary-release.yml
# never attached them to stable tags. Anything else falls through to a clear
# error pointing at `cargo install aprender`.

set -eu

REPO='paiml/aprender'
BINARY_NAME='apr'
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"
VARIANT="${APR_VARIANT:-}"
TAG=''
NIGHTLY=0

# ── Presentation ──────────────────────────────────────────────────────────
# Colors and box-drawing degrade cleanly: NO_COLOR, a non-tty stdout, or a
# `TERM=dumb` all fall back to plain ASCII with no escape codes.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    BOLD='\033[1m'
    DIM='\033[2m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    WHITE='\033[37m'
    INDIGO_500='\033[38;2;99;102;241m'
    INDIGO_600='\033[38;2;79;70;229m'
    BLUE_500='\033[38;2;59;130;246m'
    SKY_400='\033[38;2;56;189;248m'
    SLATE_400='\033[38;2;148;163;184m'
    SLATE_500='\033[38;2;100;116;139m'
    NC='\033[0m'
else
    BOLD='' DIM='' RED='' GREEN='' YELLOW='' CYAN='' WHITE='' \
        INDIGO_500='' INDIGO_600='' BLUE_500='' SKY_400='' \
        SLATE_400='' SLATE_500='' NC=''
fi

# The apr logo, five lines. Column 3 (target/variant) and 4 (the release
# being installed) carry install-specific facts in the same slots the
# runtime CLI banner uses for device/model; column 5 mirrors its cwd line
# with the install destination instead.
logo() {
    target="$1"
    variant="$2"
    display_version="$3"
    dest="$4"

    printf '\n'
    printf '  %b▕▌ ▕▌ ▕▌%b   %b%bapr installer%b\n' "$INDIGO_500" "$NC" "$BOLD" "$WHITE" "$NC"
    printf ' %b▄▀▀▀▀▀▀▀▀▄%b  %baprender ML framework (pure Rust)%b\n' "$INDIGO_600" "$NC" "$SLATE_400" "$NC"
    printf ' %b▀▄▄▄▄▄▄▄▄▀%b  %btarget: %s (%s)%b\n' "$BLUE_500" "$NC" "$SKY_400" "$target" "$variant" "$NC"
    printf '  %b▕▌ ▕▌ ▕▌%b   %binstalling: %s%b\n' "$SKY_400" "$NC" "$SKY_400" "$display_version" "$NC"
    printf '              %b%s%b\n' "$SLATE_500" "$dest" "$NC"
    printf '\n'
}

header() {
    title="$1"
    underline=$(printf '%s' "$title" | sed 's/./=/g')
    printf '\n%b%b%s%b\n' "$BOLD" "$CYAN" "$title" "$NC"
    printf '%b%s%b\n' "$CYAN" "$underline" "$NC"
}

step() {
    printf '%b%b==>%b %s\n' "$BOLD" "$CYAN" "$NC" "$1"
}

ok() {
    printf '  %b[ok]%b %s\n' "$GREEN" "$NC" "$1"
}

info() {
    printf '  %s\n' "$1"
}

warn() {
    printf '  %b[warn]%b %s\n' "$YELLOW" "$NC" "$1" >&2
}

# %b (not %s) for both the color codes and the message: POSIX printf only
# expands backslash escapes (\033 color codes, a caller's embedded \n) in a
# %b argument — %s prints them as literal backslash-digit text verbatim.
error() {
    printf '\n%b%berror:%b %b\n\n' "$BOLD" "$RED" "$NC" "$1" >&2
    exit 1
}

# ── Detection ─────────────────────────────────────────────────────────────

# Rust target triple for this host. Only targets that are actually published
# (stable via binary-release.yml, nightly via nightly.yml) are accepted;
# everything else errors out with the cargo-install fallback rather than
# pretending to succeed.
detect_target() {
    os=$(uname -s)
    arch=$(uname -m)

    case "$os" in
        Linux*)
            case "$arch" in
                x86_64)
                    echo "x86_64-unknown-linux-gnu"
                    ;;
                aarch64 | arm64)
                    echo "aarch64-unknown-linux-gnu"
                    ;;
                *)
                    error "Unsupported Linux architecture: $arch. Use: cargo install aprender"
                    ;;
            esac
            ;;
        Darwin*)
            error "No pre-built apr binary for macOS yet (aprender#2869). Use: cargo install aprender"
            ;;
        MINGW* | CYGWIN* | MSYS*)
            error "No pre-built apr binary for Windows yet (aprender#2869). Use: cargo install aprender"
            ;;
        *)
            error "Unsupported operating system: $os. Use: cargo install aprender"
            ;;
    esac
}

# cpu or cuda. An explicit --cpu/--cuda flag or APR_VARIANT wins; otherwise
# probe for a working NVIDIA driver. A present-but-broken nvidia-smi (no
# driver loaded, container without device passthrough) must not pick "cuda"
# and then fail to run — hence the exit-status check, not just command -v.
detect_variant() {
    if [ -n "$VARIANT" ]; then
        case "$VARIANT" in
            cpu | cuda)
                echo "$VARIANT"
                return
                ;;
            *)
                error "variant must be 'cpu' or 'cuda', got: $VARIANT"
                ;;
        esac
    fi

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
    else
        echo "cpu"
    fi
}

# Latest non-prerelease tag. GitHub's /releases/latest endpoint already
# excludes the "nightly" prerelease, so this always lands on a stable tag.
get_latest_version() {
    version=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name":' \
        | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')
    [ -n "$version" ] || error "Could not determine the latest release from the GitHub API. Pass --version <tag> to skip this lookup."
    echo "$version"
}

# sha256 of a file, in whatever tool this platform actually has.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        error "Neither sha256sum nor shasum is available to verify the download"
    fi
}

fetch() {
    url="$1"
    dest="$2"
    label="$3"
    if ! curl -fsSL "$url" -o "$dest"; then
        error "Failed to download ${label} from:\n  ${url}"
    fi
}

# ── Install ───────────────────────────────────────────────────────────────

install() {
    target=$(detect_target)
    variant=$(detect_variant)

    variant_fell_back=0
    if [ "$NIGHTLY" -eq 1 ]; then
        if [ "$variant" = "cuda" ]; then
            variant='cpu'
            variant_fell_back=1
        fi
        display_version='nightly'
        asset="${BINARY_NAME}-${target}"
        base_url="https://github.com/${REPO}/releases/download/nightly"
    else
        tag="${TAG:-$(get_latest_version)}"
        display_version="$tag"
        version="${tag#v}"
        asset="${BINARY_NAME}-v${version}-${target}-${variant}"
        base_url="https://github.com/${REPO}/releases/download/v${version}"
    fi

    archive_url="${base_url}/${asset}.tar.gz"
    checksum_url="${archive_url}.sha256"

    logo "$target" "$variant" "$display_version" "${INSTALL_DIR}/${BINARY_NAME}"
    [ "$variant_fell_back" -eq 1 ] && warn "nightly builds are CPU-only today; installing cpu instead of cuda"

    step "Downloading"
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "${tmp_dir:?}"' EXIT

    fetch "$archive_url" "$tmp_dir/archive.tar.gz" "${asset}.tar.gz"
    ok "${asset}.tar.gz"
    fetch "$checksum_url" "$tmp_dir/archive.tar.gz.sha256" "${asset}.tar.gz.sha256"
    ok "${asset}.tar.gz.sha256"

    step "Verifying checksum"
    expected=$(awk '{print $1}' "$tmp_dir/archive.tar.gz.sha256")
    actual=$(sha256_of "$tmp_dir/archive.tar.gz")
    if [ "$expected" != "$actual" ]; then
        error "Checksum mismatch for ${asset}.tar.gz\n  expected: ${expected}\n  actual:   ${actual}\nDownload is corrupt or tampered — not installing."
    fi
    ok "sha256 ${actual}"

    step "Installing"
    tar -xzf "$tmp_dir/archive.tar.gz" -C "$tmp_dir"
    extracted="$tmp_dir/${asset}/${BINARY_NAME}"
    [ -f "$extracted" ] || error "Binary '${BINARY_NAME}' not found in archive (expected at ${asset}/${BINARY_NAME})"

    mkdir -p "$INSTALL_DIR"
    mv "$extracted" "$INSTALL_DIR/${BINARY_NAME}"
    chmod +x "$INSTALL_DIR/${BINARY_NAME}"
    ok "${INSTALL_DIR}/${BINARY_NAME}"

    step "Verifying install"
    installed_version=$("$INSTALL_DIR/${BINARY_NAME}" --version 2>&1) || warn "could not run ${INSTALL_DIR}/${BINARY_NAME} --version"
    [ -n "${installed_version:-}" ] && ok "$installed_version"

    # A stale copy earlier on PATH silently wins over the one just installed
    # here — this exact shadowing has bitten this project before (four `apr`
    # binaries coexisting, a bare `apr` resolving to a 26-day-old copy). Don't
    # tell the user to just run `apr --version`; check what PATH actually
    # resolves to and call out a mismatch by name.
    resolved=$(command -v "${BINARY_NAME}" 2>/dev/null || true)
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) on_path=1 ;;
        *) on_path=0 ;;
    esac

    header "Done"
    if [ "$on_path" -eq 0 ]; then
        warn "${INSTALL_DIR} is not on PATH yet. Add this to your shell profile:"
        printf '\n    export PATH="$PATH:%s"\n\n' "$INSTALL_DIR"
    elif [ -z "$resolved" ]; then
        warn "${INSTALL_DIR} is on PATH but the shell hasn't picked it up yet — restart your shell."
    elif [ "$resolved" != "$INSTALL_DIR/${BINARY_NAME}" ]; then
        warn "PATH resolves '${BINARY_NAME}' to ${resolved}, NOT the copy just installed at ${INSTALL_DIR}/${BINARY_NAME}."
        warn "That other copy will shadow this install. Put ${INSTALL_DIR} earlier in PATH, or remove the other one."
    else
        ok "${BINARY_NAME} on PATH resolves to this install: ${resolved}"
        printf '\nRun %b%s --help%b to get started.\n\n' "$BOLD" "$BINARY_NAME" "$NC"
    fi
}

show_help() {
    cat <<EOF
apr installer

Usage: install.sh [OPTIONS]

Options:
  --version <tag>     Install a specific stable tag, e.g. v0.67.0
                       (default: the latest stable release)
  --nightly           Install today's nightly build instead of a stable tag
                       (CPU only; rebuilt daily from main, less stable)
  --cpu               Force the CPU build even if an NVIDIA GPU is detected
  --cuda              Force the CUDA build
  --install-dir <dir> Install location (default: \$HOME/.local/bin,
                       same as \$INSTALL_DIR)
  --help, -h          Show this help message

Environment variables (flags above take precedence):
  INSTALL_DIR   Installation directory
  APR_VARIANT   "cpu" or "cuda"

Examples:
  install.sh                          # latest stable release, auto-detect cpu/cuda
  install.sh --version v0.67.0        # a specific stable release
  install.sh --nightly                # today's nightly build
  install.sh --cpu                    # force cpu even with a GPU present
  INSTALL_DIR=/usr/local/bin install.sh
EOF
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --help | -h)
                show_help
                exit 0
                ;;
            --nightly)
                NIGHTLY=1
                shift
                ;;
            --cpu)
                VARIANT='cpu'
                shift
                ;;
            --cuda)
                VARIANT='cuda'
                shift
                ;;
            --version)
                [ $# -ge 2 ] || error "--version requires an argument, e.g. --version v0.67.0"
                TAG="$2"
                shift 2
                ;;
            --version=*)
                TAG="${1#--version=}"
                shift
                ;;
            --install-dir)
                [ $# -ge 2 ] || error "--install-dir requires an argument"
                INSTALL_DIR="$2"
                shift 2
                ;;
            --install-dir=*)
                INSTALL_DIR="${1#--install-dir=}"
                shift
                ;;
            v[0-9]*)
                # Bare version positional, e.g. `install.sh v0.67.0`, kept for
                # compatibility with the plain-positional convention other
                # paiml installers use.
                TAG="$1"
                shift
                ;;
            *)
                error "Unrecognized argument: $1 (see --help)"
                ;;
        esac
    done

    if [ "$NIGHTLY" -eq 1 ] && [ -n "$TAG" ]; then
        error "--nightly and --version/${TAG} are mutually exclusive"
    fi
}

main() {
    parse_args "$@"
    install
}

main "$@"
