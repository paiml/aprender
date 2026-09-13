#!/bin/bash
# scripts/generate_reproducibility_manifest.sh
#
# Generates a complete reproducibility manifest for benchmark runs.
# This captures the exact state of the toolchain, dependencies, and source code
# to enable byte-identical reproduction of benchmark results.

set -e

OUTPUT_FILE="${1:-benchmarks/metadata/toolchain_manifest.txt}"
# Path-traversal guard: OUTPUT_FILE comes straight from $1, so refuse a '..'
# component before it is ever used as a write target.
case "$OUTPUT_FILE" in
    *..*)
        echo "refusing OUTPUT_FILE with a '..' path component: $OUTPUT_FILE" >&2
        exit 1
        ;;
esac

echo "=== Certeza Reproducibility Manifest ===" > "$OUTPUT_FILE"
# "Generated:" records real wall-clock time by default (this manifest exists
# to capture the state of an actual toolchain/build); SOURCE_DATE_EPOCH, when
# set, pins it for a reproducible/test invocation.
echo "Generated: $(date -u -d "@${SOURCE_DATE_EPOCH:-$(date +%s)}" -Iseconds)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Rust Toolchain ===" >> "$OUTPUT_FILE"
rustc --version --verbose >> "$OUTPUT_FILE" 2>&1 || echo "rustc not found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
cargo --version >> "$OUTPUT_FILE" 2>&1 || echo "cargo not found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Dependency Tree (locked) ===" >> "$OUTPUT_FILE"
cargo tree --locked --frozen >> "$OUTPUT_FILE" 2>&1 || echo "Failed to generate dependency tree" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Cargo.lock Hash ===" >> "$OUTPUT_FILE"
if [ -f "Cargo.lock" ]; then
    sha256sum Cargo.lock >> "$OUTPUT_FILE"
else
    echo "Cargo.lock not found" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "=== Source Tree Hash (excluding target/) ===" >> "$OUTPUT_FILE"
if command -v sha256sum &> /dev/null; then
    find src -type f -name "*.rs" -exec sha256sum {} \; 2>/dev/null | sort | sha256sum >> "$OUTPUT_FILE"
else
    echo "sha256sum command not found" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "=== Build Profile Configuration ===" >> "$OUTPUT_FILE"
if [ -f "Cargo.toml" ]; then
    grep -A 20 "\[profile.release\]" Cargo.toml >> "$OUTPUT_FILE" || echo "No [profile.release] section found" >> "$OUTPUT_FILE"
else
    echo "Cargo.toml not found" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "=== Git Information ===" >> "$OUTPUT_FILE"
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "Commit: $(git rev-parse HEAD)" >> "$OUTPUT_FILE"
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)" >> "$OUTPUT_FILE"
    echo "Status:" >> "$OUTPUT_FILE"
    git status --short >> "$OUTPUT_FILE"
else
    echo "Not a git repository or git not installed" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "=== System Information ===" >> "$OUTPUT_FILE"
uname -a >> "$OUTPUT_FILE" 2>&1 || echo "uname not available" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ -f /etc/os-release ]; then
    echo "OS Release:" >> "$OUTPUT_FILE"
    cat /etc/os-release >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

echo "Reproducibility manifest generated: $OUTPUT_FILE"

# Also save with git commit hash in filename for archival
if command -v git &> /dev/null && [ -d ".git" ]; then
    COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
    # git's short hash is always [0-9a-f]+ (or the "nogit" fallback above), but
    # it names part of ARCHIVE_FILE below, so validate it before that path is
    # built rather than trust the command that produced it.
    case "$COMMIT_HASH" in
        *..*|*/*)
            echo "refusing unsafe COMMIT_HASH: $COMMIT_HASH" >&2
            exit 1
            ;;
    esac
    ARCHIVE_FILE="benchmarks/metadata/toolchain_manifest_${COMMIT_HASH}.txt"
    cp "$OUTPUT_FILE" "$ARCHIVE_FILE"
    echo "Archived as: $ARCHIVE_FILE"
fi
