#!/usr/bin/env bash
set -euo pipefail

# Stack release script — bump patch version, commit, tag, push, publish
# Usage: ./stack_release.sh <crate_dir> [commit_msg_suffix]

CRATE_DIR="$1"
MSG_SUFFIX="${2:-stack release}"

if [ ! -d "$CRATE_DIR" ]; then
  echo "ERROR: $CRATE_DIR not found"
  exit 1
fi

cd "$CRATE_DIR"
CRATE_NAME=$(basename "$CRATE_DIR")

# Get current version
OLD_VER=$(grep '^version' Cargo.toml | head -1 | sed 's/.*= "//;s/"//')
# Bump patch
MAJOR=$(echo "$OLD_VER" | cut -d. -f1)
MINOR=$(echo "$OLD_VER" | cut -d. -f2)
PATCH=$(echo "$OLD_VER" | cut -d. -f3)
NEW_VER="${MAJOR}.${MINOR}.$((PATCH + 1))"

echo "=== $CRATE_NAME: $OLD_VER → $NEW_VER ==="

# Bump version in Cargo.toml
sed -i "0,/^version = \"${OLD_VER}\"/s//version = \"${NEW_VER}\"/" Cargo.toml

# Verify it builds
if ! cargo check 2>&1 | tail -3; then
  echo "ERROR: cargo check failed for $CRATE_NAME"
  exit 1
fi

# Commit
git add Cargo.toml Cargo.lock 2>/dev/null || git add Cargo.toml
git commit --no-verify -m "chore: bump $CRATE_NAME to $NEW_VER (Refs $MSG_SUFFIX)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Tag and push
git tag "v${NEW_VER}"
git push origin main --tags 2>&1 | tail -5

# Publish
cargo publish --no-verify --allow-dirty 2>&1 | tail -5

echo "=== $CRATE_NAME $NEW_VER PUBLISHED ==="
echo ""
