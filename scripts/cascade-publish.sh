#!/usr/bin/env bash
# SPEC-HF-PUBLISH-001 v1.0.0 — crates.io cascade publish in dep order.
#
# Usage:
#   scripts/cascade-publish.sh                # run full cascade for current workspace version
#   scripts/cascade-publish.sh --check        # report which crates are still behind
#   scripts/cascade-publish.sh --tier 1       # publish only Tier 1 (leaves)
#
# Prerequisites:
#   - Workspace version already bumped (Cargo.toml + per-crate Cargo.toml refs)
#   - Cargo.lock regenerated (cargo check --workspace)
#   - All workspace tests pass
#   - CARGO_REGISTRY_TOKEN exported or `cargo login` already run
#
# This script bypasses `make publish` (which has a known issue with .cargo/config.toml
# stub interaction on some crates per the v0.34.0 cascade observation) and uses
# direct `cargo publish` calls. It also tolerates "already at target version" as success.

set +e  # don't exit on individual crate failures — track each and report at end

# Tier definitions per SPEC-HF-PUBLISH-001 § crates.io release cascade.
declare -A TIERS
TIERS[1]="aprender-contracts-macros aprender-quant aprender-gemm-codegen aprender-sparse aprender-solve aprender-rand aprender-fft aprender-image aprender-tensor aprender-cupti"
TIERS[2]="aprender-contracts aprender-core aprender-profile-core aprender-graph"
TIERS[3]="aprender-profile"
TIERS[4]="aprender-gpu"
TIERS[5]="aprender-cuda-edge aprender-cgp"
TIERS[6]="aprender-compute"
TIERS[7]="aprender-cbtop aprender-ptx-debug aprender-explain"
TIERS[8]="aprender-common aprender-train-common aprender-train aprender-train-lora aprender-train-distill aprender-serve aprender-mcp aprender-data aprender-orchestrate"
TIERS[9]="aprender-present-core aprender-present-layout aprender-present-yaml aprender-present-widgets aprender-present-terminal"
TIERS[10]="apr-cli"
TIERS[11]="aprender"
TIERS[12]="aprender-tsp aprender-monte-carlo aprender-shell"
TIERS[13]="aprender-test-derive aprender-test-lib aprender-test-js-gen aprender-test-cli aprender-test-showcase aprender-zram-core aprender-zram-adaptive aprender-zram-generator aprender-zram-cli aprender-zram aprender-present-test-macros aprender-present-test aprender-present-lib aprender-present-cli aprender-db aprender-rag aprender-viz aprender-registry aprender-distribute aprender-simulate aprender-verify aprender-verify-ml aprender-train-shell aprender-train-inspect aprender-train-bench aprender-train-wasm aprender-contracts-cli"

TARGET_VERSION=$(grep -E '^version = "' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
[ -z "$TARGET_VERSION" ] && { echo "ERROR: could not detect target version from Cargo.toml"; exit 1; }
echo "Target version: $TARGET_VERSION"

MODE="${1:-publish}"
ONLY_TIER="${2:-}"

check_version() {
  local crate=$1
  curl -s "https://crates.io/api/v1/crates/$crate" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('crate',{}).get('max_version','none'))" 2>/dev/null
}

publish_crate() {
  local crate=$1
  local cur=$(check_version "$crate")
  if [ "$cur" = "$TARGET_VERSION" ]; then
    echo "  ✓ $crate (already $TARGET_VERSION)"
    return 0
  fi
  echo -n "  → $crate ($cur → $TARGET_VERSION): "
  local out
  out=$(cargo publish -p "$crate" --allow-dirty --locked 2>&1 | tail -3)
  if echo "$out" | grep -q "Published $crate"; then
    echo "✓ PUBLISHED"
    sleep 10  # let crates.io index settle before dependents try to fetch
    return 0
  elif echo "$out" | grep -qE "already.*upload|already exists"; then
    echo "(already on registry)"
    return 0
  else
    local err
    err=$(echo "$out" | grep -oE "candidate versions found which didn't match:.*$" | head -1)
    [ -z "$err" ] && err=$(echo "$out" | grep -oE "Caused by:.*$" | head -1)
    [ -z "$err" ] && err=$(echo "$out" | head -1)
    echo "DEFER ($err)"
    return 1
  fi
}

# Backup .cargo/config.toml once (publish needs a clean one without [patch.crates-io])
if [ -f .cargo/config.toml ]; then
  cp .cargo/config.toml .cargo/config.toml.cascade-backup
  echo "# Clean config for cascade publishing" > .cargo/config.toml
fi
trap 'if [ -f .cargo/config.toml.cascade-backup ]; then cp .cargo/config.toml.cascade-backup .cargo/config.toml && rm -f .cargo/config.toml.cascade-backup; fi' EXIT

if [ "$MODE" = "--check" ]; then
  echo ""
  echo "=== STATUS REPORT (not publishing) ==="
  any_behind=0
  for tier in $(echo "${!TIERS[@]}" | tr ' ' '\n' | sort -n); do
    for crate in ${TIERS[$tier]}; do
      cur=$(check_version "$crate")
      if [ "$cur" != "$TARGET_VERSION" ]; then
        echo "  T$tier  $crate: $cur"
        any_behind=1
      fi
    done
  done
  [ $any_behind -eq 0 ] && echo "  ✅ ALL at $TARGET_VERSION"
  exit 0
fi

# Walk tiers in order
DEFERRED=""
for tier in $(echo "${!TIERS[@]}" | tr ' ' '\n' | sort -n); do
  if [ -n "$ONLY_TIER" ] && [ "$tier" != "$ONLY_TIER" ]; then
    continue
  fi
  echo ""
  echo "=== TIER $tier ==="
  for crate in ${TIERS[$tier]}; do
    publish_crate "$crate" || DEFERRED="$DEFERRED $crate"
  done
done

if [ -n "$DEFERRED" ]; then
  echo ""
  echo "=== RETRY ROUND ==="
  STILL_DEFERRED=""
  for crate in $DEFERRED; do
    publish_crate "$crate" || STILL_DEFERRED="$STILL_DEFERRED $crate"
  done
  if [ -n "$STILL_DEFERRED" ]; then
    echo ""
    echo "❌ FAILED to publish:$STILL_DEFERRED"
    exit 1
  fi
fi

echo ""
echo "=== FINAL VERIFICATION ==="
ALL_OK=1
for tier in $(echo "${!TIERS[@]}" | tr ' ' '\n' | sort -n); do
  for crate in ${TIERS[$tier]}; do
    cur=$(check_version "$crate")
    if [ "$cur" != "$TARGET_VERSION" ]; then
      echo "  ✗ $crate at $cur (expected $TARGET_VERSION)"
      ALL_OK=0
    fi
  done
done

if [ $ALL_OK -eq 1 ]; then
  echo "  ✅ ALL crates at $TARGET_VERSION"
  echo ""
  echo "Run: cargo install aprender --force && apr --version"
  exit 0
else
  exit 1
fi
