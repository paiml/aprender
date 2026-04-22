#!/bin/bash
# GitHub Actions pre-job hook.
# 1. Disk-pressure guard: prune target/ if / > HIGH_WATER_PCT.
# 2. Fix root-owned files in the workspace (container builds leave root-owned).
set -u

# 1. Disk guard (best-effort, never block the job)
/usr/local/bin/runner-disk-guard.sh --pre-job 2>&1 || true

# 2. Chown container-leftover files
WORKSPACE="${GITHUB_WORKSPACE:-}"
if [ -n "$WORKSPACE" ] && [ -d "$WORKSPACE" ]; then
  sudo chown -R noah:noah "$WORKSPACE" 2>/dev/null || true
fi
RUNNER_DIR="$(dirname "$(dirname "$WORKSPACE")" 2>/dev/null)"
if [ -n "$RUNNER_DIR" ] && [ -d "$RUNNER_DIR" ]; then
  sudo find "$RUNNER_DIR" -maxdepth 3 -user root -exec chown -R noah:noah {} + 2>/dev/null || true
fi
