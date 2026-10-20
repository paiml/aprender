#!/usr/bin/env bash
# receipt_ok_arg.sh BODY_FILE TRUST_DIR PR_HEAD_SHA  (PMAT-5797, #5797)
# Prints ONE word for the tier step's `--receipt-ok`: 1 only when the receipt in the PR body
# verified, 0 in every other case (no receipt, a bad one, a missing file, a crashed verifier,
# a base that has no judge yet). Always exits 0: 0 leaves the PR on the tier its diff selects (quick or
# full, scripts/ci_test_tier.sh pull_request case), as if there were no receipt.
#
# This does not cover a PR that edits the CI files or this judge: merge_group ignores
# --receipt-ok but runs only the quick tier (full on tag/release), from the PR's own CI files,
# so it does not re-judge such an edit with a trusted copy. CI-edit review is the only backstop.
#
# A PR cannot edit its own judge (the #5793 pattern): the verifier, the toolchain-id recipe
# and the pinned batcher public key are read from HEAD of TRUST_DIR, the base-branch checkout
# of a job that never runs PR-tree code, never from the working
# tree. The working tree supplies only what is being judged: HEAD^{tree} and
# rust-toolchain.toml. The caller must itself run THIS script from TRUST_DIR too
# (`git -C TRUST_DIR show HEAD:scripts/ci/receipt_ok_arg.sh`). Replace refs are off for every
# git call here, so a PR cannot make its own HEAD^{tree} read as another tree.
# Cases: scripts/check_receipt_gate.sh (rows W1-W34).
set -u
export GIT_NO_REPLACE_OBJECTS=1
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
body=${1:-}; trust=${2:-}; sha=${3:-}
[ -n "$trust" ] && [ -d "$trust" ] && [ -n "$sha" ] || { printf '0\n'; exit 0; }
j=$(mktemp -d) || { printf '0\n'; exit 0; }
trap 'rm -rf "${j:?}"' EXIT
tg() { GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null git -C "$trust" "$@"; }
tg rev-parse --verify -q 'HEAD^{commit}' >/dev/null 2>&1 || { printf '0\n'; exit 0; }
tg archive HEAD scripts/ci_receipt_gate.sh scripts/receipt_toolchain_id.sh .flow009/batcher.pub 2>/dev/null \
  | tar -x -C "$j" 2>/dev/null || { printf '0\n'; exit 0; }
# rustc runs in a dir holding the BASE's rust-toolchain.toml, never in the PR cwd (a `path=`
# override there would make `rustc -vV` run PR-chosen code). No toml at the base: an empty dir.
mkdir -p "$j/rc" && { tg show HEAD:rust-toolchain.toml >"$j/rc/rust-toolchain.toml" 2>/dev/null || rm -f "${j:?}/rc/rust-toolchain.toml"; }
export RECEIPT_RUSTC_CWD="$j/rc"
# The verifier never sees the job's credentials or its output/env/path files.
# env -i: no PATH entry, BASH_ENV function, RUSTUP_*/GIT_* variable or credential survives.
first=$(env -i HOME="${HOME:-/nonexistent}" ${CARGO_HOME:+CARGO_HOME="$CARGO_HOME"} ${RUSTUP_HOME:+RUSTUP_HOME="$RUSTUP_HOME"} \
  PATH=/usr/local/bin:/usr/bin:/bin GIT_NO_REPLACE_OBJECTS=1 RECEIPT_RUSTC_CWD="$RECEIPT_RUSTC_CWD" \
  /usr/bin/bash "$j/scripts/ci_receipt_gate.sh" verify "$body" "$j/.flow009/batcher.pub" "$sha" 2>/dev/null | head -n1) || first=""
if [ "$first" = "RECEIPT_OK=1" ]; then printf '1\n'; else printf '0\n'; fi
