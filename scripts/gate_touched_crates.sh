#!/usr/bin/env bash
# gate_touched_crates.sh — BSE-16: select the minimal touched-crate test set
# for `make gate`, instead of `cargo test --workspace` (42-94 min measured in
# docs/reports/work-history-delay-optimization-report.md) or `cargo test --lib`
# (root-crate-only in an 82-crate workspace — pmat verify.rs:543, spec F-09: a
# green that means nothing).
#
# Algorithm (docs/specifications/build-system-enhancement.md BSE-16, infra repo):
#   1. Diff <comparand>...HEAD (or --diff-from FILE, for the self-test) for
#      touched paths.
#   2. Map each touched `crates/<name>/...` path to its workspace crate <name>,
#      validated against `cargo metadata` (a directory under crates/ need not
#      be a workspace member — e.g. `crates/facades/`).
#   3. Fail closed to `cargo check --workspace --tests` (full matrix left to
#      CI) when: the diff touches the ROOT Cargo.toml, Cargo.lock or
#      rust-toolchain.toml; OR the touched-crate set plus each one's DIRECT
#      reverse dependents (from `cargo metadata`'s per-package `dependencies`)
#      exceeds CAP crates.
#   4. Otherwise: `cargo test -p <crate> ...` for the touched crates and their
#      direct reverse dependents.
#
# Always prints the selection and the rule that produced it, so a human or the
# self-test can see WHY a given crate set (or the full-workspace fallback) was
# chosen.
#
# Usage: scripts/gate_touched_crates.sh [--dry-run] [--diff-from FILE]
#   --dry-run        print the plan, run nothing (used by the self-test)
#   --diff-from FILE read touched paths from FILE (one per line) instead of
#                    `git diff --name-only <comparand>...HEAD`
set -euo pipefail
cd "$(dirname "$0")/.."

CAP=3
COMPARAND="${GATE_COMPARAND:-origin/main}"
DRY_RUN=0
DIFF_FROM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --print-selection)
      PRINT_SELECTION=1
      shift
      ;;
    --diff-from)
      [ $# -ge 2 ] || { echo "gate_touched_crates: --diff-from requires a file argument" >&2; exit 2; }
      DIFF_FROM="$2"
      shift 2
      ;;
    *)
      echo "gate_touched_crates: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -n "$DIFF_FROM" ]; then
  touched="$(cat "$DIFF_FROM")"
else
  touched="$(git diff --name-only "${COMPARAND}...HEAD")"
fi

if [ -z "$touched" ]; then
  echo "gate_touched_crates: no files touched vs ${COMPARAND} — nothing to select"
  exit 0
fi

metadata="$(cargo metadata --no-deps --format-version 1)"
names_json="$(jq -c '[.packages[].name]' <<< "$metadata")"

workspace_wide=0
declare -A touched_crates=()
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    Cargo.toml|Cargo.lock|rust-toolchain.toml)
      workspace_wide=1
      ;;
    crates/*/*)
      name="${f#crates/}"
      name="${name%%/*}"
      if jq -e --arg n "$name" '. as $names | ($names | index($n)) != null' <<< "$names_json" >/dev/null; then
        touched_crates["$name"]=1
      fi
      ;;
    *)
      : # no workspace-crate test surface for this path
      ;;
  esac
done <<< "$touched"

declare -A selected=()
for c in "${!touched_crates[@]}"; do
  selected["$c"]=1
done

# BEGIN reverse-dependents-expansion
if [ "$workspace_wide" -eq 0 ] && [ "${#touched_crates[@]}" -gt 0 ]; then
  edges="$(jq -r --argjson names "$names_json" '
    .packages[] as $p
    | $p.dependencies[]?
    | select((.kind == null or .kind == "normal") and (.name as $d | $names | index($d) != null))
    | [$p.name, .name] | @tsv
  ' <<< "$metadata")"
  if [ -n "$edges" ]; then
    while IFS=$'\t' read -r dependent dependency; do
      if [ -n "${touched_crates[$dependency]+x}" ]; then
        selected["$dependent"]=1
      fi
    done <<< "$edges"
  fi
fi
# END reverse-dependents-expansion

count=${#selected[@]}

if [ "$workspace_wide" -eq 1 ]; then
  action="check"
  rule="root Cargo.toml/Cargo.lock/rust-toolchain.toml touched -> full workspace check"
elif [ "$count" -eq 0 ]; then
  action="none"
  rule="no workspace crate touched -> nothing to test"
elif [ "$count" -gt "$CAP" ]; then
  action="check"
  rule="selection of ${count} crate(s) exceeds cap (${CAP}) -> full workspace check"
else
  action="test"
  rule="touched crate(s) + direct reverse dependents: ${count} crate(s), within cap (${CAP})"
fi

sorted_list() {
  if [ "$#" -eq 0 ]; then
    printf '(none)'
    return 0
  fi
  printf '%s\n' "$@" | sort | paste -sd' ' -
}

# --print-selection (BSE-17, PMAT-1077): one machine-readable line for the CI
# tier decision, no cargo run. `selection=full` when the rule falls closed to the
# whole workspace, `none` when nothing is touched, `quick` with the sorted set.
if [ "${PRINT_SELECTION:-0}" -eq 1 ]; then
  case "$action" in
    check) printf 'selection=full crates= rule=%s\n' "$rule" ;;
    none)  printf 'selection=none crates= rule=%s\n' "$rule" ;;
    test)  printf 'selection=quick crates=%s rule=%s\n' "$(sorted_list "${!selected[@]}")" "$rule" ;;
  esac
  exit 0
fi

echo "gate_touched_crates: comparand=${COMPARAND}"
echo "gate_touched_crates: touched crate(s): $(sorted_list "${!touched_crates[@]}")"
echo "gate_touched_crates: selected (touched + direct reverse dependents): $(sorted_list "${!selected[@]}")"
echo "gate_touched_crates: rule: ${rule}"

case "$action" in
  none)
    echo "gate_touched_crates: + (nothing to run)"
    exit 0
    ;;
  check)
    cmd=(cargo check --workspace --tests)
    ;;
  test)
    cmd=(cargo test)
    for c in $(sorted_list "${!selected[@]}"); do
      cmd+=(-p "$c")
    done
    ;;
esac

echo "gate_touched_crates: + ${cmd[*]}"

if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

exec "${cmd[@]}"
