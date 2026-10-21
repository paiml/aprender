#!/usr/bin/env bash
# FLOW-010 E5: the diff -> the set of workspace packages whose tests it can change.
#
#   scripts/affected_set.sh [--filter] [--diff-from FILE] [BASE [HEAD]]
#
# Prints one package per line, or the single word ALL. With --filter it prints the
# nextest filterset instead (`package(a) | package(b)`, or `all()` for ALL).
# Fail-safe: anything this tool cannot place in a package is ALL.
#   crates/<pkg>/...        the package + its reverse-dependency closure (cargo metadata)
#   a CORE crate            ALL (its closure is the workspace; naming it would only hide that)
#   Cargo.toml of a crate   ALL (a manifest edit changes features/deps, which no path can bound)
#   docs/, root *.md, ci/ (sections.yml, explicit-test-commands.d/, ...), Cargo.lock, root Cargo.toml,
#   rust-toolchain.toml, .cargo/, .github/, Makefile, any other path outside a crate
#                           ALL (tests read README.md, CLAUDE.md, ci/sections.yml and docs/ at run time)
#   docs/findings/, .claude/, ci/shard-step-costs.tsv
#                           NO-READER list (#5809): the only run-time readers of these paths (grep of crates/**/*.rs: contracts-cli,
#                           orchestrate, core, mcp, apr-cli) are all in READERS, so the set is READERS alone. Alone or
#                           beside crate .rs files; any other path in the diff still decides ALL.
# Any non-.rs file under crates/<pkg>/ (README.md, *.json, *.toml, ...) is ALL: other crates' tests read them
# (aprender-core readme_contract.rs reads every crates/*/README.md; aprender-contracts reads aprender-serve/kernel-registry.json).
# An empty diff, or a diff git cannot compute (bad base: exit 128), is ALL: nothing was
# measured, so nothing may be skipped.
# git diff runs with --no-renames: a moved file is its old path AND its new path (a rename
# out of a crate would otherwise show only the destination and the source crate would be skipped).
# --diff-from reads the changed paths from FILE (tests). AFFECTED_META / AFFECTED_ROOT
# replace `cargo metadata` and the checkout root (tests; synthetic graphs).
set -euo pipefail
# CORE = ALL on touch. apr-cli is here because other crates' tests read its sources (aprender-core readme_contract.rs reads
# apr-cli/src/commands_enum.rs; spec_checklist_*, toyota_principles_tests). Reader edges are not in cargo metadata, so a
# crate whose source another crate reads at run time must be listed (aprender-quant: aprender-serve/capability.rs reads
# aprender-quant/src/ggml_type.rs). Any crates/*/build.rs is ALL: contracts/build_helper.rs reads crates/*/build.rs.
CORE="${AFFECTED_CORE:-aprender-core aprender-compute aprender-contracts apr-cli aprender-quant}"
# READERS = crates whose tests read OTHER crates' files at run time (CARGO_MANIFEST_DIR/.., crates/, ../<crate>). CORE ones are ALL on touch, but that does not put them in a non-ALL set; every non-ALL set includes them, because the touched crate may be what they read. scripts/check_affected_readers.sh
# goes RED on a reader this list (or CORE) lacks.
READERS="${AFFECTED_READERS:-apr-cli apr-format aprender-build-sha aprender-cbtop aprender-cgp aprender-ci-tools aprender-compute aprender-compute-xtask aprender-contracts aprender-contracts-cli aprender-contracts-macros aprender-contrastive-data aprender-core aprender-data aprender-db aprender-decide aprender-distribute aprender-explain aprender-gpu aprender-graph aprender-mcp aprender-orchestrate aprender-present-cli aprender-present-core aprender-present-layout aprender-present-lib aprender-present-terminal aprender-present-widgets aprender-present-yaml aprender-profile aprender-ptx-debug aprender-qa-certify aprender-qa-cli aprender-qa-gen aprender-qa-report aprender-qa-runner aprender-quant aprender-rag aprender-rag-cli aprender-registry aprender-review-experiment aprender-rollup aprender-serve aprender-shell aprender-simulate aprender-test-cli aprender-test-lib aprender-train aprender-train-bench aprender-train-distill aprender-train-inspect aprender-train-lora aprender-train-shell aprender-tsp aprender-update aprender-verify-ml aprender-viz aprender-viz-ttop aprender-zram aprender-zram-cli aprender-zram-generator facades}"
filter=0 from=""
while [ $# -gt 0 ]; do
  case "$1" in
    --filter) filter=1; shift ;;
    --diff-from) from="$2"; shift 2 ;;
    *) break ;;
  esac
done
case "$from" in *..*) echo "affected_set: refusing a --diff-from with \"..\": $from" >&2; exit 2 ;; esac   # bashrs SEC010
if [ -n "$from" ]; then files="$(cat "$from")"
else files="$(git diff --name-only --no-renames "${1:-origin/main}...${2:-HEAD}")" || files=""; fi
emit() { # $@ = package names, or ALL
  if [ "${1:-}" = ALL ]; then [ "$filter" = 1 ] && echo 'all()' || echo ALL; return; fi
  if [ "$filter" = 1 ]; then printf '%s\n' "$@" | sed 's/.*/package(&)/' | paste -sd'|' | sed 's/|/ | /g'; else printf '%s\n' "$@"; fi
}
[ -n "$files" ] || { emit ALL; exit 0; }
if [ -n "${AFFECTED_META:-}" ]; then meta="$(cat "$AFFECTED_META")"
else meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)"; fi
root="${AFFECTED_ROOT:-$(git rev-parse --show-toplevel)}"
dirs="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta")" || { emit ALL; exit 0; }
[ -n "$dirs" ] || { emit ALL; exit 0; }
touched="" skipped=0
while IFS= read -r f; do
  case "$f" in */*) ;; *.md) skipped=1; continue ;; esac   # root *.md: its run-time readers are in READERS
  case "$f" in
    '') continue ;;
    contracts/binding.yaml | contracts/*/binding.yaml | contracts/model-families/* | contracts/*/model-families/*) emit ALL; exit 0 ;;   # build inputs of core/compute/contracts/simulate/train: their generated code reaches dependents
    contracts/* | docs/*) skipped=1; continue ;;   # FLOW-010(b): run-time readers are in READERS
    docs/findings/* | .claude/* | ci/shard-step-costs.tsv) skipped=1; continue ;;
    crates/*/build.rs) emit ALL; exit 0 ;;
    crates/*.rs) ;;
    crates/*) emit ALL; exit 0 ;;
  esac
  # longest manifest dir that prefixes the path
  pkg="$(awk -F'\t' -v f="$f" 'index(f, $2 "/") == 1 && length($2) > best { best = length($2); n = $1 } END { print n }' <<<"$dirs")"
  [ -n "$pkg" ] || { emit ALL; exit 0; }
  case " $CORE " in *" $pkg "*) emit ALL; exit 0 ;; esac
  touched="$touched"$'\n'"$pkg"
done <<<"$files"
touched="$(printf '%s\n' "$touched" | grep -v '^$' | sort -u || true)"
[ -n "$touched" ] || [ "$skipped" = 1 ] || { emit ALL; exit 0; }   # nothing placed in a package and nothing on the no-reader list: nothing may be skipped
# The closure, then aprender-core: its falsification_spec_v10 test scans every crate's src, so ANY non-ALL set includes it (and the READERS).
res="$(jq -r --argjson t "$(printf '%s\n' "$touched" | jq -R . | jq -s .)" --arg rd "$READERS" '
  (.packages | map({name, deps: [.dependencies[].name]})) as $p
  | def grow($s): ($s + [$p|.[] | select(any(.deps[]; . as $d | $s | index($d))) | .name] | unique);
    (.packages | map(.name)) as $m
    | (($t | until(grow(.) == .; grow(.))) + ($rd | split(" ")) | unique) | .[] | select(. as $n | $m | index($n))' <<<"$meta")" || { emit ALL; exit 0; }
[ -n "$res" ] || { emit ALL; exit 0; }
mapfile -t out <<<"$res"
emit "${out[@]}"
