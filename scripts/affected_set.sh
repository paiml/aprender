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
#   ci/ (sections.yml, explicit-test-commands.d/, ...), Cargo.lock, root Cargo.toml, rust-toolchain.toml, .cargo/, .github/, Makefile,
#   any other path outside a crate            ALL
#   docs/, contracts/, root *.md              READERS only (FLOW-010 b): their run-time readers are all in READERS. Exceptions, ALL:
#                           contracts binding.yaml, contracts model-families and the BUILD_CONTRACTS files (compiled into a crate)
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
filter=0 from="" bad=0
while [ $# -gt 0 ]; do
  case "$1" in
    --filter) filter=1; shift ;;
    --diff-from) [ $# -ge 2 ] || { bad=1; break; }; from="$2"; shift 2 ;;
    *) break ;;
  esac
done
emit() { # $@ = package names, or ALL
  if [ "${1:-}" = ALL ]; then [ "$filter" = 1 ] && echo 'all()' || echo ALL; return; fi
  if [ "$filter" = 1 ]; then printf '%s\n' "$@" | sed 's/.*/package(&)/' | paste -sd'|' | sed 's/|/ | /g'; else printf '%s\n' "$@"; fi
}
[ "$bad" = 0 ] || { emit ALL; exit 0; }   # --diff-from with no FILE
case "$from" in *..*) echo "affected_set: refusing a --diff-from with \"..\": $from" >&2; emit ALL; exit 0 ;; esac   # bashrs SEC010: refuse the path (never read it), fail closed with ALL
if [ -n "$from" ]; then files="$(cat "$from")" || files=""
else files="$(git diff --name-only --no-renames "${1:-origin/main}...${2:-HEAD}")" || files=""; fi
[ -n "$files" ] || { emit ALL; exit 0; }
if [ -n "${AFFECTED_META:-}" ]; then meta="$(cat "$AFFECTED_META")" || meta=""
else meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)" || meta=""; fi
root="${AFFECTED_ROOT:-$(git rev-parse --show-toplevel)}" || root=""
[ -n "$meta" ] && [ -n "$root" ] || { emit ALL; exit 0; }   # no metadata, no checkout root: nothing measured
dirs="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta")" || { emit ALL; exit 0; }
[ -n "$dirs" ] || { emit ALL; exit 0; }
# Root contracts/ files that build.rs or include_str!/include! turn into compiled code or constants (aprender-serve build.rs generates
# arch/tensor-name code from three of them; apr-cli and mcp embed others). A change reaches dependents of the compiling crate, so these are ALL.
# check_affected_readers.sh derives this list from the tree and is RED on a file it omits.
BUILD_CONTRACTS="${AFFECTED_BUILD_CONTRACTS:-apr-mcp-tool-schemas-v1.yaml apr-model-capability-v1.yaml arch-constraints-v1.yaml architecture-requirements-v1.yaml kernel-fusion-v1.yaml layer-parity-v1.yaml softmax-kernel-v1.yaml tensor-names-v1.yaml thinking-budgets-v1.yaml}"
touched="" skipped=0 lock=0
while IFS= read -r f; do
  case "$f" in */*) ;; *.md) skipped=1; continue ;; esac   # root *.md: its run-time readers are in READERS
  case "$f" in contracts/*) case " $BUILD_CONTRACTS " in *" ${f#contracts/} "*) emit ALL; exit 0 ;; esac ;; esac
  case "$f" in
    '') continue ;;
    contracts/binding.yaml | contracts/*/binding.yaml | contracts/model-families/* | contracts/*/model-families/*) emit ALL; exit 0 ;;   # build inputs of core/compute/contracts/simulate/train: their generated code reaches dependents
    contracts/* | docs/*) skipped=1; continue ;;   # FLOW-010(b): run-time readers are in READERS
    docs/findings/* | .claude/* | ci/shard-step-costs.tsv) skipped=1; continue ;;
    # FLOW-010 P3: a lock-only change is narrowed by its [[package]] diff, but only when git can answer
    # (no --diff-from: a bare path list has no content, so it stays ALL).
    Cargo.lock) if [ -z "$from" ]; then lock=1; continue; fi ;;
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
if [ "$lock" = 1 ]; then
  b="${1:-origin/main}"; h="${2:-HEAD}"
  mb="$(git merge-base "$b" "$h" 2>/dev/null || echo "$b")"
  # one line per [[package]] block: name<TAB>block with newlines as |
  lock_pkgs() { git show "$1:Cargo.lock" 2>/dev/null | awk -v RS='\\[\\[package\\]\\]\n' 'match($0, /name = "[^"]+"/) { n = substr($0, RSTART + 8, RLENGTH - 9); gsub(/\n/, "|"); print n "\t" $0 }'; }
  old="$(lock_pkgs "$mb" | sort -u || true)"; new="$(lock_pkgs "$h" | sort -u || true)"
  # unreadable lock on either side, or no changed block: cannot place the change -> ALL
  { [ -n "$old" ] && [ -n "$new" ]; } || { emit ALL; exit 0; }
  changed="$(comm -13 <(printf '%s\n' "$old") <(printf '%s\n' "$new") | cut -f1 | sort -u)"
  [ -n "$changed" ] || { emit ALL; exit 0; }
  # reverse closure over the lock graph (a changed external crate reaches members through other externals)
  seeds="$(printf '%s\n' "$new" | awk -F'\t' -v chg="$changed" '
    BEGIN { n = split(chg, c, "\n"); for (i = 1; i <= n; i++) hit[c[i]] = 1 }
    { split($2, a, "|"); for (i = 1; i <= length(a); i++) if (a[i] ~ /^ "/) { d = a[i]; gsub(/^ *"|"?,?$/, "", d); sub(/ .*/, "", d); dep[++m] = d; who[m] = $1 } }
    END { do { g = 0; for (i = 1; i <= m; i++) if ((dep[i] in hit) && !(who[i] in hit)) { hit[who[i]] = 1; g = 1 } } while (g)
          for (k in hit) print k }')"
  touched="$touched"$'\n'"$(jq -r --argjson s "$(printf '%s\n' "$seeds" | jq -R . | jq -s .)" '.packages[].name | select(. as $n | $s | index($n))' <<<"$meta")"
fi
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
