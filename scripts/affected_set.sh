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
READERS="${AFFECTED_READERS:-apr-cli aprender-build-sha aprender-cbtop aprender-cgp aprender-ci-tools aprender-compute aprender-compute-xtask aprender-contracts aprender-contracts-cli aprender-contrastive-data aprender-core aprender-data aprender-db aprender-decide aprender-distribute aprender-explain aprender-gpu aprender-mcp aprender-orchestrate aprender-present-cli aprender-present-core aprender-present-lib aprender-present-terminal aprender-present-yaml aprender-profile aprender-qa-certify aprender-qa-cli aprender-qa-gen aprender-qa-report aprender-qa-runner aprender-quant aprender-rag aprender-registry aprender-review-experiment aprender-rollup aprender-serve aprender-shell aprender-simulate aprender-test-cli aprender-test-lib aprender-train aprender-train-distill aprender-train-inspect aprender-train-shell aprender-tsp aprender-update aprender-verify-ml aprender-viz aprender-viz-ttop facades}"
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
touched=""
# FLOW-010 P3 (#5799): Cargo.lock narrowed by its [[package]] blocks. Prints every lock package that changed between
# the two revisions or depends (transitively, in the head lock) on one that did; fails (-> ALL) on anything unreadable.
lock_closure() { # $1 base $2 head
  local b h
  b="$(git merge-base "$1" "$2" 2>/dev/null)" || return 1
  h="$(git rev-parse -q --verify "$2" 2>/dev/null)" || return 1
  { git show "$b:Cargo.lock" 2>/dev/null | awk -v side=B -f <(lock_awk); git show "$h:Cargo.lock" 2>/dev/null | awk -v side=H -f <(lock_awk); } \
    | awk -F'\t' '
        $1 == "B" { bl[$2 "\t" $3 "\t" $4] = 1; next }
        $1 == "H" { hl[$2 "\t" $3 "\t" $4] = 1; nm[$2] = 1; n = split($4, d, ","); for (i = 1; i <= n; i++) if (d[i] != "") rev[d[i]] = rev[d[i]] " " $2; next }
        END {
          for (k in bl) if (!(k in hl)) { split(k, p, "\t"); ch[p[1]] = 1 }
          for (k in hl) if (!(k in bl)) { split(k, p, "\t"); ch[p[1]] = 1 }
          for (c in ch) q[++qn] = c
          for (i = 1; i <= qn; i++) { m = split(rev[q[i]], r, " "); for (j = 1; j <= m; j++) if (!(r[j] in ch)) { ch[r[j]] = 1; q[++qn] = r[j] } }
          for (c in ch) print c
        }' | sort -u
}
lock_awk() { # one TSV row per [[package]]: side, name, version, sorted dep names
  cat <<'AWK'
function flush() { if (name != "") { n = split(deps, d, ","); asort(d); s = ""; for (i = 1; i <= n; i++) s = s (i > 1 ? "," : "") d[i]; print side "\t" name "\t" ver "\t" s } name = ""; ver = ""; deps = ""; ind = 0 }
/^\[\[package\]\]/ { flush(); next }
/^name = / { gsub(/^name = "|"$/, ""); name = $0; next }
/^version = / { v = $0; gsub(/^version = "|"$/, "", v); ver = v; next }
/^dependencies = \[/ { ind = 1; next }
ind && /^\]/ { ind = 0; next }
ind { x = $0; gsub(/^ *"|",? *$/, "", x); sub(/ .*/, "", x); deps = deps (deps == "" ? "" : ",") x; next }
END { flush() }
AWK
}
while IFS= read -r f; do
  case "$f" in
    '') continue ;;
    Cargo.lock)
      [ -z "$from" ] || { emit ALL; exit 0; }   # --diff-from carries no revisions: nothing to compare, so ALL
      lk="$(lock_closure "${1:-origin/main}" "${2:-HEAD}")" || { emit ALL; exit 0; }
      [ -n "$lk" ] || { emit ALL; exit 0; }
      while IFS= read -r l; do
        cut -f1 <<<"$dirs" | grep -qxF -- "$l" || continue   # an external package: only the members it reaches matter
        case " $CORE " in *" $l "*) emit ALL; exit 0 ;; esac
        touched="$touched"$'\n'"$l"
      done <<<"$lk"
      continue ;;
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
[ -n "$touched" ] || { emit ALL; exit 0; }   # nothing placed in a package: nothing may be skipped
# The closure, then aprender-core: its falsification_spec_v10 test scans every crate's src, so ANY non-ALL set includes it (and the READERS).
res="$(jq -r --argjson t "$(printf '%s\n' "$touched" | jq -R . | jq -s .)" --arg rd "$READERS" '
  (.packages | map({name, deps: [.dependencies[].name]})) as $p
  | def grow($s): ($s + [$p|.[] | select(any(.deps[]; . as $d | $s | index($d))) | .name] | unique);
    (.packages | map(.name)) as $m
    | (($t | until(grow(.) == .; grow(.))) + ($rd | split(" ")) | unique) | .[] | select(. as $n | $m | index($n))' <<<"$meta")" || { emit ALL; exit 0; }
[ -n "$res" ] || { emit ALL; exit 0; }
mapfile -t out <<<"$res"
emit "${out[@]}"
