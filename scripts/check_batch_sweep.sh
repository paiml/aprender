#!/usr/bin/env bash
# check_batch_sweep.sh -- the six obligations of contracts/batch-sweep-v1.yaml (PRB-001 §7).
#   bs1 REPO MAIN PR_REF...      every hunk of each absorbed PR (merge-base..PR) is on MAIN
#   bs2 REPO BASE HEAD N         BASE..HEAD = N ticket commits + 1 trailing "batch: regenerate", no merge commit
#   bs3 REPO MAIN CLOSED.tsv     "PR<TAB>sha" per closed PR: sha must be an ancestor of MAIN
#   bs4 ABSORBED.tsv             "PR<TAB>author<TAB>paths..." must hit neither the author nor the gate-path deny-list
#   bs5 BODY.md                  no closing keyword before an issue/PR number in a batch PR body
#   bs6 REPO 'REGEN CMD'         running REGEN CMD in REPO leaves the tree unchanged
#   --self-test                  each obligation on a violating fixture (must exit 1) and a conforming one (must exit 0)
# Exit: 0 held . 1 violated . 2 could not check. LAB: wired into no workflow.
set -uo pipefail
DENY_AUTHORS="${BS_DENY_AUTHORS:-guyernest}"
DENY_PATHS="${BS_DENY_PATHS:-^\.github/|^ci/|^deny\.toml$|^Makefile$|^rust-toolchain|^scripts/(ci/|contracts_gate\.sh|contracts_resolve\.sh|readme_sync\.sh|batch_fold\.sh|check_)}"
GEN_PATHS=(':!contracts/contracts.nt' ':!contracts/shapes.ttl' ':!contracts/witness' ':!contracts/census.json' ':!README.md')  # README.md: judged below, generated lines collapsed
bad() { echo "BS-$1 VIOLATED: $2"; return 1; }
bs1_readme() {  # README.md judged with its generated lines collapsed (CONTRACT_COUNT blocks, frontmatter contract_count); the rest is hand-written (PRB-001 §6)
  local repo="$1" main="$2" mb="$3" pr="$4" wt="$5" n='s/(<!-- CONTRACT_COUNT_START -->).*(<!-- CONTRACT_COUNT_END -->)/\1\2/;s/^contract_count: .*/contract_count: N/' rc=0
  git -C "$repo" show "$mb:README.md" 2> /dev/null | sed -E "$n" > "$wt.rb" || { rm -f -- "${wt:?}.rb"; return 0; }
  git -C "$repo" show "$pr:README.md" 2> /dev/null | sed -E "$n" > "$wt.rp"; git -C "$repo" show "$main:README.md" 2> /dev/null | sed -E "$n" > "$wt/README.md"
  diff -u --label a/README.md --label b/README.md "$wt.rb" "$wt.rp" > "$wt.rpatch"
  if [ -s "$wt.rpatch" ] && ! git -C "$wt" apply -R --check "$wt.rpatch" 2> /dev/null; then rc=1; fi
  rm -f -- "${wt:?}.rb" "${wt:?}.rp" "${wt:?}.rpatch"; return "$rc"
}
bs1() {
  local repo="$1" main="$2" pr mb wt rc=0; shift 2
  wt="$(mktemp -d)" || return 2
  git -C "$repo" worktree add -q --detach "$wt" "$main" || { rm -rf -- "${wt:?}"; return 2; }
  for pr in "$@"; do
    mb="$(git -C "$repo" merge-base "$main" "$pr")" || { bad 1 "no merge-base for $pr"; rc=1; continue; }
    git -C "$repo" diff "$mb" "$pr" -- . "${GEN_PATHS[@]}" > "$wt.patch" || { bad 1 "git diff failed for $pr"; rc=1; continue; }
    if [ -s "$wt.patch" ] && ! git -C "$wt" apply -R --check "$wt.patch" 2> /dev/null; then bad 1 "$pr has a hunk missing from $main"; rc=1; fi
    bs1_readme "$repo" "$main" "$mb" "$pr" "$wt" || { bad 1 "$pr has a hand-written README.md hunk missing from $main"; rc=1; }
  done
  git -C "$repo" worktree remove --force "$wt"; rm -f -- "${wt:?}.patch"; return "$rc"
}
bs2() {
  local repo="$1" base="$2" head="$3" n="$4" c m last
  c="$(git -C "$repo" rev-list --count "$base..$head")" && m="$(git -C "$repo" rev-list --merges --count "$base..$head")" || return 2
  last="$(git -C "$repo" log -1 --format=%s "$head")"
  [ "$m" = 0 ] || { bad 2 "$m merge commit(s) in $base..$head"; return 1; }
  [ "$c" = "$((n + 1))" ] || { bad 2 "$c commits, want $n tickets + 1 = $((n + 1))"; return 1; }
  case "$last" in "batch: regenerate"*) ;; *) bad 2 "last commit is '$last', not 'batch: regenerate...'"; return 1 ;; esac  # prefix: batch_fold.sh writes a longer subject
}
bs3() {
  local repo="$1" main="$2" tsv="$3" pr sha rc=0; [ -r "$tsv" ] || return 2
  while IFS=$'\t' read -r pr sha; do
    [ -n "$pr" ] || continue
    if [ -z "$sha" ] || ! git -C "$repo" merge-base --is-ancestor "$sha" "$main" 2> /dev/null; then bad 3 "#$pr closed, '$sha' is not on $main"; rc=1; fi
  done < "$tsv"; return "$rc"
}
bs4() {
  local pr author paths p rc=0; [ -r "$1" ] || return 2
  while IFS=$'\t' read -r pr author paths; do
    [ -n "$pr" ] || continue
    case ",$DENY_AUTHORS," in *",$author,"*) bad 4 "#$pr is by external author $author"; rc=1 ;; esac
    for p in $paths; do [[ "$p" =~ $DENY_PATHS ]] && { bad 4 "#$pr touches gate path $p"; rc=1; }; done
  done < "$1"; return "$rc"
}
bs5() {
  [ -r "$1" ] || return 2; local hit; hit="$(grep -inE '\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\b[: ]+([A-Za-z0-9_./-]+)?#[0-9]+' "$1")" || true
  [ -z "$hit" ] || { bad 5 "closing keyword: $hit"; return 1; }
}
bs6() {
  local repo="$1" cmd="$2" d
  (cd "$repo" && bash -c "$cmd") > /dev/null 2>&1 || return 2
  d="$(git -C "$repo" status --porcelain)"
  [ -z "$d" ] || { bad 6 "regeneration changed: $d"; return 1; }
}
selftest() {
  local fail=0; t="$(mktemp -d)" || exit 2; trap 'rm -rf -- "${t:?}"' EXIT
  g() { git -C "$t/r" -c user.name=t -c user.email=t@t "$@" > /dev/null 2>&1; }
  expect() { # label want cmd...
    local label="$1" want="$2" rc; shift 2; "$@" > "$t/out" 2>&1; rc=$?
    if [ "$rc" = "$want" ]; then printf '  ok   %s (rc %s)\n' "$label" "$rc"; else printf '  BROKE %s: rc %s, want %s\n' "$label" "$rc" "$want"; cat "$t/out"; fail=$((fail + 1)); fi
  }
  git init -q -b main "$t/r"; printf 'a\n' > "$t/r/a.txt"; printf 'b\nc\n' > "$t/r/gen.nt"; g add -A; g commit -qm base
  g checkout -qb prA; printf 'a\nA\n' > "$t/r/a.txt"; g commit -qam ticketA
  g checkout -qb prB main; printf 'B\n' > "$t/r/b.txt"; g add -A; g commit -qm ticketB
  g checkout -q main; g cherry-pick prA prB
  expect "BS-1 GREEN both PRs on main" 0 bs1 "$t/r" main prA prB
  g checkout -qb prC main~2; printf 'C\n' > "$t/r/c.txt"; g add -A; g commit -qm ticketC; g checkout -q main
  expect "BS-1 RED a PR whose hunk never landed" 1 bs1 "$t/r" main prC
  git init -q -b main "$t/rd"; printf 'contract_count: 1\nx <!-- CONTRACT_COUNT_START -->5<!-- CONTRACT_COUNT_END -->\nhand\n' > "$t/rd/README.md"; git -C "$t/rd" add -A; git -C "$t/rd" commit -qm base; git -C "$t/rd" checkout -qb prN; printf 'contract_count: 2\nx <!-- CONTRACT_COUNT_START -->6<!-- CONTRACT_COUNT_END -->\nhand\n' > "$t/rd/README.md"; git -C "$t/rd" commit -qam count; git -C "$t/rd" checkout -qb prH main; printf 'contract_count: 1\nx <!-- CONTRACT_COUNT_START -->5<!-- CONTRACT_COUNT_END -->\nhand\nlost\n' > "$t/rd/README.md"; git -C "$t/rd" commit -qam hand; git -C "$t/rd" checkout -q main; printf 'contract_count: 3\nx <!-- CONTRACT_COUNT_START -->7<!-- CONTRACT_COUNT_END -->\nhand\n' > "$t/rd/README.md"; git -C "$t/rd" commit -qam regen
  expect "BS-1 GREEN README count block differs, hand text same" 0 bs1 "$t/rd" main prN
  expect "BS-1 RED a hand-written README hunk never landed" 1 bs1 "$t/rd" main prH
  g checkout -qb good main~2; g cherry-pick prA prB; printf 'b\nc\n' > "$t/r/gen.nt"; g commit -q --allow-empty -m "batch: regenerate"
  expect "BS-2 GREEN 2 tickets + regenerate" 0 bs2 "$t/r" main~2 good 2
  g checkout -qb long main~2; g cherry-pick prA; g commit -q --allow-empty -m "batch: regenerate the generated set once (batch_fold.sh --regen)"; expect "BS-2 GREEN batch_fold.sh long subject (prefix)" 0 bs2 "$t/r" main~2 long 1
  expect "BS-2 could not check a bad ref (rc 2)" 2 bs2 "$t/r" nope good 2
  g checkout -qb merged main~2; g merge -q --no-ff -m "Merge prA" prA; g commit -q --allow-empty -m "batch: regenerate"
  expect "BS-2 RED a merge commit (what batch_fold.sh produces until PRB-001 §5.4)" 1 bs2 "$t/r" main~2 merged 1
  g checkout -qb amended main~2; g cherry-pick prA; g commit -q --amend -m "ticketA + regenerate"
  expect "BS-2 RED regenerate amended into a ticket" 1 bs2 "$t/r" main~2 amended 1
  printf '1\t%s\n' "$(git -C "$t/r" rev-parse main)" > "$t/ok.tsv"; printf '2\t%s\n' "$(git -C "$t/r" rev-parse prC)" > "$t/early.tsv"
  expect "BS-3 GREEN closed after its sha is on main" 0 bs3 "$t/r" main "$t/ok.tsv"
  expect "BS-3 RED closed before its sha is on main" 1 bs3 "$t/r" main "$t/early.tsv"
  expect "BS-3 could not check a missing file (rc 2)" 2 bs3 "$t/r" main "$t/none.tsv"
  printf '7\tnoah\tcrates/x/src/lib.rs docs/a.md\n' > "$t/a-ok.tsv"
  printf '8\tguyernest\tcrates/x/src/lib.rs\n' > "$t/a-ext.tsv"; printf '9\tnoah\t.github/workflows/ci.yml\n' > "$t/a-gate.tsv"
  expect "BS-4 GREEN internal author, source paths" 0 bs4 "$t/a-ok.tsv"
  expect "BS-4 RED external author" 1 bs4 "$t/a-ext.tsv"
  expect "BS-4 RED gate-file path" 1 bs4 "$t/a-gate.tsv"
  printf '10\tnoah\tscripts/contracts_gate.sh\n' > "$t/a-gs.tsv"; expect "BS-4 RED gate script path" 1 bs4 "$t/a-gs.tsv"
  printf 'Supersedes #5724, #5558.\nRefs #5831\n' > "$t/b-ok.md"; printf 'Batch.\nCloses #5724\n' > "$t/b-bad.md"; printf 'Fixes: paiml/aprender#5558\n' > "$t/b-bad2.md"
  expect "BS-5 GREEN Supersedes/Refs only" 0 bs5 "$t/b-ok.md"
  printf 'Batch of 3; fixed 3 flakes, closes 2 gaps.\n' > "$t/b-prose.md"; expect "BS-5 GREEN prose with no issue number" 0 bs5 "$t/b-prose.md"
  expect "BS-5 RED 'Closes #N'" 1 bs5 "$t/b-bad.md"
  expect "BS-5 RED 'Fixes: org/repo#N'" 1 bs5 "$t/b-bad2.md"
  g checkout -q main; printf 'c\nb\n' > "$t/r/gen.nt"; g commit -qam unsorted
  expect "BS-6 RED committed artifact differs from regeneration" 1 bs6 "$t/r" 'sort gen.nt -o gen.nt'
  printf 'b\nc\n' > "$t/r/gen.nt"; g commit -qam sorted
  expect "BS-6 GREEN artifact equals regeneration" 0 bs6 "$t/r" 'sort gen.nt -o gen.nt'
  printf 'check_batch_sweep self-test: %d broke\n' "$fail"; [ "$fail" -eq 0 ]
}
case "${1:-}" in
  --self-test) selftest ;;
  bs1|bs2|bs3|bs4|bs5|bs6) f="$1"; shift; "$f" "$@" ;;
  -h|--help|"") awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0" ;;
  *) echo "check_batch_sweep: unknown '$1'" >&2; exit 2 ;;
esac
