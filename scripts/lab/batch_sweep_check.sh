#!/usr/bin/env bash
# batch_sweep_check.sh -- the six obligations of contracts/batch-sweep-v1.yaml (PRB-001 §7).
#   bs1 REPO MAIN PR_REF...      every hunk of each absorbed PR (merge-base..PR) is on MAIN
#   bs2 REPO BASE HEAD N         BASE..HEAD = N ticket commits + 1 trailing "batch: regenerate", no merge commit
#   bs3 REPO MAIN CLOSED.tsv     "PR<TAB>sha" per closed PR: sha must be an ancestor of MAIN
#   bs4 ABSORBED.tsv             "PR<TAB>author<TAB>paths..." author must be on BS_ALLOW_AUTHORS (empty = none) and no path on the gate-path deny-list
#   bs5 BODY.md                  no closing keyword before an issue/PR number in a batch PR body
#   bs6 REPO 'REGEN CMD'         running REGEN CMD in REPO leaves the tree unchanged
#   --self-test                  each obligation on a violating fixture (must exit 1) and a conforming one (must exit 0)
# BS-1 judges README.md only outside CONTRACT_COUNT blocks, with one context line (-U1) pinning each hunk; BS-4 denies all of .github/, ci/ and scripts/lib/, plus Cargo.toml, scripts/arm_*, .pmat/, scripts/check_*.sh, release|publish|sign|forjar scripts and configs, gate*.sh.
# Exit: 0 held . 1 violated . 2 could not check. LAB: wired into no workflow.
# No -e: every obligation is tallied by an explicit return code (0/1/2), so one violation must not stop the rest.
set -uo pipefail
ALLOW_AUTHORS="${BS_ALLOW_AUTHORS:-}"  # BS-4: any author not listed is external; empty = nobody allowed, so the caller must pass it
DENY_PATHS="${BS_DENY_PATHS:-^\.github/|^ci/|^scripts/lib/|^\.pmat-gates\.toml$|^deny\.toml$|^scripts/ci/|^scripts/check_[^/]*\.sh$|^scripts/contracts_gate\.sh$|^Makefile$|^\.clippy\.toml$|^rust-toolchain\.toml$|^Cargo\.toml$|^scripts/arm_|^\.pmat/|(^|/)(release|publish|sign|forjar)[^/]*\.(sh|ya?ml|toml)$|(^|/)gate[^/]*\.sh$}"
# Generated: the contract artifacts, and ONLY the CONTRACT_COUNT blocks of README.md (PRB-001 §6), neutralised by readme_neutral.
GEN_PATHS=(':!contracts/contracts.nt' ':!contracts/shapes.ttl' ':!contracts/witness' ':!README.md')
readme_neutral() { sed 's/<!-- CONTRACT_COUNT_START -->.*<!-- CONTRACT_COUNT_END -->/<!-- CONTRACT_COUNT -->/'; }
bad() { echo "BS-$1 VIOLATED: $2"; return 1; }
# Every hunk of PATCH must already be on the tree at WT. An addition/replacement hunk lands iff it reverse-applies. A
# DELETION-only hunk reverse-applies anywhere (re-inserting a line always works), so it lands iff it no longer applies forward.
hunks_landed() {
  local wt="$1" patch="$2" d f ok=0
  [ -s "$patch" ] || return 0
  d="$(mktemp -d)" || return 2
  awk -v d="$d" '/^diff --git /{h=""; n=0} /^@@/{n++; f=sprintf("%s/h%06d", d, ++c); printf "%s", h > f; close(f); cur=f} !/^@@/ && n==0{h=h $0 "\n"} n>0{print >> cur}' "$patch"
  for f in "$d"/h*; do
    [ -e "$f" ] || continue
    # landed means the hunk reverse-applies (its context is intact and its lines are in place); a deletion-only hunk whose context main edited fails here too, which counts as not landed (fail closed)
    git -C "$wt" apply -R --check "$f" 2> /dev/null || ok=1
  done
  rm -rf -- "${d:?}"; return "$ok"
}
bs1() {
  local repo="$1" main="$2" pr mb wt rc=0; shift 2
  [ "$#" -gt 0 ] || { echo "BS-1 no PR refs given, nothing checked"; return 2; }
  wt="$(mktemp -d)" || return 2
  git -C "$repo" worktree add -q --detach "$wt" "$main" || { rm -rf -- "${wt:?}"; return 2; }
  for pr in "$@"; do
    git -C "$repo" rev-parse --verify -q "$pr^{commit}" > /dev/null || { echo "BS-1 could not read $pr"; rc=2; continue; }
    mb="$(git -C "$repo" merge-base "$main" "$pr")" || { bad 1 "no merge-base for $pr"; rc=1; continue; }
    # -U1: one context line on each side pins a hunk to where it belongs (zero context let a lost `}` land anywhere). A main edit ON that line is a false RED: LAB, fails closed.
    git -C "$repo" diff -U1 "$mb" "$pr" -- . "${GEN_PATHS[@]}" > "$wt.patch"
    if git -C "$repo" cat-file -e "$mb:README.md" 2> /dev/null && git -C "$repo" cat-file -e "$pr:README.md" 2> /dev/null; then
      git -C "$repo" show "$mb:README.md" | readme_neutral > "$wt.a"; git -C "$repo" show "$pr:README.md" | readme_neutral > "$wt.b"
      git -C "$wt" diff --no-index -U1 --src-prefix=a/ --dst-prefix=b/ "$wt.a" "$wt.b" 2> /dev/null | sed -e "s#a$wt\.a#a/README.md#g" -e "s#b$wt\.b#b/README.md#g" -e "s#^--- .*#--- a/README.md#" -e "s#^+++ .*#+++ b/README.md#" >> "$wt.patch"
      [ -f "$wt/README.md" ] && readme_neutral < "$wt/README.md" > "$wt.m" && cp "$wt.m" "$wt/README.md"
    else # README on one side only (added or deleted): judged whole, nothing to neutralise
      git -C "$repo" diff -U1 "$mb" "$pr" -- README.md >> "$wt.patch"
    fi
    hunks_landed "$wt" "$wt.patch"; case $? in 0) ;; 2) echo "BS-1 could not check $pr"; [ "$rc" = 1 ] || rc=2 ;; *) bad 1 "$pr has a hunk missing from $main"; rc=1 ;; esac
  done
  git -C "$repo" worktree remove --force "$wt"; rm -f -- "${wt:?}.patch" "${wt:?}.a" "${wt:?}.b" "${wt:?}.m"; return "$rc"
}
bs2() {
  local repo="$1" base="$2" head="$3" n="$4" c m last
  case "$n" in [1-7]) ;; *) echo "BS-2 N must be 1..7 (the batch cap), got '$n'"; return 2 ;; esac
  c="$(git -C "$repo" rev-list --count "$base..$head" 2> /dev/null)" && m="$(git -C "$repo" rev-list --merges --count "$base..$head" 2> /dev/null)" || { echo "BS-2 could not read $base..$head"; return 2; }
  last="$(git -C "$repo" log -1 --format=%s "$head")"
  [ "$m" = 0 ] || { bad 2 "$m merge commit(s) in $base..$head"; return 1; }
  [ "$c" = "$((n + 1))" ] || { bad 2 "$c commits, want $n tickets + 1 = $((n + 1))"; return 1; }
  [ "$last" = "batch: regenerate" ] || { bad 2 "last commit is '$last', not 'batch: regenerate'"; return 1; }
}
bs3() {
  local repo="$1" main="$2" tsv="$3" pr sha rc=0 n=0
  while IFS=$'\t' read -r pr sha; do
    [ -n "$pr" ] || continue; n=$((n + 1))
    if [ -z "$sha" ] || ! git -C "$repo" merge-base --is-ancestor "$sha" "$main" 2> /dev/null; then bad 3 "#$pr closed, '$sha' is not on $main"; rc=1; fi
  done < "$tsv"; [ "$n" -gt 0 ] || { echo "BS-3 0 rows in $tsv, nothing checked"; return 2; }; echo "BS-3 $n rows checked"; return "$rc"
}
bs4() {
  local pr author paths p rc=0 n=0
  while IFS=$'\t' read -r pr author paths; do
    [ -n "$pr" ] || continue; n=$((n + 1))
    case ",$ALLOW_AUTHORS," in *",$author,"*) ;; *) bad 4 "#$pr is by external author $author"; rc=1 ;; esac
    for p in $paths; do [[ "$p" =~ $DENY_PATHS ]] && { bad 4 "#$pr touches gate path $p"; rc=1; }; done
  done < "$1"; [ "$n" -gt 0 ] || { echo "BS-4 0 rows in $1, nothing checked"; return 2; }; echo "BS-4 $n rows checked"; return "$rc"
}
bs5() {
  local hit; [ -r "${1:-}" ] || { echo "BS-5 could not read '${1:-}'"; return 2; }
  hit="$(grep -inE '\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\b[: ]+(#|[A-Za-z0-9_./-]+#|https?://[^ ]*/(issues|pull)/)[0-9]+' "$1")" || true
  [ -z "$hit" ] || { bad 5 "closing keyword: $hit"; return 1; }
}
bs6() {
  local repo="$1" cmd="$2" d
  (cd "$repo" && bash -c "$cmd") > /dev/null 2>&1 || return 2
  d="$(git -C "$repo" status --porcelain 2> /dev/null)" || { echo "BS-6 $repo is not a git work tree"; return 2; }
  [ -z "$d" ] || { bad 6 "regeneration changed: $d"; return 1; }
}
selftest() {
  local fail=0 rows=0; t="$(mktemp -d)" || exit 2; trap 'rm -rf -- "${t:?}"' EXIT
  g() { git -C "$t/r" -c user.name=t -c user.email=t@t "$@" > /dev/null 2>&1; }
  expect() { # label want cmd...
    local label="$1" want="$2" rc; shift 2; "$@" > "$t/out" 2>&1; rc=$?; rows=$((rows + 1))
    if [ "$rc" = "$want" ]; then printf '  ok   %s (rc %s)\n' "$label" "$rc"; else printf '  BROKE %s: rc %s, want %s\n' "$label" "$rc" "$want"; cat "$t/out"; fail=$((fail + 1)); fi
  }
  git init -q -b main "$t/r"; printf 'a\n' > "$t/r/a.txt"; printf 'b\nc\n' > "$t/r/gen.nt"; g add -A; g commit -qm base
  g checkout -qb prA; printf 'a\nA\n' > "$t/r/a.txt"; g commit -qam ticketA
  g checkout -qb prB main; printf 'B\n' > "$t/r/b.txt"; g add -A; g commit -qm ticketB
  g checkout -q main; g cherry-pick prA prB
  expect "BS-1 GREEN both PRs on main" 0 bs1 "$t/r" main prA prB
  g checkout -qb prC main~2; printf 'C\n' > "$t/r/c.txt"; g add -A; g commit -qm ticketC; g checkout -q main
  expect "BS-1 RED a PR whose hunk never landed" 1 bs1 "$t/r" main prC
  b0="$(git -C "$t/r" rev-parse main~2)"
  # edits on BOTH sides of a landed hunk (git apply only trims context symmetrically) must not false-RED a landed hunk (-U1: edits two or more lines away), and README is judged only outside CONTRACT_COUNT blocks
  printf '1\n2\n3\n4\n5\n6\n7\n8\n' > "$t/r/f.txt"; g add -A; g commit -qm f-base; g checkout -qb prN; printf '1\n2\n3\nfour\n5\n6\n7\n8\n' > "$t/r/f.txt"; g commit -qam prN-edit
  g checkout -qb prD main; printf '1\n2\n3\n4\n5\n6\nseven\n8\n' > "$t/r/f.txt"; g commit -qam prD-edit; g checkout -q main
  printf 'o\n' > "$t/r/other.txt"; g add -A; g commit -qm unrelated; g cherry-pick prN; printf 'one\ntwo\n3\nfour\n5\nsix\nseven7\n8\n' > "$t/r/f.txt"; g commit -qam neighbour
  expect "BS-1 GREEN landed hunk with a neighbouring edit on main" 0 bs1 "$t/r" main prN
  expect "BS-1 RED unlanded hunk beside a neighbouring edit" 1 bs1 "$t/r" main prD
  printf 'del1\ndel2\n' > "$t/r/d.txt"; g add -A; g commit -qm del-base; g checkout -qb prDel; printf 'del1\n' > "$t/r/d.txt"; g commit -qam del-only; g checkout -q main
  expect "BS-1 RED deletion-only hunk that never landed" 1 bs1 "$t/r" main prDel
  g cherry-pick prDel > /dev/null 2>&1 || { printf 'del1\n' > "$t/r/d.txt"; g commit -qam del-landed; }
  expect "BS-1 GREEN deletion-only hunk that landed" 0 bs1 "$t/r" main prDel
  printf 'a\nb\nc\nd\n' > "$t/r/e.txt"; g add -A; g commit -qm e-base; g checkout -qb prE; printf 'a\nb\nd\n' > "$t/r/e.txt"; g commit -qam e-del-c; g checkout -q main
  printf 'a\nB\nc\nd\n' > "$t/r/e.txt"; g commit -qam e-main-edits-neighbour
  expect "BS-1 RED deletion-only hunk whose neighbour main edited (forward apply fails, not landed)" 1 bs1 "$t/r" main prE
  printf 'x <!-- CONTRACT_COUNT_START -->1<!-- CONTRACT_COUNT_END --> y\nprose\n' > "$t/r/README.md"; g add -A; g commit -qm readme-base
  g checkout -qb prR; printf 'x <!-- CONTRACT_COUNT_START -->2<!-- CONTRACT_COUNT_END --> y\nprose\n' > "$t/r/README.md"; g commit -qam count-only
  g checkout -qb prP main; printf 'x <!-- CONTRACT_COUNT_START -->1<!-- CONTRACT_COUNT_END --> y\nnew prose\n' > "$t/r/README.md"; g commit -qam prose-edit; g checkout -q main
  expect "BS-1 GREEN README CONTRACT_COUNT-only change is generated" 0 bs1 "$t/r" main prR
  expect "BS-1 RED README prose change that never landed" 1 bs1 "$t/r" main prP
  g checkout -qb prRA "$b0"; printf 'brand new readme\n' > "$t/r/README.md"; g add -A; g commit -qm readme-added; g checkout -q main
  expect "BS-1 RED README added on one side only, never landed" 1 bs1 "$t/r" main prRA
  g checkout -qb good $b0; g cherry-pick prA prB; printf 'b\nc\n' > "$t/r/gen.nt"; g commit -q --allow-empty -m "batch: regenerate"
  expect "BS-2 GREEN 2 tickets + regenerate" 0 bs2 "$t/r" $b0 good 2
  g checkout -qb merged $b0; g merge -q --no-ff -m "Merge prA" prA; g commit -q --allow-empty -m "batch: regenerate"
  expect "BS-2 RED a merge commit (the --no-ff batch_fold.sh)" 1 bs2 "$t/r" $b0 merged 1
  g checkout -qb amended $b0; g cherry-pick prA; g commit -q --amend -m "ticketA + regenerate"
  expect "BS-2 RED regenerate amended into a ticket" 1 bs2 "$t/r" $b0 amended 1
  printf '1\t%s\n' "$(git -C "$t/r" rev-parse main)" > "$t/ok.tsv"; printf '2\t%s\n' "$(git -C "$t/r" rev-parse prC)" > "$t/early.tsv"
  expect "BS-3 GREEN closed after its sha is on main" 0 bs3 "$t/r" main "$t/ok.tsv"
  expect "BS-3 RED closed before its sha is on main" 1 bs3 "$t/r" main "$t/early.tsv"
  : > "$t/empty.tsv"; expect "BS-3 could-not-check on an empty list" 2 bs3 "$t/r" main "$t/empty.tsv"
  expect "BS-2 could-not-check on an unreadable ref" 2 bs2 "$t/r" nosuchref good 1
  printf '7\tnoah\tcrates/x/src/lib.rs docs/a.md\n' > "$t/a-ok.tsv"
  printf '8\tguyernest\tcrates/x/src/lib.rs\n' > "$t/a-ext.tsv"; printf '9\tnoah\t.github/workflows/ci.yml\n' > "$t/a-gate.tsv"
  printf '3\tnoah\tci/explicit-test-commands.d/9-x.cmd\n' > "$t/a-cmd.tsv"; printf '4\tnoah\t.pmat-gates.toml\n' > "$t/a-pmat.tsv"; printf '5\tnoah\tscripts/check_x.sh\n' > "$t/a-chk.tsv"
  export BS_ALLOW_AUTHORS=noah; ALLOW_AUTHORS=noah
  printf '10\tsomeoneexternal\tcrates/x/src/lib.rs\n' > "$t/a-ext2.tsv"; printf '11\tnoah\tCargo.toml scripts/arm_pr_automerge.sh\n' > "$t/a-arm.tsv"
  expect "BS-4 GREEN internal author, source paths" 0 bs4 "$t/a-ok.tsv"
  expect "BS-4 RED any author not on the allow-list" 1 bs4 "$t/a-ext2.tsv"
  expect "BS-4 RED root Cargo.toml and scripts/arm_*" 1 bs4 "$t/a-arm.tsv"
  printf '12\tnoah\tdocs/design.md crates/x/src/signal.rs docs/release-notes.md\n' > "$t/a-name.tsv"; expect "BS-4 GREEN ordinary paths whose names merely contain sign/release" 0 bs4 "$t/a-name.tsv"
  printf '13\tnoah\tscripts/release_publish.sh\n' > "$t/a-rel.tsv"; expect "BS-4 RED a release script" 1 bs4 "$t/a-rel.tsv"
  printf '14\tnoah\tscripts/sign_artifacts.sh\n' > "$t/a-sig.tsv"; expect "BS-4 RED a signing script" 1 bs4 "$t/a-sig.tsv"
  ALLOW_AUTHORS=; expect "BS-4 RED empty allow-list fails closed" 1 bs4 "$t/a-ok.tsv"; ALLOW_AUTHORS=noah
  expect "BS-4 RED default allow-list is empty (no env, script run as the caller would)" 0 bash -c 'out="$(env -u BS_ALLOW_AUTHORS bash "$0" bs4 "$1" 2>&1)"; case "$out" in *"external author noah"*) exit 0 ;; *) exit 1 ;; esac' "$0" "$t/a-ok.tsv"
  expect "BS-4 RED external author" 1 bs4 "$t/a-ext.tsv"
  expect "BS-4 RED gate-file path" 1 bs4 "$t/a-gate.tsv"
  expect "BS-4 RED ci/explicit-test-commands.d path" 1 bs4 "$t/a-cmd.tsv"
  expect "BS-4 RED .pmat-gates.toml" 1 bs4 "$t/a-pmat.tsv"
  expect "BS-4 RED scripts/check_*.sh" 1 bs4 "$t/a-chk.tsv"
  for gp in ci/vendor/x.rs Makefile .clippy.toml rust-toolchain.toml scripts/contracts_gate.sh; do
    printf '6\tnoah\t%s\n' "$gp" > "$t/a-more.tsv"; expect "BS-4 RED $gp" 1 bs4 "$t/a-more.tsv"
  done
  printf 'Supersedes #5724, #5558.\nRefs #5831\n' > "$t/b-ok.md"; printf 'Batch.\nCloses #5724\n' > "$t/b-bad.md"; printf 'Fixes: paiml/aprender#5558\n' > "$t/b-bad2.md"
  printf 'Closes https://github.com/paiml/aprender/issues/5724\n' > "$t/b-url.md"; printf 'resolves https://github.com/paiml/aprender/pull/5558\n' > "$t/b-url2.md"
  expect "BS-5 RED 'Closes <issue URL>'" 1 bs5 "$t/b-url.md"
  expect "BS-5 RED 'resolves <pull URL>'" 1 bs5 "$t/b-url2.md"
  printf 'This fixes 2 flakes and closes 3 gaps.\n' > "$t/b-prose.md"
  expect "BS-5 GREEN prose 'fixes 2 flakes' without #/URL" 0 bs5 "$t/b-prose.md"
  expect "BS-5 GREEN Supersedes/Refs only" 0 bs5 "$t/b-ok.md"
  expect "BS-5 RED 'Closes #N'" 1 bs5 "$t/b-bad.md"
  expect "BS-5 RED 'Fixes: org/repo#N'" 1 bs5 "$t/b-bad2.md"
  g checkout -q main; printf 'c\nb\n' > "$t/r/gen.nt"; g commit -qam unsorted
  expect "BS-6 RED committed artifact differs from regeneration" 1 bs6 "$t/r" 'sort gen.nt -o gen.nt'
  printf 'b\nc\n' > "$t/r/gen.nt"; g commit -qam sorted
  expect "BS-6 GREEN artifact equals regeneration" 0 bs6 "$t/r" 'sort gen.nt -o gen.nt'
  # BS-1: -U1 pins a hunk to its place; a lost `}` must not land on some other `}` (zero context did that), and `++i;` is an addition
  printf 'x\n}\ny\nz\n' > "$t/r/g2.txt"; g add -A; g commit -qm g2-base; g checkout -qb prG; printf 'x\n}\ny\n}\nz\n' > "$t/r/g2.txt"; g commit -qam g2-add; g checkout -q main
  expect "BS-1 RED a lost '}' hunk beside another '}' on main" 1 bs1 "$t/r" main prG
  printf 'i\nj\n' > "$t/r/h2.txt"; g add -A; g commit -qm h2-base; g checkout -qb prI; printf 'i\n++k;\nj\n' > "$t/r/h2.txt"; g commit -qam h2-inc; g checkout -q main; g cherry-pick prI
  expect "BS-1 GREEN a landed '++k;' addition" 0 bs1 "$t/r" main prI
  expect "BS-1 could-not-check with no PR refs" 2 bs1 "$t/r" main
  expect "BS-1 could-not-check on a missing PR ref" 2 bs1 "$t/r" main nosuchpr
  expect "BS-2 could-not-check on N=8 (batch cap is 7)" 2 bs2 "$t/r" $b0 good 8
  expect "BS-2 could-not-check on N=0" 2 bs2 "$t/r" $b0 good 0
  for gp in .github/pr-review.pub ci/mutants-debt.tsv ci/guard-lane-steps.txt ci/step-budgets.tsv scripts/lib/ci_guard_steps.sh; do
    printf '15\tnoah\t%s\n' "$gp" > "$t/a-read.tsv"; expect "BS-4 RED gate-read file $gp" 1 bs4 "$t/a-read.tsv"
  done
  expect "BS-4 could-not-check on an empty list" 2 bs4 "$t/empty.tsv"
  expect "BS-4 could-not-check on a missing file" 2 bs4 "$t/no-such.tsv"
  expect "BS-5 could-not-check on a missing file" 2 bs5 "$t/no-such.md"
  expect "BS-6 could-not-check outside a git work tree" 2 bs6 "$t" 'true'
  expect "unknown subcommand exits 2" 2 bash "$0" bs7
  printf 'batch_sweep_check self-test: %d rows, %d broke\n' "$rows" "$fail"; [ "$rows" -gt 0 ] && [ "$fail" -eq 0 ]
}
case "${1:-}" in
  --self-test) selftest ;;
  bs1|bs2|bs3|bs4|bs5|bs6) f="$1"; shift; "$f" "$@" ;;
  -h|--help|"") awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0" ;;
  *) echo "batch_sweep_check: unknown '$1'" >&2; exit 2 ;;
esac
