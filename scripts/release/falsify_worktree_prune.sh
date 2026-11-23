#!/usr/bin/env bash
# Falsifier for the stale-worktree registration defect (run 41454580958: publish-dryrun exit 128).
# A runner workdir that was deleted leaves a registered-but-missing worktree; `git worktree add`
# to that same path then exits 128. The workflows run `git worktree prune &&` first.
# Case table: stale registration + prune -> add succeeds (rc 0); stale registration, no prune -> rc 128.
# --mutants: a step that drops the prune must be refused (the without-prune case must fail).
set -euo pipefail

stale_case() { # $1 = "prune" | "noprune"; prints the rc of the add
  local t rc
  t=$(mktemp -d)
  (
    cd "$t"
    git init -q repo && cd repo
    git -c user.email=a@b -c user.name=n commit -q --allow-empty -m init
    git worktree add -q --detach "$t/wt" HEAD
    rm -rf -- "${t:?}/wt"
    if [ "$1" = prune ]; then
      git worktree prune && git worktree add -q --detach "$t/wt" HEAD
    else
      git worktree add -q --detach "$t/wt" HEAD
    fi
  ) >/dev/null 2>&1 && rc=0 || rc=$?
  rm -rf -- "${t:?}"
  echo "$rc"
}

self_test() {
  local fail=0 got
  got=$(stale_case prune);   [ "$got" = 0 ]   || { echo "FAIL prune: rc=$got want 0"; fail=1; }
  got=$(stale_case noprune); [ "$got" = 128 ] || { echo "FAIL noprune: rc=$got want 128"; fail=1; }
  [ "$fail" = 0 ] && echo "ok: stale registration -> prune rc 0, no prune rc 128"
  return "$fail"
}

# every workflow that adds a worktree must prune in the same command
sites() {
  local bad=0 f
  for f in .github/workflows/release-lanes-nightly.yml .github/workflows/rc-cut.yml .github/workflows/guards-nightly.yml; do
    if awk '/git worktree add/ && !/git worktree prune/ {bad=1} END {exit !bad}' "$f"; then
      echo "FAIL $f: worktree add without a prune"; bad=1
    fi
  done
  return "$bad"
}

mutants() {
  # mutant: no prune -> the case must observe 128, i.e. the guard would catch the drop
  [ "$(stale_case noprune)" != 0 ] || { echo "SURVIVED: no-prune mutant passes"; return 1; }
  echo "ok: no-prune mutant refused"
}

case "${1:---self-test}" in
  --self-test) self_test && sites ;;
  --mutants) mutants ;;
  *) echo "usage: $0 [--self-test|--mutants]" >&2; exit 2 ;;
esac
