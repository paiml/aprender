#!/bin/sh
# test-no-pull-request-target-guard.sh -- the guard must go red on each evasion and green on clean files.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
guard="$here/no-pull-request-target-guard.sh"
tmp=$(mktemp -d)
[ -n "$tmp" ] && [ -d "$tmp" ] || exit 2
trap 'rm -rf "${tmp:?}"' EXIT
fail=0

# expect <name> <want-rc> <dir> [allowlist]
expect() {
  name=$1 want=$2 d=$3 al=${4:-$tmp/none}
  rc=0
  sh "$guard" "$d" "$al" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "FAIL $name: want rc=$want got rc=$rc"
    fail=$((fail + 1))
  else
    echo "ok   $name"
  fi
}

# put <name> <relative-file> : content from stdin (printf-style done by caller)
put() { mkdir -p "$tmp/$1/$(dirname "$2")"; cat > "$tmp/$1/$2"; }

printf 'on:\n  pull_request:\n    types: [opened]\n' | put clean a.yml
printf 'on: push\n' | put clean-str b.yml
printf '# pull_request_target is banned\non:\n  pull_request:\n' | put comment a.yml
printf "concurrency:\n  x: \${{ github.event_name == 'pull_request_target' }}\non: push\n" | put expr-only a.yaml
printf 'on:\n  pull_request_target:\n    types: [opened]\n' | put map a.yml
printf 'on: [push, pull_request_target]\n' | put inline a.yml
printf 'on: pull_request_target\n' | put str a.yml
printf 'on: [push, "ci #1", pull_request_target]\n' | put quoted a.yml
printf 'on:\n  pull_request_target: {}  # trailing comment\n' | put trail a.yml
printf 'on: ["pull_request_\\u0074arget"]\n' | put escu a.yml
printf 'on: ["pull_request_\\x74arget"]\n' | put escx a.yml
printf 'on: ["pull_reques\\U00000074_target"]\n' | put escU a.yml
printf 'on: ["pull_request\\\n  _target"]\n' | put dqcont a.yml
printf "on: ['pull_request\n  _target']\n" | put sqflow a.yml
printf 'x: &t pull_request_target\non: [*t]\n' | put anchors a.yml
printf 'on:\n  - push\n  - pull_request_target\n' | put blockseq a.yml
printf 'on: {pull_request_target: {}}\n' | put flowmap a.yml
printf '\357\273\277on:\n  pull_request_target:\n' | put bom a.yml
printf 'on:\r\n  pull_request_target:\r\n' | put crlf a.yml
printf 'name: x\n---\non:\n  pull_request_target:\n' | put multidoc a.yml
printf 'on:\n  PULL_request_target:\n' | put upper a.yml
printf 'on: !x {pull_request_target: {}}\n' | put tagmap a.yml
printf 'on: !foo [push, pull_request_target]\n' | put tagseq a.yml
printf 'on: !!set {pull_request_target: }\n' | put tagset a.yml
printf 'on: !!omap [pull_request_target: {}]\n' | put tagomap a.yml
printf '!root\non:\n  pull_request_target:\n' | put tagroot a.yml
printf 'on: !s pull_request_target\n' | put tagscalar a.yml
printf 'ON:\n  pull_request_target:\n' | put upperkey a.yml
printf 'true:\n  pull_request_target:\n' | put truekey a.yml
printf 'on:\n  pull_request_target:\n' | put dotfile .hidden.yml
printf 'on:\n  pull_request_target:\n' | put subdir sub/deep/a.yml
printf 'on: [push\n' | put malformed a.yml
mkdir -p "$tmp/symlink"; printf 'on:\n  pull_request_target:\n' > "$tmp/symtarget.yml"; ln -s "$tmp/symtarget.yml" "$tmp/symlink/a.yml"
mkdir -p "$tmp/dangling"; ln -s "$tmp/nowhere.yml" "$tmp/dangling/a.yml"
mkdir -p "$tmp/empty"

expect clean-passes              0 "$tmp/clean"
expect clean-string-passes       0 "$tmp/clean-str"
expect comment-only-passes       0 "$tmp/comment"
expect expression-text-passes    0 "$tmp/expr-only"
expect map-fails                 1 "$tmp/map"
expect inline-seq-fails          1 "$tmp/inline"
expect string-form-fails         1 "$tmp/str"
expect quoted-hash-fails         1 "$tmp/quoted"
expect trailing-comment-fails    1 "$tmp/trail"
expect escape-u-fails            1 "$tmp/escu"
expect escape-x-fails            1 "$tmp/escx"
expect escape-U8-fails           1 "$tmp/escU"
expect dq-line-fold-fails        1 "$tmp/dqcont"
expect sq-fold-is-a-space-passes  0 "$tmp/sqflow"   # YAML folds the newline to a space: "pull_request _target" is no event
expect anchor-alias-fails        1 "$tmp/anchors"
expect block-seq-fails           1 "$tmp/blockseq"
expect flow-map-fails            1 "$tmp/flowmap"
expect bom-fails                 1 "$tmp/bom"
expect crlf-fails                1 "$tmp/crlf"
expect multi-doc-fails           1 "$tmp/multidoc"
expect uppercase-event-fails     1 "$tmp/upper"
for t in tagmap tagseq tagset tagomap tagroot tagscalar; do expect "yaml-tag-$t-fails" 1 "$tmp/$t"; done
expect uppercase-on-fails        1 "$tmp/upperkey"
expect true-key-fails            1 "$tmp/truekey"
expect dotfile-fails             1 "$tmp/dotfile"
expect subdir-fails              1 "$tmp/subdir"
expect symlink-fails             1 "$tmp/symlink"
expect dangling-symlink-fails    1 "$tmp/dangling"
expect malformed-yaml-fails      1 "$tmp/malformed"
expect zero-files-not-a-pass     2 "$tmp/empty"
expect missing-dir-is-usage      2 "$tmp/nope"

# allowlist: pinned by sha256; every edit must fail
good='name: closer
on:
  pull_request_target:
jobs:
  j:
    runs-on: ubuntu-latest
    steps:
      - run: echo close
'
printf '%s' "$good" | put al closer.yml
sum=$(sha256sum "$tmp/al/closer.yml" | cut -d' ' -f1)
printf 'closer.yml sha256:%s  # reason\n' "$sum" > "$tmp/pinned.txt"
printf 'closer.yml  # no pin\n' > "$tmp/nopin.txt"
printf 'closer.yml sha256:%064d\n' 0 > "$tmp/wrongpin.txt"
expect allowlisted-pinned-passes 0 "$tmp/al" "$tmp/pinned.txt"
expect allowlist-without-pin-fails 1 "$tmp/al" "$tmp/nopin.txt"
expect allowlist-wrong-pin-fails 1 "$tmp/al" "$tmp/wrongpin.txt"
expect not-allowlisted-fails     1 "$tmp/al"
for v in 'Actions/Checkout@v4' 'git checkout FETCH_HEAD' '"actions/checkout@v4"'; do
  n=edit-$(printf '%s' "$v" | tr -c 'A-Za-z' '_')
  case $v in
    git*) printf '%s      - run: %s\n' "$good" "$v" | put "$n" closer.yml ;;
    *)    printf '%s      - uses: %s\n' "$good" "$v" | put "$n" closer.yml ;;
  esac
  expect "edited-$v-fails" 1 "$tmp/$n" "$tmp/pinned.txt"
done
printf '%s  k:\n    uses: ./.github/workflows/other.yml\n' "$good" | put edit-callee closer.yml
expect edited-local-callee-fails 1 "$tmp/edit-callee" "$tmp/pinned.txt"
mkdir -p "$tmp/al-sym"; ln -s "$tmp/al/closer.yml" "$tmp/al-sym/closer.yml"
expect allowlisted-symlink-fails 1 "$tmp/al-sym" "$tmp/pinned.txt"

# the repo's own workflows must be clean
repo=$(cd "$here/../.." && pwd)
if [ -d "$repo/.github/workflows" ]; then
  rc=0
  sh "$guard" "$repo/.github/workflows" "$repo/.github/pull-request-target-allowlist.txt" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAIL repo-workflows: rc=$rc"; fail=$((fail + 1)); else echo "ok   repo-workflows"; fi
fi

[ "$fail" -eq 0 ] || { echo "$fail test(s) failed"; exit 1; }
echo "all guard tests passed"
