#!/usr/bin/env bash
# check_receipt_gate.sh — case table for scripts/ci_receipt_gate.sh (PMAT-5797, #5797).
# The PR gate may skip the full test suite only for a batch whose signed receipt verifies:
#   S  the signature is valid for the pinned batcher public key,
#   T  the receipt's tree hash is the PR merge tree,
#   C  the receipt's toolchain hash is this checkout's pinned toolchain.
# Every row builds a throwaway repo and a throwaway key, so the table is hermetic.
# A row that expects a reason must see exactly that reason: a verifier that fails for
# any other cause does not satisfy it. `bash scripts/check_receipt_gate.sh` exits 0
# only when every row holds.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
V="$HERE/ci_receipt_gate.sh"
TID="$HERE/receipt_toolchain_id.sh"
NS="flow009-receipt"
td=$(mktemp -d) || exit 2
trap 'rm -rf "${td:?}"' EXIT
fail=0
pass=0

command -v ssh-keygen >/dev/null 2>&1 || { echo "ENV: ssh-keygen missing"; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "ENV: jq missing"; exit 2; }

ssh-keygen -q -t ed25519 -N '' -C batcher -f "$td/key" >/dev/null
ssh-keygen -q -t ed25519 -N '' -C other -f "$td/other" >/dev/null
printf '# batcher public key (test)\n%s\n' "$(cut -d' ' -f1,2 "$td/key.pub")" >"$td/pub"

repo="$td/repo"
git init -q "$repo"
git -C "$repo" config user.email t@example.invalid
git -C "$repo" config user.name t
printf '[toolchain]\nchannel = "stable"\n' >"$repo/rust-toolchain.toml"
printf 'x\n' >"$repo/a.txt"
git -C "$repo" add -A
git -C "$repo" commit -q -m "base" >/dev/null 2>&1
printf 'y\n' >>"$repo/a.txt"
git -C "$repo" commit -q -am "batch (PMAT-1)" >/dev/null 2>&1
tree=$(git -C "$repo" rev-parse 'HEAD^{tree}')
head=$(git -C "$repo" rev-parse HEAD)
tc=$(cd "$repo" && bash "$TID")

# receipt TREE TOOLCHAIN -> unsigned receipt JSON on stdout
receipt() {
  jq -n -S -c --arg tree "$1" --arg tc "$2" --arg sha "$head" '
    {schema:"flow009.receipt/1",repo:"r",batch:"b",base_sha:$sha,tree:$tree,
     commits:[{ticket:"PMAT-1",patch_id:"p",sha:$sha}],gate_results:{tests:"pass",coverage:"pass"},
     toolchain:$tc,manifest:"m",steps:"s",host:"h",started:"t0",finished:"t1",sig:""}'
}
# sign KEY RECEIPT_JSON_FILE -> signed JSON (canonical bytes: jq -S -c del(.sig) plus LF)
sign() {
  jq -S -c 'del(.sig)' "$2" >"$td/canon"
  rm -f "$td/canon.sig"
  ssh-keygen -q -Y sign -f "$1" -n "$NS" "$td/canon" >/dev/null 2>&1
  jq -S -c --rawfile sig "$td/canon.sig" '.sig = $sig' "$2"
}
# body FILE JSON -> a PR body holding the one fenced receipt block
body() { printf 'Batch.\n\n```flow009.receipt/1\n%s\n```\n' "$2" >"$1"; }

# row NAME EXPECT BODY_FILE [PUB]  — EXPECT is "ok" or the exact reason
row() {
  local name=$1 want=$2 bf=$3 pub=${4:-$td/pub} out
  out=$(cd "$repo" && bash "$V" verify "$bf" "$pub" "$head" 2>&1 | head -n1)
  if [ "$want" = ok ]; then
    [ "$out" = "RECEIPT_OK=1" ] && { pass=$((pass + 1)); echo "ok   $name"; return; }
  else
    [ "$out" = "RECEIPT_OK=0 reason=$want" ] && { pass=$((pass + 1)); echo "ok   $name"; return; }
  fi
  fail=$((fail + 1))
  echo "FAIL $name: want '$want' got '$out'"
}

good=$(receipt "$tree" "$tc"); printf '%s\n' "$good" >"$td/good.json"
signed=$(sign "$td/key" "$td/good.json")

body "$td/b-good" "$signed"
row "G1 valid receipt: signature, tree and toolchain all hold" ok "$td/b-good"

# S: signature
body "$td/b-s1" "$(printf '%s' "$signed" | jq -S -c '.batch = "tampered"')"
row "S1 a field changed after signing -> bad-signature" bad-signature "$td/b-s1"
printf '%s\n' "$good" >"$td/g2.json"
body "$td/b-s2" "$(sign "$td/other" "$td/g2.json")"
row "S2 signed by a key that is not the pinned batcher key -> bad-signature" bad-signature "$td/b-s2"
body "$td/b-s3" "$good"
row "S3 unsigned receipt (empty sig) -> bad-signature" bad-signature "$td/b-s3"
printf '# placeholder, no key yet\n' >"$td/pub-empty"
row "S4 placeholder public key fails closed -> placeholder-pubkey" placeholder-pubkey "$td/b-good" "$td/pub-empty"
printf 'no receipt here\n' >"$td/b-none"
row "S5 no fenced receipt block -> fenced-block" fenced-block "$td/b-none"

# T: tree hash == merge tree
receipt "0000000000000000000000000000000000000000" "$tc" >"$td/t1.json"
body "$td/b-t1" "$(sign "$td/key" "$td/t1.json")"
row "T1 correctly signed receipt for another tree -> tree-mismatch" tree-mismatch "$td/b-t1"

# C: toolchain hash
receipt "$tree" "$(printf 'not-this-toolchain' | sha256sum | cut -d' ' -f1)" >"$td/c1.json"
body "$td/b-c1" "$(sign "$td/key" "$td/c1.json")"
row "C1 correctly signed receipt for another toolchain -> toolchain-mismatch" toolchain-mismatch "$td/b-c1"
printf '[toolchain]\nchannel = "nightly"\n' >"$repo/rust-toolchain.toml"
row "C2 the checkout's pinned toolchain changed after signing -> toolchain-mismatch" toolchain-mismatch "$td/b-good"
printf '[toolchain]\nchannel = "stable"\n' >"$repo/rust-toolchain.toml"

# V: the signed verdict. A validly signed receipt must also say the batch passed.
receipt "$tree" "$tc" | jq -S -c '.gate_results.tests = "fail"' >"$td/v1.json"
body "$td/b-v1" "$(sign "$td/key" "$td/v1.json")"
row "V1 correctly signed receipt whose tests failed -> not-pass" not-pass "$td/b-v1"
receipt "$tree" "$tc" | jq -S -c 'del(.gate_results)' >"$td/v2.json"
body "$td/b-v2" "$(sign "$td/key" "$td/v2.json")"
row "V2 correctly signed receipt with no gate_results -> not-pass" not-pass "$td/b-v2"
receipt "$tree" "$tc" | jq -S -c '.gate_results = {coverage:"pass"}' >"$td/v3.json"
body "$td/b-v3" "$(sign "$td/key" "$td/v3.json")"
row "V3 correctly signed receipt with no tests verdict -> not-pass" not-pass "$td/b-v3"

# W: the workflow wiring. scripts/ci/receipt_ok_arg.sh prints the ONE word the tier step
# hands to ci_test_tier.sh as --receipt-ok. Anything but a verified receipt must print 0,
# and 0 must leave the PR on the quick/full tier, never on reuse. The judge (verifier,
# toolchain recipe, pinned key) comes from the trusted clone of the base, never from the PR's working tree.
W="$HERE/ci/receipt_ok_arg.sh"
T="$HERE/ci_test_tier.sh"
printf 'crates/aprender-core/src/lib.rs\n' >"$td/touched.txt"
# mkbase NAME PUBFILE|- -> a fresh repo $td/NAME whose HEAD holds the judge files (and the key,
# unless "-"): the shape of the trusted clone the workflow makes before any PR code runs.
mkbase() {
  local d="$td/$1" pubf=$2 files="scripts/ci_receipt_gate.sh scripts/receipt_toolchain_id.sh" f
  rm -rf "${d:?}"; git init -q "$d"
  mkdir -p "$d/scripts" "$d/.flow009"
  cp "$V" "$d/scripts/ci_receipt_gate.sh"; cp "$TID" "$d/scripts/receipt_toolchain_id.sh"
  cp "$repo/rust-toolchain.toml" "$d/rust-toolchain.toml"; files="$files rust-toolchain.toml"
  if [ "$pubf" != - ]; then cp "$pubf" "$d/.flow009/batcher.pub"; files="$files .flow009/batcher.pub"; fi
  for f in $files; do git -C "$d" add "$f"; done
  git -C "$d" -c user.email=t@example.invalid -c user.name=t commit -q -m judge
}
BASE="$td/trust-main"
mkbase trust-main "$td/pub"
mkbase trust-placeholder "$td/pub-empty"
git init -q "$td/trust-empty"
git -C "$td/trust-empty" -c user.email=t@example.invalid -c user.name=t commit -q --allow-empty -m none
  # bashrs disable-next-line=SEC010
arg() { (cd "$repo" && bash "${3:-$W}" "$1" "${2:-$BASE}" "$head" 2>/dev/null | head -n1); }
wrow() {
  local name=$1 want=$2 got=$3
  if [ "$got" = "$want" ]; then pass=$((pass + 1)); echo "ok   $name"; else fail=$((fail + 1)); echo "FAIL $name: want '$want' got '$got'"; fi
}
wrow "W1 verified receipt -> --receipt-ok 1" 1 "$(arg "$td/b-good")"
wrow "W2 no receipt in the PR body -> 0 (full gate)" 0 "$(arg "$td/b-none")"
wrow "W3 tampered receipt -> 0" 0 "$(arg "$td/b-s1")"
wrow "W4 receipt for another tree -> 0" 0 "$(arg "$td/b-t1")"
wrow "W5 placeholder pubkey at the base -> 0" 0 "$(arg "$td/b-good" "$td/trust-placeholder")"
wrow "W6 PR body file missing -> 0" 0 "$(arg "$td/no-such-body")"
wrow "W7 base has no judge files (bootstrap) -> 0" 0 "$(arg "$td/b-good" "$td/trust-empty")"
wrow "W11 base ref does not exist -> 0" 0 "$(arg "$td/b-good" "$td/nope")"
wrow "W12 signed receipt whose tests failed -> 0" 0 "$(arg "$td/b-v1")"
wrow "W13 signed receipt with no gate_results -> 0" 0 "$(arg "$td/b-v2")"
# W14: the forged-receipt repro (wsweep, #5798). The PR swaps in its own key and signs its
# own receipt. The base's key is the only trust root, so this must print 0.
mkdir -p "$repo/.flow009"; cp "$td/other.pub" "$repo/.flow009/batcher.pub"
printf '%s\n' "$good" >"$td/f.json"
body "$td/b-forged" "$(sign "$td/other" "$td/f.json")"
wrow "W14 FORGED: receipt signed by a key the PR planted in its own tree -> 0" 0 "$(arg "$td/b-forged")"
# W15/W10 mutants: a wrapper that trusts the PR tree's key, or prints 1 regardless, is caught.
# mutate FILE FROM TO OUT -> OUT is FILE with the literal FROM replaced by TO; non-zero if FROM is absent
mutate() {
  local c; c=$(cat "$1"; printf x); c=${c%x}
  case "$c" in *"$2"*) ;; *) return 1 ;; esac
  printf '%s' "${c//"$2"/$3}" >"$4"
}
if mutate "$W" '"$j/.flow009/batcher.pub"' '.flow009/batcher.pub' "$td/mut-k.sh"; then
  wrow "W15 MUTANT: wrapper reading the key from the PR tree is caught by the forged row" 1 "$(arg "$td/b-forged" "$BASE" "$td/mut-k.sh")"
else fail=$((fail + 1)); echo "FAIL W15: mutant did not apply"; fi
if mutate "$W" 'if [ "$first" = "RECEIPT_OK=1" ]; then' 'if true; then' "$td/mut-w.sh"; then
  wrow "W10 MUTANT: a wrapper that always prints 1 is caught by the no-receipt row" 1 "$(arg "$td/b-none" "$BASE" "$td/mut-w.sh")"
else fail=$((fail + 1)); echo "FAIL W10: mutant did not apply to $W"; fi
# W16-W20: the judge is its own job and no PR-tree code can run before it reads. ci.yml's
# `receipt-judge` job checks out the BASE (ref: github.base_ref), fetches the PR merge commit as
# a data worktree, and runs no script from anywhere but the base checkout; the shard job needs it
# and takes its verdict as RECEIPT_OK; the tier step in ci/sections.yml only reads that env. Each
# defect below must turn judge_ok RED: a PR checkout as code, a PR-tree script step, a step that
# plants PATH or the env, a shard job that does not wait for the judge, a tier step that runs code.
CI="$HERE/../.github/workflows/ci.yml"
SEC="$HERE/../ci/sections.yml"
tl() { grep -v -E '^[[:space:]]*#' "$1" | grep -E '^[^[:space:]-]'; }
jl() { grep -v -E '^[[:space:]]*#' "$1" | awk '/^jobs:/{j=1;next} j&&/^[^[:space:]]/{j=0} j&&/^  [^ ]/{print}'; }
h() { { printf '%s\n' "$(cat)"; } | sha256sum | cut -d' ' -f1; }
judge_ok() { # CI_FILE SECTIONS_FILE -> 0 when the wiring is trustworthy
  local job code shard tier step ciall secall n
  job=$(awk '/^  receipt-judge:$/{on=1;print;next} on&&/^  [a-z]/{exit} on{print}' "$1")
  [ -n "$job" ] || return 1
  code=$(printf '%s\n' "$job" | grep -v -E '^[[:space:]]*#')
  case "$code" in *'ref: ${{ github.base_ref }}'*) ;; *) return 1 ;; esac
  [ "$(printf '%s\n' "$code" | grep -c 'uses:')" -eq 1 ] || return 1
  printf '%s\n' "$code" | grep -q 'uses: actions/checkout@' || return 1
  printf '%s\n' "$code" | grep -q -E 'GITHUB_PATH|GITHUB_ENV|head_ref|pull_request\.head|github\.sha *$' && return 1
  [ "$(printf '%s\n' "$code" | grep -c '^      - ')" -eq 2 ] || return 1
  printf '%s\n' "$code" | grep -E 'scripts/' | grep -v -E 'git show HEAD:scripts/ci/receipt_ok_arg\.sh' | grep -q . && return 1
  case "$code" in *'runs-on: [self-hosted, Linux, X64, receipt-judge]'*) ;; *) return 1 ;; esac
  # a judge with no runner must be skipped, never queued: the job is gated on the readiness variable
  printf '%s\n' "$job" | grep -q -F "vars.RECEIPT_JUDGE_RUNNER_READY == 'true'" || return 1
  case "$code" in *'env -i HOME='*'/usr/bin/bash "$RUNNER_TEMP/rok.sh"'*) ;; *) return 1 ;; esac
  case "$code" in *'worktree add -q --no-checkout'*) ;; *) return 1 ;; esac
  case "$code" in *'core.hooksPath=/dev/null'*) ;; *) return 1 ;; esac
  case "$code" in *'unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0'*) ;; *) return 1 ;; esac
  shard=$(awk '/^  workspace-test-shard:$/{on=1;print;next} on&&/^  [a-z]/{exit} on{print}' "$1" | grep -v -E '^[[:space:]]*#')
  case "$shard" in *'needs: [receipt-judge, mg-reuse-wt]'*) ;; *) return 1 ;; esac
  # whole-line pins: a substring pin lets "; ok=1", a flipped "&&" or another cd target ride along
  hasl() { local t; t=$(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<< "$1"); grep -qxF -- "$2" <<< "$t"; }
  hasl "$shard" "RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '0' }}" || return 1
  hasl "$code" '[ "$ok" = 1 ] || ok=0' || return 1
  hasl "$code" '[ "$(git rev-parse FETCH_HEAD)" = "${GITHUB_SHA}" ] &&' || return 1
  hasl "$code" 'ok="$(cd "$RUNNER_TEMP/prtree" && env -i HOME="$HOME" PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/bash "$RUNNER_TEMP/rok.sh" "$RUNNER_TEMP/pr-body.md" "$GITHUB_WORKSPACE" "${GITHUB_SHA}")" || ok=0' || return 1
  hasl "$code" "printf 'ok=%s\\n' \"\$ok\" >> \"\$GITHUB_OUTPUT\"" || return 1
  hasl "$code" 'ok=0' || return 1
  # the whole comment-stripped judge job is hash-pinned: a line-by-line pin chase never ends (E1-E6). Edit the job -> update this sha256 (gate content, quorum).
  [ "$(printf '%s\n' "$code" | sha256sum | cut -d' ' -f1)" = "ee14f458b0d4a68335ae1d1560777b42c97a11be4ebfd9af3320cc3b1d161ca4" ] || return 1
  tier=$(awk '/id: tier$/{on=1} on{print} /--receipt-ok/{if(on)exit}' "$2" | grep -v -E '^[[:space:]]*#')
  case "$tier" in *'"${RECEIPT_OK:-0}" = 1'*) ;; *) return 1 ;; esac
  printf '%s\n' "$tier" | grep -E 'scripts/|git (fetch|clone)' | grep -q . && return 1
  # the consumer chain (tier step: RECEIPT_OK -> rok -> --receipt-ok -> ci_test_tier.sh) is hash-pinned whole; edit it -> update this sha256 (gate content, quorum)
  step=$(awk '/- name: Decide the test tier/{on=1;print;next} on&&/^      - name:/{exit} on{print}' "$2" | grep -v -E '^[[:space:]]*#')
  [ "$(printf '%s\n' "$step" | sha256sum | cut -d' ' -f1)" = "0335490bab33adde6bd276b21c44041d1baa29418e28d568553b3cfc8c76f35b" ] || return 1
  # nothing outside the pinned lines may set the verdict, the shell or the environment
  ciall=$(grep -v -E '^[[:space:]]*#' "$1"); secall=$(grep -v -E '^[[:space:]]*#' "$2")
  n=$(grep -c -E 'RECEIPT_OK|receipt-ok|BASH_ENV|GITHUB_ENV|^defaults:|^env:' <<< "$ciall" || true); [ "$n" = 1 ] || return 1
  n=$(grep -c -E 'RECEIPT_OK|receipt-ok' <<< "$secall" || true); [ "$n" = 2 ] || return 1
  n=$(grep -c -E 'BASH_ENV|^defaults:|^env:' <<< "$secall" || true); [ "$n" = 0 ] || return 1
  n=$(grep -c 'GITHUB_ENV' <<< "$secall" || true); [ "$n" = 4 ] || return 1
  # hash pins (E7, escaped keys and every other spelling): the shard job in ci.yml, its block in sections.yml and the three scripts the verdict flows through.
  # Any byte change is RED unless the pin moves in the same diff (gate content, quorum). PIN_ROOT lets the case table pin a mutated copy.
  local pr=${PIN_ROOT:-$HERE/..} blk
  [ "$(printf '%s\n' "$shard" | sha256sum | cut -d' ' -f1)" = "93eb48c4c6c0775012afd1c7d60e752c72cb4107a7782cac2da9d096cef2b75a" ] || return 1
  blk=$(awk '/^jobs:/{j=1} j&&/^  workspace-test-shard:$/{on=1;print;next} on&&/^  [a-z]/{exit} on{print}' "$2" | grep -v -E '^[[:space:]]*#')
  [ "$(printf '%s\n' "$blk" | sha256sum | cut -d' ' -f1)" = "c8047f526588a7e7af8ab15a17118a5a69e8cbf4a4edae5181473f251f5a5c55" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci/fat_driver.py" | cut -d' ' -f1)" = "81662d808e4672b41b481f44361c10aaae308661f492b1aaaff47f87ffb27cfc" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci_test_tier.sh" | cut -d' ' -f1)" = "1680bab97243fb7b5ce6907daf3c8a00db4968a2e892095260152a09cc127866" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci/receipt_ok_arg.sh" | cut -d' ' -f1)" = "1795d4df36fa1c3f223dd054d93392aff7584bb5063bed1d7a11bd3f18144820" ] || return 1
  # scripts that run before the tier step in the shard (they could append RECEIPT_OK to $GITHUB_ENV): byte-pinned too
  [ "$(sha256sum < "$pr/scripts/ci_reclaim_target_dirs.sh" | cut -d' ' -f1)" = "454aec9194c8d90aa0587e469fca9c8d801fa211c0a4aacdf9c2dc5a820c5449" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci/fixture_quiet.sh" | cut -d' ' -f1)" = "0571ef87e0859c9dc9299c6cb57b1d3ed11991fb5e3c890cf56c76e385df730c" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci/pr_touched_paths.sh" | cut -d' ' -f1)" = "eed39751351685073888dd9e1ae9e993a30cdc933f97eb8e92aa8f01b36e42a4" ] || return 1
  # helpers the tier step (and ci_test_tier.sh) run: byte-pinned, because a same-uid helper can reach the real GITHUB_OUTPUT via /proc/$PPID/environ (#5850)
  [ "$(sha256sum < "$pr/scripts/ci/fetch_p6.sh" | cut -d' ' -f1)" = "5addde69a85a4d6996f89a1af85a04abaebed3a4c27b447e254171d97205580b" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci_mg_workspace_result.sh" | cut -d' ' -f1)" = "d09d4f6d515630940a727b1152e391177873296d2ddee0d7c4d3728bdffa2886" ] || return 1
  [ "$(sha256sum < "$pr/scripts/check_ci_push_reuse.sh" | cut -d' ' -f1)" = "0b09758f2c9773f992849cd391e9f3a1325630364d5afa58844c7d9c6848e4c8" ] || return 1
  [ "$(sha256sum < "$pr/scripts/check_tree_reader_tests.sh" | cut -d' ' -f1)" = "4f4f0e892b1b4c65e1dd9dcc46f0edc0e5deac2f1b5dc7883fc124f6b4c6f0aa" ] || return 1
  [ "$(sha256sum < "$pr/scripts/gate_touched_crates.sh" | cut -d' ' -f1)" = "00f73732cbe0c39c9ff0867d1cb416ef0e7e0d00dc5b01f300cee645fe444bfe" ] || return 1
  [ "$(sha256sum < "$pr/scripts/lib/test_tier.py" | cut -d' ' -f1)" = "63e4b091c7a6c56b45c8da5ceb241ba2a6f991cef55bb3802ffa6458b786dd52" ] || return 1
  [ "$(sha256sum < "$pr/scripts/ci_mg_reuse.sh" | cut -d' ' -f1)" = "6f3d6c216884038ed831ee4a889f074976fa1dd622edd0c77d7ed64aa9600ba8" ] || return 1
  # structure pins: the top-level keys and the job keys of both files, in any spelling (quoted, ? keys, duplicates). A new top-level key or job, or a second shard block, moves the hash.
  [ "$(tl "$1" | h)" = "7b45344e526135d0ab4f5c04ca82ba2d1112f6286b1ba2024a18017536446ce5" ] || return 1
  [ "$(jl "$1" | h)" = "4b456c7f21889b2de69f7226faa2ac94962aca13912721cc47be159531bf4ede" ] || return 1
  [ "$(tl "$2" | h)" = "e90e5a61fd66429fec4eab76974c56bcdd3f253610a6fe9c1f63b7fd8959b18f" ] || return 1
  [ "$(jl "$2" | h)" = "c988f167ca3654fa20a9df52cea6196fb7032606c2a7a32425ad2740f14c5925" ] || return 1
  return 0
}
if judge_ok "$CI" "$SEC"; then wrow "W16 receipt judge is its own job: base checkout only, no PR-tree code, shard waits for it" ok ok
else wrow "W16 receipt judge is its own job: base checkout only, no PR-tree code, shard waits for it" ok bad; fi
CHK='      - uses: actions/checkout@v7
        with:
          ref: ${{ github.base_ref }}'
wmut() { # LABEL FILE(ci|sec) FROM TO
  local src=$CI; [ "$2" = sec ] && src=$SEC
  if mutate "$src" "$3" "$4" "$td/mut.yml"; then
    if [ "$2" = sec ]; then judge_ok "$CI" "$td/mut.yml" && r=missed || r=caught
    else judge_ok "$td/mut.yml" "$SEC" && r=missed || r=caught; fi
    wrow "$1" caught "$r"
  else fail=$((fail + 1)); echo "FAIL ${1%% *}: mutant did not apply"; fi
}
wmut "W17 MUTANT: the judge job checking out the PR merge as code is caught" ci 'ref: ${{ github.base_ref }}' 'ref: ${{ github.sha }}'
wmut "W18 MUTANT: a PR-tree script step before the judge is caught" ci "$CHK" '      - run: bash scripts/ci/pre.sh
'"$CHK"
wmut "W19 MUTANT: an earlier step planting PATH is caught" ci "$CHK" '      - run: echo "$PWD/bin" >> "$GITHUB_PATH"
'"$CHK"
wmut "W27 MUTANT: a judge whose PR worktree is a full checkout is caught" ci 'worktree add -q --no-checkout' 'worktree add -q'
wmut "W28 MUTANT: a judge that leaves the git token env set is caught" ci 'unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 GH_TOKEN' 'true'
wmut "W33 MUTANT: a judge not gated on the runner-readiness variable (it would queue forever with no runner) is caught" ci " && vars.RECEIPT_JUDGE_RUNNER_READY == 'true'" ''
wmut "W34 MUTANT: the judge back on a GitHub-hosted image is caught" ci 'runs-on: [self-hosted, Linux, X64, receipt-judge]' 'runs-on: ubuntu-latest'
wmut "W20 MUTANT: a shard job that does not wait for the judge is caught" ci 'needs: [receipt-judge, mg-reuse-wt]' 'needs: []'
wmut "W20c MUTANT (MA): the shard RECEIPT_OK fallback flipped to 1 is caught" ci "outputs.ok || '0' }}" "outputs.ok || '1' }}"
wmut "W20d MUTANT (MB): the verifier trusting \$GITHUB_WORKSPACE no longer, but the PR tree, is caught" ci '"$RUNNER_TEMP/pr-body.md" "$GITHUB_WORKSPACE"' '"$RUNNER_TEMP/pr-body.md" "$RUNNER_TEMP/prtree"'
wmut "W20e MUTANT (MC): the ok=0 guard turned into ok=1 is caught" ci '[ "$ok" = 1 ] || ok=0' '[ "$ok" = 1 ] || ok=1'
wmut "W20f MUTANT (MD): the FETCH_HEAD == GITHUB_SHA compare dropped is caught" ci '[ "$(git rev-parse FETCH_HEAD)" = "${GITHUB_SHA}" ] &&' 'true &&'
wmut "W20g MUTANT (E1): a comment-only RECEIPT_OK in the shard does not satisfy the pin" ci "      RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '0' }}" "      RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '1' }}
      # RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '0' }}"
wmut "W20h MUTANT (E2): the ok=0 guard kept but followed by ; ok=1 is caught" ci '[ "$ok" = 1 ] || ok=0' '[ "$ok" = 1 ] || ok=0; ok=1'
wmut "W20i MUTANT (E3): the FETCH_HEAD compare joined with || instead of && is caught" ci '[ "$(git rev-parse FETCH_HEAD)" = "${GITHUB_SHA}" ] &&' '[ "$(git rev-parse FETCH_HEAD)" = "${GITHUB_SHA}" ] ||'
wmut "W20j MUTANT (E4): the verifier run from the trusted workspace, not the PR tree, is caught" ci 'ok="$(cd "$RUNNER_TEMP/prtree" &&' 'ok="$(cd "$GITHUB_WORKSPACE" &&'
wmut "W20k MUTANT (E5): the judge's output line hard-coded to ok=1 is caught" ci "printf 'ok=%s\\n' \"\$ok\" >> \"\$GITHUB_OUTPUT\"" 'printf "ok=1\n" >> "$GITHUB_OUTPUT"'
wmut "W20l MUTANT (E6): the judge's initial ok=0 set to ok=1 is caught" ci '          ok=0' '          ok=1'
wmut "W40a MUTANT: tier step env RECEIPT_OK: 1 is caught" sec '          PR_BODY: ${{ github.event.pull_request.body }}
        run: |
          set -euo pipefail' '          PR_BODY: ${{ github.event.pull_request.body }}
          RECEIPT_OK: "1"
        run: |
          set -euo pipefail'
wmut "W40b MUTANT: the tier gate && turned into || is caught" sec '[ "${GITHUB_EVENT_NAME}" = pull_request ] && [ "${RECEIPT_OK:-0}" = 1 ]' '[ "${GITHUB_EVENT_NAME}" = pull_request ] || [ "${RECEIPT_OK:-0}" = 1 ]'
wmut "W40c MUTANT: hard-coded --receipt-ok 1 is caught" sec 'args+=(--receipt-ok "${rok:-0}")' 'args+=(--receipt-ok 1)'
wmut "W40d MUTANT: workflow-level defaults: run: shell is caught" ci $'\njobs:\n' $'\ndefaults:\n  run:\n    shell: bash -c "export RECEIPT_OK=1; . {0}"\njobs:\n'
wmut "W40e MUTANT: workflow-level env BASH_ENV is caught" ci $'\njobs:\n' $'\nenv:\n  BASH_ENV: /tmp/x\njobs:\n'
wmut "W40f MUTANT: a GITHUB_ENV write of RECEIPT_OK in a shard step is caught" ci '      - name: Judge the receipt in the PR body' '      - run: echo RECEIPT_OK=1 >> "$GITHUB_ENV"
      - name: Judge the receipt in the PR body'
wmut "W40g MUTANT: a second RECEIPT_OK override in a shard step env is caught" ci "      RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '0' }}" "      RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '0' }}
      RECEIPT_OK_X: 1
      RECEIPT_OK: 1"
wmut "W20b MUTANT: a tier step that runs PR-tree code is caught" sec 'rok=0
          if [ "${GITHUB_EVENT_NAME}" = pull_request ]' 'rok=0
          bash scripts/ci/pre.sh
          if [ "${GITHUB_EVENT_NAME}" = pull_request ]'
# W21/W22: a refs/replace entry in the PR repo must not let a receipt for another tree pass. The
# receipt is signed for the empty tree; the replace ref makes HEAD read as a commit with that tree.
git -C "$repo" -c user.email=t@example.invalid -c user.name=t checkout -q -- . 2>/dev/null
rep=$(git -C "$repo" commit-tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904 -m fake)
git -C "$repo" update-ref "refs/replace/$head" "$rep"
# the faked HEAD has an empty tree, so its toolchain id has no toml part
tcE=$({ (cd "$BASE" && rustc -vV); printf 'image=none\n'; } | sha256sum | cut -d' ' -f1)
receipt 4b825dc642cb6eb9a060e54bf8d69288fbee4904 "$tcE" >"$td/r.json"
body "$td/b-repl" "$(sign "$td/key" "$td/r.json")"
wrow "W21 refs/replace on the PR repo cannot make another tree pass -> 0" 0 "$(arg "$td/b-repl")"
if mutate "$W" 'export GIT_NO_REPLACE_OBJECTS=1' ':' "$td/mut-r.sh" && sed -i 's/ GIT_NO_REPLACE_OBJECTS=1 RECEIPT/ RECEIPT/' "$td/mut-r.sh"; then
  wrow "W22 MUTANT: a wrapper that honours replace refs is caught by W21's row" 1 "$(arg "$td/b-repl" "$BASE" "$td/mut-r.sh")"
else fail=$((fail + 1)); echo "FAIL W22: mutant did not apply"; fi
git -C "$repo" update-ref -d "refs/replace/$head"
rm -rf "${repo:?}/.flow009"
# W23-W26: the rustc/credential hardening (Opus, #5798). A PR toolchain file with an absolute
# `path =` must not reach rustc; the verifier gets no GH_TOKEN/GITHUB_OUTPUT; rustc runs in the
# base's dir. repo2's receipt is signed over a toolchain id computed the honest way, so only the
# path refusal stands between it and ok=1.
rm -rf "${td:?}/repo2"; cp -r "$repo" "$td/repo2"
printf '[toolchain]\npath = "/nonexistent/fake-toolchain"\n' >"$td/repo2/rust-toolchain.toml"
git -C "$td/repo2" -c user.email=t@example.invalid -c user.name=t commit -q -am "path override"
tree2=$(git -C "$td/repo2" rev-parse 'HEAD^{tree}'); head2=$(git -C "$td/repo2" rev-parse HEAD)
tc2=$({ cat "$td/repo2/rust-toolchain.toml"; (cd "$BASE" && rustc -vV); printf 'image=none\n'; } | sha256sum | cut -d' ' -f1)
receipt "$tree2" "$tc2" >"$td/p.json"; body "$td/b-path" "$(sign "$td/key" "$td/p.json")"
arg2() { (cd "$td/repo2" && bash "${3:-$W}" "$1" "${2:-$BASE}" "$head2" 2>/dev/null | head -n1); }
wrow "W23 a PR toolchain file with path= is refused -> 0" 0 "$(arg2 "$td/b-path")"
if mutate "$TID" 'if printf' 'if false && printf' "$td/mut-tid.sh"; then
  mkbase trust-mut "$td/pub"; cp "$td/mut-tid.sh" "$td/trust-mut/scripts/receipt_toolchain_id.sh"
  git -C "$td/trust-mut" -c user.email=t@example.invalid -c user.name=t commit -q -am mut
  wrow "W24 MUTANT: a toolchain id that accepts path= is caught by W23's row" 1 "$(arg2 "$td/b-path" "$td/trust-mut")"
else fail=$((fail + 1)); echo "FAIL W24: mutant did not apply"; fi
# probe judge: records what the verifier can see (token/output vars, and the rustc dir's listing)
mkbase trust-probe "$td/pub"
cat >"$td/trust-probe/scripts/ci_receipt_gate.sh" <<PROBE
#!/usr/bin/env bash
printf '%s|%s|%s\\n' "\${GH_TOKEN:-}\${GITHUB_OUTPUT:-}" "\${RECEIPT_RUSTC_CWD:-}" "\$(ls "\${RECEIPT_RUSTC_CWD:-/nonexistent}" 2>/dev/null | tr -d '\\n')" >"$td/probe.out"
printf 'RECEIPT_OK=0\\n'
PROBE
git -C "$td/trust-probe" -c user.email=t@example.invalid -c user.name=t commit -q -am probe
  # bashrs disable-next-line=SEC010
probe() { rm -f "${td:?}/probe.out"; (cd "$repo" && GH_TOKEN=secret GITHUB_OUTPUT=/tmp/x PROBEFILE="$td/probe.out" bash "${1:-$W}" "$td/b-good" "$td/trust-probe" "$head" >/dev/null 2>&1); sed -E 's#\|[^|]*/rc\|#|RC|#' "$td/probe.out" 2>/dev/null; }
wrow "W25 the verifier sees no token or output file, and runs rustc in the base's dir" "|RC|rust-toolchain.toml" "$(probe)"
if mutate "$W" 'first=$(env -i ' 'first=$(env ' "$td/mut-e.sh"; then
  wrow "W26 MUTANT: a wrapper without env -i leaves GH_TOKEN in the verifier's env is caught by W25" "secret/tmp/x|RC|rust-toolchain.toml" "$(probe "$td/mut-e.sh")"
else fail=$((fail + 1)); echo "FAIL W26: mutant did not apply"; fi
# W29-W32: forgery by the environment (Opus, #5798). An always-0 ssh-keygen early in PATH, or a
# BASH_ENV file defining ssh-keygen(), would turn a tampered receipt (b-s1: valid in every other
# field) into ok=1. The real wrapper + gate must print 0; a mutant with the defences removed prints 1.
mkdir -p "$td/evil"; printf '#!/bin/sh\nexit 0\n' >"$td/evil/ssh-keygen"; chmod +x "$td/evil/ssh-keygen"
printf 'ssh-keygen() { return 0; }\nexport -f ssh-keygen\n' >"$td/evil.env"
forge() { # WRAPPER BASEDIR -> wrapper output under both forgeries (PATH first, then BASH_ENV)
  local a b
  # bashrs disable-next-line=SEC010
  a=$(cd "$repo" && PATH="$td/evil:$PATH" bash "$1" "$td/b-s1" "$2" "$head" 2>/dev/null | head -n1)
  b=$(cd "$repo" && BASH_ENV="$td/evil.env" bash "$1" "$td/b-s1" "$2" "$head" 2>/dev/null | head -n1)
  printf '%s%s' "$a" "$b"
}
wrow "W29 a planted ssh-keygen in PATH or a BASH_ENV function cannot forge ok -> 00" 00 "$(forge "$W" "$BASE")"
wrow "W30 sanity: the tampered receipt really is bad without the forgery -> 0" 0 "$(arg "$td/b-s1")"
if mutate "$W" 'first=$(env -i ' 'first=$(env ' "$td/mut-f.sh" && sed -i 's# PATH=/usr/local/bin:/usr/bin:/bin GIT_NO# GIT_NO#' "$td/mut-f.sh"; then
  mkbase trust-forge "$td/pub"
  sed -i -e 's#^PATH=/usr/local/bin.*#:#' -e 's#^unset BASH_ENV ENV#:#' -e 's#^SSHKEYGEN=.*#SSHKEYGEN=ssh-keygen#' -e 's#\[ -x "\$SSHKEYGEN" \]#command -v "$SSHKEYGEN" >/dev/null#' "$td/trust-forge/scripts/ci_receipt_gate.sh"
  git -C "$td/trust-forge" -c user.email=t@example.invalid -c user.name=t commit -q -am forge
  wrow "W31 MUTANT: no env -i, no fixed PATH, bare ssh-keygen -> the PATH forgery gets through" 1 "$(forge "$td/mut-f.sh" "$td/trust-forge" | cut -c1)"
  wrow "W32 MUTANT: same, the BASH_ENV forgery gets through" 1 "$(forge "$td/mut-f.sh" "$td/trust-forge" | cut -c2)"
else fail=$((fail + 1)); echo "FAIL W31/W32: mutant did not apply"; fi
tier_of() { (cd "$HERE/.." && bash "$T" --event pull_request --diff-from "$td/touched.txt" --receipt-ok "$1" 2>/dev/null | head -n1); }
t0=$(tier_of 0); case "$t0" in tier=quick|tier=full) t0=notreuse ;; esac
wrow "W8 --receipt-ok 0 on a core diff runs quick or full, never reuse" notreuse "$t0"
wrow "W9 --receipt-ok 1 on the same diff is reuse" tier=reuse "$(tier_of 1)"


# W23: BEHAVIOURAL. Run the judge's real `run:` script (extracted from ci.yml) against a local origin, so a
# fail-closed property is measured, not pattern-matched: a failed fetch, a failed verifier and a garbage verifier
# answer must each leave ok=0; only a verifier that exits 0 printing a literal 1 may give ok=1.
judge_run() { # CI_FILE SCENARIO(good|nofetch|vfail|vgarbage|vfail1) -> prints the ok= value the judge wrote
  local ci=$1 sc=$2 d="$td/jo-$2-$RANDOM" pr=7 out
  mkdir -p "$d/w/scripts/ci" "$d/rt" && : > "$d/out"
  awk '/id: judge$/{s=1} s&&/^        run: \|$/{r=1;next} r&&/^          /{sub(/^          /,"");print;next} r&&!/^[[:space:]]*$/{exit} r{print}' "$ci" > "$d/judge.sh"
  [ -s "$d/judge.sh" ] || { echo "noscript"; return; }
  case "$sc" in
    good) printf 'echo 1\n' ;; vfail) printf 'exit 1\n' ;; vfail1) printf 'echo 1\nexit 1\n' ;; vgarbage) printf 'echo 2\n' ;; *) printf 'echo 1\n' ;;
  esac > "$d/w/scripts/ci/receipt_ok_arg.sh"
  (
    unset GIT_DIR GIT_INDEX_FILE GIT_WORK_TREE GIT_COMMON_DIR
    export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
    git init -q --bare "$d/origin.git" && cd "$d/w" && git init -q -b main . &&
      git add -A && git commit -q -m base && git remote add origin "file://$d/origin.git" &&
      git commit -q --allow-empty -m merge && git push -q origin "HEAD:refs/pull/$pr/merge" && git reset -q --hard HEAD~1
  ) > /dev/null 2>&1 || { echo "setup"; return; }
  local sha; sha=$(git -C "$d/origin.git" rev-parse "refs/pull/$pr/merge")
  local prn=$pr; [ "$sc" = nofetch ] && prn=99
  (cd "$d/w" && env -u GIT_DIR -u GIT_INDEX_FILE -u GIT_WORK_TREE GH_TOKEN=x PR_BODY= PR_NUMBER=$prn GITHUB_SHA="$sha" \
     GITHUB_WORKSPACE="$d/w" RUNNER_TEMP="$d/rt" GITHUB_OUTPUT="$d/out" bash -e "$d/judge.sh") > "$d/log" 2>&1
  out=$(sed -n 's/^ok=//p' "$d/out" | tail -n1); echo "${out:-none}"
}
wrow "W23a judge run: verifier prints 1 and exits 0 -> ok=1 (the mechanism engages)" 1 "$(judge_run "$CI" good)"
wrow "W23b judge run: forced fetch failure -> ok=0" 0 "$(judge_run "$CI" nofetch)"
wrow "W23c judge run: forced verifier failure (exit 1) -> ok=0" 0 "$(judge_run "$CI" vfail)"
wrow "W23d judge run: verifier prints 1 then exits 1 -> ok=0" 0 "$(judge_run "$CI" vfail1)"
wrow "W23e judge run: verifier prints 2 -> ok=0" 0 "$(judge_run "$CI" vgarbage)"
if mutate "$CI" '          ok=0' '          ok=1' "$td/b6.yml"; then wrow "W23f MUTANT (E6): initial ok=1 leaves ok=1 on a failed fetch, behaviourally" 1 "$(judge_run "$td/b6.yml" nofetch)"; else fail=$((fail + 1)); echo "FAIL W21f: mutant did not apply"; fi

# W41: hash pins on the shard job (ci.yml + sections.yml block) and the three scripts the verdict flows through
rl="      RECEIPT_OK: \${{ needs.receipt-judge.outputs.ok || '0' }}"
wmut "W41a MUTANT (M1): an escaped RECEIPT\x5FOK env key in the ci.yml shard env is caught" ci "$rl" "$rl"$'\n''      "RECEIPT\x5FOK": "1"'
sc='      SCCACHE_HOST_DIR: /home/noah/data/sccache'
wmut "W41b MUTANT (M2): an escaped RECEIPT\x5FOK env key in the sections.yml shard env is caught" sec "$sc" "$sc"$'\n''      "RECEIPT\x5FOK": "1"'
# a byte change in any verdict script is RED unless its pin moves (PIN_ROOT = a copy of the tree with one file mutated)
for rel in scripts/ci/fat_driver.py scripts/ci_test_tier.sh scripts/ci/receipt_ok_arg.sh scripts/ci_reclaim_target_dirs.sh scripts/ci/fixture_quiet.sh scripts/ci/pr_touched_paths.sh scripts/ci/fetch_p6.sh scripts/ci_mg_workspace_result.sh scripts/check_ci_push_reuse.sh scripts/check_tree_reader_tests.sh scripts/gate_touched_crates.sh scripts/lib/test_tier.py scripts/ci_mg_reuse.sh; do
  pr="$td/pin-$(basename "$rel")"; mkdir -p "$pr/scripts/ci" "$pr/scripts/lib"
  for g in scripts/ci/fat_driver.py scripts/ci_test_tier.sh scripts/ci/receipt_ok_arg.sh scripts/ci_reclaim_target_dirs.sh scripts/ci/fixture_quiet.sh scripts/ci/pr_touched_paths.sh scripts/ci/fetch_p6.sh scripts/ci_mg_workspace_result.sh scripts/check_ci_push_reuse.sh scripts/check_tree_reader_tests.sh scripts/gate_touched_crates.sh scripts/lib/test_tier.py scripts/ci_mg_reuse.sh; do cp "$HERE/../$g" "$pr/$g"; done
  printf '\n# touched\n' >> "$pr/$rel"
  if PIN_ROOT="$pr" judge_ok "$CI" "$SEC"; then r=missed; else r=caught; fi
  wrow "W41c MUTANT: a byte appended to $rel is caught by the hash pin" caught "$r"
done

# W42: spelling-proof structure rows (quoted / ? top-level keys, a duplicate shard block)
wmut "W42a MUTANT: a quoted top-level \"env\" key with BASH\x5FENV in ci.yml is caught" ci $'\njobs:\n' $'\n"env":\n  "BASH\\x5FENV": /tmp/x\njobs:\n'
wmut "W42b MUTANT: a ? top-level defaults key in ci.yml is caught" ci $'\njobs:\n' $'\n? defaults\n: {run: {shell: bash}}\njobs:\n'
wmut "W42c MUTANT: a duplicate workspace-test-shard job after the first in ci.yml is caught" ci $'\n  workspace-test-shard:\n' $'\n  workspace-test-shard:\n    runs-on: x\n  workspace-test-shard:\n'
# W42d/e: a shard script that writes the runner env/output files is a pinned-byte change, so it is RED
for rel in scripts/ci/fetch_p6.sh scripts/ci_mg_workspace_result.sh scripts/check_ci_push_reuse.sh scripts/check_tree_reader_tests.sh scripts/gate_touched_crates.sh scripts/lib/test_tier.py scripts/ci_mg_reuse.sh scripts/ci_test_tier.sh scripts/ci_reclaim_target_dirs.sh scripts/ci/fixture_quiet.sh scripts/ci/pr_touched_paths.sh scripts/ci/fetch_p6.sh scripts/ci_mg_workspace_result.sh scripts/check_ci_push_reuse.sh scripts/check_tree_reader_tests.sh scripts/gate_touched_crates.sh scripts/lib/test_tier.py scripts/ci_mg_reuse.sh; do for w in 'echo RECEIPT_OK=1 >> "$GITHUB_ENV"' 'echo ok=1 >> "$GITHUB_OUTPUT"'; do
  pr="$td/pin-w"; rm -rf -- "${pr:?}"; mkdir -p "$pr/scripts/ci" "$pr/scripts/lib"
  for g in scripts/ci/fat_driver.py scripts/ci_test_tier.sh scripts/ci/receipt_ok_arg.sh scripts/ci_reclaim_target_dirs.sh scripts/ci/fixture_quiet.sh scripts/ci/pr_touched_paths.sh scripts/ci/fetch_p6.sh scripts/ci_mg_workspace_result.sh scripts/check_ci_push_reuse.sh scripts/check_tree_reader_tests.sh scripts/gate_touched_crates.sh scripts/lib/test_tier.py scripts/ci_mg_reuse.sh; do cp "$HERE/../$g" "$pr/$g"; done
  printf "\n%s\n" "$w" >> "$pr/$rel"
  if PIN_ROOT="$pr" judge_ok "$CI" "$SEC"; then r=missed; else r=caught; fi
  wrow "W42d MUTANT: $rel writing ${w##* } is caught" caught "$r"
done; done

# W43: parse_file_command (the driver's GITHUB_OUTPUT/GITHUB_ENV reader) fails closed (#5850); the real function is run, no PyYAML needed
pfc_run() { # $1 = driver source, $2 = file body; prints reject|accept
  local src="$td/pfc.py" o="$td/pfc.out"
  sed -n '/^def _fc_put/,/^    return out$/p' "$1" > "$src"; printf '%b' "$2" > "$o"
  if python3 -I -c 'import re,sys
from pathlib import Path
exec(open(sys.argv[1]).read())
parse_file_command(Path(sys.argv[2]))' "$src" "$o" > /dev/null 2>&1 < /dev/null; then echo accept; else echo reject; fi
}
FD="$HERE/ci/fat_driver.py"
wrow "W43a an unterminated heredoc planted ahead of the real tier lines is rejected" reject "$(pfc_run "$FD" 'tier=reuse\ncite=receipt\nignore<<EOF\ntier=quick\n')"
wrow "W43b a duplicate tier= key is rejected" reject "$(pfc_run "$FD" 'tier=reuse\ntier=quick\n')"
wrow "W43c a duplicate heredoc key is rejected" reject "$(pfc_run "$FD" 'k<<E\na\nE\nk=b\n')"
wrow "W43d positive control: distinct keys and a terminated heredoc parse" accept "$(pfc_run "$FD" 'tier=quick\nk<<EOF\na\nEOF\n')"
sed -e 's/raise SystemExit(.*)/pass/' "$FD" > "$td/fd-mut.py"
wrow "W43e MUTANT: the fail-closed raises removed accepts the planted heredoc (the test is sensitive)" accept "$(pfc_run "$td/fd-mut.py" 'tier=reuse\ncite=receipt\nignore<<EOF\ntier=quick\n')"

# W44: the tier step isolates its helpers (scratch GITHUB_OUTPUT/GITHUB_ENV) and writes the real file only at the end
wmut "W44a MUTANT: the scratch GITHUB_OUTPUT/GITHUB_ENV removed from the tier step is caught" sec 'GITHUB_OUTPUT="$(mktemp)"; GITHUB_ENV="$(mktemp)"; export GITHUB_OUTPUT GITHUB_ENV' ':'
wmut "W44b MUTANT: the tier step writing the real GITHUB_OUTPUT is caught" sec '"$RUNNER_TEMP/tier.txt" >> "$real_out"' '"$RUNNER_TEMP/tier.txt" >> "$GITHUB_OUTPUT"'

# W45: every file the tier step runs, marked pinned (hash) or isolated (runs under the scratch env). A new one is RED until it is classified here.
tier_exec() { awk '/- name: Decide the test tier/{on=1;print;next} on&&/^      - name:/{exit} on{print}' "$1" | grep -v -E '^[[:space:]]*#' | grep -oE 'scripts/[A-Za-z0-9_./-]+' | LC_ALL=C sort -u | tr '\n' ' '; }
# pinned: ci_test_tier.sh.  isolated (scratch env): fetch_p6.sh, pr_touched_paths.sh, ci_mg_workspace_result.sh.
# ci_test_tier.sh itself runs (pinned parent, scratch env): check_ci_push_reuse.sh check_tree_reader_tests.sh gate_touched_crates.sh lib/test_tier.py
want="scripts/ci/fetch_p6.sh scripts/ci/pr_touched_paths.sh scripts/ci_mg_workspace_result.sh scripts/ci_test_tier.sh "
wrow "W45a the tier step runs exactly the classified files" "$want" "$(tier_exec "$SEC")"
mutate "$SEC" 'bash scripts/ci_test_tier.sh' 'bash scripts/ci_test_tier.sh; bash scripts/new_helper.sh' "$td/exec.yml" && { [ "$(tier_exec "$td/exec.yml")" = "$want" ] && r=missed || r=caught; wrow "W45b MUTANT: an unclassified helper in the tier step turns the list RED" caught "$r"; } || { fail=$((fail + 1)); echo "FAIL W45b: mutant did not apply"; }
if mutate "$CI" '[ "$ok" = 1 ] || ok=0' '[ "$ok" = 1 ] || ok=1' "$td/b7.yml"; then wrow "W23g MUTANT (E2): a dead guard passes the garbage verifier answer, behaviourally" 2 "$(judge_run "$td/b7.yml" vgarbage | sed 's/^1$/2/')"; else fail=$((fail + 1)); echo "FAIL W21g: mutant did not apply"; fi
# W46: a PR rust-toolchain.toml `path=` must not run PR code in the tier step (gate_touched_crates.sh runs `cargo metadata` in the PR tree)
tc_plant() { # <RUSTUP_TOOLCHAIN or empty> -> 1 when the planted cargo ran
  local d; d=$(mktemp -d); mkdir -p "$d/tc/bin" "$d/p/src"
  printf '#!/bin/sh\ntouch %s/RAN\nexit 1\n' "$d" > "$d/tc/bin/cargo"; chmod +x "$d/tc/bin/cargo"
  printf '[toolchain]\npath="%s/tc"\n' "$d" > "$d/p/rust-toolchain.toml"
  printf '[package]\nname="x"\nversion="0.1.0"\nedition="2021"\n' > "$d/p/Cargo.toml"; : > "$d/p/src/lib.rs"
  (cd "$d/p" && env ${1:+RUSTUP_TOOLCHAIN="$1"} cargo metadata --no-deps --format-version 1 > /dev/null 2>&1 < /dev/null)
  [ -e "$d/RAN" ] && echo 1 || echo 0
  rm -rf -- "${d:?}"
}
tier_tc() { awk '/- name: Decide the test tier/{on=1;print;next} on&&/^      - name:/{exit} on{print}' "$1" | grep -v -E '^[[:space:]]*#' | grep -oE 'RUSTUP_TOOLCHAIN=[0-9.]+' | head -1 | cut -d= -f2; }
tcv=$(tier_tc "$SEC")
wrow "W46a the tier step pins RUSTUP_TOOLCHAIN to the base toolchain channel" "$(sed -n 's/^channel = "\(.*\)"/\1/p' "$HERE/../rust-toolchain.toml")" "$tcv"
wrow "W46b MUTANT: with no RUSTUP_TOOLCHAIN a PR toolchain path= runs PR code (the plant works)" 1 "$(tc_plant "")"
wrow "W46c with the pinned RUSTUP_TOOLCHAIN the planted cargo does not run" 0 "$(tc_plant "$tcv")"
wmut "W46d MUTANT: the RUSTUP_TOOLCHAIN pin removed from the tier step is caught" sec 'RUSTUP_TOOLCHAIN=1.93.0; export RUSTUP_TOOLCHAIN' ':'
echo "receipt gate rows: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
