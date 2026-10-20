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
judge_ok() { # CI_FILE SECTIONS_FILE -> 0 when the wiring is trustworthy
  local job code shard tier
  job=$(awk '/^  receipt-judge:$/{on=1;print;next} on&&/^  [a-z]/{exit} on{print}' "$1")
  [ -n "$job" ] || return 1
  code=$(printf '%s\n' "$job" | grep -v -E '^[[:space:]]*#')
  case "$code" in *'ref: ${{ github.base_ref }}'*) ;; *) return 1 ;; esac
  [ "$(printf '%s\n' "$code" | grep -c 'uses:')" -eq 1 ] || return 1
  printf '%s\n' "$code" | grep -q 'uses: actions/checkout@' || return 1
  printf '%s\n' "$code" | grep -q -E 'GITHUB_PATH|GITHUB_ENV|head_ref|pull_request\.head|github\.sha *$' && return 1
  [ "$(printf '%s\n' "$code" | grep -c '^      - ')" -eq 2 ] || return 1
  printf '%s\n' "$code" | grep -E 'scripts/' | grep -v -E 'GITHUB_WORKSPACE/scripts/ci/receipt_ok_arg\.sh|-f scripts/ci/receipt_ok_arg\.sh' | grep -q . && return 1
  shard=$(awk '/^  workspace-test-shard:$/{on=1;print;next} on&&/^  [a-z]/{exit} on{print}' "$1")
  case "$shard" in *'needs: [receipt-judge]'*) ;; *) return 1 ;; esac
  case "$shard" in *'RECEIPT_OK: ${{ needs.receipt-judge.outputs.ok'*) ;; *) return 1 ;; esac
  tier=$(awk '/id: tier$/{on=1} on{print} /--receipt-ok/{if(on)exit}' "$2" | grep -v -E '^[[:space:]]*#')
  case "$tier" in *'"${RECEIPT_OK:-0}" = 1'*) ;; *) return 1 ;; esac
  printf '%s\n' "$tier" | grep -E 'scripts/|git (fetch|clone)' | grep -q . && return 1
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
wmut "W20 MUTANT: a shard job that does not wait for the judge is caught" ci 'needs: [receipt-judge]' 'needs: []'
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
if mutate "$W" 'export GIT_NO_REPLACE_OBJECTS=1' ':' "$td/mut-r.sh"; then
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
cat >"$td/trust-probe/scripts/ci_receipt_gate.sh" <<'PROBE'
#!/usr/bin/env bash
printf '%s|%s|%s\n' "${GH_TOKEN:-}${GITHUB_OUTPUT:-}" "${RECEIPT_RUSTC_CWD:-}" "$(ls "${RECEIPT_RUSTC_CWD:-/nonexistent}" 2>/dev/null | tr -d '\n')" >"$PROBEFILE"
printf 'RECEIPT_OK=0\n'
PROBE
git -C "$td/trust-probe" -c user.email=t@example.invalid -c user.name=t commit -q -am probe
probe() { rm -f "${td:?}/probe.out"; (cd "$repo" && GH_TOKEN=secret GITHUB_OUTPUT=/tmp/x PROBEFILE="$td/probe.out" bash "${1:-$W}" "$td/b-good" "$td/trust-probe" "$head" >/dev/null 2>&1); sed -E 's#\|[^|]*/rc\|#|RC|#' "$td/probe.out" 2>/dev/null; }
wrow "W25 the verifier sees no token or output file, and runs rustc in the base's dir" "|RC|rust-toolchain.toml" "$(probe)"
if mutate "$W" '-u GH_TOKEN' '-u XX_TOKEN' "$td/mut-e.sh"; then
  wrow "W26 MUTANT: a wrapper that leaves GH_TOKEN in the verifier's env is caught by W25" "secret|RC|rust-toolchain.toml" "$(probe "$td/mut-e.sh")"
else fail=$((fail + 1)); echo "FAIL W26: mutant did not apply"; fi
tier_of() { (cd "$HERE/.." && bash "$T" --event pull_request --diff-from "$td/touched.txt" --receipt-ok "$1" 2>/dev/null | head -n1); }
t0=$(tier_of 0); case "$t0" in tier=quick|tier=full) t0=notreuse ;; esac
wrow "W8 --receipt-ok 0 on a core diff runs quick or full, never reuse" notreuse "$t0"
wrow "W9 --receipt-ok 1 on the same diff is reuse" tier=reuse "$(tier_of 1)"

echo "receipt gate rows: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
