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

# W: the workflow wiring. scripts/ci/receipt_ok_arg.sh prints the ONE word the tier step
# hands to ci_test_tier.sh as --receipt-ok. Anything but a verified receipt must print 0,
# and 0 must leave the PR on the quick/full tier, never on reuse.
W="$HERE/ci/receipt_ok_arg.sh"
T="$HERE/ci_test_tier.sh"
printf 'crates/aprender-core/src/lib.rs\n' >"$td/touched.txt"
arg() { (cd "$repo" && bash "$W" "$1" "${2:-$td/pub}" "$head" 2>/dev/null | head -n1); }
wrow() {
  local name=$1 want=$2 got=$3
  if [ "$got" = "$want" ]; then pass=$((pass + 1)); echo "ok   $name"; else fail=$((fail + 1)); echo "FAIL $name: want '$want' got '$got'"; fi
}
wrow "W1 verified receipt -> --receipt-ok 1" 1 "$(arg "$td/b-good")"
wrow "W2 no receipt in the PR body -> 0 (full gate)" 0 "$(arg "$td/b-none")"
wrow "W3 tampered receipt -> 0" 0 "$(arg "$td/b-s1")"
wrow "W4 receipt for another tree -> 0" 0 "$(arg "$td/b-t1")"
wrow "W5 placeholder pubkey -> 0" 0 "$(arg "$td/b-good" "$td/pub-empty")"
wrow "W6 PR body file missing -> 0" 0 "$(arg "$td/no-such-body")"
wrow "W7 no pubkey file -> 0" 0 "$(arg "$td/b-good" "$td/no-such-pub")"
tier_of() { (cd "$HERE/.." && bash "$T" --event pull_request --diff-from "$td/touched.txt" --receipt-ok "$1" 2>/dev/null | head -n1); }
t0=$(tier_of 0); case "$t0" in tier=quick|tier=full) t0=notreuse ;; esac
wrow "W8 --receipt-ok 0 on a core diff runs quick or full, never reuse" notreuse "$t0"
wrow "W9 --receipt-ok 1 on the same diff is reuse" tier=reuse "$(tier_of 1)"
# Mutant: a wrapper that prints 1 whatever the verifier said turns W2-W7 RED.
sed 's|if \[ "\$first" = "RECEIPT_OK=1" \]; then|if true; then|' "$W" >"$td/mut-w.sh"
if ! cmp -s "$W" "$td/mut-w.sh"; then
  m=$(cd "$repo" && bash "$td/mut-w.sh" "$td/b-none" "$td/pub" "$head" 2>/dev/null | head -n1)
  wrow "W10 MUTANT: a wrapper that always prints 1 is caught by the no-receipt row" 1 "$m"
else
  fail=$((fail + 1)); echo "FAIL W10: mutant did not apply to $W"
fi

echo "receipt gate rows: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
