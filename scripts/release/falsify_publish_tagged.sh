#!/usr/bin/env bash
# falsify_publish_tagged.sh -- case table for scripts/release/publish_tagged.sh (aprender#6886)
#   bash scripts/release/falsify_publish_tagged.sh             # exit 0 = every row held
#   bash scripts/release/falsify_publish_tagged.sh --mutants   # exit 0 = every mutant caught
# Real git, no network: a throwaway bare "origin" with a tagged commit and a nightly-evidence branch, and a
# clone the script runs in. The publish command is a stub that records what it was handed. Nothing real is touched.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${PUB_TAGGED_SRC:-$HERE/publish_tagged.sh}"
if [ "${1:-}" = --mutants ]; then
  M=$(mktemp -d) || exit 2; trap 'rm -rf -- "${M:?}"' EXIT; miss=0
  mut() { sed "$2" "$HERE/publish_tagged.sh" > "$M/m.sh"; cmp -s "$HERE/publish_tagged.sh" "$M/m.sh" && { echo "MUTANT $1 did not apply"; miss=$((miss+1)); return; }
    PUB_TAGGED_SRC="$M/m.sh" bash "$0" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED"; miss=$((miss+1)); }; }
  mut dirty-tree-allowed 's/^  \[ -z "\$(git status --porcelain 2>\/dev\/null)" \] || die.*$/  :/'
  mut tag-fetch-forced 's|"refs/tags/\$T:refs/tags/\$T" \|\| die|"+refs/tags/$T:refs/tags/$T" \|\| die|'
  mut bundle-of-any-commit 's|^  B="models-crux/\$SHA"$|  B="models-crux/$(git ls-tree --name-only "$REMOTE/$EVBR:models-crux" \| grep -v keep \| head -n 1)"|'
  mut checksum-unchecked 's/^  (cd "\$C" \&\& sha256sum .*$/  :/'
  mut uncovered-file-allowed 's/^    grep -q -F -x -e "\$f" <(sed.*$/    :/'
  mut verdict-commit-unread 's/^  for want in "commit=\$SHA" /  for want in /'
  mut verdict-version-unread 's/ "version=\$V" / /'
  mut verdict-state-unread 's/ "state=green"; do$/; do/'
  mut verdict-substring 's/^    grep -q -F -x -e "\$want" "\$C\/verdict"/    grep -q -F -e "$want" "$C\/verdict"/'
  mut no-checkout 's/^  git checkout -q --detach "\$SHA" || die.*$/  :/'
  mut cert-unchecked 's/^  \[ -s "evidence\/crux\/\$V\/prompt-certification.json" \] || die.*$/  :/'
  mut dry-run-ignored 's/^  if \[ "\$DRY" = 1 \]; then$/  if false; then/'
  mut crux-dir-not-handed 's/^  export MODEL_LADDER_CRUX_DIR="\$C" /  export /'
  mut cert-not-handed 's/ CRUX_CERT="\$PWD\/evidence\/crux\/\$V\/prompt-certification.json"$//'
  mut publish-red-swallowed 's/^  \$PUBLISH || die.*$/  $PUBLISH || :/'
  mut publish-skipped 's/^  \$PUBLISH || die.*$/  :/'
  mut scratch-left-behind "s/^  trap 'rm -rf -- .*$/  :/"
  [ "$miss" = 0 ] && { echo "all mutants caught"; exit 0; }; exit 1
fi
[ -f "$SRC" ] || { echo "FAIL publish_tagged.sh missing at $SRC"; exit 1; }
T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
G() { git -c user.name=t -c user.email=t@example.invalid -c init.defaultBranch=main -c advice.detachedHead=false "$@"; }
V=1.2.3
# bundle DIR SHA VERSION STATE -> writes DIR/models-crux/<SHA>/, checksummed.
bundle() {
  local d="$1/models-crux/$2"; mkdir -p "$d"
  echo '{"host":"lambda"}' > "$d/lambda-gpu.json"; echo '{"host":"gx10"}' > "$d/gx10-gpu.json"
  printf 'commit=%s\nsha9=%s\nversion=%s\nmeasure=crux\nt1_rc=0\nstate=%s\n' "$2" "${2:0:9}" "$3" "$4" > "$d/verdict"
  (cd "$d" && sha256sum gx10-gpu.json lambda-gpu.json verdict > SHA256SUMS)
}
# seed DIR: a bare origin with main (two commits; the first is tagged v1.2.3) and nightly-evidence, and a clone.
#   env: NOTAG NOCERT NOBUNDLE OTHERBUNDLE RELABEL BVERSION BSTATE TAMPER UNCOVERED VERDICTPREFIX
seed() {
  local r="$1" sha
  mkdir -p "$r/seed" && G init -q --bare "$r/origin.git" && G init -q "$r/seed" || return 1
  ( cd "$r/seed" || exit 1
    mkdir -p "evidence/crux/$V"; echo tagged > marker; echo keep > "evidence/crux/$V/.keep"
    [ -n "${NOCERT:-}" ] || echo '{"cert":1}' > "evidence/crux/$V/prompt-certification.json"
    G add -A && G commit -q -m tagged || exit 1
    sha=$(git rev-parse HEAD)
    [ -n "${NOTAG:-}" ] || G tag -a "v$V" -m "v$V" "$sha"
    echo later > marker; G commit -q -am later || exit 1
    G checkout -q --orphan nightly-evidence && G rm -r -f -q . && mkdir -p models-crux || exit 1
    [ -n "${NOBUNDLE:-}" ] || bundle "$PWD" "$sha" "${BVERSION:-$V}" "${BSTATE:-green}"
    [ -z "${OTHERBUNDLE:-}" ] || bundle "$PWD" "$OTHERBUNDLE" "$V" green
    if [ -n "${RELABEL:-}" ]; then   # another commit's bundle under this sha's path: the path is right, the verdict is not
      bundle "$PWD" "$RELABEL" "$V" green; rm -rf -- "models-crux/${sha:?}"; mv "models-crux/$RELABEL" "models-crux/$sha"; fi
    [ -z "${TAMPER:-}" ] || echo '{"host":"other"}' > "models-crux/$sha/lambda-gpu.json"
    [ -z "${UNCOVERED:-}" ] || echo '{"host":"extra"}' > "models-crux/$sha/extra-gpu.json"
    if [ -n "${VERDICTPREFIX:-}" ]; then   # "state=green" only inside a longer line, checksummed honestly
      sed -i 's/^state=green$/prev_state=green\nstate=red/' "models-crux/$sha/verdict"
      (cd "models-crux/$sha" && sha256sum gx10-gpu.json lambda-gpu.json verdict > SHA256SUMS); fi
    echo keep > models-crux/.keep; G add -A && G commit -q -m evidence || exit 1
    G push -q "$r/origin.git" main nightly-evidence --tags ) || return 1
  G clone -q "$r/origin.git" "$r/clone" || return 1
}
fails=0; n=0
row() { # name want_rc want_out_regex want_calls(0|1) dry
  local name=$1 wrc=$2 want=$3 wcalls=$4 dry=$5 r rc tagsha refs0 refs1
  n=$((n+1)); r="$T/$n"; mkdir -p "$r/tmp"
  seed "$r" > "$r/seed.log" 2>&1 || { echo "FAIL $name: fixture did not build: $(tail -n 1 "$r/seed.log")"; fails=$((fails+1)); return; }
  tagsha=$(git -C "$r/origin.git" rev-parse --verify --quiet "refs/tags/v$V^{commit}" 2>/dev/null) || tagsha=none
  [ -z "${DIRTY:-}" ] || echo x > "$r/clone/untracked"
  [ -z "${LOCALTAG:-}" ] || G -C "$r/clone" tag -f -a "v$V" -m moved origin/main > /dev/null 2>&1
  refs0=$(git -C "$r/origin.git" for-each-ref | sort)
  cat > "$r/publish" <<STUB
#!/usr/bin/env bash
{ echo "head=\$(git rev-parse HEAD)"; echo "marker=\$(cat marker)"; echo "cert=\${CRUX_CERT:-}"; [ -s "\${CRUX_CERT:-/nonexistent}" ] && echo cert_readable=1
  echo "crux_commit=\$(sed -n 's/^commit=//p' "\${MODEL_LADDER_CRUX_DIR:-/nonexistent}/verdict" 2>/dev/null)"; } > "$r/calls"
exit ${PUBRC:-0}
STUB
  chmod +x "$r/publish"
  ( cd "$r/clone" && PUB_TAGGED_PUBLISH="$r/publish" PUB_TAGGED_TMP="$r/tmp" bash "$SRC" "${VERARG-$V}" "$dry" ) > "$r/out" 2>&1; rc=$?
  [ -z "${DBG:-}" ] || cat "$r/out"
  refs1=$(git -C "$r/origin.git" for-each-ref | sort)
  [ "$rc" = "$wrc" ] || { echo "FAIL $name: rc=$rc want $wrc: $(tail -n 1 "$r/out")"; fails=$((fails+1)); return; }
  grep -q -E -e "$want" "$r/out" || { echo "FAIL $name: output lacks /$want/: $(tail -n 1 "$r/out")"; fails=$((fails+1)); return; }
  [ "$refs0" = "$refs1" ] || { echo "FAIL $name: a ref on origin changed"; fails=$((fails+1)); return; }
  [ -z "$(ls -A "$r/tmp")" ] || { echo "FAIL $name: scratch dir left behind"; fails=$((fails+1)); return; }
  if [ "$wcalls" = 0 ]; then
    [ ! -e "$r/calls" ] || { echo "FAIL $name: publish was called"; fails=$((fails+1)); return; }
  else
    grep -q -F -x -e "head=$tagsha" "$r/calls" && grep -q -F -x -e "marker=tagged" "$r/calls" \
      && grep -q -F -x -e "crux_commit=$tagsha" "$r/calls" && grep -q -F -x -e "cert_readable=1" "$r/calls" \
      && grep -q -E -e "^cert=/.*/evidence/crux/$V/prompt-certification\.json$" "$r/calls" \
      || { echo "FAIL $name: publish was not handed the tag commit and its receipts: $(tr '\n' ' ' < "$r/calls" 2>/dev/null)"; fails=$((fails+1)); return; }
  fi
  echo "ok   $name"
}
OTHER=3333333333333333333333333333333333333333
row green_publishes_the_tag_commit                   0 'PUBLISHED v1\.2\.3 at [0-9a-f]{40}' 1 0
PUBRC=1 row publish_refusal_is_red                   1 'STOP publish refused or failed' 1 0
row dry_run_uploads_nothing                          0 'WOULD publish v1\.2\.3' 0 1
VERARG=1.2 row bad_version_refuses                   2 'STOP bad version' 0 0
VERARG='' row no_version_refuses                     2 'STOP bad version' 0 0
NOTAG=1 row untagged_version_refuses                 1 'STOP tag v1\.2\.3 is not on origin' 0 0
LOCALTAG=1 row local_tag_elsewhere_refuses           1 'STOP tag v1\.2\.3 is not on origin \(or differs' 0 0
DIRTY=1 row dirty_tree_refuses                       1 'STOP the tree is not clean' 0 0
NOBUNDLE=1 row no_bundle_refuses                     1 'STOP no CRUX bundle for [0-9a-f]{40}' 0 0
NOBUNDLE=1 OTHERBUNDLE=$OTHER row bundle_of_another_commit_refuses 1 'STOP no CRUX bundle for [0-9a-f]{40}' 0 0
RELABEL=$OTHER row relabelled_bundle_refuses         1 "verdict has no line 'commit=" 0 0
BVERSION=1.2.2 row bundle_of_another_version_refuses 1 "verdict has no line 'version=1\.2\.3'" 0 0
BSTATE=red row red_verdict_refuses                   1 "verdict has no line 'state=green'" 0 0
VERDICTPREFIX=1 row green_as_substring_refuses       1 "verdict has no line 'state=green'" 0 0
TAMPER=1 row tampered_receipt_refuses                1 'SHA256SUMS does not hold' 0 0
UNCOVERED=1 row uncovered_receipt_refuses            1 'extra-gpu\.json is not covered by SHA256SUMS' 0 0
NOCERT=1 row no_committed_cert_refuses               1 'carries no evidence/crux/1\.2\.3/prompt-certification\.json' 0 0
# the script itself holds no ref-moving or tagging command
if grep -n -E 'git (push|tag)\b|--force|gh release' "$SRC" | grep -v -E '^[0-9]+:\s*#'; then
  echo "FAIL source carries a push, tag or force"; fails=$((fails+1)); else echo "ok   source_has_no_push_tag_or_force"; fi
[ "$fails" = 0 ] && { echo "all rows held"; exit 0; }
echo "$fails row(s) failed"; exit 1
