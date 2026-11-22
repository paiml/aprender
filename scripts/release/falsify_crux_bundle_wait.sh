#!/usr/bin/env bash
# falsify_crux_bundle_wait.sh -- case table + mutants for scripts/release/crux_bundle_wait.sh.
#   bash scripts/release/falsify_crux_bundle_wait.sh             # exit 0 = every row held
#   bash scripts/release/falsify_crux_bundle_wait.sh --mutants   # exit 0 = every mutant caught
# Real git, no network: a throwaway bare origin with a commit and a nightly-evidence branch.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${CRUX_WAIT_SRC:-$HERE/crux_bundle_wait.sh}"
if [ "${1:-}" = --mutants ]; then
  M=$(mktemp -d) || exit 2; trap 'rm -rf -- "${M:?}"' EXIT; miss=0
  mut() { sed "$2" "$HERE/crux_bundle_wait.sh" > "$M/m.sh"; cmp -s "$HERE/crux_bundle_wait.sh" "$M/m.sh" && { echo "MUTANT $1 did not apply"; miss=$((miss+1)); return; }
    CRUX_WAIT_SRC="$M/m.sh" bash "$0" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED"; miss=$((miss+1)); }; }
  mut bundle-of-any-commit 's|^b="models-crux/\$sha"$|b="models-crux/$(git ls-tree --name-only "$remote/$branch:models-crux" \| grep -v keep \| head -n 1)"|'
  mut fetch-unchecked 's/|| die "cannot fetch \$branch from \$remote"$/|| :/'
  mut wait-skips-refetch 's/^  git fetch -q "\$remote" "+refs\/heads\/\$branch:refs\/remotes\/\$remote\/\$branch" \\$/  true \\/'
  mut wait-never-waits 's/^  if \[ "\$try" -ge "\$wait_tries" \]; then$/  if true; then/'
  mut wait-ignores-zero-tries 's/^  if \[ "\$try" -ge "\$wait_tries" \]; then$/  if [ "$try" -gt "$wait_tries" ]; then/'
  mut checksum-unchecked 's/^env -C "\$out" sha256sum .*$/:/'
  mut uncovered-file-allowed 's/^  grep -q -F -x -e "\$f" <(sed.*$/  :/'
  mut non-blob-allowed 's/^  \[ "\$(git cat-file -t "\$remote\/\$branch:\$b\/\$f" 2>\/dev\/null)" = blob \] || die.*$/  :/'
  mut verdict-commit-unread 's/^for want in "commit=\$sha" /for want in /'
  mut verdict-version-unread 's/ "version=\$ver" / /'
  mut verdict-measure-unread 's/ "measure=crux" / /'
  mut verdict-state-unread 's/ "state=green"; do$/; do/'
  mut verdict-substring 's/^  grep -q -F -x -e "\$want" "\$out\/verdict"/  grep -q -F -e "$want" "$out\/verdict"/'
  mut cert-unchecked 's/^  \[ -s "evidence\/crux\/\$ver\/prompt-certification\.json" \] \\$/  true \\/'
  [ "$miss" = 0 ] && { echo "all mutants caught"; exit 0; }; exit 1
fi
[ -f "$SRC" ] || { echo "FAIL crux_bundle_wait.sh missing at $SRC"; exit 1; }
T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
G() { git -c user.name=t -c user.email=t@example.invalid -c init.defaultBranch=main -c advice.detachedHead=false "$@"; }
FXV=1.2.3
bundle() {
  local d="$1/models-crux/$2"; mkdir -p "$d"
  echo '{"host":"lambda"}' > "$d/lambda-gpu.json"; echo '{"host":"gx10"}' > "$d/gx10-gpu.json"
  printf 'commit=%s\nsha9=%s\nversion=%s\nmeasure=%s\nt1_rc=0\nstate=%s\n' "$2" "${2:0:9}" "$3" "${5:-crux}" "$4" > "$d/verdict"
  (cd "$d" && sha256sum gx10-gpu.json lambda-gpu.json verdict > SHA256SUMS)
}
seed() {
  local r="$1" sha
  mkdir -p "$r/seed" && G init -q --bare "$r/origin.git" && G init -q "$r/seed" || return 1
  ( cd "$r/seed" || exit 1
    mkdir -p "evidence/crux/$FXV"; echo tagged > marker; echo keep > "evidence/crux/$FXV/.keep"
    [ -n "${NOCERT:-}" ] || echo '{"cert":1}' > "evidence/crux/$FXV/prompt-certification.json"
    G add -A && G commit -q -m tagged || exit 1
    sha=$(git rev-parse HEAD)
    G checkout -q --orphan nightly-evidence && G rm -r -f -q . && mkdir -p models-crux || exit 1
    [ -n "${NOBUNDLE:-}" ] || bundle "$PWD" "$sha" "${BVERSION:-$FXV}" "${BSTATE:-green}" "${BMEASURE:-crux}"
    [ -z "${OTHERBUNDLE:-}" ] || bundle "$PWD" "$OTHERBUNDLE" "$FXV" green crux
    if [ -n "${RELABEL:-}" ]; then
      bundle "$PWD" "$RELABEL" "$FXV" green crux; rm -rf -- "models-crux/${sha:?}"; mv "models-crux/$RELABEL" "models-crux/$sha"; fi
    [ -z "${TAMPER:-}" ] || echo '{"host":"other"}' > "models-crux/$sha/lambda-gpu.json"
    [ -z "${UNCOVERED:-}" ] || echo '{"host":"extra"}' > "models-crux/$sha/extra-gpu.json"
    [ -z "${SUBDIR:-}" ] || { install -d "models-crux/$sha/sub" && echo x > "models-crux/$sha/sub/a"; }
    if [ -n "${VERDICTPREFIX:-}" ]; then
      sed -i 's/^state=green$/prev_state=green\nstate=red/' "models-crux/$sha/verdict"
      env -C "models-crux/$sha" sha256sum gx10-gpu.json lambda-gpu.json verdict > "models-crux/$sha/SHA256SUMS"; fi
    echo keep > models-crux/.keep; G add -A && G commit -q -m evidence || exit 1
    G push -q "$r/origin.git" main nightly-evidence ) || return 1
  G clone -q "$r/origin.git" "$r/clone" || return 1
}
fails=0; n=0
row() { # name want_rc want_out_regex forbid_out_regex wait_tries
  local name=$1 wrc=$2 want=$3 forbid=$4 wtries=$5 r rc sha refs0 refs1
  n=$((n+1)); r="$T/$n"; mkdir -p "$r/tmp" "$r/bin"
  seed "$r" > "$r/seed.log" 2>&1 || { echo "FAIL $name: fixture did not build: $(tail -n 1 "$r/seed.log")"; fails=$((fails+1)); return; }
  sha=$(git -C "$r/origin.git" rev-parse refs/heads/main)
  refs0=$(git -C "$r/origin.git" for-each-ref | sort)
  if [ -n "${DELAYED:-}" ]; then
    G clone -q -b nightly-evidence "$r/origin.git" "$r/prod" >> "$r/seed.log" 2>&1 || { echo "FAIL $name: producer clone failed"; fails=$((fails+1)); return; }
    printf '%s\n%s\n%s\n%s\n%s\n' "$r" "$sha" "$FXV" "${DELAYED_AT:-1}" "${DELAYED_STATE:-green}" > "$r/sleep.env"
    cat > "$r/bin/sleep" <<'SLEEPSTUB'
#!/usr/bin/env bash
{ read -r r; read -r sha; read -r fxv; read -r at; read -r st; } < "${0%/*}/../sleep.env"
[ -e "$r/pushed" ] && exit 0
cur=$(cat "$r/polls" 2>/dev/null) || cur=0
if [ "$cur" -ge "$at" ]; then
  : > "$r/pushed"
  (
    # bashrs disable-line=SEC010
    cd "$r/prod" || exit 1
    # bashrs disable-line=SEC010
    mkdir -p "models-crux/$sha"
    echo '{"host":"lambda"}' > "models-crux/$sha/lambda-gpu.json"
    echo '{"host":"gx10"}' > "models-crux/$sha/gx10-gpu.json"
    printf 'commit=%s\nsha9=%s\nversion=%s\nmeasure=crux\nt1_rc=0\nstate=%s\n' "$sha" "${sha:0:9}" "$fxv" "$st" > "models-crux/$sha/verdict"
    # bashrs disable-line=SEC010
    (cd "models-crux/$sha" && sha256sum gx10-gpu.json lambda-gpu.json verdict > SHA256SUMS)
    git -c user.name=t -c user.email=t@example.invalid add -A && git -c user.name=t -c user.email=t@example.invalid commit -q -m delayed && git push -q origin nightly-evidence
  ) >> "$r/seed.log" 2>&1
else
  echo "$(( cur + 1 ))" > "$r/polls"
fi
exit 0
SLEEPSTUB
    chmod +x "$r/bin/sleep"
    echo 1 > "$r/polls"
  fi
  [ -z "${BADREMOTE:-}" ] || git -C "$r/clone" remote set-url origin "$r/missing.git"
  local extra=(); [ -z "${REQCERT:-}" ] || extra+=(--require-cert)
  # bashrs disable-line=SEC010
  env -C "$r/clone" PATH="$r/bin:$PATH" bash "$SRC" --version "${VERARG-$FXV}" --sha "${SHAARG-$sha}" --out "$r/out/models-crux/$sha" --wait-tries "$wtries" --poll "${WAITPOLL:-0}" "${extra[@]}" > "$r/outlog" 2>&1; rc=$?
  refs1=$(git -C "$r/origin.git" for-each-ref | sort)
  [ "$rc" = "$wrc" ] || { echo "FAIL $name: rc=$rc want $wrc: $(tail -n 1 "$r/outlog")"; fails=$((fails+1)); return; }
  grep -q -E -e "$want" "$r/outlog" || { echo "FAIL $name: output lacks /$want/: $(tail -n 1 "$r/outlog")"; fails=$((fails+1)); return; }
  if [ -n "$forbid" ] && grep -q -E -e "$forbid" "$r/outlog"; then echo "FAIL $name: output matched forbidden /$forbid/"; fails=$((fails+1)); return; fi
  if [ -z "${DELAYED:-}" ]; then
    [ "$refs0" = "$refs1" ] || { echo "FAIL $name: origin ref changed"; fails=$((fails+1)); return; }
  fi
  echo "ok   $name"
}
OTHER=3333333333333333333333333333333333333333
row green_bundle_verifies                  0 'checksums hold, verdict green for 1\.2\.3' '' 0
VERARG=1.2 row bad_version_usage           2 'bad version' '' 0
SHAARG=abc row bad_sha_usage               2 'bad sha' '' 0
BADREMOTE=1 row fetch_failure_refuses      1 'STOP cannot fetch nightly-evidence from origin' '' 0
NOBUNDLE=1 row no_bundle_zero_tries_refuses 1 'STOP no CRUX bundle for [0-9a-f]{40} on nightly-evidence \(models-crux/[0-9a-f]{40}/verdict absent\)' 'waiting for CRUX bundle' 0
NOBUNDLE=1 row wait_exhausted_refuses      1 'STOP no CRUX bundle for [0-9a-f]{40} on nightly-evidence after 2 poll\(s\)' 'poll 3/' 2
NOBUNDLE=1 DELAYED=1 DELAYED_AT=2 WAITPOLL=1 row delayed_bundle_arrives_green 0 'checksums hold, verdict green for 1\.2\.3' '' 3
NOBUNDLE=1 DELAYED=1 DELAYED_AT=2 DELAYED_STATE=red WAITPOLL=1 row delayed_red_bundle_refuses 1 "verdict has no line 'state=green'" '' 3
NOBUNDLE=1 OTHERBUNDLE=$OTHER row other_commit_bundle_refuses 1 'STOP no CRUX bundle for [0-9a-f]{40}' '' 0
RELABEL=$OTHER row relabelled_bundle_refuses 1 "verdict has no line 'commit=" '' 0
BVERSION=1.2.2 row other_version_refuses   1 "verdict has no line 'version=1\.2\.3'" '' 0
BMEASURE=ladder row ladder_measure_refuses 1 "verdict has no line 'measure=crux'" '' 0
BSTATE=red row red_verdict_refuses         1 "verdict has no line 'state=green'" '' 0
VERDICTPREFIX=1 row substring_green_refuses 1 "verdict has no line 'state=green'" '' 0
TAMPER=1 row tampered_receipt_refuses      1 'SHA256SUMS does not hold' '' 0
UNCOVERED=1 row uncovered_file_refuses     1 'extra-gpu\.json is not covered by SHA256SUMS' '' 0
SUBDIR=1 row non_blob_entry_refuses        1 'sub is not a file' '' 0
NOCERT=1 REQCERT=1 row missing_cert_refuses 1 'carries no evidence/crux/1\.2\.3/prompt-certification\.json' '' 0
REQCERT=1 row present_cert_passes          0 'checksums hold, verdict green for 1\.2\.3' '' 0
if grep -n -E 'git (push|tag)\b|--force|gh release' "$SRC" | grep -v -E '^[0-9]+:\s*#'; then
  echo "FAIL source carries a push, tag or force"; fails=$((fails+1)); else echo "ok   source_has_no_push_tag_or_force"; fi
[ "$fails" = 0 ] && { echo "all rows held"; exit 0; }
echo "$fails row(s) failed"; exit 1
