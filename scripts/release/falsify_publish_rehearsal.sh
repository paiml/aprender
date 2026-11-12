#!/usr/bin/env bash
# falsify_publish_rehearsal.sh -- case table for scripts/release/publish_rehearsal.sh (PMAT-6914)
#   bash scripts/release/falsify_publish_rehearsal.sh             # exit 0 = every row held
#   bash scripts/release/falsify_publish_rehearsal.sh --mutants   # exit 0 = every planted defect caught
# Real git, no network, no credential: each row copies a throwaway repo (tags, the crate, the script under test)
# and runs the script there with fakes for the clean-room gate, the upload and the index read. `cargo` and `curl`
# on PATH are tripwires that record a call and fail; any call to them is a failed row. Two rows at the end need
# a real cargo (workspace membership, offline package) and run on the build host.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
ROOT=$(cd "$HERE/../.." && pwd) || exit 2
SRC="${PUB_REH_SRC:-$HERE/publish_rehearsal.sh}"
CRATE_SRC="$ROOT/ci/publish-rehearsal"
CASCADE_REAL="$ROOT/scripts/cascade-publish.sh"
SCRATCH_BASE="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"

if [ "${1:-}" = --mutants ]; then
  M=$(mktemp -d -p "$SCRATCH_BASE" falsify-pubreh-mut.XXXXXX) || exit 2; trap 'rm -rf -- "${M:?}"' EXIT; miss=0; total=0
  mut() { # name sed-expression: the mutated script must turn at least one row red
    total=$((total+1)); sed "$2" "$HERE/publish_rehearsal.sh" > "$M/m.sh"
    cmp -s "$HERE/publish_rehearsal.sh" "$M/m.sh" && { echo "MUTANT $1 did not apply"; miss=$((miss+1)); return; }
    PUB_REH_SRC="$M/m.sh" PUB_REH_SCRIPT_ROWS_ONLY=1 bash "$0" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED"; miss=$((miss+1)); }; return 0; }
  mut run-number-unchecked        's/^  \[\[ \$N =~ .*$/  :/'
  mut run-number-leading-zero     's/\^\[1-9\]\[0-9\]\*\$ \]\] || { echo/^[0-9][0-9]*$ ]] || { echo/'
  mut dry-arg-unchecked           's/^  case "\$DRY" in 0|1) ;; .*$/  :/'
  mut version-prefix-wrong        's/V="0\.0\.\$N"/V="0.1.$N"/'
  mut tag-filter-removed          "s/ | grep -E '[^']*'//"
  mut oldest-tag-picked           's/--sort=-creatordate/--sort=creatordate/'
  mut no-tag-allowed              's/^  \[ -n "\$TAG" \] || die .*$/  :/'
  mut gate-refusal-swallowed      's/^  \[ "\$grc" = 0 \] || die .*$/  :/'
  mut gate-skipped                's/^  ( unset CARGO_REGISTRY_TOKEN; run_gate .*$/  grc=0/'
  mut gate-sees-the-token         's/unset CARGO_REGISTRY_TOKEN; //'
  mut gate-function-unchecked     's/^    grep -q .*fns.*$/    :/'
  mut gate-gets-wrong-tag         's/clean_room_gate "\$1" "\$2"/clean_room_gate "$1" "v0.0.0"/'
  mut cascade-top-level-run       's/^    \. "\$fns" || exit 1$/    . "$cascade" || exit 1/'
  mut dry-run-ignored             's/^  if \[ "\$DRY" = 1 \]; then$/  if false; then/'
  mut empty-token-allowed         's/^  \[ -n "\${CARGO_REGISTRY_TOKEN:-}" \] || die .*$/  :/'
  mut version-not-set             's/^  sed -i .*$/  :/'
  mut version-set-unchecked       's/^  grep -q -F -x -e "version = .*$/  :/'
  mut version-set-in-the-tree     's/"\$TMP\/crate\/Cargo.toml"/"$CRATE_DIR\/Cargo.toml"/g'
  mut publish-runs-in-the-tree    's/( cd "\$TMP\/crate" \&\& publish_cmd )/( cd "$CRATE_DIR" \&\& publish_cmd )/'
  mut upload-refusal-swallowed    's/^    die "upload: refused or failed.*$/    :/'
  mut already-uploaded-not-green  's/^  if \[ "\$prc" != 0 \] \&\& already_uploaded .*; then$/  if false; then/'
  mut any-failure-is-uploaded     's/^  if \[ "\$prc" != 0 \] \&\& already_uploaded .*; then$/  if [ "$prc" != 0 ]; then/'
  mut other-version-is-uploaded   's/^    \&\& grep -q -E -e "(.*$/    \&\& true/'
  mut longer-version-is-uploaded  's/(\[^0-9\]|\\\$)" "\$1"$/" "$1"/'
  mut index-unchecked             's/^    \$INDEX "\$CRATE" "\$V" > .dev.null 2>&1 \&\& { seen=1; break; }$/    seen=1; break/'
  mut index-asked-for-wrong-version 's/\$INDEX "\$CRATE" "\$V"/$INDEX "$CRATE" "0.0.0"/'
  mut index-retry-off-by-five     's/^  while \[ "\$i" -lt "\$TRIES" \]; do$/  while [ "$i" -lt $((TRIES+5)) ]; do/'
  mut index-never-retried         's/^    \[ "\$i" -lt "\$TRIES" \] \&\& sleep.*$/    break/'
  mut index-miss-swallowed        's/^  \[ "\$seen" = 1 \] || die .*$/  :/'
  mut output-not-scrubbed         's/^    \[ -z "\$tok" \] || line=.*$/    :/'
  mut scratch-left-behind         's/^  trap .*EXIT$/  :/'
  [ "$miss" = 0 ] && { echo "mutants: $total of $total killed"; exit 0; }
  echo "mutants: $((total-miss)) of $total killed"; exit 1
fi

[ -f "$SRC" ] || { echo "FAIL publish_rehearsal.sh missing at $SRC"; exit 1; }
T=$(mktemp -d -p "$SCRATCH_BASE" falsify-pubreh.XXXXXX) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
G() { git -c user.name=t -c user.email=t@example.invalid -c init.defaultBranch=main -c advice.detachedHead=false -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
MARK="TOK-MARKER-7f3a9c1e-do-not-print"

# mkrepo DIR TAGGED(1|0): the tree release day has -- the crate, the script under test, tags with distinct dates.
mkrepo() {
  local d=$1 tagged=$2 c
  mkdir -p "$d/scripts/release" "$d/ci" && G init -q "$d" || return 1
  cp -R "$CRATE_SRC" "$d/ci/publish-rehearsal" && rm -rf -- "${d:?}/ci/publish-rehearsal/target" "${d:?}/ci/publish-rehearsal/Cargo.lock" || return 1
  cp "$SRC" "$d/scripts/release/publish_rehearsal.sh" || return 1
  for c in 1 2 3; do
    echo "c$c" > "$d/marker"; G -C "$d" add -A
    GIT_AUTHOR_DATE="2026-01-0${c}T00:00:00Z" GIT_COMMITTER_DATE="2026-01-0${c}T00:00:00Z" G -C "$d" commit -q -m "c$c" || return 1
    if [ "$tagged" = 1 ]; then
      [ "$c" = 1 ] && G -C "$d" tag v1.2.2
      [ "$c" = 2 ] && G -C "$d" tag v1.2.3
    fi
    [ "$c" = 3 ] && G -C "$d" tag v1.2.4-rc1 && G -C "$d" tag nightly-20260103 && G -C "$d" tag v1.2
  done
  return 0
}
mkrepo "$T/tpl-tagged" 1 || { echo "FAIL fixture repo did not build"; exit 1; }
mkrepo "$T/tpl-untagged" 0 || { echo "FAIL fixture repo did not build"; exit 1; }

# Fixture cascade files: the script loads the five clean-room functions from a cascade-publish.sh and must not run its top level.
FX="$T/fx"; mkdir -p "$FX"
fxfile() { # FILE GATE_BODY [omit-function]
  { echo 'touch "$(dirname "${BASH_SOURCE[0]}")/toplevel-ran"'; echo 'exit 77'
    for f in clean_room_parse_runs clean_room_parse_job clean_room_tested_abbrevs clean_room_tested_shas; do
      [ "${3:-}" = "$f" ] || printf '%s() {\n  return 0\n}\n' "$f"; done
    printf 'clean_room_gate() {\n  %s\n}\n' "$2"; } > "$1"; }
fxfile "$FX/ok.sh" 'echo "ARGS root=$1 tag=$2 n=$#"; echo "CLEAN-ROOM PROCEED: fixture"; return 0'
fxfile "$FX/refuse.sh" 'echo "CLEAN-ROOM REFUSE: fixture says no"; return 1'
fxfile "$FX/missing.sh" 'return 0' clean_room_tested_shas

fails=0; n=0
fail() { echo "FAIL $1: $2"; fails=$((fails+1)); }
# row NAME WANT_RC WANT_OUT_REGEX WANT_GATE_CALLS(1|0|x) WANT_PUBLISH_CALLS(1|0) DRY
#   env knobs: RUN TOKEN GATERC PUBRC PUBOUT INDEX_AFTER IDXN TPL USEGATE CASCADE_FX GHRC
row() {
  local name=$1 wrc=$2 want=$3 wgate=$4 wpub=$5 dry=$6 r rc head0 tags0 tok run envs=() ver idxn
  n=$((n+1)); r="$T/$n"; mkdir -p "$r/tmp" "$r/bin"; rm -f -- "${FX:?}/toplevel-ran"
  cp -a "$T/${TPL:-tpl-tagged}" "$r/repo" || { fail "$name" "fixture copy"; return; }
  run=${RUN-42}; ver="0.0.$run"; tok=${TOKEN-$MARK}
  # tripwires: the table makes no network call; gh exists only so the real gate can ask whether it is logged in
  for b in cargo curl; do printf '#!/usr/bin/env bash\necho "%s $*" >> "%s/net.calls"\nexit 99\n' "$b" "$r" > "$r/bin/$b"; done
  printf '#!/usr/bin/env bash\necho "gh $*" >> "%s/gh.calls"\nexit %s\n' "$r" "${GHRC:-1}" > "$r/bin/gh"
  cat > "$r/gate" <<STUB
#!/usr/bin/env bash
echo "ROOT=\$1 TAG=\$2 token=\${CARGO_REGISTRY_TOKEN+set}" >> "$r/gate.calls"
if [ ${GATERC:-0} = 0 ]; then echo "CLEAN-ROOM PROCEED: fixture run for \$2"; else echo "CLEAN-ROOM REFUSE: fixture says no for \$2"; fi
exit ${GATERC:-0}
STUB
  cat > "$r/publish" <<STUB
#!/usr/bin/env bash
{ grep -E '^version = ' Cargo.toml; echo "cwd=\$PWD"; [ "\${CARGO_REGISTRY_TOKEN:-}" = "$MARK" ] && echo token_ok=1
  case "\$PWD/" in "$r/repo/"*) echo in_tree=1 ;; esac; } >> "$r/publish.calls"
: > Cargo.lock
[ -z "\${PUBOUT_TEXT:-}" ] || printf '%s\n' "\$PUBOUT_TEXT"
exit ${PUBRC:-0}
STUB
  cat > "$r/index" <<STUB
#!/usr/bin/env bash
echo "\$*" >> "$r/index.calls"
[ ${INDEX_AFTER:-1} -gt 0 ] && [ "\$(wc -l < "$r/index.calls")" -ge ${INDEX_AFTER:-1} ]
STUB
  chmod +x "$r/gate" "$r/publish" "$r/index" "$r/bin/"*
  envs=(RUNNER_TEMP="$r/tmp" PUB_REH_TRIES=3 PUB_REH_WAIT=0 PUB_REH_PUBLISH="$r/publish" PUB_REH_INDEX="$r/index" PUBOUT_TEXT="${PUBOUT-}")
  [ "${USEGATE-1}" = 1 ] && envs+=(PUB_REH_GATE="$r/gate")
  [ -z "${CASCADE_FX:-}" ] || envs+=(PUB_REH_CASCADE="$CASCADE_FX")
  [ "$tok" != EMPTY ] || tok=""                                 # TOKEN=EMPTY is an empty token, TOKEN=UNSET is none at all
  [ "$tok" = UNSET ] || envs+=(CARGO_REGISTRY_TOKEN="$tok")
  if [ -n "${BADCRATE:-}" ]; then   # a committed placeholder other than 0.0.0: nothing to set the run version on
    sed -i 's/^version = "0\.0\.0"$/version = "0.3.0"/' "$r/repo/ci/publish-rehearsal/Cargo.toml" && G -C "$r/repo" commit -q -am badcrate; fi
  head0=$(G -C "$r/repo" rev-parse HEAD); tags0=$(G -C "$r/repo" tag | sort | tr '\n' ' ')
  ( cd "$r/repo" && env -u GH_TOKEN -u GITHUB_TOKEN "${envs[@]}" PATH="$r/bin:$PATH" bash scripts/release/publish_rehearsal.sh "$run" "$dry" ) > "$r/out" 2>&1; rc=$?
  [ -z "${DBG:-}" ] || cat "$r/out"
  [ "$rc" = "$wrc" ] || { fail "$name" "rc=$rc want $wrc: $(tail -n 1 "$r/out")"; return; }
  grep -q -E -e "$want" "$r/out" || { fail "$name" "output lacks /$want/: $(tail -n 1 "$r/out")"; return; }
  ! grep -q -F -e "$MARK" "$r/out" || { fail "$name" "the token value is in the output"; return; }
  [ -z "$(ls -A "$r/tmp")" ] || { fail "$name" "scratch dir left behind"; return; }
  [ -z "$(G -C "$r/repo" status --porcelain)" ] || { fail "$name" "the committed tree was modified: $(G -C "$r/repo" status --porcelain | head -n 2 | tr '\n' ' ')"; return; }
  [ "$(G -C "$r/repo" rev-parse HEAD)" = "$head0" ] && [ "$(G -C "$r/repo" tag | sort | tr '\n' ' ')" = "$tags0" ] || { fail "$name" "HEAD or a tag moved"; return; }
  [ ! -e "$r/net.calls" ] || { fail "$name" "a network tool was called: $(head -n 1 "$r/net.calls")"; return; }
  [ ! -e "$FX/toplevel-ran" ] || { fail "$name" "the cascade's top level ran"; return; }
  case "$wgate" in
    0) [ ! -e "$r/gate.calls" ] || { fail "$name" "the gate was called"; return; } ;;
    1) grep -q -E -e "^ROOT=.*/repo TAG=v1\.2\.3 token=$" "$r/gate.calls" 2>/dev/null \
         || { fail "$name" "the gate was not asked for ROOT and the newest release tag without the token: $(cat "$r/gate.calls" 2>/dev/null)"; return; } ;;
  esac
  if [ "$wpub" = 0 ]; then
    [ ! -e "$r/publish.calls" ] || { fail "$name" "publish was called"; return; }
  else
    { grep -q -F -x -e "version = \"$ver\"" "$r/publish.calls" && grep -q -F -x -e "token_ok=1" "$r/publish.calls" \
      && ! grep -q -F -x -e "in_tree=1" "$r/publish.calls"; } 2>/dev/null \
      || { fail "$name" "publish was not run from a scratch copy at $ver with the token in its environment: $(tr '\n' ' ' < "$r/publish.calls" 2>/dev/null)"; return; }
  fi
  idxn=0; [ ! -e "$r/index.calls" ] || idxn=$(wc -l < "$r/index.calls")
  [ -z "${IDXN:-}" ] || [ "$idxn" = "$IDXN" ] || { fail "$name" "index read $idxn time(s), want $IDXN"; return; }
  [ -z "${IDXN:-}" ] || [ "$IDXN" = 0 ] || [ "$(head -n 1 "$r/index.calls")" = "aprender-publish-rehearsal $ver" ] || { fail "$name" "index asked for '$(head -n 1 "$r/index.calls")'"; return; }
  echo "ok   $name"
}

OKRE='REHEARSED aprender-publish-rehearsal 0\.0\.42: clean-room read ok, uploaded, on the index'
row green_rehearses_the_throwaway_crate               0 "$OKRE" 1 1 0
IDXN=1 row green_reads_the_index_once_it_shows        0 "$OKRE" 1 1 0
INDEX_AFTER=3 IDXN=3 row index_lag_is_retried         0 "$OKRE" 1 1 0
RUN=7 row green_names_the_run_number                  0 'REHEARSED aprender-publish-rehearsal 0\.0\.7:' 1 1 0
for bad in '' 0 01 -1 1.2 abc '4 2' '42x' ' 42' $'42\n' 0x2a; do
  RUN="$bad" row "bad_run_number_refuses[$(printf '%q' "$bad")]" 2 'STOP bad run-number' 0 0 0
done
row bad_dry_run_flag_refuses                          2 'STOP bad dry_run' 0 0 2
TPL=tpl-untagged row no_release_tag_refuses           1 'STOP clean-room read: no release tag vX\.Y\.Z' 0 0 0
GATERC=1 row clean_room_read_refused                  1 'STOP clean-room read: gate refused v1\.2\.3: CLEAN-ROOM REFUSE: fixture says no for v1\.2\.3' 1 0 0
TOKEN=UNSET row unset_token_refuses                   1 'STOP upload: CARGO_REGISTRY_TOKEN is empty' 1 0 0
TOKEN=EMPTY row empty_token_refuses                     1 'STOP upload: CARGO_REGISTRY_TOKEN is empty' 1 0 0
BADCRATE=1 row placeholder_version_missing_refuses    1 'STOP upload: the committed Cargo\.toml has no version = "0\.0\.0" placeholder' 1 0 0
PUBRC=1 PUBOUT='error: 403 forbidden' row upload_refused  1 'STOP upload: refused or failed' 1 1 0
PUBRC=1 PUBOUT="error: bad token $MARK" row upload_refusal_output_is_scrubbed 1 'upload output: .*bad token \*\*\*' 1 1 0
PUBRC=101 PUBOUT='error: crate aprender-publish-rehearsal@0.0.42 already exists on crates.io index' \
  row already_on_the_index_is_green                   0 'step c upload ok: .* was already uploaded' 1 1 0
PUBRC=101 PUBOUT='error: crate version `0.0.42` is already uploaded' \
  row already_uploaded_is_green                       0 'step c upload ok: .* was already uploaded' 1 1 0
PUBRC=101 PUBOUT='error: crate aprender-publish-rehearsal@0.0.7 already exists on crates.io index' \
  row already_uploaded_other_version_is_red           1 'STOP upload: refused or failed' 1 1 0
PUBRC=101 PUBOUT='error: crate aprender-publish-rehearsal@0.0.421 already exists on crates.io index' \
  row already_uploaded_longer_version_is_red          1 'STOP upload: refused or failed' 1 1 0
INDEX_AFTER=0 IDXN=3 row index_never_shows_version    1 'STOP index read-back: aprender-publish-rehearsal 0\.0\.42 is not on the sparse index after 3 reads' 1 1 0
row dry_run_uploads_nothing                           0 'WOULD publish aprender-publish-rehearsal 0\.0\.42' 1 0 1
IDXN=0 row dry_run_reads_no_index                     0 'WOULD publish' 1 0 1
GATERC=1 row dry_run_still_asks_the_gate              1 'STOP clean-room read' 1 0 1
TOKEN=UNSET row dry_run_needs_no_token                     0 'WOULD publish' 1 0 1
# the gate is loaded from cascade-publish.sh at run time (one copy of the function), never by running the file
USEGATE=0 CASCADE_FX="$FX/ok.sh" row loaded_gate_is_asked_for_root_and_tag \
                                                      0 'gate: ARGS root=.*/repo tag=v1\.2\.3 n=2' x 1 0
USEGATE=0 CASCADE_FX="$FX/refuse.sh" row loaded_gate_refusal_is_red \
                                                      1 'STOP clean-room read: gate refused v1\.2\.3: CLEAN-ROOM REFUSE: fixture says no' x 0 0
USEGATE=0 CASCADE_FX="$FX/missing.sh" row missing_gate_function_refuses \
                                                      1 'STOP clean-room read: gate refused v1\.2\.3: cascade-publish\.sh does not define clean_room_tested_shas' x 0 0
USEGATE=0 CASCADE_FX="$FX/nowhere.sh" row absent_cascade_file_refuses \
                                                      1 'STOP clean-room read: gate refused v1\.2\.3: cannot read ' x 0 0
USEGATE=0 CASCADE_FX="$CASCADE_REAL" GHRC=1 row real_gate_refuses_when_gh_is_logged_out \
                                                      1 'STOP clean-room read: gate refused v1\.2\.3: CLEAN-ROOM REFUSE: looked for [0-9a-f]{40} \(tag v1\.2\.3\); gh is unauthenticated' x 0 0
# the script itself holds no ref-moving, tagging, forcing or hook-skipping command
if grep -n -E 'git (push|commit)|--force|gh (release|pr|issue)|--allow-dirty|--no-verify|set -x|xtrace' "$SRC" | grep -v -E '^[0-9]+:\s*#'; then
  fail source_has_no_push_commit_force_or_trace "see above"; else echo "ok   source_has_no_push_commit_force_or_trace"; fi

if [ -z "${PUB_REH_SCRIPT_ROWS_ONLY:-}" ]; then   # rows about the crate itself, not the script: the mutant runs skip them
  if ! command -v cargo > /dev/null 2>&1; then fail crate_rows "cargo not found: these two rows need the build host"; else
    # not a workspace member: the root workspace does not list it, and it is its own root
    before=$(git -C "$ROOT" status --porcelain -- ci/publish-rehearsal 2>/dev/null | sort)
    ( cd "$ROOT" && cargo metadata --offline --no-deps --format-version 1 ) > "$T/root.meta" 2> "$T/root.err"; rrc=$?
    ( cd "$ROOT" && cargo metadata --offline --no-deps --format-version 1 --manifest-path ci/publish-rehearsal/Cargo.toml ) > "$T/crate.meta" 2> "$T/crate.err"; crc=$?
    after=$(git -C "$ROOT" status --porcelain -- ci/publish-rehearsal 2>/dev/null | sort)
    if [ "$rrc" != 0 ] || ! grep -q -F '"name":"aprender-core"' "$T/root.meta"; then fail crate_is_not_a_workspace_member "root cargo metadata failed or lists no aprender-core (rc=$rrc): $(tail -n 1 "$T/root.err")"
    elif grep -q -F '"name":"aprender-publish-rehearsal"' "$T/root.meta"; then fail crate_is_not_a_workspace_member "the root workspace lists aprender-publish-rehearsal"
    elif [ "$crc" != 0 ] || ! grep -q -F '"name":"aprender-publish-rehearsal"' "$T/crate.meta"; then fail crate_is_not_a_workspace_member "the crate is not its own workspace root (rc=$crc): $(tail -n 1 "$T/crate.err")"
    elif [ "$before" != "$after" ]; then fail crate_is_not_a_workspace_member "reading the crate wrote into the tree"
    else echo "ok   crate_is_not_a_workspace_member"; fi
    # tests and packages offline from a scratch copy at a run version (no dependencies, no network)
    C="$T/crate"; mkdir -p "$C" && cp -R "$CRATE_SRC/." "$C/" && rm -rf -- "${C:?}/target" "${C:?}/Cargo.lock"
    sed -i 's/^version = "0\.0\.0"$/version = "0.0.42"/' "$C/Cargo.toml"
    if ! ( cd "$C" && export CARGO_TARGET_DIR="$C/target" && cargo generate-lockfile --offline && cargo test --offline --locked && cargo package --offline --locked ) > "$T/crate.log" 2>&1; then
      fail crate_tests_and_packages_offline "cargo failed: $(tail -n 3 "$T/crate.log" | tr '\n' ' ')"
    elif ! tar -tzf "$C/target/package/aprender-publish-rehearsal-0.0.42.crate" | grep -q 'src/lib\.rs$'; then fail crate_tests_and_packages_offline "the package has no src/lib.rs"
    elif [ "$(grep -c -E '^(description|license|repository) = ' "$CRATE_SRC/Cargo.toml")" != 3 ]; then fail crate_tests_and_packages_offline "description, license or repository missing"
    elif ! grep -q -F -x -e 'version = "0.0.0"' "$CRATE_SRC/Cargo.toml"; then fail crate_tests_and_packages_offline "the committed version is not the 0.0.0 placeholder"
    else echo "ok   crate_tests_and_packages_offline"; fi
  fi
fi
[ "$fails" = 0 ] && { echo "all rows held ($n table rows)"; exit 0; }
echo "$fails row(s) failed"; exit 1
