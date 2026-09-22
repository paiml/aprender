#!/usr/bin/env bash
# check_dogfood_version_row.sh -- the dogfood's version row is PHASE-AWARE, and the banner names the
# phase that ran (#3543).
#
# THE DEFECT, measured on the 0.68.2 train (2026-09-20, right after a 74/74 cascade):
#   `scripts/dogfood.sh --phase post-publish` printed
#     [FAIL] version-unpublished   aprender 0.68.2 is ALREADY on crates.io — bump the version
#     ══ dogfood pre-release: aprender v0.68.2 ══
#   The row is a PRE-publish gate; a successful publish turns it RED forever, so the post-publish
#   dogfood could never say GO, and the DEFERs it exists to discharge never were. The banner lied
#   about which phase ran.
#
# WHAT THIS RUNS. The REAL "2 + 10. version + publish dry-run" section of scripts/dogfood.sh --
# extracted from its section header to the next one, so the call sites run, not just the helpers
# -- with stubs for `mark` (records rows), the publish dry-run (the `cargo` name on PATH) and `curl`
# (serves a fixture crates.io sparse index: present / absent / 404 / an HTML body / down). Plus
# the phase_banner line.
#
# THE ROWS (#3543's falsifier, both directions, both phases)
#   pre-present     pre-publish, version in the index      -> version-unpublished FAIL
#   pre-absent      pre-publish, index lacks the version   -> version-unpublished PASS
#   pre-404         pre-publish, crate not in the index    -> version-unpublished PASS
#   pre-down        pre-publish, index unreachable         -> version-unpublished FAIL, UNKNOWN
#   post-present    post-publish, version in the index AND it installs from crates.io
#                   (`--version =9.9.9 --locked`, its binary reports 9.9.9) -> version-published PASS,
#                   and NO version-unpublished row (it is a pre-publish gate)
#   post-install-fails   in the index, but the install of it fails -> version-published FAIL
#   post-install-wrongver  it installs, but the binary reports another version -> FAIL
#   post-install-nobin     it "installs" but no binary lands -> FAIL
#   post-absent     post-publish, index lacks the version  -> version-published FAIL, and no install is
#                   attempted
#   post-404        post-publish, crate not in the index   -> version-published FAIL
#   post-html       post-publish, a 200 that is not the index -> version-published FAIL, UNKNOWN
#   post-dry-run    post-publish still runs the publish dry-run: row 10 reads its exit code
#   full-unchanged  full phase: the dry-run path as before (already exists -> FAIL), no index call
#   banner          post-publish prints "══ dogfood post-publish: aprender v9.9.9 ══"
#   index-path      crate "Aprender" is looked up at index.crates.io/ap/re/aprender (lowercased)
# THE MUTANTS -- one per row; each must turn its row RED. See the `mutant` lines at the bottom.
#
# Exit 0 = every row green and every mutant killed. 1 = a row RED or a mutant survived.
# 2 = ENV: the subject or a mutation anchor is missing -- the table judged nothing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="$ROOT/scripts/dogfood.sh"
SEC_START='# ── 2 + 10. version + publish dry-run'
SEC_END='# ── 3. changelog mentions the version'

env_die() { printf 'ENV   %s -- the table judged nothing, not a pass\n' "$*" >&2; exit 2; }
[ -r "$SUBJECT" ] || env_die "no $SUBJECT"
for t in python3 sed; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done
for a in "$SEC_START" "$SEC_END" 'phase_banner() {' 'strip_ansi() {'; do
    grep -qF -- "$a" "$SUBJECT" || env_die "dogfood.sh has no '$a' -- the subject moved"
done

TMP=$(mktemp -d) || exit 2
# SEC011: validate before rm -rf. An empty or '/' value must never reach it.
cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
trap cleanup EXIT

mkdir -p "$TMP/bin"
# curl: -o FILE -w '%{http_code}' ... URL. FX_INDEX picks what the fixture index answers.
cat > "$TMP/bin/curl" <<'STUB'
#!/usr/bin/env bash
out=""; url=""
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift 2 ;; -w|-A) shift 2 ;; -*) shift ;; *) url=$1; shift ;; esac; done
printf 'curl %s\n' "$url" >> "$FX_LOG"
case "${FX_INDEX:-}" in
  present) printf '{"name":"aprender","vers":"9.9.8"}\n{"name":"aprender","vers":"9.9.9"}\n' > "$out"; printf 200 ;;
  absent)  printf '{"name":"aprender","vers":"9.9.8"}\n' > "$out"; printf 200 ;;
  404)     : > "$out"; printf 404 ;;
  html)    printf '<html><body>captive portal</body></html>\n' > "$out"; printf 200 ;;
  down)    echo "curl: (6) Could not resolve host: index.crates.io" >&2; exit 6 ;;
  *)       exit 99 ;;
esac
STUB
# the publish dry-run: FX_DRY=already prints the already-exists warning; exit FX_DRC (default 0).
# install: FX_INSTALL = ok (default) | fail | wrongver | nobin; the binary lands under --root
cat > "$TMP/bin/cargo" <<'STUB'
#!/usr/bin/env bash
if [ "${1:-}" = install ]; then
  printf 'install %s\n' "$*" >> "$FX_LOG"
  root=""; prev=""; for a in "$@"; do [ "$prev" = --root ] && root=$a; prev=$a; done
  case "${FX_INSTALL:-ok}" in
    fail)  echo "error: failed to compile \`aprender v9.9.9\`, intermediate artifacts can be found at ..."; exit 101 ;;
    nobin) exit 0 ;;
  esac
  v=9.9.9; [ "${FX_INSTALL:-ok}" = wrongver ] && v=9.9.8
  mkdir -p "$root/bin" && printf '#!/usr/bin/env bash\necho "apr %s (fixture)"\n' "$v" > "$root/bin/apr" && chmod +x "$root/bin/apr"
  exit 0
fi
printf 'dry-run %s\n' "$*" >> "$FX_LOG"
[ "${FX_DRY:-}" = already ] && echo "warning: crate aprender@9.9.9 already exists on crates.io index"
exit "${FX_DRC:-0}"
STUB
chmod +x "$TMP/bin/curl" "$TMP/bin/cargo"

# run_section SRC NAME PHASE [ENV=VAL ...] -> rows in $TMP/NAME/rows, calls in $TMP/NAME/log, DRC in .../drc
run_section() {
    local src=$1 d="$TMP/$2" phase=$3; shift 3
    mkdir -p "$d/work" || return 2
    : > "$d/rows"; : > "$d/log"
    awk -v s="$SEC_START" -v e="$SEC_END" 'index($0, s) == 1 {f = 1} index($0, e) == 1 {f = 0} f' "$src" > "$d/section.sh"
    grep -E '^strip_ansi\(\) \{' "$src" > "$d/prelude.sh"
    [ -s "$d/section.sh" ] && [ -s "$d/prelude.sh" ] || return 2
    ( export PATH="$TMP/bin:$PATH" FX_LOG="$d/log" DOGFOOD_PHASE="$phase" VERSION=9.9.9 WORKLOG="$d/work"
      for kv in "$@"; do export "${kv?}"; done
      # AFTER the per-row exports: a crate name set before them never saw FX_CRATE (the index-path
      # row passed for the wrong reason until its mutant survived)
      export CRATE="${FX_CRATE:-aprender}"
      mark() { printf '%s %s %s\n' "$1" "$2" "$3" >> "$FX_ROWS"; }
      export FX_ROWS="$d/rows"
      . "$d/prelude.sh"; . "$d/section.sh"
      printf '%s\n' "${DRC:-unset}" > "$d/drc" ) > "$d/out" 2>&1
}
# expect NAME ROW STATUS NEEDLE -> 0 when the section marked ROW STATUS with a note holding NEEDLE
expect() {
    grep -qF -- "$2 $3 " "$TMP/$1/rows" || { printf 'no "%s %s" row: %s\n' "$2" "$3" "$(tr '\n' '|' < "$TMP/$1/rows")"; return 1; }
    [ -z "$4" ] || grep -qF -- "$4" <<< "$(grep -F -- "$2 $3 " "$TMP/$1/rows")" || { printf '"%s %s" never said "%s": %s\n' "$2" "$3" "$4" "$(grep -F "$2 " "$TMP/$1/rows")"; return 1; }
    return 0
}

# ---- row bodies: TAG SRC -> 0 green, 1 red (prints why), 2 fixture ------------------------------
pre_present()  { run_section "$2" "prep-$1" pre-publish FX_INDEX=present || return 2; expect "prep-$1" version-unpublished FAIL "ALREADY in the crates.io index"; }
pre_absent()   { run_section "$2" "prea-$1" pre-publish FX_INDEX=absent || return 2; expect "prea-$1" version-unpublished PASS "absent from the crates.io index"; }
pre_404()      { run_section "$2" "pre4-$1" pre-publish FX_INDEX=404 || return 2; expect "pre4-$1" version-unpublished PASS "HTTP 404"; }
pre_down()     { run_section "$2" "pred-$1" pre-publish FX_INDEX=down || return 2; expect "pred-$1" version-unpublished FAIL "UNKNOWN"; }
post_present() {
    run_section "$2" "posp-$1" post-publish FX_INDEX=present || return 2
    expect "posp-$1" version-published PASS "of it works: 1 binary(ies) installed from crates.io" || return 1
    grep -q -- '^install install aprender --version =9.9.9 --locked ' "$TMP/posp-$1/log" \
        || { printf 'not installed as a user would (--version =9.9.9 --locked): %s\n' "$(grep '^install ' "$TMP/posp-$1/log")"; return 1; }
    ! grep -q '^version-unpublished ' "$TMP/posp-$1/rows" || { printf 'post-publish still emitted the pre-publish row: %s\n' "$(grep '^version-unpublished ' "$TMP/posp-$1/rows")"; return 1; }
    return 0
}
post_absent()  {
    run_section "$2" "posa-$1" post-publish FX_INDEX=absent || return 2
    expect "posa-$1" version-published FAIL "is NOT in the crates.io index" || return 1
    ! grep -q '^install ' "$TMP/posa-$1/log" || { printf 'an install was attempted for a version the index does not carry\n'; return 1; }
    return 0
}
post_install_fails()   { run_section "$2" "posif-$1" post-publish FX_INDEX=present FX_INSTALL=fail || return 2; expect "posif-$1" version-published FAIL "aprender --version =9.9.9 --locked\` exited 101"; }
post_install_wrongver() { run_section "$2" "posiw-$1" post-publish FX_INDEX=present FX_INSTALL=wrongver || return 2; expect "posiw-$1" version-published FAIL "do not report 9.9.9"; }
post_install_nobin()   { run_section "$2" "posin-$1" post-publish FX_INDEX=present FX_INSTALL=nobin || return 2; expect "posin-$1" version-published FAIL "installed no binary"; }
post_404()     { run_section "$2" "pos4-$1" post-publish FX_INDEX=404 || return 2; expect "pos4-$1" version-published FAIL "HTTP 404"; }
post_html()    { run_section "$2" "posh-$1" post-publish FX_INDEX=html || return 2; expect "posh-$1" version-published FAIL "UNKNOWN"; }
post_dry_run() {
    run_section "$2" "posd-$1" post-publish FX_INDEX=present FX_DRY=already FX_DRC=0 || return 2
    grep -q '^dry-run publish --dry-run' "$TMP/posd-$1/log" || { printf 'post-publish no longer runs the publish dry-run (row 10 reads its exit code)\n'; return 1; }
    [ "$(cat "$TMP/posd-$1/drc")" = 0 ] || { printf 'DRC=%s after the post-publish section, want 0\n' "$(cat "$TMP/posd-$1/drc")"; return 1; }
    return 0
}
full_unchanged() {
    run_section "$2" "full-$1" full FX_INDEX=present FX_DRY=already || return 2
    expect "full-$1" version-unpublished FAIL "ALREADY on crates.io" || return 1
    ! grep -q '^curl ' "$TMP/full-$1/log" || { printf 'the full phase consulted the index\n'; return 1; }
    return 0
}
banner() {
    local got
    grep -E '^phase_banner\(\) \{' "$2" > "$TMP/banner-$1.sh" || return 2
    got=$(. "$TMP/banner-$1.sh"; phase_banner post-publish aprender 9.9.9)
    [ "$got" = "══ dogfood post-publish: aprender v9.9.9 ══" ] || { printf 'banner: %s\n' "$got"; return 1; }
    grep -qE '^phase_banner "\$DOGFOOD_PHASE" "\$CRATE" "\$VERSION"$' "$2" || { printf 'the banner is not printed from DOGFOOD_PHASE\n'; return 1; }
    return 0
}
index_path() {
    run_section "$2" "path-$1" post-publish FX_INDEX=present FX_CRATE=Aprender || return 2
    grep -qx 'curl https://index.crates.io/ap/re/aprender' "$TMP/path-$1/log" || { printf 'looked up: %s\n' "$(grep '^curl ' "$TMP/path-$1/log")"; return 1; }
    return 0
}

fails=0; rows=0
row() { # row NAME RC MESSAGE
    rows=$((rows + 1))
    if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"
    elif [ "$2" = 2 ]; then env_die "row $1 could not build its fixture"
    else printf 'FAIL  %s: %s\n' "$1" "$3" >&2; fails=$((fails + 1)); fi
}

for spec in "pre-present pre_present" "pre-absent pre_absent" "pre-404 pre_404" "pre-down pre_down" \
            "post-present post_present" "post-absent post_absent" "post-404 post_404" "post-html post_html" \
            "post-install-fails post_install_fails" "post-install-wrongver post_install_wrongver" "post-install-nobin post_install_nobin" \
            "post-dry-run post_dry_run" "full-unchanged full_unchanged" "banner banner" "index-path index_path"; do
    set -- $spec
    msg=$($2 real "$SUBJECT"); row "$1" "$?" "$msg"
done

# ---- the mutants ------------------------------------------------------------------------------
# mutant NAME ANCHOR REPLACEMENT ROWFN -- ANCHOR must occur exactly once in dogfood.sh
mutant() {
    local name=$1 m="$TMP/m-$1.sh" msg mrc
    python3 - "$SUBJECT" "$m" "$2" "$3" <<'PY' || env_die "$name mutant: anchor not found exactly once"
import sys
src, dst, anchor, repl = sys.argv[1:5]
s = open(src).read()
assert s.count(anchor) == 1, anchor
open(dst, "w").write(s.replace(anchor, repl, 1))
PY
    msg=$($4 "m-$name" "$m"); mrc=$?
    [ "$mrc" = 2 ] && env_die "$name mutant could not build its fixture"
    [ "$mrc" != 0 ]; row "mutant $name is killed by $4 (${msg:-survived})" "$?" "the mutant PASSED -- the row does not discriminate"
}
mutant pre-present-pass  "pre-publish:present)       printf 'version-unpublished FAIL" "pre-publish:present)       printf 'version-unpublished PASS" pre_present
mutant pre-absent-fail   "pre-publish:absent)        printf 'version-unpublished PASS" "pre-publish:absent)        printf 'version-unpublished FAIL" pre_absent
mutant pre-404-fail      "pre-publish:crate-absent)  printf 'version-unpublished PASS" "pre-publish:crate-absent)  printf 'version-unpublished FAIL" pre_404
mutant down-is-absent    "printf 'unknown index.crates.io not consulted (curl exit=%s): %s\\n'" "printf 'absent\\n'; : " pre_down
mutant post-phase-blind  "post-publish:present)      printf 'version-published PASS" "post-publish:present)      printf 'version-unpublished FAIL" post_present
mutant post-absent-pass  "post-publish:absent)       printf 'version-published FAIL" "post-publish:absent)       printf 'version-published PASS" post_absent
mutant post-404-pass     "post-publish:crate-absent) printf 'version-published FAIL" "post-publish:crate-absent) printf 'version-published PASS" post_404
mutant html-is-absent    "    *) printf 'unknown index.crates.io answered HTTP 200 but the body is not the index" "    *) printf 'absent\\n'; : printf 'unknown" post_html
mutant no-post-call      "  mark_version_row post-publish
" "" post_present
mutant post-skips-dry-run 'publish --dry-run --allow-dirty 2>&1); DRC=$?' 'publish --help > /dev/null 2>&1); DRY=""; DRC=0' post_dry_run
mutant full-uses-index   'if [ "$DOGFOOD_PHASE" = post-publish ]; then
  # #3543' 'if [ "$DOGFOOD_PHASE" != pre-publish ]; then
  # #3543' full_unchanged
mutant banner-fixed      "phase_banner() { printf '══ dogfood %s: %s v%s ══\\n' \"\$1\"" "phase_banner() { printf '══ dogfood pre-release: %s v%s ══\\n' \"\$2\"" banner
mutant index-case        "name=\$(printf '%s' \"\$1\" | tr '[:upper:]' '[:lower:]')" "name=\$1" index_path
mutant install-ignored   '      *)     st=FAIL; note="$CRATE $VERSION is in the crates.io index but is NOT installable' '      *)     note="$CRATE $VERSION is in the crates.io index but is NOT installable' post_install_fails
mutant version-unchecked '    case "$v" in *"$2"*) ;; *) bad="$bad $(basename "$b")='"'"'$v'"'"'" ;; esac' '    :' post_install_wrongver
mutant nobin-ok          '  [ "$n" -gt 0 ] || { printf' '  true || { printf' post_install_nobin
mutant install-on-absent '  if [ "$1" = post-publish ] && [ "$state" = present ]; then' '  if [ "$1" = post-publish ]; then' post_absent
mutant unlocked          'install "$1" --version "=$2" --locked --root' 'install "$1" --version "=$2" --root' post_present

# VACUITY FLOOR: a table that ran fewer rows than it declares is not a pass.
[ "$rows" -ge 33 ] || { printf 'VACUOUS %s row(s) ran, fewer than the 33 declared\n' "$rows" >&2; exit 1; }
[ "$fails" -eq 0 ] || { printf 'RED   %s of %s row(s) failed\n' "$fails" "$rows" >&2; exit 1; }
printf 'PASS  %s row(s): the dogfood version row is phase-aware and the banner names the phase (#3543)\n' "$rows"
