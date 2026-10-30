#!/usr/bin/env bash
# falsify_tag_publish_andon.sh -- the case table and mutants for tag_publish_andon.sh + lib_crates_io.sh (PMAT-5306)
#
#   bash scripts/release/falsify_tag_publish_andon.sh            # every row (~seconds)
#   bash scripts/release/falsify_tag_publish_andon.sh --mutants  # the table, then one mutant per check: each must turn it red
#
# Operator ruling 2026-10-30: "a release = the version is on crates.io and passed the clean-container smoke. A tag alone is not a
# release. A tag not on crates.io within 2 h of push is an andon that pages Noah live. The post-publish step closes the
# milestone/epic ONLY after crates.io shows the version. Fixture: tag pushed, publish fails -> milestone stays open, andon fires."
# The scripts under test run from SANDBOX COPIES with a .crates-io-test-seams file beside the lib; crates.io, gh and curl are
# stubs. Nothing real is touched. Exit 0 = every row held (and every mutant caught), 1 = not.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${ANDON_SRC:-$HERE/tag_publish_andon.sh}"
LIB="${ANDON_LIB:-$HERE/lib_crates_io.sh}"
if [ "${1:-}" = --mutants ]; then
    T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
    bash "$0" > "$T/base.log" 2>&1 || { tail -n 5 "$T/base.log"; echo "mutants: the unmutated table is not green"; exit 1; }
    miss=0 n=0
    mut() { # name file sed-expr
        local f=$HERE/$2 out=$T/m-$2
        n=$((n + 1)); sed "$3" "$f" > "$out"
        if cmp -s "$f" "$out"; then printf 'MUTANT %s did not apply\n' "$1" >&2; miss=$((miss + 1)); return; fi
        if [ "$2" = lib_crates_io.sh ]; then ANDON_LIB="$out" bash "$0" > "$T/m.log" 2>&1; else ANDON_SRC="$out" bash "$0" > "$T/m.log" 2>&1; fi
        if [ $? = 0 ]; then printf 'MUTANT %s SURVIVED\n' "$1" >&2; miss=$((miss + 1)); fi
    }
    mut window-dropped tag_publish_andon.sh 's/\[ "\$age" -ge "\$win" \] || continue/true/'
    mut since-dropped tag_publish_andon.sh 's/\[ "\$ts" -ge "\$since_s" \] || continue/true/'
    mut present-still-andon tag_publish_andon.sh 's/^        0) continue ;;$/        0) ;;/'
    mut not-measured-is-absent tag_publish_andon.sh 's/^        \*) nm=1;.*$/        *) ;;/'
    mut page-every-run tag_publish_andon.sh 's/if \[ ! -e "\$state\/paged-\$tag" \]; then/if true; then/'
    mut page-stamp-dropped tag_publish_andon.sh 's/^        : > "\$state\/paged-\$tag"$/        :/'
    mut escalate-dropped tag_publish_andon.sh '/>> "\$esc" ||/d'
    mut inbox-line-dropped tag_publish_andon.sh 's/^    inbox_line "\$line"$/    :/'
    mut exit-1-dropped tag_publish_andon.sh 's/^\[ "\$open" = 1 \] && exit 1$/[ "$open" = 1 ] \&\& exit 0/'
    mut exit-2-dropped tag_publish_andon.sh 's/^\[ "\$nm" = 1 \] && exit 2$/:/'
    mut page-sanitizer-dropped tag_publish_andon.sh 's/^page_text() .*$/page_text() { printf "%s" "$1"; }/'
    mut prerelease-tag-allowed tag_publish_andon.sh 's/^    \[\[ \$tag =~ \^v.*$/    :/'
    mut step-not-named tag_publish_andon.sh 's/; blocking step: \$step//'
    mut bad-body-is-empty-list lib_crates_io.sh '/type == "array"/d; s/\.versions\[\] | select/.versions[]? | select/'
    mut seam-without-file lib_crates_io.sh 's/if \[ -f "\$_CIO_HERE\/.crates-io-test-seams" \] \&\& \[ -n/if [ -n/'
    mut gate-absent-passes lib_crates_io.sh 's/^        1) printf .*$/        1) return 0 ;;/'
    mut gate-unmeasured-passes lib_crates_io.sh 's/^        \*) printf .*$/        *) return 0 ;;/'
    mut has-version-substring lib_crates_io.sh 's/grep -qxF -- "\$v"/grep -qF -- "$v"/'
    mut yanked-counts lib_crates_io.sh 's/select(.yanked != true) | //'
    printf 'falsify_tag_publish_andon --mutants: %d mutants, %d not caught\n' "$n" "$miss"
    [ "$miss" = 0 ]; exit
fi

TMP=$(mktemp -d) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); printf 'FAIL: %s\n' "$1" >&2; }
expect() { if [ "$2" = "$3" ]; then ok; else bad "$1: want $2, got $3"; fi; }
has() { if grep -qF -- "$3" "$2"; then ok; else bad "$1: lacks '$3'"; fi; }
hasnt() { if grep -qF -- "$3" "$2"; then bad "$1: has '$3'"; else ok; fi; }
lines() { wc -l < "$1" | tr -d ' '; }

BIN="$TMP/bin"; mkdir -p "$BIN" "$TMP/stub"
cp "$SRC" "$BIN/tag_publish_andon.sh"; cp "$LIB" "$BIN/lib_crates_io.sh"; : > "$BIN/.crates-io-test-seams"
# production-shaped copy: the lib WITHOUT a seams file
mkdir -p "$TMP/prod"; cp "$SRC" "$TMP/prod/tag_publish_andon.sh"; cp "$LIB" "$TMP/prod/lib_crates_io.sh"
# gh stub: logs every call; a GitHub release exists only when FX_REL=1. curl stub: always a transport failure (no network, ever)
cat > "$TMP/stub/gh" <<'SH'
#!/usr/bin/env bash
echo "gh $*" >> "$FX_GH_LOG"
case "$*" in "release view"*) [ "${FX_REL:-0}" = 1 ] ;; *) exit 1 ;; esac
SH
printf '#!/usr/bin/env bash\necho "curl $*" >> "$FX_GH_LOG"\nexit 22\n' > "$TMP/stub/curl"
chmod +x "$TMP/stub/gh" "$TMP/stub/curl"
export FX_GH_LOG="$TMP/gh.log"; : > "$FX_GH_LOG"

NOW=$(date -u +%s) # bashrs disable-line=DET002
R="$TMP/repo"; git init -q "$R"
g() { git -C "$R" -c user.email=f@x -c user.name=f -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
printf x > "$R/f"; g add f; g commit -q -m c0
mktag() { # name age_min
    GIT_COMMITTER_DATE="@$((NOW - $2 * 60)) +0000" g tag -a -m "$1" "$1"
}
cj() { printf '{"versions":[%s]}\n' "$2" > "$1"; }
J_NONE="$TMP/none.json"; cj "$J_NONE" '{"num":"0.72.1","yanked":false}'
J_HAS="$TMP/has.json"; cj "$J_HAS" '{"num":"0.72.1","yanked":false},{"num":"1.0.0","yanked":false}'
J_YANK="$TMP/yank.json"; cj "$J_YANK" '{"num":"1.0.0","yanked":true}'
J_BAD="$TMP/bad.json"; printf '{"errors":[{"detail":"rate limited"}]}\n' > "$J_BAD"

# andon NAME JSON [args...] -> rc in $RC; out in $TMP/out-NAME; inbox $TMP/inb-NAME; escalate $TMP/esc-NAME; state $TMP/st-NAME
andon() {
    local n=$1 j=$2; shift 2
    CRATES_IO_JSON_FILE="$j" PATH="$TMP/stub:$PATH" bash "${ANDON_BIN:-$BIN/tag_publish_andon.sh}" --repo "$R" --state "$TMP/st-$n" \
        --inbox "$TMP/inb-$n" --escalate "$TMP/esc-$n" --page-dry "$@" > "$TMP/out-$n" 2>&1
    RC=$?
    touch "$TMP/inb-$n" "$TMP/esc-$n"
}

# ---- rows ----
mktag v1.0.0 180
andon old-absent "$J_NONE"; expect "3h old + absent: rc" 1 "$RC"
has "3h old + absent: inbox" "$TMP/inb-old-absent" "tag-andon"; has "inbox names the tag" "$TMP/inb-old-absent" "v1.0.0"
expect "escalate lines" 1 "$(lines "$TMP/esc-old-absent")"
has "page printed to a3" "$TMP/out-old-absent" "PAGE apr:aprender-a3: OPERATOR-FACING ESCALATION"
has "page printed to cop" "$TMP/out-old-absent" "PAGE apr:cop: OPERATOR-FACING ESCALATION"
has "page names the age" "$TMP/out-old-absent" "pushed 180 min ago"
has "page names the blocking step" "$TMP/out-old-absent" "blocking step: no GitHub release"
FX_REL=1 andon rel-exists "$J_NONE"
has "blocking step moves on when the release exists" "$TMP/out-rel-exists" "crates.io publish not started or incomplete"
# second run: no second page, no second escalate line, still an inbox line
andon old-absent "$J_NONE"; expect "second run rc" 1 "$RC"
expect "second run: no second escalate" 1 "$(lines "$TMP/esc-old-absent")"
expect "second run: inbox line again" 2 "$(grep -c 'tag-andon' "$TMP/inb-old-absent")"
expect "second run: no second page" 0 "$(grep -c "^PAGE" "$TMP/out-old-absent")"
expect "second run: one page per target (state stamp)" 1 "$(ls "$TMP/st-old-absent" | grep -c '^paged-v1.0.0$')"
# version present: silent, exit 0
andon present "$J_HAS"; expect "present rc" 0 "$RC"
expect "present: no inbox" 0 "$(lines "$TMP/inb-present")"; expect "present: no escalate" 0 "$(lines "$TMP/esc-present")"; hasnt "present: no page" "$TMP/out-present" "PAGE"
# yanked is not published
andon yanked "$J_YANK"; expect "yanked version is still an andon" 1 "$RC"
# a longer version string never satisfies a shorter one (1.0.0 vs 1.0.01 / 11.0.0)
cj "$TMP/sub.json" '{"num":"11.0.0","yanked":false}'; andon substring "$TMP/sub.json"; expect "11.0.0 does not satisfy v1.0.0" 1 "$RC"
# crates.io read fails: exit 2, no page, an inbox line saying so
andon bad-body "$J_BAD"; expect "garbage body rc" 2 "$RC"; hasnt "garbage body: no page" "$TMP/out-bad-body" "PAGE"
has "garbage body: NOT_MEASURED inbox" "$TMP/inb-bad-body" "NOT_MEASURED"; expect "garbage body: no escalate" 0 "$(lines "$TMP/esc-bad-body")"
andon missing-file "$TMP/nonexistent.json"; expect "unreadable source rc" 2 "$RC"; hasnt "unreadable: no page" "$TMP/out-missing-file" "PAGE"
# a NOT_MEASURED does not stamp the tag: the next good read still pages
andon bad-then-good "$J_BAD"; andon bad-then-good "$J_NONE"; expect "page after a not-measured run" 1 "$RC"
has "page after a not-measured run" "$TMP/out-bad-then-good" "PAGE apr:aprender-a3"
# window: a 1 h old tag is nothing
mktag v1.1.0 60
git -C "$R" tag -d v1.0.0 > /dev/null; andon young "$J_NONE"; expect "1h old rc" 0 "$RC"; expect "1h old: no inbox" 0 "$(lines "$TMP/inb-young")"; hasnt "1h old: no page" "$TMP/out-young" "PAGE"
andon young-narrow "$J_NONE" --window-min 30; expect "1h old with a 30 min window" 1 "$RC"
# --since: an old tag before the cutoff is ignored (the historical eight)
mktag v1.0.0 180
andon since "$J_NONE" --since "$(date -u -d "@$((NOW - 3600))" +%FT%TZ)"; expect "older than --since rc" 0 "$RC"; expect "older than --since: no inbox" 0 "$(lines "$TMP/inb-since")"
andon default-since "$J_NONE"; expect "a 2026-era tag is inside the default --since" 1 "$RC"
# pre-release and non-version tags are not releases
git -C "$R" tag -d v1.0.0 v1.1.0 > /dev/null; mktag v2.0.0-rc.1 300; mktag nightly-x 300; mktag v3.0 300
andon prerel "$J_NONE"; expect "pre-release / odd tags rc" 0 "$RC"; expect "pre-release / odd tags: no inbox" 0 "$(lines "$TMP/inb-prerel")"
git -C "$R" tag -d v2.0.0-rc.1 nightly-x v3.0 > /dev/null
# usage errors
bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-u" --inbox "$TMP/i" > /dev/null 2>&1; expect "no --escalate" 2 "$?"
bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-u" --inbox "$TMP/i" --escalate "$TMP/e" --since garbage > /dev/null 2>&1; expect "bad --since" 2 "$?"
# the seam is not honoured without its file (production copy); curl is stubbed to fail, so honouring it would read the fixture and pass
mktag v1.0.0 180
ANDON_BIN="$TMP/prod/tag_publish_andon.sh" andon prod-seam "$J_HAS"; expect "production copy ignores CRATES_IO_JSON_FILE (curl fails -> not measured)" 2 "$RC"
has "production copy tried the network" "$FX_GH_LOG" "curl"
git -C "$R" tag -d v1.0.0 > /dev/null
# lock: a second run while the first holds it does nothing
mktag v1.0.0 180; mkdir -p "$TMP/st-lock"; : > "$FX_GH_LOG"
( exec 8> "$TMP/st-lock/.lock"; flock -n 8; CRATES_IO_JSON_FILE="$J_NONE" PATH="$TMP/stub:$PATH" bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-lock" --inbox "$TMP/inb-lock" --escalate "$TMP/esc-lock" --page-dry > "$TMP/out-lock" 2>&1; echo $? > "$TMP/lock-rc" )
expect "lock held: rc" 0 "$(cat "$TMP/lock-rc")"; has "lock held: says so" "$TMP/out-lock" "holds the lock"; hasnt "lock held: no page" "$TMP/out-lock" "PAGE"

# ---- library rows ----
eval "$(grep "^page_text()" "$BIN/tag_publish_andon.sh")"
expect "page_text strips quotes, backticks, dollar, bang, backslash" "abcdef" "$(page_text "a'b\`c\$d!e\\f")"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_HAS" crates_io_has_version aprender 1.0.0 ); expect "has_version present" 0 "$?"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_HAS" crates_io_has_version aprender 9.9.9 ); expect "has_version absent" 1 "$?"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_BAD" crates_io_has_version aprender 1.0.0 ); expect "has_version not measured" 2 "$?"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_BAD" crates_io_versions aprender > "$TMP/v.out" 2> /dev/null ); expect "versions of a bad body fails" 2 "$?"
expect "versions of a bad body prints nothing" 0 "$(lines "$TMP/v.out")"
( . "$BIN/lib_crates_io.sh" && crates_io_versions 'a b' ) > /dev/null 2>&1; expect "bad crate name" 2 "$?"
# option neutrality: sourcing leaves the caller's options alone
( set +e +u; . "$BIN/lib_crates_io.sh"; case "$-" in *e* | *u*) exit 1 ;; esac ); expect "sourcing leaves errexit/nounset alone" 0 "$?"

# ---- THE OPERATOR FIXTURE: tag pushed, publish fails -> the close gate refuses (milestone stays open) and the andon fires ----
: > "$FX_GH_LOG"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_NONE" crates_io_close_gate aprender 1.0.0 ) > "$TMP/gate.out" 2>&1; expect "operator fixture: close gate refuses" 1 "$?"
has "operator fixture: gate says why" "$TMP/gate.out" "crates.io does not show 1.0.0: milestone and epic stay open"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_BAD" crates_io_close_gate aprender 1.0.0 ) > "$TMP/gate2.out" 2>&1; expect "operator fixture: unmeasured read also refuses" 2 "$?"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_JSON_FILE="$J_HAS" crates_io_close_gate aprender 1.0.0 ) > /dev/null 2>&1; expect "close gate opens once crates.io shows it" 0 "$?"
andon operator "$J_NONE"; expect "operator fixture: andon fires" 1 "$RC"; has "operator fixture: paged" "$TMP/out-operator" "PAGE apr:aprender-a3"
hasnt "operator fixture: no milestone/epic write was attempted" "$FX_GH_LOG" "close"
hasnt "operator fixture: no PATCH was attempted" "$FX_GH_LOG" "PATCH"
# the autopilot calls the gate in step 9, before any close
grep -q 'crates_io_close_gate aprender "\$V"' "$HERE/autopilot.sh" && ok || bad "autopilot.sh step 9 does not call crates_io_close_gate"

printf 'falsify_tag_publish_andon: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]
