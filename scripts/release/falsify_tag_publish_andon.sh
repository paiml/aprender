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
    mut page-every-run tag_publish_andon.sh 's/if \[ ! -e "\$state\/\$pstamp\$tag" \]; then/if true; then/'
    mut page-stamp-dropped tag_publish_andon.sh 's/then : > "\$state\/\$pstamp\$tag"; else/then :; else/'
    mut escalate-dropped tag_publish_andon.sh 's/>> "\$esc" 2> \/dev\/null/> \/dev\/null/'
    mut inbox-line-dropped tag_publish_andon.sh 's/^    inbox_line "\$line"$/    :/'
    mut exit-1-dropped tag_publish_andon.sh 's/^\[ "\$open" = 1 \] && exit 1$/[ "$open" = 1 ] \&\& exit 0/'
    mut exit-2-dropped tag_publish_andon.sh 's/^\[ "\$nm" = 1 \] && exit 2$/:/'
    mut page-sanitizer-dropped tag_publish_andon.sh 's/^page_text() .*$/page_text() { printf "%s" "$1"; }/'
    mut prerelease-tag-allowed tag_publish_andon.sh 's/^    \[\[ \$tag =~ \^v.*$/    :/'
    mut step-not-named tag_publish_andon.sh 's/; blocking step: \$step//'
    mut stamp-without-delivery tag_publish_andon.sh 's/if page "\$line"; then : >/if page "$line" || true; then : >/'
    mut lightweight-by-commit-date tag_publish_andon.sh 's/    ts=$(tag_time "$tag")/    ts=$(G for-each-ref --format=%(creatordate:unix) "refs\/tags\/$tag")/'
    mut baseline-dropped tag_publish_andon.sh 's/^baseline=1; last=/baseline=0; last=/'
    mut gap-ignored tag_publish_andon.sh 's/-le \$((gap \* 60))/-ge 0/'
    mut dry-stamps-live tag_publish_andon.sh 's/|| pstamp=dry-paged-/|| pstamp=paged-/'
    mut undatable-is-silent tag_publish_andon.sh 's/^    \[\[ \$ts =~ \^\[0-9\]+\$ \]\] || {.*$/    [[ $ts =~ ^[0-9]+$ ]] || continue/'
    mut baseline-lightweight-zero tag_publish_andon.sh 's/G log -1 --format=%ct "refs\/tags\/$t" > "$seen"/echo 0 > "$seen"/'
    mut push-dated-by-tagger tag_publish_andon.sh 's/if \[ "\$baseline" = 1 \] && \[ "\$lw" = 0 \]; then/if [ "$lw" = 0 ]; then/'
    mut lsremote-fail-is-absent tag_publish_andon.sh 's/^            \*) printf .origin unreadable.*$/            *) printf "tag not on origin"; return ;;/'
    mut page-always-succeeds tag_publish_andon.sh 's/^    \[ "\$delivered" = 1 \]$/    true/'
    mut undelivered-inbox-dropped tag_publish_andon.sh 's/else inbox_line "PAGE UNDELIVERED[^"]*"; fi/else :; fi/'
    mut has-session-dropped tag_publish_andon.sh 's/^        tmux has-session .*$/        :/'
    mut page-target-not-required tag_publish_andon.sh 's/^\[ "\$dry" = 1 \] || \[ "\${#targets\[@\]}" -gt 0 \] || usage.*$/:/'
    mut escalate-every-run tag_publish_andon.sh 's/if \[ ! -e "\$state\/escalated-\$tag" \]; then/if true; then/'
    mut lock-exit-0 tag_publish_andon.sh 's/nothing done"; exit 3; }/nothing done"; exit 0; }/'
    mut http-500-is-absent lib_crates_io.sh 's/^        \*) return 2 ;;$/        *) return 1 ;;/'
    mut http-404-is-unmeasured lib_crates_io.sh 's/^        404) return 1 ;;$/        404) return 2 ;;/'
    mut num-mismatch-ignored lib_crates_io.sh 's/jq -e --arg v "\$v" .\.version\.num == \$v. > \/dev\/null 2>&1 || return 2/:/'
    mut yanked-counts lib_crates_io.sh 's/ || return 1$/ || return 0/'
    mut seam-without-file lib_crates_io.sh 's/if \[ -f "\$_CIO_HERE\/.crates-io-test-seams" \] \&\& \[ -n/if [ -n/'
    mut gate-absent-passes lib_crates_io.sh 's/^        1) printf .*$/        1) return 0 ;;/'
    mut gate-unmeasured-passes lib_crates_io.sh 's/^        \*) printf .*$/        *) return 0 ;;/'
    mut version-unvalidated lib_crates_io.sh 's/^    \[\[ \$v =~ .*$/    :/'
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
# gh stub: a GitHub release exists only when FX_REL=1. curl stub: always a transport failure (no network, ever).
# tmux stub: has-session fails for the session named in FX_BAD_SESSION (or all when it is "all"); send-keys fails when FX_SEND_RC != 0.
# date stub: the FAKE CLOCK -- `date -u +%s` answers FX_NOW; everything else is the real date.
cat > "$TMP/stub/gh" <<'SH'
#!/usr/bin/env bash
echo "gh $*" >> "$FX_GH_LOG"
case "$*" in "release view"*) [ "${FX_REL:-0}" = 1 ] ;; *) exit 1 ;; esac
SH
printf '#!/usr/bin/env bash\necho "curl $*" >> "$FX_GH_LOG"\nexit 22\n' > "$TMP/stub/curl"
cat > "$TMP/stub/tmux" <<'SH'
#!/usr/bin/env bash
echo "tmux $*" >> "$FX_TMUX_LOG"
case "$1" in
    has-session) [ "${FX_BAD_SESSION:-}" != all ] && [ "${FX_BAD_SESSION:-}" != "$3" ] ;;
    send-keys) exit "${FX_SEND_RC:-0}" ;;
    *) exit 1 ;;
esac
SH
cat > "$TMP/stub/date" <<'SH'
#!/usr/bin/env bash
if [ "$*" = "-u +%s" ]; then echo "$FX_NOW"; exit 0; fi
exec "$FX_REALDATE" "$@"
SH
chmod +x "$TMP/stub/gh" "$TMP/stub/curl" "$TMP/stub/tmux" "$TMP/stub/date"
export FX_GH_LOG="$TMP/gh.log" FX_TMUX_LOG="$TMP/tmux.log"; : > "$FX_GH_LOG"; : > "$FX_TMUX_LOG"
FX_REALDATE=$(command -v date); export FX_REALDATE
NOW=$("$FX_REALDATE" -u -d 2026-11-05T12:00:00Z +%s); export FX_NOW=$NOW   # the fake clock: never the wall clock

R="$TMP/repo"; git init -q "$R"
g() { git -C "$R" -c user.email=f@x -c user.name=f -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
printf x > "$R/f"; g add f; g commit -q -m c0
mktag() { # name age_min
    GIT_COMMITTER_DATE="@$((NOW - $2 * 60)) +0000" g tag -a -m "$1" "$1"
}
# response fixtures: line 1 = HTTP status, rest = body (the seam format)
resp() { printf '%s\n%s\n' "$2" "$3" > "$1"; }
J_NONE="$TMP/none.json"; resp "$J_NONE" 404 '{"errors":[{"detail":"Not Found"}]}'
J_HAS="$TMP/has.json"; resp "$J_HAS" 200 '{"version":{"num":"1.0.0","yanked":false}}'
J_YANK="$TMP/yank.json"; resp "$J_YANK" 200 '{"version":{"num":"1.0.0","yanked":true}}'
J_500="$TMP/500.json"; resp "$J_500" 500 '{"errors":[{"detail":"boom"}]}'
J_TIMEOUT="$TMP/timeout.json"; resp "$J_TIMEOUT" 000 ''
J_429="$TMP/429.json"; resp "$J_429" 429 '{"errors":[{"detail":"rate limited"}]}'
J_OTHERNUM="$TMP/othernum.json"; resp "$J_OTHERNUM" 200 '{"version":{"num":"11.0.0","yanked":false}}'
J_GARBAGE="$TMP/garbage.json"; resp "$J_GARBAGE" 200 'not json'

# andon NAME JSON [args...] -> rc in $RC; out in $TMP/out-NAME; inbox $TMP/inb-NAME; escalate $TMP/esc-NAME; state $TMP/st-NAME
# default mode is --page-dry with two targets; ANDON_LIVE=1 sends through the tmux stub instead
andon() {
    local n=$1 j=$2; shift 2; local mode=(--page-dry --page-target apr:cop --page-target ops:a3)
    [ -z "${ANDON_LIVE:-}" ] || mode=(--page-target apr:cop --page-target ops:a3)
    CRATES_IO_RESPONSE_FILE="$j" PATH="$TMP/stub:$PATH" bash "${ANDON_BIN:-$BIN/tag_publish_andon.sh}" --repo "$R" --state "$TMP/st-$n" \
        --inbox "$TMP/inb-$n" --escalate "$TMP/esc-$n" "${mode[@]}" "$@" > "$TMP/out-$n" 2>&1
    RC=$?
    touch "$TMP/inb-$n" "$TMP/esc-$n"
}
SINCE_OLD=2000-01-01T00:00:00Z

# ---- rows ----
mktag v1.0.0 180
andon old-absent "$J_NONE" --since $SINCE_OLD; expect "3h old + absent: rc" 1 "$RC"
has "3h old + absent: inbox" "$TMP/inb-old-absent" "tag-andon"; has "inbox names the tag" "$TMP/inb-old-absent" "v1.0.0"
expect "escalate lines" 1 "$(lines "$TMP/esc-old-absent")"
has "page printed to cop" "$TMP/out-old-absent" "PAGE apr:cop: OPERATOR-FACING ESCALATION"
has "page printed to every --page-target" "$TMP/out-old-absent" "PAGE ops:a3: OPERATOR-FACING ESCALATION"
hasnt "no hard-coded target" "$TMP/out-old-absent" "aprender-a3"
has "page names the age" "$TMP/out-old-absent" "pushed 180 min ago"
has "page names the blocking step" "$TMP/out-old-absent" "blocking step: no GitHub release"
FX_REL=1 andon rel-exists "$J_NONE" --since $SINCE_OLD
has "blocking step moves on when the release exists" "$TMP/out-rel-exists" "crates.io publish not started or incomplete"
# second run: no second page, no second escalate line, still an inbox line
andon old-absent "$J_NONE" --since $SINCE_OLD; expect "second run rc" 1 "$RC"
expect "second run: no second escalate" 1 "$(lines "$TMP/esc-old-absent")"
expect "second run: inbox line again" 2 "$(grep -c 'tag-andon' "$TMP/inb-old-absent")"
expect "second run: no second page" 0 "$(grep -c '^PAGE' "$TMP/out-old-absent")"
expect "second run: state stamp" 1 "$(ls "$TMP/st-old-absent" | grep -c '^dry-paged-v1.0.0$')"
# version present: silent, exit 0
andon present "$J_HAS" --since $SINCE_OLD; expect "present rc" 0 "$RC"
expect "present: no inbox" 0 "$(lines "$TMP/inb-present")"; expect "present: no escalate" 0 "$(lines "$TMP/esc-present")"; hasnt "present: no page" "$TMP/out-present" "PAGE"
andon yanked "$J_YANK" --since $SINCE_OLD; expect "yanked version is still an andon" 1 "$RC"
# a 200 whose body names another version is a failed read, never "present"
andon othernum "$J_OTHERNUM" --since $SINCE_OLD; expect "200 naming another version: not measured" 2 "$RC"
# crates.io status: 404 = absent (above); 500, 429, timeout (000), 200 with a garbage body = NOT MEASURED, no page
for nm in 500 429 TIMEOUT GARBAGE; do
    eval "j=\$J_$nm"; n=nm-$nm
    andon "$n" "$j" --since $SINCE_OLD; expect "$nm: rc" 2 "$RC"; hasnt "$nm: no page" "$TMP/out-$n" "PAGE"
    has "$nm: NOT_MEASURED inbox" "$TMP/inb-$n" "NOT_MEASURED"; expect "$nm: no escalate" 0 "$(lines "$TMP/esc-$n")"
done
andon missing-file "$TMP/nonexistent.json" --since $SINCE_OLD; expect "unreadable source rc" 2 "$RC"; hasnt "unreadable: no page" "$TMP/out-missing-file" "PAGE"
# a NOT_MEASURED does not stamp the tag: the next good read still pages
andon bad-then-good "$J_500" --since $SINCE_OLD; andon bad-then-good "$J_NONE" --since $SINCE_OLD; expect "page after a not-measured run" 1 "$RC"
has "page after a not-measured run" "$TMP/out-bad-then-good" "PAGE apr:cop"
# window: a 1 h old tag is nothing
mktag v1.1.0 60
git -C "$R" tag -d v1.0.0 > /dev/null; andon young "$J_NONE" --since $SINCE_OLD; expect "1h old rc" 0 "$RC"; expect "1h old: no inbox" 0 "$(lines "$TMP/inb-young")"; hasnt "1h old: no page" "$TMP/out-young" "PAGE"
andon young-narrow "$J_NONE" --since $SINCE_OLD --window-min 30; expect "1h old with a 30 min window" 1 "$RC"
# --since (explicit, fake clock): an old tag before the cutoff is ignored (the historical eight)
mktag v1.0.0 180
andon since "$J_NONE" --since "$("$FX_REALDATE" -u -d "@$((NOW - 3600))" +%FT%TZ)"; expect "older than --since rc" 0 "$RC"; expect "older than --since: no inbox" 0 "$(lines "$TMP/inb-since")"
andon since-edge "$J_NONE" --since "$("$FX_REALDATE" -u -d "@$((NOW - 180 * 60))" +%FT%TZ)"; expect "a tag exactly at --since counts" 1 "$RC"
# the default --since (2026-10-30) against the fake clock 2026-11-05: today's tag is inside it, a 2026-10-29 tag is not
andon default-since "$J_NONE"; expect "default --since: a tag after 2026-10-30 is inside" 1 "$RC"
git -C "$R" tag -d v1.0.0 v1.1.0 > /dev/null; mktag v1.2.0 $(( (NOW - $("$FX_REALDATE" -u -d 2026-10-29T12:00:00Z +%s)) / 60 ))
andon default-since-old "$J_NONE"; expect "default --since: a 2026-10-29 tag is ignored" 0 "$RC"; expect "default --since: no inbox" 0 "$(lines "$TMP/inb-default-since-old")"
git -C "$R" tag -d v1.2.0 > /dev/null
# pre-release and non-version tags are not releases
mktag v2.0.0-rc.1 300; mktag nightly-x 300; mktag v3.0 300
andon prerel "$J_NONE" --since $SINCE_OLD; expect "pre-release / odd tags rc" 0 "$RC"; expect "pre-release / odd tags: no inbox" 0 "$(lines "$TMP/inb-prerel")"
git -C "$R" tag -d v2.0.0-rc.1 nightly-x v3.0 > /dev/null
# usage errors
bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-u" --inbox "$TMP/i" --page-dry > /dev/null 2>&1; expect "no --escalate" 2 "$?"
bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-u" --inbox "$TMP/i" --escalate "$TMP/e" --page-dry --since garbage > /dev/null 2>&1; expect "bad --since" 2 "$?"
bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-u" --inbox "$TMP/i" --escalate "$TMP/e" > /dev/null 2>&1; expect "no --page-target and no --page-dry" 2 "$?"
# the seam is not honoured without its file (production copy); curl is stubbed to fail, so honouring it would read the fixture and pass
mktag v1.0.0 180
ANDON_BIN="$TMP/prod/tag_publish_andon.sh" andon prod-seam "$J_HAS" --since $SINCE_OLD; expect "production copy ignores CRATES_IO_RESPONSE_FILE (curl fails -> not measured)" 2 "$RC"
has "production copy tried the network" "$FX_GH_LOG" "curl"
has "production copy asks the per-version endpoint" "$FX_GH_LOG" "crates/aprender/1.0.0"
# ---- live paging: the stamp means DELIVERED ----
ANDON_LIVE=1 andon live-ok "$J_NONE" --since $SINCE_OLD; expect "live: rc" 1 "$RC"
has "live: each target verified with has-session" "$FX_TMUX_LOG" "has-session -t apr"; has "live: second target verified" "$FX_TMUX_LOG" "has-session -t ops"
has "live: cop page is an echo" "$FX_TMUX_LOG" "send-keys -t apr:cop echo"; has "live: other target gets send-keys -l" "$FX_TMUX_LOG" "send-keys -t ops:a3 -l"
expect "live: delivered -> stamped" 1 "$(ls "$TMP/st-live-ok" | grep -c '^paged-v1.0.0$')"; hasnt "live: delivered -> no UNDELIVERED line" "$TMP/inb-live-ok" "PAGE UNDELIVERED"
: > "$FX_TMUX_LOG"; FX_BAD_SESSION=all ANDON_LIVE=1 andon live-nosession "$J_NONE" --since $SINCE_OLD; expect "no session: rc" 1 "$RC"
expect "no session: NOT stamped" 0 "$(ls "$TMP/st-live-nosession" | grep -c '^paged-')"; has "no session: PAGE UNDELIVERED inbox line" "$TMP/inb-live-nosession" "PAGE UNDELIVERED v1.0.0"
hasnt "no session: nothing sent to a dead target" "$FX_TMUX_LOG" "send-keys"
expect "no session: the escalate line is still written once" 1 "$(lines "$TMP/esc-live-nosession")"
FX_BAD_SESSION=all ANDON_LIVE=1 andon live-nosession "$J_NONE" --since $SINCE_OLD
expect "no session, again: escalate not duplicated" 1 "$(lines "$TMP/esc-live-nosession")"; expect "no session, again: UNDELIVERED again" 2 "$(grep -c 'PAGE UNDELIVERED' "$TMP/inb-live-nosession")"
ANDON_LIVE=1 andon live-nosession "$J_NONE" --since $SINCE_OLD; expect "retry when the target is back: stamped" 1 "$(ls "$TMP/st-live-nosession" | grep -c '^paged-v1.0.0$')"
FX_SEND_RC=1 ANDON_LIVE=1 andon live-sendfail "$J_NONE" --since $SINCE_OLD
expect "send-keys fails: NOT stamped" 0 "$(ls "$TMP/st-live-sendfail" | grep -c '^paged-')"; has "send-keys fails: UNDELIVERED" "$TMP/inb-live-sendfail" "PAGE UNDELIVERED v1.0.0"
FX_BAD_SESSION=apr ANDON_LIVE=1 andon live-one-dead "$J_NONE" --since $SINCE_OLD
expect "one dead target, one live: stamped" 1 "$(ls "$TMP/st-live-one-dead" | grep -c '^paged-v1.0.0$')"; hasnt "one dead target, one live: not undelivered" "$TMP/inb-live-one-dead" "PAGE UNDELIVERED"
# a --page-dry run on the live state dir never suppresses the real page (lane round 7)
andon drylive "$J_NONE" --since $SINCE_OLD; : > "$FX_TMUX_LOG"; ANDON_LIVE=1 andon drylive "$J_NONE" --since $SINCE_OLD
has "dry run first, live run still pages" "$FX_TMUX_LOG" "send-keys -t apr:cop"; expect "dry then live: live stamp" 1 "$(ls "$TMP/st-drylive" | grep -c '^paged-v1.0.0$')"
git -C "$R" tag -d v1.0.0 > /dev/null
# lock: a second run while the first holds it does nothing and says so (exit 3)
mktag v1.0.0 180; mkdir -p "$TMP/st-lock"
( exec 8> "$TMP/st-lock/.lock"; flock -n 8; CRATES_IO_RESPONSE_FILE="$J_NONE" PATH="$TMP/stub:$PATH" bash "$BIN/tag_publish_andon.sh" --repo "$R" --state "$TMP/st-lock" --inbox "$TMP/inb-lock" --escalate "$TMP/esc-lock" --page-dry --since $SINCE_OLD > "$TMP/out-lock" 2>&1; echo $? > "$TMP/lock-rc" )
expect "lock held: rc 3" 3 "$(cat "$TMP/lock-rc")"; has "lock held: says so" "$TMP/out-lock" "holds the lock"; hasnt "lock held: no page" "$TMP/out-lock" "PAGE"

# ---- library rows ----
eval "$(grep "^page_text()" "$BIN/tag_publish_andon.sh")"
expect "page_text strips quotes, backticks, dollar, bang, backslash" "abcdef" "$(page_text "a'b\`c\$d!e\\f")"
lib() { ( . "$BIN/lib_crates_io.sh" && CRATES_IO_RESPONSE_FILE="$1" "${@:2}" ) > /dev/null 2>&1; }
lib "$J_HAS" crates_io_has_version aprender 1.0.0; expect "has_version 200 + matching num" 0 "$?"
lib "$J_NONE" crates_io_has_version aprender 1.0.0; expect "has_version 404" 1 "$?"
lib "$J_500" crates_io_has_version aprender 1.0.0; expect "has_version 500" 2 "$?"
lib "$J_TIMEOUT" crates_io_has_version aprender 1.0.0; expect "has_version timeout (000)" 2 "$?"
lib "$J_HAS" crates_io_has_version aprender 9.9.9; expect "has_version 200 but another num" 2 "$?"
lib "$J_YANK" crates_io_has_version aprender 1.0.0; expect "has_version yanked" 1 "$?"
lib "$J_HAS" crates_io_has_version 'a b' 1.0.0; expect "bad crate name" 2 "$?"
J_INJ="$TMP/inj.json"; resp "$J_INJ" 200 '{"version":{"num":"1.0.0/../x","yanked":false}}'
lib "$J_INJ" crates_io_has_version aprender '1.0.0/../x'; expect "bad version (path injection) even if the body echoes it" 2 "$?"
( set +e +u; . "$BIN/lib_crates_io.sh"; case "$-" in *e* | *u*) exit 1 ;; esac ); expect "sourcing leaves errexit/nounset alone" 0 "$?"

# ---- THE OPERATOR FIXTURE: tag pushed, publish fails -> the close gate refuses (milestone stays open) and the andon fires ----
: > "$FX_GH_LOG"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_RESPONSE_FILE="$J_NONE" crates_io_close_gate aprender 1.0.0 ) > "$TMP/gate.out" 2>&1; expect "operator fixture: close gate refuses" 1 "$?"
has "operator fixture: gate says why" "$TMP/gate.out" "crates.io does not show 1.0.0: milestone and epic stay open"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_RESPONSE_FILE="$J_500" crates_io_close_gate aprender 1.0.0 ) > "$TMP/gate2.out" 2>&1; expect "operator fixture: unmeasured read also refuses" 2 "$?"
( . "$BIN/lib_crates_io.sh" && CRATES_IO_RESPONSE_FILE="$J_HAS" crates_io_close_gate aprender 1.0.0 ) > /dev/null 2>&1; expect "close gate opens once crates.io shows it" 0 "$?"
andon operator "$J_NONE" --since $SINCE_OLD; expect "operator fixture: andon fires" 1 "$RC"; has "operator fixture: paged" "$TMP/out-operator" "PAGE apr:cop"
# that autopilot step 9 itself closes nothing on this refusal is proven in check_release_autopilot_dogfood_close.sh
# (row close-crates-absent): the andon never closes anything, so asserting "no close" on its gh log would be vacuous

# a failed escalate append never blocks the live page: inbox line, no stamp, the page still goes out
mkdir -p "$TMP/esc-is-a-dir"; : > "$FX_TMUX_LOG"; ANDON_LIVE=1 andon escfail "$J_NONE" --since $SINCE_OLD --escalate "$TMP/esc-is-a-dir"; expect "escalate write fails: andon still open" 1 "$RC"
has "escalate write fails: inbox says so" "$TMP/inb-escfail" "ESCALATE WRITE FAILED"
has "escalate write fails: page still delivered" "$FX_TMUX_LOG" "send-keys -t apr:cop"
[ ! -e "$TMP/st-escfail/escalated-v1.0.0" ]; expect "escalate write fails: not stamped" 0 "$?"
# ---- lightweight tags: dated by first sight, never by their commit date (lane round 3) ----
for t in $(g tag -l); do g tag -d "$t" > /dev/null; done
GIT_COMMITTER_DATE="@$((NOW - 3 * 86400)) +0000" g commit -q --allow-empty -m old-lw
g tag v0.9.0 # lightweight, present before the first run
SINCE_4H=$("$FX_REALDATE" -u -d "@$((NOW - 240 * 60))" +%FT%TZ)
andon lw "$J_NONE" --since "$SINCE_4H"; expect "baseline run: a lightweight tag on a commit older than --since never pages" 0 "$RC"
expect "baseline run: lightweight dated by its commit (a lower bound of the push)" "$((NOW - 3 * 86400))" "$(cat "$TMP/st-lw/seen-v0.9.0")"
expect "baseline run: lastrun stamped" "$NOW" "$(cat "$TMP/st-lw/lastrun")"
# a lightweight tag pushed after --since but before the first run is still caught (dated by its commit: early, never silent)
GIT_COMMITTER_DATE="@$((NOW - 180 * 60)) +0000" g commit -q --allow-empty -m recent-lw; g tag v0.9.5
andon lwb "$J_NONE" --since "$SINCE_4H"; expect "baseline run: lightweight tag pushed before install, absent: andon" 1 "$RC"
has "baseline andon names the tag" "$TMP/out-lwb" "v0.9.5"; hasnt "baseline andon skips the pre---since tag" "$TMP/out-lwb" "v0.9.0"
g tag -d v0.9.5 > /dev/null
g tag v2.0.0 "v0.9.0^{commit}" # lightweight, pushed now on the 3-day-old commit (older than --since)
andon lw "$J_NONE" --since "$SINCE_4H"; expect "new lightweight tag on an old commit: no page at once" 0 "$RC"
has "new lightweight tag: not skipped silently, inbox says so" "$TMP/inb-lw" "LIGHTWEIGHT TAG v2.0.0"
expect "new lightweight tag: dated by first sight" "$NOW" "$(cat "$TMP/st-lw/seen-v2.0.0")"
echo $((NOW - 180 * 60)) > "$TMP/st-lw/seen-v2.0.0" # 3 h after first sight
andon lw "$J_NONE" --since "$SINCE_4H"; expect "lightweight tag 3 h after first sight, absent: andon" 1 "$RC"
has "lightweight andon names the tag" "$TMP/out-lw" "v2.0.0"; hasnt "baselined tag never pages" "$TMP/out-lw" "v0.9.0"

# ---- an annotated tag is dated by its push (first sight on origin), not its tagger date (lane round 4) ----
mkdir -p "$TMP/st-lwa"; echo "$NOW" > "$TMP/st-lwa/lastrun" # a later run: the previous run was just now
GIT_COMMITTER_DATE="@$((NOW - 86400)) +0000" g tag -a -m v3.0.0 v3.0.0 # created a day ago (before --since), pushed now
andon lwa "$J_NONE" --since "$SINCE_4H"; expect "annotated tag created before --since, pushed after: no page at once" 0 "$RC"
expect "annotated tag: dated by first sight" "$NOW" "$(cat "$TMP/st-lwa/seen-v3.0.0")"
hasnt "annotated tag: no LIGHTWEIGHT line" "$TMP/inb-lwa" "LIGHTWEIGHT TAG v3.0.0"
echo $((NOW - 180 * 60)) > "$TMP/st-lwa/seen-v3.0.0" # 3 h after its push
andon lwa "$J_NONE" --since "$SINCE_4H"; expect "annotated tag 3 h after its push, absent: andon" 1 "$RC"; has "annotated andon names the tag" "$TMP/out-lwa" "v3.0.0"
# the timer was down 3 h: a tag pushed then is dated by its tagger date (early), never by first sight (late) (lane round 8)
mkdir -p "$TMP/st-gap"; echo $((NOW - 180 * 60)) > "$TMP/st-gap/lastrun"
GIT_COMMITTER_DATE="@$((NOW - 170 * 60)) +0000" g tag -a -m v5.0.0 v5.0.0
andon gap "$J_NONE" --since "$SINCE_4H"; expect "timer down 3 h, tag pushed meanwhile, absent: andon on the first run back" 1 "$RC"
has "gap andon names the tag" "$TMP/out-gap" "v5.0.0"; expect "gap run: dated by tagger date" "$((NOW - 170 * 60))" "$(cat "$TMP/st-gap/seen-v5.0.0")"
andon gapq "$J_NONE" --since "$SINCE_4H" --max-gap-min nope; expect "bad --max-gap-min: usage" 2 "$RC"
g tag -d v5.0.0 > /dev/null

# ---- blocking step: tag absent on origin (ls-remote rc 2) vs origin unreadable (any other rc: not measured, never "absent") ----
git init -q --bare "$TMP/origin.git"; g remote add origin "$TMP/origin.git"
FX_REALGIT=$(command -v git)
printf '#!/usr/bin/env bash\ncase " $* " in *" ls-remote "*) [ -z "${FX_LSREMOTE_RC:-}" ] || exit "$FX_LSREMOTE_RC" ;; esac\nexec %s "$@"\n' "$FX_REALGIT" > "$TMP/stub/git"; chmod +x "$TMP/stub/git"
andon lwa "$J_NONE" --since "$SINCE_4H"; has "step: tag absent on origin" "$TMP/out-lwa" "blocking step: tag not on origin"
FX_LSREMOTE_RC=128 andon lwa "$J_NONE" --since "$SINCE_4H"; has "step: origin unreadable is not measured" "$TMP/out-lwa" "origin unreadable (tag-on-origin not measured)"
hasnt "step: origin unreadable is never 'not on origin'" "$TMP/out-lwa" "blocking step: tag not on origin"
mv "$TMP/stub/git" "$TMP/git-stub.off"; g remote remove origin
# a tag the andon cannot date (state unwritable) is not measured (exit 2 + inbox line), never a silent skip
g tag -a -m v4.0.0 v4.0.0; mkdir -p "$TMP/st-lwa/seen-v4.0.0"
andon lwa "$J_NONE" --since "$SINCE_4H"; has "undatable tag: NOT_MEASURED inbox line" "$TMP/inb-lwa" "NOT_MEASURED tag v4.0.0: cannot date it"
g tag -d v4.0.0 > /dev/null

# the autopilot calls the gate in step 9 (a code line, not a comment); that it runs BEFORE any close is the behavioural
# row close-crates-absent in check_release_autopilot_dogfood_close.sh
grep -qE '^[[:space:]]+crates_io_close_gate aprender "\$V"' "$HERE/autopilot.sh" && ok || bad "autopilot.sh step 9 does not call crates_io_close_gate"

printf 'falsify_tag_publish_andon: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]
