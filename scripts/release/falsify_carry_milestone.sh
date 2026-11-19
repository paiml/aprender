#!/usr/bin/env bash
# falsify_carry_milestone.sh -- contract test for carry_milestone.sh (milestone auto-move on release)
#   bash scripts/release/falsify_carry_milestone.sh      # exit 0 = every row held
# A fake gh serves milestones 0.72.3 (open, items 1 2), 0.73.0, 0.74.0, backlog; PATCH results are
# controlled per row. PLANTED DEFECT rows prove the script refuses to report success when it should not.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SUBJECT="${CARRY_SCRIPT:-$HERE/carry_milestone.sh}"
TMP=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}") || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
cat > "$TMP/gh" <<'FAKE'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
case "$1 $2" in
    "api repos/"*) :;;
esac
if [ "$1" = api ] && [ "$2" = -X ]; then
    n=${4##*/}; [ "$FAKE_PATCH" = null ] && { echo null; exit 0; }
    [ "$FAKE_PATCH" = wrong ] && { echo 99; exit 0; }
    [ "$FAKE_PATCH" = "fail$n" ] && exit 1
    cat > /dev/null; echo "$FAKE_NEXT"; exit 0
elif [ "$1" = api ]; then
    case "$2" in "repos/"*"/issues?milestone="*)
        [ "$FAKE_ITEMS" = down ] && exit 1
        case "$2" in *milestone=31\&*) printf '1\n2\n'; exit 0;; *) exit 1;; esac;;
    esac
    [ "$FAKE_MS" = down ] && exit 1
    printf '%s' "$FAKE_MS"; exit 0
elif [ "$1" = issue ] || [ "$1" = pr ]; then echo "fake gh: $1 list is not used (no --milestone on pr)" >&2; exit 1
fi
FAKE
chmod +x "$TMP/gh"
FIXTURE_MS='[{"number":31,"title":"0.72.3"},{"number":16,"title":"0.74.0"},{"number":15,"title":"0.73.0"},{"number":11,"title":"backlog"}]'
pass=0 fail=0
row() { # name want_rc must_match must_not_match -- args ; env via exports
    local name=$1 want=$2 yes=$3 no=$4; shift 5
    : > "$TMP/log"; out=$(FAKE_LOG="$TMP/log" GH="$TMP/gh" CARRY_PACE=0 bash "$SUBJECT" "$@" 2>&1); rc=$?
    all="$out$(cat "$TMP/log")"
    ok=1; [ "$rc" = "$want" ] || ok=0
    [ -z "$yes" ] || grep -Eq -- "$yes" <<<"$all" || ok=0
    [ -z "$no" ] || ! grep -Eq -- "$no" <<<"$all" || ok=0
    if [ $ok = 1 ]; then pass=$((pass+1)); else fail=$((fail+1)); printf 'FAIL %s (rc=%s want %s)\n%s\n' "$name" "$rc" "$want" "$out"; fi
}
export FAKE_MS="$FIXTURE_MS" FAKE_NEXT=15 FAKE_PATCH=ok FAKE_ITEMS=ok
export FAKE_PATCH=ok FAKE_ITEMS=ok
row dry-run 0 'dry-run: #2 0.72.3 -> 0.73.0' 'api -X PATCH' -- 0.72.3
export FAKE_PATCH=ok
row execute-moves-to-next-not-backlog 0 'moved=2 failed=0' 'milestone":11' -- 0.72.3 --execute
row execute-patch-body-targets-0.73.0 0 'moved=2' 'FAILED' -- v0.72.3 --execute
FAKE_PATCH=null row PLANTED-null-return-is-failure 1 'FAILED #1' 'moved=2' -- 0.72.3 --execute
FAKE_PATCH=wrong row PLANTED-wrong-milestone-is-failure 1 'FAILED' 'moved=2' -- 0.72.3 --execute
FAKE_PATCH=fail2 row PLANTED-one-http-failure-exit-1 1 'moved=1 failed=1' '' -- 0.72.3 --execute
FAKE_MS=down row PLANTED-unreadable-milestones-refuses 1 'not_measured' 'moved=' -- 0.72.3 --execute
FAKE_MS='[{"number":31,"title":"0.72.3"},{"number":11,"title":"backlog"}]' row PLANTED-no-next-milestone-refuses 1 'no open milestone newer' 'moved=' -- 0.72.3 --execute
FAKE_MS='[{"number":15,"title":"0.73.0"}]' row released-milestone-not-open-noop 0 'nothing to carry' 'PATCH' -- 0.72.3 --execute
FAKE_ITEMS=down row PLANTED-unreadable-items-refuses 1 'not_measured' 'moved=' -- 0.72.3 --execute
row bad-version-usage 2 'X.Y.Z' '' -- 0.72 --execute
row unknown-flag-usage 2 'unknown argument' '' -- 0.72.3 --bogus
printf 'falsify_carry_milestone: pass=%s fail=%s\n' "$pass" "$fail"
[ "$fail" = 0 ]
