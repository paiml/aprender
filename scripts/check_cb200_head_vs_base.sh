#!/usr/bin/env bash
# CB-200 (TDG grade gate) as a HEAD-vs-BASE comparison: no stored limit.
#
# `pmat comply check` judges CB-200 against `.pmat-gates.toml [tdg] baseline`, a number somebody
# typed on some day under some pmat. The release path (scripts/dogfood.sh `pmat-comply`, run by
# autopilot.sh T-1 and t2_preflight.sh) inherited that number, so a stale limit and a moved
# instrument (3.41.1 -> 3.42.0) turned FINAL red with no code change. This gate measures BOTH
# trees with ONE pmat in ONE run and passes iff head <= base. The stored baseline is neutralised
# (set to 0) in both scratch trees, so its value cannot decide anything: only the count comes out.
#
# Usage:
#   scripts/check_cb200_head_vs_base.sh --base <ref> [--head <ref>]    # head defaults to HEAD
#   scripts/check_cb200_head_vs_base.sh --selftest
# Env: PMAT_BIN (REQUIRED: the pinned pmat, from scripts/verifier_pin.sh; unset = rc 3).
# Exit: 0 head <= base | 1 head > base (new debt) | 2 usage | 3 NOT MEASURED (never a pass).
set -uo pipefail
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
BASE=""; HEAD_REF="HEAD"; MODE=run

while [ $# -gt 0 ]; do
    case "$1" in
        --base) BASE=${2:-}; shift 2 ;;
        --head) HEAD_REF=${2:-}; shift 2 ;;
        --selftest) MODE=selftest; shift ;;
        *) printf 'check_cb200_head_vs_base.sh: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
done

[ "$MODE" = selftest ] || [ -n "${PMAT_BIN:-}" ] || { echo "check_cb200_head_vs_base.sh: PMAT_BIN unset — source scripts/verifier_pin.sh (a gate measured with an unknown pmat is not a gate)" >&2; exit 3; }

WT_ROOT=""
cleanup() {
    [ -n "$WT_ROOT" ] || return 0
    for d in "$WT_ROOT"/*/; do
        [ -d "$d" ] && git worktree remove --force "$d" >/dev/null 2>&1
    done
    rm -rf -- "${WT_ROOT:?}"
}
trap cleanup EXIT

# count_at <label> <ref>: prints the number of definitions below min_grade, or returns 3.
count_at() {
    local label=$1 ref=$2 wt out
    git rev-parse --verify --quiet "$ref^{commit}" >/dev/null || { printf 'NOT-MEASURED %s: ref %s does not resolve\n' "$label" "$ref" >&2; return 3; }
    wt="$WT_ROOT/$label"
    git worktree add -q --detach "$wt" "$ref" >/dev/null 2>&1 || { printf 'NOT-MEASURED %s: cannot check out %s\n' "$label" "$ref" >&2; return 3; }
    python3 - "$wt/.pmat-gates.toml" <<'PY' || return 3
import re, sys
p = sys.argv[1]
try:
    s = open(p, encoding="utf-8").read()
except OSError:
    sys.exit(0)  # no gates file: nothing stored to neutralise
head, sep, rest = s.partition("[tdg]")
if sep:
    sec, nl, tail = rest.partition("\n[")
    sec = re.sub(r"(?m)^(\s*baseline\s*=\s*)\d+", r"\g<1>0", sec)
    s = head + sep + sec + nl + tail
open(p, "w", encoding="utf-8").write(s)
PY
    ( cd "$wt" && timeout 900 "$PMAT_BIN" query x --limit 1 >/dev/null 2>&1 )
    out=$( cd "$wt" && timeout 900 "$PMAT_BIN" comply check --format json 2>/dev/null ) || true
    printf '%s' "$out" | python3 -c '
import json, re, sys
try:
    d = json.load(sys.stdin)
    c = next(c for c in d["checks"] if c["name"].startswith("CB-200"))
except Exception as e:
    print("NOT-MEASURED: no parsable CB-200 row (%s)" % type(e).__name__, file=sys.stderr); sys.exit(3)
if c.get("status") == "Skip":
    print("NOT-MEASURED: CB-200 is Skip (no .pmat/context.db?)", file=sys.stderr); sys.exit(3)
m = re.search(r"(\d+) definition\(s\) below minimum grade", c.get("message", ""))
if not m:
    print("NOT-MEASURED: CB-200 %s carries no count: %r" % (c.get("status"), c.get("message", "")[:80]), file=sys.stderr); sys.exit(3)
print(m.group(1))' || { printf 'NOT-MEASURED %s (%s)\n' "$label" "$ref" >&2; return 3; }
}

run() {
    [ -n "$BASE" ] || { printf 'check_cb200_head_vs_base.sh: --base <ref> is required (no stored default)\n' >&2; return 2; }
    WT_ROOT=$(mktemp -d) || return 3
    local ver head base
    ver=$("$PMAT_BIN" --version 2>/dev/null | head -1)
    base=$(count_at base "$BASE") || return 3
    head=$(count_at head "$HEAD_REF") || return 3
    if [ "$head" -le "$base" ]; then
        printf 'ok    CB-200 head %s <= base %s (%s vs %s; one scanner, one run: %s)\n' "$head" "$base" "$HEAD_REF" "$BASE" "$ver"
        return 0
    fi
    printf 'FAIL  CB-200 head %s > base %s: %s definition(s) below min_grade were added between %s and %s (%s). Fix them; no baseline to raise.\n' \
        "$head" "$base" "$((head - base))" "$BASE" "$HEAD_REF" "$ver"
    return 1
}

# ------------------------------------------------------------------ selftest ---
# A fake pmat whose CB-200 count is the integer in ./count (committed per ref), so the case table
# drives every branch through the same code the release uses. `mode` file selects the output shape.
selftest() {
    local d bad=0 pm base head
    d=$(mktemp -d) || return 3
    pm="$d/fakepmat"
    cat > "$pm" <<'FAKE'
#!/usr/bin/env bash
case "$1" in
    --version) echo "pmat 0.0.0-fake"; exit 0 ;;
    query) exit 0 ;;
    comply)
        m=$(cat mode 2>/dev/null || echo ok); n=$(cat count 2>/dev/null || echo 0)
        case "$m" in
            ok)   printf '{"summary":{},"checks":[{"name":"CB-200: TDG Grade Gate","status":"Fail","message":"%s definition(s) below minimum grade B - baseline"}]}' "$n" ;;
            skip) printf '{"summary":{},"checks":[{"name":"CB-200: TDG Grade Gate","status":"Skip","message":"Not measured"}]}' ;;
            nocount) printf '{"summary":{},"checks":[{"name":"CB-200: TDG Grade Gate","status":"Pass","message":"fine"}]}' ;;
            garbage) printf 'not json' ;;
        esac ;;
esac
FAKE
    chmod +x "$pm"
    (
        cd "$d" && git init -q r && cd r && git config user.email t@t && git config user.name t
        # stored baseline says 999: a comparison that read it would pass everything below
        printf '[tdg]\nbaseline = 999\nmin_grade = "B"\n' > .pmat-gates.toml
        commit() { printf '%s' "$1" > count; printf '%s' "${2:-ok}" > mode; git add -A; git commit -q -m "$3"; git tag -f "$3" >/dev/null; }
        commit 10 ok base
        commit 10 ok equal
        commit 9  ok lower
        commit 11 ok higher
        commit 3  skip skipped
        commit 3  nocount nocount
        commit 3  garbage garbage
    ) || { rm -rf "${d:?}"; return 3; }
    check() { # <name> <base> <head> <want-rc> [<rc-source-dir override>]
        local rc
        ( cd "$d/r" && PMAT_BIN="$pm" bash "$SELF" --base "$2" --head "$3" >"$d/out" 2>&1 ); rc=$?
        if [ "$rc" = "$4" ]; then printf '  ok    %-28s rc=%s\n' "$1" "$rc"
        else printf '  FAIL  %-28s want rc=%s got rc=%s: %s\n' "$1" "$4" "$rc" "$(tail -1 "$d/out")"; bad=$((bad + 1)); fi
    }
    check "equal is green"              base equal    0
    check "lower is green"              base lower    0
    check "higher is RED (999 stored)"  base higher   1
    check "head Skip is not measured"   base skipped  3
    check "no count is not measured"    base nocount  3
    check "garbage is not measured"     base garbage  3
    check "base Skip is not measured"   skipped equal 3
    check "unresolvable base"           no-such-ref equal 3
    ( cd "$d/r" && PMAT_BIN="$pm" bash "$SELF" --head equal >"$d/out" 2>&1 ); [ $? = 2 ] \
        && printf '  ok    %-28s rc=2\n' "no --base is a usage error" \
        || { printf '  FAIL  no --base must be rc 2\n'; bad=$((bad + 1)); }
    rm -rf "${d:?}"
    printf -- '--- %s bad\n' "$bad"
    [ "$bad" -eq 0 ]
}

case "$MODE" in
    selftest) selftest; exit $? ;;
    run) run; exit $? ;;
esac
