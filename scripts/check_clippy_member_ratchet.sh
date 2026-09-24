#!/usr/bin/env bash
# check_clippy_member_ratchet.sh — clippy over EVERY workspace member, as a shrink-only ratchet (#4152).
#
# WHY THIS EXISTS
#   Every clippy gate this repo ran (Makefile:161/230/246, `cargo clippy -- -D warnings`) linted the
#   ROOT FACADE package only, which is nearly empty. Measured on main aa7c6ef03 (2026-09-24):
#   `cargo clippy --workspace --all-targets --no-deps -- -D warnings` = 87 failing targets,
#   3807 diagnostics, 15 crates. No gate had ever seen them, so no gate could keep them from growing.
#
# WHY A RATCHET AND NOT `-D warnings`
#   A member-wide `-D warnings` gate cannot be first-green without fixing ~3800 findings first
#   (2288 are the `.clippy.toml` unwrap ban, tracked in its own sweep ticket). A ratchet is green
#   on main TODAY and still refuses every NEW finding (cop ruling on #4152, 2026-09-24).
#
# THE RULE, keyed on (crate, target kind, lint) — scripts/clippy_member_baseline.txt:
#   * a key that is not in the baseline          -> FAIL (a new kind of finding somewhere)
#   * a count above its baseline                 -> FAIL (more of an old kind)
#   * a count BELOW its baseline, or a key gone  -> FAIL until the baseline is shrunk to match
#                                                   (--update-baseline). Progress is recorded,
#                                                   never left as slack a later regression can use.
#   * the baseline FILE may only shrink against origin/main (lib_baseline_ratchet.sh `keyed`),
#     so "record it instead of fixing it" is refused in-branch.
#   * clippy that did not FINISH (a target failed to compile) is a FAIL, never a verdict: an
#     unfinished census hides every finding in the targets it never reached, which would read as
#     "improvement".
#   * the baseline records the clippy version; a different clippy is refused (its lint set differs,
#     CLAUDE.md: "Clippy's lint set is not monotonic").
#
# USAGE
#   bash scripts/check_clippy_member_ratchet.sh                  # measure + judge (CI)
#   bash scripts/check_clippy_member_ratchet.sh --update-baseline  # rewrite the baseline from a measurement
#   bash scripts/check_clippy_member_ratchet.sh --from-json F    # judge a saved `--message-format=json` run
#   bash scripts/check_clippy_member_ratchet.sh --self-test      # the case table (no cargo)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
BASELINE_REL=scripts/clippy_member_baseline.txt
BASELINE="$ROOT/$BASELINE_REL"
# Same exclusions as CI's workspace-test: these need a GPU toolchain and are gated separately.
EXCLUDES=(--exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute)

# census <json-file> -> "<crate>|<kind>|<lint>\t<count>" sorted, on stdout; rc 3 if clippy did not finish.
census() {
    python3 - "$1" <<'PY'
import json, sys, collections
def pkg_name(pkg):
    # Cargo's package-id spec: `path+file:///.../crates/<dir>#<name>@<ver>`, or `...#<ver>` alone
    # when the name equals the directory's last component (measured: aprender-core came out "0.69.0").
    head, _, frag = pkg.partition("#")
    if "@" in frag:
        return frag.split("@", 1)[0]
    return head.rstrip("/").rsplit("/", 1)[-1] or pkg
counts = collections.Counter()
finished = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        o = json.loads(line)
    except ValueError:
        continue
    r = o.get("reason")
    if r == "build-finished":
        finished = bool(o.get("success"))
    if r != "compiler-message":
        continue
    m = o.get("message") or {}
    if m.get("level") not in ("warning", "error"):
        continue
    code = (m.get("code") or {}).get("code")
    if not code:
        continue  # "aborting due to N errors" and summary lines carry no code
    name = pkg_name(o.get("package_id", ""))
    kind = ((o.get("target") or {}).get("kind") or ["?"])[0]
    counts[f"{name}|{kind}|{code}"] += 1
if finished is None:
    print("census: no build-finished record: this is not a complete cargo JSON run", file=sys.stderr)
    sys.exit(3)
if not finished:
    print("census: clippy did not finish (a target failed to compile)", file=sys.stderr)
    sys.exit(3)
for k in sorted(counts):
    print(f"{k}\t{counts[k]}")
PY
}

# checked_members <json-file> -> how many distinct WORKSPACE packages clippy produced an artifact for.
checked_members() {
    python3 - "$1" <<'PY'
import json, sys
def pkg_name(pkg):
    # Cargo's package-id spec: `path+file:///.../crates/<dir>#<name>@<ver>`, or `...#<ver>` alone
    # when the name equals the directory's last component (measured: aprender-core came out "0.69.0").
    head, _, frag = pkg.partition("#")
    if "@" in frag:
        return frag.split("@", 1)[0]
    return head.rstrip("/").rsplit("/", 1)[-1] or pkg
names = set()
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if '"compiler-artifact"' not in line:
        continue
    try:
        o = json.loads(line)
    except ValueError:
        continue
    pkg = o.get("package_id", "")
    if pkg.startswith("path+file://"):
        names.add(pkg_name(pkg))
print(len(names))
PY
}

# errors_of <json-file> -> one line per ERROR-level diagnostic: what stopped the build, and where.
errors_of() {
    python3 - "$1" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if '"compiler-message"' not in line:
        continue
    try:
        o = json.loads(line)
    except ValueError:
        continue
    m = o.get("message") or {}
    code = (m.get("code") or {}).get("code")
    if m.get("level") != "error" or not code:
        continue
    span = next((s for s in m.get("spans", []) if s.get("is_primary")), {})
    where = f'{span.get("file_name", "?")}:{span.get("line_start", "?")}'
    print(f"      {code} at {where}: {m.get('message', '')[:160]}")
    n += 1
    if n >= 10:
        print("      (first 10 shown)")
        break
if n == 0:
    print("      (no coded error-level diagnostic: see cargo's stderr)")
PY
}

# judge <baseline-file> <census-file> -> verdict rows; rc 0 clean, 1 on any new/rise/stale key.
judge() {
    python3 - "$1" "$2" <<'PY'
import sys
def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        k, n = line.rstrip("\n").split("\t")
        d[k] = int(n)
    return d
base, cur = load(sys.argv[1]), load(sys.argv[2])
bad = []
for k, n in sorted(cur.items()):
    if k not in base:
        bad.append(f"FAIL  NEW    {k}: {n} finding(s), a key the baseline does not have")
    elif n > base[k]:
        bad.append(f"FAIL  ROSE   {k}: {base[k]} -> {n}")
for k, n in sorted(base.items()):
    c = cur.get(k, 0)
    if c < n:
        bad.append(f"FAIL  STALE  {k}: baseline {n}, measured {c} -- shrink it (--update-baseline)")
tb, tc = sum(base.values()), sum(cur.values())
for line in bad:
    print(line)
print(f"{'FAIL' if bad else 'ok  '}  clippy member ratchet: {tc} finding(s) measured, baseline {tb}, {len(cur)} key(s)")
sys.exit(1 if bad else 0)
PY
}

clippy_version() { cargo clippy --version 2>/dev/null | head -1; }

measure() { # measure <json-out>
    ( cd "$ROOT" && cargo clippy --workspace --all-targets --no-deps "${EXCLUDES[@]}" --keep-going \
        --message-format=json > "$1" 2> "$1.stderr" ) || true
}

self_test() {
    local t fails=0
    t=$(mktemp -d) || exit 2
    trap 'rm -rf -- "${t:?}"' RETURN
    # a synthetic cargo JSON stream: two lints in serve's lib, one in core's tests
    msg() { printf '{"reason":"compiler-message","package_id":"path+file:///x/crates/%s#%s@0.1.0","target":{"kind":["%s"]},"message":{"level":"%s","code":{"code":"%s"}}}\n' "$1" "$1" "$2" "$3" "$4"; }
    fin() { printf '{"reason":"build-finished","success":%s}\n' "$1"; }
    { msg aprender-serve lib warning clippy::undocumented_unsafe_blocks
      msg aprender-serve lib warning clippy::undocumented_unsafe_blocks
      msg aprender-core test warning clippy::disallowed_methods
      printf '{"reason":"compiler-message","message":{"level":"error","code":null}}\n'
      fin true; } > "$t/base.json"
    census "$t/base.json" > "$t/base.txt"
    row() { # row <want-rc> <label> <json>
        local want=$1 label=$2 got
        if census "$3" > "$t/cur.txt" 2>/dev/null; then
            if judge "$t/base.txt" "$t/cur.txt" > /dev/null; then got=0; else got=1; fi
        else got=3; fi
        if [ "$got" = "$want" ]; then printf '  ok    %s (rc %s)\n' "$label" "$got"
        else printf '  BROKE %s: want rc %s, got %s\n' "$label" "$want" "$got"; fails=$((fails + 1)); fi
    }
    row 0 "an unchanged census is green" "$t/base.json"
    { cat "$t/base.json" | sed '$d'; msg aprender-serve lib warning clippy::undocumented_unsafe_blocks; fin true; } > "$t/rise.json"
    row 1 "one more undocumented unsafe in serve (the planted mutant's shape) is RED" "$t/rise.json"
    { cat "$t/base.json" | sed '$d'; msg aprender-graph lib warning clippy::needless_range_loop; fin true; } > "$t/new.json"
    row 1 "a lint in a crate/kind the baseline never had is RED" "$t/new.json"
    { msg aprender-serve lib warning clippy::undocumented_unsafe_blocks; msg aprender-core test warning clippy::disallowed_methods; fin true; } > "$t/shrunk.json"
    row 1 "a fixed finding with the baseline NOT shrunk is RED (stale)" "$t/shrunk.json"
    { msg aprender-serve lib warning clippy::undocumented_unsafe_blocks; msg aprender-serve lib warning clippy::undocumented_unsafe_blocks; fin true; } > "$t/gone.json"
    row 1 "a whole key disappearing with the baseline not shrunk is RED" "$t/gone.json"
    { cat "$t/base.json" | sed '$d'; fin false; } > "$t/unfinished.json"
    row 3 "clippy that did not finish is refused, never judged" "$t/unfinished.json"
    grep -v build-finished "$t/base.json" > "$t/nofin.json"
    row 3 "a stream with no build-finished record is refused" "$t/nofin.json"
    { msg aprender-serve lib warning clippy::undocumented_unsafe_blocks; msg aprender-serve lib warning clippy::undocumented_unsafe_blocks
      msg aprender-core test warning clippy::disallowed_methods; msg aprender-core test note clippy::disallowed_methods; fin true; } > "$t/note.json"
    row 0 "a note-level message is not a finding" "$t/note.json"
    # the version-only package-id form must name the crate, not the version
    printf '{"reason":"compiler-message","package_id":"path+file:///x/crates/aprender-core#0.69.0","target":{"kind":["lib"]},"message":{"level":"warning","code":{"code":"clippy::x"}}}\n{"reason":"build-finished","success":true}\n' > "$t/verid.json"
    if census "$t/verid.json" | grep -q '^aprender-core|lib|clippy::x'; then printf '  ok    a version-only package id names the crate\n'
    else printf '  BROKE a version-only package id was not named by its directory\n'; fails=$((fails + 1)); fi
    if [ "$fails" -eq 0 ]; then echo "SELF-TEST OK: 9 rows"; return 0; fi
    echo "SELF-TEST FAIL: $fails row(s) broke"; return 1
}

write_baseline() { # write_baseline <census-file>
    local ver total
    ver=$(clippy_version)
    total=$(awk -F'\t' '{s+=$2} END {print s+0}' "$1")
    {
        echo "# tool_version=none (versioned by the clippy_version line below, which check_clippy_member_ratchet.sh checks itself: the library's probe runs '<tool> --version' and cargo-clippy prints 'clippy ...')"
        echo "# clippy_version=$ver"
        echo "# clippy_member_baseline.txt — #4152. <crate>|<target kind>|<lint><TAB><count>, shrink-only."
        echo "# Measured by: cargo clippy --workspace --all-targets --no-deps ${EXCLUDES[*]} --keep-going --message-format=json"
        echo "# (no -D warnings: a warning must not stop its dependents from being checked; every warn-level lint is counted)"
        echo "# First measurement, main aa7c6ef03 (2026-09-24): 3807 diagnostics under -D warnings across 87 failing targets."
        echo "# This census: $total finding(s). Regenerate with: bash scripts/check_clippy_member_ratchet.sh --update-baseline"
        cat "$1"
    } > "$BASELINE"
    printf 'baseline written: %s key(s), %s finding(s), %s\n' "$(grep -c . "$1")" "$total" "$ver"
}

check_version() {
    local want have
    want=$(grep -m1 '^# clippy_version=' "$BASELINE" | sed 's/^# clippy_version=//')
    have=$(clippy_version)
    if [ -z "$want" ]; then echo "FAIL  $BASELINE_REL carries no '# clippy_version=' line"; return 1; fi
    if [ "$want" != "$have" ]; then
        echo "FAIL  $BASELINE_REL was measured with '$want', this runner has '$have' -- two instruments, not comparable"
        return 4
    fi
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --update-baseline)
        tmp=$(mktemp -d); measure "$tmp/run.json"
        census "$tmp/run.json" > "$tmp/census.txt" || { tail -5 "$tmp/run.json.stderr" >&2; exit 3; }
        write_baseline "$tmp/census.txt"; rm -rf -- "${tmp:?}"; exit 0 ;;
    --from-json)
        [ -n "${2:-}" ] || { echo "--from-json needs a file" >&2; exit 2; }
        json=$2 ;;
    "") json="" ;;
    *) echo "unknown argument '$1'" >&2; exit 2 ;;
esac

echo "=== clippy over every workspace member, shrink-only (check_clippy_member_ratchet.sh, #4152) ==="
[ -f "$BASELINE" ] || { echo "FAIL  $BASELINE_REL is missing"; exit 1; }
check_version || exit $?
# The comparand. This guard runs as a ci/explicit-test-commands.d fragment, and that job fetches
# origin/main only on pull_request. On a push the ref is absent, and the library would (correctly)
# refuse. So fetch it here, with the exact command the library's own refusal prescribes. The
# library still judges; this only supplies the ref it needs. A PR cannot rewrite the ref either way.
if ! git -C "$ROOT" rev-parse -q --verify refs/remotes/origin/main >/dev/null 2>&1; then
    echo "note  origin/main is not fetched here; fetching it as the ratchet's comparand"
    git -C "$ROOT" fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main >/dev/null 2>&1 \
        || echo "note  the fetch failed; the ratchet below will say UNRESOLVABLE and refuse"
fi
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "$ROOT/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "$ROOT" "$BASELINE_REL" keyed || exit 1

tmp=$(mktemp -d)
trap 'rm -rf -- "${tmp:?}"' EXIT
if [ -z "$json" ]; then json="$tmp/run.json"; measure "$json"; fi
if ! census "$json" > "$tmp/census.txt"; then
    echo "FAIL  clippy did not finish, so the census is incomplete and NOT judged. The error(s) that stopped it:"
    errors_of "$json"
    exit 1
fi
# VACUITY: an empty census is legitimate the day every finding is fixed, so emptiness proves
# nothing either way. What proves clippy LOOKED is that it checked every member it was asked to.
want_members=$( cd "$ROOT" && cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c '
import json, sys
ex = {"aprender-gpu", "aprender-cuda-edge", "aprender-compute"}
print(len([p for p in json.load(sys.stdin)["packages"] if p["name"] not in ex]))' ) || want_members=""
got_members=$(checked_members "$json")
if [ -z "$want_members" ] || [ "$got_members" -lt "$want_members" ]; then
    echo "FAIL (vacuity)  clippy checked $got_members workspace member(s), expected ${want_members:-<cargo metadata failed>}: a census of a partial workspace is not a census"
    exit 1
fi
echo "ok    clippy checked all $got_members workspace members (GPU trio excluded, as CI does)"
judge "$BASELINE" "$tmp/census.txt"
