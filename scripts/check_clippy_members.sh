#!/usr/bin/env bash
# check_clippy_members.sh — clippy findings in workspace MEMBERS may only shrink (#4152).
#
# `ci / lint` runs `cargo clippy $CLIPPY_ARGS`, and without -p/--workspace that
# lints the ROOT FACADE package: two files. No gate ever ran clippy over the 77
# member crates, so their findings grew invisibly — the #2370 shape, one scope
# over. `cargo clippy -p aprender-serve --all-targets -- -D warnings` alone was
# RED on 1280 excessive_precision + 5 undocumented_unsafe_blocks, one of which
# documented an avx2-only check guarding an avx2+fma function (unsound; fixed).
#
# Findings are keyed <package>/<kind>:<target>/<lint> and counted. The baseline
# (scripts/clippy_members_baseline.txt, `keyed` in check_baseline_ratchets.sh)
# is the ceiling: a NEW key or a RAISED count is RED. A LOWERED count is green
# and the gate prints the rows to shrink; the meta-guard refuses any baseline
# edit that grows against origin/main, so the ceiling cannot be raised by hand.
#
# MEASUREMENT. `--keep-going` + `--cap-lints warn`: under -D warnings a crate
# with findings stops its dependents from being linted at all, so the count
# would depend on build order and shrink by hiding. The three GPU crates are
# excluded exactly as CI's workspace-test excludes them (need a GPU toolchain);
# their cuda-feature lint is check_clippy_cuda.sh.
#
# INSTRUMENT. Clippy's lint set is not monotonic across releases (CLAUDE.md,
# #2370), so a count from another rustc is not comparable. The baseline records
# `# rustc=<version>`; a different rustc exits 2, never 0.
#
# SCOPE. A cold whole-workspace clippy is ~20 min, over the PR budget. The
# default gate lints only the member packages this diff touches (vs
# resolve_base: merge-base with origin/main) and compares only their rows —
# a key is <package>/..., so an unlinted package's rows cannot move. A change
# to the lint instrument (rust-toolchain.toml, clippy.toml, this gate or its
# baseline) can move every package, so it forces --full. Residual, seen only
# by --full (nightly / release): a Cargo.lock / [workspace.lints] change, or a
# dependency API change, that adds findings in an UNCHANGED package. Those are
# not escalated because a batch PR always touches Cargo.lock, and a cold
# whole-workspace run on every batch is exactly the cost this scope removes.
#
#   bash scripts/check_clippy_members.sh                   # the gate, changed packages
#   bash scripts/check_clippy_members.sh --full            # the gate, whole workspace
#   bash scripts/check_clippy_members.sh --update-baseline # re-measure, rewrite
#   bash scripts/check_clippy_members.sh --self-test       # planted lint -> RED
#
# Exit: 0 within baseline, 1 new/raised findings, 2 cannot measure, 3 restore failed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$ROOT/scripts/clippy_members_baseline.txt"
WORKSPACE_SCOPE=(--workspace --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute)
EXCLUDED_PKGS=" aprender-gpu aprender-cuda-edge aprender-compute "

die2() { echo "check_clippy_members: $*; refusing to report clean" >&2; exit 2; }

rustc_version() {
    local v
    v="$(cd "$ROOT" && rustc --version 2>/dev/null)" || die2 "rustc not runnable"
    v="${v#rustc }"
    printf '%s\n' "${v%% *}"
}

# measure <out-keyed-file> <scope args...> — clippy JSON -> sorted "<key>\t<count>".
measure() {
    local out="$1" json err rc=0
    shift
    json="$(mktemp)"
    err="$(mktemp)"
    (cd "$ROOT" && cargo clippy --keep-going "$@" --all-targets --no-deps \
        --message-format=json -- --cap-lints warn -W warnings) >"$json" 2>"$err" || rc=$?
    # A non-zero rc with no finished build is a toolchain/resolution failure, not a count.
    if ! grep -q '"reason":"build-finished"' "$json"; then
        tail -n 20 "$err" >&2
        rm -f "$json" "$err"
        die2 "clippy did not finish (rc=$rc)"
    fi
    jq -r 'select(.reason == "compiler-message" and .message.code != null)
        | (.package_id | split("#") | .[0] as $p | .[1]
            | if test("@") then sub("@.*"; "") else ($p | sub(".*/"; "")) end) as $pkg
        | "\($pkg)/\(.target.kind[0]):\(.target.name)/\(.message.code.code)"' "$json" \
        | LC_ALL=C sort | uniq -c | awk '{ printf "%s\t%s\n", $2, $1 }' >"$out"
    rm -f "$json" "$err"
}

# compare <baseline> <current> — prints NEW/RAISED to stderr (rc 1) and LOWERED to stdout.
# FILENAME, not NR == FNR: an EMPTY baseline (a clean package) makes NR == FNR
# true on the current file too, and every finding then compares green.
compare() {
    LC_ALL=C awk -F'\t' '
        FILENAME == ARGV[1] { if ($0 !~ /^#/ && NF == 2) b[$1] = $2; next }
        {
            if (!($1 in b))          { printf "  + NEW KEY  %s (%s)\n", $1, $2 > "/dev/stderr"; bad = 1 }
            else if ($2 + 0 > b[$1] + 0) { printf "  + RAISED   %s  %s -> %s\n", $1, b[$1], $2 > "/dev/stderr"; bad = 1 }
            else if ($2 + 0 < b[$1] + 0) { printf "  - lowered  %s  %s -> %s (shrink the baseline)\n", $1, b[$1], $2 }
        }
        END { exit bad }
    ' "$1" "$2"
}

require_instrument() {
    local want have
    want="$(grep -m1 -E '^# rustc=' "$BASELINE" 2>/dev/null)" || die2 "$BASELINE has no '# rustc=' header"
    want="${want#\# rustc=}"
    have="$(rustc_version)"
    [ "$want" = "$have" ] || die2 "baseline measured under rustc $want, this box runs $have — two instruments"
}

# changed_packages — member packages touched since the base, one per line, or
# FULL when a workspace-wide input changed.
changed_packages() {
    local files pkgs
    # shellcheck source=scripts/lib/resolve_base.sh
    . "$ROOT/scripts/lib/resolve_base.sh" || die2 "cannot load resolve_base.sh"
    REPO_ROOT="$ROOT" PROG=check_clippy_members resolve_base HEAD || die2 "no base to diff against"
    files="$(git -C "$ROOT" diff --name-only "$BASE_REF" HEAD)" || die2 "git diff $BASE_REF failed"
    files="$files
$(git -C "$ROOT" diff --name-only HEAD)"
    if grep -qxE 'rust-toolchain(\.toml)?|\.?clippy\.toml|scripts/check_clippy_members\.sh|scripts/clippy_members_baseline\.txt' <<<"$files"; then
        echo FULL
        return 0
    fi
    pkgs="$(cd "$ROOT" && cargo metadata --no-deps --format-version 1 2>/dev/null \
        | jq -r '.packages[] | "\(.name)\t\(.manifest_path | sub("/Cargo.toml$"; ""))"')" \
        || die2 "cargo metadata failed"
    [ -n "$pkgs" ] || die2 "cargo metadata listed no packages"
    # Longest package-dir prefix wins: the root facade's dir is a prefix of every crate's.
    # `|| true`: a diff with no .rs/Cargo.toml path is "nothing to lint", not a
    # pipefail death before gate() can say so.
    { grep -E '(\.rs|Cargo\.toml)$' <<<"$files" || true; } | awk -F'\t' -v root="$ROOT/" '
        FILENAME == ARGV[1] { d = $2 "/"; if (d == root) d = ""; else if (index(d, root) == 1) d = substr(d, length(root) + 1); dir[$1] = d; next }
        $0 != "" { best = ""; bl = -1
          for (p in dir) if (index($0, dir[p]) == 1 && length(dir[p]) > bl) { best = p; bl = length(dir[p]) }
          if (best != "") print best }
    ' <(printf '%s\n' "$pkgs") - | LC_ALL=C sort -u | while IFS= read -r p; do
        case "$EXCLUDED_PKGS" in *" $p "*) ;; *) printf '%s\n' "$p" ;; esac
    done
}

gate() {
    command -v jq >/dev/null 2>&1 || die2 "jq not found"
    [ -f "$BASELINE" ] || die2 "no baseline at $BASELINE"
    require_instrument
    local cur base rc=0 full="${1:-}" pkgs="" label p
    local scope=()
    if [ "$full" != --full ]; then
        pkgs="$(changed_packages)"
        [ "$pkgs" = FULL ] && full="--full"
    fi
    if [ "$full" != --full ] && [ -z "$pkgs" ]; then
        echo "check_clippy_members: no member package changed since the base — nothing to lint (--full lints all)"
        return 0
    fi
    cur="$(mktemp)"
    base="$(mktemp)"
    if [ "$full" = --full ]; then
        scope=("${WORKSPACE_SCOPE[@]}")
        label="workspace"
        cp "$BASELINE" "$base"
    else
        while IFS= read -r p; do scope+=(-p "$p"); done <<<"$pkgs"
        label="$(printf '%s' "$pkgs" | tr '\n' ' ')"
        # Only the linted packages' rows: an unlinted package must not read as "lowered".
        awk -F'\t' 'FILENAME == ARGV[1] { want[$0] = 1; next } /^#/ { next } { split($1, k, "/"); if (k[1] in want) print }' \
            <(printf '%s\n' "$pkgs") "$BASELINE" >"$base"
    fi
    measure "$cur" "${scope[@]}"
    if [ "$full" = --full ] && ! [ -s "$cur" ]; then
        die2 "measured zero findings — an empty count from a 77-crate workspace is a broken measurement"
    fi
    compare "$base" "$cur" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "check_clippy_members: RED — findings above scripts/clippy_members_baseline.txt in [$label] (fix them; the baseline never grows)" >&2
        rm -f "$cur" "$base"
        return 1
    fi
    echo "check_clippy_members: within baseline [$label] ($(wc -l <"$cur") keys, $(awk -F'\t' '{ s += $2 } END { print s + 0 }' "$cur") findings)"
    rm -f "$cur" "$base"
}

update_baseline() {
    command -v jq >/dev/null 2>&1 || die2 "jq not found"
    local cur
    cur="$(mktemp)"
    measure "$cur" "${WORKSPACE_SCOPE[@]}"
    {
        echo "# tool_version=none (cargo clippy under rust-toolchain.toml; rustc pinned by the '# rustc=' line, enforced by scripts/check_clippy_members.sh)"
        echo "# rustc=$(rustc_version)"
        echo "# Workspace-member clippy findings, <package>/<kind>:<target>/<lint><TAB><count> (#4152)."
        echo "# SHRINK-ONLY: written by scripts/check_clippy_members.sh --update-baseline; never hand-raise a row."
        cat "$cur"
    } >"$BASELINE"
    echo "check_clippy_members: wrote $(wc -l <"$cur") keys to $BASELINE"
    rm -f "$cur"
}

self_test() {
    command -v jq >/dev/null 2>&1 || die2 "jq not found"
    local target="$ROOT/crates/aprender-common/src/lib.rs" backup base cur
    backup="$(mktemp)"
    base="$(mktemp)"
    cur="$(mktemp)"
    # Scoped to a small member lib that is on main (two dependencies): the property under test
    # is the key/compare path, and it must stay cheap on a cold CI target dir.
    measure "$base" -p aprender-common --lib
    # A SECOND measurement, not `compare "$base" "$base"`: one path passed twice
    # is ARGV[1] on both reads, so compare() would never reach its row logic.
    measure "$cur" -p aprender-common --lib
    if ! compare "$base" "$cur" >/dev/null 2>&1; then
        echo "SELF-TEST FAILED: an unchanged tree compared RED against a re-measurement of itself" >&2
        return 1
    fi
    cp "$target" "$backup"
    # shellcheck disable=SC2064  # expand now: absolute paths, restore survives any cd
    # The backup is deleted ONLY after a successful restore; a failed cp keeps it
    # and names it, so the original is never lost with the tree left mutated.
    trap "if cp '$backup' '$target'; then rm -f '$backup'; else echo \"check_clippy_members: RESTORE FAILED — original kept at $backup\" >&2; exit 3; fi" EXIT
    cat >>"$target" <<'PLANT'

#[allow(dead_code)]
fn check_clippy_members_planted() -> u8 {
    return 1;
}
PLANT
    measure "$cur" -p aprender-common --lib
    if compare "$base" "$cur" >/dev/null 2>&1; then
        echo "SELF-TEST FAILED: a planted needless_return compared green" >&2
        return 1
    fi
    local why
    why="$(compare "$base" "$cur" 2>&1 || true)"
    if [[ "$why" != *needless_return* ]]; then
        echo "SELF-TEST FAILED: RED, but not on the planted needless_return" >&2
        return 1
    fi
    rm -f "$base" "$cur"
    echo "SELF-TEST PASSED: unchanged tree -> green; planted needless_return -> RED"
}

case "${1:-}" in
    --self-test) self_test ;;
    --update-baseline) update_baseline ;;
    --full) gate --full ;;
    "") gate ;;
    *) echo "usage: $0 [--full | --self-test | --update-baseline]" >&2; exit 2 ;;
esac
