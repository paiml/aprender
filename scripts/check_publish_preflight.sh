#!/usr/bin/env bash
# check_publish_preflight.sh — the ONLY way into `cargo publish` (F-9, PMAT-745).
#
# WHY. Until 0.65.0 the cascade ran `cargo publish --allow-dirty --locked` with
# no precondition of its own: a dirty tree, an untagged commit, a commit that
# was not on main, or a release whose dogfood verdict was NO-GO could all be
# uploaded to an immutable registry. The operator's release rule reads
# "publish only through a workflow gate; never --allow-dirty". This script is
# that gate: `scripts/cascade-publish.sh` calls it before the first upload and
# refuses to continue on any non-zero exit, and the `--allow-dirty` is gone.
#
# RULES (each prints its own line; the verdict is the AND of all of them)
#   R1  the tree is clean: no tracked change, no untracked file. cargo package
#       ships every non-ignored file in the tree, so an untracked file is a
#       file the registry would receive that git never saw.
#   R2  the version comes from `cargo metadata` (the root manifest), never
#       from an argument.
#   R3  the tag `v<version>` points at HEAD: the crate that is uploaded is the
#       commit that is tagged, not a neighbour of it.
#   R4  HEAD is an ancestor of the main ref: nothing publishes from a branch.
#   R5  the newest dogfood receipt (`.dogfood/receipt-*.json`, written by
#       scripts/dogfood.sh) says `verdict: GO` for THIS commit and THIS
#       version. A stale receipt, a NO-GO, or no receipt at all refuses.
#
# EXIT  0 every rule holds · 1 a rule refused · 2 the box cannot answer
#       (no git/cargo/python3, not a repository). 2 is not a pass.
#
# SEAMS (the selftest builds a throwaway repository and drives every rule to
# both verdicts through them; production never sets them):
#   PUBLISH_PREFLIGHT_ROOT         repository root (default: this script's repo)
#   PUBLISH_PREFLIGHT_MAIN_REF     the main ref for R4 (default: origin/main)
#   PUBLISH_PREFLIGHT_RECEIPT_DIR  the dogfood receipt dir (default: $ROOT/.dogfood)
#
# USAGE
#   bash scripts/check_publish_preflight.sh             # the gate
#   bash scripts/check_publish_preflight.sh --selftest  # case table, both polarities
set -uo pipefail

PROG=${0##*/}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

die_env() { printf '%s: ENV %s\n' "$PROG" "$*" >&2; exit 2; }

root_version() { # root -> the root manifest's package version, from cargo metadata
    local root="$1"
    cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null \
    | python3 -c '
import json, os, sys
m = json.load(sys.stdin)
root = os.path.realpath(sys.argv[1])
for p in m.get("packages", []):
    if os.path.realpath(p["manifest_path"]) == root:
        print(p["version"]); sys.exit(0)
sys.exit(1)' "$root/Cargo.toml"
}

newest_receipt() { # dir -> path of the newest receipt-*.json, or nothing
    local dir="$1"
    [ -d "$dir" ] || return 0
    find "$dir" -maxdepth 1 -name 'receipt-*.json' -type f 2>/dev/null | LC_ALL=C sort | tail -n 1
}

gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" main_ref="${PUBLISH_PREFLIGHT_MAIN_REF:-origin/main}"
    local fails=0 status version tags head rdir receipt verdict rcommit rversion
    for t in git cargo python3; do
        command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
    done
    if [ -z "$root" ]; then
        root="$(cd -- "$SCRIPT_DIR/.." && pwd)"
    fi
    git -C "$root" rev-parse --verify --quiet HEAD >/dev/null || die_env "$root is not a git repository with a HEAD"
    head="$(git -C "$root" rev-parse HEAD)"

    # R1 clean tree
    status="$(git -C "$root" status --porcelain --untracked-files=all 2>/dev/null)"
    if [ -n "$status" ]; then
        printf 'FAIL  R1 the tree is not clean; cargo package would ship what git never saw:\n%s\n' \
            "$(printf '%s\n' "$status" | sed 's/^/        /' | head -n 20)"
        fails=1
    else
        echo "ok    R1 clean tree (no tracked change, no untracked file)"
    fi

    # R2 version from cargo metadata
    version="$(root_version "$root")" || version=""
    if [ -z "$version" ]; then
        echo "FAIL  R2 cargo metadata names no version for the root manifest"
        fails=1
    else
        echo "ok    R2 version $version (cargo metadata, root manifest)"
    fi

    # R3 the tag points at HEAD
    tags="$(git -C "$root" tag --points-at HEAD 2>/dev/null)"
    if [ -n "$version" ] && printf '%s\n' "$tags" | grep -qx -- "v$version"; then
        echo "ok    R3 tag v$version points at HEAD ${head:0:9}"
    else
        printf 'FAIL  R3 tag v%s does not point at HEAD %s (tags here: %s)\n' \
            "${version:-?}" "${head:0:9}" "${tags:-none}"
        fails=1
    fi

    # R4 HEAD is on main
    if git -C "$root" rev-parse --verify --quiet "${main_ref}^{commit}" >/dev/null \
       && git -C "$root" merge-base --is-ancestor "$head" "$main_ref" 2>/dev/null; then
        echo "ok    R4 HEAD is an ancestor of $main_ref"
    else
        echo "FAIL  R4 HEAD ${head:0:9} is not an ancestor of $main_ref (or that ref does not exist)"
        fails=1
    fi

    # R5 dogfood receipt: GO, this commit, this version
    rdir="${PUBLISH_PREFLIGHT_RECEIPT_DIR:-$root/.dogfood}"
    receipt="$(newest_receipt "$rdir")"
    if [ -z "$receipt" ]; then
        echo "FAIL  R5 no dogfood receipt under $rdir (run scripts/dogfood.sh on this commit)"
        fails=1
    else
        read -r verdict rcommit rversion < <(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("UNREADABLE - -"); sys.exit(0)
print(d.get("verdict") or "-", d.get("commit") or "-", d.get("version") or "-")' "$receipt")
        if [ "$verdict" = GO ] && [ "$rcommit" = "$head" ] && [ "$rversion" = "$version" ]; then
            echo "ok    R5 dogfood receipt $(basename "$receipt"): GO for ${head:0:9} at $version"
        else
            printf 'FAIL  R5 dogfood receipt %s: verdict=%s commit=%s version=%s (need GO, %s, %s)\n' \
                "$(basename "$receipt")" "$verdict" "${rcommit:0:9}" "$rversion" "${head:0:9}" "${version:-?}"
            fails=1
        fi
    fi

    if [ "$fails" -ne 0 ]; then
        echo "REFUSE $PROG: publishing is not allowed from this tree (see the FAIL rows)."
        return 1
    fi
    echo "PASS  $PROG: clean, versioned, tagged, on $main_ref, dogfood GO"
    return 0
}

# --------------------------------------------------------------- selftest ---
selftest() {
    local tmp pass=0 fail=0
    tmp="$(mktemp -d)"
    case "$tmp" in /tmp/*|/var/folders/*|/mnt/*) : ;; *) die_env "mktemp gave ${tmp:-<empty>}, refusing to rm -rf it" ;; esac
    # SEC011: the delete is guarded by the same case the creation was, and an
    # empty or root path is left alone rather than deleted carefully.
    _rm_scratch() {
        local victim="${tmp:-}"
        [ -n "$victim" ] || return 0
        [ "$victim" != "/" ] || return 0
        case "$victim" in
            /tmp/?*|/var/folders/?*|/mnt/?*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
            *) return 0 ;;
        esac
    }
    trap _rm_scratch RETURN

    # A throwaway package repository with a lockfile committed, so `cargo
    # metadata --offline` writes nothing into the tree it is judging.
    build_repo() { # dir
        local d="$1"
        mkdir -p "$d/src"
        printf '[package]\nname = "preflight-fixture"\nversion = "1.2.3"\nedition = "2021"\n\n[dependencies]\n' > "$d/Cargo.toml"
        printf 'pub fn f() {}\n' > "$d/src/lib.rs"
        printf '.dogfood/\n' > "$d/.gitignore"
        git -C "$d" init -q -b fixture-main
        git -C "$d" -c user.name=t -c user.email=t@t config commit.gpgsign false
        ( cd "$d" && cargo metadata --no-deps --offline --format-version 1 >/dev/null 2>&1 )
        git -C "$d" add -A
        git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'fixture' >/dev/null
        git -C "$d" tag v1.2.3
        mkdir -p "$d/.dogfood"
        write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    }
    write_receipt() { # dir, verdict, commit, version
        printf '{"crate":"preflight-fixture","version":"%s","timestamp":"20260903T000000Z","commit":"%s","gates":[],"verdict":"%s"}\n' \
            "$4" "$3" "$2" > "$1/.dogfood/receipt-20260903T000000Z.json"
    }
    row() { # name, expect(0|1), needle, dir
        local name="$1" expect="$2" needle="$3" d="$4" out rc=0
        out="$( PUBLISH_PREFLIGHT_ROOT="$d" PUBLISH_PREFLIGHT_MAIN_REF=fixture-main gate 2>&1 )" || rc=$?
        if [ "$rc" != "$expect" ]; then
            printf '  BROKE %-36s expected exit %s got %s\n' "$name" "$expect" "$rc"; fail=$((fail + 1)); return 0
        fi
        case "$out" in
            *"$needle"*) printf '  ok    %-36s exit=%s said %s\n' "$name" "$expect" "$needle"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-36s exit %s but never said %s\n' "$name" "$expect" "$needle"; fail=$((fail + 1)) ;;
        esac
    }

    local d
    d="$tmp/ok"; build_repo "$d"
    row all_rules_hold                 0 "PASS" "$d"

    d="$tmp/dirty"; build_repo "$d"; printf 'pub fn g() {}\n' >> "$d/src/lib.rs"
    row tracked_change_refuses         1 "FAIL  R1" "$d"

    d="$tmp/untracked"; build_repo "$d"; printf 'x\n' > "$d/stray.out"
    row untracked_file_refuses         1 "FAIL  R1" "$d"

    d="$tmp/notag"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    row missing_tag_refuses            1 "FAIL  R3" "$d"

    d="$tmp/tagelse"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    printf 'pub fn h() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'second' >/dev/null
    git -C "$d" tag v1.2.3 HEAD~1; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row tag_on_another_commit_refuses  1 "FAIL  R3" "$d"

    d="$tmp/branch"; build_repo "$d"; git -C "$d" checkout -q -b topic
    printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'topic' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row head_off_main_refuses          1 "FAIL  R4" "$d"

    d="$tmp/nogo"; build_repo "$d"; write_receipt "$d" NO-GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row dogfood_no_go_refuses          1 "FAIL  R5" "$d"

    d="$tmp/stale"; build_repo "$d"; write_receipt "$d" GO 0000000000000000000000000000000000000000 1.2.3
    row dogfood_stale_commit_refuses   1 "FAIL  R5" "$d"

    d="$tmp/otherver"; build_repo "$d"; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.4
    row dogfood_other_version_refuses  1 "FAIL  R5" "$d"

    d="$tmp/noreceipt"; build_repo "$d"; rm -f "$d/.dogfood/receipt-20260903T000000Z.json"; rmdir "$d/.dogfood"
    row dogfood_receipt_absent_refuses 1 "FAIL  R5" "$d"

    d="$tmp/unreadable"; build_repo "$d"; printf '{not json' > "$d/.dogfood/receipt-20260903T000000Z.json"
    row dogfood_receipt_unreadable_refuses 1 "FAIL  R5" "$d"

    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --selftest) selftest ;;
    '')         gate ;;
    -h|--help)  sed -n '2,40p' "$0" ;;
    *)          printf '%s: unknown argument %s\n' "$PROG" "$1" >&2; exit 2 ;;
esac
