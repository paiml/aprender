#!/usr/bin/env bash
# roadmap_to_fragments.sh — turn a branch's hand edit of docs/roadmaps/roadmap.yaml into fragments
# (T21, operator ruling RQ-8: one writer; contract contracts/apr-roadmap-one-writer-v1.yaml).
#
# WHY. `pmat work add` / `pmat work complete` rewrite roadmap.yaml in place and write no fragment.
# Under one writer a PR commits docs/roadmaps/entries/<ID>.yaml only. This converts the branch's
# change: every entry ADDED or CHANGED between the merge-base and the worktree becomes (or
# overwrites) entries/<ID>.yaml with the worktree's bytes, and roadmap.yaml goes back to the
# merge-base copy. It then proves the conversion: aggregate(merge-base copy, fragments) holds
# every worktree entry byte for byte (order is the aggregator's, not the hand edit's).
#
# An entry REMOVED from roadmap.yaml cannot be expressed as a fragment (fragments supersede,
# they never delete): exit 1, nothing written. Ids that are not filename-safe: exit 1.
#
#   bash scripts/roadmap_to_fragments.sh [--base <ref>]   # default base: merge-base with origin/main
#   bash scripts/roadmap_to_fragments.sh --selftest | --mutants
# EXIT 0 converted (or nothing to convert) · 1 refused · 2 environment
set -uo pipefail
PROG=roadmap_to_fragments.sh
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SELF_DIR/$PROG"
RM=docs/roadmaps/roadmap.yaml
ENT=docs/roadmaps/entries
MUT="${ROADMAP_TO_FRAGMENTS_MUTANT:-}"

env2() { printf '%s: ENV %s — nothing written\n' "$PROG" "$*" >&2; exit 2; }

# blocks: one block per top-level `- id:` line, keyed by its id in memory (legacy ids hold spaces,
# slashes and paths, so they never become file names unless they are ADDED or CHANGED).
AWK_LIB='
function idval(l) { sub(/^- id:[ \t]*/, "", l); gsub(/^[\x27"]|[\x27"][ \t]*$/, "", l); sub(/[ \t]+$/, "", l); return l }
'
# diff_blocks <base> <head> <newdir>: prints REMOVED/UNSAFE/ADDED/CHANGED <tab> id; writes ADDED/CHANGED blocks to newdir
diff_blocks() {
    LC_ALL=C awk -v out="$3" -v mut="$MUT" "$AWK_LIB"'
        FNR == 1 { side = (FILENAME == ARGV[1]) ? 1 : 2; cur = "" }   # by name: an EMPTY first file has no FNR == 1
        /^- id:/ { id = idval($0); if (side == 1) { B[id] = B[id] $0 "\n"; cur = id } else { if (!(id in H)) HO[++hn] = id; H[id] = H[id] $0 "\n"; cur = id }; next }
        cur != "" { if (side == 1) B[cur] = B[cur] $0 "\n"; else H[cur] = H[cur] $0 "\n" }
        END {
            if (mut != "noremove") for (id in B) if (!(id in H)) printf "REMOVED\t%s\n", id
            for (i = 1; i <= hn; i++) {
                id = HO[i]; kind = ""
                if (!(id in B)) kind = "ADDED"; else if (mut != "nochanged" && B[id] != H[id]) kind = "CHANGED"
                if (kind == "") continue
                if (id !~ /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/) { printf "UNSAFE\t%s\n", id; continue }
                printf "%s\t%s\n", kind, id; f = out "/" id ".yaml"; printf "%s", H[id] > f; close(f)
            }
        }' "$1" "$2"
}
# same_blocks <head> <agg>: prints DIFF <tab> id for every head block the aggregate does not hold byte for byte
same_blocks() {
    LC_ALL=C awk "$AWK_LIB"'
        FNR == 1 { side = (FILENAME == ARGV[1]) ? 1 : 2; cur = "" }   # by name: an EMPTY first file has no FNR == 1
        /^- id:/ { cur = idval($0); if (side == 1) { if (!(cur in H)) HO[++hn] = cur; H[cur] = H[cur] $0 "\n" } else A[cur] = A[cur] $0 "\n"; next }
        cur != "" { if (side == 1) H[cur] = H[cur] $0 "\n"; else A[cur] = A[cur] $0 "\n" }
        END { for (i = 1; i <= hn; i++) if (H[HO[i]] != A[HO[i]]) printf "DIFF\t%s\n", HO[i] }' "$1" "$2"
}

convert() {
    local root=$1 base=$2 t n_add n_chg kind id
    t=$(mktemp -d) || env2 "mktemp failed"
    mkdir -p "$t/new" "$t/ent"
    git -C "$root" show "$base:$RM" > "$t/base.yaml" 2> /dev/null || { rm -rf -- "${t:?}"; env2 "cannot read $RM at $base"; }
    diff_blocks "$t/base.yaml" "$root/$RM" "$t/new" > "$t/diff" || { rm -rf -- "${t:?}"; env2 "awk failed on the roadmap diff"; }
    if grep -q -e '^REMOVED' -e '^UNSAFE' "$t/diff"; then
        sed -n -e 's/^REMOVED\t\(.*\)/FAIL  REMOVED \1: a fragment cannot delete an entry; restore it or remove it in its own reviewed PR/p' \
               -e 's/^UNSAFE\t\(.*\)/FAIL  id \1 is not filename-safe; it cannot be a fragment/p' "$t/diff"
        rm -rf -- "${t:?}"; return 1
    fi
    # The header (every line before the first entry) is not entry content, so no fragment can carry it, and the
    # restore below would drop a branch's edit of it without a word (quorum r2 I). Refuse; it needs its own PR.
    if [ "$MUT" != nopreamble ] &&
        ! cmp -s <(LC_ALL=C awk '/^- id:/ { exit } 1' "$t/base.yaml") <(LC_ALL=C awk '/^- id:/ { exit } 1' "$root/$RM"); then
        printf 'FAIL  the %s header (before the first entry) changed; a fragment cannot carry it, so nothing was written\n' "$RM"
        rm -rf -- "${t:?}"; return 1
    fi
    # A CHANGED id whose fragment already says something else than the base copy: the hand edit started from a
    # stale roadmap.yaml, and overwriting the fragment would drop what it holds. Edit the fragment instead.
    # An ADDED id whose fragment already exists with other bytes (it landed on main, unaggregated, after this branch
    # copied roadmap.yaml) would be overwritten the same way (quorum r2 D).
    : > "$t/stale"
    while IFS="$(printf '\t')" read -r kind id; do
        [ "$kind" = CHANGED ] && [ -f "$root/$ENT/$id.yaml" ] && [ "$MUT" != nostale ] &&
            [ -n "$(same_blocks "$root/$ENT/$id.yaml" "$t/base.yaml")" ] && printf '%s\n' "$id" >> "$t/stale"
        [ "$kind" = ADDED ] && [ -f "$root/$ENT/$id.yaml" ] && [ "$MUT" != noadded ] &&
            ! cmp -s -- "$root/$ENT/$id.yaml" "$t/new/$id.yaml" && printf '%s\n' "$id" >> "$t/stale"
    done < "$t/diff"
    if [ -s "$t/stale" ]; then
        sed "s|.*|FAIL  & : $ENT/&.yaml already says something else than the $RM copy you edited; edit the fragment, not $RM|" "$t/stale"
        rm -rf -- "${t:?}"; return 1
    fi
    # Stage: existing fragments + converted blocks, in a temp dir. Nothing in the worktree changes until the proof holds.
    [ -d "$root/$ENT" ] && cp -R -- "$root/$ENT/." "$t/ent/"
    : > "$t/converted.yaml"
    while IFS="$(printf '\t')" read -r kind id; do
        cp -- "$t/new/$id.yaml" "$t/ent/$id.yaml"; cat -- "$t/new/$id.yaml" >> "$t/converted.yaml"
    done < "$t/diff"
    # proof: every CONVERTED entry is in aggregate(base, staged fragments) byte for byte. Only converted ids are
    # compared: an id this branch never touched may legitimately differ when main's own copy is stale.
    if [ "$MUT" != noverify ]; then
        if ! bash "$root/scripts/roadmap_aggregate.sh" --print --roadmap "$t/base.yaml" --entries "$t/ent" > "$t/agg.yaml" 2> "$t/agg.err"; then
            tail -n 5 "$t/agg.err"; rm -rf -- "${t:?}"; printf 'FAIL  the aggregator refuses the converted fragments; nothing written\n'; return 1
        fi
        same_blocks "$t/converted.yaml" "$t/agg.yaml" > "$t/same"
        if [ -s "$t/same" ]; then
            sed 's/^DIFF\t\(.*\)/FAIL  \1: aggregate(base, fragments) differs from the worktree entry; nothing written/' "$t/same" | head -n 10
            rm -rf -- "${t:?}"; return 1
        fi
    fi
    mkdir -p "$root/$ENT"
    while IFS="$(printf '\t')" read -r kind id; do
        cp -- "$t/new/$id.yaml" "$root/$ENT/$id.yaml"; printf '%-8s %s -> %s/%s.yaml\n' "$kind" "$id" "$ENT" "$id"
    done < "$t/diff"
    n_add=$(grep -c -e '^ADDED' "$t/diff"); n_chg=$(grep -c -e '^CHANGED' "$t/diff")
    [ "$MUT" = norestore ] || cp -- "$t/base.yaml" "$root/$RM"
    rm -rf -- "${t:?}"
    printf 'ok    %s added, %s changed -> fragments; %s restored to %s\n' "$n_add" "$n_chg" "$RM" "$base"
}

# ---------------------------------------------------------------- self-test ----
P=0; F=0
row() { if [ "$2" = 0 ]; then printf 'PASS  %s\n' "$1"; P=$((P+1)); else printf 'FAIL  %s — %s\n' "$1" "$3"; F=$((F+1)); fi; }
g() { git -C "$1" -c core.hooksPath=/dev/null -c user.name=st -c user.email=noreply@invalid "${@:2}"; }
fx() {   # fx <dir>: repo with base roadmap A-1, A-3 (+ fragment A-3), aggregator copied
    mkdir -p "$1/scripts" "$1/$ENT" && git init -q "$1" && cp -- "$SELF_DIR/roadmap_aggregate.sh" "$1/scripts/"
    printf "roadmap:\n- id: A-1\n  title: 'one'\n- id: A-3\n  title: 'three'\n" > "$1/$RM"
    printf -- "- id: A-3\n  title: 'three'\n" > "$1/$ENT/A-3.yaml"
    g "$1" add -A && g "$1" commit -q -m base
}
selftest() {
    local d rc
    d=$(mktemp -d) || return 2
    # C1 pmat-style add of A-2 (inserted mid-file) + retitle of A-1 -> two fragments, roadmap restored
    fx "$d/1" > /dev/null 2>&1
    printf "roadmap:\n- id: A-1\n  title: 'uno'\n- id: A-2\n  title: 'two'\n- id: A-3\n  title: 'three'\n" > "$d/1/$RM"
    ( cd "$d/1" && bash "$SELF" --base HEAD ) > "$d/o1" 2>&1; rc=$?
    [ "$rc" = 0 ] && [ -z "$(git -C "$d/1" diff --name-only -- "$RM")" ] && grep -q "title: 'two'" "$d/1/$ENT/A-2.yaml" &&
        grep -q "title: 'uno'" "$d/1/$ENT/A-1.yaml" && grep -q 'ok    1 added, 1 changed' "$d/o1"
    row 'C1 add + retitle -> A-2 and A-1 fragments, roadmap.yaml back at base' $? "rc $rc: $(tail -n 2 "$d/o1")"
    # C2 nothing changed -> no fragment written
    fx "$d/2" > /dev/null 2>&1; ( cd "$d/2" && bash "$SELF" --base HEAD ) > "$d/o2" 2>&1; rc=$?
    [ "$rc" = 0 ] && [ -z "$(git -C "$d/2" status --porcelain)" ] && grep -q 'ok    0 added, 0 changed' "$d/o2"
    row 'C2 untouched roadmap -> nothing written' $? "rc $rc: $(tail -n 2 "$d/o2")"
    # C3 an entry removed -> refused, nothing written
    fx "$d/3" > /dev/null 2>&1; printf "roadmap:\n- id: A-3\n  title: 'three'\n- id: A-4\n  title: 'four'\n" > "$d/3/$RM"
    ( cd "$d/3" && bash "$SELF" --base HEAD ) > "$d/o3" 2>&1; rc=$?
    [ "$rc" = 1 ] && grep -q 'REMOVED A-1' "$d/o3" && [ ! -f "$d/3/$ENT/A-4.yaml" ]
    row 'C3 removed A-1 -> refused, no A-4 fragment' $? "rc $rc: $(tail -n 2 "$d/o3")"
    # C4 an id that cannot be a filename -> refused
    fx "$d/4" > /dev/null 2>&1; printf -- "- id: a/b\n  title: 'x'\n" >> "$d/4/$RM"
    ( cd "$d/4" && bash "$SELF" --base HEAD ) > "$d/o4" 2>&1; rc=$?
    [ "$rc" = 1 ] && grep -q 'not filename-safe' "$d/o4"
    row 'C4 id a/b -> refused' $? "rc $rc: $(tail -n 2 "$d/o4")"
    # C5 a change the aggregator cannot reproduce (a CR in the entry) -> refused by the proof
    fx "$d/5" > /dev/null 2>&1; printf "roadmap:\n- id: A-1\n  title: 'one'\r\n- id: A-3\n  title: 'three'\n" > "$d/5/$RM"
    ( cd "$d/5" && bash "$SELF" --base HEAD ) > "$d/o5" 2>&1; rc=$?
    [ "$rc" = 1 ] && [ ! -f "$d/5/$ENT/A-1.yaml" ] && LC_ALL=C grep -q "$(printf "'one'\r")" "$d/5/$RM"
    row 'C5 CR in a changed entry -> refused, NOTHING written (no fragment, roadmap.yaml untouched)' $? "rc $rc: $(tail -n 2 "$d/o5")"
    # C6 an UNCHANGED legacy id that is no file name (spaces, slashes) next to an add -> converted, not refused
    fx "$d/6" > /dev/null 2>&1; printf -- "- id: GH-1/2: a legacy title\n  title: 'old'\n" >> "$d/6/$RM"; g "$d/6" commit -q -am legacy
    printf -- "- id: A-9\n  title: 'nine'\n" >> "$d/6/$RM"
    ( cd "$d/6" && bash "$SELF" --base HEAD ) > "$d/o6" 2>&1; rc=$?
    [ "$rc" = 0 ] && [ -f "$d/6/$ENT/A-9.yaml" ] && grep -q 'ok    1 added, 0 changed' "$d/o6"
    row 'C6 unchanged legacy id GH-1/2 beside an add -> converted, not refused' $? "rc $rc: $(tail -n 2 "$d/o6")"
    # C7 main's roadmap.yaml is stale vs its own fragment A-3; the branch only adds A-2 -> converted (only A-2 compared)
    fx "$d/7" > /dev/null 2>&1; printf -- "- id: A-3\n  title: 'THREE NEW'\n" > "$d/7/$ENT/A-3.yaml"; g "$d/7" commit -q -am 'fragment, no regen'
    printf "roadmap:\n- id: A-1\n  title: 'one'\n- id: A-2\n  title: 'two'\n- id: A-3\n  title: 'three'\n" > "$d/7/$RM"
    ( cd "$d/7" && bash "$SELF" --base HEAD ) > "$d/o7" 2>&1; rc=$?
    [ "$rc" = 0 ] && [ -f "$d/7/$ENT/A-2.yaml" ] && grep -q "THREE NEW" "$d/7/$ENT/A-3.yaml"
    row 'C7 main stale vs fragment A-3, branch adds A-2 -> converted, A-3 fragment kept' $? "rc $rc: $(tail -n 2 "$d/o7")"
    # C8 the branch edits A-3 in a STALE roadmap.yaml while fragment A-3 says more -> refused, fragment kept
    fx "$d/8" > /dev/null 2>&1; printf -- "- id: A-3\n  title: 'THREE NEW'\n" > "$d/8/$ENT/A-3.yaml"; g "$d/8" commit -q -am 'fragment, no regen'
    printf "roadmap:\n- id: A-1\n  title: 'one'\n- id: A-3\n  title: 'tres'\n" > "$d/8/$RM"
    ( cd "$d/8" && bash "$SELF" --base HEAD ) > "$d/o8" 2>&1; rc=$?
    [ "$rc" = 1 ] && grep -q 'edit the fragment' "$d/o8" && grep -q "THREE NEW" "$d/8/$ENT/A-3.yaml"
    row 'C8 edit of A-3 in a stale roadmap.yaml under a newer fragment -> refused, fragment kept' $? "rc $rc: $(tail -n 2 "$d/o8")"
    # C9 the branch adds A-5 in roadmap.yaml while fragment A-5 already exists with other bytes -> refused, kept (D)
    fx "$d/9" > /dev/null 2>&1; printf -- "- id: A-5\n  title: 'FIVE MAIN'\n" > "$d/9/$ENT/A-5.yaml"; g "$d/9" add -A; g "$d/9" commit -q -m 'fragment, no regen'
    printf -- "- id: A-5\n  title: 'five'\n" >> "$d/9/$RM"
    ( cd "$d/9" && bash "$SELF" --base HEAD ) > "$d/o9" 2>&1; rc=$?
    [ "$rc" = 1 ] && grep -q 'edit the fragment' "$d/o9" && grep -q "FIVE MAIN" "$d/9/$ENT/A-5.yaml"
    row 'C9 add of A-5 over an existing, different fragment A-5 -> refused, fragment kept' $? "rc $rc: $(tail -n 2 "$d/o9")"
    # C10 the same add when the fragment already holds those bytes -> converted (idempotent re-run)
    fx "$d/10" > /dev/null 2>&1; printf -- "- id: A-5\n  title: 'five'\n" > "$d/10/$ENT/A-5.yaml"; g "$d/10" add -A; g "$d/10" commit -q -m frag
    printf -- "- id: A-5\n  title: 'five'\n" >> "$d/10/$RM"
    ( cd "$d/10" && bash "$SELF" --base HEAD ) > "$d/o10" 2>&1; rc=$?
    [ "$rc" = 0 ] && grep -q 'ok    1 added, 0 changed' "$d/o10"
    row 'C10 add of A-5 equal to its existing fragment -> converted' $? "rc $rc: $(tail -n 2 "$d/o10")"
    # C11 a header edit beside an add -> refused, nothing written (the restore would drop it) (I)
    fx "$d/11" > /dev/null 2>&1
    printf "# owner: x\nroadmap:\n- id: A-1\n  title: 'one'\n- id: A-3\n  title: 'three'\n- id: A-6\n  title: 'six'\n" > "$d/11/$RM"
    ( cd "$d/11" && bash "$SELF" --base HEAD ) > "$d/o11" 2>&1; rc=$?
    [ "$rc" = 1 ] && grep -q 'header' "$d/o11" && [ ! -f "$d/11/$ENT/A-6.yaml" ] && grep -q '# owner: x' "$d/11/$RM"
    row 'C11 header edit beside an add -> refused, nothing written, roadmap.yaml untouched' $? "rc $rc: $(tail -n 2 "$d/o11")"
    rm -rf -- "${d:?}"
    printf '%s --selftest: %s PASS, %s FAIL\n' "$PROG" "$P" "$F"
    [ "$F" = 0 ]
}
mutants() {
    local m k=0 n=0
    for m in noremove nochanged norestore noverify nostale noadded nopreamble; do
        n=$((n+1))
        if ROADMAP_TO_FRAGMENTS_MUTANT=$m bash "$SELF" --selftest > /dev/null 2>&1; then printf 'SURVIVED  %s\n' "$m"
        else k=$((k+1)); printf 'killed    %s\n' "$m"; fi
    done
    printf '%s --mutants: %s/%s killed\n' "$PROG" "$k" "$n"
    [ "$k" = "$n" ]
}

case "${1:-}" in
    --selftest) selftest; exit $? ;;
    --mutants) mutants; exit $? ;;
    -h|--help) sed -n '2,17p' "$SELF"; exit 0 ;;
esac
ROOT=$(git rev-parse --show-toplevel 2> /dev/null) || env2 "not inside a git repository"
BASE=""
if [ "${1:-}" = --base ]; then BASE=${2:-}; [ -n "$BASE" ] || env2 "--base needs a ref"; fi
if [ -z "$BASE" ]; then BASE=$(git -C "$ROOT" merge-base HEAD origin/main 2> /dev/null) || env2 "no merge-base with origin/main"; fi
convert "$ROOT" "$BASE"
