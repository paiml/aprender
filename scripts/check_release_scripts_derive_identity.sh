#!/usr/bin/env bash
# check_release_scripts_derive_identity.sh -- scripts/release/ may not carry a release's
# identity as a literal, nor any path outside the repository (#3618).
#
# THE DEFECT. The release scripts were ported from an out-of-tree 0.68.2 bundle (#3599)
# and kept V=0.68.2, MS=12, EPIC=3477, LAST_TAG=v0.68.1 and
# AP=<an agent worktree under the /mnt RAID>. Cutting 0.69 through them would have
# targeted the old version and a directory that may not exist. The path was invisible
# to every gate: check_hardcoded_paths.sh does not match that mount at all (#3592,
# proved with a two-path probe scoring +1, not +2). The only thing that named it was a
# paragraph in #3617's PR body. A prose warning is not a guard.
#
# THE RULES, over every file under scripts/release/:
#   R1  no absolute path under /mnt /home /Users /opt /srv /media /root -- on ANY line,
#       comments included. Erring strict is correct in a seven-file directory: a comment
#       naming an operator-box path is how the last one survived review.
#   R2  no LITERAL value assigned to a release identity name (V T MS EPIC LAST_TAG AP)
#       on a non-comment line: a value with no `$` in it. `MS=$V`, `T="v$V"` and
#       `AP="${RELEASE_AP:-...}"` are derivations; `V=0.69.0` and `EPIC=3477` are not.
#       scripts/release/lib_release_params.sh is where the identity is derived.
# Zero files scanned is ENV rc=2, never a pass.
#
#   check_release_scripts_derive_identity.sh              judge scripts/release/
#   check_release_scripts_derive_identity.sh --self-test  the case table (fixtures)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
DIR="${RELEASE_SCRIPTS_DIR:-$ROOT/scripts/release}"

R1_RE='(^|[^A-Za-z0-9_.$}-])/(mnt|home|Users|opt|srv|media|root)/'
R2_RE='(^|[;&|[:space:]])(V|T|MS|EPIC|LAST_TAG|AP)=[^[:space:];$]*([[:space:];]|$)'

# judge <dir> -> 0 clean, 1 a finding (each printed), 2 ENV
judge() {
    local dir=$1 f n=0 bad=0 hits
    [ -d "$dir" ] || { printf 'ENV   %s: no such directory -- cannot judge, not a pass\n' "$dir" >&2; return 2; }
    while IFS= read -r -d '' f; do
        n=$((n + 1))
        hits=$(grep -nE "$R1_RE" "$f") || hits=""
        if [ -n "$hits" ]; then
            bad=1
            printf 'FAIL  R1 %s: a path outside the repository:\n' "${f#"$ROOT/"}"
            printf '%s\n' "$hits" | sed 's/^/        /'
        fi
        # R2 reads non-comment lines only; the line number is kept for the report
        hits=$(awk -v re="$R2_RE" '!/^[[:space:]]*#/ && $0 ~ re { printf "%d:%s\n", NR, $0 }' "$f")
        if [ -n "$hits" ]; then
            bad=1
            printf 'FAIL  R2 %s: a release identity assigned a literal (derive it: lib_release_params.sh):\n' "${f#"$ROOT/"}"
            printf '%s\n' "$hits" | sed 's/^/        /'
        fi
    done < <(find "$dir" -maxdepth 1 -type f -print0)
    [ "$n" -gt 0 ] || { printf 'ENV   %s: zero files scanned -- a scan of nothing is not a pass\n' "$dir" >&2; return 2; }
    [ "$bad" -eq 0 ] && printf 'ok    %s file(s) under %s: no out-of-repo path, no literal release identity\n' "$n" "${dir#"$ROOT/"}"
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== release scripts derive their identity: case table ==="
    d=$(mktemp -d) || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    row() { # row WANT_RC LABEL FILE-BODY
        local want=$1 label=$2 body=$3 rc=0; n=$((n + 1))
        rmtree "$d/r"; mkdir -p "$d/r"
        printf '%s' "$body" > "$d/r/x.sh"
        judge "$d/r" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label" >&2; bad=1; fi
    }
    M=/mnt; H=/home   # built, so this file itself carries no literal for R1 to find
    row 0 "derivations only -> clean" \
        $'REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"\nrelease_params "$1" "$REPO_ROOT"\nMS=$V\nT="v$V"\nAP="${RELEASE_AP:-$ROOT/target/release-train/$T}"\nexport PATH="$HOME/.cargo/bin:$PATH"\ncmd > /dev/null 2>&1\n'
    row 1 "the #3618 shape: AP under the RAID mount -> RED (R1 and R2)"  "AP=$M/nvme-raid0/agent-wt/rel-0682-autopilot"$'\n'
    row 1 "that path in a COMMENT -> RED (R1 reads comments)"            "# the precedent is $M/nvme-raid0/agent-wt/rel-067-autopilot/"$'\n'
    row 1 "a home-directory path -> RED"                                 "cd $H/noah/src/aprender"$'\n'
    row 1 "V=0.69.0 -> RED (R2)"                                         $'V=0.69.0; T="v$V"\n'
    row 1 "MS and EPIC literals after another statement -> RED"          $'REPO=paiml/aprender; MS=12; EPIC=3477\n'
    row 1 "LAST_TAG=v0.68.1 -> RED"                                      $'LAST_TAG=v0.68.1\n'
    row 1 "a quoted literal V=\"0.69.0\" -> RED"                         $'V="0.69.0"\n'
    row 0 "an assignment in a COMMENT is documentation -> clean (R2 skips comments)" $'# V=0.68.2 was the old literal\n'
    row 0 "lowercase locals and names that merely end in V/T -> clean"  $'local v=$1 t=$2\nENV=prod\nPLOT=1\n'
    row 0 "/dev/null, /tmp and \$HOME paths are not operator-box literals -> clean" $'x > /dev/null\nmktemp -p /tmp\nls "$HOME/.cargo/bin"\n'
    n=$((n + 1)); rc=0; rmtree "$d/empty"; mkdir -p "$d/empty"; judge "$d/empty" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 2 ] && printf 'ok    row %-2s rc=2  zero files -> ENV, never a pass\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s (wanted 2)  zero files\n' "$n" "$rc" >&2; bad=1; }
    n=$((n + 1)); rc=0; judge "$d/no-such-dir" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 2 ] && printf 'ok    row %-2s rc=2  missing directory -> ENV\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s (wanted 2)  missing directory\n' "$n" "$rc" >&2; bad=1; }
    # and the real directory, so a red tree cannot hide behind green fixtures
    n=$((n + 1)); rc=0; judge "$DIR" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 0 ] && printf 'ok    row %-2s rc=0  the real %s is clean\n' "$n" "${DIR#"$ROOT/"}" \
        || { printf 'FAIL  row %-2s rc=%s  the real %s\n' "$n" "$rc" "${DIR#"$ROOT/"}" >&2; bad=1; }
    [ "$bad" -eq 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== scripts/release/ derives the release identity (check_release_scripts_derive_identity.sh) ==="
judge "$DIR"; rc=$?
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL (rc=$rc)" >&2
exit "$rc"
