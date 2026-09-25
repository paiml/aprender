# cascade_universe.sh - the bash+jq twin of cascade_universe.py, for the release path (#4352).
#
# The cop's 0.70.1 ruling put python3 out of the release path. publish_strict.sh read its
# universe from cascade_universe.py; this is the same enumeration with the same TSV, the same
# refusals and the same exit codes, so publish_strict needs no python3. The .py stays for the
# other consumers (the coverage guard, the publish-safety scan, the Makefile, the drain) until
# they move; the parity case table in #4352 runs both on the real tree and on refusal fixtures.
#
# Source it (option-neutral: no `set` here; failure is the return status):
#     . scripts/lib/cascade_universe.sh || exit 2
#     cascade_universe <repo-root>            # <name>\t<version>\t<manifest>\t<workspace root>
#     cascade_universe --names <repo-root>    # names only
# or run it: bash scripts/lib/cascade_universe.sh [--names] <repo-root>
#
# Every rule is the .py's, kept in the same order:
#   - every workspace in CU_WORKSPACES is read with `cargo metadata --no-deps`; a failure or an
#     empty stdout is rc 2 (the enumeration is broken, never "no crates");
#   - `publish = false` (metadata "publish": []) is not in the universe;
#   - a name publishable from two DIFFERENT manifests is rc 2; the same manifest twice is the
#     later workspace's row;
#   - the workspace root is normpath(abspath(root)/ws): abspath joins the PHYSICAL cwd (python's
#     os.getcwd()), normpath is lexical and never resolves a symlink;
#   - rows sort by name (codepoint order, as python sorts str); fewer than CU_MIN_CRATES is rc 2.
CU_WORKSPACES=(. crates/facades)
CU_MIN_CRATES=70

# normpath: python's posixpath.normpath, lexically (exactly two leading slashes survive).
CU_JQ='def normpath:
  . as $p
  | (if ($p | startswith("//")) and (($p | startswith("///")) | not) then 2
     elif ($p | startswith("/")) then 1 else 0 end) as $lead
  | (reduce ($p | split("/")[]) as $c ([];
       if $c == "" or $c == "." then .
       elif $c == ".." then
         (if length > 0 and .[-1] != ".." then .[:-1] elif $lead > 0 then . else . + [".."] end)
       else . + [$c] end)) as $parts
  | ([range($lead)] | map("/") | join("")) + ($parts | join("/"))
  | if . == "" then "." else . end;'

# the rows: one per name (a second manifest for a name is refused), sorted, at least $min
read -r -d '' CU_ROWS_JQ <<'JQ' || :
    (reduce .[] as $r ({};
        if has($r[0]) and .[$r[0]][1] != $r[2] then
            error("cascade_universe: `\($r[0])` is publishable from two workspaces:\n  \(.[$r[0]][1])\n  \($r[2])")
        else .[$r[0]] = $r[1:] end)) as $seen
    | if ($seen | length) < $min then
        error("cascade_universe: enumerated only \($seen | length) publishable crate(s), expected at least \($min). The ENUMERATION is broken, not the repo.")
      else $seen | keys[] as $n
        | if $names == 1 then $n else ([$n] + .[$n]) | join("\t") end
      end
JQ

cascade_universe() {
    local names=0 root="" a ws manifest out wsroot cwd rows
    for a in "$@"; do
        case $a in
            --names) names=1 ;;
            --*) ;;
            *) [ -n "$root" ] || root=$a ;;
        esac
    done
    [ -n "$root" ] || root=.
    command -v jq > /dev/null 2>&1 || { echo "cascade_universe: jq not found" >&2; return 2; }
    cwd=$(pwd -P) || return 2
    case $root in /*) a=$root ;; *) a="${cwd%/}/$root" ;; esac  # posixpath.join: no second slash after "/"
    rows=""
    for ws in "${CU_WORKSPACES[@]}"; do
        manifest="$root/$ws/Cargo.toml"
        if ! out=$(cargo metadata --no-deps --format-version 1 --manifest-path "$manifest" 2> /dev/null) \
            || [ -z "${out//[[:space:]]/}" ]; then
            printf 'cascade_universe: cargo metadata failed for %s\n' "$manifest" >&2
            return 2
        fi
        wsroot=$(jq -rn --arg p "$a/$ws" "$CU_JQ"' $p | normpath') || return 2
        out=$(printf '%s' "$out" | jq -c --arg w "$wsroot" \
            '.packages[] | select(.publish != []) | [.name, .version, .manifest_path, $w]') || {
            printf 'cascade_universe: unreadable cargo metadata for %s\n' "$manifest" >&2
            return 2
        }
        rows+="$out"$'\n'
    done
    printf '%s' "$rows" | jq -rs --argjson min "$CU_MIN_CRATES" --argjson names "$names" "$CU_ROWS_JQ" \
        2> >(sed 's/^jq: error (at <stdin>:[0-9]*): //' >&2) || return 2
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    cascade_universe "$@"
    exit $?
fi
