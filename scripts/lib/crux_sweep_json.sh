# crux_sweep_json.sh — the JSON steps of scripts/crux_sweep_shards.sh in bash + jq, with no Python.
#
# Each function replaces one inline python3 snippet of the sweep and keeps its output. Where the two can
# differ, the difference is listed here and pinned by a case in scripts/check_crux_sweep_json.sh:
#
#   - A JSON file the sweep rewrites (the merged meta, the greedy-only receipt) keeps Python's layout:
#     same indent, ASCII-only escapes, no final newline. jq writes a number in exponent form canonically
#     (Python's 1e-05 comes out as 0.00001): the value is the same, the bytes are not.
#   - Python carried NaN and Infinity through; jq cannot write them back. Those two files are REFUSED
#     when they hold a non-finite number, never rewritten with a changed value.
#   - A greedy row is copied verbatim, plus a final newline if the shard's last row had none (Python
#     glued the next shard's first row onto it).
#   - A malformed certification or plan input (a non-string id, a mode list that is not a list, a model
#     path holding a tab or newline) is refused. Python crashed on most of these, or wrote a plan line
#     the sweep then misread.
#
# SOURCED: option-neutral (no `set`); every function fails by return status, never by exit.

# Python truthiness of a JSON value (jq alone counts 0, "", [] and {} as true); a --slurpfile that must
# hold exactly one JSON value, as json.load required; the refusal of a non-finite number before a rewrite.
_CRUX_JQ_DEFS='
def pytrue: . != null and . != false and . != 0 and . != "" and . != [] and . != {};
def one($what): if length == 1 then .[0] else error("\($what) is not one JSON value") end;
def finite_only:
  if [.. | numbers | select(isnan or isinfinite)] | length > 0
  then error("it holds a non-finite number (NaN or Infinity), which jq cannot write back unchanged")
  else . end;
'

_crux_sha256() { # <file> — the hex sha256 of one file, or return 1
  local out
  out=$(sha256sum < "$1") || return 1
  printf '%s\n' "${out%% *}"
}

# crux_sweep_plan <certification> <prompts> <scope> <plan> <models-dir>...
# One plan line per certified (sha, mode), sorted by sha, off before on:
#   RUN <sha> <mode> <path> <ids>  |  SKIP <sha> <mode> <path> <why>  |  ABSENT <sha> - - <why>
# A model is a *.gguf entry of a models dir (dotfiles too), hashed in name order; the first path with a
# given sha256 wins. A models dir that does not exist is skipped, as before.
crux_sweep_plan() {
  local cert="$1" prompts="$2" scope="$3" plan="$4"; shift 4
  local bound actual d fd names sorted rest f p sha idx='{}'
  bound=$(jq -nr --slurpfile c "$cert" "$_CRUX_JQ_DEFS"'
    $c | one("the certification") | .prompts_sha256
    | if pytrue | not then "" elif type == "string" then . else error("prompts_sha256 is not a string") end') || return 1
  if [ -n "$bound" ]; then
    actual=$(_crux_sha256 "$prompts") || return 1
    if [ "$bound" != "$actual" ]; then
      printf 'the certification binds prompts sha %s, but %s is %s\n' "${bound:0:12}" "$prompts" "${actual:0:12}" >&2
      return 1
    fi
  fi
  for d in "$@"; do
    [ -d "$d" ] || continue
    # '/' cannot occur in a file name, so it separates the names safely (newlines included).
    fd=$d; case $fd in -*) fd=./$fd ;; esac   # find would read a leading - as an option
    names=$(find -H "$fd" -mindepth 1 -maxdepth 1 -name '*.gguf' -printf '%f/') || return 1
    sorted=$(printf '%s' "$names" | tr '/' '\0' | LC_ALL=C sort -z | tr '\0' '/')
    rest=$sorted
    while [ -n "$rest" ]; do
      f=${rest%%/*}; rest=${rest#*/}
      p="${d%/}/$f"
      sha=$(_crux_sha256 "$p") || return 1
      idx=$(jq -c --arg s "$sha" --arg p "$p" 'if has($s) then . else .[$s] = $p end' <<< "$idx") || return 1
    done
  done
  jq -nr "$_CRUX_JQ_DEFS"'
    ($ARGS.positional | join(" ")) as $dirs
    | ($c | one("the certification")) as $c
    | ($p | one("the prompt set")) as $p
    | [ $p.prompts[] | select(.control | pytrue)
        | if has("id") then .id else error("a control prompt has no id") end ] as $controls
    | ($c.admitted_by_sha_thinking
       | if type == "object" then . else error("the certification has no admitted_by_sha_thinking table") end) as $adm
    | $adm | keys[] as $sha | $adm[$sha] as $modes
    | if ($local | has($sha)) | not then
        "ABSENT\t\($sha)\t-\t-\tcertified, but no file with this sha256 in \($dirs)"
      elif ($modes | type) != "object" then error("admitted_by_sha_thinking[\($sha)] is not an object")
      else
        $local[$sha] as $path
        | if ($path | test("[\t\n]")) then error("model path \($path | tojson) holds a tab or newline") else . end
        | ("off", "on") as $mode
        | (if $modes | has($mode) then $modes[$mode] else [] end) as $listed
        | if ($listed | type) != "array" or ($listed | any(type != "string"))
          then error("admitted ids for \($sha) \($mode) are not a list of strings") else . end
        | [ $listed[] | select($scope == "admitted" or (. as $i | $controls | any(. == $i))) ] as $ids
        | if ($ids | length) == 0 then
            (if $listed | pytrue then "no CONTROL admitted (scope=controls)" else "nothing admitted" end) as $why
            | "SKIP\t\($sha)\t\($mode)\t\($path)\t\($why)"
          else "RUN\t\($sha)\t\($mode)\t\($path)\t\($ids | join(","))" end
      end' \
    --slurpfile c "$cert" --slurpfile p "$prompts" --arg scope "$scope" --argjson local "$idx" \
    --args "$@" > "$plan" || return 1
}

# crux_first_control <prompts> — the id of the first control prompt; return 1 (and print nothing) if
# there is none, as the python3 one-liner did by crashing.
crux_first_control() {
  local id
  id=$(jq -nr --slurpfile p "$1" "$_CRUX_JQ_DEFS"'
    first($p | one("the prompt set") | .prompts[] | select(.control | pytrue)) // error("no control prompt")
    | .id | if type == "string" then . else error("the first control prompt has no string id") end') || return 1
  printf '%s\n' "$id"
}

# crux_greedy_rows <manifest.jsonl> — the rows whose "kind" is "greedy", byte for byte. A row that is not
# a JSON object stops the copy there, with the rows before it already written (as before).
crux_greedy_rows() {
  jq -nRr 'inputs
    | select(fromjson | if type == "object" then .kind == "greedy" else error("a manifest row is not a JSON object") end)' \
    < "$1"
}

# crux_merge_meta <first-meta> <out> <shards.tsv> — the first certified shard's meta, with `models` the
# union (first wins, keyed by sha256) over every certified shard's meta and `merged_shards` the number of
# shard lines. Greedy shards, shards with no work dir and work dirs with no meta.json are left out.
crux_merge_meta() {
  local first="$1" out="$2" tsv="$3" works work k n metas=()
  # The work dir is the 3rd tab field of a line whose name is not greedy-*; empty fields are kept, as split() did.
  works=$(jq -nRr 'inputs | (split("\t") + ["", "", ""])[:3]
    | select((.[0] | startswith("greedy-") | not) and .[2] != "") | .[2]' < "$tsv") || return 1
  while IFS= read -r work; do
    [ -n "$work" ] || continue
    [ -e "$work/meta.json" ] || continue
    # json.load read exactly one value per file; jq inputs would take an empty file as none and two as two.
    k=$(jq -n '[inputs] | length' < "$work/meta.json") || return 1
    if [ "$k" != 1 ]; then printf '%s holds %s JSON values, not one\n' "$work/meta.json" "$k" >&2; return 1; fi
    metas+=("$work/meta.json")
  done <<< "$works"
  n=$(jq -nR '[inputs] | length' < "$tsv") || return 1
  # /dev/null is always the last input, so jq never falls back to stdin when no shard meta qualifies.
  jq -nj -a --indent 2 "$_CRUX_JQ_DEFS"'
    ($first | one("the first meta")) as $meta
    | (reduce inputs as $m ({models: [], seen: []};
        if ($m | type) != "object" then error("a shard meta is not a JSON object") else . end
        | reduce ($m | if has("models") then .models else [] end
                  | if type == "array" then .[] else error("a shard meta models is not a list") end) as $x (.;
            if ($x | type) != "object" then error("a models entry is not an object") else . end
            | $x.sha256 as $s
            | if any(.seen[]; . == $s) then . else .seen += [$s] | .models += [$x] end))
       | .models) as $models
    | $meta | if type == "object" then . else error("the first meta is not a JSON object") end
    | if ($models | length) > 0 then .models = $models else . end
    | .merged_shards = $n
    | finite_only' \
    --slurpfile first "$first" --argjson n "$n" "${metas[@]}" /dev/null > "$out" || return 1
  printf 'merged meta: %s model(s)\n' "$(jq '.models | length' "$out")"
}

# crux_mark_greedy_only <receipt.json> — mark a greedy-only receipt in place: `"greedy_only": true` and
# `"cells": []`. Refused if the judge produced a cell or an empty greedy[].
crux_mark_greedy_only() {
  local r="$1" new
  new=$(jq -nj -a --indent 1 --slurpfile r "$r" "$_CRUX_JQ_DEFS"'
    $r | one("the receipt")
    | if type != "object" then error("the receipt is not a JSON object")
    elif .cells | pytrue then error("a greedy-only merge produced \(.cells | length) judged cells; it must produce none")
    elif .greedy | pytrue | not then error("a greedy-only receipt with an EMPTY greedy[] proves nothing")
    elif (.summary | type) != "object" then error("the receipt has no summary")
    else .greedy_only = true | .cells = [] | finite_only end') || return 1
  printf '%s' "$new" > "$r" || return 1
  jq -r '"greedy_only receipt: \(.greedy | length) greedy entries, verdict \(.summary.verdict
    | if . == null then "None" elif type == "string" then . else tojson end)"' "$r"
}
