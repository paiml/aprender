# crux_dogfood_json.sh — the JSON and YAML steps of scripts/crux_inference_dogfood.sh in bash + jq, with no Python.
#
# Each function replaces one inline python3 snippet of the dogfood and writes the bytes Python wrote: the same
# key order, `json.dumps` separators (", " and ": "), ASCII-only escapes (é, surrogate pairs past U+FFFF,
# \u007f), Python's float repr (0.0, 1e-05, 1e+16) and int(), and indent=2 for meta.json. Where the two can
# differ, the difference is listed here and pinned by a case in scripts/tests/crux_dogfood_json_test.sh:
#
#   - NaN and Infinity were written through by Python; jq cannot write them back, so a value holding one is
#     REFUSED. So are a lone surrogate escape (jq does not parse one) and a NUL inside a text file's content or
#     an id (a shell variable cannot hold it).
#   - A JSON number keeps the literal it was read with: 1.0 stays a float and 2 an int, as in Python. jq
#     folds an exponent literal into a whole number when its mantissa has a fraction or its exponent is 0
#     (1.5e1, 0.5E1, 100e0), and that reads back as an int, where Python wrote 15.0, 5.0 and 100.0. An
#     integer mantissa with a nonzero exponent (1E5, 1e2) keeps its exponent and is written as Python does.
#   - A number with more than 17 significant digits is refused wherever it is read as a double (a float
#     literal, int() of one, float() of text): jq rounds it to 17 digits and then to a double, which can land
#     one step away from the double Python read. 17 digits or fewer convert exactly as Python does.
#   - int() of a JSON float past 2^53-1 is refused (Python printed every digit). int() of text reads ASCII
#     digits with ASCII whitespace around them, and float() of text reads ASCII decimals; underscores, other
#     scripts' digits, "inf" and "nan" are refused.
#   - A prompt set that is refused writes no file at all (Python left the files of the prompts before the bad
#     one). A prompt id must be a string with no "/" (Python wrote an int id's files, and a "/" outside $WORK).
#   - An exit status that is not plain ASCII digits is null (Python crashed on "--5"); an ollama version that
#     is not a string is no answer (Python printed it).
#   - Invalid UTF-8 in an input becomes U+FFFD (Python refused the file); an invalid byte in an argument
#     becomes U+FFFD (Python wrote a lone-surrogate escape). A UTF-8 BOM before the JSON and a number with a
#     leading zero (01) are read (Python refused the file).
#   - A lone CR inside a manifest line is refused (Python's text mode split the line there); CR LF is read.
#   - evidence/crux/hf-sources.yaml is read as a strict subset of YAML: two-space block mappings, plain
#     scalars of [A-Za-z0-9_./-] and comments. Anything else (flow style, quotes, tabs, CR LF, a key YAML 1.1
#     reads as a bool, null, number or date) is REFUSED with a message, so the GGUF gets no declared HF
#     source; scripts/tests/crux_dogfood_json_test.sh holds the real file to the subset. A null dtype or a
#     non-string repo, revision or dtype is refused too (Python printed None, or the value's repr).
#
# SOURCED: option-neutral (no `set`); every function fails by return status, never by exit.

_CRUX_DF_JQ='
def pytrue: . != null and . != false and . != 0 and . != "" and . != [] and . != {};
def one($what): if length == 1 then .[0] else error("\($what) is not one JSON value") end;
def pyblank: test("^[\t\n\u000b\f\r\u001c-\u001f \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$");
def zeros($n): if $n > 0 then "0" * $n else "" end;
# The significant digits of a decimal literal. jq rounds a literal to 17 of them and then to a double, so
# past 17 the double can land one step away from the correctly rounded one Python reads; such a literal is refused.
def _sig: sub("[eE].*$"; "") | gsub("[^0-9]"; "") | sub("^0+"; "") | sub("0+$"; "") | length;
def sig17: if tojson | _sig > 17 then error("a number with more than 17 significant digits is refused (jq rounds it twice)") else . end;
def _hex4: [(. / 4096 | floor) % 16, (. / 256 | floor) % 16, (. / 16 | floor) % 16, . % 16]
  | map("0123456789abcdef"[.:. + 1]) | join("");
# json.dumps of a string, ensure_ascii: tojson escapes the controls and DEL as Python does; past ASCII, \uXXXX.
def pystr:
  if test("^[ -~]*$") then tojson
  else [tojson | explode[]
        | if . < 127 then .
          elif . < 65536 then ("\\u" + _hex4 | explode[])
          else (. - 65536) as $c
          | ("\\u" + (55296 + ($c / 1024 | floor) | _hex4) + "\\u" + (56320 + $c % 1024 | _hex4) | explode[]) end]
    | implode end;
# repr() of a finite double: the shortest round-trip digits (jq and Python both use dtoa mode 0), laid out as
# Python does. pp is where the point falls in the digits; exponent form iff pp < -3 or pp > 16.
def _pyfloat:
  (tojson | capture("^(?<s>-?)(?<i>[0-9]*)(?:\\.(?<f>[0-9]*))?(?:[eE](?<e>[-+]?[0-9]+))?$")
    // error("unexpected number text \(tojson)")) as $m
  | ($m.i + ($m.f // "")) as $d0
  | ($d0 | sub("^0+"; "")) as $d1
  | (($m.i | length) + ($m.e // "0" | tonumber) - (($d0 | length) - ($d1 | length))) as $pp
  | ($d1 | sub("0+$"; "")) as $d
  | if $d == "" then $m.s + "0.0"
    elif $pp < -3 or $pp > 16 then
      ($pp - 1) as $x
      | $m.s + $d[0:1] + (if ($d | length) > 1 then "." + $d[1:] else "" end) + "e"
        + (if $x < 0 then "-" else "+" end) + ($x | fabs | tostring | if length < 2 then "0" + . else . end)
    elif $pp <= 0 then $m.s + "0." + zeros(-$pp) + $d
    elif $pp >= ($d | length) then $m.s + $d + zeros($pp - ($d | length)) + ".0"
    else $m.s + $d[0:$pp] + "." + $d[$pp:] end;
# json.dumps of a number: an int literal is a Python int, anything else a float.
def pynum:
  tojson as $t
  | if $t | test("^-?[0-9]+$") then (if $t == "-0" then "0" else $t end)
    elif isnan or isinfinite then error("a non-finite number (NaN or Infinity) cannot be written back unchanged")
    else sig17 | . * 1 | _pyfloat end;
# json.dumps($v) with indent null (", " and ": ") or a number of spaces; no final newline.
def pydump($ind):
  def d($pad):
    if type == "object" or type == "array" then
      (if type == "object" then ["{", "}", [to_entries[] | [(.key | pystr) + ": ", .value]]]
       else ["[", "]", [.[] | ["", .]]] end) as [$o, $c, $kv]
      | if ($kv | length) == 0 then $o + $c
        elif $ind == null then $o + ([$kv[] | .[0] + (.[1] | d($pad))] | join(", ")) + $c
        else ($pad + (" " * $ind)) as $p
        | $o + "\n" + ([$kv[] | $p + .[0] + (.[1] | d($p))] | join(",\n")) + "\n" + $pad + $c end
    elif type == "string" then pystr
    elif type == "number" then pynum
    elif type == "boolean" then tostring
    else "null" end;
  d("");
# int() of a JSON value, as a jq number (an int literal is kept exactly).
def _intlit: capture("^(?<s>[-+]?)0*(?<d>[0-9]+)$") | if .d == "0" then 0 else (if .s == "-" then "-" else "" end) + .d | fromjson end;
def pyint:
  if type == "boolean" then (if . then 1 else 0 end)
  elif type == "number" then
    if tojson | test("^-?[0-9]+$") then .
    elif isnan or isinfinite then error("int() of a non-finite number")
    elif fabs > 9007199254740991 then error("int() of a float past 2^53-1 is refused here")
    else sig17 | . * 1 | trunc end
  elif type == "string" then
    (capture("^[ \t\n\r\f\u000b]*(?<n>[-+]?[0-9]+)[ \t\n\r\f\u000b]*$") // error("int() of \(tojson)")) | .n | _intlit
  else error("int() of a \(type)") end;
# str(int) of what pyint gave: digits, never -0.
def intstr: tojson | if . == "-0" then "0" else . end;
# float() of text, as a jq number that pynum writes as a float.
def pyfloat_of_text:
  (capture("^[ \t\n\r\f\u000b]*(?<s>[-+]?)(?:(?<i>[0-9]+)(?:\\.(?<f>[0-9]*))?|\\.(?<g>[0-9]+))(?:[eE](?<e>[-+]?[0-9]+))?[ \t\n\r\f\u000b]*$")
    // error("float() of \(tojson)"))
  | ((if .s == "-" then "-" else "" end) + ((.i // "0") | sub("^0+(?=[0-9])"; "")) + "." + ((.f // .g // "") + "0")
     + (if .e then "e" + .e else "" end) | fromjson | sig17 | . * 1)
  | if isnan or isinfinite then error("float() gave a non-finite number") else _pyfloat | fromjson end;
'

# crux_dogfood_jq_ok — return 0 when this jq keeps a number's literal (1.0 stays 1.0, every digit kept) and
# has --raw-output0, which every step below relies on (jq 1.7 and later); else say why on stderr, return 1.
# An older jq reads 1.0 back as 1 and would write different bytes without failing.
crux_dogfood_jq_ok() {
  local t
  t=$(jq -rn '[1.0, 100000000000000000001] | tojson' 2>/dev/null) || t=""
  if [ "$t" != '[1.0,100000000000000000001]' ]; then
    printf '%s does not keep number literals (it wrote %s); jq 1.7 or later is needed\n' \
      "$(jq --version 2>/dev/null || printf 'jq')" "${t:-nothing}" >&2
    return 1
  fi
  if ! jq -n --raw-output0 '""' > /dev/null 2>&1; then
    printf 'this jq has no --raw-output0; jq 1.7 or later is needed\n' >&2
    return 1
  fi
}

# crux_ollama_version — `version` of the /api/version answer on stdin; return 1 (and print nothing) when the
# answer is not one JSON object with a string version.
crux_ollama_version() {
  jq -sr "$_CRUX_DF_JQ"'one("the /api/version answer")
    | if type != "object" then error("the answer is not an object")
      elif has("version") | not then ""
      elif .version | type == "string" then .version
      else error("version is not a string") end'
}

# crux_prompt_files <prompts> <work> <only> — the prompt set's per-prompt files in <work>, and on stdout one
# "run <id>" / "chat <id>" line per runnable (verb, prompt). <only>: comma-separated ids, or empty for all.
# An id <only> names that the set does not hold is refused with status 3.
crux_prompt_files() {
  local prompts="$1" work="$2" only="$3" ops op name body rc=0
  ops=$(mktemp) || return 1
  # The program writes op, name, body triples, NUL-terminated; they are replayed only once all of them exist.
  jq -n --raw-output0 --slurpfile d "$prompts" --arg only "$only" "$_CRUX_DF_JQ"'
    def id_ok: if type == "string" and (test("/") | not) then . else error("prompt id \(tojson) is not a string free of /") end;
    def txt: if type == "string" then . else error("a message content is not a string") end;
    ($d | one("the prompt set") | if type == "object" then . else error("the prompt set is not an object") end) as $d
    | ($d.prompts | if type == "array" then . else error("the prompt set has no prompts list") end) as $ps
    | [$ps[] | if type == "object" then .id | id_ok else error("a prompt is not an object") end] as $ids
    | [$only | split(",")[] | select(. != "")] as $only
    | ([$only[] | select(. as $o | $ids | any(.[]; . == $o) | not)] | unique) as $unknown
    | if ($unknown | length) > 0 then
        "x", "", "--only-prompts names ids the prompt set does not hold: \($unknown | join(", "))"
      else
        ("w", "serve-prompts.jsonl", "", "w", "code-prompts.txt", ""),
        ($d.max_tokens as $glob
        | $ps[] | select(($only | length) == 0 or (.id as $i | $only | any(.[]; . == $i)))
        | . as $p | .id as $id
        | (.messages | if type == "array" and length > 0 then . else error("prompt \($id): messages is not a non-empty list") end) as $msgs
        | ($msgs | map(if type == "object" then . else error("prompt \($id): a message is not an object") end)) as $msgs
        | (if $p | has("max_tokens") then $p.max_tokens else $glob end) as $mt
        | [("off", "on") as $th | if ($mt | type) == "object" then $mt[$th] else $mt end | pyint | intstr] as [$off, $on]
        | ($p | if has("verb") then .verb else "run" end) as $v
        | (if ($v | type) == "array" then $v else [$v] + (if $v == "run" then ["serve run", "serve stream"] else [] end) end) as $verbs
        | [$verbs[] | select(. == "serve run" or . == "serve stream")] as $sv
        | [$msgs[] | select(.role == "user") | if has("content") then .content | txt else error("prompt \($id): a user message has no content") end] as $users
        | ("w", "prompt-\($id).txt", ($msgs[-1] | if has("content") then .content | txt else error("prompt \($id): the last message has no content") end)),
          ("w", "prompt-\($id).json", ($p | pydump(null))),
          ("w", "messages-\($id).json", ({messages: $p.messages} | pydump(null))),
          ("w", "turns-\($id).json", ($users | pydump(null))),
          ("w", "turns-\($id).txt", ($users | map(. + "\n") | join(""))),
          ("w", "maxtok-\($id)-off.txt", $off), ("w", "maxtok-\($id)-on.txt", $on), ("w", "maxtok-\($id).txt", $off),
          (if ($sv | length) > 0 then ("a", "serve-prompts.jsonl", ([$id, $sv] | pydump(null)) + "\n") else empty end),
          (if any($verbs[]; . == "code") then ("a", "code-prompts.txt", $id + "\n") else empty end),
          ($verbs[] | select(. == "run" or . == "chat") | ("p", "", "\(.) \($id)")))
      end' > "$ops" || { rm -f "${ops:?}"; return 1; }
  while IFS= read -r -d '' op && IFS= read -r -d '' name && IFS= read -r -d '' body; do
    case $op in
      w) printf '%s' "$body" > "$work/$name" || rc=1 ;;
      a) printf '%s' "$body" >> "$work/$name" || rc=1 ;;
      p) printf '%s\n' "$body" ;;
      x) printf '%s\n' "$body" >&2; rc=3 ;;
    esac
  done < "$ops"
  rm -f "${ops:?}"
  return "$rc"
}

# crux_max_tokens <prompts> <off|on> — the set's global max_tokens, else the largest per-prompt budget of the
# mode over EVERY prompt of the set.
crux_max_tokens() {
  jq -nr --slurpfile d "$1" --arg mode "$2" "$_CRUX_DF_JQ"'
    ($d | one("the prompt set") | if type == "object" then . else error("the prompt set is not an object") end) as $d
    | if $d.max_tokens != null then $d.max_tokens | pyint | intstr
      else [$d.prompts | if type == "array" then .[] else error("no prompts list") end
            | if type == "object" and (.max_tokens | type) == "object" and (.max_tokens | has($mode))
              then .max_tokens[$mode] | pyint else error("a prompt has no max_tokens.\($mode)") end]
        | if length == 0 then error("no prompt to take max_tokens from") else max | intstr end end'
}

# crux_control_ids <prompts> — the ids of the control prompts, space-separated, in set order.
crux_control_ids() {
  jq -nr --slurpfile d "$1" "$_CRUX_DF_JQ"'
    [$d | one("the prompt set") | .prompts | if type == "array" then .[] else error("the prompt set has no prompts list") end
     | if type == "object" then . else error("a prompt is not an object") end | select(.control | pytrue)
     | .id | if type == "string" then . else error("a control prompt id is not a string") end] | join(" ")'
}

# crux_emit_gen <manifest> <engine> <prompt_id> <rc> <stdout> <stderr> <refused> <ollama_unloaded> <sha> <host>
#               <verb> <thinking> <backend> <mode> — append one gen row.
crux_emit_gen() {
  jq -nj --arg eng "$2" --arg pid "$3" --arg rc "$4" --arg o "$5" --arg e "$6" --arg ref "$7" --arg unl "$8" \
    --arg sha "$9" --arg host "${10}" --arg verb "${11}" --arg think "${12}" --arg be "${13}" --arg mode "${14}" \
    "$_CRUX_DF_JQ"'
    def orNull: if . == "" then null else . end;
    {kind: "gen", engine: $eng, prompt_id: $pid,
     rc: ($rc | if test("^-?[0-9]+$") then _intlit else null end),
     stdout: ($o | orNull), stderr: ($e | orNull), refused: ($ref | orNull),
     model_sha256: $sha, host: $host, verb: $verb, thinking: $think, backend: $be}
    | if $unl != "" then .ollama_unloaded = ($unl == "true") else . end
    | if $mode != "" then .mode = $mode else . end
    | pydump(null) + "\n"' >> "$1"
}

# crux_rows_for <manifest> <engine> <prompt_id> <sha> <verb> — how many gen rows match.
crux_rows_for() {
  jq -nR --arg eng "$2" --arg pid "$3" --arg sha "$4" --arg verb "$5" "$_CRUX_DF_JQ"'
    [inputs | select(pyblank | not) | fromjson
     | if type == "object" then . else error("a manifest row is not a JSON object") end
     | select(.kind == "gen" and .engine == $eng and .prompt_id == $pid and .model_sha256 == $sha and .verb == $verb)]
    | length' < "$1"
}

# crux_thinking_capable <props.json> — true/false: llama-server /props chat_template mentions enable_thinking
# or <think>; return 1 (and print nothing) when that cannot be read.
crux_thinking_capable() {
  jq -nr --slurpfile p "$1" "$_CRUX_DF_JQ"'
    $p | one("props") | if type == "object" then . else error("props is not an object") end
    | (.chat_template | if pytrue then . else "" end) as $t
    | if ($t | type) == "string" then ($t | contains("enable_thinking") or contains("<think>"))
      elif ($t | type) == "array" then any($t[]; . == "enable_thinking" or . == "<think>")
      elif ($t | type) == "object" then ($t | has("enable_thinking") or has("<think>"))
      else error("chat_template is a \($t | type)") end'
}

# crux_tokreq <rendered.json> <out> — the /tokenize request for a /apply-template answer; <out> is written
# only when the answer has a prompt.
crux_tokreq() {
  local req
  req=$(jq -nj --slurpfile r "$1" "$_CRUX_DF_JQ"'
    $r | one("the rendered answer")
    | if type == "object" and has("prompt") then {content: .prompt, add_special: true, parse_special: true} | pydump(null)
      else error("the rendered answer has no prompt") end') || return 1
  printf '%s' "$req" > "$2"
}

# crux_emit_tok <manifest> <sha> <prompt_id> <rendered> <ids> — append one tok row.
crux_emit_tok() {
  jq -nj --arg sha "$2" --arg pid "$3" --arg rendered "$4" --arg ids "$5" "$_CRUX_DF_JQ"'
    {kind: "tok", engine: "llama.cpp", model_sha256: $sha, prompt_id: $pid, rendered: $rendered, ids: $ids}
    | pydump(null) + "\n"' >> "$1"
}

# crux_hf_source <hf-sources.yaml> <sha256> — "repo<TAB>revision<TAB>dtype" of the GGUF's declared HF source,
# or nothing. A missing or unreadable file declares nothing; a file outside the subset is refused (status 1).
crux_hf_source() {
  [ -f "$1" ] && [ -r "$1" ] || return 0
  jq -nRr --arg sha "$2" '
    def y11: test("^(yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF|null|Null|NULL)$")
      or test("^(0b[0-1_]+|0[0-7_]+|0|[1-9][0-9_]*|0x[0-9a-fA-F_]+)$")
      or test("^[0-9][0-9_]*\\.[0-9_]*([eE][-+][0-9]+)?$") or test("^[0-9]{4}-[0-9][0-9]?-[0-9][0-9]?");
    def shown: .[0:80] | tojson;
    def obj($what): if type == "object" then . else error("\($what) is not a mapping") end;
    def str($what): if type == "array" and .[0] == "s" then .[1] else error("\($what) is not a plain string") end;
    # st: the key path of the last line; open: that key had no value, so the next line may nest under it.
    reduce (inputs | select(test("^ *(#.*)?$") | not)) as $l ({t: {}, st: [], open: false};
      ($l | capture("^(?<ind> *)(?<k>[A-Za-z0-9_][A-Za-z0-9_./-]*):(?: +(?<v>[A-Za-z0-9_][A-Za-z0-9_./-]*))?(?: +#.*| *)$")
        // error("a line outside the YAML subset read here: \($l | shown)")) as $m
      | ($m.ind | length) as $ind
      | ($ind / 2) as $dep
      | if $ind % 2 != 0 or $ind > 6 then error("indent \($ind) is not 0, 2, 4 or 6: \($l | shown)")
        elif $dep > (.st | length) or ($dep == (.st | length) and $dep > 0 and (.open | not))
        then error("a line nested under a key that holds a value, or two levels deeper: \($l | shown)")
        elif $m.k | y11 then error("key \($m.k | tojson) is not a string in YAML 1.1")
        else . end
      | (.st[0:$dep] + [$m.k]) as $path
      | .st = $path
      | if $m.v == null then .t |= setpath($path; null) | .open = true
        else .t |= setpath($path; [(if $m.v | y11 then "x" else "s" end), $m.v]) | .open = false end)
    | .t
    | (if has("sources") then .sources | obj("sources") else {} end)
    | (if has($sha) then .[$sha] | obj("sources.\($sha)") else {} end)
    | (.hf | if . == null then {} else obj("sources.\($sha).hf") end)
    | if .repo == null or .revision == null then empty
      else [(.repo | str("repo")), (.revision | str("revision")),
            (if has("dtype") then .dtype | str("dtype") else "bfloat16" end)] | join("\t") end' < "$1"
}

# crux_emit_model <models.jsonl> <name> <sha> <thinking true|false|other> <ollama blob> <template sha>
#                 <ollama refused> <tokenization refused> — append one models row.
crux_emit_model() {
  jq -nj --arg name "$2" --arg sha "$3" --arg think "$4" --arg blob "$5" --arg tsha "$6" --arg oref "$7" \
    --arg twhy "$8" "$_CRUX_DF_JQ"'
    def orNull: if . == "" then null else . end;
    {name: $name, sha256: $sha,
     thinking_capable: ({"true": true, "false": false} | if has($think) then .[$think] else null end),
     ollama: {blob_sha256: ($blob | orNull), blob_is_input: (if $blob != "" then $blob == $sha else null end),
              template_sha256: ($tsha | orNull), refused: ($oref | orNull)},
     tokenization: {refused: ($twhy | orNull)}}
    | pydump(null) + "\n"' >> "$1"
}

# crux_ext_meta <out> (<engine> <probe> <unavailable>)... — the plugin engines' probe lines, by engine.
crux_ext_meta() {
  local out="$1" i=0 a=()
  shift
  while [ $# -ge 3 ]; do
    a+=(--arg "e$i" "$1" --arg "p$i" "$2" --arg "w$i" "$3")
    i=$((i + 1)); shift 3
  done
  [ $# -eq 0 ] || { printf 'crux_ext_meta: arguments come in threes\n' >&2; return 1; }
  jq -nj "${a[@]}" --argjson n "$i" "$_CRUX_DF_JQ"'
    def orNull: if . == "" then null else . end;
    reduce range($n) as $k ({}; .[$ARGS.named["e\($k)"]] =
      {probe: ($ARGS.named["p\($k)"] | orNull), unavailable: ($ARGS.named["w\($k)"] | orNull)})
    | pydump(null)' > "$out"
}

# crux_meta <out> <models.jsonl> <ext-engines.json> name=value... — meta.json, indent 2. The names are those
# of the python3 heredoc it replaces: version host backend isa engines verbs apr_line lbuild lpin lwhy osrv
# ocli owhy temp seed ctx maxtok gpu hsha lockvia.
crux_meta() {
  local out="$1" models="$2" ext="$3" kv a=() meta
  shift 3
  for kv in "$@"; do a+=(--arg "${kv%%=*}" "${kv#*=}"); done
  meta=$(jq -nj --rawfile models "$models" --slurpfile ext "$ext" "${a[@]}" "$_CRUX_DF_JQ"'
    def orNull: if . == "" then null else . end;
    def py_split: if . == "" then [""] else split(",") end;
    ($verbs | py_split) as $vs
    | ($ext | one("the ext-engines file") | if type == "object" then . else error("the ext-engines file is not an object") end) as $ext
    | {version: $version, host: $host, backend: $backend, isa: $isa, gpu: ($gpu | orNull),
       engines: ($engines | py_split), verbs: $vs, thinking: ["off"],
       sampling: {temperature: ($temp | pyfloat_of_text), seed: ($seed | pyint), context: ($ctx | pyint),
                  max_tokens: ($maxtok | pyint), source: "scripts/llama_pin.toml [protocol]"},
       apr: {version_line: $apr_line},
       harness: {sha: ($hsha | orNull), driver: "scripts/crux_inference_dogfood.sh", gpu_lock: $lockvia},
       llama_cpp: {build: ($lbuild | orNull), pin: ($lpin | orNull), unavailable: ($lwhy | orNull)},
       ollama: {server_version: ($osrv | orNull), client_version: ($ocli | orNull), unavailable: ($owhy | orNull)}}
    + $ext
    + {models: [$models | split("\n")[] | select(pyblank | not) | fromjson],
       not_covered: [
         "verbs not run here: " + ([("chat", "serve", "code") | select(. as $v | $vs | any(.[]; . == $v) | not)] | join(", ")),
         (if any($vs[]; . == "serve") then
            "apr serve'"'"'s backend is unverified: its responses report none",
            "apr serve reads no per-request thinking toggle; the comparators are told enable_thinking explicitly"
          else empty end),
         (if any($vs[]; . == "code") then
            "apr code has no backend, max-tokens or thinking control: it spawns `apr serve --gpu`, so its rows are judged by executing the code, and it is refused on the cpu lane"
          else empty end),
         (if any($vs[]; . == "chat") then "apr chat'"'"'s backend is unverified: apr chat reports none (#3794)" else empty end),
         "thinking ON until #3723 adds an apr toggle",
         "consumer-brief context rungs (#3716) and each engine'"'"'s max accepted context",
         "TTFT and decode rate: the serve verb'"'"'s measurement goes through `apr test llm bench` (PERF-009), the next increment"]}
    | pydump(2)') || return 1
  printf '%s' "$meta" > "$out"
}
