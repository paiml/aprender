#!/usr/bin/env bash
# crux_dogfood_json_test.sh — the case table of scripts/lib/crux_dogfood_json.sh, the bash + jq port of the
# JSON and YAML steps of scripts/crux_inference_dogfood.sh, and of the dogfood's bash free_port. Three checks:
#
#   1. CASES. Each case feeds a function a small synthetic input and compares its output, byte for byte,
#      with the output the replaced python3 snippet gave (written out below, not computed by the port), or
#      expects the refusal or the difference the library header lists. free_port runs on stand-in /proc files.
#   2. HF-SOURCES. The real evidence/crux/hf-sources.yaml stays inside the YAML subset the port reads: every
#      sha256 in it is read with no refusal, and each declared source is "repo<TAB>40-hex revision<TAB>dtype".
#   3. RATCHET. The dogfood's python lines are exactly the ones not ported yet (the plugin engines' *.py
#      runner, the PTY chat driver and the judge). Each port moves this table down.
#
#   bash scripts/tests/crux_dogfood_json_test.sh              # all three; exit 1 on any failure
#   bash scripts/tests/crux_dogfood_json_test.sh --self-test  # every planted defect must turn a check RED
#
# The self-test plants each defect in a copy of the library (or of the dogfood), asserts the plant changed
# the file, and requires the named case to FAIL on it with a wrong answer. A plant that changes nothing, or
# a case table that stays green, fails the self-test.
#
# Not wired yet. It lives under scripts/tests/ rather than scripts/check_*.sh because guard_tree.sh runs
# every scripts/check_*.sh in the required CI job, and a new check blocks only after it has been run green
# by hand for three nights. Wiring it is its own change.
set -euo pipefail
cd "$(dirname "$0")/../.."
PROG=crux_dogfood_json_test
LIB=scripts/lib/crux_dogfood_json.sh
DOGFOOD=scripts/crux_inference_dogfood.sh
HF=evidence/crux/hf-sources.yaml
case "${1:-}" in
  "" | --self-test) ;;
  -h | --help) printf 'usage: bash scripts/tests/%s.sh [--self-test]\n' "$PROG"; exit 0 ;;
  *) printf '%s: usage: bash scripts/tests/%s.sh [--self-test]\n' "$PROG" "$PROG" >&2; exit 2 ;;
esac
command -v jq > /dev/null || { printf '%s: FAIL: jq not found; nothing was measured\n' "$PROG" >&2; exit 1; }
T=$(mktemp -d)
trap 'rm -rf -- "${T:?}"' EXIT

# ---------------------------------------------------------------------------------------------------------
# Fixtures (synthetic; built once, read-only to the cases).
FX="$T/fx"
mkdir -p "$FX"
noeol() { local s; s=$(cat); printf '%s' "$s"; }   # a heredoc without its final newline, as json.dump wrote
# One file per line of the expected listing: "--- <name> (<bytes> bytes)", the content, then a newline.
dump() {
  local f
  find "$1" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort | while IFS= read -r f; do
    printf -- '--- %s (%s bytes)\n' "$f" "$(wc -c < "$1/$f")"; cat "$1/$f"; printf '\n'
  done
}

# The prompt set: every number form json.dumps lays out its own way, \u escapes past ASCII and for DEL, a
# string, a float and a bool as int() inputs, the three verb shapes, and Python truthiness of control.
cat > "$FX/prompts.json" <<'JSON'
{"max_tokens": null, "prompts": [
  {"id": "a1", "control": true, "note": "tab\there \u007f ~",
   "x": [1e-05, 1E16, 1234567890123456.0, 0.0001, 0.1, -0.0, -0, 100, 2.5e-7, 123456789012345678, 0.12345678901234567, true, null],
   "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "caf\u00e9 \ud83d\ude00"}],
   "max_tokens": {"off": 64, "on": 2.9}},
  {"id": "b2", "verb": "chat", "control": "yes",
   "messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three\nlines"}],
   "max_tokens": {"off": "  48\n", "on": 512}},
  {"id": "c3", "verb": ["code", "serve stream"], "control": 0,
   "messages": [{"role": "user", "content": "def f():"}], "max_tokens": {"off": true, "on": 1024}}]}
JSON
cat > "$FX/prompts.expected" <<'EOF'
--- code-prompts.txt (3 bytes)
c3

--- maxtok-a1-off.txt (2 bytes)
64
--- maxtok-a1-on.txt (1 bytes)
2
--- maxtok-a1.txt (2 bytes)
64
--- maxtok-b2-off.txt (2 bytes)
48
--- maxtok-b2-on.txt (3 bytes)
512
--- maxtok-b2.txt (2 bytes)
48
--- maxtok-c3-off.txt (1 bytes)
1
--- maxtok-c3-on.txt (4 bytes)
1024
--- maxtok-c3.txt (1 bytes)
1
--- messages-a1.json (112 bytes)
{"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "caf\u00e9 \ud83d\ude00"}]}
--- messages-b2.json (136 bytes)
{"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three\nlines"}]}
--- messages-c3.json (55 bytes)
{"messages": [{"role": "user", "content": "def f():"}]}
--- prompt-a1.json (339 bytes)
{"id": "a1", "control": true, "note": "tab\there \u007f ~", "x": [1e-05, 1e+16, 1234567890123456.0, 0.0001, 0.1, -0.0, 0, 100, 2.5e-07, 123456789012345678, 0.12345678901234566, true, null], "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "caf\u00e9 \ud83d\ude00"}], "max_tokens": {"off": 64, "on": 2.9}}
--- prompt-a1.txt (10 bytes)
café 😀
--- prompt-b2.json (226 bytes)
{"id": "b2", "verb": "chat", "control": "yes", "messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three\nlines"}], "max_tokens": {"off": "  48\n", "on": 512}}
--- prompt-b2.txt (11 bytes)
three
lines
--- prompt-c3.json (156 bytes)
{"id": "c3", "verb": ["code", "serve stream"], "control": 0, "messages": [{"role": "user", "content": "def f():"}], "max_tokens": {"off": true, "on": 1024}}
--- prompt-c3.txt (8 bytes)
def f():
--- serve-prompts.jsonl (63 bytes)
["a1", ["serve run", "serve stream"]]
["c3", ["serve stream"]]

--- turns-a1.json (26 bytes)
["caf\u00e9 \ud83d\ude00"]
--- turns-a1.txt (11 bytes)
café 😀

--- turns-b2.json (23 bytes)
["one", "three\nlines"]
--- turns-b2.txt (16 bytes)
one
three
lines

--- turns-c3.json (12 bytes)
["def f():"]
--- turns-c3.txt (9 bytes)
def f():

EOF
printf '%s\n' 'run a1' 'chat b2' > "$FX/prompts.stdout"
cat > "$FX/only.expected" <<'EOF'
--- code-prompts.txt (0 bytes)

--- maxtok-b2-off.txt (2 bytes)
48
--- maxtok-b2-on.txt (3 bytes)
512
--- maxtok-b2.txt (2 bytes)
48
--- messages-b2.json (136 bytes)
{"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three\nlines"}]}
--- prompt-b2.json (226 bytes)
{"id": "b2", "verb": "chat", "control": "yes", "messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three\nlines"}], "max_tokens": {"off": "  48\n", "on": 512}}
--- prompt-b2.txt (11 bytes)
three
lines
--- serve-prompts.jsonl (0 bytes)

--- turns-b2.json (23 bytes)
["one", "three\nlines"]
--- turns-b2.txt (16 bytes)
one
three
lines

EOF
# One-prompt sets, each holding one input the library refuses or reads differently from Python.
p1='{"id": "a1", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8'
printf '{"prompts": [%s}, {"id": "b", "messages": [], "max_tokens": 8}]}\n' "$p1" > "$FX/prompts-badmsg.json"
printf '{"prompts": [{"id": "../x", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}]}\n' > "$FX/prompts-slash.json"
printf '{"prompts": [{"id": 7, "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}]}\n' > "$FX/prompts-intid.json"
for v in fold:'[1.5e1, 100e0]' nan:NaN surrogate:'"\ud800"' sig18:0.123456789012345678 inf:1e400; do
  printf '{"prompts": [%s, "x": %s}]}\n' "$p1" "${v#*:}" > "$FX/prompts-${v%%:*}.json"
done
for v in str:'"  7 "' float:1.9 under:'"4_2"' huge:1e300 sig21:0.123456789012345678901; do
  printf '{"max_tokens": %s, "prompts": [%s}]}\n' "${v#*:}" "$p1" > "$FX/glob-${v%%:*}.json"
done
printf '{"prompts": [{"id": "a", "max_tokens": {"off": 3}}]}\n' > "$FX/glob-nomode.json"

# llama-server answers.
printf '%s\n' '{"chat_template": "x {% if enable_thinking %}"}' > "$FX/props-et.json"
printf '%s\n' '{"chat_template": "a <think> b"}' > "$FX/props-think.json"
printf '%s\n' '{"chat_template": ["<think>"]}' > "$FX/props-list.json"
printf '%s\n' '{"chat_template": "plain"}' > "$FX/props-plain.json"
printf '%s\n' '{"chat_template": null}' > "$FX/props-null.json"
printf '%s\n' '{}' > "$FX/props-none.json"
printf '%s\n' '{"chat_template": 5}' > "$FX/props-num.json"
printf '%s\n' '{"chat_template": {"<think>": 1}}' > "$FX/props-obj.json"   # a mapping: its keys count
printf '%s\n' '{"chat_template": {"x": "enable_thinking"}}' > "$FX/props-objval.json"
printf '%s\n' '{"prompt": "<s>h\u00e9 \ud83d\ude00", "x": 1}' > "$FX/rendered.json"
printf '%s\n' '{"tokens": []}' > "$FX/rendered-none.json"
printf '%s' '{"content": "<s>h\u00e9 \ud83d\ude00", "add_special": true, "parse_special": true}' > "$FX/tokreq.expected"

# Manifest rows: three match (apr, a1, s1, run), one of them CR LF and one with no final newline.
{
  printf '%s\n' '{"kind": "gen", "engine": "apr", "prompt_id": "a1", "rc": 0, "model_sha256": "s1", "host": "h", "verb": "run"}' \
    '{"kind": "gen", "engine": "ollama", "prompt_id": "a1", "model_sha256": "s1", "verb": "run"}' \
    '{"kind": "tok", "engine": "apr", "prompt_id": "a1", "model_sha256": "s1", "verb": "run"}'
  printf '%s\r\n\r\n' '{"kind": "gen", "engine": "apr", "prompt_id": "a1", "model_sha256": "s1", "verb": "run", "x": 1}'
  printf '%s\n' '{"kind": "gen", "engine": "apr", "prompt_id": "a2", "model_sha256": "s1", "verb": "run"}' \
    '{"kind": "gen", "engine": "apr", "prompt_id": "a1", "model_sha256": "s2", "verb": "run"}' \
    '{"kind": "gen", "engine": "apr", "prompt_id": "a1", "model_sha256": "s1", "verb": "chat"}' ''
  printf ' \302\240\343\200\200\n'   # blank to str.strip(): NBSP and U+3000
  printf '%s' '{"kind": "gen", "engine": "apr", "prompt_id": "a1", "model_sha256": "s1", "verb": "run"}'
} > "$FX/rows.jsonl"
printf '%s\n' '{"kind": "gen"}' '[1]' > "$FX/rows-array.jsonl"
printf '%s\r%s\n' '{"kind": "gen"}' '{"kind": "gen"}' > "$FX/rows-lone-cr.jsonl"

# Rows the emit functions append (json.dumps, one per line).
cat > "$FX/gen.expected" <<'EOF'
{"kind": "gen", "engine": "apr", "prompt_id": "a1", "rc": 0, "stdout": "out \u00e9", "stderr": null, "refused": null, "model_sha256": "s1", "host": "h", "verb": "run", "thinking": "off", "backend": "gpu"}
{"kind": "gen", "engine": "ollama", "prompt_id": "a1", "rc": -7, "stdout": null, "stderr": "err\nx", "refused": "why", "model_sha256": "s1", "host": "h", "verb": "run", "thinking": "off", "backend": "gpu", "ollama_unloaded": true, "mode": "on"}
{"kind": "gen", "engine": "apr", "prompt_id": "a1", "rc": 7, "stdout": null, "stderr": null, "refused": null, "model_sha256": "s1", "host": "h", "verb": "run", "thinking": "off", "backend": "gpu", "ollama_unloaded": false}
{"kind": "gen", "engine": "apr", "prompt_id": "a1", "rc": null, "stdout": null, "stderr": null, "refused": null, "model_sha256": "s1", "host": "h", "verb": "run", "thinking": "off", "backend": "gpu"}
EOF
printf '%s\n' '{"kind": "tok", "engine": "llama.cpp", "model_sha256": "s1", "prompt_id": "a1", "rendered": "r.json", "ids": "i.json"}' \
  > "$FX/tok.expected"
cat > "$FX/models.expected" <<'EOF'
{"name": "m \u00e9", "sha256": "s1", "thinking_capable": true, "ollama": {"blob_sha256": "s1", "blob_is_input": true, "template_sha256": "t1", "refused": null}, "tokenization": {"refused": null}}
{"name": "m2", "sha256": "s2", "thinking_capable": null, "ollama": {"blob_sha256": null, "blob_is_input": null, "template_sha256": null, "refused": "no ollama"}, "tokenization": {"refused": "no tok"}}
{"name": "m3", "sha256": "s3", "thinking_capable": false, "ollama": {"blob_sha256": "s9", "blob_is_input": false, "template_sha256": null, "refused": null}, "tokenization": {"refused": null}}
EOF
printf '%s' '{"vllm": {"probe": "p2", "unavailable": null}, "sglang": {"probe": null, "unavailable": "not installed"}}' \
  > "$FX/ext.expected"
{ cat "$FX/models.expected"; printf '\n \302\240\343\200\200\n'; } > "$FX/models.jsonl"   # blank lines are skipped
cp "$FX/ext.expected" "$FX/ext.json"
noeol > "$FX/meta.expected" <<'JSON'
{
  "version": "0.70.0",
  "host": "h",
  "backend": "gpu",
  "isa": "@ISA@",
  "gpu": null,
  "engines": [
    "apr",
    "llama.cpp"
  ],
  "verbs": [
    "run",
    "serve"
  ],
  "thinking": [
    "off"
  ],
  "sampling": {
    "temperature": 0.0,
    "seed": 42,
    "context": 4096,
    "max_tokens": 64,
    "source": "scripts/llama_pin.toml [protocol]"
  },
  "apr": {
    "version_line": "apr 0.70.0 (abc)"
  },
  "harness": {
    "sha": "abc",
    "driver": "scripts/crux_inference_dogfood.sh",
    "gpu_lock": "flock"
  },
  "llama_cpp": {
    "build": null,
    "pin": "b123",
    "unavailable": null
  },
  "ollama": {
    "server_version": "0.12.3",
    "client_version": null,
    "unavailable": null
  },
  "vllm": {
    "probe": "p2",
    "unavailable": null
  },
  "sglang": {
    "probe": null,
    "unavailable": "not installed"
  },
  "models": [
    {
      "name": "m \u00e9",
      "sha256": "s1",
      "thinking_capable": true,
      "ollama": {
        "blob_sha256": "s1",
        "blob_is_input": true,
        "template_sha256": "t1",
        "refused": null
      },
      "tokenization": {
        "refused": null
      }
    },
    {
      "name": "m2",
      "sha256": "s2",
      "thinking_capable": null,
      "ollama": {
        "blob_sha256": null,
        "blob_is_input": null,
        "template_sha256": null,
        "refused": "no ollama"
      },
      "tokenization": {
        "refused": "no tok"
      }
    },
    {
      "name": "m3",
      "sha256": "s3",
      "thinking_capable": false,
      "ollama": {
        "blob_sha256": "s9",
        "blob_is_input": false,
        "template_sha256": null,
        "refused": null
      },
      "tokenization": {
        "refused": null
      }
    }
  ],
  "not_covered": [
    "verbs not run here: chat, code",
    "apr serve's backend is unverified: its responses report none",
    "apr serve reads no per-request thinking toggle; the comparators are told enable_thinking explicitly",
    "thinking ON until #3723 adds an apr toggle",
    "consumer-brief context rungs (#3716) and each engine's max accepted context",
    "TTFT and decode rate: the serve verb's measurement goes through `apr test llm bench` (PERF-009), the next increment"
  ]
}
JSON
ISA=$(uname -m)
sed -i "s/@ISA@/$ISA/" "$FX/meta.expected"
META1=(version=0.70.0 host=h backend=gpu "isa=$ISA" engines=apr,llama.cpp verbs=run,serve 'apr_line=apr 0.70.0 (abc)'
  'lbuild=' lpin=b123 'lwhy=' osrv=0.12.3 'ocli=' 'owhy=' temp=0 seed=42 ctx=4096 maxtok=64 'gpu=' hsha=abc lockvia=flock)
META2=(version=1 host=h backend=cpu "isa=$ISA" engines=apr verbs=run,chat,code apr_line=a=b lbuild=bb 'lpin='
  'lwhy=no llama' 'osrv=' ocli=0.1 'owhy=no ollama' 'temp= 1e-5 ' 'seed= 7' ctx=+2048 maxtok=+3 'gpu=RTX é' 'hsha=' lockvia=none)
META3=(version=1 host=h backend=cpu "isa=$ISA" engines=apr 'verbs=' 'apr_line=' 'lbuild=' 'lpin=' 'lwhy=' 'osrv=' 'ocli=' 'owhy='
  temp=1.5E3 seed=0 ctx=0 maxtok=0 'gpu=' 'hsha=' lockvia=none)

# hf-sources: the subset, and one input outside it per refusal the library header lists.
cat > "$FX/hf.yaml" <<'YAML'
# a comment
schema: crux-hf-sources/v1
sources:
  aaa:
    gguf:
      repo: g/r
    hf:
      repo: Org/Model-1.5B
      revision: abc123
  bbb:   # a comment after a key
    hf:
      repo: Org/M2
      revision: def456
      dtype: float16

  ccc:
    hf:
      repo: Org/M3
YAML
sed 's#repo: Org/M2#repo: "Org/M2"#' "$FX/hf.yaml" > "$FX/hf-quoted.yaml"
sed 's#^  ccc:#  yes:#' "$FX/hf.yaml" > "$FX/hf-boolkey.yaml"
sed 's#revision: def456#revision: 0123#' "$FX/hf.yaml" > "$FX/hf-octal.yaml"
sed 's#^      dtype: float16#\tdtype: float16#' "$FX/hf.yaml" > "$FX/hf-tab.yaml"
sed 's#^      dtype: float16#     dtype: float16#' "$FX/hf.yaml" > "$FX/hf-odd.yaml"
sed 's#$#\r#' "$FX/hf.yaml" > "$FX/hf-crlf.yaml"
sed 's#^    hf:$#    hf: {repo: a, revision: b}#' "$FX/hf.yaml" > "$FX/hf-flow.yaml"

# Stand-in /proc files for free_port: tcp holds 40000, tcp6 holds 40001 and 40002.
P="$FX/proc"
mkdir -p "$P"
tcp_line() { printf '   %s: 0100007F:%s 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000 0 1 1\n' "$1" "$2"; }
tcp_head='  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode'
{ printf '%s\n' "$tcp_head"; tcp_line 0 9C40; } > "$P/tcp"
{ printf '%s\n' "$tcp_head"; tcp_line 0 9C41; tcp_line 1 9c42; } > "$P/tcp6"
printf '%s\n' "$tcp_head" > "$P/tcp-empty"
{ printf '%s\n' "$tcp_head"; tcp_line 0 ZZZZ; } > "$P/tcp-bad"
printf '40000\t40003\n' > "$P/range"
printf '40000 40001' > "$P/range-noeol"
printf '40000 40999\n' > "$P/range-wide"
printf '40003 40000\n' > "$P/range-backwards"
printf '0 3\n' > "$P/range-zero"
printf 'a b\n' > "$P/range-text"
printf '\n' > "$P/resv-none"
printf '40001,40003\n' > "$P/resv-two"
printf '40003\n' > "$P/resv-last"
printf '4000a\n' > "$P/resv-bad"

# ---------------------------------------------------------------------------------------------------------
# The case table. run_cases <lib> <dogfood> prints one FAIL line per failed case and exits with the count.
run_cases() (
  # shellcheck source=scripts/lib/crux_dogfood_json.sh
  . "$1"
  W=$(mktemp -d "$T/run.XXXXXX")
  n=0
  bad() { printf 'FAIL %s: %s\n' "$1" "$2"; n=$((n + 1)); }
  same() { cmp -s "$2" "$3" || bad "$1" "output differs from $(basename "$2"): $(cmp "$2" "$3" 2>&1 | head -n 1)"; }
  ok() { [ "$2" = 0 ] || bad "$1" "rc $2, expected 0: $(head -c 300 "$W/err")"; }
  refused() { [ "$2" != 0 ] || bad "$1" "rc 0, expected a refusal"; }
  is() { [ "$2" = "$3" ] || bad "$1" "got '$2', expected '$3'"; }
  empty_dir() { [ -z "$(find "$2" -mindepth 1 -print -quit)" ] || bad "$1" "files were written: $(find "$2" -mindepth 1 -printf '%f ')"; }
  run() { rc=0; "$@" > "$W/out" 2> "$W/err" || rc=$?; }   # out and err of one call, its status in rc

  rc=0; jq_why=$(crux_dogfood_jq_ok 2>&1) || rc=$?
  [ "$rc" = 0 ] || bad jq-ok "this jq was judged unfit: $jq_why"

  # crux_prompt_files
  mkdir "$W/p"; run crux_prompt_files "$FX/prompts.json" "$W/p" ""
  ok prompts-files "$rc"; dump "$W/p" > "$W/p.dump"; same prompts-files "$FX/prompts.expected" "$W/p.dump"
  same prompts-files-stdout "$FX/prompts.stdout" "$W/out"
  mkdir "$W/o"; run crux_prompt_files "$FX/prompts.json" "$W/o" b2
  ok prompts-only "$rc"; dump "$W/o" > "$W/o.dump"; same prompts-only "$FX/only.expected" "$W/o.dump"
  is prompts-only-stdout "$(cat "$W/out")" "chat b2"
  mkdir "$W/u"; run crux_prompt_files "$FX/prompts.json" "$W/u" zz,b2,yy,zz
  is prompts-only-unknown "$rc" 3; empty_dir prompts-only-unknown "$W/u"
  is prompts-only-unknown "$(cat "$W/err")" "--only-prompts names ids the prompt set does not hold: yy, zz"
  # Refused: nothing is written (Python wrote the files of the prompts before the bad one, or wrote NaN).
  for v in badmsg slash intid nan surrogate sig18 inf; do
    mkdir "$W/r-$v"; run crux_prompt_files "$FX/prompts-$v.json" "$W/r-$v" ""
    refused "prompts-refused-$v" "$rc"; empty_dir "prompts-refused-$v" "$W/r-$v"
  done
  # A literal whose exponent folds it into a whole number is written as an int (Python wrote 15.0, 100.0).
  mkdir "$W/f"; run crux_prompt_files "$FX/prompts-fold.json" "$W/f" ""
  ok prompts-folded-literal "$rc"
  is prompts-folded-literal "$(cat "$W/f/prompt-a1.json")" \
    '{"id": "a1", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8, "x": [15, 100]}'

  # crux_max_tokens
  is max-tokens-off "$(crux_max_tokens "$FX/prompts.json" off 2> /dev/null)" 64
  is max-tokens-on "$(crux_max_tokens "$FX/prompts.json" on 2> /dev/null)" 1024
  is max-tokens-global-text "$(crux_max_tokens "$FX/glob-str.json" on 2> /dev/null)" 7
  is max-tokens-global-float "$(crux_max_tokens "$FX/glob-float.json" off 2> /dev/null)" 1
  for v in under huge sig21; do run crux_max_tokens "$FX/glob-$v.json" off; refused "max-tokens-refused-$v" "$rc"; done
  run crux_max_tokens "$FX/glob-nomode.json" on; refused max-tokens-no-mode "$rc"

  # crux_control_ids
  is control-ids "$(crux_control_ids "$FX/prompts.json" 2> /dev/null)" "a1 b2"

  # crux_emit_gen and crux_rows_for
  run crux_emit_gen "$W/gen.jsonl" apr a1 0 "out é" "" "" "" s1 h run off gpu ""
  ok gen-rows "$rc"
  crux_emit_gen "$W/gen.jsonl" ollama a1 -7 "" "err"$'\n'"x" why true s1 h run off gpu on 2> /dev/null || bad gen-rows "row 2 refused"
  crux_emit_gen "$W/gen.jsonl" apr a1 007 "" "" "" false s1 h run off gpu "" 2> /dev/null || bad gen-rows "row 3 refused"
  crux_emit_gen "$W/gen.jsonl" apr a1 x "" "" "" "" s1 h run off gpu "" 2> /dev/null || bad gen-rows "row 4 refused"
  same gen-rows "$FX/gen.expected" "$W/gen.jsonl"
  # "--5" is no exit status: null (Python crashed on it).
  crux_emit_gen "$W/gen-dd.jsonl" apr a1 --5 "" "" "" "" s1 h run off gpu "" 2> /dev/null || bad gen-rc-dash-dash "refused"
  tail -n 1 "$FX/gen.expected" > "$W/gen-dd.expected"; same gen-rc-dash-dash "$W/gen-dd.expected" "$W/gen-dd.jsonl"
  is rows-for "$(crux_rows_for "$FX/rows.jsonl" apr a1 s1 run 2> /dev/null)" 3
  run crux_rows_for "$FX/rows-array.jsonl" apr a1 s1 run; refused rows-for-not-an-object "$rc"
  run crux_rows_for "$FX/rows-lone-cr.jsonl" apr a1 s1 run; refused rows-for-lone-cr "$rc"

  # crux_thinking_capable
  for v in et:true think:true list:true obj:true objval:false plain:false null:false none:false; do
    is "thinking-${v%%:*}" "$(crux_thinking_capable "$FX/props-${v%%:*}.json" 2> /dev/null)" "${v#*:}"
  done
  run crux_thinking_capable "$FX/props-num.json"; refused thinking-number "$rc"; is thinking-number "$(cat "$W/out")" ""

  # crux_tokreq and crux_emit_tok
  run crux_tokreq "$FX/rendered.json" "$W/tokreq.json"; ok tokreq "$rc"; same tokreq "$FX/tokreq.expected" "$W/tokreq.json"
  run crux_tokreq "$FX/rendered-none.json" "$W/tokreq-none.json"; refused tokreq-no-prompt "$rc"
  [ ! -e "$W/tokreq-none.json" ] || bad tokreq-no-prompt "a request was written"
  run crux_emit_tok "$W/tok.jsonl" s1 a1 r.json i.json; ok tok-row "$rc"; same tok-row "$FX/tok.expected" "$W/tok.jsonl"

  # crux_emit_model, crux_ext_meta, crux_meta
  run crux_emit_model "$W/models.jsonl" "m é" s1 true s1 t1 "" ""; ok model-rows "$rc"
  crux_emit_model "$W/models.jsonl" m2 s2 unknown "" "" "no ollama" "no tok" 2> /dev/null || bad model-rows "row 2 refused"
  crux_emit_model "$W/models.jsonl" m3 s3 false s9 "" "" "" 2> /dev/null || bad model-rows "row 3 refused"
  same model-rows "$FX/models.expected" "$W/models.jsonl"
  run crux_ext_meta "$W/ext.json" vllm "probe 1" "" sglang "" "not installed" vllm p2 ""
  ok ext-meta "$rc"; same ext-meta "$FX/ext.expected" "$W/ext.json"
  run crux_ext_meta "$W/ext0.json"; ok ext-meta-none "$rc"; is ext-meta-none "$(cat "$W/ext0.json")" "{}"
  run crux_ext_meta "$W/ext4.json" a b c d; refused ext-meta-not-threes "$rc"
  run crux_meta "$W/meta1.json" "$FX/models.jsonl" "$FX/ext.json" "${META1[@]}"
  ok meta "$rc"; same meta "$FX/meta.expected" "$W/meta1.json"
  run crux_meta "$W/meta2.json" "$FX/models.jsonl" "$FX/ext.json" "${META2[@]}"; ok meta-text-numbers "$rc"
  for l in '  "gpu": "RTX \u00e9",' '    "temperature": 1e-05,' '    "seed": 7,' '    "context": 2048,' '    "max_tokens": 3,' \
    '    "verbs not run here: serve",' "    \"apr chat's backend is unverified: apr chat reports none (#3794)\","; do
    grep -qxF -- "$l" "$W/meta2.json" 2> /dev/null || bad meta-text-numbers "no line: $l"
  done
  run crux_meta "$W/meta3.json" "$FX/models.jsonl" "$FX/ext.json" "${META3[@]}"; ok meta-empty-verbs "$rc"
  for l in '    "temperature": 1500.0,' '    "verbs not run here: chat, serve, code",'; do
    grep -qxF -- "$l" "$W/meta3.json" 2> /dev/null || bad meta-empty-verbs "no line: $l"
  done
  jq -e '.verbs == [""]' "$W/meta3.json" > /dev/null 2>&1 || bad meta-empty-verbs 'verbs is not [""]'
  for t in 1_0 inf 0.10000000000000000555; do
    run crux_meta "$W/meta-$t.json" "$FX/models.jsonl" "$FX/ext.json" "${META3[@]/#temp=*/temp=$t}"
    refused "meta-temperature-refused-$t" "$rc"
  done

  # crux_hf_source
  is hf-source "$(crux_hf_source "$FX/hf.yaml" aaa 2> /dev/null)" "Org/Model-1.5B"$'\t'"abc123"$'\t'"bfloat16"
  is hf-source-dtype "$(crux_hf_source "$FX/hf.yaml" bbb 2> /dev/null)" "Org/M2"$'\t'"def456"$'\t'"float16"
  for s in ccc zzz; do run crux_hf_source "$FX/hf.yaml" "$s"; ok "hf-source-none-$s" "$rc"; is "hf-source-none-$s" "$(cat "$W/out")" ""; done
  run crux_hf_source "$FX/no-such.yaml" bbb; ok hf-source-no-file "$rc"; is hf-source-no-file "$(cat "$W/out")" ""
  for v in quoted boolkey octal tab odd crlf flow; do
    run crux_hf_source "$FX/hf-$v.yaml" bbb; refused "hf-source-refused-$v" "$rc"; is "hf-source-refused-$v" "$(cat "$W/out")" ""
  done

  # crux_ollama_version
  is ollama-version "$(printf '%s' '{"version": "0.12.3"}' | crux_ollama_version 2> /dev/null)" 0.12.3
  rc=0; out=$(printf '%s' '{"x": 1}' | crux_ollama_version 2> /dev/null) || rc=$?; ok ollama-version-absent "$rc"; is ollama-version-absent "$out" ""
  for v in number:'{"version": 5}' two:'{"version": "a"} {"version": "b"}' array:'[1]'; do
    rc=0; out=$(printf '%s' "${v#*:}" | crux_ollama_version 2> /dev/null) || rc=$?
    refused "ollama-version-refused-${v%%:*}" "$rc"; is "ollama-version-refused-${v%%:*}" "$out" ""
  done

  # free_port, taken from the dogfood itself, on stand-in /proc files.
  sed -n '/^free_port() {/,/^}/p' "$2" > "$W/free_port.sh"
  grep -q '^free_port() {' "$W/free_port.sh" || bad free-port "no free_port() in $2"
  # shellcheck disable=SC1091
  . "$W/free_port.sh"
  fp() { rc=0; port=$(free_port "$@" 2> /dev/null) || rc=$?; }
  # The start is random: one call lands on the right port by luck 1 time in 4 even with a skip dropped, so 12
  # calls miss a dropped skip with odds (1/4)^12.
  for i in $(seq 12); do fp "$P/range" "$P/resv-none" "$P/tcp" "$P/tcp6"; ok free-port-skips-used "$rc"; is free-port-skips-used "$port" 40003; done
  for i in $(seq 12); do fp "$P/range" "$P/resv-two" "$P/tcp" "$P/no-tcp6"; ok free-port-skips-reserved "$rc"; is free-port-skips-reserved "$port" 40002; done
  fp "$P/range" "$P/resv-last" "$P/tcp" "$P/tcp6"; refused free-port-none-free "$rc"; is free-port-none-free "$port" ""
  fp "$P/range-noeol" "$P/no-resv" "$P/tcp"; ok free-port-range-no-newline "$rc"; is free-port-range-no-newline "$port" 40001
  fp "$P/range" "$P/resv-none" "$P/no-tcp" "$P/no-tcp6"; refused free-port-no-table "$rc"
  fp "$P/range" "$P/resv-none" "$P/tcp-bad"; refused free-port-bad-table "$rc"
  for r in backwards zero text; do fp "$P/range-$r" "$P/resv-none" "$P/tcp"; refused "free-port-range-$r" "$rc"; done
  fp "$P/range" "$P/resv-bad" "$P/tcp"; refused free-port-bad-reserved "$rc"
  : > "$W/ports"
  for i in $(seq 20); do fp "$P/range-wide" "$P/resv-none" "$P/tcp-empty"; printf '%s\n' "$port" >> "$W/ports"; done
  [ "$(grep -cxE '40[0-9]{3}' "$W/ports")" = 20 ] || bad free-port-random-start "a port outside 40000-40999"
  [ "$(sort -u "$W/ports" | wc -l)" -gt 1 ] || bad free-port-random-start "20 calls gave one port"
  fp; ok free-port-real-proc "$rc"
  read -r lo hi < /proc/sys/net/ipv4/ip_local_port_range
  { [ "${port:-0}" -ge "$lo" ] && [ "${port:-0}" -le "$hi" ]; } 2> /dev/null || bad free-port-real-proc "$port is not in $lo-$hi"
  exit "$n"
)

# ---------------------------------------------------------------------------------------------------------
# The real hf-sources.yaml: inside the subset, every sha256 in it read with no refusal.
hf_real() ( # <lib> <hf-sources.yaml>
  # shellcheck source=scripts/lib/crux_dogfood_json.sh
  . "$1"
  local HF=$2 shas sha out n=0 got=0
  shas=$(grep -oE '^  [0-9a-f]{64}:' "$HF" | tr -d ' :' || true)
  [ -n "$shas" ] || { printf 'FAIL hf-real: no sha256 key in %s\n' "$HF"; exit 1; }
  for sha in $shas; do
    if ! out=$(crux_hf_source "$HF" "$sha" 2> "$T/hf.err") || [ -s "$T/hf.err" ]; then
      printf 'FAIL hf-real: %s: %s\n' "${sha:0:12}" "$(head -c 300 "$T/hf.err")"; n=$((n + 1)); continue
    fi
    [ -z "$out" ] && continue
    if printf '%s\n' "$out" | grep -qxE $'[^\t]+\t[0-9a-f]{40}\t[a-z0-9]+'; then got=$((got + 1))
    else printf 'FAIL hf-real: %s: not repo<TAB>revision<TAB>dtype: %s\n' "${sha:0:12}" "$out"; n=$((n + 1)); fi
  done
  [ "$got" -gt 0 ] || { printf 'FAIL hf-real: no sha256 in %s declares an HF source\n' "$HF"; n=$((n + 1)); }
  [ "$n" = 0 ] && printf 'PASS hf-real: %s sha256 keys read, %s declare an HF source\n' "$(printf '%s\n' "$shas" | wc -l)" "$got"
  exit "$n"
)

# ---------------------------------------------------------------------------------------------------------
# The ratchet: the python lines of the dogfood, comments excluded, must be exactly the ones not ported yet.
# A line names python when it says python or pypy in any case ($PYTHON too), runs the `py` launcher, or names
# a .py file. Each table entry matches one whole line, so a call appended to an allowed line is a new line.
python_lines() {
  grep -niE 'python|pypy|(^|[^[:alnum:]_.])py[[:space:]]|\.py([^[:alnum:]_]|$)' "$1" | grep -vE '^[0-9]+:[[:space:]]*#' || true
}
# <count> <ERE on the line, leading blanks stripped>
RATCHET=(
  '1 ^for f in "scripts/crux_engine_\$eng\.sh" "scripts/crux_engine_\$eng\.py"; do$'
  '2 ^case "\$\{EXT_SCRIPT\[\$eng\]\}" in \*\.py\) ext_run=\(python3\) ;; \*\) ext_run=\(bash\) ;; esac$'
  '1 ^\[ "\$HAVE_LLAMA" = 1 \] && cell_add "\$cell" "\$d/llama-\$pid" python3 scripts/lib/crux_pty_chat\.py \\$'
  '1 ^cell_add "\$cell" "\$d/ollama-\$pid" python3 scripts/lib/crux_pty_chat\.py \\$'
  "1 ^if grep -q -- '--certification' scripts/lib/crux_inference_judge\.py; then\$"
  '1 ^python3 scripts/lib/crux_inference_judge\.py collect --manifest "\$MANIFEST" --prompts "\$PROMPTS" "\$\{CERT_ARGS\[@\]\}" \\$'
)
only_unported_python() { # <dogfood script>
  local lines want re got
  lines=$(python_lines "$1" | sed -E 's/^[0-9]+:[[:space:]]*//')
  for want in "${RATCHET[@]}"; do
    re=${want#* }
    got=$(printf '%s\n' "$lines" | grep -cE -- "$re" || true)
    [ "$got" = "${want%% *}" ] || return 1
    lines=$(printf '%s\n' "$lines" | grep -vE -- "$re" || true)
  done
  [ -z "$lines" ]
}

if [ "${1:-}" != --self-test ]; then
  fails=0
  run_cases "$LIB" "$DOGFOOD" || fails=$?
  hf_real "$LIB" "$HF" || fails=$((fails + $?))
  if only_unported_python "$DOGFOOD"; then
    printf 'PASS ratchet: %s calls python3 only for the plugin *.py runner, the PTY chat driver and the judge\n' "$DOGFOOD"
  else
    printf 'FAIL ratchet: %s python lines differ from the table:\n' "$DOGFOOD"; python_lines "$DOGFOOD" | sed 's/^/  /'
    fails=$((fails + 1))
  fi
  if [ "$fails" -gt 0 ]; then printf '%s: FAIL (%s)\n' "$PROG" "$fails"; exit 1; fi
  printf '%s: PASS — case table green, hf-sources inside the subset, dogfood python3 = the unported lines\n' "$PROG"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------
# --self-test: the unplanted library is green; every plant below must turn the case table RED.
st=0
if run_cases "$LIB" "$DOGFOOD" > "$T/base.out"; then printf 'ok   unplanted library: case table green\n'
else printf 'FAIL unplanted library is not green:\n'; sed 's/^/  /' "$T/base.out"; st=1; fi
plant_in() { # <lib|dogfood> <name> <case> <perl substitution> — on a copy; <case> must go RED with a wrong
  # answer (a crash of a case that expects rc 0 is a RED that proves nothing about the planted defect)
  local src m out rc=0 lib="$LIB" df="$DOGFOOD"
  if [ "$1" = lib ]; then src=$LIB; else src=$DOGFOOD; fi
  m="$T/plant.$2.sh"
  perl -pe "$4" "$src" > "$m"
  if cmp -s "$src" "$m"; then printf 'FAIL plant %s changed nothing (the substitution no longer matches)\n' "$2"; st=1; return; fi
  if [ "$1" = lib ]; then lib=$m; else df=$m; fi
  out=$(run_cases "$lib" "$df") || rc=$?
  if [ "$rc" = 0 ]; then printf 'FAIL plant %s SURVIVED: the case table stayed green\n' "$2"; st=1
  elif ! printf '%s\n' "$out" | grep -q "^FAIL $3:"; then printf 'FAIL plant %s: RED, but not on case %s\n' "$2" "$3"; st=1
  elif printf '%s\n' "$out" | grep -qE "^FAIL $3: rc [1-9][0-9]*, expected 0"; then
    printf 'FAIL plant %s: case %s crashed instead of answering wrong\n' "$2" "$3"; st=1
  else printf 'ok   plant %s: %s RED\n' "$2" "$3"; fi
}
plant() { plant_in lib "$@"; }
plant ascii-escapes-off     prompts-files 's/if test\("\^\[ -~\]\*\$"\) then tojson/if true then tojson/'
plant tilde-escaped         prompts-files 's/\| if \. < 127 then \./| if . < 126 then ./'
plant astral-not-paired     prompts-files 's/elif \. < 65536 then/elif . < 1114112 then/'
plant exponent-from-16      prompts-files 's/elif \$pp < -3 or \$pp > 16 then/elif \$pp < -3 or \$pp > 15 then/'
plant exponent-below-3      prompts-files 's/elif \$pp < -3 or \$pp > 16 then/elif \$pp < -2 or \$pp > 16 then/'
plant exponent-one-digit    prompts-files 's/if length < 2 then "0" \+ \. else \. end/if length < 1 then "0" + . else . end/'
plant negative-zero-int     prompts-files 's/\(if \$t == "-0" then "0" else \$t end\)/\$t/'
plant sig17-off             prompts-refused-sig18 's/if tojson \| _sig > 17 then/if tojson | _sig > 18 then/'
plant fold-not-int          prompts-folded-literal 's/if \$t \| test\("\^-\?\[0-9\]\+\$"\) then/if \$t | test("^-?[0-9]+\$") and (\$t | test("e";"i") | not) and false then/'
plant int-of-bool           prompts-files 's/then \(if \. then 1 else 0 end\)/then (if . then 0 else 1 end)/'
plant int-rounds            prompts-files 's/else sig17 \| \. \* 1 \| trunc end/else sig17 | . * 1 | round end/'
plant run-verb-not-served   prompts-files 's/if \$v == "run" then \["serve run", "serve stream"\]/if \$v == "chat" then ["serve run", "serve stream"]/'
plant only-ignored          prompts-only 's/select\(\(\$only \| length\) == 0 or/select(true or/'
plant unknown-unsorted      prompts-only-unknown 's/\| unique\) as \$unknown/) as \$unknown/'
plant maxtok-modes-swapped  prompts-files 's/\[\("off", "on"\) as \$th/[("on", "off") as \$th/'
plant max-tokens-min        max-tokens-on 's/else max \| intstr end end/else min | intstr end end/'
plant control-jq-truth      control-ids 's/\| select\(\.control \| pytrue\)/| select(.control != null and .control != false)/'
plant gen-rc-zero           gen-rows 's/then _intlit else null end\)/then _intlit else 0 end)/'
plant gen-unloaded-always   gen-rows 's/\.ollama_unloaded = \(\$unl == "true"\)/.ollama_unloaded = true/'
plant gen-empty-stderr      gen-rows 's/stderr: \(\$e \| orNull\)/stderr: \$e/'
plant rows-any-kind         rows-for 's/select\(\.kind == "gen" and/select(.kind != "x" and/'
plant think-tag-ignored     thinking-think 's/contains\("enable_thinking"\) or contains\("<think>"\)\)/contains("enable_thinking"))/'
plant think-map-values      thinking-obj 's/has\("enable_thinking"\) or has\("<think>"\)/has("enable_thinking")/'
plant tokreq-no-special     tokreq 's/add_special: true/add_special: false/'
plant blob-always-input     model-rows 's/then \$blob == \$sha else null end/then true else null end/'
plant ext-empty-probe       ext-meta 's/\{probe: \(\$ARGS\.named\["p\\\(\$k\)"\] \| orNull\)/{probe: \$ARGS.named["p\\(\$k)"]/'
plant meta-indent-4         meta 's/\| pydump\(2\)\x27/| pydump(4)\x27/'
plant meta-temp-int         meta 's/temperature: \(\$temp \| pyfloat_of_text\)/temperature: (\$temp | pyint)/'
plant meta-split-empty      meta-empty-verbs 's/def py_split: if \. == "" then \[""\] else split\(","\) end;/def py_split: split(",");/'
plant float-text-sig17-off  meta-temperature-refused-0.10000000000000000555 's/\| fromjson \| sig17 \| \. \* 1\)/| fromjson | . * 1)/'
plant blank-misses-nbsp     rows-for 's/ \\u0085\\u00a0\\u1680/ \\u0085\\u1680/'
plant hf-bool-key           hf-source-refused-boolkey 's/elif \$m\.k \| y11 then error/elif false then error/'
plant hf-dtype-default      hf-source 's/else "bfloat16" end\)/else "float16" end)/'
plant ollama-number-version ollama-version-refused-number 's/else error\("version is not a string"\) end/else .version end/'
plant jq-ok-wrong-literal   jq-ok "s/if \\[ \"\\\$t\" != \x27\\[1\\.0,100000000000000000001\\]\x27 \\]/if [ \"\\\$t\" != \x27[1,100000000000000000001]\x27 ]/"
plant_in dogfood port-in-use-taken   free-port-skips-used 's/^(\s*)case "\$used" in \*" \$p "\*\) continue ;; esac/$1:/'
plant_in dogfood port-reserved-taken free-port-skips-reserved 's/&& \[ "\$p" -le "\$\{res_hi\[r\]\}" \] && continue 2/\&\& false/'
plant_in dogfood port-start-fixed    free-port-random-start 's/^(\s*)start=\$\(\( \$\{SRANDOM:-\$\(\(RANDOM \* 32768 \+ RANDOM\)\)\} % n \)\)/$1start=0/'

# The ratchet's own table: must go RED on a new python call, a changed or removed one, and stay green on a
# comment that names python.
ratchet() { # <name> <want: green|red> <perl substitution> — on a copy of the dogfood
  local s="$T/dogfood.$1.sh" got=red
  perl -pe "$3" "$DOGFOOD" > "$s"
  if cmp -s "$DOGFOOD" "$s"; then printf 'FAIL ratchet plant %s changed nothing\n' "$1"; st=1; return; fi
  only_unported_python "$s" && got=green
  if [ "$got" = "$2" ]; then printf 'ok   ratchet %s: %s\n' "$1" "$got"
  else printf 'FAIL ratchet %s: %s, expected %s\n' "$1" "$got" "$2"; st=1; fi
}
if only_unported_python "$DOGFOOD"; then printf 'ok   ratchet on the real dogfood: green\n'
else printf 'FAIL ratchet on the real dogfood is not green\n'; st=1; fi
ratchet python3-snippet    red   's/^(free_port\(\) \{)/X=\$(python3 -c 1)\n$1/'
ratchet bare-python        red   's/^(free_port\(\) \{)/python -c 1\n$1/'
ratchet env-python         red   's/^(free_port\(\) \{)/\/usr\/bin\/env python3 x.py\n$1/'
ratchet a-py-script        red   's/^(free_port\(\) \{)/bash -c "x scripts\/lib\/y.py"\n$1/'
ratchet third-pty-chat     red   's/^(free_port\(\) \{)/cell_add "\$cell" "\$d\/llama-\$pid" python3 scripts\/lib\/crux_pty_chat.py \\\n$1/'
ratchet judge-changed      red   's/^python3 scripts\/lib\/crux_inference_judge\.py collect /python3 -m crux_judge collect /'
ratchet judge-removed      red   's/^python3 scripts\/lib\/crux_inference_judge\.py collect /: scripts\/lib\/crux_judge collect /'
ratchet judge-appended     red   's/^(python3 scripts\/lib\/crux_inference_judge\.py collect .*) \\$/$1 \&\& python3 -c 1 \\/'
ratchet pty-prefixed       red   's/^(\s*)(cell_add "\$cell" "\$d\/ollama-\$pid" python3)/$1python3 -c 1; $2/'
ratchet py-launcher        red   's/^(free_port\(\) \{)/py -3 x\n$1/'
ratchet python-variable    red   's/^(free_port\(\) \{)/"\$PYTHON" -c 1\n$1/'
ratchet pypy               red   's/^(free_port\(\) \{)/pypy3 -c 1\n$1/'
ratchet capital-python     red   's/^(free_port\(\) \{)/Python3 -c 1\n$1/'
ratchet comment-python     green 's/^(free_port\(\) \{)/# python3 is named in a comment only\n$1/'
ratchet indented-comment   green 's/^(free_port\(\) \{)/  # see crux_inference_judge.py\n$1/'

# The real-file check: green on the real file, RED on a file with no sha256 key and on a library whose
# hf-sources reader refuses everything.
if hf_real "$LIB" "$HF" > /dev/null; then printf 'ok   hf-real on the real file: green\n'
else printf 'FAIL hf-real on the real file is not green\n'; st=1; fi
if hf_real "$LIB" "$FX/hf.yaml" > /dev/null; then printf 'FAIL hf-real SURVIVED a file with no sha256 key\n'; st=1
else printf 'ok   hf-real: a file with no sha256 key is RED\n'; fi
perl -pe 's/^crux_hf_source\(\) \{$/crux_hf_source() { echo refused >&2; return 1;/' "$LIB" > "$T/lib.hf-refuses.sh"
if cmp -s "$LIB" "$T/lib.hf-refuses.sh"; then printf 'FAIL plant hf-refuses changed nothing\n'; st=1
elif hf_real "$T/lib.hf-refuses.sh" "$HF" > /dev/null; then printf 'FAIL hf-real SURVIVED a reader that refuses everything\n'; st=1
else printf 'ok   hf-real: a reader that refuses everything is RED\n'; fi
if [ "$st" = 0 ]; then printf '%s --self-test: PASS\n' "$PROG"; else printf '%s --self-test: FAIL\n' "$PROG"; fi
exit "$st"
