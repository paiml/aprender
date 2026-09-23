#!/usr/bin/env bash
# check_moe_routes_through_one_dispatch.sh — every verb's model-construction entry point routes
# qwen3moe to the ONE dispatch, never to the dense model (#3987).
#
# WHY. #3714 landed a qwen3moe CUDA forward and routed it inside `run_inference`, so `apr run`
# and `apr qa` were green -- and 2 of the 4 verbs the operator's bar names never reached it:
#   chat   built the DENSE `OwnedQuantizedModelCuda`; a MoE file has no dense `ffn_gate`, so the
#          first forward dereferenced a null weight ("ffn_gate_ptr is null (0)") -> rc 8
#   serve  built the same dense model and a state with no quantized model -> 501 on every chat
#          route, 500 on /v1/completions; and its one MoE branch called the CPU-ONLY generator
#   code   spawns `apr serve`, so it inherited all of that
# Measured on gx10 at 65baa3eb8, Qwen3-30B-A3B-Instruct-2507-Q4_K_M. One defect, N entry points.
#
# TWO LAYERS.
#   1. THE KNOWN ENTRY POINTS. In each function below, the MoE predicate and the dispatch
#      (`run_qwen3_moe_generate_dispatch`) must appear BEFORE the dense construction / dense
#      generation it would otherwise reach -- order, not mere presence, because a check that
#      runs after the dense model is built has already crashed.
#   2. THE DERIVED UNIVERSE. Every dense GGUF CUDA construction site in apr-cli and serve's API
#      (`OwnedQuantizedModelCuda::new|with_max_seq_len(`) is ENUMERATED FROM THE SOURCE and must
#      be one this table has audited. A new site -- a new verb, a new path -- goes RED until
#      someone decides what qwen3moe does there. A hand-kept list of verbs is exactly how three
#      of them were missed.
#
# Exit: 0 all as expected · 1 a check failed · 2 could not check.
#       --self-test: 0 when each planted mutation turns this RED, 1 otherwise.
set -euo pipefail
ROOT="${MOE_GUARD_ROOT:-.}"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) [ $# -ge 2 ] || { echo "--root needs a value" >&2; exit 2; }; ROOT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_moe_routes_through_one_dispatch: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

check() { # check <root> -> 0 ok
  python3 - "$1" <<'PY'
import re, sys, os, subprocess
root = sys.argv[1]
def fn_body(path, name):
    src = open(os.path.join(root, path), encoding="utf-8").read()
    m = re.search(r'\bfn\s+' + re.escape(name) + r'\s*(<[^>]*>)?\s*\(', src)
    if not m: return None
    i = src.index("{", m.end()); d = 0; j = i
    while j < len(src):
        c = src[j]
        if c == "{": d += 1
        elif c == "}":
            d -= 1
            if d == 0: return src[i:j+1]
        j += 1
    return None
DISPATCH = "run_qwen3_moe_generate_dispatch("
# (verb, file, function, the MoE predicate, what the MoE route must CALL, what must come AFTER)
# Generation sites must call the dispatch. Construction sites must route AWAY from the dense
# model before building it (the dispatch runs later, per turn / per request).
ENTRY = [
  ("run",   "crates/aprender-serve/src/infer/inference_result.rs", None,
            "moe_forward_handles(", DISPATCH, "run_qwen35_generate_dispatch("),
  ("chat preload", "crates/apr-cli/src/commands/chat_session_02.rs", "preload_gguf",
            "is_qwen3_moe_gguf(", "cuda: None", "try_init_gguf_cuda("),
  ("chat turn", "crates/apr-cli/src/commands/chat_generate_session_02.rs", "generate_gguf_with_prompt",
            "is_qwen3_moe_gguf(", DISPATCH, "cached_gguf_cuda"),
  ("serve cuda start", "crates/apr-cli/src/commands/serve/handler_gpu_completion.rs", "start_gguf_server_cuda",
            "moe_forward_handles(", "with_moe_gpu(", "OwnedQuantizedModelCuda::with_max_seq_len("),
  ("serve chat", "crates/aprender-serve/src/api/cuda_chat_backend.rs", "try_qwen3_moe_backend",
            "is_qwen3_moe_arch(", DISPATCH, None),
  ("serve completions", "crates/aprender-serve/src/api/realize_handlers_embed_completion.rs", "try_quantized_completions",
            "moe_forward_handles(", DISPATCH, "generate_with_cache("),
]
bad = 0
for verb, path, fn, pred, must, after in ENTRY:
    if fn is None:
        body = open(os.path.join(root, path), encoding="utf-8").read()
        body = body[body.find("moe_forward_handles(") - 2000:] if "moe_forward_handles(" in body else body
    else:
        body = fn_body(path, fn)
    if body is None:
        print(f"  FAIL  {verb:18s} {fn}() not found in {path} -- the entry point moved; re-audit it"); bad = 1; continue
    p, d = body.find(pred), body.find(must)
    a = body.find(after) if after else len(body)
    if verb == "serve chat" and "qwen3_moe_generate::run_qwen3_moe_generate(" in body:
        print(f"  FAIL  {verb:18s} still calls the CPU-ONLY generator, not the dispatch"); bad = 1; continue
    # #3987b: the live CPU streamer is legitimate only on a CPU server; a CUDA server's
    # stream=true must reach the dispatch (measured: gx10 streamed on the CPU, U+FFFD).
    st = body.find("run_qwen3_moe_generate_streaming(")
    if verb == "serve chat" and st >= 0 and not (0 <= body.find("request.stream && state.moe_no_gpu()") < st):
        print(f"  FAIL  {verb:18s} stream=true reaches the CPU-ONLY streamer on a CUDA server"); bad = 1; continue
    if p < 0 or d < 0:
        print(f"  FAIL  {verb:18s} predicate {'present' if p >= 0 else 'MISSING'}, `{must}` {'present' if d >= 0 else 'MISSING'}"); bad = 1; continue
    if a >= 0 and not (p < a and d < a):
        print(f"  FAIL  {verb:18s} the MoE route comes AFTER `{after}` -- the dense path runs first"); bad = 1; continue
    print(f"  ok    {verb:18s} MoE predicate + `{must.rstrip('(')}` before the dense path")
# Layer 2: the derived universe of dense GGUF CUDA construction sites
AUDITED = {  # routed for qwen3moe by #3987, or MoE returns before the dense site
  "crates/apr-cli/src/commands/chat_load_tokenizers.rs",          # try_init_gguf_cuda: preload_gguf skips it for MoE
  "crates/apr-cli/src/commands/chat_generate_session_02.rs",       # generate_gguf fallback: MoE returns earlier
  "crates/apr-cli/src/commands/serve/handler_gpu_completion.rs",   # start_gguf_server_cuda: MoE returns earlier
}
# DEBT, not approval: dense GGUF CUDA construction outside the four release verbs, with no
# qwen3moe route yet. Tracked file-by-file in #3992; a file leaves this list only by moving
# to AUDITED. Listing it here is what lets a NEW file fail the guard.
KNOWN_UNROUTED = {
  "crates/apr-cli/src/commands/bench_moe.rs", "crates/apr-cli/src/commands/bench_safetensors.rs",
  "crates/apr-cli/src/commands/benchmark.rs", "crates/apr-cli/src/commands/cbtop_measure_batch.rs",
  "crates/apr-cli/src/commands/distill_q4k_teacher.rs", "crates/apr-cli/src/commands/gguf.rs",
  "crates/apr-cli/src/commands/gguf_generate_result.rs", "crates/apr-cli/src/commands/golden_output.rs",
  "crates/apr-cli/src/commands/gpu_isolation_result.rs", "crates/apr-cli/src/commands/kernel.rs",
  "crates/apr-cli/src/commands/output_verification.rs", "crates/apr-cli/src/commands/parity_03.rs",
  "crates/apr-cli/src/commands/parity_per_op.rs", "crates/apr-cli/src/commands/serve/handlers_include_01.rs",
  "crates/apr-cli/src/commands/showcase/benchmark.rs", "crates/apr-cli/src/commands/showcase/pipeline.rs",
  "crates/apr-cli/src/commands/speedup.rs",
}
out = subprocess.run(["git", "-C", root, "grep", "-lE", r"OwnedQuantizedModelCuda::(new|with_max_seq_len)\(",
                      "--", "crates/apr-cli/src/**/*.rs", "crates/aprender-serve/src/api/**/*.rs"],
                     capture_output=True, text=True).stdout.split()
sites = {f for f in out if not re.search(r'(_tests?\.rs|/tests/)', f)}
# #3992 (aprender-30): OwnedQuantizedModelCuda::with_max_seq_len / new now REFUSE MoE by name
# before CUDA init, so the 17 former KNOWN_UNROUTED files are audited by the constructor
# itself. Flipped only once the gate was an ancestor of the release (460dd0613), never on a
# promise. Layer 3 below is what makes that claim true rather than asserted.
AUDITED |= KNOWN_UNROUTED
KNOWN_UNROUTED = set()
# Layer 3: the gate that makes those AUDITED claims true.
def body_after(path, marker):
    src = open(os.path.join(root, path), encoding="utf-8").read()
    k = src.find(marker)
    if k < 0: return None
    i = src.index("{", k); d = 0; j = i
    while j < len(src):
        if src[j] == "{": d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0: return src[i:j+1]
        j += 1
    return None
GATE = "crates/aprender-serve/src/gguf/cuda/mod.rs"
wm = body_after(GATE, "pub fn with_max_seq_len(")
nw = body_after(GATE, "pub fn new(model: OwnedQuantizedModel")
ok_gate = (wm is not None and nw is not None
           and "Self::check_not_moe(model)" in wm
           and "Self::build(" in wm
           and wm.find("Self::check_not_moe(model)") < wm.find("Self::build(")
           and "CudaExecutor::new" not in wm
           and "Self::with_max_seq_len(" in nw)
if ok_gate:
    print(f"  ok    {'moe gate':18s} with_max_seq_len refuses MoE before build(); new delegates to it")
else:
    print(f"  FAIL  {'moe gate':18s} with_max_seq_len must call check_not_moe before build() (no CUDA init in it), and new must delegate"); bad = 1
# The one escape hatch, enumerated from source like layer 2: an escape hatch nobody
# enumerates is how the next bypass happens.
ALLOW_MOE_FORWARD = {"crates/apr-cli/src/commands/bench_moe.rs"}
esc = subprocess.run(["git", "-C", root, "grep", "-lF", "new_for_moe_forward(", "--",
                      "crates/apr-cli/src", "crates/aprender-serve/src"], capture_output=True, text=True).stdout.split()
esc = {f for f in esc if not f.endswith("gguf/cuda/mod.rs") and not re.search(r'(_tests?\.rs|/tests/)', f)}
if esc - ALLOW_MOE_FORWARD:
    print(f"  FAIL  {'moe escape hatch':18s} new_for_moe_forward used outside the allow-list: {sorted(esc - ALLOW_MOE_FORWARD)}"); bad = 1
else:
    print(f"  ok    {'moe escape hatch':18s} new_for_moe_forward only in {sorted(esc) or '(none)'}")
unaudited = sorted(sites - AUDITED - KNOWN_UNROUTED)
if unaudited:
    print(f"  FAIL  {'new dense sites':18s} dense GGUF CUDA construction in an UNAUDITED file: {unaudited}")
    print("        decide what qwen3moe does there, route it, then add the file to AUDITED"); bad = 1
else:
    print(f"  ok    {'derived universe':18s} {len(sites)} dense-construction file(s): {len(sites & AUDITED)} audited, {len(sites & KNOWN_UNROUTED)} known debt (#3992), 0 new")
sys.exit(bad)
PY
}

if [ "$SELF_TEST" = 1 ]; then
  T=$(mktemp -d) || exit 2
  # Remove only what mktemp made: never empty, never "/", never outside a temp root.
  _rm() {
    case "${T:-}" in
      /tmp/?*|/var/folders/?*) [ -d "$T" ] && rm -rf -- "$T" ;;
      *) : ;;
    esac
  }
  trap _rm EXIT
  git -C "$ROOT" ls-files -z crates/apr-cli/src crates/aprender-serve/src | (cd "$ROOT" && xargs -0 cp --parents -t "$T")
  git -C "$T" init -q && git -C "$T" add -A >/dev/null
  check "$T" > /dev/null || { echo "SELF-TEST FAILED: the shipped tree is already red" >&2; exit 1; }
  echo "self-test: shipped tree GREEN"
  mut() { # mut <label> <file> <python-replace-expr>
    cp "$T/$2" "$T/$2.orig"
    python3 -c "import io,sys; p=sys.argv[1]; s=io.open(p).read(); n=$3; assert n!=s, 'mutant changed nothing'; io.open(p,'w').write(n)" "$T/$2"
    git -C "$T" add -A >/dev/null
    if check "$T" > /dev/null; then echo "SELF-TEST FAILED: mutant '$1' stayed green" >&2; exit 1; fi
    echo "self-test: mutant '$1' RED"
    mv "$T/$2.orig" "$T/$2"; git -C "$T" add -A >/dev/null
  }
  mut "chat preload builds the dense model for MoE" crates/apr-cli/src/commands/chat_session_02.rs \
      "s.replace('    if is_qwen3_moe_gguf(mapped) {\n        return Ok(GgufPreload', '    if false {\n        return Ok(GgufPreload', 1)"
  mut "serve chat back on the CPU-only generator" crates/aprender-serve/src/api/cuda_chat_backend.rs \
      "s.replace('crate::infer::qwen3_moe_dispatch::run_qwen3_moe_generate_dispatch(', 'crate::infer::qwen3_moe_generate::run_qwen3_moe_generate(', 1)"
  mut "serve chat streams on the CPU on a CUDA server" crates/aprender-serve/src/api/cuda_chat_backend.rs \
      "s.replace('if request.stream && state.moe_no_gpu() {', 'if request.stream {', 1)"
  mut "serve completions loses its MoE route" crates/aprender-serve/src/api/realize_handlers_embed_completion.rs \
      "s.replace('run_qwen3_moe_generate_dispatch(', 'removed_dispatch(', 1)"
  mkdir -p "$T/crates/apr-cli/src/commands"
  printf 'fn new_verb() { let _ = OwnedQuantizedModelCuda::new(m, 0); }\n' > "$T/crates/apr-cli/src/commands/new_verb.rs"
  git -C "$T" add -A >/dev/null
  if check "$T" > /dev/null; then echo "SELF-TEST FAILED: a NEW dense construction site stayed green" >&2; exit 1; fi
  echo "self-test: mutant 'a new verb builds the dense model' RED"
  rm -f "$T/crates/apr-cli/src/commands/new_verb.rs"; git -C "$T" add -A >/dev/null
  mut "the MoE gate is removed from the dense constructor" crates/aprender-serve/src/gguf/cuda/mod.rs \
      "s.replace('        let model = Self::check_not_moe(model)?;\n', '', 1)"
  mkdir -p "$T/crates/apr-cli/src/commands"
  printf 'fn sneaky() { let _ = OwnedQuantizedModelCuda::new_for_moe_forward(m, 0); }\n' > "$T/crates/apr-cli/src/commands/sneaky.rs"
  git -C "$T" add -A >/dev/null
  if check "$T" > /dev/null; then echo "SELF-TEST FAILED: the escape hatch used outside the allow-list stayed green" >&2; exit 1; fi
  echo "self-test: mutant 'escape hatch used outside the allow-list' RED"
  echo "self-test: PASS"
  exit 0
fi

echo "qwen3moe routing: every verb reaches the ONE dispatch before any dense model (#3987)"
if check "$ROOT"; then echo "OK"; exit 0; fi
echo "FAIL: a verb can reach the dense model with a qwen3moe file (#3987)"; exit 1
