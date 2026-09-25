#!/usr/bin/env python3
"""A fake `apr` for check_ladder_cells_producer.sh: the envelopes of run / chat / code / serve / inspect,
shaped as crates/apr-cli prints them, with ONE defect switched on by FAKE_APR_MODE:
  good | fellback (run+chat ran on cpu after asking for gpu) | noneedle (answers without the needle)
  | refuse (a pre-load capacity refusal at >= 8192 tokens) | noclose (a think block never closes)
  | undercount (the tokenizer is denser than the producer's first guess: 1 token per 6 chars)."""
import http.server, json, os, sys

MODE = os.environ.get("FAKE_APR_MODE", "good")
NEEDLE = "TANGERINE-4417"
CPT = 6.0 if MODE == "undercount" else 3.0  # chars per token of this fake tokenizer
TMPL = "{% if enable_thinking %}<think>{% endif %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}"


def answer(prompt, thinking):
    tok = int(len(prompt) / CPT)
    if MODE == "refuse" and tok >= 8192:
        sys.stderr.write("error: GPU capacity refused: weights 20000 MiB + KV 9000 MiB (8192 positions x 4 B at F32) "
                         "+ workspace 100 MiB + overhead 512 MiB = 29612 MiB, against 24000 MiB free of 24564 MiB "
                         "(discrete GPU: cuMemGetInfo). Pass --no-gpu to run the CPU forward instead\n")
        sys.exit(1)
    body = "Paris." if MODE == "noneedle" else NEEDLE
    think = ("<think>checking the first line" + ("" if MODE == "noclose" else "</think>")) if thinking == "on" else ""
    return tok, think + ("" if MODE == "noclose" and thinking == "on" else body)


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def main(a):
    verb = a[0]
    if verb == "inspect":
        md = {"general.architecture": "qwen2", "qwen2.context_length": "32768", "qwen2.block_count": "28",
              "qwen2.attention.head_count": "12", "qwen2.attention.head_count_kv": "2",
              "qwen2.embedding_length": "1536", "tokenizer.chat_template": TMPL, "general.file_type": "15"}
        print(json.dumps({"architecture": "qwen2", "metadata": md})); return 0
    gpu = MODE != "fellback"
    if verb == "run":
        tok, text = answer(open(opt(a, "--input")).read(), opt(a, "--thinking", "off"))
        print(json.dumps({"text": text, "prompt_tokens": tok, "used_gpu": gpu,
                          "backend": {"requested": "gpu", "ran": "gpu" if gpu else "cpu", "fell_back": not gpu}}))
        return 0
    if verb == "chat":
        lines = sys.stdin.read().split("\n")
        _, text = answer(lines[0], opt(a, "--thinking", "off"))
        print(text)
        print(json.dumps({"backend": {"requested": "gpu", "ran": "gpu" if gpu else "cpu", "fell_back": not gpu}}))
        return 0
    if verb == "code":
        _, text = answer(sys.stdin.read(), opt(a, "--thinking", "off"))
        print(json.dumps({"type": "result", "subtype": "success", "result": text, "session_id": "x", "duration_ms": 1}))
        return 0
    if verb == "serve":
        port = int(opt(a, "--port"))

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                self._send(200, {"status": "ok"})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                on = (req.get("chat_template_kwargs") or {}).get("enable_thinking")
                try:
                    tok, text = answer(req["messages"][0]["content"], "on" if on else "off")
                except SystemExit:
                    return self._send(503, {"error": "GPU capacity refused: = 29612 MiB, against 24000 MiB free of 24564 MiB"})
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}],
                                 "usage": {"prompt_tokens": tok}, "used_gpu": gpu})

        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
    sys.stderr.write(f"fake apr: unknown verb {verb}\n"); return 2


sys.exit(main(sys.argv[1:]))
