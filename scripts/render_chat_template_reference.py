#!/usr/bin/env python3
"""Regenerate the chat-template byte-equality oracle fixtures (#3755).

For every GGUF in the model directories, take the model's OWN `tokenizer.chat_template`,
and render it with HF's own renderer (transformers.utils.chat_template_utils
.render_jinja_template, the function `apply_chat_template` calls) for a fixed message
set x thinking {on, off, unset}. apr's `EmbeddedChatTemplate` must reproduce every
rendering byte for byte (crates/aprender-serve/src/chat_template_embedded_oracle.rs).

    python3 scripts/render_chat_template_reference.py [--models DIR ...] [--out DIR]

Writes <out>/<sha12>.jinja per distinct template, <out>/index.json (which models ship
which template, and their architecture), and <out>/reference.json (every rendering,
plus the renderer's name and version). Message contents contain no `<|`: apr sanitizes
that sequence in user content (F-SEC-220) and HF does not, a deliberate difference.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

MESSAGE_SETS = {
    "single": [{"role": "user", "content": "What is 2+2?"}],
    "system": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "multi": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
        {"role": "user", "content": "And 3+3?"},
    ],
}
MODES = {"on": {"enable_thinking": True}, "off": {"enable_thinking": False}, "unset": {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[os.path.expanduser("~/models")])
    ap.add_argument("--out", default="crates/aprender-serve/tests/fixtures/chat_templates")
    ap.add_argument(
        "--llama-server",
        help="also render every template with this llama-server (--jinja, i.e. minja) into "
        "llama_reference.json; the release comparator (CRUX) renders with llama.cpp, not HF",
    )
    ap.add_argument("--llama-model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct-IQ3_M.gguf"))
    args = ap.parse_args()
    if args.llama_server:
        return render_with_llama(args)

    from gguf import GGUFReader
    import transformers
    from transformers.utils.chat_template_utils import render_jinja_template

    os.makedirs(args.out, exist_ok=True)
    index = {}
    for d in args.models:
        for path in sorted(glob.glob(os.path.join(d, "*.gguf"))):
            reader = GGUFReader(path)
            field = reader.fields.get("tokenizer.chat_template")
            if field is None:
                continue
            arch_field = reader.fields.get("general.architecture")
            arch = bytes(arch_field.parts[arch_field.data[0]]).decode() if arch_field else "?"
            template = bytes(field.parts[field.data[0]]).decode("utf-8")
            sha = hashlib.sha256(template.encode()).hexdigest()[:12]
            with open(os.path.join(args.out, f"{sha}.jinja"), "w", encoding="utf-8") as f:
                f.write(template)
            entry = index.setdefault(sha, {"architectures": [], "models": []})
            if arch not in entry["architectures"]:
                entry["architectures"].append(arch)
            entry["models"].append(os.path.basename(path))
    if not index:
        print("no GGUF with a tokenizer.chat_template found", file=sys.stderr)
        return 2

    cases = {}
    for sha in sorted(index):
        with open(os.path.join(args.out, f"{sha}.jinja"), encoding="utf-8") as f:
            template = f.read()
        for set_name, messages in MESSAGE_SETS.items():
            for mode, kwargs in MODES.items():
                rendered = render_jinja_template(
                    conversations=[messages], chat_template=template, add_generation_prompt=True, **kwargs
                )
                prompt = rendered[0][0] if isinstance(rendered, tuple) else rendered[0]
                cases[f"{sha}/{set_name}/{mode}"] = prompt

    reference = {
        "renderer": f"transformers {transformers.__version__} render_jinja_template",
        "message_sets": MESSAGE_SETS,
        "cases": cases,
    }
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(index.items())), f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(args.out, "reference.json"), "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(index)} templates, {len(cases)} renderings, {reference['renderer']}")
    return 0


def render_with_llama(args) -> int:
    """Render each fixture template with llama-server's /apply-template (minja, --jinja).

    The small --llama-model is loaded on CPU and its template REPLACED with each fixture
    via --chat-template-file, so one tiny model renders every template. No inventory
    template reads bos_token/eos_token, so the carrier model does not enter the output.
    """
    import json as _json
    import subprocess
    import time
    import urllib.request

    version = subprocess.run([args.llama_server, "--version"], capture_output=True, text=True)
    version = (version.stdout + version.stderr).strip().splitlines()
    version = next((l for l in version if "version" in l), "unknown")
    index = _json.load(open(os.path.join(args.out, "index.json"), encoding="utf-8"))
    cases = {}
    port = 18000 + os.getpid() % 900
    for sha in sorted(index):
        proc = subprocess.Popen(
            [args.llama_server, "-m", args.llama_model, "--jinja", "--chat-template-file",
             os.path.join(args.out, f"{sha}.jinja"), "-ngl", "0", "-c", "2048", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                    break
                except Exception:
                    time.sleep(0.5)
            for set_name, messages in MESSAGE_SETS.items():
                for mode, kwargs in MODES.items():
                    body = {"messages": messages}
                    if kwargs:
                        body["chat_template_kwargs"] = kwargs
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/apply-template",
                        data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    reply = _json.load(urllib.request.urlopen(req, timeout=30))
                    cases[f"{sha}/{set_name}/{mode}"] = reply["prompt"]
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    reference = {"renderer": f"llama-server {version} /apply-template (--jinja)", "cases": cases}
    with open(os.path.join(args.out, "llama_reference.json"), "w", encoding="utf-8") as f:
        _json.dump(reference, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(cases)} renderings, {reference['renderer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
