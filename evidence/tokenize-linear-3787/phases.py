#!/usr/bin/env python3
"""#3787: time the phases before a model loads, per prompt, for one binary.
Runs `apr run MODEL -i PROMPT --chat -n 1 -v --no-gpu`, stamps every stderr line on arrival,
and stops the process at the "Model loaded in" line (CPU prefill after it is out of scope).
Before (no [prepare] line): template ends at "[DEBUG] formatted_prompt", tokenize ends at
"[DEBUG] add_bos=... encoded N tokens". After: the [prepare] line gives parse/template/tokenize."""
import json, re, subprocess, sys, time
binary, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = sys.argv[4:]
rows = []
for p in prompts:
    t0 = time.monotonic()
    proc = subprocess.Popen([binary, "run", model, "-i", p, "--chat", "-n", "1", "-v", "--no-gpu"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    marks, prepare, load_ms, ntok = {}, None, None, None
    for line in proc.stderr:
        t = time.monotonic() - t0
        if line.startswith("[DEBUG] formatted_prompt") and "template_end" not in marks:
            marks["template_end"] = t
        m = re.search(r"encoded (\d+) tokens", line)
        if m and "tokenize_end" not in marks:
            marks["tokenize_end"] = t; ntok = int(m.group(1))
        m = re.match(r"\[prepare\] parse ([\d.]+) ms, template ([\d.]+) ms, tokenize ([\d.]+) ms: (\d+) tokens from (\d+) bytes", line)
        if m:
            prepare = dict(parse_ms=float(m.group(1)), template_ms=float(m.group(2)), tokenize_ms=float(m.group(3)), tokens=int(m.group(4)), bytes=int(m.group(5)))
        m = re.match(r"Model loaded in ([\d.]+)ms", line)
        if m:
            load_ms = float(m.group(1)); marks["loaded"] = t
            proc.kill(); break
    proc.wait()
    row = dict(prompt=p, wall_to_loaded_s=marks.get("loaded"), load_ms=load_ms, tokens=ntok, marks=marks, prepare=prepare)
    if prepare is None and "template_end" in marks and "tokenize_end" in marks:
        row["tokenize_ms_from_marks"] = (marks["tokenize_end"] - marks["template_end"]) * 1000
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(out, "w"), indent=1)
