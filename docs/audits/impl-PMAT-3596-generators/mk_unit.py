"""#3596 unit-level parity readings from the tests-<sha>-<host>.log files ([3596] lines)."""
import re, sys
from collections import OrderedDict

CFG = re.compile(r"\[3596\] \S+?/([^/ ]+)\.gguf n=(\d+) \(chunk rows (\d+), attention (.+?), rows/pass (\d+)\)(.*?): (.*)")
CMP = re.compile(r"argmax batched (\d+) / per-token (\d+), cosine ([\d.]+), rel L∞ ([\d.e+-]+)")
ST = re.compile(r"worst state rel L∞ ([\d.e+-]+) at (.*)")
FL = re.compile(r"\[3596\] flash prefill heads (\d+)/(\d+) rows (\d+) pos0 (\d+): rel L∞ vs f16-input reference ([\d.e+-]+), vs full-f32 reference ([\d.e+-]+)")

agg, flash = OrderedDict(), []
for arg in sys.argv[1:]:
    host, path = arg.split("=", 1)
    for line in open(path, errors="replace"):
        if m := FL.search(line):
            flash.append((host, *m.groups()))
            continue
        m = CFG.search(line)
        if not m:
            continue
        model, n, chunk, att, rpp, _what, rest = m.groups()
        att = "flash" if att.startswith("flash") else "cuBLAS f32"
        k = (host, model, int(n), att, int(chunk), int(rpp))
        a = agg.setdefault(k, {"cmp": 0, "argmax_eq": 0, "cos": 1.0, "logits": 0.0, "state": 0.0, "where": ""})
        if c := CMP.search(rest):
            a["cmp"] += 1
            a["argmax_eq"] += c.group(1) == c.group(2)
            a["cos"] = min(a["cos"], float(c.group(3)))
            a["logits"] = max(a["logits"], float(c.group(4)))
        elif s := ST.search(rest):
            if float(s.group(1)) >= a["state"]:
                a["state"], a["where"] = float(s.group(1)), s.group(2)

print("| host | model | positions | attention | chunk rows | query rows/pass | argmax equal (last, split, decode) | min cosine (7 d.p.) | worst logits rel L∞ | worst state rel L∞ |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (host, model, n, att, chunk, rpp), a in agg.items():
    print(f"| {host} | {model} | {n} | {att} | {chunk} | {rpp} | {a['argmax_eq']}/{a['cmp']} | {a['cos']:.7f} | {a['logits']:.2e} | {a['state']:.2e} ({a['where']}) |")
if flash:
    print("\nThe flash kernel alone, against exact attention in f64 (`gdn_flash_prefill_matches_reference_*`):\n")
    print("| host | q heads / kv heads | query rows | pos0 (cached prefix) | rel L∞ vs f16-input reference (bound 2e-3) | vs full-f32 reference |")
    print("|---|---|---|---|---|---|")
    for host, h, kv, rows, pos0, e16, e32 in flash:
        print(f"| {host} | {h}/{kv} | {rows} | {pos0} | {float(e16):.2e} | {float(e32):.2e} |")
