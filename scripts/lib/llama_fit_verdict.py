"""#4016: llama.cpp's fit verdict for one (model, host) cell.

Operator rule (2026-09-23): until apr has a placement tool at parity, llama.cpp's
fit logic is the placement authority for every model certification and test. A
cell may be measured only if the pinned `llama-fit-params` says the model fits
FULLY on this host's GPU at ctx >= 4096. A cell without that verdict is refused.

`llama-fit-params --model <gguf>` (tools/fit-params/fit-params.cpp at the pin)
exits 1 with "failed to fit CLI arguments to free memory" when nothing fits, and
on success prints ONE stdout line: `-c <ctx> -ngl <n>` followed by optional
`-ts <split>` and `-ot "<pattern>=<buft>,..."`. Per the #3994 fleet table,
`-ngl -1` means every layer on the GPU; a lower `-ngl` or any `-ot` override is
partial offload, which the ladder declares for no cell, so it is refused too.

Pure: every input is a value, so the case table needs no GPU and no tool.
"""

import json
import re
import sys

MIN_CTX = 4096

VERDICTS = ("fits", "does-not-fit", "partial-offload", "missing", "tool-absent", "unpinned")


def verdict(tool_found, version_out, pin, rc, stdout, free_mib, min_ctx=MIN_CTX):
    """The cell's fit record. `verdict == "fits"` is the only value that admits it."""
    rec = {"tool": "llama-fit-params", "pin": pin, "llama_cpp": None, "free_mib": free_mib,
           "ctx": None, "ngl": None, "rc": rc, "raw": (stdout or "").strip()[:300]}

    def out(v, reason):
        rec["verdict"], rec["reason"] = v, reason
        return rec

    if not tool_found:
        return out("tool-absent", "llama-fit-params is not installed on this host")
    m = re.search(r"\(([0-9a-f]{7,40})\)|commit ([0-9a-f]{7,40})", version_out or "")
    built = (m.group(1) or m.group(2)) if m else None
    rec["llama_cpp"] = built
    k = min(len(built or ""), len(pin))
    if not built or k < 7 or built[:k] != pin[:k]:
        return out("unpinned", f"llama-fit-params is built from {built or 'an unknown commit'}, not the pin {pin}")
    if rc != 0:
        return out("does-not-fit", f"llama-fit-params exit {rc}: the model does not fit this host's free memory")
    line = (stdout or "").strip().splitlines()[-1] if (stdout or "").strip() else ""
    c = re.search(r"(?:^|\s)-c (\d+)(?:\s|$)", line)
    n = re.search(r"(?:^|\s)-ngl (-?\d+)(?:\s|$)", line)
    if not (c and n):
        return out("missing", "llama-fit-params printed no `-c N -ngl N` line: no verdict")
    rec["ctx"], rec["ngl"] = int(c.group(1)), int(n.group(1))
    if rec["ngl"] != -1 or " -ot " in f" {line} " or " -ts " in f" {line} ":
        return out("partial-offload", f"fitted `{line}` is not full GPU placement")
    if rec["ctx"] < min_ctx:
        return out("does-not-fit", f"fitted ctx {rec['ctx']} < {min_ctx}")
    return out("fits", f"full GPU at ctx {rec['ctx']}")


def main(argv):
    # argv: tool_found(0/1) pin rc free_mib version_file stdout_file
    tool_found, pin, rc, free_mib, vfile, sfile = argv
    read = lambda p: open(p, errors="replace").read() if p and p != "-" else ""
    free = int(free_mib) if free_mib.lstrip("-").isdigit() else None
    print(json.dumps(verdict(tool_found == "1", read(vfile), pin, int(rc), read(sfile), free)))


if __name__ == "__main__":
    main(sys.argv[1:])
