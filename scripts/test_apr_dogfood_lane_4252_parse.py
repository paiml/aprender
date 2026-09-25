#!/usr/bin/env python3
"""#4252: the gx10 receipt names the served apr version, and the GPU proof survives the
HEALTH line. parse_gx10 is fed the line protocol gx10_ask's remote command prints.
Run: python3 scripts/test_apr_dogfood_lane_4252_parse.py
"""

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apr_dogfood_lane_4252 as lane  # noqa: E402

RESP = {"choices": [{"message": {"content": '{"verdict":"PASS"}'}}]}


def stdout(health_line, pid=220388, weights=None):
    st = {"pid": pid, "serving": True}
    if weights is not None:
        st["weights"] = weights
    return "\n".join([
        json.dumps(st),
        json.dumps(RESP),
        "CURL_RC=0",
        "PROBE=True",
        health_line,
        f"{pid}, 243 MiB",
        "TRACE_BEGIN",
        "gpu-layers: requested=all resolved=32 total=32 (backend=cuda)",
    ]) + "\n"


def main():
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
        if not ok:
            fails.append(name)

    resp, prov = lane.parse_gx10(stdout('HEALTH={"status":"ok","version":"0.69.3","compute_mode":"cuda"}'))
    check("response parsed", resp, RESP)
    check("apr version from gx10 /health", prov["apr"], "0.69.3")
    check("HEALTH line is not a GPU app", prov["gpu_apps"], ["220388, 243 MiB"])
    check("server holds the GPU", prov["server_holds_gpu"], True)
    check("used_gpu probe", prov["used_gpu_probe"], True)

    _, prov = lane.parse_gx10(stdout("HEALTH="))
    check("health down: apr is null, not a crash", (prov["apr"], prov["health"]), (None, None))
    check("health down: the GPU proof still stands", prov["server_holds_gpu"], True)

    try:
        lane.parse_gx10(stdout("HEALTH=").replace("CURL_RC=0", "CURL_RC=7"))
        check("curl failure raises", "no raise", "raise")
    except RuntimeError:
        check("curl failure raises", "raise", "raise")

    # REX-07 Ask 2: the served weights' sha256 rides in state.json, line 0 of the protocol
    w = {"model": "/m/q.gguf", "model_sha256": "ab" * 32, "size": 3, "mtime_ns": 1}
    _, prov = lane.parse_gx10(stdout("HEALTH=", weights=w))
    check("weights sha256 from the watcher's state", (prov["model"], prov["model_sha256"]),
          ("/m/q.gguf", "ab" * 32))
    _, prov = lane.parse_gx10(stdout("HEALTH="))
    check("a watcher without weights: null, not a crash", prov["model_sha256"], None)

    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "m.gguf")
        with open(m, "wb") as f:
            f.write(b"weights-v1")
        ident = lane.weights_identity(m)
        check("sha256 is the file's", ident["model_sha256"], hashlib.sha256(b"weights-v1").hexdigest())
        fake = {**ident, "model_sha256": "cached"}
        check("unchanged stat reuses the hash", lane.weights_identity(m, fake)["model_sha256"], "cached")
        check("the file unchanged: still bound", lane.weights_still(ident)["model_sha256"], ident["model_sha256"])
        with open(m, "wb") as f:
            f.write(b"weights-v2-longer")
        check("changed after hashing: sha256 withdrawn", lane.weights_still(ident)["model_sha256"], None)
        check("changed file is re-hashed, not reused", lane.weights_identity(m, fake)["model_sha256"],
              hashlib.sha256(b"weights-v2-longer").hexdigest())
        gone = lane.weights_identity(os.path.join(d, "absent.gguf"))
        check("missing file: null with a reason", (gone["model_sha256"], "why" in gone), (None, True))

    print(("RED" if fails else "GREEN") + f" parse_gx10: {len(fails)} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
