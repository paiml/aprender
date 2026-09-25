#!/usr/bin/env python3
"""#4252: the gx10 receipt names the served apr version, and the GPU proof survives the
HEALTH line. parse_gx10 is fed the line protocol gx10_ask's remote command prints.
Run: python3 scripts/test_apr_dogfood_lane_4252_parse.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apr_dogfood_lane_4252 as lane  # noqa: E402

RESP = {"choices": [{"message": {"content": '{"verdict":"PASS"}'}}]}


def stdout(health_line, pid=220388):
    return "\n".join([
        json.dumps({"pid": pid, "serving": True}),
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

    print(("RED" if fails else "GREEN") + f" parse_gx10: {len(fails)} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
