#!/usr/bin/env python3
"""serve_parity_gate.py -- the #4218 release gate's verdict, and nothing else.

apr serve vs the pinned llama-server, on the same Qwen3.5 GGUF, in the same run.
Every number this file compares against comes from contracts/serve-parity-gate-v1.yaml
(`gate:`) or scripts/llama_pin.toml; the file has no defaults of its own.

    verdict <run.json> [--write-receipt]   PASS=0  RED=1  NO-GO=3  usage=2
    promote <run.json>                     writes the baseline from a PASS-shaped run
    check-release <tag_sha> <asset_sha256> the release decision surfaces call this
    selftest                               the obligations, each flipped by a mutant

The run record is written by the driver (scripts/serve_parity_gate.sh). Its fields
are claims until this file has read them: a record's own `pass` or `verdict` is
never read, and every conclusion is recomputed here.
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile
import tomllib

import yaml

CONTRACT = "contracts/serve-parity-gate-v1.yaml"
PIN = "scripts/llama_pin.toml"
RUN_SCHEMA = "serve-parity-run/v1"
RECEIPT_SCHEMA = "serve-parity-receipt/v1"
BASELINE_SCHEMA = "serve-parity-baseline/v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

PASS, RED, NO_GO = "PASS", "RED", "NO-GO"
EXIT = {PASS: 0, RED: 1, NO_GO: 3}


def _finite(x):
    """A real, finite number. NaN compares False both ways, so it would pass
    every one-sided bound; a bool is an int in python and is never a measurement."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _count(x):
    return isinstance(x, int) and not isinstance(x, bool)


def load_gate(path=CONTRACT):
    with open(path) as fh:
        gate = (yaml.safe_load(fh) or {}).get("gate")
    need = ("host_class", "band_concurrency", "tolerance", "parity_target",
            "disk_floor_gb", "disk_path", "correctness", "baseline_file", "receipt_dir", "models")
    missing = [k for k in need if not isinstance(gate, dict) or k not in gate]
    corr = gate.get("correctness") if isinstance(gate, dict) else None
    missing += ["correctness." + k for k in ("tau", "max_rank", "min_teacher_forced_steps", "template")
                if isinstance(corr, dict) and k not in corr]
    if not missing:
        nums = [gate[k] for k in ("tolerance", "parity_target", "disk_floor_gb")]
        nums += [corr.get(k) for k in ("tau", "max_rank", "min_teacher_forced_steps")]
        if not all(_finite(x) for x in nums) or not 0 <= gate["tolerance"] < 1:
            missing = ["a finite number for every threshold (tolerance in [0, 1))"]
    if missing:
        raise SystemExit("serve_parity_gate: %s gate: is missing %s -- no defaults "
                         "are assumed" % (path, ", ".join(missing)))
    return gate


def load_pin(path=PIN):
    with open(path, "rb") as fh:
        return tomllib.load(fh)["comparator"]["build_commit"]


# ------------------------------------------------------------- the checks --
def preflight_reasons(run, gate):
    """NO-GO causes: the host, not the code."""
    pre = run.get("preflight") or {}
    out = []
    if pre.get("gpu_lock") != "free":
        out.append("gpu lock is %r, not 'free'" % (pre.get("gpu_lock"),))
    apps = pre.get("foreign_compute_apps")
    if not isinstance(apps, list):
        out.append("foreign_compute_apps was not recorded, so nobody asked "
                   "nvidia-smi who else was on the device")
    elif apps:
        out.append("foreign process(es) on the GPU: %s" % (apps,))
    disk = pre.get("disk") or {}
    free = disk.get("free_gb")
    if disk.get("path") != gate["disk_path"]:
        out.append("free disk was measured at %r, not the gate's %s"
                   % (disk.get("path"), gate["disk_path"]))
    if not isinstance(disk.get("resolved"), str) or not disk.get("resolved"):
        out.append("disk path %s was not resolved (readlink -f), so the filesystem "
                   "measured is unknown" % (gate["disk_path"],))
    if not _finite(free):
        out.append("free disk at %s was not recorded" % (gate.get("disk_path"),))
    elif free < gate["disk_floor_gb"]:
        out.append("free disk %.1f GB at %s (resolved %s) is under the %s GB floor"
                   % (free, disk.get("path"), disk.get("resolved"), gate["disk_floor_gb"]))
    return out


def correctness_reasons(model, gate):
    """A wrong output is RED. Recomputed from the disagreements, never `pass`."""
    c = gate["correctness"]
    rec = model.get("correctness") or {}
    size = model.get("size")
    out = []
    if rec.get("template") != c["template"]:
        out.append("%s: prompt template is %r, not %r" % (size, rec.get("template"), c["template"]))
    prompts = rec.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return out + ["%s: no correctness prompts recorded" % size]
    for i, p in enumerate(prompts):
        if p.get("prompt_ids_match") is not True:
            out.append("%s prompt %d: apr's prompt ids differ from the oracle's" % (size, i))
        steps = p.get("teacher_forced_steps")
        if not _count(steps) or steps < c["min_teacher_forced_steps"]:
            out.append("%s prompt %d: %r teacher-forced steps < %d"
                       % (size, i, steps, c["min_teacher_forced_steps"]))
        dis = p.get("disagreements")
        if not isinstance(dis, list):
            out.append("%s prompt %d: disagreements not recorded" % (size, i))
            continue
        for d in dis:
            gap = d.get("gap")
            rank = d.get("rank")
            if not _count(rank) or not 1 <= rank <= c["max_rank"]:
                out.append("%s prompt %d step %s: the oracle's token is apr's rank %r, outside [1, %d] "
                           "(rank 1 is a disagreement tied on the logit)"
                           % (size, i, d.get("step"), rank, c["max_rank"]))
            if not _finite(gap) or not 0 <= gap < c["tau"]:
                out.append("%s prompt %d step %s: apr disagrees with the oracle at gap "
                           "%r >= tau %s" % (size, i, d.get("step"), gap, c["tau"]))
    return out


def band_ratio(receipt, gate):
    """(point, lcb95, reason) of the gated band's decode ratio."""
    bands = (receipt or {}).get("bands") or []
    hits = [b for b in bands if b.get("concurrency") == gate["band_concurrency"]]
    if len(hits) != 1:
        return None, None, "%d c=%d bands in the receipt, need exactly 1" % (
            len(hits), gate["band_concurrency"])
    band = hits[0]
    if band.get("status") != "MEASURED":
        return None, None, "c=%d band status is %r, not MEASURED (%s)" % (
            gate["band_concurrency"], band.get("status"), band.get("status_reasons"))
    dec = (band.get("ratios") or {}).get("dec") or {}
    point, lcb = dec.get("point"), dec.get("lcb95")
    if not _finite(point) or not _finite(lcb) or point <= 0:
        return None, None, "c=%d band has no decode ratio point/lcb95" % gate["band_concurrency"]
    return float(point), float(lcb), None


def ratchet_floor(baseline_point, gate):
    return baseline_point * (1.0 - gate["tolerance"])


def below_ratchet(point, baseline_point, gate):
    """Fail closed: anything not provably at or above the floor is below it."""
    return not point >= ratchet_floor(baseline_point, gate)


def model_reasons(model, pinned, gate, pin_commit):
    size = model.get("size")
    out = []
    if model.get("gguf_sha256") != pinned["sha256"]:
        out.append("%s: GGUF sha %r is not the pinned %s" % (size, model.get("gguf_sha256"), pinned["sha256"]))
    receipt = model.get("receipt") or {}
    comp = ((receipt.get("provenance") or {}).get("comparator") or {}).get("commit")
    if comp != pin_commit:
        out.append("%s: comparator commit %r is not the pinned %s" % (size, comp, pin_commit))
    mem = model.get("memory") or {}
    for key in ("mem_available_kb", "apr_rss_kb", "comparator_rss_kb"):
        if not _count(mem.get(key)) or mem.get(key) <= 0:
            out.append("%s: memory.%s not recorded (unified memory: the ratio is "
                       "uninterpretable without it)" % (size, key))
    return out + correctness_reasons(model, gate)


def evaluate(run, gate, pin_commit, baseline):
    """The whole verdict. Returns (verdict, reasons, rows). A record too
    malformed to read is RED, never a crash that a caller might misread."""
    try:
        return _evaluate(run, gate, pin_commit, baseline)
    except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:
        return RED, ["run record is malformed: %s: %s" % (type(exc).__name__, exc)], []


def _evaluate(run, gate, pin_commit, baseline):
    if run.get("schema") != RUN_SCHEMA:
        return RED, ["run schema is %r, not %r" % (run.get("schema"), RUN_SCHEMA)], []
    nogo = preflight_reasons(run, gate)
    if nogo:
        return NO_GO, nogo, []
    reasons, rows = [], []
    if run.get("host_class") != gate["host_class"]:
        reasons.append("host_class %r is not the gate's %r" % (run.get("host_class"), gate["host_class"]))
    ver = ((run.get("binding") or {}).get("comparator") or {}).get("version_output") or ""
    if pin_commit not in ver:
        reasons.append("llama-server --version output %r does not name the pinned %s"
                       % (ver, pin_commit))
    base = (baseline or {}).get("models") or {}
    sizes = [m.get("size") for m in run.get("models") or []]
    dup = sorted({str(x) for x in sizes if sizes.count(x) > 1})
    if dup:
        reasons.append("model size(s) %s recorded more than once: which one was "
                       "measured is ambiguous" % ", ".join(dup))
    by_size = {m.get("size"): m for m in run.get("models") or []}
    for pinned in gate["models"]:
        size = pinned["size"]
        model = by_size.get(size)
        if model is None:
            reasons.append("%s: not measured" % size)
            continue
        mr = model_reasons(model, pinned, gate, pin_commit)
        point, lcb, why = band_ratio(model.get("receipt"), gate)
        if why:
            mr.append("%s: %s" % (size, why))
        row = {"size": size, "point": point, "lcb95": lcb}
        b = (base.get(size) or {}).get("point")
        if not _finite(b) or b <= 0:
            mr.append("%s: no committed baseline for host class %s" % (size, gate["host_class"]))
        elif point is not None:
            row.update(baseline_point=b, floor=ratchet_floor(b, gate))
            if below_ratchet(point, b, gate):
                mr.append("%s: decode ratio %.4f < ratchet floor %.4f (baseline %.4f x (1 - %s))"
                          % (size, point, ratchet_floor(b, gate), b, gate["tolerance"]))
        if point is not None:
            row["parity"] = "met" if point >= gate["parity_target"] else "unmet"
        row["reasons"] = mr
        rows.append(row)
        reasons.extend(mr)
    reasons.extend(control_reasons(run, gate, base, rows, pin_commit))
    return (RED if reasons else PASS), reasons, rows


def control_reasons(run, gate, base, rows, pin_commit):
    """A harness that cannot see a slowdown is blind: its PASS proves nothing.

    The control is judged by the SAME ratchet as the real run. Its reference
    is the committed baseline when one exists, and otherwise this run's own
    real point, which is what a promotion would commit. A control model is
    bound to the size it claims by the pinned GGUF sha and to the pinned
    comparator, or a lenient size's floor could judge another model's control.
    """
    ctl = run.get("control") or {}
    if not ctl.get("models"):
        return ["positive control absent: nothing showed this harness can see a slowdown"]
    if ctl.get("harness_sha") != run.get("harness_sha") or not run.get("harness_sha"):
        return ["positive control ran on harness %r, the run on %r"
                % (ctl.get("harness_sha"), run.get("harness_sha"))]
    real = {r["size"]: r.get("point") for r in rows}
    pins = {p["size"]: p["sha256"] for p in gate["models"]}
    out = []
    for m in ctl["models"]:
        size = m.get("size")
        if size not in pins or m.get("gguf_sha256") != pins[size]:
            out.append("control %s: GGUF sha %r is not the one pinned for that size"
                       % (size, m.get("gguf_sha256")))
            continue
        comp = (((m.get("receipt") or {}).get("provenance") or {}).get("comparator") or {}).get("commit")
        if comp != pin_commit:
            out.append("control %s: comparator commit %r is not the pinned %s" % (size, comp, pin_commit))
            continue
        point, _, why = band_ratio(m.get("receipt"), gate)
        ref = (base.get(size) or {}).get("point") or real.get(size)
        if why or not _finite(ref):
            out.append("control %s: unmeasurable (%s)" % (size, why or "no reference point"))
        elif not below_ratchet(point, ref, gate):
            out.append("control %s: the slowed fixture (%s ms/token) measured %.4f, not "
                       "under the ratchet floor %.4f -- the harness cannot see a slowdown"
                       % (size, ctl.get("delay_ms"), point, ratchet_floor(ref, gate)))
    return out


# ------------------------------------------------------------ the receipt --
def build_receipt(run, verdict, reasons, rows, gate):
    apr = (run.get("binding") or {}).get("apr") or {}
    return {
        "schema": RECEIPT_SCHEMA,
        "target": run.get("target"),
        "host_class": gate["host_class"],
        "green_sha": run.get("green_sha"),
        "harness_sha": run.get("harness_sha"),
        "bin_sha256": apr.get("bin_sha256"),
        "manifest_identity": apr.get("manifest_identity"),
        "manifest_run_id": apr.get("manifest_run_id"),
        "verdict": verdict,
        "reasons": reasons,
        "models": rows,
        "run_url": run.get("run_url"),
    }


def check_release(tag_sha, asset_sha256, receipt_dir):
    """(ok, reason). Every refusal names the one thing that was wrong."""
    tag_sha, asset_sha256 = tag_sha.lower(), asset_sha256.lower()
    if not SHA40.match(tag_sha):
        return False, "tag sha %r is not a full 40-hex commit" % tag_sha
    if not SHA256.match(asset_sha256):
        return False, "asset sha256 %r is not 64 hex" % asset_sha256
    path = os.path.join(receipt_dir, tag_sha + ".json")
    if not os.path.isfile(path):
        return False, "no serve-parity receipt at %s" % path
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, "receipt %s is unreadable: %s" % (path, exc)
    if rec.get("schema") != RECEIPT_SCHEMA:
        return False, "receipt schema %r is not %r" % (rec.get("schema"), RECEIPT_SCHEMA)
    if (rec.get("green_sha") or "").lower() != tag_sha:
        return False, "receipt measured %r, the tag is %s" % (rec.get("green_sha"), tag_sha)
    if "verdict" not in rec:
        return False, "receipt carries no verdict; absent is a refusal, never a pass"
    if rec["verdict"] != PASS:
        return False, "receipt verdict is %r: %s" % (rec["verdict"], rec.get("reasons"))
    if (rec.get("bin_sha256") or "").lower() != asset_sha256:
        return False, "receipt measured apr %r, the release asset is %s" % (rec.get("bin_sha256"), asset_sha256)
    return True, "PASS receipt bound to %s and apr %s" % (tag_sha, asset_sha256)


def promote(run, gate, pin_commit):
    """A baseline from a run that PASSes everything a baseline cannot supply.

    It is evaluated against its own points as the baseline. That makes the
    ratchet trivially true and leaves every other check live, the control
    included: the control is judged against this run's own points.
    """
    own = {"models": {}}
    for m in run.get("models") or []:
        point, _, _ = band_ratio(m.get("receipt"), gate)
        if point is not None:
            own["models"][m.get("size")] = {"point": point}
    verdict, reasons, rows = evaluate(run, gate, pin_commit, own)
    if verdict != PASS:
        return None, reasons
    return {
        "schema": BASELINE_SCHEMA,
        "host_class": gate["host_class"],
        "green_sha": run.get("green_sha"),
        "harness_sha": run.get("harness_sha"),
        "run_url": run.get("run_url"),
        "models": {r["size"]: {"point": r["point"], "lcb95": r["lcb95"]} for r in rows},
    }, []


# ------------------------------------------------------------------- CLI --
def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def cmd_verdict(args):
    gate, pin = load_gate(args.contract), load_pin(args.pin)
    base = _read_json(gate["baseline_file"]) if os.path.isfile(gate["baseline_file"]) else None
    run = _read_json(args.run)
    verdict, reasons, rows = evaluate(run, gate, pin, base)
    receipt = build_receipt(run, verdict, reasons, rows, gate)
    print(json.dumps(receipt, indent=2))
    if args.write_receipt:
        green = run.get("green_sha") or ""
        if not SHA40.match(green):
            print("serve_parity_gate: green_sha %r is not a full commit; no receipt "
                  "written" % green, file=sys.stderr)
            return 2
        os.makedirs(gate["receipt_dir"], exist_ok=True)
        with open(os.path.join(gate["receipt_dir"], green + ".json"), "w") as fh:
            json.dump(receipt, fh, indent=2)
            fh.write("\n")
    return EXIT[verdict]


def cmd_promote(args):
    gate, pin = load_gate(args.contract), load_pin(args.pin)
    if os.path.isfile(gate["baseline_file"]) and not args.replace:
        print("serve_parity_gate: %s exists; a ratchet is moved deliberately "
              "(--replace), never by accident" % gate["baseline_file"], file=sys.stderr)
        return 2
    base, reasons = promote(_read_json(args.run), gate, pin)
    if base is None:
        print("serve_parity_gate: NOT promoted:\n  " + "\n  ".join(reasons), file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(gate["baseline_file"]), exist_ok=True)
    with open(gate["baseline_file"], "w") as fh:
        json.dump(base, fh, indent=2)
        fh.write("\n")
    print("promoted %s from %s" % (gate["baseline_file"], base["green_sha"]))
    return 0


def cmd_check_release(args):
    gate = load_gate(args.contract)
    ok, why = check_release(args.tag_sha, args.asset_sha256, args.receipt_dir or gate["receipt_dir"])
    print(("PASS  " if ok else "REFUSED  ") + why)
    return 0 if ok else 1


# --------------------------------------------------------------- selftest --
def _fx_receipt(point, pin, status="MEASURED"):
    return {"provenance": {"comparator": {"commit": pin}},
            "bands": [{"concurrency": 1, "status": status, "status_reasons": [],
                       "ratios": {"dec": {"point": point, "lcb95": point * 0.95}}}]}


def _fx_run(gate, pin, point=0.9, ctl_point=0.4):
    models = []
    for m in gate["models"]:
        models.append({
            "size": m["size"], "gguf_sha256": m["sha256"],
            "receipt": _fx_receipt(point, pin),
            "memory": {"mem_available_kb": 100_000_000, "apr_rss_kb": 2_000_000,
                       "comparator_rss_kb": 2_000_000},
            "correctness": {"template": gate["correctness"]["template"], "prompts": [
                {"prompt_ids_match": True, "teacher_forced_steps": 32,
                 "disagreements": [{"step": 3, "gap": 0.01, "rank": 2}]}]}})
    return {
        "schema": RUN_SCHEMA, "host_class": gate["host_class"], "target": "aarch64-unknown-linux-gnu",
        "green_sha": "a" * 40, "harness_sha": "b" * 40, "run_url": "https://example.invalid/run/1",
        "binding": {"apr": {"bin_sha256": "c" * 64, "manifest_identity": "absent"},
                    "comparator": {"version_output": "version: 1 (%s)" % pin}},
        "preflight": {"gpu_lock": "free", "foreign_compute_apps": [],
                      "disk": {"path": "/mnt/nvme-raid0", "resolved": "/", "free_gb": 500.0}},
        "models": models,
        "control": {"harness_sha": "b" * 40, "delay_ms": 20,
                    "models": [{"size": gate["models"][0]["size"],
                                "gguf_sha256": gate["models"][0]["sha256"],
                                "receipt": _fx_receipt(ctl_point, pin)}]},
    }


def _fx_base(gate, point=0.9):
    return {"models": {m["size"]: {"point": point} for m in gate["models"]}}


def _set(path, value):
    def edit(run):
        node = run
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        return run
    return edit


def verdict_rows(gate, pin):
    """(name, run-edit, baseline point | None | baseline-edit, expected verdict)."""
    return [
        ("good run PASSes", None, 0.9, PASS),
        ("slowed run is RED (ratchet)", _set(["models", 1, "receipt"], _fx_receipt(0.7, pin)), 0.9, RED),
        ("within tolerance PASSes", _set(["models", 1, "receipt"], _fx_receipt(0.82, pin)), 0.9, PASS),
        ("no baseline is RED", None, None, RED),
        ("scrambled output is RED even when pass=true",
         _set(["models", 0, "correctness", "prompts", 0],
              {"pass": True, "prompt_ids_match": True, "teacher_forced_steps": 32,
               "disagreements": [{"step": 0, "gap": 3.0, "rank": 2}]}), 0.9, RED),
        ("NaN gap is RED", _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0, "gap"], float("nan")), 0.9, RED),
        ("bool gap is RED", _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0, "gap"], False), 0.9, RED),
        ("bool rank is RED", _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0, "rank"], True), 0.9, RED),
        ("too few teacher-forced steps is RED", _set(["models", 2, "correctness", "prompts", 0, "teacher_forced_steps"], 8), 0.9, RED),
        ("NaN decode ratio is RED", _set(["models", 2, "receipt", "bands", 0, "ratios", "dec", "point"], float("nan")), 0.9, RED),
        ("NaN decode lcb95 is RED", _set(["models", 2, "receipt", "bands", 0, "ratios", "dec", "lcb95"], float("nan")), 0.9, RED),
        ("NaN baseline is RED", None, float("nan"), RED),
        ("two c=1 bands is RED",
         lambda r: (r["models"][0]["receipt"]["bands"].insert(0, _fx_receipt(0.95, pin)["bands"][0]),
                    r["models"][0]["receipt"]["bands"][1]["ratios"]["dec"].update(point=0.5), r)[2], 0.9, RED),
        ("duplicate model size is RED",
         lambda r: (r["models"].insert(0, dict(r["models"][1], gguf_sha256="e" * 64)), r)[1], 0.9, RED),
        ("model on an unpinned comparator is RED",
         _set(["models", 3, "receipt", "provenance", "comparator", "commit"], "0" * 9), 0.9, RED),
        ("control of an ungated size is RED",
         _set(["control", "models", 0], {"size": "70B", "receipt": _fx_receipt(0.4, pin)}),
         lambda b: (b["models"].update({"70B": {"point": 0.9}}), b)[1], RED),  # a stale baseline key
        ("negative gap is RED", _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0, "gap"], -0.5), 0.9, RED),
        ("rank below 1 is RED", _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0, "rank"], 0), 0.9, RED),
        ("tied-logit disagreement (rank 1, gap 0) PASSes",
         _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0], {"step": 3, "gap": 0.0, "rank": 1}), 0.9, PASS),
        ("zero RSS is RED", _set(["models", 0, "memory", "apr_rss_kb"], 0), 0.9, RED),
        ("malformed models is RED", _set(["models"], "not-a-list"), 0.9, RED),
        ("oracle token deep in apr's ranking is RED",
         _set(["models", 1, "correctness", "prompts", 0, "disagreements", 0, "rank"], 3), 0.9, RED),
        ("prompt ids differ is RED", _set(["models", 2, "correctness", "prompts", 0, "prompt_ids_match"], False), 0.9, RED),
        ("apr-derived template is RED", _set(["models", 0, "correctness", "template"], "apr"), 0.9, RED),
        ("blind control is RED", _set(["control", "models", 0, "receipt"], _fx_receipt(0.9, pin)), 0.9, RED),
        ("absent control is RED", _set(["control"], {}), 0.9, RED),
        ("control with no models is RED", _set(["control", "models"], []), 0.9, RED),
        ("control mislabeled to another size is RED",
         _set(["control", "models", 0, "size"], gate["models"][4]["size"]), 0.9, RED),
        ("control on an unpinned comparator is RED",
         _set(["control", "models", 0, "receipt"], _fx_receipt(0.4, "0" * 9)), 0.9, RED),
        ("control on another harness is RED", _set(["control", "harness_sha"], "d" * 40), 0.9, RED),
        ("busy GPU lock is NO-GO", _set(["preflight", "gpu_lock"], "busy"), 0.9, NO_GO),
        ("foreign GPU process is NO-GO", _set(["preflight", "foreign_compute_apps"], [{"pid": 7, "name": "llama-server"}]), 0.9, NO_GO),
        ("unrecorded GPU apps is NO-GO", _set(["preflight", "foreign_compute_apps"], None), 0.9, NO_GO),
        ("low disk is NO-GO", _set(["preflight", "disk", "free_gb"], 5.0), 0.9, NO_GO),
        ("NaN free disk is NO-GO", _set(["preflight", "disk", "free_gb"], float("nan")), 0.9, NO_GO),
        ("unrecorded free disk is NO-GO", _set(["preflight", "disk", "free_gb"], None), 0.9, NO_GO),
        ("disk measured elsewhere is NO-GO", _set(["preflight", "disk", "path"], "/tmp"), 0.9, NO_GO),
        ("unresolved disk path is NO-GO", _set(["preflight", "disk", "resolved"], ""), 0.9, NO_GO),
        ("wrong GGUF is RED", _set(["models", 3, "gguf_sha256"], "e" * 64), 0.9, RED),
        ("unpinned comparator is RED", _set(["binding", "comparator", "version_output"], "version: 1 (39173bcac)"), 0.9, RED),
        ("band not MEASURED is RED", _set(["models", 4, "receipt"], _fx_receipt(0.9, pin, "COMPARATOR_STALE")), 0.9, RED),
        ("unrecorded RSS is RED", _set(["models", 0, "memory", "apr_rss_kb"], None), 0.9, RED),
        ("missing model is RED", lambda r: (r["models"].pop(), r)[1], 0.9, RED),
    ]


def release_rows():
    """(name, receipt-or-None, tag, asset, expected ok). aprender-36's table."""
    tag, asset = "a" * 40, "c" * 64
    good = {"schema": RECEIPT_SCHEMA, "green_sha": tag, "bin_sha256": asset, "verdict": PASS}
    return [
        ("missing receipt", None, tag, asset, False),
        ("sha != commit", dict(good, green_sha="f" * 40), tag, asset, False),
        ("verdict FAIL", dict(good, verdict=RED), tag, asset, False),
        ("verdict absent", {k: v for k, v in good.items() if k != "verdict"}, tag, asset, False),
        ("binary sha != asset", dict(good, bin_sha256="d" * 64), tag, asset, False),
        ("PASS bound to tag and asset", good, tag, asset, True),
    ]


# Each mutant is a textual edit of THIS file, run as a fresh module against
# the same tables. A mutant is caught when at least one row changes its
# outcome. The rows it must flip are listed, so a mutant that is "caught" by
# an unrelated row does not count.
MUTANTS = [
    ("model ratchet skipped", "if below_ratchet(point, b, gate):", "if False:",
     ["slowed run is RED (ratchet)"]),
    # The control is judged by the same function, so disabling it blinds the
    # control too: the good run must then turn RED, because its control no
    # longer goes RED.
    ("ratchet function disabled", "return not point >= ratchet_floor(baseline_point, gate)", "return False",
     ["good run PASSes"]),
    ("tolerance ignored", "return baseline_point * (1.0 - gate[\"tolerance\"])", "return baseline_point",
     ["within tolerance PASSes"]),
    ("missing baseline passes", "mr.append(\"%s: no committed baseline", "pass  # (\"%s: no committed baseline",
     ["no baseline is RED"]),
    ("gap check dropped", "0 <= gap < c[\"tau\"]", "0 <= gap",
     ["scrambled output is RED even when pass=true"]),
    ("prompt id check dropped", "if p.get(\"prompt_ids_match\") is not True:", "if False:",
     ["prompt ids differ is RED"]),
    ("control not judged", "elif not below_ratchet(point, ref, gate):", "elif False:",
     ["blind control is RED"]),
    ("control optional", "return [\"positive control absent", "return [] or [\"positive control absent",
     []),  # an equivalent edit: it must NOT flip anything (proves the harness is not trigger-happy)
    ("control presence skipped", "if not ctl.get(\"models\"):", "if not ctl.get(\"models\") and False:",
     ["control with no models is RED"]),
    ("control identity not bound", "if size not in pins or m.get(\"gguf_sha256\") != pins[size]:", "if False:",
     ["control mislabeled to another size is RED"]),
    ("control comparator not bound", "        if comp != pin_commit:\n            out.append(\"control", "        if False:\n            out.append(\"control",
     ["control on an unpinned comparator is RED"]),
    ("rank check dropped", "1 <= rank <= c[\"max_rank\"]", "1 <= rank",
     ["oracle token deep in apr's ranking is RED"]),
    ("unrecorded disk read as fine", "out.append(\"free disk at %s was not recorded\"", "pass  # (\"free disk at %s was not recorded\"",
     ["unrecorded free disk is NO-GO"]),
    ("disk path not bound", "if disk.get(\"path\") != gate[\"disk_path\"]:", "if False:",
     ["disk measured elsewhere is NO-GO"]),
    ("disk resolution not read", "if not isinstance(disk.get(\"resolved\"), str) or not disk.get(\"resolved\"):", "if False:",
     ["unresolved disk path is NO-GO"]),
    # The negated range already fails closed on NaN; _finite is what refuses a
    # bool gap (0 <= False < tau holds). Round-4 quorum measured it live.
    ("gap admits bool", "if not _finite(gap) or not 0", "if not isinstance(gap, (int, float)) or not 0",
     ["bool gap is RED"]),
    ("rank admits bool", "if not _count(rank) or not 1", "if not isinstance(rank, int) or not 1",
     ["bool rank is RED"]),
    ("steps floor dropped", "steps < c[\"min_teacher_forced_steps\"]", "False",
     ["too few teacher-forced steps is RED"]),
    # Equivalent today: band_ratio and the baseline check already refuse a
    # non-finite point, so the fail-closed ratchet is defence in depth. It must
    # flip nothing; if it ever does, an upstream guard was lost.
    ("ratchet NaN-blind", "return not point >= ratchet_floor(baseline_point, gate)", "return point < ratchet_floor(baseline_point, gate)",
     []),
    ("point NaN-blind", "if not _finite(point) or not _finite(lcb)", "if not isinstance(point, (int, float)) or not isinstance(lcb, (int, float))",
     ["NaN decode lcb95 is RED"]),
    ("free disk NaN-blind", "if not _finite(free):", "if not isinstance(free, (int, float)):",
     ["NaN free disk is NO-GO"]),
    ("duplicates not refused", "    if dup:\n", "    if False:\n",
     ["duplicate model size is RED"]),
    ("model comparator not bound", "    if comp != pin_commit:\n        out.append(\"%s: comparator", "    if False:\n        out.append(\"%s: comparator",
     ["model on an unpinned comparator is RED"]),
    ("control size not required gated", "if size not in pins or ", "if size in pins and ",
     ["control of an ungated size is RED"]),
    ("first c=1 band taken", "if len(hits) != 1:", "if not hits:",
     ["two c=1 bands is RED"]),
    ("gap lower bound dropped", "not 0 <= gap < c[\"tau\"]", "not gap < c[\"tau\"]",
     ["negative gap is RED"]),
    ("rank lower bound dropped", "not 1 <= rank <= c[\"max_rank\"]", "not rank <= c[\"max_rank\"]",
     ["rank below 1 is RED"]),
    ("rank floor too strict", "not 1 <= rank <= c[\"max_rank\"]", "not 2 <= rank <= c[\"max_rank\"]",
     ["tied-logit disagreement (rank 1, gap 0) PASSes"]),
    ("memory sign not read", "if not _count(mem.get(key)) or mem.get(key) <= 0:", "if not _count(mem.get(key)):",
     ["zero RSS is RED"]),
    ("malformed record crashes", "    except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:\n", "    except ZeroDivisionError as exc:\n",
     ["malformed models is RED"]),
    ("lock not read", "if pre.get(\"gpu_lock\") != \"free\":", "if False:",
     ["busy GPU lock is NO-GO"]),
    ("foreign apps not read", "elif apps:", "elif False:",
     ["foreign GPU process is NO-GO"]),
    ("disk floor dropped", "elif free < gate[\"disk_floor_gb\"]:", "elif False:",
     ["low disk is NO-GO"]),
    ("gguf sha not checked", "if model.get(\"gguf_sha256\") != pinned[\"sha256\"]:", "if False:",
     ["wrong GGUF is RED"]),
    ("comparator version not checked", "if pin_commit not in ver:", "if False:",
     ["unpinned comparator is RED"]),
    ("release: sha not bound", "if (rec.get(\"green_sha\") or \"\").lower() != tag_sha:", "if False:",
     ["sha != commit"]),
    ("release: absent verdict read as pass", "if \"verdict\" not in rec:", "if False:",
     ["verdict absent"]),
    ("release: verdict not read", "if rec[\"verdict\"] != PASS:", "if rec.get(\"verdict\", PASS) not in (PASS, RED):",
     ["verdict FAIL"]),
    ("release: binary not bound", "if (rec.get(\"bin_sha256\") or \"\").lower() != asset_sha256:", "if False:",
     ["binary sha != asset"]),
]


def run_tables(mod, gate, pin):
    """{row name: outcome} over both tables, under module `mod`."""
    out = {}
    for name, edit, bpoint, _ in verdict_rows(gate, pin):
        run = mod._fx_run(gate, pin)
        if edit:
            run = edit(run)
        if callable(bpoint):
            base = bpoint(mod._fx_base(gate, 0.9))
        else:
            base = mod._fx_base(gate, bpoint) if bpoint is not None else None
        try:
            out[name] = mod.evaluate(run, gate, pin, base)[0]
        except Exception as exc:  # a crash is an outcome, and a different one
            out[name] = "CRASH:%s" % type(exc).__name__
    with tempfile.TemporaryDirectory() as tmp:
        for name, rec, tag, asset, _ in release_rows():
            d = os.path.join(tmp, re.sub(r"\W+", "_", name))
            os.makedirs(d)
            if rec is not None:
                with open(os.path.join(d, tag + ".json"), "w") as fh:
                    json.dump(rec, fh)
            try:
                out[name] = mod.check_release(tag, asset, d)[0]
            except Exception as exc:
                out[name] = "CRASH:%s" % type(exc).__name__
    return out


def _load_variant(source, tmp, n):
    import importlib.util
    path = os.path.join(tmp, "spg_mutant_%d.py" % n)
    with open(path, "w") as fh:
        fh.write(source)
    spec = importlib.util.spec_from_file_location("spg_mutant_%d" % n, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_selftest(args):
    gate, pin = load_gate(args.contract), load_pin(args.pin)
    me = sys.modules[__name__]
    rc = 0
    expected = {r[0]: r[3] for r in verdict_rows(gate, pin)}
    expected.update({r[0]: r[4] for r in release_rows()})
    got = run_tables(me, gate, pin)
    for name, want in expected.items():
        ok = got.get(name) == want
        rc |= not ok
        print("%-5s %-48s expected %-6s got %s" % ("ok" if ok else "FAIL", name, want, got.get(name)))
    kinds = {PASS: 0, RED: 0, NO_GO: 0, True: 0, False: 0}
    for v in expected.values():
        kinds[v] += 1
    if min(kinds.values()) == 0:
        print("FAIL  the tables are one-sided: %s" % kinds)
        rc = 1
    with open(os.path.abspath(__file__)) as fh:
        source = fh.read()
    table_start = source.index("MUTANTS = [")
    with tempfile.TemporaryDirectory() as tmp:
        for n, (name, old, new, must_flip) in enumerate(MUTANTS):
            hits = source[:table_start].count(old)
            if hits != 1:
                print("FAIL  mutant %-40s anchor occurs %d times before the table (need 1)" % (name, hits))
                rc = 1
                continue
            mod = _load_variant(source.replace(old, new, 1), tmp, n)
            mgot = run_tables(mod, gate, pin)
            flipped = sorted(k for k in got if mgot.get(k) != got[k])
            if must_flip:
                ok = all(k in flipped for k in must_flip)
                verdict = "caught" if ok else "SURVIVED"
            else:
                ok = not flipped
                verdict = "inert (equivalent edit, as expected)" if ok else "FLIPPED an equivalent edit"
            rc |= not ok
            print("%-5s mutant %-40s %s %s" % ("ok" if ok else "FAIL", name, verdict, flipped))
    print("PASS" if rc == 0 else "FAIL")
    return 1 if rc else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", default=CONTRACT)
    ap.add_argument("--pin", default=PIN)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("run")
    v.add_argument("--write-receipt", action="store_true")
    p = sub.add_parser("promote")
    p.add_argument("run")
    p.add_argument("--replace", action="store_true")
    c = sub.add_parser("check-release")
    c.add_argument("tag_sha")
    c.add_argument("asset_sha256")
    c.add_argument("--receipt-dir")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    return {"verdict": cmd_verdict, "promote": cmd_promote,
            "check-release": cmd_check_release, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
