#!/usr/bin/env python3
"""adjudicate_known_red_receipt.py -- an OPERATOR KNOWN-RED adjudication of a NO-GO dogfood receipt.

WHY. A release cut's own check_publish_preflight.sh (called with no arguments by cascade-publish.sh)
reads R5 as: the newest `receipt-*.json` under PUBLISH_PREFLIGHT_RECEIPT_DIR says `verdict: GO` for this
commit and version. The cut's code cannot change. When the operator rules "ship as known failures"
(0.69.1, 2026-09-23), the dogfood receipt is honestly NO-GO on the declared rows, and R5 has no hook
that could read a ruling. This tool is that hook, OUTSIDE the cut: it never edits the real receipt.

WHAT IT DOES. Reads the real receipt and a known-red list, and REFUSES (exit 1, every reason printed) unless
  * the source is the NEWEST receipt in its directory, is `verdict: NO-GO`, has no deferral, and names
    exactly the expected commit and version, phase pre-publish;
  * EVERY row with result FAIL is on the list, and fails for the list's PINNED reason (a regex over its
    note -- the declared failure, not a new one under the same name);
  * every listed row DID fail (an entry that covers nothing is stale, and is refused, not tolerated).
Only then does it write `receipt-<source timestamp>-ADJUDICATED.json` into --out: the source's fields,
`verdict: GO` (what the cut's R5 can read), and the provenance R5 cannot print -- `verdict_basis`,
`source_receipt`, `source_sha256`, `source_verdict: NO-GO`, `known_red` (gate, ticket, reason) and the
ruling. The FILE NAME says ADJUDICATED, so the R5 line that prints it says so too.

argv: --source <receipt.json> --known-red <list.json> --commit <40-hex> --version <v> --out <dir>
      --self-test
list.json: {"release": "0.69.1", "ruling": "...", "date": "...",
            "rows": [{"gate": "...", "ticket": "#NNNN", "reason": "<regex over the row's note>"}]}
Exit: 0 written · 1 refused · 2 unreadable input.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import tempfile

TICKET = re.compile(r"#\d+")


def refusals(src_path, src, known, commit, version):
    why = []
    newest = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(src_path)), "receipt-*.json")))
    if not newest or os.path.abspath(newest[-1]) != os.path.abspath(src_path):
        why.append(f"the source is not the NEWEST receipt in its directory (newest: {newest[-1] if newest else None}) -- a later run would be ignored")
    if src.get("verdict") != "NO-GO":
        why.append(f"the source verdict is {src.get('verdict')!r}, not NO-GO -- a GO receipt needs no adjudication; use it as is")
    if src.get("commit") != commit:
        why.append(f"the source commit is {src.get('commit')!r}, not the cut {commit}")
    if src.get("version") != version:
        why.append(f"the source version is {src.get('version')!r}, not {version}")
    if src.get("phase") != "pre-publish":
        why.append(f"the source phase is {src.get('phase')!r}, not pre-publish")
    if src.get("deferred"):
        why.append(f"the source DEFERS {src.get('deferred')} -- DEFER is abolished (#3957 F1b); no ruling adjudicates it")
    if str(known.get("release")) != str(version):
        why.append(f"the known-red list is for release {known.get('release')!r}, not {version}")
    for k in ("ruling", "date"):
        if not str(known.get(k) or "").strip():
            why.append(f"the known-red list carries no {k!r} -- a known-red is recorded, never improvised")
    rows = known.get("rows") or []
    if not rows:
        why.append("the known-red list names no rows")
    byname = {}
    for r in rows:
        g, t, rx = r.get("gate"), r.get("ticket"), r.get("reason")
        if not g or not isinstance(t, str) or not TICKET.search(t) or not rx:
            why.append(f"known-red entry {r!r} needs a gate, a #NNNN ticket and a reason regex")
            continue
        try:
            byname[g] = (t, re.compile(rx))
        except re.error as e:
            why.append(f"known-red entry {g}: reason is not a valid regex ({e})")
    fails = [g for g in (src.get("gates") or []) if isinstance(g, dict) and g.get("result") == "FAIL"]
    for g in fails:
        k = byname.get(g.get("gate"))
        if k is None:
            why.append(f"FAIL row {g.get('gate')!r} is NOT on the operator known-red list: {str(g.get('note'))[:160]!r}")
        elif not k[1].search(str(g.get("note") or "")):
            why.append(f"FAIL row {g.get('gate')!r} fails for a reason the list did not pin ({k[1].pattern!r}): {str(g.get('note'))[:160]!r}")
    failed = {g.get("gate") for g in fails}
    for name in byname:
        if name not in failed:
            why.append(f"known-red row {name!r} did not FAIL on this receipt -- a stale entry is refused, not tolerated; drop it from the list")
    return why, fails


def adjudicate(src_path, known_path, commit, version, out_dir, say=print):
    try:
        raw = open(src_path, "rb").read()
        src = json.loads(raw)
        known = json.load(open(known_path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        say(f"UNREADABLE  {e}")
        return 2
    why, fails = refusals(src_path, src, known, commit, version)
    if why:
        for w in why:
            say(f"REFUSE  {w}")
        say(f"REFUSED: {len(why)} reason(s); nothing written")
        return 1
    ts = src.get("timestamp") or "unknown"
    rows = {r["gate"]: r for r in known["rows"]}
    adj = dict(src)
    adj.update({
        "verdict": "GO",
        "verdict_basis": "OPERATOR KNOWN-RED ADJUDICATION of a NO-GO dogfood receipt: every FAIL row is a recorded "
                         "known-red with its ticket, failing for its pinned reason. NOT a green dogfood.",
        "source_receipt": os.path.abspath(src_path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_verdict": "NO-GO",
        "known_red": [{"gate": g["gate"], "ticket": rows[g["gate"]]["ticket"], "note": g.get("note")} for g in fails],
        "ruling": {"text": known["ruling"], "date": known["date"], "release": known["release"]},
    })
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"receipt-{ts}-ADJUDICATED.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(adj, fh, indent=2)
    json.load(open(out, encoding="utf-8"))
    say(f"ADJUDICATED  {len(fails)} known-red FAIL row(s) ({', '.join(g['gate'] for g in fails)}) -> {out}")
    return 0


def self_test():
    C, V = "d" * 40, "0.69.1"
    known = {"release": V, "ruling": "ship now as known failures", "date": "2026-09-23",
             "rows": [{"gate": "check_model_ladder", "ticket": "#4001", "reason": "no scope seam"},
                      {"gate": "check_no_claim_literals", "ticket": "#4002", "reason": "claim literal"}]}
    def rc(**kw):
        return {"crate": "aprender", "version": V, "timestamp": "20260923T200000Z", "commit": C, "phase": "pre-publish",
                "deferred": [], "open_obligations": [], "verdict": "NO-GO",
                "gates": [{"gate": "fmt", "result": "PASS", "note": ""},
                          {"gate": "check_model_ladder", "result": "FAIL", "note": "judge: no scope seam at X2"},
                          {"gate": "check_no_claim_literals", "result": "FAIL", "note": "7 claim literal(s) drifted"}], **kw}
    def gates_with(extra=None, drop=None, note=None):
        r = rc()
        g = [x for x in r["gates"] if x["gate"] != drop]
        if note:
            g = [dict(x, note=note) if x["gate"] == "check_model_ladder" else x for x in g]
        if extra:
            g.append(extra)
        r["gates"] = g
        return r
    cases = [
        ("exactly the two known-red rows -> ADJUDICATED", rc(), known, 0, "ADJUDICATED"),
        ("an OTHER FAIL row -> refused", gates_with(extra={"gate": "cargo_test", "result": "FAIL", "note": "3 failed"}), known, 1, "NOT on the operator known-red list"),
        ("a listed row failing for another reason -> refused", gates_with(note="judge crashed: KeyError"), known, 1, "did not pin"),
        ("a listed row that PASSED (stale entry) -> refused", gates_with(drop="check_no_claim_literals"), known, 1, "did not FAIL"),
        ("a GO source -> refused", rc(verdict="GO"), known, 1, "needs no adjudication"),
        ("another commit -> refused", rc(commit="e" * 40), known, 1, "not the cut"),
        ("a deferral -> refused", rc(deferred=["x"]), known, 1, "DEFERS"),
        ("another phase -> refused", rc(phase="full"), known, 1, "not pre-publish"),
        ("a list for another release -> refused", rc(), dict(known, release="0.70.0"), 1, "not 0.69.1"),
        ("a list entry without a ticket -> refused", rc(), dict(known, rows=[dict(known["rows"][0], ticket="soon"), known["rows"][1]]), 1, "#NNNN ticket"),
        ("a list without its ruling -> refused", rc(), dict(known, ruling=""), 1, "no 'ruling'"),
    ]
    bad = 0
    for name, src, kn, want, needle in cases:
        with tempfile.TemporaryDirectory() as d:
            sd, od = os.path.join(d, "dogfood"), os.path.join(d, "out")
            os.makedirs(sd)
            sp = os.path.join(sd, "receipt-20260923T200000Z.json")
            json.dump(src, open(sp, "w"))
            kp = os.path.join(d, "known.json")
            json.dump(kn, open(kp, "w"))
            lines = []
            got = adjudicate(sp, kp, C, V, od, lines.append)
            wrote = glob.glob(os.path.join(od, "*-ADJUDICATED.json"))
            ok = got == want and any(needle in ln for ln in lines) and (bool(wrote) == (want == 0))
            if ok and want == 0:
                a = json.load(open(wrote[0]))
                ok = (a["verdict"] == "GO" and a["source_verdict"] == "NO-GO" and len(a["known_red"]) == 2
                      and a["source_sha256"] == hashlib.sha256(open(sp, "rb").read()).hexdigest())
            print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f" -> rc={got} wrote={wrote} {lines[-2:]}"))
            bad |= not ok
    # the NEWEST rule: a later receipt beside the source refuses
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "receipt-20260923T200000Z.json"); json.dump(rc(), open(sp, "w"))
        json.dump(rc(), open(os.path.join(d, "receipt-20260923T210000Z.json"), "w"))
        kp = os.path.join(d, "known.json"); json.dump(known, open(kp, "w"))
        lines = []
        got = adjudicate(sp, kp, C, V, os.path.join(d, "out"), lines.append)
        ok = got == 1 and any("not the NEWEST" in ln for ln in lines)
        print(("ok    " if ok else "FAIL  ") + "a later receipt beside the source -> refused")
        bad |= not ok
    print("SELF-TEST " + ("PASSED" if not bad else "FAILED"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--source"); ap.add_argument("--known-red"); ap.add_argument("--commit")
    ap.add_argument("--version"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not all([a.source, a.known_red, a.commit, a.version, a.out]):
        ap.error("--source, --known-red, --commit, --version and --out are all required")
    sys.exit(adjudicate(a.source, a.known_red, a.commit, a.version, a.out))


if __name__ == "__main__":
    main()
