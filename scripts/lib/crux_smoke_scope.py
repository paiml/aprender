"""crux_smoke_scope.py -- the 0.69.1 OPERATOR EMERGENCY SCOPE: CRUX smoke only.

OPERATOR, verbatim, relayed by the release cop aprender-cf, 2026-09-23: "for this particular release
we need emergency surgery.  after a binary is build we really only care about CRUX on gx10 and lambda
(but "smoke test") style.  anything huge, must be nightly only".

So for the ONE release named in its contract entry (`ladder.emergency_scopes`, name `crux-smoke`),
the release gate is NOT the model ladder. It is CRUX smoke receipts from the RELEASE BINARY:
  - every host the entry names has CRUX receipts;
  - every receipt is bound to the cut EXACTLY (apr sha == cut, no equivalence: it is the binary
    being released) and did not DECLINE;
  - for every host x certified model x thinking mode the certification ADMITS for that model
    (`admitted_by_sha_thinking`; cop correction 2026-09-23: the operator never said "OFF/ON"): at least
    one cell exists, every such cell is GREEN, and a POSITIVE-CONTROL cell is GREEN (a lane that cannot
    answer its own control proves nothing). A certified model with NO admitted mode is RED, and a cell
    in a mode the certification does not admit never counts toward the pass.
Used for any other release, the scope is refused. The output always says it ran under the emergency
scope, and never that every rung was green.
"""

import glob
import json
import os
import re

import model_ladder_crux

HEX40 = re.compile(r"[0-9a-f]{40}")


def _semver(v):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(v or ""))
    return tuple(int(g) for g in m.groups()) if m else None


def judge(L, version, crux_dir, cert_p, cut_sha, scope_name, out):
    """-> True when the scope is NOT satisfied (RED).

    `release` is the NORMAL release gate from 0.70.0 (#4045): `ladder.release_gate.crux_smoke`, the same
    smoke rules, for every release from `release_gate.from` on -- no per-release override. The caller
    (check_model_ladder.sh --scope release) ALSO requires a nightly admitted and bound to the cut."""
    failed = False
    if scope_name == "release":
        g = L.get("release_gate") or {}
        cs = g.get("crux_smoke") or {}
        missing = [k for k in ("from", "ruling") if not g.get(k)] + [k for k in ("hosts", "thinking") if not cs.get(k)]
        if missing:
            out(f"FAIL  ladder.release_gate is missing {missing} -- the normal release gate is recorded, never improvised (#4045)")
            return True
        frm, cur = _semver(g["from"]), _semver(version)
        if frm is None or cur is None or cur < frm:
            out(f"FAIL  the normal release gate applies from {g['from']}, and this cut is {version} -- "
                f"an earlier release is judged by the full gate or its own recorded scope")
            return True
        out(f"RELEASE GATE (normal, #4045): CRUX smoke on the release binary + the nightly long certification -- {g.get('ruling')}")
        entry = {"hosts": cs["hosts"], "thinking": cs["thinking"]}
        return _smoke(entry, crux_dir, cert_p, cut_sha, out)
    entry = next((e for e in (L.get("emergency_scopes") or [])
                  if isinstance(e, dict) and e.get("name") == scope_name), None)
    if entry is None:
        out(f"FAIL  no emergency scope named {scope_name!r} in the ladder contract -- a scope is recorded, never improvised")
        return True
    missing = [k for k in ("release", "date", "quote", "hosts", "thinking") if not entry.get(k)]
    if missing:
        out(f"FAIL  emergency scope {scope_name} is missing {missing} -- it must carry the release, date and verbatim ruling")
        return True
    out(f"OPERATOR EMERGENCY SCOPE: CRUX smoke only -- release {entry['release']}, {entry['date']}, operator: {entry['quote']}")
    if str(version) != str(entry["release"]):
        out(f"FAIL  emergency scope {scope_name} is recorded for release {entry['release']} ONLY, and this cut is {version} "
            f"-- the full gate applies")
        return True
    return _smoke(entry, crux_dir, cert_p, cut_sha, out)


def _smoke(entry, crux_dir, cert_p, cut_sha, out):
    """The smoke rules, shared by the emergency scope and the normal release gate. -> True when RED."""
    failed = False
    if not HEX40.fullmatch(cut_sha or ""):
        out(f"FAIL  the cut {cut_sha!r} is not a full 40-hex sha -- smoke receipts are bound to the release binary's commit")
        return True
    certified, cfail = model_ladder_crux.load_certified(cert_p, out)
    if cfail or not certified:
        return True
    try:
        with open(cert_p, encoding="utf-8") as fh:
            by_mode = json.load(fh).get("admitted_by_sha_thinking")
    except (OSError, ValueError):
        by_mode = None
    if not isinstance(by_mode, dict):
        out("FAIL  the certification carries no admitted_by_sha_thinking -- the smoke matrix (model x admitted mode) is unknown")
        return True
    hosts = list(entry["hosts"])
    matrix = {}
    for sha in sorted(certified):
        m = by_mode.get(sha) if isinstance(by_mode.get(sha), dict) else {}
        matrix[sha] = [t for t in entry["thinking"] if m.get(t)]
    out("smoke matrix (certified model -> admitted thinking modes): "
        + "; ".join(f"{model_sha[:12]} -> {','.join(models) or 'NONE'}" for model_sha, models in matrix.items()))
    for sha, m in matrix.items():
        if not m:
            out(f"FAIL  certified model {sha[:12]} has NO admitted thinking mode -- nothing about it can be smoke-proven"); failed = True
    cells = {}          # (host, sha, thinking) -> [(verdict, positive_control)]
    seen_hosts = set()
    files = sorted(f for f in glob.glob(os.path.join(crux_dir or "", "*.json"))
                   if not os.path.basename(f).startswith("prompt-certification"))
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                R = json.load(fh)
        except (OSError, ValueError) as exc:
            out(f"FAIL  CRUX receipt {os.path.basename(f)} unreadable: {exc}"); failed = True
            continue
        if not isinstance(R, dict) or R.get("schema") != "crux-inference-receipt/v1":
            # crux_sweep_shards.sh writes the merge's meta as <host>-<backend>.meta.json BESIDE the receipt
            # (aprender-3a, 0.69.1 phase 3). It claims no schema and no cells: named and skipped. Any other
            # non-receipt here is still a FAIL.
            if os.path.basename(f).endswith(".meta.json") and isinstance(R, dict) and "schema" not in R and "cells" not in R:
                out(f"note  {os.path.basename(f)} is the sweep's merge meta, not a receipt -- skipped"); continue
            out(f"FAIL  {os.path.basename(f)} is not a crux-inference-receipt/v1"); failed = True
            continue
        asha = model_ladder_crux.apr_sha_of(R)
        if asha != cut_sha:
            out(f"FAIL  CRUX receipt {os.path.basename(f)} is from apr sha {asha!r}, not the release binary {cut_sha[:12]} "
                f"-- the emergency scope accepts only receipts from the binary being released"); failed = True
            continue
        if (R.get("summary") or {}).get("verdict") == "DECLINE":
            out(f"FAIL  CRUX receipt {os.path.basename(f)} DECLINED ({(R.get('summary') or {}).get('declined_because')})"); failed = True
            continue
        host = R.get("host")
        seen_hosts.add(host)
        for c in R.get("cells") or []:
            k = c.get("key") or {}
            cells.setdefault((k.get("host") or host, k.get("model_sha256"), k.get("thinking")), []).append(
                (c.get("verdict"), bool(c.get("positive_control"))))
    for h in hosts:
        if h not in seen_hosts:
            out(f"FAIL  host {h} has no CRUX receipt from the release binary -- the smoke gate needs every named host"); failed = True
            continue
        for sha in sorted(certified):
            for mode in matrix[sha]:
                got = cells.get((h, sha, mode)) or []
                if not got:
                    out(f"FAIL  {h} certified model {sha[:12]} thinking={mode}: no CRUX cell -- not smoke-tested"); failed = True
                    continue
                red = [v for v, _ in got if v != "GREEN"]
                ctl = [v for v, pc in got if pc]
                if red:
                    out(f"FAIL  {h} certified model {sha[:12]} thinking={mode}: {len(red)} of {len(got)} cell(s) not GREEN"); failed = True
                elif not ctl or any(v != "GREEN" for v in ctl):
                    out(f"FAIL  {h} certified model {sha[:12]} thinking={mode}: the positive control is missing or not GREEN "
                        f"-- a lane that cannot answer its control proves nothing"); failed = True
                elif all(pc for _, pc in got):
                    # every cell IS a control: "control GREEN" is then implied by "all GREEN", not a second check
                    out(f"ok    {h} certified model {sha[:12]} thinking={mode}: {len(got)} CRUX cell(s) GREEN -- every cell is a "
                        f"positive control (control-only smoke)")
                else:
                    out(f"ok    {h} certified model {sha[:12]} thinking={mode}: {len(got)} CRUX cell(s) GREEN, control GREEN")
    return failed


def release_gate_applies(L, version):
    """#4117: does the NORMAL release gate (`ladder.release_gate`, #4045) judge `version`? -> (bool, why).
    ONE rule for every caller -- autopilot's models step, R7, prepare_bump --ship, the candidate watch -- so they
    cannot disagree about which release has retired Phase 2. No `release_gate`, or a version before its `from`:
    False, and the full ladder still applies. A `from` or version that is not X.Y.Z raises: the caller must refuse,
    never guess which gate applies."""
    g = L.get("release_gate")
    if not isinstance(g, dict):
        return False, "the ladder contract records no release_gate -- the full ladder applies"
    frm, cur = _semver(g.get("from")), _semver(version)
    if frm is None or cur is None:
        raise ValueError(f"release_gate.from {g.get('from')!r} or version {version!r} is not X.Y.Z")
    if cur < frm:
        return False, f"the normal release gate applies from {g['from']}; {version} is before it -- the full ladder applies"
    return True, f"the normal release gate applies to {version} (ladder.release_gate.from {g['from']})"


if __name__ == "__main__":
    # python3 scripts/lib/crux_smoke_scope.py applies <ladder contract> <version>
    #   -> exit 0 the release gate applies, 1 it does not (the full ladder), 2 the question cannot be answered.
    #   The reason is printed either way, so a caller can say which gate it chose and why.
    import sys
    import yaml
    if len(sys.argv) != 4 or sys.argv[1] != "applies":
        print("usage: crux_smoke_scope.py applies <ladder contract> <version>", file=sys.stderr)
        sys.exit(2)
    try:
        applies, why = release_gate_applies(yaml.safe_load(open(sys.argv[2]))["ladder"], sys.argv[3])
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"the release gate cannot be decided for {sys.argv[3]} from {sys.argv[2]}: {exc}")
        sys.exit(2)
    print(why)
    sys.exit(0 if applies else 1)
