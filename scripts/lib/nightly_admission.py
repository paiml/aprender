"""nightly_admission.py -- may a release be judged on last night's receipts? (#4040, #4045)

The long certification (full ladder + full CRUX) runs nightly on main (scripts/certify_nightly.sh). A release
is admitted only on a nightly that is, for EVERY required host:
  GREEN     its verdict says green (apr-nightly-certification/v1);
  FRESH     t_end within `max_age_h` hours (default 24) of now;
  UPSTREAM  measured at the cut or at an ANCESTOR of it -- never a sibling branch, never a later commit;
  COHERENT  its green is RE-DERIVED from the receipts it names, never trusted: the ladder receipt is at the
            verdict's sha with executed >= 1 and red == 0, and every CRUX lane receipt (gpu AND cpu -- every
            rung claims both) is a PASS. A t_end in the future is refused (a clock or a forgery).
The newest such verdict per host is chosen. A host with none is refused BY NAME, with the nearest miss.

Admission does NOT bind the receipts to the cut: check_model_ladder.sh does that as for any receipt. A nightly
at an ancestor binds only through equivalence -- evidence-only, the scoped hotfix, or the #4037 carry-forward
(no path in nightly..cut reaches apr inference). Otherwise it is STALE BY SHA and the release re-measures.

    python3 scripts/lib/nightly_admission.py <nightly root> <cut sha> <out dir> <host>...
      -> <out>/receipts/<host>.json, <out>/crux/<host>-{gpu,cpu}.json, <out>/crux/prompt-certification.json
         (symlinks to the chosen night's files); one line per host; exit 0 admitted, 1 refused
"""

import glob
import json
import os
import subprocess
import sys
import time


REQUIRED_LANES = ("gpu", "cpu")
SKEW_S = 300.0


def _load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError, TypeError):
        return None


def resolve(v, p):
    """#4117: a receipt path recorded in a verdict -> (path on THIS host, why-not). A RELATIVE path is resolved
    against the directory of the verdict.json it was read from, so a night measured on one host and copied under
    another root (the two-host release gate judges on one) is still found; one that climbs out of that directory
    (`..`) is refused, never followed. An ABSOLUTE path is taken as recorded: nights written before this change."""
    if not p:
        return None, "no receipt recorded"
    if os.path.isabs(p):
        return p, None
    base = os.path.dirname(os.path.abspath(v.get("_path") or ""))
    q = os.path.normpath(os.path.join(base, p))
    if not q.startswith(base + os.sep):
        return None, "the recorded path %r escapes the verdict's directory" % p
    return q, None


def coherent(v):
    """-> None when the verdict's own receipts say green, else why not. Re-derived, never read from `green`."""
    lad_p, why = resolve(v, (v.get("ladder") or {}).get("receipt"))
    if why and "escapes" in why:
        return "its ladder receipt: %s" % why
    lad = _load(lad_p) if lad_p else None
    if not isinstance(lad, dict):
        return "its ladder receipt %s is unreadable" % lad_p
    if lad.get("apr_sha") != v.get("sha"):
        return "its ladder receipt is at %r, not the nightly's sha" % lad.get("apr_sha")
    if int(lad.get("executed") or 0) < 1 or int(lad.get("red") or 0) != 0:
        return "its ladder receipt is RED (executed=%s red=%s)" % (lad.get("executed"), lad.get("red"))
    lanes = (v.get("crux") or {}).get("lanes") or {}
    for lane in REQUIRED_LANES:
        rp, why = resolve(v, (lanes.get(lane) or {}).get("receipt"))
        if why and "escapes" in why:
            return "its CRUX %s receipt: %s" % (lane, why)
        r = _load(rp) if rp else None
        if not isinstance(r, dict) or (r.get("summary") or {}).get("verdict") != "PASS":
            return "its CRUX %s receipt is missing or not PASS" % lane
        # the same binding as the ladder receipt: the lane measured THIS nightly's binary (degraded quorum, Sonnet)
        got = _crux_sha(r)
        if got != v.get("sha"):
            return "its CRUX %s receipt measured apr %r, not the nightly's sha" % (lane, got)
    return None


def _crux_sha(r):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import model_ladder_crux   # the judge's own binding rule: apr.sha, or the version line resolved to a full sha
    return model_ladder_crux.apr_sha_of(r)


def load_verdicts(root):
    out = []
    for f in glob.glob(os.path.join(root, "*", "*", "verdict.json")):
        v = _load(f)
        if isinstance(v, dict) and v.get("schema") == "apr-nightly-certification/v1":
            v["_path"] = f
            v["_incoherent"] = coherent(v)
            out.append(v)
    return out


def select(verdicts, hosts, now, is_ancestor, max_age_h=24.0):
    """-> ({host: verdict}, {host: why}). Pure: the caller supplies time and ancestry."""
    chosen, refused = {}, {}
    for h in hosts:
        mine = [v for v in verdicts if v.get("host") == h]
        ok, misses = [], []
        for v in mine:
            sha, age = v.get("sha") or "", (now - float(v.get("t_end") or 0)) / 3600.0
            if v.get("green") is not True:
                misses.append("%s is RED (%s)" % (sha[:9], "; ".join(v.get("why") or [])[:120]))
            elif v.get("_incoherent"):
                misses.append("%s says green but %s" % (sha[:9], v["_incoherent"]))
            elif age < -SKEW_S / 3600.0:
                misses.append("%s has a t_end %.1f h in the FUTURE" % (sha[:9], -age))
            elif age > max_age_h:
                misses.append("%s is %.1f h old (> %g h)" % (sha[:9], age, max_age_h))
            elif not is_ancestor(sha):
                misses.append("%s is not the cut or an ancestor of it" % sha[:9])
            else:
                ok.append(v)
        if ok:
            chosen[h] = max(ok, key=lambda v: float(v.get("t_end") or 0))
        else:
            refused[h] = ("no nightly at all" if not mine else
                          "no GREEN nightly within %g h at an ancestor of the cut -- %s" % (max_age_h, "; ".join(misses[:3])))
    return chosen, refused


def git_is_ancestor(cut):
    def f(sha):
        if not sha:
            return False
        r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, cut], capture_output=True)
        return r.returncode == 0
    return f


def assemble(chosen, out):
    """Symlink each chosen night's receipts where check_model_ladder.sh reads them. -> [why] on failure."""
    bad = []
    os.makedirs(os.path.join(out, "receipts"), exist_ok=True)
    os.makedirs(os.path.join(out, "crux"), exist_ok=True)
    cert = None
    for h, v in sorted(chosen.items()):
        lad = resolve(v, (v.get("ladder") or {}).get("receipt"))[0]
        lanes = {k: resolve(v, (x or {}).get("receipt"))[0] for k, x in ((v.get("crux") or {}).get("lanes") or {}).items()}
        gone = [p for p in [lad] + [lanes.get(k) for k in REQUIRED_LANES] if not (p and os.path.isfile(p))]
        if gone:
            bad.append("%s: the nightly's receipts are gone (%s)" % (h, ", ".join(map(str, gone))))
            continue
        os.symlink(os.path.abspath(lad), os.path.join(out, "receipts", h + ".json"))
        for k in REQUIRED_LANES:
            os.symlink(os.path.abspath(lanes[k]), os.path.join(out, "crux", "%s-%s.json" % (h, k)))
        c = resolve(v, (v.get("certification") or {}).get("path"))[0]
        if c and os.path.isfile(c) and cert is None:
            cert = c
            os.symlink(os.path.abspath(c), os.path.join(out, "crux", "prompt-certification.json"))
    return bad


def main(argv):
    if len(argv) < 4:
        sys.stderr.write("usage: nightly_admission.py <nightly root> <cut sha> <out dir> <host>...\n")
        return 2
    root, cut, out, hosts = argv[0], argv[1], argv[2], argv[3:]
    max_age = float(os.environ.get("NIGHTLY_MAX_AGE_H", "24"))
    chosen, refused = select(load_verdicts(root), hosts, time.time(), git_is_ancestor(cut), max_age)
    for h in hosts:
        if h in chosen:
            v = chosen[h]
            print("ok    NIGHTLY %s: GREEN at %s, %.1f h old -- %s" % (
                h, v["sha"][:9], (time.time() - float(v["t_end"])) / 3600.0, v["_path"]))
        else:
            print("FAIL  NIGHTLY %s: %s (#4040)" % (h, refused[h]))
    if refused:
        return 1
    bad = assemble(chosen, out)
    for b in bad:
        print("FAIL  NIGHTLY %s (#4040)" % b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
