"""nightly_admission.py -- may a release be judged on last night's receipts? (#4040, #4045)

The long certification (full ladder + full CRUX) runs nightly on main (scripts/certify_nightly.sh). A release
is admitted only on a nightly that is, for EVERY required host:
  GREEN     its verdict says green (apr-nightly-certification/v1);
  FRESH     t_end within `max_age_h` hours (default 24) of now;
  UPSTREAM  measured at the cut or at an ANCESTOR of it -- never a sibling branch, never a later commit.
The newest such verdict per host is chosen. A host with none is refused BY NAME, with the nearest miss.

Admission does NOT bind the receipts to the cut: check_model_ladder.sh does that as for any receipt. A nightly
at an ancestor binds only through equivalence -- evidence-only, the scoped hotfix, or the #4037 carry-forward
(no path in nightly..cut reaches apr inference). Otherwise it is STALE BY SHA and the release re-measures.

    python3 scripts/lib/nightly_admission.py <nightly root> <cut sha> <out dir> <host>...
      -> <out>/receipts/<host>.json, <out>/crux/<host>-gpu.json, <out>/crux/prompt-certification.json
         (symlinks to the chosen night's files); one line per host; exit 0 admitted, 1 refused
"""

import glob
import json
import os
import subprocess
import sys
import time


def load_verdicts(root):
    out = []
    for f in glob.glob(os.path.join(root, "*", "*", "verdict.json")):
        try:
            v = json.load(open(f))
        except (OSError, ValueError):
            continue
        if isinstance(v, dict) and v.get("schema") == "apr-nightly-certification/v1":
            v["_path"] = f
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
        lad, crux = (v.get("ladder") or {}).get("receipt"), (v.get("crux") or {}).get("receipt")
        if not (lad and os.path.isfile(lad) and crux and os.path.isfile(crux)):
            bad.append("%s: the nightly's receipts are gone (%s, %s)" % (h, lad, crux))
            continue
        os.symlink(os.path.abspath(lad), os.path.join(out, "receipts", h + ".json"))
        os.symlink(os.path.abspath(crux), os.path.join(out, "crux", h + "-gpu.json"))
        c = (v.get("certification") or {}).get("path")
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
