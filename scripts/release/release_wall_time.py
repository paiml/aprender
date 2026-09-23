"""release_wall_time.py -- release-night wall time, MEASURED from the systems of record (#4045 M7, #4033).

The ledger (scripts/release/ledger.py) carries `t4_wall_minutes` under "unmeasured", and 0.69.1 ran by hand, so no
autopilot STATUS exists to derive it from. This reads the anchors that exist for EVERY release, whoever drove it:
  freeze      the freeze commit's committer time (--freeze <sha>: the first candidate the train measured)
  cut         the tagged commit's committer time
  tag         the annotated tag's tagger time
  cascade     the first and last crates.io publish of this version, over scripts/release/publish-order.txt
  release     the GitHub release's publishedAt
and derives the spans the #4033 levers are judged by. Nothing is typed in: a missing anchor is null, with the reason,
and its spans are null too. Never a guess.

    python3 scripts/release/release_wall_time.py <version> --freeze <sha> [--out <json>]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request

UA = "aprender-release-wall-time (noah@paiml.com)"


def _ts(s):
    """ISO-8601 (any offset, 'Z', fractional seconds) -> UTC epoch seconds, or None."""
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _iso(t):
    return None if t is None else datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spans(a):
    """a: {anchor: epoch|None} -> {span: minutes|None}. Pure: this is the part the case table drives."""
    def d(x, y):
        return round((a[y] - a[x]) / 60.0, 1) if a.get(x) is not None and a.get(y) is not None else None
    return {
        "freeze_to_release_min": d("freeze", "release"),
        "cut_to_release_min": d("cut", "release"),
        "cut_to_tag_min": d("cut", "tag"),
        "tag_to_cascade_start_min": d("tag", "cascade_first"),
        "cascade_min": d("cascade_first", "cascade_last"),
        "cascade_end_to_release_min": d("cascade_last", "release"),
    }


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _crates_io(name, version):
    req = urllib.request.Request("https://crates.io/api/v1/crates/%s/%s" % (name, version), headers={"User-Agent": UA})
    try:
        return json.load(urllib.request.urlopen(req, timeout=20))["version"]["created_at"]
    except Exception:
        return None


def measure(version, freeze):
    tag = "v" + version
    why, a = {}, {}
    a["freeze"] = _ts(_git("log", "-1", "--format=%cI", freeze)) if freeze else None
    if a["freeze"] is None:
        why["freeze"] = "no --freeze commit given or it does not resolve"
    a["cut"] = _ts(_git("log", "-1", "--format=%cI", tag + "^{commit}"))
    a["tag"] = _ts(_git("for-each-ref", "--format=%(taggerdate:iso-strict)", "refs/tags/" + tag))
    order = _git("show", tag + ":scripts/release/publish-order.txt") or ""
    names = [ln.split()[0] for ln in order.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    pub, missing = [], []
    for n in names:
        t = _ts(_crates_io(n, version))
        (pub.append(t) if t is not None else missing.append(n))
        time.sleep(1.0)   # crates.io's crawler policy: one request per second
    a["cascade_first"], a["cascade_last"] = (min(pub), max(pub)) if pub else (None, None)
    if missing or not names:
        why["cascade"] = "%d of %d crate(s) have no crates.io %s: %s" % (len(missing), len(names), version, missing[:10])
    r = subprocess.run(["gh", "release", "view", tag, "--repo", "paiml/aprender", "--json", "publishedAt", "-q", ".publishedAt"],
                       capture_output=True, text=True)
    a["release"] = _ts(r.stdout.strip()) if r.returncode == 0 else None
    for k in ("cut", "tag", "release"):
        if a[k] is None:
            why[k] = "not readable (git tag %s / gh release)" % tag
    return a, why, len(names), len(pub)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("version")
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    a, why, n, npub = measure(args.version, args.freeze)
    doc = {"schema": "apr-release-wall-time/v1", "version": args.version, "freeze_commit": args.freeze,
           "anchors_utc": {k: _iso(v) for k, v in a.items()}, "spans": spans(a), "crates": {"in_order": n, "published": npub},
           "unmeasured": why, "measured_at": _iso(time.time())}
    txt = json.dumps(doc, indent=2) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(txt)
    sys.stdout.write(txt)
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
