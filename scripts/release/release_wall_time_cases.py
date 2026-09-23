"""release_wall_time_cases.py -- case table + mutants for release_wall_time.py (#4045 M7).

    python3 scripts/release/release_wall_time_cases.py [--mutants]
The spans are driven with injected anchors, so no row reads the network; every mutant must break its NAMED row.
"""

import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(mod):
    iso = "2026-09-23T%s"
    a = {k: mod._ts(iso % v) for k, v in {"freeze": "14:54:22Z", "cut": "18:00:49+02:00", "tag": "17:48:43Z",
         "cascade_first": "21:09:06.864113Z", "cascade_last": "21:50:28Z", "release": "21:51:46Z"}.items()}
    s = mod.spans(a)
    res = {}
    res["offset-normalised-to-utc"] = (s["cut_to_tag_min"] == 107.9, s["cut_to_tag_min"])
    res["freeze-to-release"] = (s["freeze_to_release_min"] == 417.4, s["freeze_to_release_min"])
    res["tag-to-cascade-start"] = (s["tag_to_cascade_start_min"] == 200.4, s["tag_to_cascade_start_min"])
    res["cascade-span"] = (s["cascade_min"] == 41.4, s["cascade_min"])
    missing = dict(a, cascade_first=None)
    s2 = mod.spans(missing)
    res["missing-anchor-is-null-never-zero"] = (s2["cascade_min"] is None and s2["tag_to_cascade_start_min"] is None
                                               and s2["cut_to_release_min"] == 350.9, s2)
    res["unparseable-anchor-is-null"] = (mod._ts("yesterday") is None and mod._ts(None) is None, mod._ts("yesterday"))
    return res


MUTANTS = [
    ("naive-local-time", 'return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()',
     'return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None).timestamp()', "offset-normalised-to-utc"),
    ("missing-as-zero", "if a.get(x) is not None and a.get(y) is not None else None",
     "if a.get(x) is not None and a.get(y) is not None else 0.0", "missing-anchor-is-null-never-zero"),
    ("span-swapped", '"tag_to_cascade_start_min": d("tag", "cascade_first")', '"tag_to_cascade_start_min": d("cut", "cascade_first")',
     "tag-to-cascade-start"),
    ("parse-error-guessed", "    except ValueError:\n        return None", "    except ValueError:\n        return 0.0",
     "unparseable-anchor-is-null"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    src = os.path.join(HERE, "release_wall_time.py")
    res = run(load(src, "release_wall_time"))
    bad = [r for r, (ok, _) in res.items() if not ok]
    for r, (ok, got) in res.items():
        print("%s %-34s %s" % ("PASS" if ok else "FAIL", r, "" if ok else got))
    print("rows: %d pass, %d fail" % (len(res) - len(bad), len(bad)))
    if bad:
        return 1
    if "--mutants" not in sys.argv:
        return 0
    text, survived, tmp = open(src).read(), 0, tempfile.mkdtemp(prefix="wall-time-mut-")
    for name, old, new, row in MUTANTS:
        if text.count(old) != 1:
            print("MUTANT %-20s ANCHOR %d != 1" % (name, text.count(old))); survived += 1; continue
        p = os.path.join(tmp, name + ".py"); open(p, "w").write(text.replace(old, new))
        try:
            killed = not run(load(p, "m_" + name.replace("-", "_")))[row][0]; how = ""
        except Exception as exc:
            killed, how = False, " (CRASHED: %r)" % exc
        print("MUTANT %-20s %s by %s%s" % (name, "KILLED" if killed else "SURVIVED", row, how)); survived += not killed
    print("mutants: %d/%d killed by their named row" % (len(MUTANTS) - survived, len(MUTANTS)))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
