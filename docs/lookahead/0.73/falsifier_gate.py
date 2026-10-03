#!/usr/bin/env python3
"""Falsifier gate for the 0.73 contract drafts, and the case table in K9-falsifier-gate.md (#3999).

The doc states what the installed pv does with a contract that has fewer falsifiers than
obligations. This script runs pv on each case of the doc's table, and fails when an exit code
differs from the table. A case is a set of copies of the drafts in contracts-draft/, written to
a temp dir: as drafted, with falsification_tests emptied, or cut to its first n obligations.

Usage: python3 docs/lookahead/0.73/falsifier_gate.py [--corpus REF]   (needs PyYAML, and pv on PATH or in $PV)
Exit 0 when every case matches, 1 when one differs, 2 when pv or a draft is missing.
--corpus REF prints the doc's numbers for the contracts/ tree at git ref REF, and checks that the
validate gate of `pv lint contracts/` (CI and the release run that command; PROVABILITY-001 is in
this gate) passes that tree with no PV-PRV-001 finding. It runs that one gate, because the full lint
also runs gates that need crates/ and a git work tree (verify, duplicate-stems), which an archive lacks.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PV = os.environ.get("PV") or shutil.which("pv")
WEIGHTS = '{"spec_depth":0,"falsification":1,"kani":0,"lean":0,"binding":0}'
CHECKS = {  # column -> pv subcommand and flags; the directory goes after the subcommand
    "lint": ("lint", "--no-cache"),
    "strict": ("lint", "--no-cache", "--strict"),
    "lint06": ("lint", "--no-cache", "--min-score", "0.6"),
    "gate": ("score", "--weights", WEIGHTS, "--min-score", "1.0", "--exit-code"),
}
COLUMNS = ("validate", *CHECKS)
BPM, CPU, NEON, WGF = "backend-parity-matrix-v1", "cpu-q4k-activation-quant-v1", "neon-q4k-q6k-v1", "wgpu-forward-v1"
DRAFTS = (BPM, CPU, NEON, WGF)
CASES = [  # case, files as (draft, how), exit codes in COLUMNS order
    ("BPM as drafted", [(BPM, "drafted")], (0, 0, 0, 0, 0)),
    ("CPU as drafted", [(CPU, "drafted")], (0, 0, 0, 0, 0)),
    ("NEON as drafted", [(NEON, "drafted")], (0, 0, 0, 1, 0)),
    ("WGF as drafted", [(WGF, "drafted")], (0, 0, 0, 0, 0)),
    ("BPM, falsifiers stripped", [(BPM, "stripped")], (0, 0, 0, 1, 1)),
    ("WGF, falsifiers stripped", [(WGF, "stripped")], (0, 0, 0, 1, 1)),
    ("CPU, falsifiers stripped", [(CPU, "stripped")], (1, 1, 1, 1, 1)),
    ("NEON, falsifiers stripped", [(NEON, "stripped")], (1, 1, 1, 1, 1)),
    ("the four drafts and stripped BPM", [(d, "drafted") for d in DRAFTS] + [(BPM, "stripped")], (0, 0, 0, 1, 1)),
    ("NEON cut to its first 5 obligations", [(NEON, 5)], (0, 0, 0, 1, 0)),
]
VERDICT = {True: "PASS", False: "FAIL"}
failures = []


def die(msg):
    print(f"falsifier_gate: {msg}", file=sys.stderr)
    sys.exit(2)


def pv(*args):
    return subprocess.run([PV, *map(str, args)], capture_output=True, text=True, check=False)


def score_rows(path):
    """pv's score records for a file or a directory: one dict for a file, a list under 'scores' for a directory."""
    run = pv("score", path, "--format", "json")
    if run.returncode != 0:
        die(f"pv score {path} exited {run.returncode}")
    out = json.loads(run.stdout)
    return out.get("scores", [out])


def obligations(row):
    """pv's falsification probe for each obligation; a probe whose detail ends '(unmatched)' is a test with no obligation."""
    return [p for p in row["probes"] if p["dimension"] == "falsification" and not str(p["detail"]).endswith("(unmatched)")]


def make(draft, how):
    """The text of one case file: the draft as written, with no falsifiers, or cut to its first `how` obligations."""
    path = HERE / "contracts-draft" / f"{draft}.yaml"
    if not path.exists():
        die(f"missing {path}")
    if how == "drafted":
        return path.read_text()
    doc = yaml.safe_load(path.read_text())
    if how == "stripped":
        doc["falsification_tests"] = []
    else:
        doc["proof_obligations"] = doc["proof_obligations"][:how]
        doc["verification_summary"]["total_obligations"] = how
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def outcomes(files):
    """Exit codes in COLUMNS order (validate is the worst file), and pv's default composite per file."""
    with tempfile.TemporaryDirectory(prefix="k9-") as tmp:
        d = Path(tmp)
        for draft, how in files:
            name = draft if how == "drafted" else f"{draft}-{how}"
            (d / f"{name}.yaml").write_text(make(draft, how))
        codes = [max(pv("validate", p).returncode for p in d.glob("*.yaml"))]
        codes += [pv(args[0], d, *args[1:]).returncode for args in CHECKS.values()]
        return tuple(codes), {r["stem"]: r["composite"] for r in score_rows(d)}


def check(name, codes, want, composite):
    ok = codes == want
    cells = [f"{c} {got}" + ("" if got == exp else f" (want {exp})") for c, got, exp in zip(COLUMNS, codes, want)]
    comps = ", ".join(f"{v:.3f}" for _, v in sorted(composite.items()))
    print(f"{VERDICT[ok]} {name}: {'; '.join(cells)}; composite {comps}")
    if not ok:
        failures.append(name)
    return composite


def run_cases():
    seen = {}
    for name, files, want in CASES:
        codes, composite = outcomes(files)
        seen[name] = check(name, codes, want, composite)
    drafted, cut = seen["NEON as drafted"][NEON], seen["NEON cut to its first 5 obligations"][f"{NEON}-5"]
    ok = cut > drafted
    print(f"{VERDICT[ok]} cutting NEON to 5 obligations raises its composite: {drafted:.3f} -> {cut:.3f}")
    if not ok:
        failures.append("cut")
    rows = [r for d in DRAFTS for r in score_rows(HERE / "contracts-draft" / f"{d}.yaml")]
    probes = [p for r in rows for p in obligations(r)]
    print(f"drafts: {sum(p['outcome'] is True for p in probes)} of {len(probes)} obligations have a test whose rule names them")
    print(f"{VERDICT[not failures]}: {len(failures)} of {len(CASES) + 1} checks differ from the table")
    return 1 if failures else 0


def toplevel():
    run = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if run.returncode != 0:
        die("not inside a git checkout")
    return run.stdout.strip()


def archive(ref, dest):
    """contracts/ and .pv.toml at ref, extracted under dest: the tree that CI and the release lint."""
    run = subprocess.run(["git", "-C", toplevel(), "archive", ref, "contracts", ".pv.toml"], capture_output=True, check=False)
    if run.returncode != 0:
        die(f"git archive {ref} exited {run.returncode}: {run.stderr.decode().strip()[:200]}")
    subprocess.run(["tar", "-x", "-C", str(dest)], input=run.stdout, check=True)
    return dest / "contracts"


def short(row):
    """Fewer falsification_tests than proof_obligations: pv's D2 is min(tests, obligations) / obligations."""
    return bool(obligations(row)) and row["falsification_coverage"] < 1


def resolve(rows, root):
    """The file behind each row. pv keys a score row by file stem, and stems repeat across directories."""
    by_stem = defaultdict(list)
    for p in sorted(root.rglob("*.yaml")):
        by_stem[p.stem].append(p)
    found = []
    for stem in sorted({r["stem"] for r in rows}):
        paths = by_stem[stem]
        found += paths if len(paths) == 1 else [p for p in paths if short(score_rows(p)[0])]
    return found


def statuses(root):
    """pv status of every YAML file under root, 16 at a time."""
    files = sorted(root.rglob("*.yaml"))
    with ThreadPoolExecutor(16) as pool:
        return dict(zip(files, pool.map(lambda p: pv("status", p).stdout, files)))


def metadata(path):
    """The raw kind and registry fields; pv resolves them in Contract::kind() (schema/types.rs)."""
    meta = (yaml.safe_load(path.read_text()) or {}).get("metadata") or {}
    return f"kind: {meta.get('kind', '(none)')}" + (", registry: true" if meta.get("registry") else "")


def pct(n, d):
    return f"{n} of {d} ({100 * n / d:.1f}%)" if d else f"{n} of 0"


def corpus_lines(rows, shorts, files, status):
    probes = [p for r in rows for p in obligations(r)]
    inert = [o for o in status.values() if "is INERT" in o]
    with_obligations = sum(bool(re.search(r"^Proof obligations: [1-9]", o, re.M)) for o in inert)
    kinds = Counter(map(metadata, files)).most_common()
    return [
        f"with proof_obligations: {sum(bool(obligations(r)) for r in rows)}; fewer falsification_tests than "
        f"obligations: {len(shorts)} ({sum(r['falsification_coverage'] == 0 for r in shorts)} with none)",
        "those files by metadata: " + "; ".join(f"{k} {n}" for k, n in kinds),
        f"obligations with a test whose rule names them: {pct(sum(p['outcome'] is True for p in probes), len(probes))}",
        f"composite below 0.6 at the default weights: {pct(sum(r['composite'] < 0.6 for r in rows), len(rows))}",
        f"legacy top-level falsifier block: {sum('legacy top-level' in o for o in status.values())} files; "
        f"INERT: {len(inert)} ({with_obligations} with proof_obligations)",
        f"the gate would fail {sum(r['falsification_coverage'] < 1 for r in rows)} rows here: the {len(shorts)} above, and "
        "the rest declare neither proof_obligations nor falsification_tests (D2 is then 0.0)",
    ]


def corpus(ref):
    with tempfile.TemporaryDirectory(prefix="k9-corpus-") as tmp:
        root = archive(ref, Path(tmp))
        rows = score_rows(root)
        shorts = [r for r in rows if short(r)]
        files = resolve(shorts, root)
        status = statuses(root)
        gate = subprocess.run([PV, "lint", "contracts/", "--gate", "validate"], cwd=tmp, capture_output=True, text=True, check=False)
        print(f"corpus {ref}, {pv('--version').stdout.splitlines()[0]}: {len(rows)} contracts scored, {len(status)} YAML files")
        for line in corpus_lines(rows, shorts, files, status):
            print(f"  {line}")
    try:
        prv = sum(f.get("rule_id") == "PV-PRV-001" for f in json.loads(gate.stdout).get("findings", []))
    except ValueError:
        die(f"pv lint --gate validate printed no JSON (exit {gate.returncode})")
    ok = gate.returncode == 0 and prv == 0 and len(files) == len(shorts)
    print(f"{VERDICT[ok]} `pv lint contracts/ --gate validate` exits {gate.returncode} with {prv} PV-PRV-001 findings "
          f"on this tree, which holds the {len(files)} files above ({len(shorts)} score rows)")
    return 0 if ok else 1


def main():
    if not PV:
        die("pv not found; set $PV")
    if len(sys.argv) == 3 and sys.argv[1] == "--corpus":
        return corpus(sys.argv[2])
    if sys.argv[1:]:
        die("usage: falsifier_gate.py [--corpus REF]")
    return run_cases()


if __name__ == "__main__":
    sys.exit(main())
