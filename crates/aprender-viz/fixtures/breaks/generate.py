#!/usr/bin/env python3
"""Regenerate manifest.json from the reference transcription in README.md.

The reference is kept verbatim in README.md rather than as a sibling .py, because it is an
external artefact reproduced for provenance rather than this project's source (see that file).
This script extracts the fenced python block and executes it, so the transcription is exercised
on every regeneration and cannot drift into something that no longer runs.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))


def load_reference():
    """Return the reference `extended` from README.md's single fenced python block."""
    readme = open(os.path.join(HERE, "README.md")).read()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    if len(blocks) != 1:
        raise SystemExit(f"README.md must carry exactly one python block, found {len(blocks)}")
    ns = {}
    exec(compile(blocks[0], "README.md#reference", "exec"), ns)  # noqa: S102 - the provenance record
    return ns["extended"]


extended = load_reference()

DOMAINS = [
    (0.0, 100.0, 5), (0.0, 100.0, 3), (0.0, 100.0, 10),
    (0.0, 1.0, 5), (0.0, 10.0, 3), (0.0, 200.0, 5),
    (-50.0, 50.0, 6), (-100.0, -10.0, 5), (-1.0, 1.0, 5),
    (116.0, 180.0, 5), (2.3, 97.7, 4), (17.0, 83.0, 6),
    (0.0, 0.001, 4), (0.0, 1000000.0, 5), (1234.0, 5678.0, 5),
    (0.5, 0.7, 5), (99.0, 101.0, 5), (-0.05, 0.05, 5),
    (3.0, 3.14159, 4), (0.0, 7.0, 7), (1.0, 9.0, 4),
    (-273.15, 100.0, 6), (1e-6, 2e-6, 3), (0.0, 33.0, 4),
    # A sweep of m at one fixed range, so the density term is exercised across its whole
    # useful span rather than at one point, and a matching sweep of offsets at fixed m, so the
    # simplicity term's "includes zero" bonus is exercised both ways. Chosen structurally:
    # neither sweep was selected by checking which entries the weight-swap mutant fails.
    (0.0, 100.0, 2), (0.0, 100.0, 4), (0.0, 100.0, 6), (0.0, 100.0, 7),
    (0.0, 100.0, 8), (0.0, 100.0, 12),
    (1.0, 101.0, 5), (7.0, 107.0, 5), (13.0, 113.0, 5), (55.0, 155.0, 5),
    (-100.0, 0.0, 5), (-7.0, 93.0, 5),
]

rows = [{"dmin": a, "dmax": b, "m": m,
         "breaks": [float(f"{x:.12g}") for x in extended(a, b, m)]}
        for (a, b, m) in DOMAINS]

manifest = {
    "algorithm": "talbot-lin-hanrahan-2010-extended",
    "reference": "CRAN labeling::extended (Justin Talbot); transcribed literally in README.md, extracted and run by generate.py",
    "Q": [1, 5, 2, 2.5, 4, 3],
    "w": [0.25, 0.2, 0.5, 0.05],
    "w_order": ["simplicity", "coverage", "density", "legibility"],
    "erratum": (
        "The paper's prose (section 3.2) states 0.2*simplicity + 0.25*coverage, but BOTH of the "
        "first author's implementations -- the R labeling package and the C# ExtendedAxisLabeler "
        "-- use w = [0.25, 0.2, 0.5, 0.05] as w[0]*simplicity + w[1]*coverage. The implementations "
        "are normative and APEX-001 EV-2b's stated default agrees with them. That exact swap is "
        "this row's stated mutation, so these goldens are what detects it."
    ),
    "domains": rows,
}
path = os.path.join(os.environ["WT"], "crates/aprender-viz/fixtures/breaks/manifest.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(manifest, f, indent=2); f.write("\n")
print("domains:", len(rows))
for r in rows[:5]:
    print(f"  ({r['dmin']}, {r['dmax']}, m={r['m']}) -> {r['breaks']}")
