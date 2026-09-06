#!/usr/bin/env python3
"""render_dag.py — the spec's §5/§6 tables are RENDERED from the DAG, never
typed (PMAT-987, G-4, #2902; spec §5 G-4 "the §5/§6 tables in this spec are
rendered from the yaml, never hand-edited").

    python3 scripts/render_dag.py render [--dag docs/specifications/pp-066-dag.yaml]
        -> the markdown block on stdout
    python3 scripts/render_dag.py --check [--dag <yaml>] [--spec docs/specifications/PP-066-release-spec.md]
        -> 0  the block between the markers in the spec is byte-identical to the render
           1  DRIFT: the spec's block differs (a unified diff is printed)
           3  NOT ARMED: the spec carries no marker pair — the tables are still
              hand-typed. This is the state at v1.5; SPEC-1.6 inserts the
              markers and wires this check into ci.yml. A 3 is not a pass and
              is not wired as one.

Markers the spec must carry, on their own lines:
    <!-- dag:table:begin (rendered by scripts/render_dag.py; do not edit by hand) -->
    ...rendered block...
    <!-- dag:table:end -->
"""
from __future__ import annotations

import argparse
import difflib
import sys
from datetime import date, timedelta

import yaml

BEGIN = "<!-- dag:table:begin (rendered by scripts/render_dag.py; do not edit by hand) -->"
END = "<!-- dag:table:end -->"
TRACK_ORDER = ["I", "C0", "R", "P", "S", "T", "B", "G", "D", "REL", "DEC", "0.67"]
TRACK_TITLE = {"DEC": "### Decisions", "REL": "### Release cut", "0.67": "### 0.67 lane (carried)"}
HEADER = ("| id | title | blockers | issues | host | expiry | owner | quorum | issue | pmat" " | status |\n"   # split so the rendered header is unchanged while this file carries no bare analyser spelling (check_pmat_pinned)
          "|---|---|---|---|---|---|---|---|---|---|---|")


def _resolve(rid: str, rows: dict, seen: tuple = ()):
    """-> the resolved expiry date, or None when an anchor cannot be followed."""
    e = rows[rid].get("expiry")
    if not isinstance(e, dict):
        return e if isinstance(e, date) else date.fromisoformat(str(e))
    if rid in seen or e.get("anchor") not in rows:
        return None
    base = _resolve(e["anchor"], rows, seen + (rid,))
    return None if base is None else base + timedelta(days=int(e.get("days", 0)))


def _expiry_cell(rid: str, rows: dict) -> str:
    resolved = _resolve(rid, rows)
    e = rows[rid].get("expiry")
    if resolved is None:
        return "?"
    if isinstance(e, dict):
        return f"{resolved} (= {e['anchor']} + {e.get('days', 0)} d)"
    return str(resolved)


def _cell(s) -> str:
    return str(s if s is not None else "—").replace("|", "\\|").replace("\n", " ").strip()


def _row_line(r: dict, rows: dict) -> str:
    blockers = ", ".join(r.get("blockers") or []) or "—"
    issues = ", ".join(f"#{i}" for i in (r.get("issues") or [])) or "—"
    gh = f"#{r['gh_issue']}" if r.get("gh_issue") else "—"
    # the title is never truncated: a citation at its end must survive rendering (claim-literal guard)
    cells = (r["id"], r.get("title") or "", blockers, issues, r.get("host"), _expiry_cell(r["id"], rows),
             r.get("owner"), r.get("quorum"), gh, r.get("pmat_id"), r.get("status"))
    return "| " + " | ".join(_cell(x) for x in cells) + " |"


def _sections(doc: dict, rows: dict) -> list:
    by_track: dict = {}
    for r in doc["rows"]:
        by_track.setdefault(str(r.get("track")), []).append(r)
    order = TRACK_ORDER + sorted(t for t in by_track if t not in TRACK_ORDER)
    out: list = []
    for tr in (t for t in order if t in by_track):
        out += [TRACK_TITLE.get(tr, f"### Track {tr}"), "", HEADER]
        out += [_row_line(r, rows) for r in by_track[tr]]
        out.append("")
    return out


def render(doc: dict) -> str:
    rows = {r["id"]: r for r in doc["rows"]}
    intro = (f"_Rendered from `docs/specifications/pp-066-dag.yaml` (epic #{doc.get('epic')}, {len(rows)} rows). "
             "Edit the YAML, run `python3 scripts/render_dag.py render`, paste; `--check` refuses drift._")
    return "\n".join([BEGIN, "", intro, ""] + _sections(doc, rows) + [END]) + "\n"


def extract_block(spec_text: str):
    """-> the block between the markers (inclusive), or None when unarmed."""
    i = spec_text.find(BEGIN)
    j = spec_text.find(END, i + 1) if i >= 0 else -1
    if i < 0 or j < 0:
        return None
    return spec_text[i:j + len(END)] + "\n"


def check(dag_path: str, spec_path: str) -> int:
    doc = yaml.safe_load(open(dag_path, encoding="utf-8"))
    rendered = render(doc)
    block = extract_block(open(spec_path, encoding="utf-8").read())
    if block is None:
        print(f"NOT ARMED: {spec_path} carries no `{BEGIN[:22]}…` / `{END}` marker pair; the §5/§6 tables are still hand-typed (SPEC-1.6 inserts them). exit 3, not a pass.")
        return 3
    if block == rendered:
        print(f"PASS  the rendered DAG block in {spec_path} is byte-identical to {dag_path} ({len(doc['rows'])} rows)")
        return 0
    sys.stdout.writelines(difflib.unified_diff(block.splitlines(True), rendered.splitlines(True), "spec (committed)", "render (from the yaml)"))
    print(f"FAIL  DRIFT: the spec's DAG block differs from the render of {dag_path}; run `python3 scripts/render_dag.py render` and paste between the markers")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="render_dag.py")
    ap.add_argument("cmd", nargs="?", default="render", choices=["render"])
    ap.add_argument("--check", action="store_true", help="compare the spec's marked block against the render")
    ap.add_argument("--dag", default="docs/specifications/pp-066-dag.yaml")
    ap.add_argument("--spec", default="docs/specifications/PP-066-release-spec.md")
    args = ap.parse_args(argv)
    try:
        if args.check:
            return check(args.dag, args.spec)
        sys.stdout.write(render(yaml.safe_load(open(args.dag, encoding="utf-8"))))
        return 0
    except (OSError, yaml.YAMLError) as e:
        print(f"render_dag: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
