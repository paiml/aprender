#!/usr/bin/env python3
"""Dump a GitHub workflow's push/pull_request `paths:` filters, one per line.

Lives in its own file rather than a heredoc inside
scripts/check_workflow_path_filters.sh: bashrs parses an embedded Python program
as shell and reported 9 phantom errors (SC1007 against Python assignments,
SC1078 against a multi-line string), which would bury a real one. Same reason
scripts/lib/assertions_exclude.awk was extracted.

Exit 3 means the file could not be parsed as YAML - the caller must treat that
as a hard failure, never as "no filters".
"""
import sys, yaml
try:
    d = yaml.safe_load(open(sys.argv[1]))
except Exception as e:
    print(f"PARSE_ERROR {e}", file=sys.stderr); sys.exit(3)
if not isinstance(d, dict):
    sys.exit(0)
on = d.get(True) if d.get(True) is not None else d.get('on')
if not isinstance(on, dict):
    sys.exit(0)
for ev, tag in (('push', 'PUSH'), ('pull_request', 'PR')):
    spec = on.get(ev)
    if not isinstance(spec, dict):
        continue
    paths = spec.get('paths')
    if paths is None:
        print(f"{tag}\t<unfiltered>")
        continue
    for p in paths:
        print(f"{tag}\t{p}")
