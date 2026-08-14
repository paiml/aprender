"""List every include!() target in a crate, resolved against the including file.

Separate file rather than an inline heredoc: bashrs parses an embedded heredoc
as shell, so python assignments read as SC1007 "space after =" -- eight phantom
errors. Same reason assertions_exclude.awk and workflow_path_filters.py live
here.

argv: <crate-dir>
stdout: "<crate-relative target>\t<crate-relative including file>" per line
"""
import os
import re
import sys

crate = sys.argv[1]
src = os.path.join(crate, "src")
pat = re.compile(r'include!\s*\(\s*"([^"]+)"\s*\)')

for root, _dirs, files in os.walk(src):
    for fn in files:
        if not fn.endswith(".rs"):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for m in pat.finditer(text):
            target = os.path.normpath(os.path.join(root, m.group(1)))
            print(f"{os.path.relpath(target, crate)}\t{os.path.relpath(path, crate)}")
