"""Print "<package name>\t<crate dir>" for every publishable workspace crate.

Reads `cargo metadata --no-deps --format-version 1` on stdin. Separate file for
the same reason as its neighbours: inline python inside a shell script is parsed
as shell by bashrs (`m = json.load(...)` reads as SC1078, an unterminated string).
"""
import json
import os
import sys

meta = json.load(sys.stdin)
for pkg in meta["packages"]:
    if pkg.get("publish") == []:
        continue
    print(pkg["name"] + "\t" + os.path.dirname(pkg["manifest_path"]))
