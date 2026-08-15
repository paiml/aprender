"""Print "<package name>\t<crate dir>" for every publishable workspace crate.

Reads `cargo metadata --no-deps --format-version 1` on stdin. A separate file
because bashrs parses inline python inside a shell script as shell.
"""
import json
import os
import sys

meta = json.load(sys.stdin)
for pkg in meta["packages"]:
    if pkg.get("publish") == []:
        continue
    print(pkg["name"] + "\t" + os.path.dirname(pkg["manifest_path"]))
