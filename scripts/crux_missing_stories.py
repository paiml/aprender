#!/usr/bin/env python3
"""Read JSON stories on stdin, emit TSV (id,title,demand,competitor) for status==missing."""
import json
import sys

stories = json.load(sys.stdin)
for s in stories:
    if s.get("status") == "missing":
        print("\t".join([s["id"], s["title"], str(s["demand_score"]), s["competitor"]]))
