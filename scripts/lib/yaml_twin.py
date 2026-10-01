#!/usr/bin/env python3
"""yaml_twin.py -- read a YAML file where PyYAML exists, and its JSON TWIN where it does not (#3731).

WHY. The parity and bench block helpers (bench_receipt.py, perf_receipt.py) read
scripts/perf-matrix.yaml and scripts/perf-receipt-fields.yaml. mini's only python3 is
CommandLineTools' 3.9.6 with no PyYAML, so parity refused there ("No module named 'yaml'",
2026-09-21), and installing into that python is an undeclared host change. The host-receipt KIT is
built on the train host, which has PyYAML, so the kit carries a JSON twin beside each YAML
(`<name>.json`), written by this file. A host that cannot `import yaml` loads the twin with the
standard library. Twins exist ONLY in kits: they are never committed, so they cannot drift from the
YAML they copy. Both files hold only JSON types (dict, list, str, int, float, bool, None; str keys),
so the twin is the same object, and the case table proves it after a canonical dump.

    python3 scripts/lib/yaml_twin.py write A.yaml [B.yaml ...]   # writes A.json beside each
    load(path)   # the parsed YAML; or, when `import yaml` fails, the twin beside it; or ImportError
"""
import json
import os
import sys


def twin_path(path):
    return os.path.splitext(path)[0] + ".json"


def load(path):
    try:
        import yaml
    except ImportError:
        twin = twin_path(path)
        if not os.path.exists(twin):
            raise ImportError(
                f"PyYAML is not importable and there is no JSON twin {twin} beside {path}; "
                "a host-receipt kit carries one (scripts/lib/yaml_twin.py write), a checkout does not")
        with open(twin, encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write(paths):
    import yaml
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        with open(twin_path(path), "w", encoding="utf-8") as handle:
            json.dump(doc, handle, sort_keys=True, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "write":
        sys.stderr.write("usage: yaml_twin.py write A.yaml [B.yaml ...]\n")
        sys.exit(2)
    write(sys.argv[2:])
