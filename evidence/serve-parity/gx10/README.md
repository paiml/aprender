# serve-parity receipts: gx10 (GB10, aarch64, sm_121)

aprender#4218, contract `contracts/serve-parity-gate-v1.yaml`.

- `<green_sha>.json`: one receipt per measured commit, written by
  `python3 scripts/lib/serve_parity_gate.py verdict <run.json> --write-receipt`.
  The release decision surfaces read it through `scripts/check_serve_parity_receipt.sh`.
- `baseline.json`: the ratchet. It does not exist yet. It is created only by
  `serve_parity_gate.py promote` from a run that passed every check, including a
  positive control that went RED on the same harness. Until then every gx10 run is
  RED ("no committed baseline"), and that verdict is honest.
