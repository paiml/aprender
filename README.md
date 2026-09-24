# fleet-state

Data branch, not code. `fleet/cells.tsv` is the fleet's host × binary matrix,
written by `fleet-drift` (infra `machines/fleet-hosts/fleet-bins/fleet-drift.sh`,
aprender#4328 C3) on its 15-min timer, and read by the rc cut's
`scripts/release/fleet_cells_gate.sh`. Never edit by hand: the gate trusts
the `# measured` stamp.
