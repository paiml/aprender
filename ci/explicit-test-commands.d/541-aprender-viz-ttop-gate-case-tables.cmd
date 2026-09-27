# #4511: the ttop RSS-soak and --timings gate classifiers (case tables only; the soak run and the planted-leak/planted-sleep mutants are release-time, never a required PR check).
bash scripts/ttop_soak_gate.sh --self-test && bash scripts/ttop_timings_gate.sh --self-test
