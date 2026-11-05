# R34-3: check_release_models_t1.sh builds its fixtures under $RUNNER_TEMP, not bare /tmp; the mutant (bare mktemp -d) must be caught
bash scripts/release/falsify_t1_tmp_root.sh --self-test && bash scripts/release/falsify_t1_tmp_root.sh --mutants
