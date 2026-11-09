# The runner owns what it restores: no fixed uid and no silenced chown in ci/sections.yml; the self-test kills three planted mutants
bash scripts/check_chown_loud.sh --self-test
