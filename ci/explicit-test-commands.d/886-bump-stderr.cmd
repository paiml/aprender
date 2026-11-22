# bump-version.sh surfaces cargo stderr when the lockfile refresh fails (falsifier + pre-fix mutant)
bash scripts/release/falsify_bump_stderr.sh && bash scripts/release/falsify_bump_stderr.sh --mutants
