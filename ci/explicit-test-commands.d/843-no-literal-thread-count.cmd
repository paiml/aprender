# Thread caps come from RUNNER_THREAD_CAP: no workflow or section carries a literal CARGO_BUILD_JOBS / RAYON_NUM_THREADS / NEXTEST_TEST_THREADS / RUST_TEST_THREADS count (#6151); the self-test plants each literal form and must turn RED.
bash scripts/check_no_literal_thread_count.sh
