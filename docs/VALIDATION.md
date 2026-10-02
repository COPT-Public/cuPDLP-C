# cuPDLP-C validation — 2026-10-01

Build/install/test script exit: **0**. Source was built on the user-authorized
Linux validation server. The archive's 1,062 code/build/test files across the four
projects match the server copies by SHA-256; the corresponding per-project
files are included in this archive's SOURCE_MANIFEST.sha256.

Environment: Ubuntu 22.04.5 x86_64; GCC/G++ 11.4.0; CMake 3.30.5;
Python 3.10.12; NumPy 1.26.4. GPU runs: NVIDIA RTX 4090, device 0,
driver 580.82.07, CUDA Toolkit 12.8.93, architecture 89.
HDSDP uses oneMKL 2023.1.0; SOLNP_plus uses BLAS/LAPACK 3.10.0 and OSQP 0.6.3;
cuPDLP-C uses bundled HiGHS 1.6.0. Dependencies were installed in user prefixes.
HDSDP and SOLNP_plus were CPU runs. Optional interfaces and native Windows were
not tested. UTC timestamps on 2026-09-30 in raw logs are 2026-10-01 in Shanghai.

| Case | Objective | Primal residual | Result |
|---|---:|---:|---|
| cpu | 7.99999998913 | 1.081e-08 | PASS |
| gpu | 7.99999998913 | 1.081e-08 | PASS |

Additional checks: {"ctest_cpu": "4/4", "installed_suite_cpu": "4/4", "ctest_gpu": "4/4", "installed_suite_gpu": "4/4", "checker_negative_regressions": "2/2"}.
See validation/SUMMARY.json and build-and-tests.log for full evidence.
An initial installed-run attempt exposed a missing HiGHS runtime path. The final CMake installation adds the configured dependency paths; CPU/GPU installed runs now pass. CUDA deprecation warnings remain but do not fail these checks.

These are installation correctness tests, not performance benchmarks or proof
for every solver feature.
