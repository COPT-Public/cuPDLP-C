# Installation and checker tests

Run `bash build.sh` after following INSTALL. All tests use bundled small models
with analytic solutions and return process exit 0 only if accepted.

Four LP cases: bounded=-10; equality=3; mixed_bounds=-1; compressed_mps=-10.
Independent bound/row feasibility <=1e-6; objective error <=5e-4; reported
relative primal/dual feasibility and gap <=5e-6.
Extra tiny_lp.mps: x=(1,2), objective 8; independent primal/dual feasibility
<=1e-6, objective/gap/coordinate errors <=1e-4, OPTIMAL termination.

`run.py` uses NumPy to independently recompute the specified feasibility and
objective conditions from returned solution data. Every invocation creates a
new run-* output folder. The parent verification.json is reset before execution
and replaced with the result; no earlier solution is read. Missing output,
timeout, nonzero process exit, nonfinite values or failed checks cause failure.
The run folder contains command.json, solver.log and raw solver outputs.

`python3 tests/check_checker.py --project cuPDLP-C` verifies two failure paths:
old correct results plus a no-op process must fail; newly generated incorrect
solutions must also fail. The real build.sh solve is the positive control.

To add a case:
1. Add a small model with an analytic expected answer to examples/ and describe
   its mathematics/expected answer in examples/README.md.
2. Extend generate_problem.py if the model is generated. In run.py add an explicit
   --case choice and the corresponding model, expected optimum and residual checks.
3. Add its command to build.sh (and add_test in the relevant CMakeLists if using
   CTest). Use the installed executable/consumer where applicable.
4. Confirm a real solve passes and deliberately bad output fails; record tolerances.
For cuPDLP-C's four-case suite, add a model in tests/data/ and an entry in
tests/cases.json, then register it in tests/CMakeLists.txt. Compressed input is
generated temporarily; no external benchmark downloads are required.

These tests do not benchmark performance or validate optional language interfaces.
