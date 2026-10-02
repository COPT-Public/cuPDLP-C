cuPDLP-C preparation changes — 2026-10-01-r2

Uses the earlier prepared package build/install/four-case framework, on the
same 7b94c41 upstream numerical source as the newer package. Bundles the newer
package HiGHS source and analytic tiny LP check. Adds CPU/GPU/all build modes,
installed testing, strict residual checks and false-positive regressions.
Preserves configurable CUDA architecture/includes and adds sparse license notices.
Explicit installed runtime search paths locate bundled-build HiGHS and CUDA.
The installed test runner excludes generated test logs from installation.

Declared project-owned copyright holder: Dongdong Ge, per submitting team.
All existing upstream/third-party attribution retained. No forms signed or sent.
