#!/usr/bin/env python3
"""Check an installed or build-tree cuPDLP-C solver; Python standard library only."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FEAS_TOL = 1e-6
OBJ_TOL = 5e-4


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite(value, name):
    require(isinstance(value, (int, float)) and math.isfinite(value),
            f"{name} is missing, non-numeric, or non-finite: {value!r}")
    return value


def check_case(solver, name, cases):
    compressed = name == 'compressed_mps'
    case = cases['bounded' if compressed else name]
    with tempfile.TemporaryDirectory(prefix='cupdlp-test-') as tmp:
        tmp = Path(tmp)
        model = HERE / 'data' / case['mps']
        if compressed:
            compressed_model = tmp / (model.name + '.gz')
            with gzip.open(compressed_model, 'wb') as f:
                f.write(model.read_bytes())
            model = compressed_model
        summary_path, solution_path = tmp/'summary.json', tmp/'solution.json'
        command = [str(solver), '-fname', str(model), '-out', str(summary_path),
                   '-outSol', str(solution_path), '-savesol', '1', '-ifPre', '0',
                   '-nIterLim', '20000', '-dTimeLim', '30',
                   '-dPrimalTol', '1e-7', '-dDualTol', '1e-7', '-dGapTol', '1e-7']
        result = subprocess.run(command, cwd=tmp, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=45)
        try:
            require(result.returncode == 0, f"solver exit code {result.returncode}")
            summary = json.loads(summary_path.read_text())
            solution = json.loads(solution_path.read_text())
            require(summary.get('terminationCode') == 'OPTIMAL',
                    f"terminationCode={summary.get('terminationCode')!r}")
            require(solution.get('nCols') == len(case['cost']), 'wrong column count')
            require(solution.get('nRows') == len(case['rows']), 'wrong row count')
            x = solution['col_value']
            require(len(x) == len(case['cost']), 'wrong solution length')
            for i, xi in enumerate(x):
                finite(xi, f'x[{i}]')
                lower, upper = case['bounds'][i]
                require(lower is None or xi >= lower-FEAS_TOL, f'x[{i}] violates lower bound')
                require(upper is None or xi <= upper+FEAS_TOL, f'x[{i}] violates upper bound')
            for row in case['rows']:
                activity = sum(a*xi for a, xi in zip(row['coefficients'], x))
                rhs, sense = row['rhs'], row['sense']
                residual = {'L': activity-rhs, 'G': rhs-activity, 'E': abs(activity-rhs)}[sense]
                require(residual <= FEAS_TOL, f'row {row} violated by {residual}')
            objective = sum(c*xi for c, xi in zip(case['cost'], x))
            require(abs(objective-case['objective']) <= OBJ_TOL,
                    f'objective {objective} != expected {case["objective"]}')
            iterate = summary.get('terminationIterate')
            require(iterate in ('LAST_ITERATE', 'AVERAGE_ITERATE'), 'unknown termination iterate')
            suffix = 'Average' if iterate == 'AVERAGE_ITERATE' else ''
            for key in ['dPrimalObj'+suffix, 'dDualObj'+suffix]:
                value = finite(summary.get(key), key)
                require(abs(value-case['objective']) <= OBJ_TOL, f'{key}={value} is inaccurate')
            for key in ['dRelPrimalFeas', 'dRelDualFeas', 'dRelDualityGap']:
                value = finite(summary.get(key), key)
                require(0 <= value <= 5e-6, f'{key}={value} is outside tolerance')
            print(f'PASS {name}: objective={objective:.10g}; expected={case["objective"]}; OPTIMAL')
        except (AssertionError, OSError, ValueError, KeyError, TypeError):
            print('COMMAND:', ' '.join(command), file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            raise


def main():
    cases = json.loads((HERE/'cases.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', required=True, type=Path, help='Path to plc executable')
    parser.add_argument('--case', choices=list(cases)+['compressed_mps'], help='Run only this case')
    args = parser.parse_args()
    solver = args.solver.expanduser().resolve()
    if not solver.is_file():
        print(f'FAIL: solver not found: {solver}', file=sys.stderr)
        return 1
    names = [args.case] if args.case else list(cases)+['compressed_mps']
    failures = 0
    for name in names:
        try:
            check_case(solver, name, cases)
        except (AssertionError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            failures += 1
            print(f'FAIL {name}: {error}', file=sys.stderr)
    print(f'{len(names)-failures}/{len(names)} installation tests passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
