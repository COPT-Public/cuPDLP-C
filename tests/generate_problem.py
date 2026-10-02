from pathlib import Path
DATA={'tiny_lp.mps': 'NAME          TINYLP\nROWS\n N  COST\n G  R1\n G  R2\nCOLUMNS\n    X1        COST      2             R1        1\n    X1        R2        2\n    X2        COST      3             R1        2\n    X2        R2        1\nRHS\n    RHS1      R1        5             R2        4\nBOUNDS\n LO BND1      X1        0\n LO BND1      X2        0\nENDATA\n', 'expected.json': '{\n  "x": [\n    1,\n    2\n  ],\n  "objective": 8,\n  "solution_tolerance": 0.0001,\n  "residual_tolerance": 1e-05\n}'}
root=Path(__file__).resolve().parents[1]/"examples"
root.mkdir(exist_ok=True)
for name,text in DATA.items(): (root/name).write_text(text,encoding="utf-8")
print("Generated analytical problems in",root)
