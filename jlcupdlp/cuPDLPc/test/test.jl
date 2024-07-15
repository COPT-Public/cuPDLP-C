using cuPDLPc
using SparseArrays

"""
The problem to solve:

  Maximize:
    1.2 x + 1.8 y + 2.1 z

  Subject to:
    1.5 x + 1.2 y + 1.8 z <= 2.6
    0.8 x + 0.6 y + 0.9 z >= 1.2

  where:
    0.1 <= x <= 0.6
    0.2 <= y <= 1.5
    0.3 <= z <= 2.8
"""

cost = [1.2, 1.8, 2.1]
A = sparse(
  [1.5 1.2 1.8;
    0.8 0.6 0.9]
)
lhs = [-Inf, 1.2]
rhs = [2.6, Inf]
l = [0.1, 0.2, 0.3]
u = [0.6, 1.5, 2.8]
sense = -1
offset = 0.0

solver = cuPDLP_C()
solver.load_lp!(cost, A, lhs, rhs, l, u, sense, offset)
solver.setParam!("dPrimalTol", 1e-6)
solver.setParam!("dDualTol", 1e-6)
solver.setParam!("dGapTol", 1e-6)
# solver.help()
solver.solve!()
