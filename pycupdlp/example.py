import pycupdlp
import numpy as np
import scipy as sp

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

c = pycupdlp.cupdlp()
A = sp.sparse.coo_matrix(
    ([-1.5, -1.2, -1.8, 0.8, 0.6, 0.9], ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]))
)
cost = np.array([1.2, 1.8, 2.1])
rhs = np.array([-2.6, 1.2])
lb = np.array([0.1, 0.2, 0.3])
ub = np.array([0.6, 1.5, 2.8])
neqs = 0
c.loadData(A, cost, rhs, lb, ub, neqs)
c.solve()
res = c.getSolution()
print(res)
