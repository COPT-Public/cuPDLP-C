# cuPDLP-C

Code for solving LP on GPU using first-order methods. 

This is the C implementation of the Julia version [cuPDLP.jl](https://github.com/jinwen-yang/cuPDLP.jl).

## Compile
We use CMAKE to build CUPDLP. The current version is built on the [Coin-OR CLP project](https://github.com/coin-or/Clp). Please install the dependencies therein.

Once you setup CLP and CUDA, set the following environment variables.

```shell
export CLP_HOME=/path-to-clp
export COIN_HOME=/path-to-coinutils
export CUDA_HOME=/path-to-cuda
```

You can build the project with CUDA by setting `-DBUILD_CUDA=ON` (by default OFF, i.e., the CPU version):

- in the debug mode:
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_CUDA=ON ..
cmake --build . --target plc
```
then you can find the binary `plc` in the folder `build/bin/`.

When using the release mode, we suggest the following options,
```
cmake -DBUILD_CUDA=ON \
-DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
-DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" \
-DCMAKE_CUDA_FLAGS_RELEASE="-O2 -DNDEBUG" ..
```  

## Usage

Usage example: set `nIterLim` to `5000` and solve.

```shell
./bin/plc -fname <mpsfile> -nIterLim 5000
```

| Param | Type | Range | Default | Description |
|:---:|:---:|:---:|:---:|:---:|
|`fname`|`str`|` `|` `|`.mps` or `.mps.gz` file of the LP instance|
|`fout`|`str`|` `|`./solution.json`|`.json` file to save result|
|`savesol`|`bool`|`true, false`|`false`|whether to write solution to `.json` output|
|`ifScaling`|`bool`|`true, false`|`true`|Whether to use scaling|
|`ifRuizScaling`|`bool`|`true, false`|`true`|Whether to use Ruiz scaling (10 times)|
|`ifL2Scaling`|`bool`|`true, false`|`false`|Whether to use L2 scaling|
|`ifPcScaling`|`bool`|`true, false`|`true`|Whether to use Pock-Chambolle scaling|
|`nIterLim`|`int`|`>=0`|`10000000`|Maximum iteration number|
|`eLineSearchMethod`|`int`|`0-2`|`2`|Choose line search: 0-fixed, 1-Malitsky, 2-Adaptive|
|`dPrimalTol`|`double`|`>=0`|`1e-4`|Primal feasibility tolerance for termination|
|`dDualTol`|`double`|`>=0`|`1e-4`|Dual feasibility tolerance for termination|
|`dGapTol`|`double`|`>=0`|`1e-4`|Duality gap tolerance for termination|
|`dFeasTol`|`double`|`>=0`|`1e-8`|Not used yet, maybe infeasibility tolerance|
|`dTimeLim`|`double`|`>0`|`3600`|Time limit (in seconds)|
|`eRestartMethod`|`int`|`0-1`|`1`|Choose restart: 0-none, 1-GPU|
<!-- |`dScalingLimit`|`double`|`>0`|`1`|Maybe to control scaling magnitude| -->
<!-- |`iScalingMethod`|`int`|`0-5`|`0`|Which scaling to use: 0-Column, 1-Row, 2-Col&Row, 3-Ruiz, 4-Col&Row&Obj, 5-Ruiz| -->
<!-- |``|``|``|``|| -->





## The PDLP Algorithm
Consider the generic linear programming problem:

$$
\begin{aligned}
\min\ & c^\top x \\
\text{s.t.}\ & A x = b \\
& Gx \geq h \\
& l \leq x \leq u
\end{aligned}
$$

Equivalently, we solve the following saddle-point problem,

$$
\max_{y_1\,\text{free}, y_2\geq 0}\min_{l\leq x\leq u}c^\top x - y^\top Kx + q^\top y
$$

where dual variables $y^\top=(y_1^\top, y_2^\top)$, $K^\top = (A^\top, G^\top)$, $q^\top=(b^\top, h^\top)$.

Primal-Dual Hybrid Gradient (PDHG) algorithm takes the step as follows,

$$
\begin{aligned}
x^{k+1} &= \Pi_{l\leq x\leq u} (x^k - \tau (c - K^\top y^k)) \\
y^{k+1} &= \Pi_{y_2\geq 0} (y^k + \sigma (q - K(2x^{k+1} - x^k)))
\end{aligned}
$$

The termination criteria contain the primal feasibility, dual feasibility, and duality gap.

$$
\begin{aligned}
\left\\| \begin{matrix} Ax-b \\\\ (h - Gx)^+ \end{matrix} \right\\| &\leq \epsilon (1 + \\|q\\|) \\
\\|c - K^\top y - \lambda\\|&\leq \epsilon(1 + \\|c\\|) \\
|q^\top y + l^\top \lambda^+ - u^\top \lambda^- - c^\top x| &\leq \epsilon(1 + |c^\top x| + |q^\top y + l^\top \lambda^+ - u^\top \lambda^-|)
\end{aligned}
$$

where $\lambda = \Pi_\Lambda(c - K^\top y)$

$$
\lambda_i \begin{cases} = 0 & l_i=-\infty, u_i=+\infty\\
\leq 0 & l_i=-\infty, u_i<+\infty\\ 
\geq 0 & l_i<-\infty, u_i=+\infty\\ 
\text{free} & l_i>-\infty, u_i<+\infty  \end{cases}.$$

$\\|\cdot\\|$ is 2-norm, and $|\cdot|$ is absolute value.

## Authors

Dongdong Ge, Haodong Hu, Qi Huangfu, Jinsong Liu, Tianhao Liu, Haihao Lu, Jinwen Yang, Yinyu Ye, Chuwen Zhang

### Contact
- Jinsong Liu  <github.com/JinsongLiu6>
- Tianhao Liu  <github.com/SkyLiu0>
- Chuwen Zhang <github.com/bzhangcw>
