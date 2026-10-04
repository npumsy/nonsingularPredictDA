# Changelog

## 2026-10-04

### 增广状态 DA：阻力参数可微（M3）
- 新增 `daJ234DragAugCoeffs`：把阻力缩放参数 $\kappa$ 升为第 7 个 DA 状态（$\kappa'=0$），用原生 DACE 在 7 变量、阶 2 上积分并导出 6 个状态的密集泰勒系数；积分前 `DA::setEps(0)`，避免微小阻力灵敏度被默认 `eps` 丢弃。
- `TBPfull` 的 `beta` 形参由 `double` 泛化为 `T`，并新增增广 RHS `TBPfull_param`（阻力从状态向量读参数）。
- pybind 暴露 `qoe.daJ234DragAugCoeffs(rv0, kappa0, tf, order, step) -> (rvf, coeffs, mons)`。
- 新增 `script/testDA_learning.py`：系数/中心敏感度对拍 STM 与有限差分、`torch.autograd.Function` 的 VJP 对拍有限差分、Theseus `IMPLICIT` 学习阻力参数。

### 多参数 / 位置相关残差力场（M5）
- 新增 `daAugCoeffs` 与增广 RHS `TBPfull_aug`：任意 $m$ 个乘性阻力参数（面质比、阻力系数、密度倍率）作为增广 DA 状态。
- 新增位置相关 RBF 引力异常场 `setRBFParams` + `TBPfull_rbf` + `daAugRBFCoeffs`：势 $U=\sum_k\theta_k e^{-|r-c_k|^2/(2s^2)}$、残差加速度 $-\nabla U$，$\theta$ 为可学习系数，中心与宽度固定。
- pybind 暴露 `qoe.daAugCoeffs(rv0, params, tf, order, step)` 与 `qoe.daAugRBFCoeffs(rv0, thetas, centers, s, tf, order, step)`。
- 新增 `script/testDA_augM.py`：用例 A（drag，乘性参数）与用例 B（rbf，位置相关力场）各验 VJP 对拍有限差分与 Theseus `IMPLICIT` 学习；命令行 `drag`/`rbf` 选择用例。
- `DACoeffs` 支持**批量输入**（前缀 batch 维）与**批量向量化反向**：移位恒等式按 batch 张量一次算，去除逐样本 Python 循环；批量前向/反向与逐样本结果逐位一致，`testDA_augM.py` 增 `check_batch`。
- 新增**低阶非带谐球谐异常场** `TBPfull_sh` + `daAugSHCoeffs`（笛卡尔实球谐表示，学 $C_{lm},S_{lm}$、$m\ge1$，带谐仍由 J234 显式），及数值自检 `shResidualAccel`（返回残差加速度供对拍 $\nabla U$）；`testDA_augM.py` 增 `sh` 用例与 `check_sh_physics`。
- 新增**路径 A：记录型标量 + 反向 tape** `include/RecordingScalar.h`（`DACE::RecordTape` + `DACE::Scalar`，重载算术/初等函数，thread-local 活跃 tape，反向为标准反向模式）；`daRecordFlowField` 把既有 RHS 以 `Scalar` 实例化、整条 RK4 自动记录，经 pybind 接 PyTorch（`daRecordFlowRBF`/`daRecordFlowSH`/`daRecordFlowBackward`）。不改 DA 内核、不把 DA 重写为 torch；梯度代价随图规模而非 $m$ 二项式增长。`testDA_augM.py` 增 `check_pathA`（`rbf` 与 `sh`）。
- 把 RBF 与球谐力场统一为**单一残差加速度接口** `fieldAccel<T>` + `TBPfull_field<T>`（由 `FieldSpec` 经 `setRBFParams`/`setSHParams` 设定种类），`double`/`DA`/记录标量三条路径共用；`TBPfull_rbf`/`TBPfull_sh`、`daAugRBFCoeffs`/`daAugSHCoeffs`、`daRecordFlowRBF`/`daRecordFlowSH` 均改为其薄封装。
- 新增**多历元 Theseus 因子学习**：`testDA_augM.py` 的 `_MultiDyn`/`_MultiObs` 把 $K$ 个历元并入单个动力学因子（优化变量 $X{=}[x_1,\dots,x_K]$、$\theta$ 为 aux），各历元用记录型标量算子 `DARecordFlow`（`tf` 可变）传播；`check_multiepoch`（`rbf`/`sh`）验 loss 下降与系数恢复。（重检查，改由 `full` 参数显式开启。）
- 新增**变分灵敏度多历元可微算子** `daVarMultiEpoch`（pybind `daVarMultiEpochRBF`/`daVarMultiEpochSH`）：DA **只作用于状态**（$N{=}6$）求 $A=\partial f/\partial x$，积分增广 $[x,\ \Phi{=}\partial x/\partial x_0,\ S{=}\partial x/\partial\theta]$，**$\theta$ 不进 DA**（对 $m$ 线性，$m{=}300$ 可用；而 $\theta$-in-DA 在 $m{=}100$ 触发 `daceInitialize ID:911`）。`DAFlowMulti` 的 backward 仅为矩阵乘（Gauss–Newton），无 tape、无逐历元 Python 循环；默认路径改为它（1/4 轨道周期下 rbf 11 ms/step、sh 20 ms/step）。
- 场景按应用改回 **1/4 轨道周期**（$T{=}\mathrm{PERIOD}/4{\approx}1437$ s、step=10、143 步），取代此前的 21600 s（3.8 圈，纯浪费）。

### 积分伴随：大参数情形的深度反向与 Theseus 参数学习
- 新增解析求力场基对状态的雅可比 `fieldBasisJacobian`（即状态方程对参数的混合二阶导，RBF 与球谐），以及残差加速度访问器 `fieldResidualAccel`；把 `fieldAccel` 改为基于单个基 `basisAccel<T>` 求和。
- 新增积分伴随反向算子 `daDeepForward`/`daDeepBackward`（Python 接口 `daDeepForwardRBF`/`daDeepForwardSH`、`daDeepBackwardRBF`/`daDeepBackwardSH` 与 `DeepFlow`）。做法：前向同时积分状态与状态转移矩阵，参数不放进 DA（变量数固定为 6）；前向记录每步 Runge–Kutta 的各个阶段，反向把这些阶段逐步取转置、从后往前累加，从而一次给出「含深度项 $\partial\Phi/\partial\theta$ 的 $\partial L/\partial\theta$」和「$\partial L/\partial x_0$」。交叉项用一次二阶状态 DA 求出。
- `testDA_augM.py` 新增 PyTorch 侧封装 `DAFlowDeep` 与 Theseus 因子 `_DynPhi`（决策量为初始状态、雅可比取状态转移矩阵 $\Phi(\theta)$），并新增以下检查（均进入默认路径）：T1 `check_deep_adjoint`（反向梯度对拍有限差分）、T5 `check_adjoint_identity`（伴随恒等式，检验反向确实是正向的伴随算子）、T6 `check_deep_necessary`（只依赖状态转移矩阵的损失下直接项恒为 0、深度项正确）、T7 `check_cross_method`（与「参数进 DA」的独立实现互相验证）、T2 `check_gn_deep`（高斯–牛顿解对参数的梯度，深度项与直接项对比）、T4 `check_gn_learn`（RBF、30 个参数端到端学习，损失降约 600 倍）、效果实验 `check_effect_deep`（直接项梯度误差随场强升到约 48%、深度项始终正确）、T3 `check_deep_scaling`（正向加反向耗时随参数个数近似线性）。原有只适用于小参数的 `check_learning` 移入 `full`。
- 效率：让 DA 只作用于与参数无关的 `TBPfull`（变量数固定为 6），RBF 力场的加速度及其一、二阶导、每个基的导数都用解析公式一次遍历求出（`rbfAllFieldDerivs`，复用缓冲），去掉了原先「逐个基做 DA」的高开销路径；在真实训练规模（四分之一轨道周期、143 步）下，30 个参数的正向加反向从 152 毫秒降到 26.6 毫秒（100 个参数 42 毫秒、300 个参数 89 毫秒，随参数个数近似线性）。
- 效果实验的构造、运行命令和期望输出写在 `src/test.md` 第七节；设计取舍记录在 `doc/dyn_para_learning.md` 第 3.7 节；结果汇总见 `src/experiment.md` 的 M5 一节。
