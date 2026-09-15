# 生成式离线多目标基线

本目录把 `2026-ICLR/model_generative` 中的三个独有方法接入
dual-ranking 的统一实验协议：DOMOO、PCD 和 ParetoFlow。这里是协议适配实现，
不是把原目录连同它各自的数据集、评估脚本和重复网络整棵复制进来。

## 对比结论

| 方面 | ICLR `model_generative` | dual-ranking 中的实现 |
| --- | --- | --- |
| 数据 | 每个方法自行加载、裁剪或归一化 | 统一使用 `src.official_pool` 的固定子集和 train/test 划分 |
| 代理 | DOMOO/ParetoFlow 各带一套多模型代理 | 复用 `experiments/DL_baseline` 的 `MultipleModels-Vallina` |
| ParetoFlow 网络 | 方法目录内另有 `FlowMatching`/`VectorFieldNet` | 复用 `external/offline-moo` 已存在的同名实现 |
| 真实函数 | 原脚本在各自评估阶段直接调用 | 只在最终候选已选定后，由统一评估器调用一次 |
| 指标 | 常用任务 min-max、`1.1` 参考点和 D-best | 统一使用论文 HV 参考点、official/full-pool 归一化与 IGD+ |
| 失败与复现 | 方法各自记录 | 配置哈希、子集哈希、失败行、独立 model/opt seed、断点续跑 |

三个方法保留的核心机制如下：

- **DOMOO**：逐目标代理、Langevin 负样本训练的能量置信度、风险抑制的
  Pareto-set learning，以及 PSL 候选和代理 NSGA-II 候选的联合筛选。
- **PCD**：Pareto 层级/目标密度重加权、目标条件 dropout、EDM
  预条件损失、classifier-free guidance 和向理想点外推的条件采样。
- **ParetoFlow**：无条件 flow matching、参考方向和后段代理梯度引导。

PCD 是直接条件生成器，没有额外目标代理，因此结果中的 `MSEpre`、
`MSEsur_real`、`HVsur` 和 `IGDplus_sur` 为 NaN；真实 HV/IGD+ 正常计算。

## 运行

依赖与现有 DL baseline 共用：

```bash
git submodule update --init external/offline-moo
python -m pip install -r experiments/generative_baseline/requirements.txt
```

Off-MOO 的任务数据文件也必须存在于它预期的 `data/<task>/` 目录；仅初始化
代码子模块而没有数据时，正式运行会明确报出缺失的 `.npy` 文件。

先检查完整实验计划（不会加载 PyTorch 或数据）：

```bash
python experiments/generative_baseline/run.py --dry-run
```

用极小训练设置验证三个方法的完整链路：

```bash
python experiments/generative_baseline/run.py \
  --methods DOMOO,PCD,ParetoFlow \
  --problems zdt1 \
  --training-sizes 50 \
  --offline-seeds 1 \
  --optimization-seeds 1 \
  --device cpu \
  --smoke
```

正式单任务示例：

```bash
python experiments/generative_baseline/run.py \
  --methods PCD \
  --problems zdt1 \
  --training-sizes 1000 \
  --offline-seeds 1 \
  --optimization-seeds 1
```

默认配置在 `config.yaml`。结果写入
`results/generative_baselines.csv`，最终候选写入
`results/candidates/<method>/*.npz`。成功行默认断点续跑；使用
`--no-resume` 可以强制重跑。

## 实现边界

当前 benchmark 的 30 个任务都是本目录支持的定长连续决策表示；portfolio
会继续经过仓库现有 repair。没有搬入 ICLR 目录中的 wandb、gin、任务副本、
画图脚本、结果后处理、代理副本和 ParetoFlow 网络副本。这样可避免同一协议里
出现两份定义逐渐漂移。
