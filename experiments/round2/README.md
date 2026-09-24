# 第二轮：先检验 FM 的作用机制

这是独立的机制实验目录。第一轮 `official_aligned/`、协议锁和已发布结果保持不变。
这里没有声称达到 80% OA，也没有新增注意力、伪标签 CE、教师或其他模块。

## 已有证据

诊断脚本与原始数据：`../../investigations/round2/diagnose.py`、`diagnostics.json`。
每个 seed 的 A/C source-val-best checkpoint 各使用四个固定 batch，CPU、train 模式。
每次探测前恢复 checkpoint 的 BN 状态，没有 optimizer step，没有写回权重。
只读取共享 ILDA 图像、已保存的采样位置和源标签；未重新读取目标 GT。
混淆归因另外使用第一轮已经公开的混淆矩阵。

| Seed | C 的位移路径/插值路径梯度范数之比* | FM 与 CE 平均梯度 cosine | A → C 源特征平均范数 |
|---|---:|---:|---:|
| 1341 | 32.5 | -0.137 | 10.78 → 8.37 |
| 1174 | 25.7 | -0.132 | 10.77 → 8.12 |
| 1370 | 58.0 | -0.050 | 10.82 → 8.72 |

*两条路径各自的 batch 平均梯度范数的比，不是逐 batch 比值的平均。
Flow 预测 MSE 相对零速度 MSE 仅改善约 3.4%、7.7%、2.7%。
这些是选中模型上的局部观测，不是完整训练轨迹，也不能单独证明退化原因。
A/C 的 selected epoch 不一定相同，因此范数对比不是等 epoch 因果对照。

第一轮 C 的主要稳定混淆变化为类2→类1、类7→类6。
类4虽有明显 recall 降幅，但实际只包含22个测试像素，不能当作 OA 下降主因。
三个 seed 中类2与类7合计造成的平均 OA 变化约 -2.44 个百分点，其他类别合计抵消约 +0.41 个百分点。
不据此设置类别专属权重或利用目标真值筛选训练样本。

## 最小实验矩阵

共用第一轮 Houston 数据、ILDA cache、采样/增强/BN、SGD、100 epochs、source-val-best 和测试口径。
FM 权重仍为1，OT及MLP参数不变，不进行超参数搜索。

| 方法 | 改动 | 要回答的问题 |
|---|---|---|
| A/B/C | 保留原实现作为复现入口 | 原始对照；正式比较可复用首轮已核验结果 |
| C_state_only | 仅将速度监督 `end-start` detach；插值状态保留源梯度 | 移除位移监督反传是否减轻源分类损害？ |
| OT_pull | 同一 OT 配对上直接最小化 `mean((end-start)^2)`；无 Flow 网络 | 现有 FM 是否主要表现为直接配对拉近？ |

`C_state_only` 与 C 在相同参数、特征、配对与 t 下数值 loss 完全相同，Flow 参数梯度也完全相同；仅编码器梯度路径不同。
`OT_pull` 是零速度对照，不是一个新的 flow matching 方法。
插值路径梯度原本较弱，C_state_only 可能只退回 A 附近；不能把恢复基线解释成 Flow 有效。

## 执行与验证

使用第一轮相同 Python 环境：

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/round2/check_objectives.py
```

梯度契约检查：原 C 与首轮数值/梯度/RNG 精确一致；state_only 不改 Flow 梯度；target/q 保持 detach；OT_pull 等价于配对 MSE；B 保持 detached。

短跑必须显式提供 `--audit`，LR horizon 固定100，不能用 final_test.py 测目标。

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/round2/train.py \
  --method C_state_only --seed 1341 --epochs 1 --audit --device cpu \
  --out experiments/round2/runs/example_smoke
```

正式预算为两个候选 × 三个原有 seed = 六次100-epoch训练。用户已授权启动，GPU 0/1 各一个 worker 排队执行。
正式命令省略 `--audit`，epochs和LR horizon均为100；程序校验原始输入和ILDA cache哈希。
所有候选全部训练结束后再使用各自 source-val-best 进行目标测评；报告全部候选和seed，不只保留赢家。
后台控制器 `run_formal.py` 在六组全部训练完成并通过步数/划分检查后，统一运行最终测评。
状态：`runs/formal/status.json`；进程记录：`launch_receipt.json`；任务日志：`runs/formal/train_*.log`。
成功完成后生成 `runs/formal/RESULTS.md`、`summary.json`，并归档到仓库 `results/round2/`。
若任一任务失败，会停止派发后续任务并记录错误；不会自动改变算法或缩短训练。

## 决策规则与研究方向

1. 两个候选先通过一轮真实 CPU 训练/验证 smoke；不计算目标准确率。
2. 在获准的算力范围内执行完整配对三 seed 比较；不在中间 epoch 计算目标准确率。
3. 若 state_only 仅恢复 A，结论是移除有害梯度有效，不能声称 Flow 已带来适应收益。
4. 若 OT_pull 与 C 接近，应把“Flow 是否必要”作为下一轮要解决的问题。
5. 性能目标是多seed平均OA超过80%，同时监控AA/Kappa/各类表现。不是选择某个超过80%的seed。
6. 若保留Flow路线，下一项可研究的改动是将运输得到的特征用于有源标签的分类训练，形成明确的分类目标；需要单独证明标签保持、与无Flow插值增强比较，不能直接堆入本轮。
7. 若优先性能，可另设完整MLUDA基线上的单一改动实验；不能把更换整个骨干/损失的增益归因于Flow。

Houston目标结果已经用于研究方向判断，后续实验属于同一已观察基准上的开发。
source-val-best并不能消除跨轮算法选择对已看过测试集的适应；额外seed也不是新的独立场景。
论文需要额外数据集/迁移任务的独立验证，并完整披露本轮负结果及所有候选。

## 文献定位（初步检索，不是完整新颖性审查）

- [FlowEO](https://arxiv.org/abs/2512.05140)：已有遥感图像空间的 flow-matching UDA；不能主张首次将Flow用于遥感域适应。
- [DisRFM](https://arxiv.org/abs/2602.00656)：已有图域适应中的结构保持与条件流匹配；需要辨别特征几何和语义保持层面的重叠。
- [Better Practices for Domain Adaptation](https://proceedings.mlr.press/v224/ericsson23a.html)：UDA的验证与模型选择会显著影响算法评价。

80%本身不是论文创新点或录用保证；应证明具体问题、机制、跨任务稳定增益与计算代价。
