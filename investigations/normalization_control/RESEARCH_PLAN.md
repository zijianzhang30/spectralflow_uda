# BN 与无目标标签选模：调研计划及下一轮协议建议

日期：2026-09-23。本文是调研与预登记草案，不是已经生效的新训练协议。
本次核对了官方 Houston 入口、本地控制实验记录，以及下列论文的原始摘要页面；没有宣称已逐篇完成全文和官方实现复现。
没有启动新训练、评估新的 target 成绩、修改已有协议锁或覆盖历史结果。

## 1. 先回答：MLUDA 官方到底怎样选 epoch？

证据来自归档的 `official_aligned/reference/`，对应已核验的官方 Houston 发布入口：

| 位置 | 实际代码/行为 | 结论 |
|---|---|---|
| `config_Houston.py:4` | `epochs = 100` | 固定训练预算 100 轮 |
| `MLUDA_hu.py:171` | `if epoch % epochs == 0` | 默认循环只在第 100 轮进入测试 |
| `MLUDA_hu.py:200` | target loader 上的 `test_accuracy` | 此处计算的是 target OA |
| `MLUDA_hu.py:213` | `if test_accuracy > last_accuracy` | 默认只有一次候选评估，不是从 100 轮中寻找 target-best |
| `MLUDA_hu.py:215` | `torch.save(...)` 被注释 | 发布入口没有实际保存该 checkpoint |
| `utils.py:202` | 验证集返回值被注释 | 没有生效的 source-val-best 选模逻辑 |

因此，默认发布代码实际报告 **最后第 100 轮**，不是 source-val-best，也不是逐 epoch target-best。
打印语句使用 `epoch + 1`，可能显示 101；不能把打印数字当作实际训练轮数。
结论限定于核验的发布代码，不据此反推论文所有实验或未公开实现。
代码身份记录见 `reference/official_selection_verification.json`。

第一轮及第二轮已采用的 source-val-best 结果继续保留原定义。下一轮若换规则，应作为新协议，不能把历史表悄悄换成另一个更高的数。

## 2. 当前证据能说明什么？

- `checks.json` 证明 A/C 在 CPU、seed1341、38 个真实数据更新中，BN 路由前后的训练 logits、损失、梯度、可学习参数和 Flow RNG 一致，且两套 buffer 可正确保存/恢复。尚未证明 GPU、多 seed、100 epoch 的一致性或 target 收益。
- `../bn_augmentation/REPORT.md` 显示源域原图统计量校准后，A 的源验证接近 100%；它是既有 checkpoint 的校准诊断，不是分域 BN 全训练曲线已经饱和的证明。
- 1270 个源验证中心都位于至少一个训练 patch 内；中心标签划分没有交集，但图像区域高度重叠。source-val 高分不能直接解释为跨场景泛化良好。
- 当前原型同时做了“源/目标 buffer 分开”和“增强视图不提交统计量”。这两个因素必须区分，不能把所有效果叫作单纯分域 BN 的收益。
- 官方 flip 对 `[N,C,H,W]` 使用 `fliplr/flipud`，分别翻 channel/batch。完整 MLUDA 中 batch 倒置会破坏对比视图的样本身份对应；修正应单列，不与 BN 改动捆绑归因。

## 3. 建议的选模规则

**下一轮建议：固定 epoch100 为主结果，source-val-best 为预先登记的次要结果。**

理由：与官方发布入口的固定预算一致；不需要 target 真值；避免源验证接近饱和时由少数预测变化或很早的并列最高分决定主结果。这不是宣称 epoch100 的 target 准确率最高，也不能消除终期过拟合或跨方法收敛速度差异。

| 候选规则 | 使用哪些标签 | 当前建议 | 需要承认的局限 |
|---|---|---|---|
| 固定 epoch100 | 不用标签选轮次 | 下一轮主口径 | 固定预算公平，但不保证各模型都处在各自最优点 |
| 最大 source-val OA，精确并列取最早 | 仅源验证标签 | 保留为完整次要表，与历史规则相同 | 源/目标风险不一致；源验证饱和、空间重叠时区分力弱 |
| 最小 source-val CE，精确并列取最早 | 仅源验证标签 | 阅读后可考虑独立探索，不自动加进主报告 | 饱和 OA 下仍有数值变化，但置信度更强不等于目标分类更好 |
| DEV | 源标签和无标签目标样本 | 文献/实现调研，暂不接入 | 风险估计依赖假设、密度比估计与验证拆分；空间相关性需检查 |
| SND | 无标签目标输出/表示 | 文献/实现调研，暂不接入 | 邻域结构、温度及采样影响分数；不能把低类别多样性误认为好聚类 |
| 最小 target entropy / 最大 confidence | 无目标标签 | 只作诊断，不单独选模 | 对错误类别极自信也会得到好分数 |
| target-best | 目标真值 | 不用于主结果、调参或挑选规则 | 如事先登记 oracle diagnostic，必须单列，不能回流到方法选择 |

必须在看新 target 成绩前确定规则。不能逐方法/逐 seed 从 epoch100 与 source-best 中挑较高者。
修改训练预算为 200 或 300 后再根据 Houston target 成绩选预算，同样是在用 target 调参。
由于已有 Houston target 结果被反复查看，这不是全新未触碰的测试场景；增加随机 seed 不能代替独立场景/数据集的后续验证。

## 4. 文献阅读顺序与必须回答的问题

### 第一组：BN 的训练、统计和推理

1. [Ioffe & Szegedy, Batch Normalization, ICML 2015](https://proceedings.mlr.press/v37/ioffe15.html)。先区分 train 的 batch statistics、eval 的 running statistics、可学习 affine 参数。对照 PyTorch 实现核查 EMA、方差估计、`num_batches_tracked`；不要把所有差异笼统称作“BN 参数”。
2. [Li et al., Revisiting Batch Normalization For Practical Domain Adaptation / AdaBN](https://arxiv.org/abs/1603.04779)。摘要支持利用 BN 统计量进行域适应。全文重点查：统计量来自哪些目标数据、离线还是在线更新、是否重置、是否固定网络权重、推理是否依赖目标 batch 组成。不要把我们的 buffers-only 路由直接等同于 AdaBN 完整实现。
3. [Chang et al., Domain-Specific Batch Normalization for UDA, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html)。摘要明确包含分域 BN、两阶段和伪标签训练。阅读全文/代码时逐项查 running statistics、gamma/beta 是否分域，以及收益来自哪项消融。我们目前共享 affine，只路由 buffers，不能称为完整 DSBN 复现。
4. [Xie et al., Adversarial Examples Improve Image Recognition / AdvProp, CVPR 2020](https://arxiv.org/abs/1911.09665)。摘要支持原图与对抗样本使用不同 BN。重点查辅助 BN 的训练/测试路径；它提供视图分布分离的动机，不直接证明 Houston 噪声或翻转应该怎样处理。

每篇产出一张表：共享卷积？共享 gamma/beta？哪些图像更新哪套 buffers？验证用哪个 bank？测试用哪个 bank？统计量估计时机？是否额外用伪标签？是否有仅 BN 消融？

### 第二组：没有目标真值时如何验证

5. [You et al., Towards Accurate Model Selection in Deep UDA / DEV, ICML 2019](https://proceedings.mlr.press/v97/you19a.html)。摘要提出基于适应表示的目标风险估计及控制变量降方差。全文必须提取无偏/方差结论的假设、重要性权重如何估计、数据拆分、是否需要额外域分类器以及失败情形；不能把摘要中的保证当成任意域偏移下的保证。
6. [Saito et al., Tune It the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Saito_Tune_It_the_Right_Way_Unsupervised_Validation_of_Domain_Adaptation_ICCV_2021_paper.html)。摘要利用目标样本软邻域相似分布的熵评价结构，可用于超参数与迭代次数选择。全文查输入表示、归一化、温度、自相似处理、目标子集大小和官方代码。Houston 相邻 patch 很相似，需区分空间重复和类别语义；还要做表示塌缩、类别不均衡的无标签合成反例。
7. [Ericsson et al., Better Practices for Domain Adaptation, AutoML 2023](https://proceedings.mlr.press/v224/ericsson23a.html)。摘要强调无标签验证及超参数选择会改变 DA 方法评价。优先阅读验证划分、超参数搜索预算和各准则的稳定性比较，再决定是否值得实现 DEV/SND；不要从某篇论文的平均排名直接指定本数据集最佳准则。

统一阅读记录字段：论文/实现版本；要选择的是超参数还是 checkpoint；使用的标签；验证样本与适应样本是否重叠；BN 模式；评分公式及极值方向；假设；失败案例；额外算力；是否适用于我们的空间相关 patch。

建议检索词：`domain-specific batch normalization running statistics affine parameters`、`augmentation auxiliary batch normalization`、`unsupervised domain adaptation model selection source validation`、`deep embedded validation covariate shift`、`soft neighborhood density model selection collapse`、`hyperspectral spatial train validation overlap`。

## 5. 可执行的最小验证计划（未启动）

### 阶段 A：只验证机制，不查询 target OA

保持原 split、ILDA、优化器、100 epoch 预算、算法参数不变。先以 A/C 为控制对象；若进入正式 B，仍需 A/B 完整 hash audit。

| BN 条件 | 原始 source | 原始 target | 增强视图统计量 | 用途 |
|---|---|---|---|---|
| N0 | 共享 bank | 共享 bank | 写入共享 bank | 当前官方对齐参考 |
| N1 | source bank | target bank | 写入各自所属域 bank | 单独检验源/目标分域 |
| N2 | source bank | target bank | 不提交更新 | 在 N1 基础上单独检验增强污染；当前原型属于此类 |

N1 是建议新增的控制条件，当前 `controls.py` 没有证明它的完整行为；不把表格当成已实现结果。
所有条件仍共享卷积、classifier 和 BN affine；不在本轮同时加入独立 affine 或新的损失。
增强保持官方实现，正确 H/W 翻转另开完整 MLUDA 单因素对照，不与 N0/N1/N2 混做。

先扩大到 GPU、三个 seed 的短跑：检查 logits、损失、梯度、可学习参数、RNG 一致性，允许预期中的 BN buffers 不同；验证 checkpoint 重载后的两个 bank；出现额外参数轨迹差异先停止定位。
短跑验证通过不等于目标收益成立；如果训练路径未来改为 eval、冻结 BN 或不同的 batch 混合，就不能再默认轨迹不变。

### 阶段 B：预登记选模与完整模型状态

- 主规则 fixed epoch100；次规则最大 source-val OA、并列最早，均适用于所有方法和三个 seed，不按结果挑选。
- 分域版本 source validation 固定 source bank，target test 固定 target bank；共享版本保留共享 bank。不要为提高分数在测试时切 bank。
- checkpoint 必须保存权重、source/target BN banks、配置、epoch、RNG；不能把 epoch60 权重与 epoch100 BN banks 拼接。
- 默认不额外做测试时 BN 重估；如研究校准，另立条件并固定无标签数据范围、顺序、batch、EMA/累计估计方式。
- 完整 MLUDA 的双输入 forward 必须明确 source/target 的 BN 路由及参考 source batch；不能把为单输入 Backbone 写的外层 router 不加区分地套上去。
- 若主实验加入 buffers-only 路由，所有可比较基线都应明确定义对应版本，不能只修我们的 BN 而使用未标注的历史 baseline。
- source validation 记录 OA、CE、首次达到峰值 epoch、峰值并列次数、选中 epoch；将“是否饱和”作为待测事实。

### 阶段 C：讨论无标签评分，不急于替换主规则

在独立开发问题或无标签合成控制上先排查 DEV/SND；固定子集、评分频率、温度/密度比设置和算力预算。
评分过程使用只读 eval、显式 BN bank，保存/恢复 RNG；计算分数不能偷偷修改训练 buffers。
若采用空间隔离源验证，应另列协议，并保证训练 patch 与验证 patch 的实际像素支持不相交；仅中心不同或只间隔一个半径不够。
空间划分改变了官方 split，不应覆盖官方 180/class 主协议。源标签允许的开发检查不等于可保证目标风险排名。

### 阶段 D：冻结方案之后才做统一 target 测试

每个条件训练 100 epoch 后，分别报告预登记的 epoch100 和 source-val-best 两张完整表：OA/AA/Kappa、七类 accuracy、三个 seed、mean ± sample std（ddof=1）、选中 epoch。
保存官方 OA 分母与完整预测样本口径的标注，不能混算。共享/分域、官方/修正 flip 都明确命名。
旧实验若具备完整 epoch100 checkpoint，可在另行授权的评估任务中追加 fixed100 对照而不重训；缺失 BN banks 或配对参考状态时不得靠重新校准伪造历史 checkpoint。

## 6. 本轮决策

目前最稳妥的是：先以官方固定 epoch100 为下一轮主口径，source-val-best 作为统一次要对照；BN 和翻转分开验证。DEV/SND 是需要评估的候选工具，不是必需新增模块。
这是协议建议，不是“已经发现最优选模方法”，也没有改变第一、二轮已报告结果。
