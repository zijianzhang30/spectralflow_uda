# 下一步排查：BN缓冲区路由与空间翻转

这是诊断原型，不是第三轮正式训练入口。首两轮的代码、权重、结果和协议均保持不变。

## 当前控制

`controls.py` 实现两个互相独立的改动：

1. `DomainBNBuffers`：源原图、目标原图各自维护 running_mean、running_var、num_batches_tracked；卷积、分类器及BN affine参数仍共享。增强视图可以前向，但不提交其统计量更新。
2. `spatial_flip`：只翻转H/W轴，保留batch索引、波段索引及中心像素。

BN检查中仍使用官方增强，避免同时改变两项。空间翻转仅做身份/维度契约检查，尚未接入MLUDA训练。

## 真实数据配对检查

`check_controls.py` 使用已保存的Houston划分和ILDA图像，CPU、seed1341，A/C各38次更新。
每组共享与路由版本使用相同batch、增强、初始化、SGD与Flow随机数。
它不复现首轮全局RNG采样轨迹，仅检验“BN路由前后是否保持训练过程一致”。

检查范围：

- 每步源/目标logits、CE/FM、全部可学习参数、梯度，以及Flow参数/梯度/RNG逐位相等。
- 共享BN每层更新228次；分域BN各38次，增强视图不写入原图统计量。
- 单独更新目标统计量不能改变source-bank下的eval输出。
- checkpoint同时保存网络状态和两套BN状态，内存序列化/重载后两域预测逐位相同。
- 异常退出BN上下文会恢复原buffer对象。
- 四种空间翻转组合保留样本/波段身份和中心像素。

注意：不能在有梯度的forward之后原地copy_还原BN buffer；autograd可能保存它们的版本计数。本原型通过临时替换buffer对象避免这一风险，反向过程已包含在检查中。

复现：

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python investigations/normalization_control/check_controls.py
```

结果保存在 `checks.json`。CPU单seed短跑通过不等于GPU多seed全训练已验证。

## 使用接口（尚未接入正式trainer）

```python
router = DomainBNBuffers(model)
model.train()
with router.use("source", update=True):
    source_features, source_logits = model(source_patch)
with torch.no_grad(), router.use("target", update=True):
    target_features, target_logits = model(target_patch)
with torch.no_grad(), router.use("source", update=False):
    model(source_augmented_patch)

# 验证/推理必须显式选择bank，不可裸调用model(x)。
model.eval()
with torch.no_grad(), router.use("source"):
    validation_logits = model(source_validation_patch)[1]

# 必须同时保存并重载两部分，model.state_dict()本身不包含router的bank。
checkpoint = {"model": model.state_dict(), "bn_banks": router.state_dict()}
```

目标推理选择target bank，但它的实际目标准确率尚未验证。

## 后续顺序与停止条件

1. 完成本目录的配对训练与序列化检查，先证明统计量路由没有意外改变训练目标或梯度。
2. 如进入第三轮，先明确源验证和目标测试的BN策略，并将其写成新协议；不覆盖旧mixed-BN结果。
3. 选模规则不能根据新目标成绩临时调整。源验证接近100%后如何选择checkpoint，是必须在正式实验前明确的问题。
4. 正确翻转轴的完整MLUDA实验应单列为corrected-flip基线，只改变增强轴；不与BN路由同时修改后归因。
5. 基础行为明确后，才讨论新的Flow分类目标；目前不建议继续大范围搜索FM权重或同时叠加多个模块。

## 用户调研时最有用的三个问题

- **BN**：哪些方法只分离running statistics，哪些还分离affine参数？是否把增强视图单独归一化？验证和推理具体使用哪一套统计量？
- **选模**：没有目标真值、源验证饱和时，用什么事先确定的准则选checkpoint？它对标签比例变化/空间重叠有哪些假设？
- **Flow**：端点编码器是否固定或stop-gradient？运输结果是否参与分类损失或推理？怎样保持类别语义，并证明收益超过普通插值/配对拉近？

可优先对照前一份 `../bn_augmentation/REPORT.md` 中的AdaBN/DSBN、AdvProp、DEV与OT-CFM。文献调研由用户继续，本次没有扩大搜索或宣称新颖性。

## BN 与选模调研计划

具体阅读顺序、官方选模代码证据、BN 单因素对照和下一轮选模建议见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。这是一份调研与预登记草案，尚未启动新训练或修改既有协议。
