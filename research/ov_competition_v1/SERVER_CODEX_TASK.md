# 服务器 Codex 执行任务｜OV Competition V1

## 任务目标

在 `ZiyiHuang2/OV` **独立工作树/新分支**中，将附件 `ov_competition_v1` 接入**完整预训练** CAT-Seg，验证 A Native、B Self、C Competition 三组。**先完成资源审计与真实 CUDA 短程验收，全部通过后才按冻结协议启动正式训练。**不覆盖旧输出，不擅自使用封存测试集，不向公开仓库 push，除非用户另行明确授权。

## 已有文件

- `METHOD_SPEC.md`：唯一研究假设、B/C 条件信息区别、D 的预注册定义。
- `competitive_update.py`：PyTorch 条件空间更新，独立于 Detectron2。
- `patch_repo.py`：针对 `feat/ov-adaptation-reobserve` `Aggregator.forward` 的最小前向补丁（执行前 `--check`；先审查 diff）。
- `install.py`：将新增模块接入已装载共同预训练权重的模型。
- `test_*`：CPU 算法测试；不得替代真实 CUDA 验收。
- `RESOURCE_AUDIT.md`：服务器数据、模型、协议和环境核对要求。

## 执行分段

**P0｜只读资源审计**

1. 确认目标 worktree 的代码 commit；别用任意旧 `main` 来套补丁。
2. 定位 OVRSIS95K 数据、35 类映射、mask 和独立源验证集；确认无泄漏、忽略标签与数据版本。如果没有可信的独立源 val，先交付固定训练源开发划分方案，不碰 final test。
3. 审计与 2 层 serial CAT-Seg ViT-B/16 相容的完整权重；列出 base 配置差别、CLIP attention finetune、PiNI OFF 的共同训练策略。
4. 输出 `resource_audit.json`，不猜测任何未知路径或文件状态。

**P1｜真实模型接线**

1. 将包复制到 `research/ov_competition_v1`，运行 patch 检查，审查后仅修改本地 `Aggregator.forward` 第一层 Swin→Class Attention 之间的位置。
2. 为 `train_ovrs.py` 的正式训练和评估入口添加严格的 `--competition-arm A|B|C` 或等效配置。**保证新增模块在优化器/DDP/检查点构建前注册**；预训练加载与 B/C 新增参数初始化严格可审计。保留 A 的原生模型键和行为。
3. B 与 C 使用同一组关系网络/门控/value 投影参数形状、相同随机初始化、相同 optimizer 参数组；唯一差别是局部同类对比 vs 跨类竞争对比。
4. 训练与 checkpoint 必须保存 `arm`、方法配置、原始权重 SHA256 和代码身份；禁止跨组恢复造成 silent mismatch。

**P2｜真实 CUDA preflight**

1. 真实权重、真实 384×384 RGB、真实 35 类及 GT 下逐项验收 native A / bypass B,C 数值等价。
2. B/C 各完成前向、反传、非零梯度、参数更新、状态保存/恢复；检查门控确实可以逐像素拒绝；检查接收方条件不会在邻居 softmax 中抵消。
3. 检查 K=1/2/35、类别重排、追加缺席类、有效类 mask、无效邻域及真实训练/推理分数域；不得将原生截断导致的差异误归因到新增模块。
4. 实际测量峰值显存、每步用时和可用卡数；如果 batch 8 不可行，交付对 **全部 A/B/C** 同等生效的替代优化配方和有效 batch 规则。
5. 对已有 `cat_seg` 训练/评测入口进行至少一次真实训练批次与严格 checkpoint resume；输出 `preflight_report.json`，失败则停止。

**P3｜正式 A/B/C**

P0-P2 通过后，按冻结相同配方训练三组（参考 40k / global batch8 / 2层 / 384 / seed42 / PiNI OFF；实际取值以已验收协议为准）。定期保存预设 checkpoint、源验证 mIoU、训练日志、开销和逐类统计。外部开发集只用作研究观察，不能用于从中选择 C 的最优 checkpoint。

**P4｜科研裁决**

输出 `decision_report.md`：C−A、C−B 的主任务和外部开发表现，类别支持统计、代表性相邻类别混淆、同类内部损伤与总体资源成本。单种子不得称为稳定提升。若 C 无独立增量且实现正确，停止；若 C 有正向结果，再运行事前冻结的 D pointwise 竞争对照，之后才考虑多 seed、强方法和 final test。

## 退出条件

- 任一阶段没有资源、权重或可信验证划分：明确失败或未知项，不自动切回 DLRSD 当正式主实验。
- 所有 A/B/C 保持同一个预训练来源与评测口径；数据/训练配方改变必须全部重新冻结并公开差异。
- 主报告提供真实执行命令、commit、权重及数据哈希、配置、检查点路径、每组日志与失败原因；不可只提交合成单测的通过截图。
