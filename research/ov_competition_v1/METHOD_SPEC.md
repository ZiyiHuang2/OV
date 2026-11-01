# OV-Competition V1｜计算定义与受控实验规格（待服务器验收）

## 研究任务与边界

**任务**：单 CLIP 编码器＋可训练 cost aggregation/decoder 的标准遥感 OV 语义分割。

**唯一问题**：B 的同类条件更新与 C 的跨类别竞争条件更新，在**同一训练源、同一完整 CAT-Seg 宿主、同一计算结构**下是否存在有用增量？禁止提前声称“修复原生 Swin 内部注意力”“可靠性概率”或“可普遍防止错误传播”。

实现参考：
- `ZiyiHuang2/OV` `feat/ov-adaptation-reobserve` 分支 `cat_seg/modeling/transformer/model.py`：`Aggregator.forward` 按 `corr -> corr_embed -> layers -> decoder` 运行。
- 旧 `research/semantic_structure/model.py`：同类 evidence feature 的 pairwise affinity 加入几何分数后 softmax。
- CAT-Seg 原生：`AggregatorLayer` 先 `swin_block` 再 `attention`；本 V1 额外更新位于第一层 **Swin 之后、Class Transformer 之前**。只测试此插入点。
- PCA-Seg：并行空间/类别分支、专家融合与正交化；是必须对照的近邻，不预设 V1 创新已成立。

## 同一数据结构上的 B/C

- 宿主原生相关性 `corr: [B,P,K,H,W]`；仅在经过宿主原生可能的类别截断**之后**，采用 `s=mean_prompt(corr) : [B,K,H,W]`。正式 V1 应使用 single-prompt 配置，禁止混用 decoder logits、sigmoid 分数及原始相关性。
- 中间特征 `h: [B,D,K,H,W]`，默认 `D=128`，来自宿主第一层 `swin_block` 的输出。
- B 条件 `t^B_{i,c}=s_{i,c}-mean_{j∈N(i), j≠i}s_{j,c}`：只依赖当前类；边缘和空间无效位置严格处理。
- C 条件 `t^C_{i,c}=s_{i,c}-max_{k≠c}s_{i,k}`：增加其他类竞争信息，不当作真实置信度；若有效类别不足 2 个，退化为 B 的条件。
- B/C 使用相同 `f_theta`、`g_phi`、value projection、输入维数 8、3×3 邻域、残差幅度及插入位置，区别仅为 `t^B`/`t^C` 来源。

关系输入：

`[s_i, s_j, s_j-s_i, s_i*s_j, t_i, t_j, t_i*t_j, abs(t_i-t_j)]`。

`a_ij = f_theta(features_ij)`；从合法的非自身邻居中 softmax，**允许为空**；只有在合法邻居存在时才计算 `sum_j w_ij*(h_j-h_i)`。接收门控 `g_i=hard_sigmoid(g_phi([s_i,t_i]))`。

输出 `h'_i = h_i + alpha * g_i * Value(sum_j w_ij*(h_j-h_i))`。不覆盖原生 `h_i`。`hard_sigmoid` 可以在单个位置严格输出 0，表示拒绝新增邻居更新；其饱和区间梯度为零需在真实训练中监测。初始化推荐 `alpha=0.01`，不是 0；显式 `enabled=False` 用于严格等价性测试。`alpha=0` 时主损失第一步不会驱动关系与门控参数，是预期梯度现象，不是失败。

**计算性限制**：B 与 C 的参数/结构/FLOPs 应一致。信息统计类型不同，仍非严格的统计因果隔离。B 是“同类条件＋残差接收”的完整宿主对照，并非原旧 conditioned 网络的精确复刻。

## 四个实验角色（D 暂不训练）

- **A（Native）**：原完整 CAT-Seg；不安装新增模块。
- **B（Self）**：同类局部对比的条件空间更新。
- **C（Competition）**：跨类别竞争对比的条件空间更新。
- **D（预注册，条件触发）**：以同样的竞争条件 `t^C`，**只在当前位置做特征重标定，不读取邻域状态**：`h'_i = h_i + alpha*g_phi([s_i,t_i])*V(h_i)`；相同插入点、训练配方和初始宿主。与 C 的容量/FLOPs 不完全相同，**独立报告**，不通过无效参数伪装匹配。D 回答“竞争信息改善类决策，是否真的需要邻居交换”。若 C 在第一轮无独立收益，D 不运行。可进一步采用严格等参数 pointwise MLP 作为补充，但必须在看到正式结果前说明替换原则。

## 实现关口

1. `enabled=False` 直接返回原输入，并在**真实完整权重**上比对原生推理输出（不能仅以合成张量说明宿主等价）。
2. `alpha=0.01` 下关系层、门控层、value projection、alpha 都有可用梯度；允许 `alpha=0` 的第一步受门控梯度为 0。
3. `K=1` 退化 B；类别重排等变（对未触发宿主 top-k 截断的词表）；类别 padding、空间越界与无有效邻居不会产生 NaN。
4. 验证接收方/邻居互换时关系输入与结果可以不同；不得把仅依赖 i 的项作为 softmax 常数声称方向性。
5. 训练/推理的原生类别表、prompt、padding、top-k 截断、后处理必须保留。由于原生 `pad_len` 可以截断 K>256，V1 只保证不受截断干扰的有效词表下自身等变；截断场景须单独审计。
6. V1 不修改 CLIP、原生 Swin attention 内部权重、类别聚合器、损失或解码头。不得在 C 特别使用不同数据、额外视图或更长日程。

## 训练关口与主结论约束

官方 [Pi-Seg 配置](https://github.com/LiBingyu01/Pi-Seg/blob/main/configs/vitb_384_OVRSIS95K.yaml) 是 **候选参考**（ViT-B/16；35 类；384 输入；2 层聚合；24×24；全局 batch 8；40k steps；lr 2e-4；CLIP attention 微调；PiNI 原配置 ON）。研究 A/B/C 必须统一关闭 PiNI，不得宣称属于 Pi-Seg 官方基线复现。`OVRSIS95K_val` 是否是真正可用于选择 checkpoint 的独立源验证仍待资源与划分审计。

- 单 seed 第一轮给出候选效果，不可称“稳定优于”；若 C>A、B 且针对错误有有意义收益，再补 D、多 seed 和匹配强方法。
- 外部开发集参与方向决策后不得当封存 test；DLRSD 200 图作为历史开发集，不单独支撑最终论文。
- 预先冻结 checkpoint 选择规则，不能按外部集选择 C 最优权重；严格区分跨域迁移与下游留类。
- 若 C 无独立收益且工程链路正确，终止该具体机制；不追加无限调门控、学习率或额外模块。

## 当前状态

**已完成（本地 CPU）**：独立模块、B/C 形状与梯度、数值边界、动态词表测试。**未完成**：服务器上的数据/权重确认、完整 CAT-Seg 数值等价、Detectron2 Trainer 优化器/checkpoint 一体化、真实 CUDA 短训、正式 A/B/C 训练。未经后者不得宣布进入正式训练。
