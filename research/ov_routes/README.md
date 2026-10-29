# OV 下一阶段：两个独立、可执行的研究分支

状态：实现与 CPU 算法/模拟入口测试已完成，**尚未完成真实 CAT-Seg 权重、DLRSD 或 GPU 验收，也没有新效果结果**。
真实运行前，入口自动执行完整模型的原生输出一致性、真实图像 meta 梯度和裁块重编码检查；失败即停止。
不会退回随机初始化、267k 缓存解码器或合成数据。旧 semantic_structure 结果不被改写。

## 研究取舍

七组 80k 说明在旧冻结缓存原型里继续堆叠模板与传播没有足够依据；它没有否定所有适配、尺度和语义方法。
本轮选择改变两个不同环节，而不是把新模块继续加到旧原型上：

* **A：源类别内部的留类适配。** 从完整预训练 CAT-Seg 出发，冻结 CLIP，更新原有 aggregation/decoder/guidance。
  每次从训练类别中选一个临时留出类，用另一张图上的其余类别做一步临时 SGD，再用 query 图的留出类像素检验更新。
  query 使用完整 SOURCE 词表竞争。真实目标 unseen 标签不参与任何训练；临时留出类仍参与外层训练，不能称其为真正 unseen。
* **B：根据语义竞争选择第二次视觉观察。** 冻结完整 host，用首次预测的候选类别文本差、局部竞争和窗口几何，
  预测重新编码某个原图窗口的收益。训练时穷举 8 个真实裁块，用训练标签计算融合后 BCE 的改善；推理只看初始预测与文本，
  选择指定数量窗口，每个窗口重新运行完整视觉编码器。不是对旧缓存做 ROI pooling。

A 是 first-order meta-learning 候选，不声称发明 episodic training；B 是 learned crop selection 候选，不声称发明多尺度。
当前 B 的动作集合是固定位置的 .5/.75 尺度窗口，默认一次补充观察。尚不包含自适应 proposal、上下文扩展、循环推理或 RL。
它足以检验“语义特征能否帮助选择有效的新观察”，不能直接支持更强的主动推理机制结论。

属性组合 C 与多源兼容监督 D 保留为独立后续路线，本次没有放入空实现。
C 需要可信的属性定义与图像判别对照；D 需要实际数据源、类别层级及标注覆盖表，不能凭类名自动猜测。
四个方向不是已确认创新，也不应未经验证合成一个网络。

## A 的真实优化定义与对照

令 S 为 source 类别，H 为本 episode 暂时留出的一个 source 类别。
support loss 在 S\H 词表和相应像素上计算；其他位置 ignore。
临时参数为 `theta_fast = theta - inner_lr * stop_gradient(grad(loss_support))`。
外层目标为 `loss_support(theta) + weight * loss_query_H(theta_fast)`。
query 只在 H 的有效像素上取损失，但候选轴保留完整 S，使其必须与 source 其他类别竞争。
实现保留 fast 参数对原参数的 identity 梯度，省略 Hessian；并非对 fast 参数 detach 后丢弃 query 梯度。

三个独立运行必须从同一完整 checkpoint 起步：

| mode | 内层 | 外层 | 用途 |
|---|---|---|---|
| erm | 无 | support/query 两图全部有效 source 像素 | 普通冻结 CLIP 的任务适配 |
| joint | 无 | 同 meta 的 support 子集与 query 留类像素 | 排除采样/监督重加权带来的收益 |
| meta | 一步 stateless SGD | 同 joint，但 query 用 fast 参数 | 检验适配更新本身的作用 |

三组使用同样的 episode、图像和增强 RNG，且采用相同空 episode 跳过规则。ERM 与另两组的监督像素不同是明确对照变量。
每个有效更新读取两个不同训练图，FP32；全部模块保持 eval 模式以冻结 dropout/BN，梯度仍更新非 CLIP 参数。
在线增强为双线性 resize 384、mask nearest、随机翻转/90 度旋转。
本入口暂不做 CLIP finetuning、AMP、DDP 或 batch>1；不是与旧 80k 配方的匹配对照。
训练 loss 延用 host 的有效像素 BCE，**不是**旧原型的 CE。

## B 的动作、监督与对照

每张图的候选集：两个尺度，各四个角落，共八个窗口（极小图去重）。选图特征不含类别 ID、seen/unseen 标志或标签。
文本输入是初始局部均值分数前两类的归一化文本向量绝对差；另有 9 个竞争/几何统计。
类别重排保持结果对应（精确分数并列时 top-k 的任意 tie-breaking 除外）。
选择器为 `(text_dim+9)->64->1` MLP，回归训练图上单个动作的全图有效像素平均 BCE 改善，目标乘 100 改善数值尺度。
评测比较 budget=0、fixed、random、learned；后三者使用同样动作集合、预算、融合方式和同一完整 host。
默认 budget=1；多个重叠动作的收益不保证相加，不将单动作训练解释成最优联合选择。
固定预算不允许 learned 通过少执行计算获得不公平成本优势。记录实际窗口和编码次数。
融合使用 sigmoid score 平均，与原 host 的分数语义一致；窗口重叠计数归一化，未覆盖像素保持原值。
先 sigmoid 后插值的推理路径与训练时先插值 logits 的路径分开实现，避免偷偷改变 baseline。

## 输入与环境

在**完整 OV 仓库根目录**运行。使用服务器已经可运行 CAT-Seg 的环境（PyTorch >=2.5、Detectron2、仓库 runtime 依赖）。
本轮本机 CPU 测试环境为 Python 3.12 / PyTorch 2.15.0+cpu / pytest 9.2；没有验证服务器 2.5.1/CUDA 下数值一致性。

必须提供：

1. 现有 version=1、ignore_label=255 的 train/val manifest。train mask 已映射成 source 连续 ID；val mask 保留完整原始 ID。
   不能把 semantic_structure 的 feature `.pt` 当 manifest。`test_classes` 只读取词表，任何 test 图像/标签记录均不访问。
2. **完整、已经训练过分割的 CAT-Seg ViT-B/16 权重**及其准确匹配 YAML。包括 CLIP、guidance 与 aggregator 全部参数。
   官方 COCO 权重不是 strict source-unseen 保证；必须报告源类别覆盖。旧随机头训练终点不能冒充合理分割预训练起点。
3. 原仓库 CLIP 初始化文件 `~/.cache/clip/ViT-B-16.pt`。原加载器会验证官方 SHA256；缺失时原加载器会尝试下载。
   服务器可把已有同哈希文件链接到此位置。完整分割 checkpoint 随后严格覆盖模型参数。

权重由现有 EvaluationCheckpointer 严格加载；缺键、多键、形状错误均拒绝。
仅当确认官方旧 checkpoint 无配置 metadata 时传 `--legacy-weights`；它只放行 metadata 缺失，不放行张量不匹配。
不要为了加载成功删除断言、猜配置或随机补参数。类别顺序由 manifest 和显式词表传递。

## 命令

先设置自己的真实路径；以下环境变量不得留空：

```bash
export OV_MANIFEST=/absolute/path/to/frozen_data/train_val/manifest.json
export OV_CONFIG=/absolute/path/to/matching_pretrained_config.yaml
export OV_WEIGHTS=/absolute/path/to/complete_pretrained_model.pth
export OV_OUTPUT=/absolute/path/to/new_experiment_directory
# 仅对于已核实的无 metadata 官方旧权重：
export OV_LEGACY_WEIGHTS=1

python -m research.ov_routes.run preflight \
  --manifest "$OV_MANIFEST" --config "$OV_CONFIG" --weights "$OV_WEIGHTS" \
  --legacy-weights --output "${OV_OUTPUT}_preflight"
```

上面命令针对旧权重；有 metadata 时去掉 `--legacy-weights`。预检只读训练数据，不做参数更新。
下面脚本根据 OV_LEGACY_WEIGHTS 自动处理该开关。更新数要求显式指定，示例数值仅用于启动实验，不代表已证实充分训练：

```bash
A_STEPS=16000 B_STEPS=4000 bash research/ov_routes/run_comparisons.sh
```

脚本串行运行三组 A，以及同 host 上 B 的零视图、固定、随机和学习选择。
**16k 不是收敛结论或硬上限**；看固定终点与轨迹决定下一轮预算，不能混用不同预算的最佳点。
A 的每步计算和 B 的每步编码数不同，不把更新数或共享 GPU 时间直接当公平算力比较。
也可以单独运行 `train-a --mode meta|joint|erm`、`train-b` 和 `evaluate`，参见 `--help`。

```bash
python -m research.ov_routes.run evaluate \
  --manifest "$OV_MANIFEST" --config "$OV_CONFIG" --weights "$OV_WEIGHTS" \
  --legacy-weights --artifact "$OV_OUTPUT/a_meta/checkpoint_016000.pt" \
  --budget 0 --output "${OV_OUTPUT}_a_meta_eval"
```

评测 A 时用 budget=0；评测 B 的 selector artifact 时指定 `--policy learned --budget 1`。
不支持直接将 A 与 B 合并；必须先独立回答各自问题。
恢复时指定 `--resume` 和**全新**输出目录，总 steps、配方、代码、manifest、基础权重须一致。
checkpoint 保存可训练参数、优化器和随机状态，重新加载时先严格恢复同一完整基础模型；不可单独把它当完整模型。

## 输出与判断

保存协议、基础权重/代码/manifest/词表哈希、日志、周期 checkpoint、完整词表混淆矩阵和逐图选择记录。
没有运行最佳点选择；没有在 validation unseen GT 上拟合选择器、阈值或动作。
当前校验记录不替代数据原始文件的字节/内容去重审计；沿用已冻结的数据来源。

A 要同时超过 ordinary ERM 与 joint 对照，并检查是否只是学习更小的更新。
B 要超过相同预算的 fixed/random，而不仅超过无额外视图；还要检查 gains 是否集中于单类、漏检/误检是否交换。
第一个有可信增益的候选应进入另一数据集，而不是反复调同一 200 图。一个种子可用于路线筛选，不能声称统计稳健。
若 A/B 失败，分别回到适配目标或观察策略；不自动组合两条失败路线。

## 验证边界

```bash
python -m pytest -q research/ov_routes/test_routes.py
```

本地已通过 14 项 CPU 测试，包括 first-order 梯度与手算公式对照、类子集/全词表竞争、ignore 梯度、
真实裁块调用次数、坐标/融合、类别重排、选择器梯度、manifest test 封存，以及读取真实 PNG 文件的**模拟 host**训练/重载/续训入口。
模拟 host 只存在测试文件中；生产入口没有 mock/fallback。模拟入口测试不等于真实 CAT-Seg 或 DLRSD 验收。
真实权重上的 parity/梯度/显存和实际效果必须由服务器 preflight/训练结果补充，本次没有这些结果。

## 设计依据

已有证据：[Notion 实验记录](https://app.notion.com/p/3f6445aea41f81a8b4b6f2a47156a6b6)、
[论文路线笔记](https://app.notion.com/p/3d4445aea41f81f3a411c01e9e43666e)、
[七组 80k](https://app.notion.com/p/408445aea41f812498dee8e57a842291)。
借鉴边界：SAN 的区域识别接口、OVRS 的方向/尺度、GeoAgent 的尺度选择、episodic/meta-learning 思想均有先行工作。
本实现没有声称复现这些论文的完整方法，也没有完成新颖性排他检索。
