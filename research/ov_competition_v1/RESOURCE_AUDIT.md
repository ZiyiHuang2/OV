# OVRSIS95K｜服务器验收单（未执行）

本文件是待服务器 Codex 执行的**只读核查任务**，不是资源已经可用的证明。禁止自动下载/重组数据并把未审计文件记作正式数据；禁止使用封存 test 进行选点。

## 1. 数据与类别

- 识别 OVRSIS95K 的真实根目录、版本及 95K train 的实际图像/mask 数量；检查是否有官方 val、来源与空间独立性；无可信源 val 则从训练源建立固定开发划分，预先记录 seed 与分组规则。
- 读取 35 类的 JSON/metadata；核查 mask ID 映射、ignore=255、背景语义、非法标签像素、每类图像/像素支持量。抽样人眼核验 RGB 与 mask 配对；保存 hash 和 manifest。
- 核查外部测试有无训练数据内容/地理场景重叠；分类为 source val、external dev、sealed final。

## 2. 完整权重及软件

- 记录服务器 GPU、显存、PyTorch、CUDA、Detectron2、实际 Git commit、worktree 状态、配置哈希。
- 三组采用同一个完整已训练 CAT-Seg checkpoint；逐键记录共享参数 `missing/unexpected`（新增模块键例外），不能用不同预训练宿主或 CLIP 编码器混比。
- 审核 2 层聚合、hidden=128、24×24、window=12 与权重是否严格兼容；若改架构需明确重新预训练或单独报告新增初始化。
- 审核 CLIP attention 微调与冻结策略；A/B/C 一致。PiNI 在三个组均关闭。
- 记录 prompt 处理、动态类数、原生 `pad_len` 截断、文本权重及推理恢复顺序。

## 3. 训练入口关键工程风险

当前仓库 `train_ovrs.py` 使用自定义 `OVRSTrainer` 和 `OVRSCheckpointer`。**不能仅在训练开始后安装新增模块**：那会使 optimizer 缺失 B/C 参数；也不能仅在预训练加载前加新模块却默许未审计的权重缺失。服务器实现必须满足以下顺序：

1. 构建共同的模型结构并核实原始权重加载身份；
2. B/C 新增模块在 optimizer/DDP/训练 checkpoint 注册前完成；
3. 预训练初始化仅允许新增模块参数为未匹配键，其他共享键严格验证；
4. 恢复训练时必须完整匹配 B 或 C 模型和配置（不得跨臂无提示恢复）；
5. A/B/C 记录共同初始化的键哈希、采样身份与训练日程，并真正反传到新增模块。

## 4. CUDA preflight（必须实际运行）

- A 原生 vs 装上旁路模块：真实权重单图输出一致。
- B/C 各至少一次 384 图像、35 类、batch>0 的前向/反向/更新、保存/严格重载与续训一致；确认新增参数包含在 optimizer。
- 记录 CUDA allocated/reserved、峰值显存、每步时间；若 batch8 超过资源，只能修改**全部组相同**的有效 batch 与日程，不能只缩小 C。
- 动态词表 K=1/2/35、重排、padding 和缺席类追加 smoke；完整模型区别与模块区别分开报告。

## 5. 验收输出格式

`resource_audit.json`：`data_version, train_count, source_val_count, num_classes, label_audit, overlap_audit, weights_sha256, model_keys, code_commit, versions, gpu, native_parity, optimizer_coverage, b_grad, c_grad, resume_parity, memory, unresolved`。

**如果任何关键项 FAIL 或 UNKNOWN：不得填写正式训练 READY。**
