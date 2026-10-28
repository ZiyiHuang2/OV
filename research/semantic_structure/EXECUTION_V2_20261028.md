# OV semantic-structure v2：服务器执行报告（2026-10-28）

本次使用真实冻结编码器特征和 DLRSD 训练/验证划分完成四组单种子匹配训练。固定第 4000 次更新的验证结果显示：自适应描述选择比均值对照高 0.90 个 mIoU 百分点；条件传播相对相应对照没有显示收益。完整配置优于 B，但略低于 C，且存在类别退化和误检增加。这是单数据集、seen 类、单种子结果；四条验证曲线到终点仍在上升，不能宣称收敛、统计稳健或跨域泛化。

实验目录：`/home/wangchengjun/huangziyi/semantic_structure_acceptance_20261028/model_v2_fulltrain_20261028`。实验执行代码 SHA `da51ba1bd2801f3c828659ca97328ac7e3a6c54b`，文件级哈希见 `code_identity.json`。原始请求 SHA 为 `2729ac95aaf5cbae0e3af6aacf270082ab0d968d`；执行期间合入的 v2 上游提交为 `fb433b644f26de4ac454accc3d00e1421fd094d4`。v1 融合/传播实现与结果保存在 `research/semantic_structure/archive/v1_pilot_20261028/`，不能与 v2 数值当成同一架构的对照。

## A. 单元和合成测试：工程证据

`tests_fixed_cuda_init.log`：15 个 CPU/GPU 测试通过。GPU 合成测试覆盖 4 种配置、K=1/3/5、batch=2、非方形多尺度输入、不同有效描述数、输出尺寸、路径差异、梯度与严格保存重载。测试包含类别顺序、padding、边界、标签范围、全 ignore batch 不更新 AdamW，以及像素展平交叉熵与标准交叉熵的一致性。这些测试不能说明真实模型效果。

修复包括：拒绝空词表、非有限结构特征和错误标签；全 ignore batch 跳过优化器；使用等价的像素展平交叉熵和确定性算法，以便 CUDA 严格续训；在重置显存统计前初始化 CUDA。首个 v2 工程启动在该初始化问题上失败，**没有进行参数更新**，失败日志保存在 `engineering_failed_cuda_init/` 和 `runner.log_failed_cuda_init`；修复后新增独立进程 GPU 启动测试，全部测试重跑通过。未削弱断言。

## B. 真实数据和工程验收

训练集/验证集来自既有 DLRSD 自定义 decoded-content 去重划分，分别 1604/200 张；样本索引和 ID 见预先固定的 `protocol.json`，图像和 mask 对应、文件哈希和类别映射见 `features/*_metadata.json`。提取前对所选图像检查文件字节与解码 RGB 唯一性、train/val 不重合，以及图像/mask 原尺寸一致；`features/input_audit.json` 保存核查结果。与 v1 已保存的 32 train / 16 val 子集相比，新提取的图像、结构、文本、有效掩码和标签逐值一致，见 `cache_replay_and_coverage.json`。

原始验证标签按 `seen_class_ids` 映射到 14 个连续训练类别，unseen 像素映射为 ignore=255。四组使用相同有效像素范围，14 个类别在本次验证中均有 GT 像素。现有划分是自定义划分，未经验证为 DLRSD 官方划分。封存 test manifest、图像及标签均未读取；`classes_test.json` 仅用于取得类别词表和 ID 对应关系。

| 资源 | 实际路径 | SHA256 |
|---|---|---|
| OpenAI CLIP ViT-B/16 | `/home/wangchengjun/huangziyi/open-ov/asset_store/backbones/clip/ViT-B-16.pt` | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| RSIB ViT-B/8 | `/home/wangchengjun/huangziyi/open-ov/asset_store/backbones/dino/RSIB.pth` | `fd60a7c5358e7f51bcad202885530331852f902dbac4868cb229e4b589e15392` |
| 训练/验证 manifest | `/home/wangchengjun/huangziyi/ovrs_formal_20261011_dlrsd_v1/frozen_data/train_val/manifest.json` | `c859ac4bdfc7c64cb14831d4fe8b7471a45e89c6c6f0bd95f78c8964e1bef99b` |

CLIP 权重哈希与已安装 OpenCLIP 官方 URL 所载哈希一致。RSIB 的 teacher 主干加载无缺键，仅预训练 head 为额外键；RSIB 源码路径及哈希记录在协议与缓存中。未重新下载第三方权重比对，也未独立验证其预训练数据和本数据集无重合。编码器都处于 eval/frozen，提取前后状态哈希一致，可训练编码器参数数目为 0；解码器训练过程不加载编码器（`features/encoder_audit.json`）。

两个编码器使用同一完整 RGB 图像的 224×224 bicubic resize，各自采用既定归一化；mask 最近邻 resize 至 224×224。CLIP 最后一层 patch token 经 `ln_post` 与同一 CLIP 的 `visual.proj` 变换，形成 512×14×14 特征，与同一模型的文本编码相配；每张图的 CLS 路径与原生 CLIP 全局输出核对。结构特征是 RSIB 最后一层归一化 patch token，768×28×28。逐 patch 投影是全局对齐空间的空间外推，不能因此宣称 dense 对齐质量已验证。使用原生 224 输入绕开现有 RSIB 位置插值的尺寸限制。

文本仅使用事先固定的两个类名模板 `a satellite image of {class}` 和 `an aerial photo of {class}`，去重后编码，未依据验证错误图修改。完整缓存由 `extract_rsib.py` 复现，和仓库原有 CLIP+ResNet 适配器是不同的明确配置。缓存 SHA256：train `0a9512f1d0062fcfa9985f5f1ccd31267ac97f3827909f9f9b908df51a957387`；val `0c5699333e9d7397b7d604388da5de2afa3b71151c8f89e7e9a0882ecef3cef5`。缓存无在线图像增强。

四组先在固定 4 张训练图上训练 20 次，batch=2。B/C/D/E 的固定四图 loss 分别由 2.639081/2.639067/2.639081/2.639067 降至 2.609243/2.608865/2.609267/2.608821；输出 `[4,14,224,224]`，保存重载逐值一致，连续 20 步与 10 步后恢复至 20 步的模型、优化器、RNG 和采样轨迹完全一致（`engineering_acceptance.json`）。这是工程检查，并未证明四图已经过拟合。

## C. 真实数据四组匹配训练

用户在 v1 100 步验收后授权扩大训练。`protocol.json` 在特征提取及训练前固定：训练 1604 图、验证 200 图、seed=42、独立采样器 seed=43、随机有放回抽样、batch=8、AdamW lr=0.0002、weight decay=0.01、梯度裁剪 1、width=64、温度 0.1、语义强度 1.0，每组 4000 次真实更新（约 19.95 个等效采样遍数），每 1000 步验证，**只报告固定终点 4000，不选最佳 checkpoint**。共享参数的初始哈希、采样轨迹、数据/文本、优化控制和更新数均一致（`matched_checks.json`）。B/E 使用 GPU 0 串行，C 使用 GPU 1，D 使用 GPU 2；运行时没有终止其他作业。

| 组 | v2 配置 | 终点 mIoU | 像素准确率 | 可训练/有效参数 | 峰值 allocated | 平均每步 | 运行总时间 |
|---|---|---:|---:|---:|---:|---:|---:|
| B | mean + geometry | 55.1804% | 77.8119% | 266561 / 266560 | 662 MiB | 57.55 ms | 234.3 s |
| C | adaptive + geometry | 56.0815% | 77.7131% | 266561 / 266560 | 696 MiB | 62.55 ms | 254.7 s |
| D | mean + conditioned | 55.1659% | 77.9092% | 266561 / 266560 | 901 MiB | 72.80 ms | 295.6 s |
| E | adaptive + conditioned | 56.0278% | 77.5855% | 266561 / 266560 | 933 MiB | 76.18 ms | 309.5 s |

有效参数排除在类别 softmax 中抵消的共享标量 head bias；四组注册参数完全相同。显存为各独立训练进程的 PyTorch 最大 allocated，不包含编码器预训练和完整模型系统开销；时间包括周期验证和重载，且不是隔离的标准化性能基准。所有组的前 100 步平均训练 loss 约 2.31，最后 100 步约 0.49。四条验证曲线从第 3000 步的 50.40/51.13/50.36/51.06% 升到第 4000 步的 55.18/56.08/55.17/56.03%，不能视为收敛。

终点差值：C−B **+0.9011 pp**，D−B **−0.0146 pp**，E−B **+0.8474 pp**，E−C **−0.0536 pp**，E−D **+0.8620 pp**；交互项 E−C−D+B 为 **−0.0391 pp**。在本次设置中，语义选择有可观测的正向点估计，但无法由单种子量化稳定性；条件传播没有可观测的总体改善，不能据此证明其在所有数据或预算上无效。完整方案不超过更简单的 C。

收益伴随 seen 类差异：E 相对 B，dock IoU 从 32.06% 增至 46.73%，但 court 从 74.62% 降至 70.44%，grass 从 57.36% 降至 56.45%，sand 从 52.29% 降至 50.44%。E 的 grass 错误预测像素为 439168，B 为 374323；pavement 分别为 393858 和 353826，且 E 的总体像素准确率下降 0.23 pp。field 在全部四组的 IoU 和召回均为 0%，无任何该类预测，precision 因分母为零记为 null。完整逐类 IoU、precision、recall 和 FP 见 `per_class.csv`；类别例子是观察到的结果，不是提前规定的失败组。

每组 `training/<arm>/validation_004000.json` 保存总/逐图原始混淆矩阵（GT 行、预测列）和逐类指标；`summary.csv`、`per_class.csv`、`per_image.csv`、`learning_curve.csv` 和 `comparison.json` 为导出。mIoU 对非零 union 类取平均；全部组图像顺序和 GT 像素计数相同。无 unseen 指标或 hIoU，因为 unseen 验证像素被忽略。

## 结论与资源边界

1. **真实训练条件：** 在已验收的固定特征缓存条件下具备，四组均完成，训练 loss 下降，保存重载和小规模续训通过。尚非端到端图像训练，未进行随机裁剪、在线增强或编码器微调。
2. **模块收益：** 自适应描述选择在此单次试验有 +0.90 pp 的点估计；条件传播差值接近零且训练更慢、更耗显存。它们的跨种子稳定性和跨域效果未知。
3. **完整方案：** E 高于 B 0.85 pp，但比 C 低 0.05 pp；不能宣称两个模块叠加带来额外收益。部分 seen 类 IoU 退化、部分误检增加。
4. **下一步决策：** 保留 C 作为值得进一步验证的候选，暂缓扩展条件传播及继续堆叠模块；当前结果不足以永久否定条件传播。四组曲线仍在上升且仅单种子，应先验证优化充分性与重复性，再提出更强方法结论。本轮不自动追加实验。

此前资源核查还检查了 `~/huangziyi/open-ov/experiments/BATCH2_DIAGNOSTICS_20261016/protocol_A/RESULTS_A.md`（DLRSD 类别划分诊断，非匹配跨数据集主协议），以及 `~/huangziyi/open-ov/asset_store/datasets/ovrsis_v1` 和 `~/huangziyi/open-ov/assets/ovrsisbenchv2/datasets`（当时为空）。服务器还存在其他遥感数据，但尚未为此原型建立和验收匹配的 source/target 协议。因此完整 HG 复现、跨数据集、unseen/hIoU 和多种子结论仍缺资源身份或协议验收，不能以本次结果代替。

环境：Python 3.10.14、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、open-clip-torch 2.16.0、timm 0.9.16，3×RTX 4090 24 GB、驱动 580.159.03；`environment_full.txt` 含完整环境。复跑命令见实验目录 `RUN_COMMANDS.sh`，应指定全新输出目录；原始训练日志、缓存、权重路径、checkpoint 和评测 JSON 均留在服务器，未提交到 GitHub。
