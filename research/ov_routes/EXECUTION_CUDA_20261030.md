# CAT-Seg 留类适配与重观察：真实 CUDA 首轮

代码：`feat/ov-adaptation-reobserve`，起始 d17283305acae111ab0970bc62c5ef4a953180bf，实验代码 0cec1c686526245601ac2fa0cce09923ed7e29b4。环境 Python 3.10、PyTorch 2.5.1+cu121、Detectron2 0.6；RTX 4090，FP32，禁用 TF32，确定性算法。完整 CAT-Seg ViT-B/16 权重 SHA256 `a201a50f94f4c04196e3083773832b736c2b8b81d27ea2c59cc13af3aff05760`；CLIP SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；配置 `configs/base/clip_vitb16.yaml`。旧 checkpoint 不含配置元数据：其官方来源依据既有服务器报告，484/484 张量严格核验，但没有重新下载远端字节核对。manifest SHA256 `c859ac4bdfc7c64cb14831d4fe8b7471a45e89c6c6f0bd95f78c8964e1bef99b`，1604 train、200 val，17 类同时竞争，test 封存。COCO-Stuff 171 预训练词表包含 airplane 和 boat，本文的 unseen 是下游留类，不是严格预训练 source-unseen。

真实 CUDA 预检最初因 cuDNN TF32 导致 host/native 相差最大 4.45e-5 而不通过；禁用 TF32 后按原断言通过，最大差 8.94e-8。完整模型严格加载，CLIP 冻结，support 和 held-query 梯度非零，裁块确实重新编码。真实 32 train/4 val 工程短训的 A meta 和 B 各两次更新；从第一步恢复与连续训练的参数、优化器、随机状态和逐图指标逐值一致。15 项测试通过。这些为工程验收，不是效果结果。

A 三组从相同完整预训练权重、seed 42，固定 16k 更新；B 独立使用未适配完整模型，learned 固定 4k 更新；每 1k 保存和评估，见运行协议及命令。A 三组逐步采样图像、留类和学习率相同；三组初始验证逐值相同。七个终点同一 200 图及 GT，逐图混淆矩阵累加与总矩阵、重新计算的 mIoU/seen/unseen/hIoU 一致；额外视图各每图 2 次编码，zero 1 次。以下均为同一次 17 类竞争、固定终点，单位 %。

| A 路线 | 总 mIoU | seen | 留类 | hIoU |
|---|---:|---:|---:|---:|
| 未适配 zero | 29.629 | 30.471 | 25.703 | 27.884 |
| ERM 16k | 46.673 | 51.952 | 22.035 | 30.945 |
| joint 16k | 43.871 | 47.956 | 24.805 | 32.698 |
| meta 16k | 43.887 | 47.978 | 24.796 | 32.695 |

meta 比 joint 总 mIoU 高 0.016 pp、hIoU 低 0.003 pp；与 joint 的差小到不足以支持机制收益，且总 mIoU 比 ERM 低 2.785 pp。相比 ERM，meta 的留类高 2.761 pp、seen 低 3.974 pp：这是权衡而非全面优势。ERM 的 ship IoU 仅 0.822%，meta 7.074%，两者 tanks 均 0%；airplane 分别 65.283%、67.314%。ERM 的 mobile home 59.008%，meta 37.363%；field 分别 6.091%、1.329%。各组 14k–16k 总指标变化小，末段并无清晰的持续改善，较早 checkpoint 的留类表现较高但不能用它替代固定终点。故不触发 meta 优于两种对照时才加的小学习率对照，也不根据这些曲线自动扩大预算。16k 不是收敛证明。

| B 路线 | 总 mIoU | seen | 留类 | hIoU |
|---|---:|---:|---:|---:|
| zero | 29.629 | 30.471 | 25.703 | 27.884 |
| fixed +1 crop | 29.504 | 30.474 | 24.975 | 27.452 |
| random +1 crop | 28.923 | 29.629 | 25.628 | 27.484 |
| learned +1 crop 4k | 29.200 | 30.100 | 24.999 | 27.313 |

learned 比 random 高 0.277 pp，但比 fixed 低 0.304 pp，比 zero 低 0.430 pp；因此不支持 learned 优于同预算简单策略，也不触发预定的无文本差特征对照。learned 4k 比 zero 的 dock IoU 7.13→4.30%、ship 14.11→12.60%、court 45.02→44.62%、field 12.23→11.18%；预测差异及误检见逐类表。总体主要混淆如 grass→trees：zero 448,771、learned 437,160 像素；pavement→cars 分别 378,766、386,506 像素。A 的 grass→trees：ERM 217,610，joint 262,796，meta 262,575；meta 的 bare soil→grass 232,312。计数均来自有效像素，不能解释为因果机制。

结论：首轮不支持元适配和 learned 重观察各自带来超过匹配简单对照的可靠增量；建议暂缓复杂机制，保留 ERM/joint 作为进一步确认 seen–留类权衡的简单参考。只有单种子、单数据划分，不能证明稳定性或跨数据集泛化；不读取 test。4k learned 末段变化很小，但预算仍不足以证明充分收敛。固定前四张验证图的七组预测已重算并与逐图混淆逐值一致；完整输出、所有节点的逐图混淆/检查点、曲线、逐类 IoU/precision/recall/预测面积和审计保存在服务器独立目录，不上传数据和权重。

复跑：在本目录 checkout 下使用服务器独立目录的 `run_formal_cuda_20261030.sh`，先指定确实空闲设备和新的 `OV_OUTPUT`；详细固定参数见其脚本和每组 `protocol.json`。检查 CPU/其他用户任务后运行，不覆盖现有输出。原始结果路径 `/home/wangchengjun/huangziyi/ov_routes_acceptance_20261029/formal_cuda_20261030`，分析 `analysis_cuda_20261030`。测试：`python -m pytest -q research/ov_routes/test_routes.py`。

参数更新幅度核验：以同一预训练 checkpoint 的可训练参数为起点，16k 终点的整体 L2 更新范数分别为 ERM 10.9677、joint 8.8324、meta 8.8345（起点参数范数 91.7684；joint 与 meta 终点间 L2 为 0.1449）。meta 与 joint 的更新幅度几乎相同，较 ERM 更小；这不能证明小学习率本身等价于其效果，仍需匹配学习率对照才能作因果判断。数据保存在 `analysis_cuda_20261030/parameter_update_norm.json`。
