# 语义与结构研究原型 v2

本目录是独立的冻结特征解码原型。代码可用于受控工程 pilot；不是 HG 官方复现，不包含论文级效果结论。当前支持图像/标签 → 本地编码器 → 特征缓存 → 匹配训练 → 固定 checkpoint 评测。没有官方 HG 权重迁移、知识图谱、伪标签或端到端编码器训练。

## 先决定实验问题

一次只选一个事先写明的假设，并指定预期改善的类别或失败组。默认基线为 mean+geometry。

| 实验因素 | 对照 | 干预 | 保持相同 |
|---|---|---|---|
| 语义选择 | 描述均匀平均 mean | 局部 cosine 权重 adaptive | 描述集合、最终单位范数、参数、初始化、结构路径 |
| 空间传播 | 纯结构邻域权重 geometry | 增加类别证据相似性 conditioned | 邻域、直接结构输入、直接语义输入、细化层、参数 |

identity 是可选的不传播对照，不等于没有结构输入。所有模式都将 state、direct evidence、projected structure、transported state 输入相同的四路细化层。选择温度和类别条件系数为固定超参数，无额外路由参数；不可在目标 unseen GT 上选它们。

动态选择可能放大错误语义；结构相似性不保证语义一致；后续卷积不具备凸组合约束。一个对照胜出只能支持该条件下的干预收益，不能证明唯一根因。只有一条描述时两种语义模式应等价。只用一个 prompt 的缓存无法检验多描述选择假设。

## 环境与测试

在仓库根目录安装适配服务器的 torch/torchvision 配对版本，其余依赖见本目录 requirements.txt。CPU 核心训练不依赖 Detectron2。

```bash
cd research/semantic_structure
python -m unittest -v test_model test_extract
```

本次实际验收环境为 Python 3.12、torch 2.14.1+cpu、torchvision 0.29.1+cpu。requirements 给的是接口最低范围，不代表所有旧版本均经过测试。GPU 未在本次环境运行；先执行小预算服务器验收。

## 从真实图像提取缓存

复制 manifest.example.json 为实际 train/validation manifest，明确原始标签到连续类别 ID 的映射。例子的类别和 label_map 仅展示格式，不能直接套用实际数据。描述是最终送入文本编码器的完整字符串；会去掉大小写不敏感的精确重复，不自动生成描述。samples 路径相对 manifest 位置；同一数据集 train/validation 类别、描述和映射必须一致。

本版提取器明确支持：
- 本仓库 model_vpt.py 的 CLIP ViT dense=True 路径，使用最后投影后的 patch tokens，排除 CLS。此路径来自现有 CAT-Seg 代码，不能称为 HG 图像特征。
- OpenAI 格式、未拆分 QKV 的本地 CLIP TorchScript 或纯 state_dict；严格加载，缺键报错。不自动识别任意封装 checkpoint。
- torchvision ResNet-50 纯 state_dict，layer3/layer2 作为粗到细结构特征；严格加载，固定 ImageNet 输入归一化。这是具体的工程适配器，不是 RS-OVSGSeg 的 RSIE。使用 ImageNet 权重时，只能报告 ImageNet 结构分支结果。不能通过改 source 字符串伪装成遥感权重。

所有权重都必须已存在于本地。不会自动下载，不会缺权重后随机初始化继续运行。DINO/Swin/其他预处理必须另写并验收适配器，不能直接套用本入口。

```bash
python extract_features.py \
  --manifest /data/protocol/train.json \
  --clip-checkpoint /weights/ViT-B-16.pt --clip-format torchscript \
  --clip-source 'OpenAI CLIP ViT-B/16' \
  --structure-encoder resnet50 --structure-checkpoint /weights/resnet50.pth \
  --structure-source '实际权重的训练来源与版本' \
  --size 384 --device cuda:0 --output /data/cache/train.pt
```

对 validation manifest 重复同一命令，输出不同文件。编码器使用同一张完整 RGB 图像的确定性双三次缩放，分别执行各自归一化，不做独立裁剪。当前输入为正方形，size 必须是 CLIP patch size 的整数倍；非正方形原图会改变长宽比，应在协议中说明，或预先生成经过审计的方形 tile。标签保持原始尺寸，模型输出恢复到标签网格。单个缓存内原始 mask 尺寸必须一致；不会偷偷缩放 mask。

提取使用 eval/no_grad 且冻结全部参数。记录权重、源码、图像、mask、manifest 的 SHA256 和预处理。图像 SHA256 是文件字节哈希，不能检测不同压缩版本或地理重叠；地理独立性需数据协议另外保证。缓存整体驻留 CPU 内存，只适合有界 pilot；先用少量样本估算容量，不把全数据集直接塞进一个缓存。

## 最小训练与匹配比较

先执行 2 次实际优化器更新，确认真实缓存、显存和保存重载可用：

```bash
python train_cached.py --cache /data/cache/train.pt \
  --val-cache /data/cache/validation.pt --output /experiments/ss_v2_smoke \
  --semantic mean --spatial geometry --steps 2 --eval-every 2 --device cuda:0
```

通过后选择一条事先定义的干预，示例为结构条件：

```bash
python run_pair.py --cache /data/cache/train.pt \
  --val-cache /data/cache/validation.pt --output /experiments/ss_v2_spatial \
  --intervention spatial --hypothesis '填入预先定义的失败组、预期变化和否定条件' \
  --steps 100 --batch-size 2 --width 64 --seed 42 --device cuda:0
```

semantic 比较 mean/geometry 与 adaptive/geometry；spatial 比较 mean/geometry 与 mean/conditioned。不会自动组合、搜索参数、选 target checkpoint 或触发大预算。100 步只是工程 pilot 默认值，不能据此否定或证明方法。

run_pair 检查初始参数哈希、采样轨迹、更新次数、输入缓存、优化控制和参数量一致。预训练编码器费用不计入 decoder 参数量，不能把该数字当作完整系统复杂度。

## 训练预算、续训和输出

steps 现在表示真实 optimizer 更新次数。训练前排除整张标签均为 255 的样本并记录数量，使用有放回的独立随机采样；局部 ignore 像素仍然保留。optimization_step 对全 ignore 批次直接跳过整个优化器步骤，保留 3542de2 的修复意图，避免 AdamW 权重衰减或动量改变参数。

```bash
python train_cached.py --cache /data/cache/train.pt \
  --val-cache /data/cache/validation.pt --output /experiments/ss_v2_resumed \
  --semantic mean --spatial geometry --steps 200 --device cuda:0 \
  --resume /experiments/ss_v2_smoke/final.pt
```

续训必须使用相同缓存、模型设置、batch、seed、学习率、源码和设备；steps 是累计目标更新数。输出目录必须是新的。checkpoint 保留 optimizer、CPU/CUDA RNG、独立 sampler、更新数和采样历史；不会覆盖已有实验。每次运行输出 protocol.json、train_metadata.json、train.jsonl、周期 latest.pt、final.pt、验证 JSON 与 summary.json。最终 checkpoint 会独立重载并检查固定输入输出逐值一致。固定 final 为主要读出，不保存 target 最优点。

训练中 validation 必须是同源、相同类别和文本的 validation。跨域或新类别只在训练结束后运行：

```bash
python evaluate.py --checkpoint /experiments/ss_v2_spatial/baseline/final.pt \
  --cache /data/cache/target_research.pt --output /experiments/target_baseline.json --device cuda:0
```

评测验证 encoder/preprocessing 兼容并拒绝已知训练图像重叠；支持不同类别数。split 必须是 validation 或 research，本 pilot 拒绝 test。split 字段是协议声明，不能替代人工确认数据集身份，禁止把封存 test 改名 research。

指标为 0–1 比例：总/逐图混淆矩阵、逐类 IoU、precision、recall、GT/pred 像素数。零 union 类不进入 mIoU，缺分母指标为 null；误检存在而 GT 缺失的类仍计入 IoU。不同 benchmark 的 absent-class 定义可能不同，不直接混表。per_image 结果可用于预定义的类别/场景分组；本版没有自动目标实例尺寸统计或完整地理 bootstrap，也不自动宣称 failure group 改善。

## v1 迁移

模型和缓存协议都升级为 v2。旧 fusion/transport 配置、旧 checkpoint 和空 metadata 缓存明确报错，不自动转换。旧研究结果保留原版本身份，不与 v2 混用。要复用旧缓存，先核实来源、图像身份、文本及编码器，再显式生成完整 v2 元数据，禁止补猜身份。

## 当前证据边界

本次通过的测试覆盖合成特征、临时图像/合成编码器权重、真实 PyTorch 优化、续训和文件读写。没有本地真实遥感数据、预训练权重或 CUDA，因此没有真实数据集成绩、冻结预训练编码器表现或 GPU 资源结论。服务器下一步是用真实权重和小型数据清单验收，再决定是否执行一条匹配 pilot。

独立原型的收益不能证明旧 DLRSD unseen 退化的根因，也不能证明改进了 HG。要研究 HG，仍需其官方实现、权重和匹配协议；不要因为此目录已存在就默认扩展模型。

参考：KG-OVRSeg (10.1016/j.isprsjprs.2025.11.029)、HG-RSOVSSeg (10.3390/rs18020213)、RS-OVSGSeg (10.1109/TGRS.2026.3726176)。本目录新代码不复制其实现；图像适配器调用仓库已有 CLIP 代码，版权沿用该代码来源。


## Preserved v1 server evidence

The original requested mean/adaptive x fusion/transport experiment is preserved
in [archive/v1_pilot_20261028](archive/v1_pilot_20261028/EXECUTION_20261028.md).
That frozen implementation, protocol and results are v1 evidence, not v2 results.
It is retained to reproduce measured checkpoints; v2 does not load those checkpoints.
The server loss fix flattens pixels for deterministic CUDA NLL and validates labels.

## Server RSIB adapter and expanded v2 experiment (2026-10-28)

`extract_rsib.py` is a separately audited OpenCLIP ViT-B/16 + RSIB ViT-B/8
adapter. It requires local identified weights, the original train/val manifest,
a fixed protocol, and the RSIB implementation SHA256. It extracts native 224
features (14x14 CLIP / 28x28 RSIB), resizes labels to 224 with nearest sampling,
and records schema-2 metadata. This differs from `extract_features.py` and must
not be reported as its ResNet50 configuration. Encoder eval/freeze, strict
backbone loading, native CLIP CLS equivalence, exact image identities, decoded
train/validation disjointness, and unchanged encoder states are checked.

The user authorized expanded training after the initial v1 100-update pilot.
`protocols/dlrsd_v2_fulltrain_20261028.json` fixes 1604 train / 200 validation,
seen-only metrics, seed 42, sampler seed 43, batch 8, AdamW 2e-4, and 4000
updates per arm. Four arms are mean/adaptive x geometry/conditioned. No sealed
test is accessed, no best checkpoint selected, and no prompt is tuned on val.

After extraction into `<experiment>/features`, run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unittest -v test_model test_extract test_gpu
python run_factorial.py --root <experiment>
```

The runner first checks four-image / 20-update real-data engineering acceptance
in every arm, including decreasing fixed-subset loss and exact 10+10 vs 20
continuation. Only then does it start the four matched full-training-set runs,
one process per GPU (B then E on GPU 0, C on GPU 1, D on GPU 2). Confirm these
GPUs are available before invoking it. Protocol, launches, logs, final/latest
checkpoints, periodic per-image/per-class/confusion metrics, and matched-control
checks remain under the independent experiment directory. This is cached,
single-seed, single-dataset seen-only evidence, not end-to-end training or a
cross-dataset open-vocabulary result. See EXECUTION_V2_20261028.md for measured
server results once the run completes.
