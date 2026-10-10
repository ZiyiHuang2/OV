# OV：基于 CAT-Seg 的遥感开放词汇分割实验工程

完整代码基于固定 CAT-Seg commit，直接整合三个可消融的改动：
**LSK 共享空间引导、等操作次数的并行聚合/EPL、单路 CAD 解码。**
首版保留原转置卷积；没有加入 DySample、Pi-Seg、FOD、DINO 或 SAM。
借鉴机制已经实现，不代表组合的训练收益已经得到验证。

全局审查后另提供 `full_selective`：保留 LSK，使用分支门控融合和类别加权 CAD。
它是一套独立候选，原 `full` 保留为对照；设计取舍见 [DESIGN.md](DESIGN.md)。

## 1. 安装

建议 Linux、Python 3.10–3.12、单卡 CUDA GPU。默认批量为 2，不能保证所有
词汇规模在 4090 上都能用相同批量；必要时统一把各实验改成批量 1。

```bash
git clone https://github.com/ZiyiHuang2/OV.git
cd OV
conda create -n ovrs python=3.10 -y
conda activate ovrs
bash scripts/install.sh
python -m pytest -q
```

安装脚本使用 PyTorch 2.5.1/CUDA 12.1、torchvision 0.20.1 和固定 Detectron2
源码版本，需要 git、gcc/g++ 编译器。CAT-Seg 路径不调用 Detectron2 的 CUDA
ROI/NMS 算子，因此安装时只编译 Detectron2 CPU 扩展，训练仍在 GPU 上运行。
这免除了 nvcc 依赖，但仍需支持 CUDA 12.1 的 NVIDIA 驱动。
若已有能运行 CAT-Seg 的环境，先备份环境，再对照 requirements-runtime.txt；
不要在重要环境中盲目覆盖依赖。

默认从 OpenAI 下载 ViT-B/16 预训练权重。也可预先将官方 ViT-B-16.pt 放到
~/.cache/clip/；分割 checkpoint 与 CLIP 预训练权重是两种不同文件。

## 2. 先检查工程，不需要真实数据

聚合器检查不下载 CLIP：

```bash
python tools/smoke_aggregator.py --classes 17
python tools/smoke_aggregator.py --classes 17 --full --device cuda
python tools/smoke_aggregator.py --classes 17 --selective --device cuda
python tools/check_baseline_equivalence.py
```

最后一个命令下载固定版本的原 Aggregator 源文件，比较新增模块关闭时的
前向输出与权重键。它不表示训练损失与上游完全相同，见第 6 节。
smoke_aggregator 的显存数字不包含 CLIP，不能作为完整训练显存。

完整管线的合成数据检查：

```bash
python tools/create_toy_dataset.py --output data/toy_source
python tools/prepare_rs_dataset.py \
  --image-dir data/toy_source/images --mask-dir data/toy_source/labels \
  --classes data/toy_source/classes.json \
  --train-list data/toy_source/train.txt --val-list data/toy_source/val.txt \
  --output data/toy
python train_ovrs.py --dataset-manifest data/toy/manifest.json \
  --experiment full --num-gpus 1 --batch-size 1 --max-iter 2 \
  --output output/toy_full DATALOADER.NUM_WORKERS 0 TEST.EVAL_PERIOD 0
```

合成图像只用来检查读图、标签、训练、保存，不能用作性能证据。

## 3. 接入 DLRSD 或其他遥感数据

你需要提供真实类别顺序和划分；本仓库没有猜测 DLRSD 的 17 类顺序、颜色表
或官方 split。图片和 mask 相对路径去掉扩展名后必须对应，例如 a/001.jpg
对应 a/001.png。classes.json 是 JSON 字符串列表，索引就是目标 mask ID。

```json
["building", "road", "vegetation"]
```

上面只是格式示例，不是 DLRSD 配置。train.txt、val.txt 每行是相对文件 stem，
例如 a/001；也支持带扩展名的相对文件名。两份列表必须不重叠。

```bash
python tools/prepare_rs_dataset.py \
  --image-dir /path/to/DLRSD/images --mask-dir /path/to/DLRSD/masks \
  --classes /path/to/classes.json \
  --train-list /path/to/train.txt --val-list /path/to/val.txt \
  --output data/dlrsd
```

默认假设 mask 为二维 ID 图，ID 为 0..K-1，255 为 ignore。P 模式 PNG 按
索引值读取，不把其调色板颜色重新解释成类别。RGB mask 必须提供 --color-map；
非连续 ID 必须提供 --label-map。未知颜色/ID 会报错，不会静默当背景。

颜色映射格式：{"255,0,0": 0, "0,255,0": 1, "0,0,0": 255}。
ID 映射格式：{"1": 0, "2": 1, "0": 255}。

不提供两份列表时可用 --val-fraction 0.2 --seed 42 生成确定性随机划分；
manifest 会标为 random_nonofficial，不能宣称它是数据集官方协议。
准备器会写入新的 mask 和类别文件，不修改原数据。

开放类别划分可加 --seen-ids 0,2：训练类别按这个顺序重映射成 0,1，其他
类别训练像素变为 ignore；评测保留完整类别顺序。默认所有类别均为 seen，
不会自动制造“未见类别”结果。该设置会使用包含未见物体的图像、忽略其像素；
若你的协议禁止这类图像，需要自己提供符合要求的训练列表。

## 4. 训练与评测

```bash
# 串行 baseline
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment baseline --num-gpus 1 --output output/baseline

# 等操作次数并行 + 简单融合
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment parallel_simple --num-gpus 1 --output output/parallel_simple

# 等操作次数并行 + EPL-inspired 融合
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment parallel_epl --num-gpus 1 --output output/parallel_epl

# 三模块完整版本
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment full --num-gpus 1 --output output/full

# 选择性融合与类别加权摘要候选
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment full_selective --num-gpus 1 --output output/full_selective

# 完整模型评测；必须匹配训练时的 experiment
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment full --num-gpus 1 --eval-only \
  --weights output/full/model_final.pth --output output/full_eval
```

从分割 checkpoint 初始化可添加 --weights /path/to/model.pth，各对比应使用
相同初始化策略。新增参数没有对应权重时会随机初始化并显示缺失键；parallel
保留原空间/类别分支的参数名，便于初始化。
真正恢复训练使用相同输出目录和 --resume，它读取 last_checkpoint、优化器和
迭代状态；--weights 单独使用不是恢复优化器。没有已有 checkpoint 的 --resume
会报错。评测要求模型权重完整匹配，选错 experiment 或缺失模块权重会报错。

默认是 attention 微调、80k iterations、2 层聚合、batch=2、AMP、seed=42。
这些只是可运行起点，不自动等于你的论文协议。改变批量、预算或 CLIP 策略时，
对所有实验使用相同设置，例如 --batch-size 1 --max-iter 30000。
标准 Detectron2 opts 放在最后，例如 SOLVER.BASE_LR 0.0001。

输出包含 config.yaml、run_provenance.json、metrics.json、checkpoints。
eval-only 另外保存 evaluation.json，包含整体、逐类与 seen/unseen mIoU。
没有 unseen 类别或对应类别没有有效评测像素时，相应结果是 NaN，不能解释成 0。

## 5. 消融配置

| experiment | 聚合 | guidance 适配 | CAD |
|---|---|---|---|
| baseline | 原串行 | 无 | 无 |
| parallel_simple | 一轮并行/层，简单融合 | 无 | 无 |
| parallel_epl | 一轮并行/层，4 个专家 | 无 | 无 |
| lsk | 原串行 | LSK | 无 |
| conv | 原串行 | 普通卷积对照 | 无 |
| cad | 原串行 | 无 | 第一级 |
| epl_lsk | 并行 EPL | LSK | 无 |
| epl_cad | 并行 EPL | 无 | 第一级 |
| full | 并行 EPL | LSK | 第一级 |
| full_cad_both | 并行 EPL | LSK | 两级 |
| parallel_gated | 分歧条件的两路门控融合 | 无 | 无 |
| cad_weighted | 原串行 | 无 | 第一级，类别加权摘要 |
| full_selective | 两路门控融合 | LSK | 第一级，类别加权摘要 |

普通卷积对照不是严格的等参数/等感受野对照；要作相应机制主张需进一步匹配。
并行版本每层只计算一次空间分支和一次类别分支，与串行计算次数相同，
但 EPL 带来额外参数/计算，因此不能称整个模型等 FLOPs。

## 6. 实现边界

- LSK 只在投影后的共享 guidance [B,C,H,W] 上计算一次；它影响空间
  attention 的 Q/K，不重算原始图文 cost。后续聚合仍随词汇数量增长。
- EPL-inspired 模块采用 GroupNorm，属于对原 PCA-Seg SyncBatchNorm 的
  明确适配；融合保留两路输出均值，再加入可学习残差。残差系数初值为 0.01
  （MODEL.OVRS.FUSION_SCALE），避免随机融合直接覆盖分支输出。它不等价于原
  串行网络，也不是原 EPL 的严格复现；没有 FOD、额外聚合轮数或 DeCLIP。
- CAD 先用原转置卷积上采样，再对真实 K 个类别的解码特征求平均，细化
  单路 CLIP guidance，然后拼接与 DoubleConv。新增细化以可学习小幅残差接入，
  初值为 0.01（MODEL.OVRS.CAD_SCALE）。没有结构教师或置信度估计。
- `parallel_gated` 按位置与类别选择空间/类别分支，另用单个轻量残差细化，
  不使用 EPL 专家。`CAD_POOLING: weighted` 用共享 scorer 对类别摘要加权；
  路由与摘要权重均是学习偏好，不能当作校准后的可靠性或类别概率。
- CAD 对类别顺序不敏感，但增删候选类别会改变共享摘要。原 class attention
  的 padding 是内部处理，送进 decoder 的是实际类别；默认超过 256 类仍沿用
  CAT-Seg 的候选截断策略，不承诺无限词汇同时参与 CAD。
- 文本缓存按词汇和模板区分，训练/评测切换及加载 checkpoint 时清空评测
  缓存，支持训练/评测类别不同。训练时文本编码仍有梯度。
- 所有实验统一修正 ignore 像素的 BCE。上游把这些像素作为全零目标计算
  BCE；本工程完全排除它们。因此这里的 baseline 训练不是原损失的逐字复现。
- 默认实验只支持 OpenAI CLIP ViT-B/16 的已检查数据流。原仓库其他入口与
  配置保留，但未作为本项目的稳定运行路径。

## 7. 代码与引用

主要文件：model.py 集成聚合与 Up；ovrs_modules.py 放三类小模块；
cat_seg_predictor.py 管理动态文本；prepare_rs_dataset.py 管理标签与 split；
train_ovrs.py 复用 CAT-Seg Trainer 的优化器、mapper 与训练循环。

CAT-Seg MIT 许可及原 copyright 均保留，原 README 见 README_UPSTREAM.md。
固定版本见 UPSTREAM.json。借鉴并引用：
- CAT-Seg：https://github.com/cvlab-kaist/CAT-Seg
- LSKNet：https://arxiv.org/abs/2303.09030 ，https://github.com/zcablii/LSKNet
- PCA-Seg：https://arxiv.org/abs/2603.17520 ，https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg
- GeoSeg-OV：https://arxiv.org/abs/2608.10426 ，https://github.com/zzaiyan/GeoSeg-OV

提交前架构审核与取舍见 AUDIT.md；实际检查范围见 VALIDATION.md。未提供用户数据、分割训练权重，
也没有宣称组合训练收益。
