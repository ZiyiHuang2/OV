# CLIP + DINOv3 双分支分割

九组 C-series 的正式实验采用 batch 24、15 轮。全量 iSAID 验证结果、分支分析、资源用量及原始报告见 [C-series 正式实验归档](results/isaid_b24_e15/README.md)。下文记录早期单轮双分支基线。

冻结 OpenAI CLIP ViT-B/16、DINOv3 ViT-B/16 Web 和 CLIP 文本编码器。输入缩放到 384 × 384，两路视觉特征在相同的 24 × 24 patch 网格对齐。拼接后的 1,280 维特征经线性层投影到 512 维，再与冻结的 CLIP 文本向量计算余弦相似度。对上采样后的像素 logits 使用交叉熵。双分支投影初始化为 `[I, 0]`，起始输出与 CLIP-only 一致。该实验独立于 CAT-Seg 和 Route B。

## iSAID 结果

采用发布的 OVRSISBench 划分，训练 18,076 张，验证 6,363 张。评估 15 个前景类，忽略标签 `255`。投影层通过全量训练集一次，共 2,260 步，AdamW 学习率 `1e-3`，有效 batch 8，seed 42。

| 模型 | 训练参数 | mIoU |
| --- | ---: | ---: |
| CLIP-only，零样本 | 0 | 12.10% |
| CLIP-only，训练投影 | 262,656 | 57.58% |
| CLIP + DINOv3-B，训练投影 | 655,872 | 74.28% |

双分支比训练投影的 CLIP-only 高 16.70 个百分点。两组分别在完整验证集上测得 643 和 972 MiB PyTorch 峰值已分配显存。在共享 RTX 4090 上交替运行同一批 512 张验证图，batch 1 平均 GPU 推理耗时分别为 36.46 和 103.52 ms/图。耗时覆盖 GPU 图像缩放、特征提取、相似度、上采样和 argmax。

## 环境与数据

从仓库根目录进入 `research/clip_dinov3`。实跑使用 Python 3.10、PyTorch 2.5.1+cu121、FP32、关闭 TF32。先在对应环境安装 CUDA PyTorch 和 torchvision，再准备其余依赖及固定版本源码。

```bash
cd research/clip_dinov3
python -m pip install numpy pillow ftfy regex tqdm
mkdir -p _deps
git clone https://github.com/openai/CLIP.git _deps/CLIP
git -C _deps/CLIP checkout --detach d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
git clone https://github.com/facebookresearch/dinov3.git _deps/dinov3
git -C _deps/dinov3 checkout --detach 6876159a11b4df116f30f667f8c9888617df0751
```

准备 [OpenAI CLIP](https://github.com/openai/CLIP) 的 `ViT-B-16.pt` 与 [Meta DINOv3](https://github.com/facebookresearch/dinov3) 的 `dinov3_vitb16_pretrain_lvd1689m.pth`，运行时会校验两个文件的 SHA-256。已有正式 iSAID 清单可以原样传给 `--manifest`。如需从 [OVRSISBench iSAID.zip](https://huggingface.co/datasets/xiximilu/OVSISBenchDataset/blob/1a29a2e77a76ddb263e1c985452bbcc2bc7f8d91/OVSISBenchDataset/iSAID.zip) 生成清单，指定包含 `iSAID_split/{train,val}/{images,D2masks}` 的数据根目录。脚本沿用发布划分，按文件名排序和配对。

```bash
out=../../output/clip_dinov3
python prepare_isaid.py --dataset-root /path/to/extracted/isaid --output "$out/data"
manifest="$out/data/manifest.json"
clip=/path/to/ViT-B-16.pt
dino=/path/to/dinov3_vitb16_pretrain_lvd1689m.pth
common=(--manifest "$manifest" --clip-weights "$clip" --size 384)
dual=(--dino-weights "$dino")
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 复跑

双分支 CUDA 前检用训练集前 16 张做两步，核对初始输出、冻结状态和 DINOv3 投影权重的梯度。后续 CLIP 投影对照可读取前检报告复核数据顺序。

```bash
python run.py train "${common[@]}" "${dual[@]}" --batch-size 8 --seed 42 \
  --limit 16 --max-steps 2 --audit --output "$out/preflight"
python run.py eval "${common[@]}" "${dual[@]}" --batch-size 1 --limit 2 --warmup 5 \
  --checkpoint "$out/preflight/projection.pt" --reference "$out/preflight/roundtrip.pt" \
  --output "$out/preflight_eval"

python run.py train "${common[@]}" "${dual[@]}" --epochs 1 --batch-size 8 \
  --lr 1e-3 --seed 42 --workers 4 --output "$out/train_dual"
python run.py eval "${common[@]}" "${dual[@]}" --batch-size 1 --workers 2 --warmup 30 \
  --checkpoint "$out/train_dual/projection.pt" --output "$out/eval_dual"
python run.py eval "${common[@]}" --batch-size 1 --workers 2 --warmup 30 \
  --output "$out/eval_clip_zero"

python control.py train "${common[@]}" --reference-dino "$dino" \
  --reference-train-audit "$out/preflight/report.json" --batch-size 8 \
  --encode-batch-size 32 --lr 1e-3 --seed 42 --workers 4 --output "$out/train_clip"
python control.py eval "${common[@]}" --batch-size 1 --workers 2 --warmup 30 \
  --checkpoint "$out/train_clip/projection.pt" --output "$out/eval_clip"

python paired_latency.py --manifest "$manifest" --clip-weights "$clip" --dino-weights "$dino" \
  --checkpoint "$out/train_dual/projection.pt" --clip-checkpoint "$out/train_clip/projection.pt" \
  --output "$out/paired_latency.json"
```

CLIP-only 对照的视觉前向 batch 为 32，每 8 张更新一次投影。它在 CPU 重放双分支的初始化以重建数据顺序，优化器只接收 CLIP 特征和标签。评估 `report.json` 包含样本顺序摘要、混淆矩阵、逐类 IoU、GPU 耗时和峰值显存。两个训练检查点均只保存各自的投影层和元数据。

## 留类与 40 轮对照

从 `research/clip_dinov3` 运行下列命令。每个划分、seed、训练预算使用自己的输出目录，且前检与训练使用相同配置。留类入口接收已核验的原始完整 iSAID 清单，前检在输出目录中生成留类清单与像素审计。训练使用 12 类，源域验证用完整 15 类。H1 留出 `storage tank`、`tennis court`、`bridge`，H2 留出 `plane`、`basketball court`、`roundabout`。

```bash
export CSERIES_HELD_SPLIT=H1 CSERIES_SEED=42 CSERIES_EPOCHS=15
export CSERIES_VARIANTS=C0,B0 CSERIES_UNATTENDED=1
bash run_isaid_suite.sh preflight "$manifest" "$clip" "$dino" "$held_output" 24
bash run_isaid_suite.sh main "$manifest" "$clip" "$dino" "$held_output" 24

unset CSERIES_HELD_SPLIT
export CSERIES_VARIANTS=B0,C1 CSERIES_EPOCHS=40
bash run_isaid_suite.sh preflight "$manifest" "$clip" "$dino" "$long_output" 24
bash run_isaid_suite.sh main "$manifest" "$clip" "$dino" "$long_output" 24
```

需要两张卡时，将 `CSERIES_GPU_IDS` 设为逗号分隔的两个 GPU UUID。`CSERIES_UNATTENDED=1` 下，单组失败后该组保留日志与状态文件，同队列后续模型继续。修复故障后用相同命令重新进入，已完成的组复用报告，中断的训练从最近兼容的 `state.pt` 恢复。40 轮使用独立的优化器调度，从原始初始化启动。

40 轮训练结束后，统一评估每两个 epoch 的轻量导出，以及第 15 轮，写入固定 512 图开发集曲线。保留第 15、30、40 轮和开发集最佳导出。默认按开发集 GT 出现类别的 mIoU 选最佳，留类按相同口径的 seen/held 调和均值选最佳，同分保留较早的 checkpoint。最终完整 val 独立报告，像素交叉熵按验证词表和非忽略像素计算。设置审计后的 `EXTERNAL_MANIFEST` 再用相同配置运行 `external`，可评估长训的第 15、30、40 轮。`KEEP_FINAL_STATE=1` 时保留已完成组的最后恢复状态。

补 C1 的两格匹配消融时，设置 `CSERIES_VARIANTS=E01,E02`。E01 使用多层语义残差和双线性上采样，E02 使用末层语义种子和 C1 的引导解码。复用历史 B0/C1 结果时，使用全词表、batch 24、15 轮、seed 42 的同一 iSAID 清单和权重版本。[四格对照与依据](C_SERIES_DESIGN.md#c1-factorial-controls)
