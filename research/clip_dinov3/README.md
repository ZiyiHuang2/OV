# CLIP + DINOv3 双分支分割

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
