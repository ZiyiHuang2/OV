# OV：遥感开放词汇分割

基于 **CAT-Seg / CLIP ViT-B/16**，提供三个可独立消融的改动：
**LSK 共享空间引导 → 并行聚合与融合 → CAD 解码引导细化**。

`full` 是并行 EPL 的借鉴组合；`full_selective` 使用两路门控融合和类别加权
CAD。两者均保留原上采样方式，真实遥感性能仍需在你的数据与协议下验证。

新增 `full_epl_uniform`、`full_class_cad` 与 `full_uniform_class_cad`，分别比较
专家路由、逐类别 guidance 细化及两者组合。接入契约、模块包含关系和实验矩阵
见 [原型说明](docs/PROTOTYPES.md)。原有默认配置保持兼容。

## 快速开始

在仓库根目录执行以下命令。建议使用独立的 Linux / Python 3.10 环境。

```bash
git clone https://github.com/ZiyiHuang2/OV.git
cd OV
conda create -n ovrs python=3.10 -y
conda activate ovrs
bash scripts/install.sh
python -m pytest -q
python tools/smoke_aggregator.py --classes 17 --full --device cuda
```

准备好数据 manifest 后，统一使用一个入口训练或评测：

```bash
# baseline；替换为 full 或 full_selective 即可运行对应组合
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment baseline --num-gpus 1 --output output/baseline

# 评测时 experiment 必须与 checkpoint 匹配
python train_ovrs.py --dataset-manifest data/dlrsd/manifest.json \
  --experiment baseline --num-gpus 1 --eval-only \
  --weights output/baseline/model_final.pth --output output/baseline_eval
```

数据格式、显式类别映射、seen/unseen 划分、合成数据检查和完整消融命令见
[运行指南](docs/USAGE.md)。默认批量为 2；4090 实际显存需要测量，必要时
对所有对比统一指定 `--batch-size 1`。

新 checkpoint 保存并校验关键模型配置；旧权重评测/恢复需显式加
`--allow-unverified-checkpoint`。每次运行的日志及独立评测结果保存在输出目录的
`runs/<运行ID>/` 中，原训练记录不会被评测或恢复覆盖。

## 三个模块在哪里

| 模块 | 实现位置 | 接入方式 |
|---|---|---|
| M1：共享空间引导 | [guidance.py](cat_seg/modeling/modules/guidance.py) | 类别展开前处理视觉 guidance；含 LSK 与普通卷积对照 |
| M2：并行分支融合 | [fusion.py](cat_seg/modeling/modules/fusion.py) | 空间与类别分支读取相同输入，再用 simple / EPL / gated 融合 |
| M3：解码引导细化 | [decoder.py](cat_seg/modeling/modules/decoder.py) | 上采样后通过共享摘要或各类别 cost 细化 guidance |

接入位置集中在 [transformer/model.py](cat_seg/modeling/transformer/model.py)；
开关定义在 [config.py](cat_seg/config.py)，实验组合在 [configs/ovrs](configs/ovrs)。
改变模块实现时，先看 [结构说明](docs/STRUCTURE.md) 和 [设计取舍](docs/DESIGN.md)。

## 目录导航

| 目录 / 文件 | 用途 |
|---|---|
| `train_ovrs.py` | 遥感训练与评测的统一入口 |
| `cat_seg/` | 模型、三个模块、训练策略、数据注册与 CLIP 实现 |
| `configs/base/` | CAT-Seg 与 CLIP ViT-B/16 的基础配置 |
| `configs/ovrs/` | baseline、单模块及组合消融配置 |
| `tools/` | 数据准备、合成数据、聚合器检查与上游一致性检查 |
| `scripts/`、`requirements/` | 环境安装脚本与依赖清单 |
| `tests/` | 模块、标签、文本缓存、权重加载及指标测试 |
| `docs/` | 运行指南、设计审查、验证记录及来源版本 |
| `datasets/` | 继承的自然图像类别词表；真实遥感数据放在本地 `data/` |
| `legacy/` | 原 CAT-Seg 入口、演示、数据准备脚本与历史说明 |

## 说明与来源

已经完成 CPU 单元测试、真实尺寸聚合器反向传播、关闭模块时的上游一致性，
以及合成数据完整训练和严格权重加载。检查范围见 [VALIDATION](docs/VALIDATION.md)；
GPU 显存与真实数据训练收益尚未验证。

CAT-Seg 的 MIT 许可与原 copyright 保留。模块借鉴 LSKNet、PCA-Seg 和
GeoSeg-OV，并做了共享 guidance、GroupNorm 和单路 CAD 等适配；它们不是
原论文完整系统的复现。来源与固定版本见 [UPSTREAM.json](docs/UPSTREAM.json)。
