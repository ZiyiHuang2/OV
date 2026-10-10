# 代码结构与修改入口

## 主运行路径

所有命令在仓库根目录执行。`train_ovrs.py` 负责解析实验配置、注册遥感
manifest、记录运行信息以及启动训练/评测。通用训练策略位于
`cat_seg/engine/trainer.py`，不会从历史 CLI 导入训练器。

模型路径为 `CATSeg → CATSegHead → CATSegPredictor → Aggregator`。
CLIP 的图像/文本表示与动态词汇缓存在 predictor 及 CATSeg 中管理；新增三个
模块集中在 `cat_seg/modeling/modules/`，由 `transformer/model.py` 接入。

| 位置 | 负责的内容 | 常见修改 |
|---|---|---|
| `cat_seg/modeling/modules/guidance.py` | M1：共享视觉引导的 LSK / 普通卷积适配 | 核大小、膨胀率与残差更新 |
| `cat_seg/modeling/modules/fusion.py` | M2：simple / EPL / gated 分支融合 | 两路输出的融合机制 |
| `cat_seg/modeling/modules/decoder.py` | M3：mean / weighted CAD | 类别摘要与 guidance 细化 |
| `cat_seg/modeling/modules/common.py` | 共用 GroupNorm 构造 | 归一化策略 |
| `cat_seg/modeling/transformer/model.py` | AggregatorLayer、Aggregator、Up | 串行/并行计算与解码阶段的接入位置 |
| `cat_seg/modeling/transformer/cat_seg_predictor.py` | 文本编码、缓存与聚合器创建 | 动态词汇及配置参数传递 |
| `cat_seg/engine/trainer.py` | 优化器、学习率、mapper、通用 evaluator | 训练策略 |
| `train_ovrs.py` | RS evaluator、严格评测加载与命令行 | 运行协议与入口 |
| `tools/prepare_rs_dataset.py` | 标签转换、seen/unseen 与 manifest | 新遥感数据集接入 |

M1 处理 `[B,C,H,W]` 共享 guidance；M2 处理 `[B,C,K,H,W]` 的两路输出；
M3 从 `[B*K,C,H,W]` 解码特征获得共享摘要。它们没有类别专属参数。
解码器收到的是真实候选类别，class attention 的内部 padding 不进入 CAD。

## 配置如何继承

基础配置逐级继承：`configs/base/catseg.yaml` →
`configs/base/clip_vitb16.yaml` → `configs/ovrs/base.yaml` → 具体实验 YAML。
共享设置只改基础层；单项消融在具体 YAML 中覆盖 `MODEL.OVRS`。
`--experiment` 的完整对照表见 [运行指南](USAGE.md#5-消融配置)。

不需要为每种组合复制训练脚本。需要短期调整时可以使用入口末尾的 Detectron2
opts，例如 `MODEL.OVRS.CAD_POOLING weighted`，实际配置会写入输出目录。

## 文档与历史代码

- [USAGE.md](USAGE.md)：数据格式、训练、评测和消融命令。
- [INSTALL.md](INSTALL.md)：环境与编译要求。
- [DESIGN.md](DESIGN.md)：`full_selective` 的机制与取舍。
- [AUDIT.md](AUDIT.md)：初版三个模块的审查。
- [VALIDATION.md](VALIDATION.md)：已验证范围与限制。
- [UPSTREAM.json](UPSTREAM.json)：依赖来源和固定版本。

`legacy/` 集中保存原入口、演示、自然图像数据准备脚本和 ViT-L 配置。
这些文件用于追溯原工程，不纳入本项目稳定的遥感运行路径。`datasets/` 仍保留
继承的类别 JSON，避免改变原注册逻辑及基础配置的资源路径。模型核心和 CLIP
代码保留在 `cat_seg/`，不需要从 archive 加载。

## 本次整理的兼容边界

`train_ovrs.py` 的 CLI、experiment 名称和 `configs/ovrs/*.yaml` 路径保持不变。
三个模块移动到独立文件，但模型属性名、参数键、前向计算与随机初始化顺序
保持不变，已训练权重可以按原 experiment 严格加载。

Python 模块导入位置改为 `cat_seg.modeling.modules`，通用 Trainer 导入位置改为
`cat_seg.engine`。原 `ovrs_modules.py` 和根目录 `train_net.py` 没有保留转发包装。
如自行编写过直接导入它们的脚本，需要更新这两个 import。

依赖清单位于 `requirements/runtime.txt`、`requirements/test.txt`；根目录
`requirements.txt` 仍是标准安装清单入口。安装脚本与 CI 已同步更新。
