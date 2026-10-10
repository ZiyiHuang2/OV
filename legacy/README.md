# 原工程归档

这里保存 CAT-Seg 继承文件，方便追溯与单独研究原训练流程。
**本项目的遥感实验使用根目录 `train_ovrs.py`，无需运行本目录脚本。**

| 文件 / 目录 | 内容 |
|---|---|
| `train_net.py` | 原 CLI；通用 Trainer 已提取到 `cat_seg/engine/trainer.py` |
| `plain_train_net.py` | 原独立训练与评价实现 |
| `run.sh`、`eval.sh` | 原自然图像训练与多数据集评测脚本 |
| `configs/vitl_336.yaml` | 未纳入本项目验证路径的 ViT-L 配置 |
| `demo/` | 原演示；包含旧版 MaskFormer 接口，未适配为 OVRS 工具 |
| `datasets/` | 自然图像数据准备脚本与旧说明 |
| `assets/`、`README_UPSTREAM.md` | 原图示与项目说明 |

如确实使用归档脚本，从仓库根目录执行，例如：

```bash
python legacy/train_net.py --config-file configs/base/clip_vitb16.yaml --help
```

入口与包的相对导入已更新；这些历史路径没有承诺与遥感主入口同等的运行验证。
数据文件仍由 `DETECTRON2_DATASETS` 或原脚本指定，类别词表保留在根目录 `datasets/`。
