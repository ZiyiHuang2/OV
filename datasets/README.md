# 继承的类别词表

本目录只保留 CAT-Seg 原有的自然图像类别 JSON，供基础配置与原数据注册逻辑使用。
它不包含实际训练图片、标注或遥感数据。

遥感数据通过 `tools/prepare_rs_dataset.py` 准备，建议输出到本地 `data/<dataset>/`。
类别 JSON 与 split 由你明确提供，训练入口会用 manifest 覆盖原词表配置。
详细格式见 [运行指南](../docs/USAGE.md#3-接入-dlrsd-或其他遥感数据)。

原自然图像数据准备脚本及说明已经移到 [legacy/datasets](../legacy/datasets)。
