# OV Competition V1｜最小实现（CPU 单元验收版本）

这是针对 `ZiyiHuang2/OV` 中 CAT-Seg 宿主的**最小计算原型**，并非已完成服务器整合或效果训练。

## 文件

- `competitive_update.py`：B/C 共用成对关系、门控与残差，区别只在条件统计；原生特征更新保留。
- `install.py`：在**完整预训练权重加载后、optimizer/DDP 构造前**注册 B/C；A 不增加模块。
- `patch_repo.py`：安全匹配现有 `Aggregator.forward` 的唯一代码锚点，仅在显式 `--apply` 时编辑**本地**源码；只对原生 `serial` 第一层生效。
- `test_competitive_update.py`：动态词表、重排、padding、空间边界、方向性、梯度、严格旁路测试。
- `METHOD_SPEC.md`、`RESOURCE_AUDIT.md`：冻结计算对照和待服务器核查清单。

## 推荐接入方式

将整个目录复制到 OV 仓库 `research/ov_competition_v1/` 后：

```bash
python research/ov_competition_v1/patch_repo.py --check
python research/ov_competition_v1/patch_repo.py --apply
python -m pytest -q research/ov_competition_v1/test_competitive_update.py
```

仅在服务器/研发分支上运行上述修改，**不要直接在已经运行其他实验的工作树中改动**。运行后检查 `git diff`；再次应用会拒绝，避免重复改写。

示意接入（**需要服务器完成 OVRSTrainer 的正式装载和优化器顺序整合**）：

```python
from research.ov_competition_v1.install import install
# model 已经加载共同完整预训练权重；这里还没有创建 optimizer/DDP
model = install(model, 'B')  # A/B/C, B=同类条件, C=跨类别竞争条件
# 随后构建 optimizer，务必覆盖 competition_update.* 的全部参数
```

**重要**：现有 `train_ovrs.py` 仍不知道 B/C。`patch_repo.py` 只接入前向，不会自动创建训练臂、优化器参数组、checkpoint 复现规则、数据注册或启动训练。因此，当前包不可直接用旧训练脚本号称完成 A/B/C 正式 CUDA 验收。

CPU 测试：PyTorch 2.10 CPU、Python 3.13；14 项自动测试通过，额外完成 B/C 在 `B=1,D=128,K=35,H=W=24` 形状下的前向/反向检查。所有测试在合成张量上运行，只证明局部实现逻辑。用户服务器是另一套环境，必须重新实际验证。
