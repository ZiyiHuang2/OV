# OV routes 执行进度 — 2026-10-29T22:09:48.436576+08:00

本报告为进行中记录，A 16k 三组、B learned 4k、CUDA 真实验收均未完成；不得将 CPU 小样本工程检查当作完整研究结论。

仓库 feat/ov-adaptation-reobserve 原始 SHA `d17283305acae111ab0970bc62c5ef4a953180bf`，实际代码修复及已推送 SHA `9f890cfd4cb747c558cbf33bab8d975131f80ac1`。服务器独立 checkout 在本目录 `checkout/`。

## 已完成工程验收

- 原测试 14 项通过；修复后 15 项通过（`cpu_tests_deterministic.log`）。设备修复：CATSegPredictor 从 cfg.MODEL.DEVICE 接收 CLIP 初始设备，避免 CPU 请求误占 CUDA；补全 IoU/precision/recall/GT 与预测面积；预检显式核验 support 更新和 query 外层梯度，启用确定性算法与 cuBLAS 配置。
- 完整官方 ViT-B CAT-Seg 权重 `~/huangziyi/open-ov/asset_store/checkpoints/cat_seg/model_base.pth`，SHA256 `a201a50f94f4c04196e3083773832b736c2b8b81d27ea2c59cc13af3aff05760`；先前服务器复现报告标识其为官方模型。checkpoint 无嵌入配置元数据，显式允许 metadata 缺失但严格检查全部张量；`strict_weight_audit.json` 中 484/484 个张量键、形状和值均一致，CLIP 可训练参数数为零，非 CLIP 可训练参数 4668641。对应配置 `checkout/configs/base/clip_vitb16.yaml` 与既有 CAT-Seg ViT-B 配方相符。官方 README 提供同名权重链接，未重新下载验证远端文件字节。
- `preflight_cpu_fixed_full` 为修复后的完整 1604/200 manifest CPU 预检，真实 support 梯度与 query 梯度均非零；早期 `preflight_cpu_visible_none` 为修复前的完整 manifest 预检：真实 host/native 输出最大差 `4.1723e-7`，meta 有效梯度，零视图 1 次编码，单裁块 2 次编码。修复后的 `smoke_a_cpu/preflight.json` 在 32 train/4 val 子集上测得 support 和 query 梯度非零、CLIP 冻结；100 个训练 episode 的临时留类和图像分离检查见 `episode_audit.json`。首个误占默认 GPU OOM 的失败日志为 `preflight_cpu.log`，没有训练更新。
- 全部原图均为 256×256；单视图动作的两种裁块尺寸为 128×128、192×192。在这些实际尺寸下与原生模型 CPU 输出一致（最大误差 7.16e-7、2.99e-7、3.88e-7），记录在 `real_native_geometry_audit.json`；此审查没有把 384 尺寸的单点等价性当作自动覆盖裁块。
- A meta 与 B selector 各完成真实完整权重、真实图像的 2 更新 CPU 短训，分别从第 1 步恢复到第 2 步；参数、优化器、随机状态及完整 17 类逐图评测逐值一致。见 `smoke_a_resume_check.json`、`smoke_b_resume_check.json`。这是工程测试，不是效果比较。

## 真实 200 图 CPU 基线：不是 learned 或适配结果

基于同一完整冻结 CAT-Seg、同一 200 张验证图与完整 17 类词表，seed 42。fixed/random 都在同一候选集上额外重新编码一块原图。已核对三组同一图像顺序、GT 像素、逐图混淆矩阵之和及编码次数（`b_cpu_baseline_audit.json`）。

| 策略 | 总 mIoU | seen mIoU | 留类 mIoU | hIoU | 每图编码 | CPU 评测约秒 |
|---|---:|---:|---:|---:|---:|---:|
| 零额外视图 | 29.6293% | 30.4707% | 25.7026% | 27.8843% | 1 | 279 |
| fixed | 29.5038% | 30.4742% | 24.9750% | 27.4519% | 2 | 557 |
| random | 28.9232% | 29.6292% | 25.6282% | 27.4839% | 2 | 576 |

三组的现象是两种额外观察策略在总 mIoU 上均低于零视图，fixed 高于 random 0.5806 pp，但单种子不能证明稳定性。fixed 相对零视图：ship IoU −2.269 pp、dock −2.828 pp、court +4.078 pp。ship precision/recall 从 52.27%/16.19% 变为 51.58%/13.32%，dock 从 56.12%/7.55% 变为 60.97%/4.43%。结果属于下游留类，而非严格 source-unseen；COCO-Stuff 预训练词表包含 airplane 和 boat。原始验证 GT 的所有 17 类参与同一次竞争，未访问封存 test。

learned 尚未训练，不能从本表断定它是否超过 fixed/random；A 三组也没有正式结果。CPU 基线要在 CUDA 真实验收及同设备匹配完成后，才能与 GPU 结果组成正式比较。原始逐图混淆矩阵、轨迹和逐类结果保存在 `b_cpu_real/`，汇总见 `b_cpu_baseline_summary.csv` 和 `b_cpu_per_class.csv`。`b_cpu_qualitative/` 保存按 manifest 固定前四图的零视图/fixed/random 预测、图像和 GT 拼图；已验证重算逐图混淆矩阵与评测逐值相同。

## 阻塞与后续

所有三张 4090 当前均存在其他用户的计算作业（GPU 0 几乎满显存、GPU 1/2 分别留有其他作业的显存且利用率波动），不能为了本实验中断或抢占这些任务。另一已知 A100 SSH 别名连接超时；不假设该服务器具备同一数据或权重。资源状态在变化，自动跟踪会检查 GPU 进程与空闲情况；只有存在确实空闲设备时再运行 CUDA preflight 和真实短训/续训，通过后执行用户指定的 A 16k 三组及 B 4k 与同预算控制。完整目标尚未实现，不能根据 CPU 测试宣布通过 CUDA 或方法有效。

其他身份见 `protocol.json`、`environment_identity.json` 和 `episode_audit.json`；模拟/真实/正式输出在不同目录。相关初始真实 CPU 验收和历史失败均保留。
