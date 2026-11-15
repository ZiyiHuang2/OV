# RCT CUDA validation and 2k exploration

状态：五组真实训练、完整 source dev 评测和核心对照的全量像素转换分析完成。这些结果均为 2,000 步单种子探索，不是 40k 收敛结果。

The upstream experiment source was `bb38fffd79bec940be3fb9d01555f14841060c63`. This work uses the historical OVRSIS95K grouped source train/dev manifest, seed 42, batch 4, FP32, 384 training crop, AdamW, base LR 0.0001 and 2,000-step cosine schedule. All arms select the final checkpoint, evaluate the same 4,700-image source dev set, and use the same full baseline initializer. This is not the historical 40k schedule or an independent external test.

Native A and Competition C completed 2,000 updates and full source-dev evaluation: A mIoU 23.8919, macro F1 34.0792; C mIoU 23.9392, macro F1 34.1250. The C minus A mIoU difference is +0.0473 percentage points. These are single-seed 2k results, not 40k training or evidence that RCT is effective. The full results including per-image RLE and confusion matrices are under `/home/wangchengjun/huangziyi/experiments/rct_2k_bb38fff_20261115/`. Checkpoint SHA-256 values: A `b93407b7bc9b26b09f845775fe5cc704ae6dbe59f99cf5e6eb563289dd4748c3`, C `7bd942f14d0770d31d2549e1fe994f1a7648c270177a163a806e065e1a30367b`.

Full-weight FP32 RCT bypass on one real 35-class image gave max and mean absolute error 0. Real two-update M/V/R smokes had finite gradients, optimizer coverage and actual parameter changes. The individual gate and mix gradient channels, relation input gradients, checkpoint resume with nonzero updates, and repeated eval in one output root were also checked. See `cuda_smoke_report.json` and channel audit JSONs. The small two-image evaluations are engineering checks, not method metrics.

Historical source dev audit reports no exact train/dev image overlap, but geographic independence is unknown and 32 near-duplicate candidate pairs remain unverified. The official val split was excluded due to reported exact train overlap. The 35-class source vocabulary is all seen; unseen mIoU and hIoU are not applicable. DLRSD has repeatedly been used for development and is not a sealed independent test.

## 匹配探索结果

| 臂 | source dev mIoU (%) | macro mF1 (%) | 相对 Native mIoU (百分点) |
|---|---:|---:|---:|
| Native A | 23.8919 | 34.0792 | 0 |
| Competition C | 23.9392 | 34.1250 | +0.0473 |
| RCT-M | 23.9102 | 34.0865 | +0.0183 |
| RCT-V | 23.8675 | 34.0390 | -0.0244 |
| RCT-R | 23.9029 | 34.0868 | +0.0110 |

V–M 为 -0.0427 pp，R–V 为 +0.0355 pp，R–M 为 -0.0072 pp。差距远小于 0.1 pp，仅一个随机种子，不能称为稳定增益。所有 35 类都是 seen；unseen mIoU、hIoU 不适用。mF1 从同一混淆矩阵按类别计算 F1 后宏平均。历史 40k 的 A/C source dev 分别为 41.5383/41.5821 mIoU，是不同预算和调度，仅供背景参考，不可与本轮直接比较。历史 A/C 曲线在 5k 后仍上升，不能声称本轮已收敛。

M 相对 Native 的最大类别 IoU 改善包括 golf field +1.3885 pp，主要下降包括 train station -1.0501 pp。V 相对 M：ground track field +0.2484、dam +0.2360、harbor +0.2268；basketball court -0.6351、overpass -0.6305、stadium -0.6173 pp。R 相对 V：train station +0.9094、basketball court +0.5239；agriculture land -0.4777、ground track field -0.2453 pp。所有类别结果见 `summary/per_class_iou.json`。

最大混淆包括 grass→背景、tree→背景、grass→agriculture land。完整混淆矩阵显示 V 的 grass→agriculture land 为 84,983,160 像素，R 为 87,239,854；绝对数受类别支持量影响。M–Native 对齐像素比较中，M 纠正 12,553,898 像素、使 12,528,150 个原正确像素变错；V–M 分别为 7,839,968 和 7,684,272；R–V 分别为 13,466,777 和 14,651,155。R 宏平均 mIoU 略升，但净正确像素减少 1,184,378，反映类别权重不同。使用逐图 RLE 重建的真实预测标签与 ground truth，并非按 `per_image.jsonl` 的面积推断转换。

固定 source dev 前 100 图的粗特征网格事后诊断中，按 GT 类与最强竞争类得分绝对差的中位数划分高/低竞争：V–M 在高竞争区纠正 49 格、退化 61 格；R–V 在竞争类空间切换区域纠正 114 格、退化 108 格，未切换区域纠正 61 格、退化 59 格。这是标签辅助的描述性诊断，并未直接测量注意力因果作用，不足以证明竞争特征消息或邻居选择有效。

## CUDA、优化与开销

真实 Trainer 短训中，M 的 mix 通道为设计性零梯度，V/R 的 gate 和 mix 通道非零；同一竞争类关系输入在 M/V 为零而 R 非零。每臂注册 8,708 个参数，但有效分支不同，不能把注册数量视为有效参数数量。现有 CPU 测试 36 项通过。

训练遥测抽样（第 0→1900 步）：V 的 alpha 0.0100→0.00247、R 0.0100→0.00323，更新 RMS 分别约 0.000432→0.000111、0.000423→0.000142；M alpha 约 0.0100→0.00997。V/R 在这个预算内的残差被明显缩小，但这本身不能确定原因。各臂 gate 均值约 0.525，记录点上的 gate 关断/饱和比例均为 0；mix 均值约 0.495，M 的 mix 记录不用于前向。margin 平均绝对值约 0.15。曲线见 `training_curves_v2.png`。

| 臂 | 峰值显存 MiB | 中位训练秒/步 | 评测纯模型秒/图 |
|---|---:|---:|---:|
| A | 12,354 | 0.2988 | 0.0272 |
| C | 12,980 | 0.3120 | 0.0296 |
| M | 12,749 | 0.3107 | 0.0522 |
| V | 12,871 | 0.4020 | 0.0385 |
| R | 12,871 | 0.3169 | 0.0300 |

这些是共享 GPU 下的观察值；M 评测与 R 训练同卡并发、V 与其他用户任务共卡，不能把时延差直接归因于结构。纯模型时间来自 Detectron2，包括同步及测量开销，并非隔离基准。

## 产物与判断

持久化服务器目录：`/home/wangchengjun/huangziyi/experiments/rct_2k_bb38fff_20261115/`；每臂包含配置、训练日志、metrics、最终 checkpoint、SHA-256、完整混淆矩阵、逐图记录和 RLE 预测。五个 checkpoint 均与临时原文件逐一校验一致。原始源码提交为 `bb38fff`；代码与报告通过独立 Draft PR 跟踪，历史 checkpoint、数据和日志未删除或覆盖。本次根据用户要求让 M 与 V 并行，R 训练与 M 评测在 GPU 0 同时运行。

完整串行复现入口是 `run_rct_2k.sh`；其每臂命令、路径和超参数与本轮相同。实际并行运行时先暂停原串行脚本的排队 shell（不中断正在运行的 M），通过 `run_rct_2k_gpu1.sh` 在 GPU 1 启动 V；然后暂停该脚本的排队 shell（不中断 V），在 GPU 0 以串行脚本中的 R 命令启动 R。M 和 V 完成后终止两个暂停的排队 shell，避免重复运行；各训练进程均完成且保留完整日志。

V/R 作为竞争特征增量的总体信号尚不可信，R 相比 V 的微小正差不能排除噪声、类别取舍或不同训练轨迹，且 V/R 残差被优化压低。**暂不进入 40k**；更长预算可能改变结论，但当前不足以优先消耗完整预算。若继续，应先冻结协议，在匹配训练调度下增加独立种子并检查更新幅度及高竞争/切换区域，再根据预先明确的信号决定是否升级；不更换数据集、评测集或 checkpoint 规则。没有外部泛化结论。
