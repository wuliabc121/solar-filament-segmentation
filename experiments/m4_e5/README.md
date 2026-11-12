# M4 精修实验与复现

基于原仓库提交 `76185bdfc296249ebf6f7f776926654b30998a5b`。第一阶段以冻结 E5 候选为 M4 输入。100 组实验先在 A 折用统一预算初筛，其中原 30 组复用核验结果，本轮新增 70 次训练。七组入围后在 E、F 两折各训练四轮。T88 用小目标加权与双倍 Dice，在两折的平均 PQ 为 0.4105。固定阈值 0.35 后使用全部 500 张训练图的 2,317 个正样本重训。

## 方案和结果

| 同一批 103 张留出图 | 输入与处理 | PQ | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| M1 | M1 语义图，阈值 0.50，八邻域连通域面积至少 400 像素 | 0.3665 | 542 | 374 | 455 |
| M2 | M2 语义图，阈值 0.35，同样的连通域设置 | 0.4094 | 654 | 504 | 343 |
| M3 | S 骨干的实例掩膜，使用发布质量头评分 | 0.3829 | 558 | 395 | 439 |
| E5 | M1 和 M2 概率等权融合，在概率至少 0.50 的区域保留 M3 实例掩膜 | 0.3816 | 536 | 350 | 461 |
| E5 → M4 T00 | 每个 E5 候选由 TinyUNet 精修，训练用 BCE 与 Dice，推理阈值 0.50 | 0.4103 | 574 | 312 | 423 |
| E5 → M4 T88 | 同一 E5 候选和网络，训练加权小目标并加倍 Dice，推理阈值 0.35 | 0.4089 | 566 | 320 | 431 |

三条 E5 路线共用 591 个候选。M4 的三个输入通道是灰度图、M1/M2 融合概率和当前候选掩膜。输出贴回原图后按原候选优先级处理重叠。T00 相比 E5 提高 2.87 个 PQ 点。T00 相比同集最强单模块 M2 高 0.09 个点，日期组配对重采样 95% 区间为 −1.65 至 +1.83 个点。

原仓库的随机 141 张开发图上，M1、M2、M3 的 PQ 分别为 0.3789、0.4331、0.3649。该批图有 101 张与 M2 的训练清单重叠。表中的 M2 使用相同发布权重和连通域规则，当前 103 张图与其训练清单互斥。M1、M3 的发布模型在当前留出图上有已知历史重叠。这些分数是冻结候选条件下的本地配对评测。

[M1–M3 原仓库指标](../../docs/random20_metrics.json) · [六方案核验指标](results/aligned_holdout.json) · [100 组完整结果](results/100_trials.csv) · [100 组核验指标](results/100_trials.json) · [逐方向实验](results/targeted.json)

## 原仓库 141 张图的同集对照

沿用原仓库种子 42 的固定清单、228 份标注和 1,541 个真实实例。整图评测逐标注版本用严格大于 0.50 的 IoU 匹配，再汇总 PQ。原仓库 CSV 评分与整图评分在六条路线的 PQ、SQ、RQ、TP、FP、FN 一致。

| 方案 | 原仓库 PQ | 同集复测 PQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| M1 | 0.3789 | 0.3789 | 866 | 583 | 675 |
| M2 | 0.4331 | 0.4327 | 1,052 | 737 | 489 |
| M3＋质量头 | 0.3649 | 0.3649 | 832 | 690 | 709 |
| E5 | — | 0.3650 | 800 | 620 | 741 |
| M2 → M4 T00 | — | 0.4010 | 1,003 | 786 | 538 |
| E5 → M4 T00 | — | 0.3784 | 846 | 574 | 695 |

M4 重新随机初始化，使用冻结上游的真实预测与 GT 构造正样本。原训练分区排除 101 张同图及另外 13 张同日期或同内容图。剩余 386 张图中，M2 路线获得 2,253 个正样本，E5 路线获得 1,788 个正样本。两条路线均固定 T00 训练四轮、阈值 0.50 和最后一步检查点。141 张图的标注仅用于最后的评分。

接在 M2 后，M4 降低 3.17 个 PQ 百分点，日期组配对 95% 区间为 −4.12 至 −2.25，误检和漏检各增加 49。接在 E5 后，M4 提高 1.33 个百分点，区间为 +0.15 至 +2.55，误检和漏检各减少 46。M2 复测较历史记录低约 0.05 个百分点。该复测使用重新核验的缓存，历史混合精度概率缓存未随源码发布。

发布版 M2 的拟合和校准分别覆盖其中 101 张和 20 张。其余 20 张属于原 M2 留出分区，并通过日期和内容隔离核查。在这 20 张上，M2 的 PQ 为 0.3971，接入 M4 后为 0.3703。全 141 张用于冻结发布上游的同集配对对照。对同一批 141 张做严格独立的端到端评测，需要重新拟合和校准所用上游，再比较新上游接入 M4 前后。

[M4 同集对照核验指标](results/random20_aligned.json) · [原仓库 141 张评测清单](../../data/splits/random20_seed42_20261104.json)

已有六组原图预测 JSONL 时，可将原仓库清单的验证 ID 传给通用评分入口。推理和训练产物保留于实验服务器。

```sh
python -c 'import json; print(json.dumps(json.load(open("data/splits/random20_seed42_20261104.json"))["validation_ids"]))' > runs/random20_observations.json

PYTHONPATH=src:. .team-env/bin/python -m m4_filament score \
  --observations runs/random20_observations.json \
  --annotations data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --prediction m1=runs/random20/m1.jsonl --prediction m2=runs/random20/m2.jsonl \
  --prediction m3=runs/random20/m3.jsonl --prediction e5=runs/random20/e5.jsonl \
  --prediction m2_m4=runs/random20/m2_m4.jsonl --prediction e5_m4=runs/random20/e5_m4.jsonl \
  --output runs/random20_score
```

## M2 真实候选的后续精修实验

从冻结的 M2 真实候选和 GT 构造 2,253 个正样本，另构造多标注版本，在标注分歧处保留原 M2 像素。排除原仓库评测及同日期、同内容图后，在 386 张训练图内固定两折，各 69 张。比较四种 M4 训练设置与九条编辑规则，以及单标注和多标注训练的轻量边界模型与六条编辑规则，共 48 组设置。按两折 TP 均不下降、最小 PQ 增益最大选方案。

八次 4090 训练的网络方案未稳定超出 M2。开发折选出单标注训练的轻量边界模型，仅允许两像素内的有限编辑。固定规则和全量训练后复测原 141 张图。

| 评测范围 | M2 PQ | M2 → M4 PQ | PQ 差值（百分点） | 日期组配对 95% 区间（百分点） | TP 变化 | FP 变化 | FN 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 开发 A 折，69 张 | 0.44268 | 0.44306 | +0.038 | +0.024 至 +0.054 | +0 | +0 | +0 |
| 开发 B 折，69 张 | 0.43801 | 0.43958 | +0.157 | +0.031 至 +0.336 | +2 | -2 | -2 |
| 原仓库同集，141 张 | 0.43266 | 0.43282 | +0.016 | -0.050 至 +0.054 | -1 | +1 | +1 |
| 原 M2 留出子集，20 张 | 0.39709 | 0.39748 | +0.038 | +0.020 至 +0.064 | +0 | +0 | +0 |

141 张图上 PQ 只提高 0.016 个百分点，配对区间包含零，且少匹配 1 个目标。20 张原 M2 留出图的 TP、FP、FN 均持平。该轮结果支持保守边界编辑能避免先前直接精修的大幅下降，发布版 M2 仍是主对照。历史 0.4331 略高于本轮边界精修。141 张仍含发布版 M2 的历史训练及校准重叠，适用范围是同集配对对照。两种评分入口交叉一致，4090 已释放。

[完整核验指标与训练配置](results/m2_refiner_followup.json)

## M2 多方向纠错实验

在冻结的 M2 上比较候选补回、边界调整、拆分合并、质量筛查和神经 M4。180 张图用于拟合，两折各 69 张用于选择，另有 68 张用于锁定方案后的比较。各分区按日期隔离。68 张仍来自发布版 M2 的原训练分区。原 141 张及团队 103 张评测标注在本轮保持封存。

第三折的原 M2 PQ 为 **0.42311**。候选组合加入恢复质量校准后，PQ 为 **0.43316**，误检减少 57 个，匹配减少 9 个。神经 M4 接在固定的筛查与补回组合后，PQ 从 0.43173 变为 0.43172，匹配数、误检和漏检均持平。

[实验报告](results/m2_error_directions.md) · [核验指标与配置](results/m2_error_directions.json)

## 权重下载

在 [model-weights-v1](https://github.com/wuliabc121/solar-filament-segmentation/releases/tag/model-weights-v1) 下载：

| 文件 | 内容 | ZIP SHA-256 |
|---|---|---|
| [solar-filament-model-weights.zip](https://github.com/wuliabc121/solar-filament-segmentation/releases/download/model-weights-v1/solar-filament-model-weights.zip) | 原 M1、M2、M3 骨干和 M3 质量头 | `ff2ea9416b1f70e6c26af6cb6cc73ac65bf76407bd08251fcf6172a5a940d2ac` |
| [solar-filament-m4-e5-weights-v1.zip](https://github.com/wuliabc121/solar-filament-segmentation/releases/download/model-weights-v1/solar-filament-m4-e5-weights-v1.zip) | T00、T88 的模型张量和清单 | `bdce2995a3c10c312994865c7de1f4ea3bc08d4dff337243afe9613d0ecae10b` |

原压缩包内的文件按 [上游权重清单](../../docs/checkpoints.json) 放置。M4 压缩包可解到任意目录。两份权重来自 103 张留出对照所用的全量检查点，导出时只保留模型参数和必要推理配置。逐文件 SHA-256 及来源见 [M4 权重清单](m4_weights.json)。

## 运行

在仓库根目录运行。使用原仓库的 Python 3.11 团队环境和 MAGFiLO 官方标注。可按 [原仓库 README](../../README.md) 生成 M1、M2 的原图概率缓存以及 M3 的实例 JSONL。训练用团队原划分中的 500 张图，留出评分默认读取同一协议的 103 张图。原图文件名使用观测 ID 加 `.jpeg`。

```sh
export PYTHONPATH=src:.
./team.sh fuse --instances runs/m3/instances.jsonl --cnn runs/m1_prob \
  --transformer runs/m2_prob --config team/configs/e5_interface_only.json \
  --run-id e5-holdout --purpose application --output runs/e5_holdout.jsonl

.team-env/bin/python -m m4_filament refine --images runs/holdout_images \
  --m1 runs/m1_prob --m2 runs/m2_prob --coarse runs/e5_holdout.jsonl \
  --checkpoint weights/m4/m4-e5-t00.pt --device cuda \
  --output runs/e5_t00.jsonl

.team-env/bin/python -m m4_filament components --model m2 \
  --images runs/holdout_images --probabilities runs/m2_prob --output runs/m2_instances.jsonl

.team-env/bin/python -m m4_filament score --annotations data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --prediction m2=runs/m2_instances.jsonl --prediction e5=runs/e5_holdout.jsonl \
  --prediction e5_t00=runs/e5_t00.jsonl --output runs/m4_aligned_score
```

训练先用同一 E5 配置在 500 张训练图上生成候选，再从匹配正样本构造裁片。每张图使用按记录 ID 排序的第一个标注版本。候选与 GT 需要互为唯一最佳，掩膜 IoU 至少 0.30，领先次佳 GT 至少 0.05。以候选为中心保留 32 像素上下文，裁片需容纳完整目标。全量训练四轮，使用 CUDA BF16 前向和 FP32 损失。

```sh
.team-env/bin/python -m m4_filament prepare --images data/raw/train/train_images \
  --m1 runs/train_m1_prob --m2 runs/train_m2_prob --coarse runs/e5_train.jsonl \
  --annotations data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --output runs/m4_positive_crops

.team-env/bin/python -m m4_filament train --dataset runs/m4_positive_crops \
  --recipe t88 --threshold 0.35 --output weights/m4/m4-e5-t88-refit.pt
```

初筛用 `--stop-after 2 --exclude experiments/m4_e5/dev_a.json`。E、F 折用四轮训练，并分别传入 `dev_e.json`、`dev_f.json` 和固定种子 20261810、20261918。评分时将对应名单传给 `score --observations`。候选融合、训练设置、推理阈值、标注及模型来源见核验指标。评分使用严格大于 0.50 的 IoU 匹配条件，逐标注版本汇总 PQ。
