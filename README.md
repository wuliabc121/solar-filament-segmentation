# 太阳暗条实例分割

团队整合项目，面向 Solar Filament Segmentation Challenge 2026。输入为 Hα 太阳图像，输出为原图坐标的实例掩膜及 COCO RLE。

| 模块 | 目录 | 职责 |
|---|---|---|
| M1 | `team/m1/source/` | U-Net + ResNet34 语义分割 |
| M2 | `team/m2/source/` | SegFormer 语义分割和 CSV 评测 |
| M3 | `src/m3_filament/`、`deployments/` | YOLO 实例分割。当前独立模型使用 S 骨干和质量评分头 |
| 整合 | `team_pipeline/`、`team_adaptation/` | 语义概率、原图坐标契约、实例融合与后处理 |
| M4 交接 | `src/m3_filament/schema.py`、`docs/methods.md` | 带目标 crop 和精修标记的粗实例接口 |

## 环境

使用 Python 3.11。运行 `sh team/setup-team.sh` 创建项目专用 `.team-env`。可通过 `TEAM_CONDA_BIN` 指定 Conda 可执行文件。原 M3 训练入口可用 `sh setup.sh` 创建独立 `.env`。锁定的依赖见 `team/requirements-team-lock.txt` 和 `requirements-lock.txt`。

通过赛事官方渠道取得 MAGFiLO 数据后，将其放入 `data/raw/`，保留 `train/train_images/` 和 `test/test_images/` 结构。模型权重由团队另行提供，保存到 [权重清单](docs/checkpoints.json) 所列位置，保留原 SHA256。仓库提供源码、配置、划分协议和聚合结果。

## 推理与评估

```sh
./predict.sh --images /path/to/images --output reports/m3_inference
./team.sh predict-semantic --model m1 --images /path/to/images \
  --output reports/m1_prob --run-id m1_local --device cpu
./team.sh predict-semantic --model m2 --images /path/to/images \
  --output reports/m2_prob --run-id m2_local --device cpu
./team.sh score-csv --gt /path/to/gt.csv --pred /path/to/pred.csv \
  --height 2048 --width 2048 --output reports/score.json
```

M3 默认部署由 `deployments/active_m3.json` 指向。新 M3 的质量分数需要单独校准后才能用于融合。`team/configs/` 提供融合接口配置。精修模型和端到端 M4 结果尚待独立验证。

## 随机 20% 开发诊断

固定种子 42，从 707 张已标注图像抽取 141 张。共 228 份独立标注记录，1541 个 GT 实例。同图的不同标注一起划分。

| 模型 | PQ | SQ | 召回率 | FP | FN |
|---|---:|---:|---:|---:|---:|
| M1 | 0.3789 | 0.6541 | 56.20% | 583 | 675 |
| M2 | 0.4331 | 0.6849 | 68.20% | 732 | 490 |
| M3（S＋质量头） | 0.3649 | 0.6717 | 53.99% | 690 | 709 |

M3 精确率 54.66%，小目标召回 53.80%。验证样本与三个模型的历史训练记录分别重叠 136、101、31 张，其中 M1 的训练成员由源代码重建。表中指标用于开发诊断。完整聚合结果与运行编号见 [评测摘要](docs/random20_metrics.json)。

历史研究脚本位于 `scripts/`、`random_validation/` 和 `team_adaptation/`。复跑历史实验需恢复对应的本地缓存、权重和冻结协议。`docs/methods.md` 记录方法及复现范围。
