# 方法与复现范围

- M1 将 RGB 图像缩放到 512×512，输出语义前景概率，再还原到原图坐标。M2 使用 512 像素滑窗、128 像素重叠和 Hann 加权还原语义概率图。
- M3 从结构骨干 S 的掩膜候选提取 24 项检测、形状和灰度特征，以已冻结的 MLP 质量头重评候选。保留 match 分数至少 0.35、面积至少 32 像素的实例，掩膜几何保持原样。配置见 `deployments/m3_quality_v1/quality_profile.json`。
- `team_pipeline/contracts.py` 验证概率图来源和原图坐标。`schemas/` 定义实例 JSONL/RLE 契约。`requires_refiner` 标记与 crop 坐标供 M4 接口使用。
- M2 CSV 入口和项目内部入口使用同一计数规则，对每个标注者单独计算严格 IoU > 0.5 的命中，再汇总 TP、FP、FN 和 IoU。交叉核对覆盖编码与聚合。
- 141 张复核重新执行质量评分，复用经哈希核验的骨干原始候选。质量头拟合及调参与诊断图像按日期、内容隔离，骨干存在 31 张历史训练重叠。

发布副本移除了本机路径、私有运行记录和原始输入，重算了受路径调整影响的配置哈希。`docs/checkpoints.json` 保留权重原始 SHA256。验证列表和关键配置作为实验协议保留。官方评测参考代码可从赛事自评估 Notebook 获取。两项逐函数一致性测试需要将 v6 Python 导出置于 `references/official-self-evaluation-v6.py`。

## 源码测试

先创建 `.env` 和 `.team-env`。下面的测试范围适用于只含源码的发布副本。原始导入清单、历史运行目录和官方参考导出到位后，可以再运行对应的完整归档校验。

```sh
PYTHONPATH=src:. .team-env/bin/python -m pytest -q tests team_tests adapt_tests random_tests \
  --ignore=tests/test_balanced_driver.py --ignore=tests/test_resume_driver.py \
  -k 'not test_final_import_manifest_matches_every_file and not test_local_score_matches_published_notebook_boundaries_and_random_matrices and not test_rle_iou_matches_original_torch_overlap_on_asymmetric_masks'
```
