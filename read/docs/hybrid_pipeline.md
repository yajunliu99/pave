# 基于视觉证据的车道与 movement pipeline

当前默认入口为 `scripts/pipeline/run_hybrid_pipeline.py`，配置的 `inference_mode` 为 `visual_evidence`。输出目录为 `experiments/hybrid_pipeline_351_20260914_visual_evidence_fused`。

## 本次改变

v5 将用户确认的车道用途和不存在区域直接写成约束，不能用来证明模型独立得出了结论。v6 把这些信息从推断输入中移除，参考答案只在运行结束后由单独的评估脚本读取。

- 恢复 v4 的候选几何，包含当时尚未删除的 EB/WB 出口区域。
- 配置按字段白名单重新构建，不包含 review_annotation、exclusive、intended_directions、excluded_regions、carriageway_envelope，以及旧修订说明和质量标签。
- 所有候选区域仍由此前人工辅助绘制。因此本实验是给定候选几何条件下的视觉判别和关联，不是完全自动车道提取，也不是开发者未见答案的独立测试集。

## 推断流程

1. 在四个方向各 25/50/100 m 位置执行 YOLO，保留三张反向辅助视图。目标框提供遮挡位置，不直接判定车道用途。
2. 复用或读取十二张原始 GSV 的独立场景观察。这类缓存只基于源图片，不来自人工修订的车道结果。
3. 对八个卫星断面执行 surface audit：同时提供保留背景的原始局部图和候选边界图。检查路缘、铺装、植被、建筑、标线、阴影及连续性；宽度只作为数值参考。
4. 对四个进口执行 usage audit：同时查看卫星候选区域与三档 GSV，复核每个区域的物理类别和存在性，再输出用途预测、证据图像、归一化位置框及物理关联说明。GSV 可以纠正卫星图把排队车辆误判为停车区域等错误；两阶段输出分别保留。
5. 模型可以自主请求最多三个原图局部裁剪。每个方向最多追加一次细节复核，裁剪坐标和原始像素来源全部保存；不根据人工参考答案选择裁剪位置。
6. movement 阶段综合模型的存在性、用途和图像证据。不存在/未确认的机动车端点不能组成有效车道对；用途冲突保留诊断标记，不按人工答案覆盖模型结果。
7. 推断结束后单独对照用户反馈，保存全部匹配、不匹配和未知项，不回写预测文件。

## 运行

```powershell
python scripts/pipeline/build_university_mill_evidence_config.py
python scripts/pipeline/run_hybrid_pipeline.py --reuse-cache-from experiments/hybrid_pipeline_351_20260913_three_positions
```

仅复跑缓存时使用 `--cache-only`。`--prepare-only` 只准备图片。更换输入、几何或模型时另设 `--output`，保留前次运行。

后验比较单独执行：

```powershell
python scripts/experiments/evaluate_visual_evidence.py --output experiments/hybrid_pipeline_351_20260914_visual_evidence_fused
```

`configs/evaluation/university_mill_351_user_review.json` 是评估参考文件；推断 runner 不导入或读取它。该参考是用户对本试点的反馈，不是独立测绘真值。

## 输出

| 文件 | 内容 |
|---|---|
| inference_config.json / input_manifest.json | 实际配置、来源哈希及版本 |
| candidate_geometry.png | 全部候选区域，包括可能不存在的区域 |
| surface_audits/ | 含背景的原图、候选图及输入面板 |
| surface_predictions.json | 首次存在性和设施类别预测 |
| usage_audits/ | 三档街景、模型请求的细节面板及像素映射 |
| lane_use_audits.json | 模型用途预测和区域绑定，不是人工约束 |
| lanes.json / lane_predictions.csv | 最终逐区域模型结果 |
| model_geometry.png | 根据模型存在性判断绘制的几何 |
| movements.json / movement_candidates.csv | 模型生成的连接候选及矛盾标记 |
| post_run_comparison.json | 推断完成后与反馈逐项比较 |
| report.md | 模型结果、来源和限制 |

结构检查可以修复一次 JSON 格式错误，但不因为语义预测与反馈不一致而要求模型重答。缓存复跑验证的是给定响应下的确定性执行，不能证明外部 VLM 新请求的统计可重复性。

## 验证范围

测试包括：人工标签 canary 被白名单过滤、推断输入拒绝答案字段、原先被否定的区域可以被模型判为存在、模型可以预测非参考转向、用途冲突不被强改、上下文像素保留和原图裁剪追溯。旧版本约束测试使用归档配置，不表示当前默认推断还启用那些人工约束。

本轮为一个已讨论路口的开发性验证，不报告泛化准确率，也不把缓存命中或单元测试通过当作视觉识别正确。此前 v5 规则与报告说明保存在 `docs/archive/hybrid_pipeline_v5.md`。

用途面板保留完整 PNG，但 API 传输使用同尺寸 JPEG（quality=95、4:4:4）。原生局部裁剪仍来自原始 PNG 源像素；transport.json 记录传输差异。首轮 PNG 用途请求失败，原运行目录及完成的表面审计保留，后续融合运行另存，不混写失败状态。
