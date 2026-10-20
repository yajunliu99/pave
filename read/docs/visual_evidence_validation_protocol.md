## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-14
- Verification Status: ANALYZED
- Version Label: evidence_input_isolation_v6

## 验证范围

这是对一个已讨论路口的开发性复核。开发者已见用户反馈，候选几何也来自之前的辅助绘制，不能称为独立盲测或自动边界提取。验证对象是：模型调用是否没有读入人工答案，以及给定候选区域时是否能从原始影像得出相关判断。

参考文件只由独立的 post-run 脚本在完整推断后读取。预测与参考不一致不会触发重答或结果改写。允许一次通用格式修复、每方向一次由模型请求的细节复核。全部尝试及失败目录保留。

首轮卫星局部存在性结果揭示了错误传播风险：进口外侧停车/排队车辆的初判可能有误。因此融合阶段对每个区域都允许重新判断物理类别，而非只预测用途。该机制适用于所有方向和区域，不指定正确类别。

## 统计与可重复性边界

没有统计显著性检验、置信区间或总体准确率估计。最终逐项比较只是与用户反馈的一致性。外部 VLM 新请求不是确定性可重跑系统；缓存复跑仅验证给定响应下的执行、校验和导出一致性。

## 方法学检查：11/11 覆盖

| 项目 | 本次适用性 |
|---|---|
| Simpson's paradox | 无分组相关或汇总效应估计，不适用 |
| Ecological fallacy | 单个区域逐项描述，不外推道路总体 |
| Berkson's paradox | 试点和错误案例具有选择性，不从它们推断总体关联 |
| Collider bias | 无回归调整或因果条件化分析，不适用 |
| Base rate neglect | 不报告灵敏度、PPV 或总体准确率；未估计类别基率 |
| Regression to the mean | 未以一次改进证明效果，保留先前失败，不作因果改善声明 |
| Survivorship bias | 保留全部完成/失败调用，不仅展示成功案例 |
| Look-elsewhere effect | 反馈对照项目固定，不仅挑选匹配项；无显著性搜索 |
| Garden of forking paths | 承认围绕已知试点开发，记录全部版本、格式修复和融合修改，不称验证性独立试验 |
| Correlation != causation | 不把当前结果变化归因于单个新增模块；多个流程因素同时改变 |
| Reverse causality | 无因果方向估计，不适用 |
