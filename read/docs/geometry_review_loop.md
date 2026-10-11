# 候选车道区域的自动生成与审查 loop

模块位置：`src/movement_fixer/autoloop/`。入口：`scripts/pipeline/run_geometry_loop.py`。

v6 hybrid 流程使用的候选区域多边形是此前在对话中由助手描绘、再写进配置的，流程本身没有生成它们的步骤。本模块把这一步放进流程：模型起草边界，坐标检查和模型审查轮流找问题，按审查意见修改，直到达到明确的停止原因。输出的 `sections` 与 hybrid 配置中的结构相同。

现有 v6 流程和已冻结的实验目录没有改动。

## 流程

1. **裁剪框**（`frames.py`）：对节点的每条驶入、驶出机动车 link，从路网几何计算一个按行驶方向朝上的裁剪框。进口数量和夹角不限。距离从节点中心沿道路轴线量取，不沿折线量：macronet 合并路口后，link 折线从原子节点出发，离合并节点可以有数米。
2. **起草**：模型看带标尺和站线的裁剪图，对每条纵向边界给出它在每条站线上的 x 坐标。
3. **坐标检查**（`checks.py`）：相对宽度、绝对宽度、沿程宽度变化、边界折点、贴近裁剪边缘。检查只提出疑点，不下结论。
4. **审查**：模型看到自己画的边界叠加回原图，以及每个区域单独抠出的像素。对每个疑点必须回答修改、驳回（说明图上依据）或无法判断。
5. **修改**（`strips.py`）：移动、拆分、合并、增加、删除边界。无法应用的修改被拒绝并记录，不做猜测。
6. 回到第 3 步，直到停止。

起草结果贴到裁剪边缘时，先加宽裁剪框重新起草一次，而不是外推。

## 停止原因与状态

| 停止原因 | 含义 |
|---|---|
| `accepted` | 审查没有再提出修改 |
| `converged` | 最后一次修改小于 2 像素 |
| `oscillation` | 修改后回到了之前出现过的状态 |
| `round_limit` | 达到轮数上限（默认 3） |
| `no_applicable_edits` | 审查提出的修改都无法应用 |
| `budget_exhausted` | 达到调用上限 |
| `review_failed` / `draft_failed` | 模型输出在一次修复后仍不合结构要求 |

每个断面的状态是 `accepted`、`accepted_after_revision` 或 `unresolved`。只有以 `accepted` 或 `converged` 停止、且没有遗留疑点时才算前两种。

## 运行

```bash
~/.cache/net2cell-vlm/venvs/mac/bin/python scripts/pipeline/run_geometry_loop.py --config configs/autoloop/node_351.json --output experiments/geometry_loop_351_20261002 --plan-only
```

`--plan-only` 只计算裁剪框和调用预算，不访问任何服务，输出 `geometry/plan.json` 和 `geometry/plan.png`。去掉该参数才会调用模型。`--cache-only` 禁止新的调用。更换输入或策略时另设 `--output`。

运行结束后与参考多边形比较：

```bash
~/.cache/net2cell-vlm/venvs/mac/bin/python scripts/experiments/evaluate_geometry_loop.py --run experiments/geometry_loop_351_20261002
```

## 输出

| 文件 | 内容 |
|---|---|
| `geometry/plan.json`、`plan.png` | 裁剪框、调用预算 |
| `geometry/sections.json` | 各断面的候选区域，结构同 hybrid 配置 |
| `geometry/history.json` | 起草结果、最终结果、每轮的疑点、修改和停止原因 |
| `geometry/<断面>/draft/`、`round_n/` | 发给模型的图、当轮边界、疑点和审查回复 |
| `geometry/summary.json` | 各断面状态、调用次数 |
| `geometry/post_run_comparison.json` | 起草和最终结果分别与参考的比较 |

## 输入隔离

路网中的车道数只用于确定裁剪框大小，记录在 `crop_prior`，不进入任何提示词；有测试检查改变车道数不会改变提示词。参考多边形只由评估脚本在运行结束后读取。

## 验证范围

48 项测试使用合成图像和脚本化的模型回复，检查的是：几何与修改操作正确、loop 必然终止并记录原因、预算生效、裁剪框位置正确。这些测试不能说明真实模型描边界的准确程度。

351 的参考多边形本身由助手描绘，西向进口部分为插值，且裁剪框的参数是在看过 351 的情况下调的，因此 351 上的比较只说明与那次描绘的接近程度，不是泛化准确率。loop 的收敛也不等于正确。

## 351 首次真实运行（2026-10-02）

输出目录 `experiments/geometry_loop_351_20261002`，模型 `openai/gpt6_astra`，共 19 次调用（8 次起草、11 次审查）。

| 断面 | 状态 | 审查轮数 | 起草：描出/参考/对上 | 最终：描出/参考/对上 | 边界误差 |
|---|---|---:|---|---|---:|
| in_NB | accepted | 1 | 5 / 5 / 5 | 5 / 5 / 5 | 0.16 m |
| in_SB | accepted | 1 | 6 / 6 / 6 | 6 / 6 / 6 | 0.12 m |
| in_EB | accepted | 1 | 4 / 5 / 4 | 4 / 5 / 4 | 0.19 m |
| in_WB | accepted_after_revision | 2 | 5 / 5 / 4 | 4 / 5 / 4 | 0.53 m |
| out_WB | accepted | 1 | 3 / 4 / 3 | 3 / 4 / 3 | 0.26 m |
| out_NB | accepted | 1 | 3 / 3 / 1 | 3 / 3 / 1 | 0.32 m |
| out_EB | unresolved | 1 | 3 / 5 / 0 | 3 / 5 / 0 | — |
| out_SB | unresolved | 3 | 3 / 4 / 1 | 1 / 4 / 0 | — |

- 进口：参考中 v6 判为机动车道的 16 个区域全部有对应的描绘区域。in_EB 未对上的是轨道区，in_WB 未对上的是自行车带（被并入相邻车道，宽度比 1.5，低于 1.65 的检查阈值）。
- 出口：参考中 v6 判为存在的 9 个设施只对上 3 个（都在 out_WB）。out_EB、out_SB 位于建筑和树木阴影内。参考中被 v6 否定为非道路的区域，模型大多没有描。
- loop 的作用：in_WB 的审查删除了一个路外绿化带区域；5 个断面的审查只是驳回了"自行车带偏窄"的疑点；out_SB 被改坏。合计对上数 24 → 23，无对应的描绘区域 8 → 6。
- out_SB 改坏的原因之一是提示词缺陷：修改后边界重新编号，而"前几轮的修改"仍用旧编号描述，模型认为合并没有生效并再次合并。已改为按位置描述，并要求遮挡时回答"无法判断"而不是合并或删除（版本 `autoloop-geometry-2`）。修正后的重跑见下一节。

参考多边形本身由助手描绘，这些数字说明的是与那次描绘的接近程度。

## 351 修正后重跑（2026-10-02，`autoloop-geometry-2`）

输出目录 `experiments/geometry_loop_351_20261002_v2`。8 次起草全部复用首次运行的缓存，审查重新请求，共 12 次新调用。

- out_SB：审查只删除了一条夹出窄条的边界，保留了与参考对上的那条车道；对宽区域的疑点回答"无法判断"。最终 2 个区域、对上 1 个（首次运行为 1 个区域、对上 0 个）。
- 其余断面的区域与首次运行相同。out_EB 的状态由"未解决"变为"通过"，但仍没有与参考对上的区域。
- 合计：对上数起草 24、最终 24；无对应的描绘区域 8 → 6；没有被改坏的断面。

两次运行的审查回复不是同一批请求，差异中包含模型输出本身的波动，不能全部归因于提示词修改。

## 接入分类阶段（2026-10-02）

`scripts/pipeline/build_evidence_config_from_geometry.py` 把描出的区域写入 visual-evidence 配置，之后由现有的 `run_hybrid_pipeline.py` 做表面审计、用途审计和 movement。宽度参考改为全部进口区域的中位数，不再手选。`scripts/experiments/draw_lane_labels.py` 输出带车道号、设施类型和 movement 的标注图。

351 上的运行：配置 `configs/pilots/university_mill_351.autogeometry.json`，输出 `experiments/hybrid_pipeline_351_20261002_autogeometry`，几何来自 `geometry_loop_351_20261002_v2`。YOLO26n 在 CPU 上运行（15 张街景），32 次模型调用，其中 3 次为细节复核。

与 9 月 14 日 v6 运行（助手描绘的几何）相比：

| 项目 | 自动几何 | v6（9 月 14 日） |
|---|---|---|
| 机动车道数 in_NB / in_SB / in_EB / in_WB | 4 / 5 / 3 / 3 | 4 / 5 / 3 / 4 |
| 机动车道数 out_NB / out_SB / out_EB / out_WB | 1 / 1 / 2 / 2 | 1 / 1 / 2 / 2 |
| movement 候选 / 车道对 | 8 / 12 | 9 / 13 |

- 差异集中在 in_WB：描边界时漏了自行车带与相邻车道之间的线，最外侧区域被判为"无法确定"，西向右转因此未解决。其余 8 个 movement 候选的车道对与 v6 相同。
- in_SB 第 2 条车道的用途这次为"未确定"（v6 为左转），但 movement 阶段仍给出了 1、2 两条车道的左转候选，两个阶段的输出不一致。
- 四个进口的直行在两次运行中都没有形成候选或只部分形成：没有直行箭头时，用途审计不把车道判为直行。

Windows 配置中的反斜杠路径已在 `hybrid/common.py` 的 `resource_path` 中统一处理，现有 351 配置可在 macOS 上加载。

## 与 dashboard 的完整流程对齐（2026-10-02）

`dashboard/dashboard.html` 背后是三个阶段的叠加：基线分类、用 48 张出口街景（向外看 24 张、回望路口 24 张）做的出口复核、信号灯与标志审计。自动几何的流程现在按同样的阶段执行，入口为 `scripts/pipeline/run_site_pipeline.py`：

| 阶段 | 脚本 | 模型调用 |
|---|---|---|
| 描边界（起草、检查、审查） | `run_geometry_loop.py` | 有 |
| 生成分类配置 | `build_evidence_config_from_geometry.py` | 无 |
| 表面、用途、movement（卫星图 + 前向/后向街景 + YOLO） | `run_hybrid_pipeline.py` | 有 |
| 出口复核（向外看 + 回望） | `audit_downstream.py` | 每个出口 1 次 |
| 证据图集 | `build_lane_evidence_atlas.py` | 无 |
| 接入信号灯与标志 | `attach_control_evidence.py` | 无（复用已有审计） |
| 标注图、单文件 dashboard | `draw_lane_labels.py`、`dashboard/build_standalone_dashboard.py` | 无 |

已完成的阶段会被跳过，命令可以在中断后重复执行而不重复调用模型。`--plan-only` 只列出将要执行的阶段。

351 的结果：`dashboard/dashboard_autogeometry.html`（图集 `experiments/control_generic_351_20261002_autogeometry/atlas`），104 张视图、37 个全景，与参考 dashboard 相同。出口复核新增 4 次调用。逐断面的对比写在 `dashboard/report_autogeometry.md`，也嵌在页面里。

与参考 dashboard 的差异：in_WB（3 条机动车道加 1 条未确定，参考为 4 条）和 out_SB（1 条，参考为 2 条），都源于阴影或车辆遮挡下漏描的边界；其余 6 个断面的机动车道数相同。和参考流程一样，movement 没有在出口复核之后重新生成。

仍然只适用于 351 的部分：`fusion/bundle.py` 里卫星图缓存目录写死为 `cache/mapbox_351`；出口街景和信号灯审计使用的是 9 月取得的缓存，新路口需要先运行 `fetch_downstream_gsv.py` 和 `run_control_pipeline.py` 取图。

`hybrid/common.py` 的 `resource_path` 现在能把 Windows 机器写下的绝对路径按项目文件夹名重新定位到本机。

## 街景对齐（`scripts/pipeline/align_gsv_views.py`）

把卫星图上描出的车道投影到每张街景里，并纠正相机位姿，使车道号和类型能直接标在街景上。代码不含路口专用的常量：

- **路段**：按各断面区域相对路口中心的方位自动聚类（`road_arms`），任意数量、任意角度的路段都适用。每张街景按拍摄位置归入最近的路段；朝向与该路段驶入方向的夹角决定它是向前看还是回头看。
- **分组**：同一路段、同一拍摄日期的街景视为同一次行驶，共享横向偏移和相机高度，各自有朝向修正。偏移在路段坐标里比较，回头看的图符号相反。
- **步骤**：① 模型读出每条投影边界对应的真实标线在几条参考行上的像素位置，拟合位姿；② 组内联合拟合；③ 把投影整体平移 −2…+2 条车道宽做成 A–E 选项，由模型按内容选择（L 带上有左转箭头、BIKE 在自行车道、+RAIL 在轨道上），组内投票；④ 选定平移后，把每个读数重新匹配到最近的投影边界并重新拟合，迭代 3 轮；⑤ 模型逐条检查最终叠加。
- **状态**：`aligned`（检查一致，组内投票一致）、`aligned_single_vote`（检查一致，平移只来自一张图的高置信选择）、`aligned_single_view`、`alignment_doubtful`、`group_pose_unchecked`（离描出路段太远无法读数，套用同组位姿，未检查）、`unreadable`。
- **调用**：每张可读街景 3 次（读数、平移选择、检查），由 `--max-api-calls` 封顶。在 `run_site_pipeline.py` 中用 `--align-gsv` 开启。

为什么需要第 ③④ 步：车道线每隔一个车道宽重复一次，模型读像素位置时会把整组标线对错一到两条车道，多张图还会以相同方式读错，组内一致也挡不住；而"箭头在哪条带里"这类语义判断是准的。

351 南向的结果（`experiments/control_generic_351_20261002_autogeometry_v2/gsv_alignment_sb_realign2`）：原拟合偏了两条车道（SB L2 压在北向出口的轨道车道上）。一张前向图以高置信度选择平移 −2 条车道，重新匹配并联合拟合后，两张前向、两张回头看的街景逐条检查全部一致。平移只来自一张图的选择，所以状态是 `aligned_single_vote`。
