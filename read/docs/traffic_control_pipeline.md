# 通用信号灯与标志采样流程

本模块从道路几何生成拍摄点，用 YOLO 和 VLM 选择特写，并在可读性不足时补充不同位置的街景。路口坐标、方位、进口数和车道侧偏移均不写在推断代码中。可以处理任意命名的进口、三岔/多岔/斜交、弯曲道路和左侧通行。

这表示无需按路口改源码，不表示所有路口都能完整识别。地图误差、街景覆盖、影像时间、遮挡与像素细节仍会留下不确定项。

## 从道路网络一键运行

在项目根目录执行，替换节点 ID 和网络文件即可；`--baseline`、`--atlas-base` 都是可选的，不依赖本试点的卫星候选。

```powershell
python scripts/pipeline/run_control_pipeline.py --node-id 0 --node-csv data/ASU/output/macronet/node.csv --link-csv data/ASU/output/macronet/link.csv --output experiments/control_node_0 --plan-only
```

去掉 `--plan-only` 才会查询街景、运行检测和识别。计划模式不访问 Google 或 VLM。独立结果保存在 `audit/index.html`、`audit/predictions.json`、`audit/views_manifest.json` 和 `audit/adaptive_history.json`。

本试点已执行的配置与输出：

```powershell
python scripts/pipeline/run_control_pipeline.py --config configs/control_sites/node_351.json --output experiments/control_generic_351_v2 --atlas-base experiments/geospatial_evidence_351_20260915_billing --cache-only
```

`--cache-only` 禁止 Google 网络补图并要求所有 VLM 阶段命中缓存；已有 acquisition manifest 不被改写，配置变化会报错。首次运行不要加此参数。请求缓存按完整坐标、搜索半径或完整全景取景参数命名，不会因重用文件名拿到旧角度的图片。

## 输入约定

道路网络适配器读取 GMNS 风格 CSV：节点 `node_id,x_coord,y_coord`；道路 `link_id,from_node_id,to_node_id,geometry,allowed_uses,from_biway`。坐标必须为 WGS84 经纬度，geometry 为 LINESTRING。由所有驶入该节点的机动车道路自动生成进口，不要求四个方向；终端折线平滑后计算朝向，局部采样沿折线弧长进行。

也可以直接传入 `control-site-1` JSON：

```json
{
  "schema": "control-site-1",
  "site_id": "your-junction",
  "origin_lonlat": [0.0, 0.0],
  "policy": {"driving_side": "right"},
  "approaches": [{
    "id": "approach-a",
    "label": "入口 A",
    "section_id": "incoming-a",
    "centerline_lonlat": [[-0.001, -0.001], [-0.0004, -0.0003], [0.0, 0.0]],
    "centerline_reference": "shared_road_axis",
    "carriageway_width_m": 10.5,
    "width_source": "map_width_prior",
    "allow_interior": false,
    "region_ids": []
  }],
  "models": {
    "vlm": {"preset": "default", "max_tokens": 6144, "temperature": 0},
    "yolo": {"weights": "PATH_TO_CHECKPOINT", "expected_sha256": "CHECKPOINT_SHA256", "device": "auto", "imgsz": 1280, "confidence": 0.25}
  }
}
```

以上坐标仅说明格式。centerline 按上游到路口排列；`shared_road_axis` 表示双向共用轴，`directed_carriageway` 表示已经位于行驶侧的道路中心线，后者不再重复施加横向偏移。独立运行允许没有 `region_ids`，此时不能输出具体车道绑定。模型配置可由网络适配器生成，已有项目的 `.env` 提供服务凭据；不会把凭据写入缓存。

## 位置与角度

1. 道路宽度来自明确宽度属性；缺失时可以使用地图车道数乘宽度的弱先验，再缺失则使用策略默认值。它们只影响相机搜索位置，绝不作为车道识别答案。
2. 若提供停止线几何和来源，投影到进口道路得到采样基准；缺失时利用相交道路宽度与夹角估计路口范围，标为 `estimated_junction_envelope`。没有把这个估计声称为自动识别出的停止线。
3. 按道路尺度在基准之前、附近以及可通行的路口内部生成三个初始站位。没有近似直行出口的 T 形进口不会把相机请求延伸到断头道路以外。
4. 按 Google 返回的实际拍摄点检查纵向偏差、道路范围、反向道路误匹配、全景 ID 和位置重复。指定时间窗口时，先尝试附近符合时期的全景；最接近的旧图会延后，只有有界搜索没有可用时才做历史回退并记录原因。失败时尝试相邻站位，仍失败则输出部分覆盖或不可用。
5. 概览朝向随道路切线变化；仰角默认根据目标相对相机高度的先验和纵向范围估计，可用策略覆盖。这只用于取景，不是测量信号灯高度。概览 FOV 默认100°。
6. VLM 输出原生图像目标框，几何函数据此计算新朝向/仰角和15—55°特写视场。特写与母图属于同一全景来源，不增加独立证据数。

站位零点是**各进口道路折线的路口端点**，负值表示上游；弯道按弧长计。它不是到路口中心、停止线或信号灯的直线距离。请求坐标、实际坐标、偏移量和角度均保留。

## 自动补图与停止条件

每轮按“概览检测 → VLM 选择目标 → 特写 → 文字/符号与适用范围复核”执行。重要设施仍不可读、斜视、被遮挡，或模型提出具体补图请求时，在现有相机前后搜索不同全景，重新取概览和特写。车道绑定未确定本身不会触发无休止补图。

默认每进口最多2轮、5个概览位置、每轮最多4张特写；每轮最多新增2个位置。全任务限64次元数据请求、64次图像请求、24次 VLM API 调用；均可在 policy 中调整。达到轮次/预算、找不到不同可用全景、没有新增文字/符号读数时停止，并记录原因。新增读数是复核启发式，不是准确率指标。

YOLO26n 仍只辅助发现其类别表中的交通灯/stop sign；其他标志和文字由 VLM 分析。检测遗漏不等于设施不存在。信号灯当前亮色不建立相位表，标志出现不等于适用于当前进路，具体车道绑定和 movement 约束保持独立。

## 可追溯输出与验证范围

- `imagery/sampling_plan.json`：道路几何、参考位置来源、请求位置。
- `imagery/manifest.json`：初始实际全景、位姿质量检查、拒绝与回退原因。
- `audit/adaptive_history.json`：每轮视图数、可读内容变化和停止原因。
- `audit/views_manifest.json`：全部图像、母图关系、来源日期、缓存哈希和额外查询记录。
- `audit/yolo/all_detections.json`：所有初始及补图概览的 YOLO 结果。
- `audit/vlm/`：每阶段的提示、原始响应、模型身份和校验结果。
- `audit/initial_execution.json` 与 `audit/executions/`：首次运行收据与后续执行记录，缓存重放不覆盖首次请求次数。
- `validation.json`：自动检查图像哈希、重复全景、来源、请求上限、车道图保持不变等结构性质，不是准确率评测。
- 可选 `atlas/`：接入已有卫星车道图；要求相同节点、有效区域 ID 和未叠加旧控制结果的底图快照。

本轮验证包括斜交、弯曲、多进口、左右行、宽度变化、停止线来源、重复全景、错误拍摄侧、缓存与预算测试；另有模拟 Google/VLM 的三岔和五岔完整两轮流程，模拟结果不计入识别准确率。真实节点0和6完成自动几何采样计划；真实影像完整识别仍以节点351为试点，尚未宣称跨路口准确率。
