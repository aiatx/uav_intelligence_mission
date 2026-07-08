# FSM 与视觉节点任务对接文档

## 1. 总体分工

本项目中，FSM 节点和视觉节点分工如下：

```text
视觉节点负责：
相机图像 → 目标识别 → 像素定位 → 米制偏差计算 → 发布视觉结果

FSM 负责：
任务状态切换 → 请求识别目标 → 根据视觉结果移动无人机 → 稳定后投放或穿越
```

视觉节点不要直接控制无人机，不要直接发布投放命令，不要修改航点。

FSM 不做复杂像素换算，不使用像素偏差控制飞机。

FSM 正式控制只使用视觉节点输出的米制偏差：

```text
offset_x_m
offset_y_m
offset_z_m
forward_m
confidence
stable_count
```

像素坐标可以发布，但只作为调试字段。

------

## 2. FSM 发给视觉节点的话题

视觉节点需要订阅：

```text
/uav/scan_enable   std_msgs/Bool
/uav/scan_target   std_msgs/String
```

含义：

```text
/uav/scan_enable = True
表示 FSM 当前需要视觉节点工作

/uav/scan_enable = False
表示 FSM 当前不需要视觉节点工作

/uav/scan_target
表示当前需要识别的目标类型
```

`/uav/scan_target` 可能取值：

```text
none
qr
image_target
special_target
ring_gate
```

对应关系：

```text
qr             二维码识别
image_target   图片靶识别和靶心定位
special_target 特殊靶中心定位
ring_gate      圆环识别和圆环中心定位
none           不识别
```

视觉节点应只在：

```text
scan_enable = True
```

时发布有效识别结果。

------

## 3. 视觉节点发给 FSM 的话题

### 3.1 二维码旧接口

二维码当前使用旧接口：

```text
/uav/qr_text   std_msgs/String
```

格式：

```text
class1,class2,left/right
```

示例：

```text
man,apple,left
```

要求：

```text
必须是 3 个字段
字段之间用英文逗号分隔
前两个字段是图片类别
第三个字段只能是 left 或 right
建议全部小写
不要带多余空格
```

------

### 3.2 统一视觉结果接口

图片靶、特殊靶、圆环使用统一接口：

```text
/uav/vision_result   std_msgs/String
```

内容是 JSON 字符串。

------

## 4. 图片靶 image_target 接口

当 FSM 发布：

```text
/uav/scan_enable = True
/uav/scan_target = image_target
```

视觉节点需要识别当前图片靶，并发布：

```json
{
  "target": "image_target",
  "detected": true,
  "class_name": "apple",
  "offset_x_m": 0.04,
  "offset_y_m": -0.03,
  "confidence": 0.90,
  "stable_count": 5
}
```

字段说明：

```text
target
必须是 image_target

detected
是否检测到图片靶

class_name
图片靶类别，必须和二维码类别命名保持一致，建议小写

offset_x_m
靶心相对当前无人机投放点的前后偏差，单位 m

offset_y_m
靶心相对当前无人机投放点的左右偏差，单位 m

confidence
视觉结果置信度，范围建议 0~1

stable_count
连续稳定识别帧数
```

坐标方向约定：

```text
offset_x_m > 0：
靶心在当前无人机机头前方，无人机应该向前移动

offset_x_m < 0：
靶心在当前无人机机头后方，无人机应该向后移动

offset_y_m > 0：
靶心在当前无人机左侧，无人机应该向左移动

offset_y_m < 0：
靶心在当前无人机右侧，无人机应该向右移动
```

注意：

```text
offset_x_m / offset_y_m 是当前无人机机体系下的米制偏差
不是像素偏差
不是起飞点坐标系偏差
不是全局 local 坐标偏差
```

建议额外发布调试字段：

```json
{
  "target_center_px": [318, 241],
  "release_center_px": [326, 238],
  "offset_px": [-8, 3],
  "mapping_method": "homography_800mm_square",
  "reprojection_error_px": 1.8
}
```

这些字段 FSM 不用于控制，只用于人工检查视觉映射是否正确。

------

## 5. 特殊靶 special_target 接口

当 FSM 发布：

```text
/uav/scan_enable = True
/uav/scan_target = special_target
```

视觉节点需要识别特殊靶中心，并发布：

```json
{
  "target": "special_target",
  "detected": true,
  "offset_x_m": 0.02,
  "offset_y_m": -0.01,
  "confidence": 0.90,
  "stable_count": 6
}
```

字段说明：

```text
target
必须是 special_target

detected
是否检测到特殊靶

offset_x_m
特殊靶中心相对当前无人机投放点的前后偏差，单位 m

offset_y_m
特殊靶中心相对当前无人机投放点的左右偏差，单位 m

confidence
检测置信度

stable_count
连续稳定识别帧数
```

坐标方向和图片靶完全一致：

```text
offset_x_m > 0：目标中心在当前无人机前方
offset_x_m < 0：目标中心在当前无人机后方

offset_y_m > 0：目标中心在当前无人机左侧
offset_y_m < 0：目标中心在当前无人机右侧
```

特殊靶不需要 `class_name`。

------

## 6. 圆环 ring_gate 接口

当 FSM 发布：

```text
/uav/scan_enable = True
/uav/scan_target = ring_gate
```

视觉节点需要识别圆环中心，并发布：

```json
{
  "target": "ring_gate",
  "detected": true,
  "forward_m": 1.20,
  "offset_y_m": -0.05,
  "offset_z_m": 0.03,
  "confidence": 0.86,
  "stable_count": 4
}
```

字段说明：

```text
target
必须是 ring_gate

detected
是否检测到圆环

forward_m
圆环中心在当前无人机机头前方的距离，单位 m

offset_y_m
圆环中心相对当前无人机左右方向的偏差，单位 m

offset_z_m
圆环中心相对当前无人机高度方向的偏差，单位 m

confidence
圆环检测置信度

stable_count
连续稳定识别帧数
```

坐标方向约定：

```text
forward_m > 0：
圆环中心在当前无人机机头前方

offset_y_m > 0：
圆环中心在当前无人机左侧

offset_y_m < 0：
圆环中心在当前无人机右侧

offset_z_m > 0：
圆环中心在当前无人机上方

offset_z_m < 0：
圆环中心在当前无人机下方
```

注意：

```text
forward_m / offset_y_m / offset_z_m 是当前无人机机体系下的米制偏差
不是像素偏差
不是起飞点坐标系偏差
不是全局 local 坐标偏差
```

建议额外发布调试字段：

```json
{
  "ring_center_px": [315, 236],
  "image_center_px": [320, 240],
  "ring_radius_px": 128,
  "mapping_method": "pnp_or_size_estimation",
  "reprojection_error_px": 2.1
}
```

这些字段 FSM 不用于控制，只用于调试。

------

## 7. FSM 如何使用图片靶结果

FSM 到达 `IMGx_DROP` 后会请求：

```text
scan_target = image_target
```

视觉节点返回 `class_name` 和靶心偏差。

FSM 会判断：

```text
class_name 是否等于二维码中的两个目标类别之一
```

如果不匹配，FSM 跳过当前图片靶。

如果匹配，FSM 使用：

```text
offset_x_m
offset_y_m
```

对准靶心。

当满足：

```text
sqrt(offset_x_m² + offset_y_m²) < align_xy_eps
confidence >= align_min_confidence
stable_count >= align_min_stable_count
无人机自身已停稳
```

FSM 发布：

```text
/uav/drop_cmd = image_drop_1
```

或：

```text
/uav/drop_cmd = image_drop_2
```

------

## 8. FSM 如何使用特殊靶结果

FSM 到达 `SPECIAL_DROP` 后会请求：

```text
scan_target = special_target
```

视觉节点返回特殊靶中心偏差。

FSM 使用：

```text
offset_x_m
offset_y_m
```

对准特殊靶中心。

满足稳定条件后，FSM 发布：

```text
/uav/drop_cmd = special_drop
```

------

## 9. FSM 如何使用圆环结果

FSM 到达 `RING_SEARCH_START` 后会请求：

```text
scan_target = ring_gate
```

视觉节点返回：

```text
forward_m
offset_y_m
offset_z_m
```

FSM 根据这些数据动态生成：

```text
RING_PRE
RING_CENTER
RING_POST
```

其中：

```text
RING_CENTER = 当前无人机位置 + 当前机体系下的圆环中心偏差
RING_PRE    = RING_CENTER 沿当前机头反方向后退一段距离
RING_POST   = RING_CENTER 沿当前机头正方向前进一段距离
```

无人机飞到 `RING_PRE` 后，FSM 会再次请求：

```text
scan_target = ring_gate
```

此时主要使用：

```text
offset_y_m
offset_z_m
```

进行二次对准。

满足：

```text
sqrt(offset_y_m² + offset_z_m²) < ring_yz_eps
confidence >= ring_min_confidence
stable_count >= ring_min_stable_count
```

后，FSM 关闭视觉请求，并连续飞过：

```text
RING_CENTER
RING_POST
```

------

## 10. 视觉结果失败格式

如果未检测到目标，视觉节点也建议发布结果，但 `detected=false`：

```json
{
  "target": "ring_gate",
  "detected": false,
  "reason": "not_found",
  "confidence": 0.0,
  "stable_count": 0
}
```

常见 reason：

```text
not_found
low_confidence
motion_blur
target_out_of_view
bad_geometry
```

FSM 收到 `detected=false` 不会使用该结果控制飞机。

------

## 11. 稳定帧数要求

视觉节点需要维护 `stable_count`。

建议规则：

```text
连续检测到同一目标，且中心位置变化不大，则 stable_count 递增
目标丢失，stable_count 清零
目标跳变过大，stable_count 清零
置信度过低，stable_count 清零
```

FSM 会用 `stable_count` 防止单帧误识别导致投放或穿环。

------

## 12. 坐标系总约定

所有正式控制字段必须是当前无人机机体系下的米制偏差。

图片靶和特殊靶：

```text
offset_x_m：当前机头前方为正
offset_y_m：当前机体左侧为正
```

圆环：

```text
forward_m：当前机头前方为正
offset_y_m：当前机体左侧为正
offset_z_m：当前无人机上方为正
```

视觉节点不要输出起飞点坐标系偏差给 FSM。
视觉节点不要输出全局 local 偏差给 FSM。
视觉节点不要只输出像素偏差给 FSM。

------

## 13. 单独测试方法

### 13.1 图片靶 / 特殊靶测试

启动视觉节点后，手动发布：

```bash
rostopic pub /uav/scan_enable std_msgs/Bool "data: true"
rostopic pub /uav/scan_target std_msgs/String "data: 'image_target'"
rostopic echo /uav/vision_result
```

测试方向：

```text
靶心在正下方：offset_x_m ≈ 0，offset_y_m ≈ 0
靶心向无人机前方移动 10 cm：offset_x_m ≈ +0.10
靶心向无人机后方移动 10 cm：offset_x_m ≈ -0.10
靶心向无人机左侧移动 10 cm：offset_y_m ≈ +0.10
靶心向无人机右侧移动 10 cm：offset_y_m ≈ -0.10
```

### 13.2 圆环测试

手动发布：

```bash
rostopic pub /uav/scan_enable std_msgs/Bool "data: true"
rostopic pub /uav/scan_target std_msgs/String "data: 'ring_gate'"
rostopic echo /uav/vision_result
```

测试方向：

```text
圆环在正前方：forward_m > 0，offset_y_m ≈ 0，offset_z_m ≈ 0
圆环向无人机左侧移动 10 cm：offset_y_m ≈ +0.10
圆环向无人机右侧移动 10 cm：offset_y_m ≈ -0.10
圆环向上移动 10 cm：offset_z_m ≈ +0.10
圆环向下移动 10 cm：offset_z_m ≈ -0.10
```

如果正负号反了，必须先在视觉节点修正。

------

## 14. 当前 FSM 侧参数

图片靶 / 特殊靶相关参数：

```text
vision_result_timeout
image_align_timeout
special_align_timeout
align_xy_eps
align_step_max
align_gain
align_min_confidence
align_min_stable_count
```

圆环相关参数：

```text
ring_search_timeout
ring_timeout_policy
ring_pre_distance
ring_post_distance
ring_align_timeout
ring_yz_eps
ring_align_step_max
ring_align_gain
ring_min_confidence
ring_min_stable_count
ring_forward_min
ring_forward_max
ring_min_z
ring_max_z
```

这些参数由 FSM 的 YAML 配置，不需要视觉节点控制。

------

## 15. 对接验收标准

视觉节点完成后，至少要满足：

```text
1. 能根据 scan_target 切换识别任务
2. qr 阶段能发布 /uav/qr_text
3. image_target 阶段能发布 class_name、offset_x_m、offset_y_m
4. special_target 阶段能发布 offset_x_m、offset_y_m
5. ring_gate 阶段能发布 forward_m、offset_y_m、offset_z_m
6. 所有米制偏差正负号符合 FSM 约定
7. 静止目标时 offset 抖动不应过大
8. confidence 和 stable_count 能真实反映识别可靠性
9. 目标丢失时 detected=false，不要继续发布旧结果
10. 像素字段只用于调试，正式控制必须使用米制偏差
```