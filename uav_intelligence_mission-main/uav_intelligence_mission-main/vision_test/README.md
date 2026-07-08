# Vision Test Scripts

这组脚本是从当前 `stage1_vision_node.py` 拆出来的独立测试版，方便单独调试视觉功能。

共同特点：

- 打开即开始识别，不需要 `/uav/scan_enable` 和 `/uav/scan_target`。
- 启动时会询问是否打开 OpenCV 图形界面：NoMachine/桌面环境输入 `y`，纯终端输入 `n`。
- 也可以用 `--gui` / `--no-gui` 跳过询问。
- 没识别到时只低频打印简短信息。
- 识别到候选目标或有效目标时按帧打印详细信息。
- 默认不依赖 ROS；加 `--ros` 后会发布原正式节点兼容的话题，方便 `rostopic echo` 调试。

文件列表：

```text
qr_test_node.py                  二维码识别测试
image_target_yolo_test_node.py   图片靶方形定位 + YOLO 分类测试
special_target_test_node.py      特殊靶方形轮廓测试
ring_gate_test_node.py           圆环检测测试
```

---

## 1. 二维码识别测试

```bash
python3 qr_test_node.py --camera-index 0 --gui
python3 qr_test_node.py --camera-index 0 --no-gui
```

默认要求二维码内容格式为：

```text
类别1,类别2,left/right
```

例如：

```text
man,apple,left
```

如果只是测试普通二维码，不想检查格式，可以用：

```bash
python3 qr_test_node.py --accept-any --gui
```

加 ROS 发布：

```bash
python3 qr_test_node.py --ros
```

发布话题：

```text
/uav/qr_text
```

---

## 2. 图片靶 YOLO 分类测试

这个脚本是必须跑 YOLO 的图片靶识别测试。流程是：

```text
下视/USB 相机取图
↓
OpenCV 找方形靶板外轮廓
↓
透视矫正成正方形图像
↓
中心裁剪
↓
YOLO 分类模型识别类别
↓
输出 class_name、confidence、offset_x_m、offset_y_m
```

默认模型路径是脚本目录下的相对路径：

```text
models/image_target_classifier.engine
```

推荐目录结构：

```text
vision_test_scripts/
├── image_target_yolo_test_node.py
├── models/
│   └── image_target_classifier.engine
```

运行：

```bash
python3 image_target_yolo_test_node.py --camera-index 0 --gui
python3 image_target_yolo_test_node.py --camera-index 0 --no-gui
```

如果模型放在别的位置，可以指定相对路径或绝对路径：

```bash
python3 image_target_yolo_test_node.py --yolo-model models/image_target_classifier.engine --gui
python3 image_target_yolo_test_node.py --yolo-model ../models/image_target_classifier.engine --gui
```

默认按分类模型加载：

```bash
--yolo-task classify
```

如果你之后换成检测模型，可以试：

```bash
python3 image_target_yolo_test_node.py --yolo-task detect --yolo-model models/image_target_detector.engine --gui
```

加 ROS 发布：

```bash
python3 image_target_yolo_test_node.py --ros --gui
```

发布话题：

```text
/uav/vision_result
```

发布 JSON 的 `target` 字段为：

```text
image_target
```

主要输出字段：

```json
{
  "target": "image_target",
  "detected": true,
  "class_name": "apple",
  "offset_x_m": 0.04,
  "offset_y_m": -0.03,
  "confidence": 0.90,
  "yolo_confidence": 0.88,
  "stable_count": 4
}
```

识别不到时低频打印：

```text
[IMAGE_YOLO] square not found
```

识别到方形但 YOLO 没分类成功时低频打印：

```text
[IMAGE_YOLO] square found but class not found
```

分类成功时按帧打印详细信息。

---

## 3. 特殊靶方形轮廓测试

特殊靶脚本默认不跑 YOLO，主要测试几何方形定位。它用于确认方形板子外轮廓能否被稳定检测到。

```bash
python3 special_target_test_node.py --camera-index 0 --gui
python3 special_target_test_node.py --camera-index 0 --no-gui
```

加 ROS 发布：

```bash
python3 special_target_test_node.py --ros
```

发布话题：

```text
/uav/vision_result
```

发布 JSON 的 `target` 字段为：

```text
special_target
```

如果后续决定特殊靶也需要 YOLO，可以再单独改这个脚本。

---

## 4. 圆环检测测试

圆环脚本使用前置 RealSense D435i。默认圆环外径为：

```text
1.20 m
```

运行：

```bash
python3 ring_gate_test_node.py --gui
python3 ring_gate_test_node.py --no-gui
```

如果圆环外径变化，必须改外径参数，注意是外径不是半径：

```bash
python3 ring_gate_test_node.py --outer-diameter 1.20 --gui
```

加 ROS 发布：

```bash
python3 ring_gate_test_node.py --ros
```

发布话题：

```text
/uav/vision_result
```

发布 JSON 的 `target` 字段为：

```text
ring_gate
```

---

## 5. 纯终端 / NoMachine 使用建议

NoMachine 或本地桌面：

```bash
python3 image_target_yolo_test_node.py --gui
```

纯 SSH 终端：

```bash
python3 image_target_yolo_test_node.py --no-gui
```

如果终端里误开了 GUI，可能会出现 OpenCV 窗口相关报错；这种情况直接改用 `--no-gui`。

---

## 6. 依赖说明

二维码脚本：

```bash
pip install pyzbar opencv-python
```

图片靶 YOLO 脚本：

```bash
pip install ultralytics opencv-python
```

圆环脚本需要 RealSense：

```bash
pip install pyrealsense2 opencv-python
```

ROS 发布是可选功能，只有加 `--ros` 时才会尝试导入 `rospy`。
