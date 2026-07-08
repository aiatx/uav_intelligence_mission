#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 vision node for the micro UAV mission.

Responsibilities:
1. Down camera:
   - QR scan.
   - Image target square localization and YOLO classification.
   - Special target square localization.
2. Front Intel RealSense D435i:
   - Ring gate localization only.

The node follows the FSM interface:
    subscribe: /uav/scan_enable, /uav/scan_target
    publish:   /uav/qr_text, /uav/vision_result
"""

# ================================================================
# 圆环尺寸重要说明（非常关键）
# ---------------------------------------------------------------
# 当前代码默认圆环“外径”为 1.20 m：ring_outer_diameter_m = 1.20
# 注意这里填的是【外径 diameter】，不是半径 radius。
# 如果之后更换圆环尺寸：
#   1) 先用尺子量圆环外边缘到外边缘的实际外径 D，单位米；
#   2) 推荐在 roslaunch/命令行里传参：
#        _ring_outer_diameter_m:=D
#      例如外径 1.00 m：
#        _ring_outer_diameter_m:=1.00
#   3) 如果只知道半径 R，要填 2*R，不要直接填 R；
#   4) 这个参数会直接影响 forward_m 距离估计。填错一倍，距离也会大约错一倍。
# ================================================================

import json
import math
import os
import re
import time
import queue
import threading
from datetime import datetime

import cv2
import numpy as np
import rospy

from std_msgs.msg import Bool, String

try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# =========================
# Image classifier model area
# =========================
# 支持两种图片靶分类后端：
#   1) yolo:      Ultralytics YOLO 分类/检测 engine，沿用原逻辑，类别名通常可从 YOLO 模型元信息读取。
#   2) resnet_trt: 原生 TensorRT ResNet18 engine，需要额外读取 classes.json。
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401, creates CUDA context
except Exception:
    trt = None
    cuda = None

IMAGE_CLASS_BACKEND = "resnet_trt"
IMAGE_CLASS_YOLO_MODEL_PATH = "/home/nvidia/catkin_ws/src/uav_inventory/models/image_target_classifier.engine"
IMAGE_CLASS_YOLO_TASK = "classify"
IMAGE_CLASS_YOLO_CONF = 0.25

IMAGE_CLASS_RESNET_ENGINE_PATH = "/home/nvidia/catkin_ws/src/uav_inventory/models/resnet18_4class_fp16.engine"
IMAGE_CLASS_RESNET_CLASSES_PATH = "/home/nvidia/catkin_ws/src/uav_inventory/models/classes.json"
IMAGE_CLASS_RESNET_INPUT_SIZE = 224
IMAGE_CLASS_RESNET_MEAN = [0.485, 0.456, 0.406]
IMAGE_CLASS_RESNET_STD = [0.229, 0.224, 0.225]

# =========================
# Down camera dataset capture
# =========================
# 收到 /uav/scan_enable=True 后，新建一个采集文件夹；
# 当 scan_target 属于 DOWN_CAPTURE_TARGETS 时，后台保存下视相机图像。
# 默认只在 image_target 识别时采集，避免 QR / 特殊靶 / 圆环阶段混入太多无关图。
DOWN_CAPTURE_ENABLE = True
DOWN_CAPTURE_TARGETS = ["image_target"]
DOWN_CAPTURE_ROOT = "/home/nvidia/catkin_ws/src/uav_inventory/down_capture_dataset"
DOWN_CAPTURE_MAX_FPS = 8.0
DOWN_CAPTURE_JPEG_QUALITY = 95
DOWN_CAPTURE_QUEUE_SIZE = 300
DOWN_CAPTURE_SAVE_RAW = True
DOWN_CAPTURE_SAVE_WARP = True
DOWN_CAPTURE_SAVE_CROP = True


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def parse_float_list(value, default):
    """Parse ROS param list/string into a float list."""
    try:
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            return [float(x.strip()) for x in value.split(",") if x.strip() != ""]
    except Exception:
        pass
    return list(default)


def parse_string_list(value, default):
    """Parse ROS param list/string into a normalized string list."""
    try:
        if isinstance(value, (list, tuple)):
            return [str(x).strip().lower() for x in value if str(x).strip() != ""]
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            value = value.replace(";", ",")
            return [x.strip().strip("'\"").lower() for x in value.split(",") if x.strip() != ""]
    except Exception:
        pass
    return [str(x).strip().lower() for x in default]


class ResNetTRTClassifier:
    """Minimal TensorRT classifier for fixed-shape ResNet18 engine.

    Expected network IO:
        input:  NCHW float tensor, usually 1x3x224x224
        output: logits, usually 1x4
    """

    def __init__(self, engine_path, class_names, input_size=224, mean=None, std=None):
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT or PyCUDA is not available")

        self.engine_path = os.path.expanduser(engine_path)
        self.class_names = list(class_names)
        self.input_size = int(input_size)
        self.mean = np.array(mean or IMAGE_CLASS_RESNET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std or IMAGE_CLASS_RESNET_STD, dtype=np.float32).reshape(3, 1, 1)

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine failed: %s" % self.engine_path)

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self._allocate_buffers()

    def _binding_shape(self, binding_name):
        shape = tuple(int(x) for x in self.engine.get_binding_shape(binding_name))
        if any(x < 0 for x in shape):
            # This node exports static ONNX by default, but keep a safe fallback.
            if self.engine.binding_is_input(binding_name):
                shape = (1, 3, self.input_size, self.input_size)
                idx = self.engine.get_binding_index(binding_name)
                self.context.set_binding_shape(idx, shape)
            else:
                idx = self.engine.get_binding_index(binding_name)
                shape = tuple(int(x) for x in self.context.get_binding_shape(idx))
        return shape

    def _allocate_buffers(self):
        for binding in self.engine:
            shape = self._binding_shape(binding)
            size = int(trt.volume(shape))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            item = {
                "name": binding,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }

            if self.engine.binding_is_input(binding):
                self.inputs.append(item)
            else:
                self.outputs.append(item)

        if len(self.inputs) != 1 or len(self.outputs) < 1:
            raise RuntimeError("Unexpected TensorRT bindings: inputs=%d outputs=%d" % (
                len(self.inputs), len(self.outputs)
            ))

    def preprocess(self, image_bgr):
        img = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, axis=0)

        input_dtype = self.inputs[0]["dtype"]
        if input_dtype == np.float16:
            img = img.astype(np.float16)
        else:
            img = img.astype(np.float32)

        return np.ascontiguousarray(img)

    def infer_logits(self, image_bgr):
        x = self.preprocess(image_bgr)
        np.copyto(self.inputs[0]["host"], x.ravel())

        for item in self.inputs:
            cuda.memcpy_htod_async(item["device"], item["host"], self.stream)

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        for item in self.outputs:
            cuda.memcpy_dtoh_async(item["host"], item["device"], self.stream)

        self.stream.synchronize()
        return np.array(self.outputs[0]["host"], dtype=np.float32).reshape(-1)

    def predict(self, image_bgr):
        logits = self.infer_logits(image_bgr)
        logits = logits[:len(self.class_names)]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        cls_id = int(np.argmax(probs))
        conf = float(probs[cls_id])
        name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
        return str(name).strip().lower(), conf


class StableCounter:
    """Tracks whether a target result stays stable across consecutive frames."""

    def __init__(self, max_jump_px):
        self.max_jump_px = float(max_jump_px)
        self.last_key = None
        self.last_center = None
        self.count = 0

    def reset(self):
        self.last_key = None
        self.last_center = None
        self.count = 0

    def update(self, key, center):
        if center is None:
            self.reset()
            return 0

        center = np.array(center, dtype=np.float32)

        if self.last_center is None or self.last_key != key:
            self.last_key = key
            self.last_center = center
            self.count = 1
            return self.count

        jump = float(np.linalg.norm(center - self.last_center))

        if jump <= self.max_jump_px:
            self.count += 1
        else:
            self.count = 1

        self.last_key = key
        self.last_center = center
        return self.count


class Stage1VisionNode:
    def __init__(self):
        rospy.init_node("stage1_vision_node")

        # ---------- FSM interface ----------
        self.scan_enabled = False
        self.scan_target = "none"
        self.last_target = "none"

        self.scan_enable_topic = rospy.get_param("~scan_enable_topic", "/uav/scan_enable")
        self.scan_target_topic = rospy.get_param("~scan_target_topic", "/uav/scan_target")
        self.qr_text_topic = rospy.get_param("~qr_text_topic", "/uav/qr_text")
        self.vision_result_topic = rospy.get_param("~vision_result_topic", "/uav/vision_result")

        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 12.0))
        self.disabled_read_rate_hz = float(rospy.get_param("~disabled_read_rate_hz", 3.0))
        self.show_debug = bool(rospy.get_param("~show_debug", False))

        # ---------- Down camera ----------
        self.down_camera_index = int(rospy.get_param("~down_camera_index", 0))
        self.down_width = int(rospy.get_param("~down_width", 640))
        self.down_height = int(rospy.get_param("~down_height", 480))
        self.down_fps = int(rospy.get_param("~down_fps", 30))
        self.down_auto_exposure = safe_float(rospy.get_param("~down_auto_exposure", 3), 3.0)
        self.down_gain = safe_float(rospy.get_param("~down_gain", 56), 56.0)
        self.down_board_size_m = float(rospy.get_param("~down_board_size_m", 0.80))
        self.down_qr_size_m = float(rospy.get_param("~down_qr_size_m", 0.20))
        self.down_release_u = safe_float(rospy.get_param("~down_release_u", -1.0), -1.0)
        self.down_release_v = safe_float(rospy.get_param("~down_release_v", -1.0), -1.0)
        self.body_forward_from_image_down = safe_float(
            rospy.get_param("~body_forward_from_image_down", -1.0),
            -1.0
        )
        self.body_left_from_image_right = safe_float(
            rospy.get_param("~body_left_from_image_right", -1.0),
            -1.0
        )

        if self.down_release_u < 0:
            self.down_release_u = self.down_width * 0.5
        if self.down_release_v < 0:
            self.down_release_v = self.down_height * 0.5

        self.min_square_area_px = float(rospy.get_param("~min_square_area_px", 5000.0))
        # 方形靶板几何过滤。默认按“接近正方形”的靶板收紧参数，减少背景矩形误检。
        self.square_aspect_min = float(rospy.get_param("~square_aspect_min", 0.75))
        self.square_aspect_max = float(rospy.get_param("~square_aspect_max", 1.30))
        self.square_extent_min = float(rospy.get_param("~square_extent_min", 0.55))
        self.square_angle_min_deg = float(rospy.get_param("~square_angle_min_deg", 55.0))
        self.square_angle_max_deg = float(rospy.get_param("~square_angle_max_deg", 125.0))
        self.square_center_weight = float(rospy.get_param("~square_center_weight", 0.15))
        # 特殊靶发布保护：避免第一帧误识别到背景方框就直接 detected=True。
        self.special_min_stable_count = int(rospy.get_param("~special_min_stable_count", 3))
        self.special_min_publish_confidence = float(rospy.get_param("~special_min_publish_confidence", 0.60))
        self.image_class_crop_ratio = float(rospy.get_param("~image_class_crop_ratio", 0.50))
        self.image_class_conf = float(rospy.get_param("~image_class_conf", IMAGE_CLASS_YOLO_CONF))
        self.min_result_confidence = float(rospy.get_param("~min_result_confidence", 0.20))

        # ---------- Down camera dataset capture ----------
        self.down_capture_enable = bool(rospy.get_param("~down_capture_enable", DOWN_CAPTURE_ENABLE))
        self.down_capture_targets = parse_string_list(
            rospy.get_param("~down_capture_targets", DOWN_CAPTURE_TARGETS),
            DOWN_CAPTURE_TARGETS
        )
        self.down_capture_root = os.path.expanduser(str(rospy.get_param(
            "~down_capture_root",
            DOWN_CAPTURE_ROOT
        )))
        self.down_capture_max_fps = float(rospy.get_param("~down_capture_max_fps", DOWN_CAPTURE_MAX_FPS))
        self.down_capture_jpeg_quality = int(rospy.get_param("~down_capture_jpeg_quality", DOWN_CAPTURE_JPEG_QUALITY))
        self.down_capture_queue_size = int(rospy.get_param("~down_capture_queue_size", DOWN_CAPTURE_QUEUE_SIZE))
        self.down_capture_save_raw = bool(rospy.get_param("~down_capture_save_raw", DOWN_CAPTURE_SAVE_RAW))
        self.down_capture_save_warp = bool(rospy.get_param("~down_capture_save_warp", DOWN_CAPTURE_SAVE_WARP))
        self.down_capture_save_crop = bool(rospy.get_param("~down_capture_save_crop", DOWN_CAPTURE_SAVE_CROP))
        self.down_capture_active = False
        self.down_capture_session_dir = None
        self.down_capture_session_name = ""
        self.down_capture_index = 0
        self.down_capture_last_time = 0.0
        self.down_capture_queue = None
        self.down_capture_stop_event = threading.Event()
        self.down_capture_thread = None

        # ---------- QR ----------
        self.qr_detector = cv2.QRCodeDetector()
        self.qr_valid_regex = rospy.get_param(
            "~qr_valid_regex",
            r"^[a-zA-Z0-9_ -]+,[a-zA-Z0-9_ -]+,(left|right)$"
        )
        self.last_qr_text = ""
        self.qr_publish_repeat = bool(rospy.get_param("~qr_publish_repeat", True))

        # ---------- Ring / front RealSense ----------
        self.enable_front_realsense = bool(rospy.get_param("~enable_front_realsense", True))
        self.front_width = int(rospy.get_param("~front_width", 848))
        self.front_height = int(rospy.get_param("~front_height", 480))
        self.front_fps = int(rospy.get_param("~front_fps", 30))
        self.ring_outer_diameter_m = float(rospy.get_param("~ring_outer_diameter_m", 1.20))
        self.ring_use_depth = bool(rospy.get_param("~ring_use_depth", False))
        self.ring_min_radius_px = float(rospy.get_param("~ring_min_radius_px", 35.0))
        self.ring_max_radius_px = float(rospy.get_param("~ring_max_radius_px", 360.0))
        self.ring_min_circularity = float(rospy.get_param("~ring_min_circularity", 0.75))
        self.ring_confidence_floor = float(rospy.get_param("~ring_confidence_floor", 0.35))
        # 圆环结果发布保护：至少连续稳定若干帧，且置信度足够，才把 detected 置为 True。
        self.ring_min_stable_count = int(rospy.get_param("~ring_min_stable_count", 3))
        self.ring_min_publish_confidence = float(rospy.get_param("~ring_min_publish_confidence", 0.55))
        # 外接圆填充率过滤：避免把长条边缘/支架/背景弧线强行套圆后误认为圆环。
        self.ring_fill_ratio_min = float(rospy.get_param("~ring_fill_ratio_min", 0.45))
        self.ring_fill_ratio_max = float(rospy.get_param("~ring_fill_ratio_max", 1.10))
        # Hough 兜底时不要盲目选最大圆；适当惩罚离画面中心过远的候选圆。
        self.ring_hough_center_weight = float(rospy.get_param("~ring_hough_center_weight", 0.002))

        # RealSense camera-frame to FSM body-frame signs.
        # RealSense color/depth frame convention: x right, y down, z forward.
        self.ring_left_from_camera_right = safe_float(
            rospy.get_param("~ring_left_from_camera_right", -1.0),
            -1.0
        )
        self.ring_up_from_camera_down = safe_float(
            rospy.get_param("~ring_up_from_camera_down", -1.0),
            -1.0
        )

        # ---------- Image classifier ----------
        # image_class_backend:
        #   yolo       -> use original Ultralytics YOLO classifier/detector path
        #   resnet_trt -> use native TensorRT ResNet18 engine + classes.json
        self.image_class_backend = str(rospy.get_param(
            "~image_class_backend",
            IMAGE_CLASS_BACKEND
        )).strip().lower()

        self.image_class_model_path = rospy.get_param(
            "~image_class_model_path",
            IMAGE_CLASS_YOLO_MODEL_PATH
        )
        self.image_class_model_task = rospy.get_param(
            "~image_class_model_task",
            IMAGE_CLASS_YOLO_TASK
        )

        self.image_class_resnet_engine_path = rospy.get_param(
            "~image_class_resnet_engine_path",
            IMAGE_CLASS_RESNET_ENGINE_PATH
        )
        self.image_class_resnet_classes_path = rospy.get_param(
            "~image_class_resnet_classes_path",
            IMAGE_CLASS_RESNET_CLASSES_PATH
        )
        self.image_class_resnet_input_size = int(rospy.get_param(
            "~image_class_resnet_input_size",
            IMAGE_CLASS_RESNET_INPUT_SIZE
        ))
        self.image_class_resnet_mean = parse_float_list(
            rospy.get_param("~image_class_resnet_mean", IMAGE_CLASS_RESNET_MEAN),
            IMAGE_CLASS_RESNET_MEAN
        )
        self.image_class_resnet_std = parse_float_list(
            rospy.get_param("~image_class_resnet_std", IMAGE_CLASS_RESNET_STD),
            IMAGE_CLASS_RESNET_STD
        )

        self.image_class_model = None
        self.image_class_names = {}
        self.image_class_resnet_model = None
        self.image_class_resnet_names = []

        # ---------- Runtime ----------
        self.frame_id = 0
        self.down_cap = None
        self.front_pipeline = None
        self.front_profile = None
        self.front_color_intrinsics = None

        self.trackers = {
            "image_target": StableCounter(rospy.get_param("~image_stable_jump_px", 25.0)),
            "special_target": StableCounter(rospy.get_param("~special_stable_jump_px", 25.0)),
            "ring_gate": StableCounter(rospy.get_param("~ring_stable_jump_px", 22.0)),
        }

        self.qr_pub = rospy.Publisher(self.qr_text_topic, String, queue_size=10)
        self.vision_result_pub = rospy.Publisher(self.vision_result_topic, String, queue_size=10)

        rospy.Subscriber(self.scan_enable_topic, Bool, self.scan_enable_cb, queue_size=5)
        rospy.Subscriber(self.scan_target_topic, String, self.scan_target_cb, queue_size=5)

        self.start_down_capture_writer()
        self.load_image_classifier()
        self.open_down_camera()
        self.open_front_realsense()

        rospy.loginfo("stage1_vision_node started.")
        rospy.loginfo("Subscribe: %s, %s", self.scan_enable_topic, self.scan_target_topic)
        rospy.loginfo("Publish: %s, %s", self.qr_text_topic, self.vision_result_topic)

    # =========================
    # Initialization
    # =========================

    def load_image_classifier(self):
        backend = self.image_class_backend

        if backend in ["yolo", "ultralytics"]:
            self.load_yolo_image_classifier()
        elif backend in ["resnet", "resnet_trt", "tensorrt", "trt"]:
            self.load_resnet_trt_image_classifier()
        else:
            rospy.logwarn(
                "Unknown image_class_backend=%s. Use 'yolo' or 'resnet_trt'. image_target classification disabled.",
                backend
            )

    def load_yolo_image_classifier(self):
        if YOLO is None:
            rospy.logwarn("ultralytics.YOLO import failed. YOLO image_target classification disabled.")
            return

        if not self.image_class_model_path:
            rospy.logwarn("image_class_model_path is empty. YOLO image_target classification disabled.")
            return

        if not os.path.exists(os.path.expanduser(self.image_class_model_path)):
            rospy.logwarn(
                "YOLO image classifier model not found: %s. Set _image_class_model_path.",
                self.image_class_model_path
            )
            return

        try:
            model_path = os.path.expanduser(self.image_class_model_path)
            rospy.loginfo("Loading YOLO image classifier: %s", model_path)
            self.image_class_model = YOLO(model_path, task=self.image_class_model_task)
            self.image_class_names = getattr(self.image_class_model, "names", {}) or {}
            rospy.loginfo("YOLO image classifier loaded. classes=%d", len(self.image_class_names))
        except Exception as e:
            self.image_class_model = None
            rospy.logerr("Failed to load YOLO image classifier: %s", str(e))

    def load_resnet_trt_image_classifier(self):
        engine_path = os.path.expanduser(self.image_class_resnet_engine_path)
        classes_path = os.path.expanduser(self.image_class_resnet_classes_path)

        if trt is None or cuda is None:
            rospy.logerr("TensorRT/PyCUDA import failed. ResNet TensorRT classification disabled.")
            return

        if not os.path.exists(engine_path):
            rospy.logerr("ResNet TensorRT engine not found: %s", engine_path)
            return

        if not os.path.exists(classes_path):
            rospy.logerr("ResNet classes.json not found: %s", classes_path)
            return

        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                names = json.load(f)

            if isinstance(names, dict):
                # Accept both {"0": "beer"} and {0: "beer"}-like mappings.
                names = [names[str(i)] if str(i) in names else names[i] for i in range(len(names))]

            self.image_class_resnet_names = [str(x).strip().lower() for x in names]

            rospy.loginfo("Loading ResNet TensorRT classifier: %s", engine_path)
            rospy.loginfo("Loading ResNet classes: %s -> %s", classes_path, self.image_class_resnet_names)

            self.image_class_resnet_model = ResNetTRTClassifier(
                engine_path=engine_path,
                class_names=self.image_class_resnet_names,
                input_size=self.image_class_resnet_input_size,
                mean=self.image_class_resnet_mean,
                std=self.image_class_resnet_std
            )

            rospy.loginfo("ResNet TensorRT classifier loaded. classes=%d", len(self.image_class_resnet_names))
        except Exception as e:
            self.image_class_resnet_model = None
            rospy.logerr("Failed to load ResNet TensorRT classifier: %s", str(e))

    def open_down_camera(self):
        self.down_cap = cv2.VideoCapture(self.down_camera_index)
        self.down_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.down_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.down_width)
        self.down_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.down_height)
        self.down_cap.set(cv2.CAP_PROP_FPS, self.down_fps)
        self.down_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.down_auto_exposure)
        self.down_cap.set(cv2.CAP_PROP_GAIN, self.down_gain)

        if not self.down_cap.isOpened():
            rospy.logerr("Cannot open down camera /dev/video%d", self.down_camera_index)
            self.down_cap = None
            return

        rospy.loginfo(
            "Down camera opened: /dev/video%d %dx%d@%d",
            self.down_camera_index,
            self.down_width,
            self.down_height,
            self.down_fps
        )

    def open_front_realsense(self):
        if not self.enable_front_realsense:
            rospy.logwarn("Front RealSense disabled by parameter.")
            return

        if rs is None:
            rospy.logwarn("pyrealsense2 import failed. ring_gate localization disabled.")
            return

        try:
            self.front_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color,
                self.front_width,
                self.front_height,
                rs.format.bgr8,
                self.front_fps
            )
            config.enable_stream(
                rs.stream.depth,
                self.front_width,
                self.front_height,
                rs.format.z16,
                self.front_fps
            )

            self.front_profile = self.front_pipeline.start(config)
            color_profile = self.front_profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.front_color_intrinsics = color_profile.get_intrinsics()
            self.configure_front_camera()

            rospy.loginfo(
                "Front RealSense D435i opened: %dx%d@%d",
                self.front_width,
                self.front_height,
                self.front_fps
            )
        except Exception as e:
            rospy.logwarn("Front RealSense start failed. ring_gate disabled: %s", str(e))
            self.front_pipeline = None
            self.front_profile = None
            self.front_color_intrinsics = None

    def configure_front_camera(self):
        try:
            color_sensor = self.front_profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if color_sensor.supports(rs.option.sharpness):
                color_sensor.set_option(rs.option.sharpness, 70)
            if color_sensor.supports(rs.option.contrast):
                color_sensor.set_option(rs.option.contrast, 55)
        except Exception as e:
            rospy.logwarn("Front camera option setup failed, ignored: %s", str(e))

    # =========================
    # ROS callbacks and publishing
    # =========================

    def scan_enable_cb(self, msg):
        previous_enabled = self.scan_enabled
        self.scan_enabled = bool(msg.data)

        if self.scan_enabled and not previous_enabled:
            self.start_new_down_capture_session()

        if not self.scan_enabled:
            self.reset_trackers()
            if self.down_capture_active:
                rospy.loginfo(
                    "Down capture session closed: %s saved_count=%d",
                    str(self.down_capture_session_dir),
                    int(self.down_capture_index)
                )
            self.down_capture_active = False

    def scan_target_cb(self, msg):
        target = msg.data.strip().lower()
        if target == "":
            target = "none"

        if target != self.scan_target:
            rospy.loginfo("Vision target changed: %s -> %s", self.scan_target, target)
            self.scan_target = target
            self.reset_trackers()

    def reset_trackers(self):
        for tracker in self.trackers.values():
            tracker.reset()

    def publish_result(self, data):
        data["stamp"] = rospy.Time.now().to_sec()
        self.vision_result_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def publish_not_found(self, target, reason):
        self.publish_result({
            "target": target,
            "detected": False,
            "reason": reason,
            "confidence": 0.0,
            "stable_count": 0
        })

    # =========================
    # Down camera dataset capture
    # =========================

    def start_down_capture_writer(self):
        if not self.down_capture_enable:
            rospy.loginfo("Down capture disabled by parameter.")
            return

        self.down_capture_queue = queue.Queue(maxsize=max(1, self.down_capture_queue_size))
        self.down_capture_stop_event.clear()
        self.down_capture_thread = threading.Thread(
            target=self.down_capture_worker,
            name="down_capture_writer",
            daemon=True
        )
        self.down_capture_thread.start()

        rospy.loginfo(
            "Down capture enabled: root=%s targets=%s max_fps=%.1f raw=%s warp=%s crop=%s",
            self.down_capture_root,
            str(self.down_capture_targets),
            self.down_capture_max_fps,
            str(self.down_capture_save_raw),
            str(self.down_capture_save_warp),
            str(self.down_capture_save_crop)
        )

    def stop_down_capture_writer(self):
        if self.down_capture_thread is None:
            return

        self.down_capture_stop_event.set()
        try:
            self.down_capture_thread.join(timeout=3.0)
        except Exception:
            pass
        self.down_capture_thread = None

    def down_capture_worker(self):
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(clamp(self.down_capture_jpeg_quality, 30, 100))
        ]

        while not self.down_capture_stop_event.is_set() or (
            self.down_capture_queue is not None and not self.down_capture_queue.empty()
        ):
            try:
                out_path, image = self.down_capture_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                continue

            try:
                out_path = os.path.expanduser(str(out_path))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", image, encode_params)
                if ok:
                    encoded.tofile(out_path)
                else:
                    rospy.logwarn_throttle(2.0, "Down capture cv2.imencode failed: %s", out_path)
            except Exception as e:
                rospy.logwarn_throttle(2.0, "Down capture save failed: %s", str(e))
            finally:
                try:
                    self.down_capture_queue.task_done()
                except Exception:
                    pass

    def start_new_down_capture_session(self):
        if not self.down_capture_enable:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.down_capture_session_name = "stage1_down_capture_%s" % timestamp
        self.down_capture_session_dir = os.path.join(self.down_capture_root, self.down_capture_session_name)

        try:
            os.makedirs(self.down_capture_session_dir, exist_ok=True)
            if self.down_capture_save_raw:
                os.makedirs(os.path.join(self.down_capture_session_dir, "raw"), exist_ok=True)
            if self.down_capture_save_warp:
                os.makedirs(os.path.join(self.down_capture_session_dir, "warp"), exist_ok=True)
            if self.down_capture_save_crop:
                os.makedirs(os.path.join(self.down_capture_session_dir, "crop"), exist_ok=True)
        except Exception as e:
            rospy.logwarn("Failed to create down capture session folder: %s", str(e))
            self.down_capture_active = False
            return

        self.down_capture_index = 0
        self.down_capture_last_time = 0.0
        self.down_capture_active = True

        rospy.loginfo("Down capture session started: %s", self.down_capture_session_dir)

    def down_capture_should_save(self):
        if not self.down_capture_enable:
            return False
        if not self.down_capture_active:
            return False
        if self.down_capture_queue is None:
            return False

        target = str(self.scan_target).strip().lower()
        targets = [str(x).strip().lower() for x in self.down_capture_targets]
        if "*" not in targets and "all" not in targets and target not in targets:
            return False

        if self.down_capture_max_fps > 0:
            now = time.time()
            interval = 1.0 / max(0.1, self.down_capture_max_fps)
            if now - self.down_capture_last_time < interval:
                return False
            self.down_capture_last_time = now

        return True

    def sanitize_capture_token(self, text, default="unknown"):
        text = str(text).strip().lower()
        if text == "":
            text = default
        text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
        text = text.strip("_")
        return text if text else default

    def enqueue_down_capture(self, subdir, image, filename):
        if image is None or self.down_capture_queue is None:
            return

        out_path = os.path.join(self.down_capture_session_dir, subdir, filename)
        try:
            self.down_capture_queue.put_nowait((out_path, image.copy()))
        except queue.Full:
            rospy.logwarn_throttle(2.0, "Down capture queue full, dropping images.")

    def capture_down_sample_set(self, frame, square=None, board_warp=None, class_name="unknown", reason="raw"):
        if not self.down_capture_should_save():
            return

        if self.down_capture_session_dir is None:
            self.start_new_down_capture_session()
            if not self.down_capture_active:
                return

        self.down_capture_index += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        target = self.sanitize_capture_token(self.scan_target, "target")
        label = self.sanitize_capture_token(class_name, "unknown")
        reason = self.sanitize_capture_token(reason, "reason")
        prefix = "%06d_%s_%s_%s_%s" % (
            int(self.down_capture_index),
            ts,
            target,
            label,
            reason
        )

        if self.down_capture_save_raw:
            self.enqueue_down_capture("raw", frame, prefix + "_raw.jpg")

        if board_warp is not None and self.down_capture_save_warp:
            self.enqueue_down_capture("warp", board_warp, prefix + "_warp.jpg")

        if board_warp is not None and self.down_capture_save_crop:
            try:
                crop = self.center_crop(board_warp, self.image_class_crop_ratio)
                self.enqueue_down_capture("crop", crop, prefix + "_crop.jpg")
            except Exception as e:
                rospy.logwarn_throttle(2.0, "Down capture crop failed: %s", str(e))

    # =========================
    # Down camera helpers
    # =========================

    def read_down_frame(self):
        if self.down_cap is None:
            return None

        ret, frame = self.down_cap.read()
        if not ret:
            rospy.logwarn_throttle(2.0, "Failed to read down camera frame.")
            return None

        return frame

    def order_quad_points(self, pts):
        pts = np.array(pts, dtype=np.float32).reshape(4, 2)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]

        # Current order is roughly top-left, top-right, bottom-right, bottom-left
        # after rotating to put the smallest x+y first.
        start = int(np.argmin(pts[:, 0] + pts[:, 1]))
        pts = np.roll(pts, -start, axis=0)

        # Ensure clockwise TL, TR, BR, BL.
        if pts[1][0] < pts[3][0]:
            pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)

        return pts

    def quad_corner_angles_deg(self, quad):
        """
        计算四边形四个角的角度。
        作用：approxPolyDP 只能保证轮廓有 4 个点，不能保证它真的像矩形；
        这里额外要求角度不要太离谱，用来过滤梯形、残缺边缘和不规则背景轮廓。
        """
        quad = np.array(quad, dtype=np.float32).reshape(4, 2)
        angles = []

        for i in range(4):
            p_prev = quad[(i - 1) % 4]
            p_curr = quad[i]
            p_next = quad[(i + 1) % 4]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 1e-6 or n2 <= 1e-6:
                return []

            cos_angle = float(np.dot(v1, v2) / (n1 * n2))
            cos_angle = clamp(cos_angle, -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_angle)))

        return angles

    def find_square_board(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        candidates = []

        # Edge-based pass.
        edges = cv2.Canny(blur, 60, 160)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

        # Threshold-based pass for high-contrast board borders.
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

        best = None
        best_score = -1.0
        image_center = np.array([frame.shape[1] * 0.5, frame.shape[0] * 0.5], dtype=np.float32)

        for contour in candidates:
            area = float(cv2.contourArea(contour))
            if area < self.min_square_area_px:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue

            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = self.order_quad_points(approx.reshape(4, 2))
            side_lengths = [
                np.linalg.norm(quad[(i + 1) % 4] - quad[i])
                for i in range(4)
            ]
            min_side = min(side_lengths)
            max_side = max(side_lengths)
            if min_side <= 1e-6:
                continue

            # 边长比例过滤：对正方形靶板更严格，减少普通矩形背景误检。
            aspect = max_side / min_side
            if aspect < self.square_aspect_min or aspect > self.square_aspect_max:
                continue

            # 轮廓饱满度过滤：避免细长边缘、破碎线框、支架边缘被四边形拟合后误认为靶板。
            x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
            rect_area = float(w * h)
            extent = area / rect_area if rect_area > 1e-6 else 0.0
            if extent < self.square_extent_min:
                continue

            # 角度过滤：approxPolyDP 得到四个点后，再检查它是否真的像矩形。
            angles = self.quad_corner_angles_deg(quad)
            if len(angles) != 4:
                continue
            min_angle = min(angles)
            max_angle = max(angles)
            if min_angle < self.square_angle_min_deg or max_angle > self.square_angle_max_deg:
                continue

            center = np.mean(quad, axis=0)
            center_dist = float(np.linalg.norm(center - image_center))
            score = area - self.square_center_weight * center_dist * center_dist

            if score > best_score:
                best_score = score
                best = {
                    "quad": quad,
                    "center": center,
                    "area": area,
                    "aspect": aspect,
                    "extent": float(extent),
                    "min_angle_deg": float(min_angle),
                    "max_angle_deg": float(max_angle),
                    "score": score
                }

        return best

    def board_homography(self, quad, board_size_m):
        half = board_size_m * 0.5
        board_pts = np.array(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            dtype=np.float32
        )
        h_img_to_board = cv2.getPerspectiveTransform(quad.astype(np.float32), board_pts)
        h_board_to_img = cv2.getPerspectiveTransform(board_pts, quad.astype(np.float32))
        return h_img_to_board, h_board_to_img

    def perspective_point(self, h, point):
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, h)
        return dst.reshape(2)

    def compute_down_offsets_from_square(self, square):
        quad = square["quad"]
        h_img_to_board, _ = self.board_homography(quad, self.down_board_size_m)

        release_px = np.array([self.down_release_u, self.down_release_v], dtype=np.float32)
        release_board = self.perspective_point(h_img_to_board, release_px)

        # Board coordinates are x=image-right, y=image-down. Target center is board origin.
        delta_right_m = -float(release_board[0])
        delta_down_m = -float(release_board[1])

        offset_x_m = self.body_forward_from_image_down * delta_down_m
        offset_y_m = self.body_left_from_image_right * delta_right_m

        target_center_px = np.mean(quad, axis=0)
        offset_px = [
            float(target_center_px[0] - self.down_release_u),
            float(target_center_px[1] - self.down_release_v)
        ]

        return offset_x_m, offset_y_m, target_center_px, offset_px

    def warp_square_board(self, frame, square, output_size=416):
        quad = square["quad"]
        dst = np.array(
            [
                [0, 0],
                [output_size - 1, 0],
                [output_size - 1, output_size - 1],
                [0, output_size - 1],
            ],
            dtype=np.float32
        )
        h = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(frame, h, (output_size, output_size))

    def center_crop(self, image, ratio):
        ratio = clamp(float(ratio), 0.10, 1.0)
        h, w = image.shape[:2]
        crop_w = int(w * ratio)
        crop_h = int(h * ratio)
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        return image[y1:y1 + crop_h, x1:x1 + crop_w]

    # =========================
    # QR
    # =========================

    def normalize_qr_text(self, text):
        text = str(text).strip()
        text = re.sub(r"\s+", "", text)
        parts = [p.strip().lower() for p in text.split(",")]

        if len(parts) != 3:
            return ""

        if parts[2] not in ["left", "right"]:
            return ""

        normalized = "{},{},{}".format(parts[0], parts[1], parts[2])

        if not re.match(self.qr_valid_regex, normalized):
            return ""

        return normalized

    def qr_preprocess_candidates(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
        sharpen = cv2.addWeighted(gray_clahe, 1.6, blur, -0.6, 0)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5
        )
        return [
            ("raw", frame),
            ("gray", gray),
            ("clahe", gray_clahe),
            ("sharpen", sharpen),
            ("adaptive", adaptive),
        ]

    def decode_qr_opencv(self, frame):
        for method, img in self.qr_preprocess_candidates(frame):
            try:
                data, points, _ = self.qr_detector.detectAndDecode(img)
            except Exception:
                continue

            data = self.normalize_qr_text(data)
            if data and points is not None:
                pts = np.array(points, dtype=np.float32).reshape(-1, 2)
                return data, pts, "opencv_" + method

        return "", None, ""

    def decode_qr_pyzbar(self, frame):
        if pyzbar is None:
            return "", None, ""

        for method, img in self.qr_preprocess_candidates(frame):
            if len(img.shape) == 3:
                decode_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                decode_img = img

            try:
                results = pyzbar.decode(decode_img)
            except Exception:
                continue

            if not results:
                continue

            best = max(results, key=lambda r: r.rect.width * r.rect.height)
            try:
                text = best.data.decode("utf-8", errors="replace")
            except Exception:
                text = str(best.data)

            text = self.normalize_qr_text(text)
            if not text:
                continue

            polygon = best.polygon
            if polygon and len(polygon) >= 4:
                pts = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
            else:
                x, y, w, h = best.rect
                pts = np.array(
                    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    dtype=np.float32
                )

            return text, pts, "pyzbar_" + method

        return "", None, ""

    def process_qr(self, frame):
        text, pts, method = self.decode_qr_opencv(frame)

        if not text:
            text, pts, method = self.decode_qr_pyzbar(frame)

        if not text:
            rospy.loginfo_throttle(1.0, "[QR] not found")
            return

        if self.qr_publish_repeat or text != self.last_qr_text:
            self.qr_pub.publish(String(data=text))

        self.last_qr_text = text

        center = np.mean(pts, axis=0) if pts is not None else [-1.0, -1.0]
        area = float(abs(cv2.contourArea(pts))) if pts is not None and len(pts) >= 4 else 0.0

        rospy.loginfo_throttle(
            0.3,
            "[QR] text=%s center=(%.1f, %.1f) area=%.0f method=%s",
            text,
            center[0],
            center[1],
            area,
            method
        )

        if self.show_debug:
            debug = frame.copy()
            if pts is not None and len(pts) >= 4:
                cv2.polylines(debug, [pts.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(debug, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("stage1_down_qr", debug)
            cv2.waitKey(1)

    # =========================
    # Image target and special target
    # =========================

    def classify_image_target(self, board_warp):
        backend = self.image_class_backend

        if backend in ["resnet", "resnet_trt", "tensorrt", "trt"]:
            return self.classify_image_target_resnet_trt(board_warp)

        return self.classify_image_target_yolo(board_warp)

    def classify_image_target_resnet_trt(self, board_warp):
        if self.image_class_resnet_model is None:
            return "", 0.0, "resnet_model_not_loaded"

        crop = self.center_crop(board_warp, self.image_class_crop_ratio)

        try:
            class_name, conf = self.image_class_resnet_model.predict(crop)
            return class_name, conf, "resnet_trt"
        except Exception as e:
            return "", 0.0, "resnet_predict_failed:%s" % str(e)

    def classify_image_target_yolo(self, board_warp):
        if self.image_class_model is None:
            return "", 0.0, "model_not_loaded"

        crop = self.center_crop(board_warp, self.image_class_crop_ratio)

        try:
            results = self.image_class_model.predict(
                crop,
                conf=self.image_class_conf,
                verbose=False
            )
        except Exception as e:
            return "", 0.0, "predict_failed:%s" % str(e)

        if not results:
            return "", 0.0, "empty_result"

        result = results[0]

        # Classification model path.
        probs = getattr(result, "probs", None)
        if probs is not None and getattr(probs, "top1", None) is not None:
            cls_id = int(probs.top1)
            conf = float(probs.top1conf)
            name = self.image_class_names.get(cls_id, str(cls_id))
            return str(name).strip().lower(), conf, "yolo_classify"

        # Detection model compatibility path. This is not the default plan, but keeps
        # the code easy to extend if the classifier is later replaced by a detector.
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            best_box = max(boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            name = self.image_class_names.get(cls_id, str(cls_id))
            return str(name).strip().lower(), conf, "yolo_detect_compat"

        return "", 0.0, "no_class"

    def process_image_target(self, frame):
        square = self.find_square_board(frame)
        if square is None:
            self.capture_down_sample_set(frame, class_name="unknown", reason="square_not_found")
            self.trackers["image_target"].reset()
            self.publish_not_found("image_target", "square_not_found")
            rospy.loginfo_throttle(1.0, "[IMAGE] square not found")
            return

        offset_x_m, offset_y_m, center_px, offset_px = self.compute_down_offsets_from_square(square)
        board_warp = self.warp_square_board(frame, square)
        class_name, yolo_conf, method = self.classify_image_target(board_warp)

        if class_name == "":
            self.capture_down_sample_set(
                frame,
                square=square,
                board_warp=board_warp,
                class_name="unknown",
                reason=method
            )
            self.trackers["image_target"].reset()
            self.publish_result({
                "target": "image_target",
                "detected": False,
                "reason": method,
                "confidence": float(yolo_conf),
                "stable_count": 0,
                "target_center_px": [float(center_px[0]), float(center_px[1])],
                "release_center_px": [float(self.down_release_u), float(self.down_release_v)],
                "offset_px": offset_px,
                "board_area_px": float(square["area"]),
                "mapping_method": "square_homography_800mm"
            })
            return

        stable_count = self.trackers["image_target"].update(class_name, center_px)
        confidence = clamp(0.15 + 0.85 * yolo_conf, 0.0, 1.0)
        detected = confidence >= self.min_result_confidence

        self.publish_result({
            "target": "image_target",
            "detected": bool(detected),
            "reason": "ok" if detected else "low_confidence",
            "class_name": class_name,
            "offset_x_m": float(offset_x_m),
            "offset_y_m": float(offset_y_m),
            "confidence": float(confidence),
            "stable_count": int(stable_count if detected else 0),
            "target_center_px": [float(center_px[0]), float(center_px[1])],
            "release_center_px": [float(self.down_release_u), float(self.down_release_v)],
            "offset_px": offset_px,
            "board_area_px": float(square["area"]),
            "board_aspect": float(square["aspect"]),
            "mapping_method": "square_homography_800mm",
            "classifier_method": method
        })

        rospy.loginfo_throttle(
            0.3,
            "[IMAGE] class=%s conf=%.2f offset=(%.3f, %.3f) stable=%d",
            class_name,
            confidence,
            offset_x_m,
            offset_y_m,
            stable_count
        )

        self.capture_down_sample_set(
            frame,
            square=square,
            board_warp=board_warp,
            class_name=class_name,
            reason=method
        )

        if self.show_debug:
            self.draw_down_square_debug("stage1_down_image", frame, square, class_name)

    def process_special_target(self, frame):
        square = self.find_square_board(frame)
        if square is None:
            self.trackers["special_target"].reset()
            self.publish_not_found("special_target", "square_not_found")
            rospy.loginfo_throttle(1.0, "[SPECIAL] square not found")
            return

        offset_x_m, offset_y_m, center_px, offset_px = self.compute_down_offsets_from_square(square)
        stable_count = self.trackers["special_target"].update("special_target", center_px)

        # Geometry-based confidence. This will be replaced or supplemented if a
        # dedicated special-target detector is added later.
        area_score = clamp(square["area"] / (self.down_width * self.down_height * 0.25), 0.0, 1.0)
        aspect_score = clamp(1.0 - abs(1.0 - square["aspect"]), 0.0, 1.0)
        extent_score = clamp((square.get("extent", 0.0) - self.square_extent_min) / 0.35, 0.0, 1.0)
        confidence = clamp(
            0.25 + 0.30 * area_score + 0.25 * aspect_score + 0.20 * extent_score,
            0.0,
            1.0
        )
        detected = (
            stable_count >= self.special_min_stable_count and
            confidence >= self.special_min_publish_confidence
        )

        self.publish_result({
            "target": "special_target",
            "detected": bool(detected),
            "reason": "ok" if detected else "unstable_or_low_confidence",
            "offset_x_m": float(offset_x_m),
            "offset_y_m": float(offset_y_m),
            "confidence": float(confidence),
            "stable_count": int(stable_count),
            "target_center_px": [float(center_px[0]), float(center_px[1])],
            "release_center_px": [float(self.down_release_u), float(self.down_release_v)],
            "offset_px": offset_px,
            "board_area_px": float(square["area"]),
            "board_aspect": float(square["aspect"]),
            "board_extent": float(square.get("extent", 0.0)),
            "board_min_angle_deg": float(square.get("min_angle_deg", 0.0)),
            "board_max_angle_deg": float(square.get("max_angle_deg", 0.0)),
            "mapping_method": "square_homography_800mm",
            "detector_method": "opencv_square"
        })

        rospy.loginfo_throttle(
            0.3,
            "[SPECIAL] detected=%s conf=%.2f offset=(%.3f, %.3f) stable=%d extent=%.2f angle=(%.1f, %.1f)",
            str(bool(detected)),
            confidence,
            offset_x_m,
            offset_y_m,
            stable_count,
            float(square.get("extent", 0.0)),
            float(square.get("min_angle_deg", 0.0)),
            float(square.get("max_angle_deg", 0.0))
        )

        if self.show_debug:
            self.draw_down_square_debug("stage1_down_special", frame, square, "special")

    def draw_down_square_debug(self, window, frame, square, label):
        debug = frame.copy()
        quad = square["quad"].astype(np.int32)
        cv2.polylines(debug, [quad], True, (0, 255, 0), 2)
        center = square["center"]
        cv2.circle(debug, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        cv2.circle(debug, (int(self.down_release_u), int(self.down_release_v)), 5, (255, 0, 0), -1)
        cv2.putText(debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, debug)
        cv2.waitKey(1)

    # =========================
    # Ring gate
    # =========================

    def read_front_frames(self):
        if self.front_pipeline is None:
            return None, None

        try:
            frames = self.front_pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                return None, None
            color = np.asanyarray(color_frame.get_data())
            return color, depth_frame
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Failed to read front RealSense frames: %s", str(e))
            return None, None

    def find_ring_circle(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 800.0:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue

            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            if circularity < self.ring_min_circularity:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < self.ring_min_radius_px or radius > self.ring_max_radius_px:
                continue

            circle_area = math.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 1e-6 else 0.0
            if fill_ratio < self.ring_fill_ratio_min or fill_ratio > self.ring_fill_ratio_max:
                continue

            image_center_x = frame.shape[1] * 0.5
            image_center_y = frame.shape[0] * 0.5
            center_dist = math.sqrt((cx - image_center_x) ** 2 + (cy - image_center_y) ** 2)
            score = area * circularity - 0.05 * center_dist * center_dist

            if score > best_score:
                best_score = score
                best = {
                    "center": np.array([cx, cy], dtype=np.float32),
                    "radius": float(radius),
                    "area": area,
                    "circularity": circularity,
                    "fill_ratio": float(fill_ratio),
                    "score": score,
                    "method": "contour_circle"
                }

        if best is not None:
            return best

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=80,
            param2=50,
            minRadius=int(self.ring_min_radius_px),
            maxRadius=int(self.ring_max_radius_px)
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(np.float32)
        image_center_x = frame.shape[1] * 0.5
        image_center_y = frame.shape[0] * 0.5

        def hough_score(c):
            cx, cy, radius = float(c[0]), float(c[1]), float(c[2])
            center_dist = math.sqrt((cx - image_center_x) ** 2 + (cy - image_center_y) ** 2)
            return radius - self.ring_hough_center_weight * center_dist * center_dist

        best_circle = max(circles, key=hough_score)
        return {
            "center": np.array([best_circle[0], best_circle[1]], dtype=np.float32),
            "radius": float(best_circle[2]),
            "area": float(math.pi * best_circle[2] * best_circle[2]),
            "circularity": 0.55,
            "fill_ratio": 1.0,
            "score": float(hough_score(best_circle)),
            "method": "hough_circle"
        }

    def depth_at_pixel(self, depth_frame, u, v):
        if depth_frame is None or not self.ring_use_depth:
            return 0.0

        values = []
        u = int(round(u))
        v = int(round(v))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x = u + dx
                y = v + dy
                if x < 0 or y < 0 or x >= self.front_width or y >= self.front_height:
                    continue
                d = depth_frame.get_distance(x, y)
                if d > 0.10:
                    values.append(d)

        if not values:
            return 0.0

        return float(np.median(values))

    def ring_distance_from_size(self, radius_px):
        if self.front_color_intrinsics is None:
            return 0.0

        fx = float(self.front_color_intrinsics.fx)
        diameter_px = max(1.0, 2.0 * float(radius_px))
        return self.ring_outer_diameter_m * fx / diameter_px

    def process_ring_gate(self):
        color, depth_frame = self.read_front_frames()
        if color is None:
            self.trackers["ring_gate"].reset()
            self.publish_not_found("ring_gate", "front_camera_no_frame")
            return

        ring = self.find_ring_circle(color)
        if ring is None:
            self.trackers["ring_gate"].reset()
            self.publish_not_found("ring_gate", "ring_not_found")
            rospy.loginfo_throttle(1.0, "[RING] not found")
            return

        if self.front_color_intrinsics is None:
            self.publish_not_found("ring_gate", "no_intrinsics")
            return

        center = ring["center"]
        u = float(center[0])
        v = float(center[1])

        depth_m = self.depth_at_pixel(depth_frame, u, v)
        size_m = self.ring_distance_from_size(ring["radius"])

        if depth_m > 0.10:
            forward_m = depth_m
            distance_method = "depth"
        else:
            forward_m = size_m
            distance_method = "known_outer_diameter"

        if forward_m <= 0.10:
            self.publish_not_found("ring_gate", "invalid_distance")
            return

        intr = self.front_color_intrinsics
        camera_right_m = (u - float(intr.ppx)) * forward_m / float(intr.fx)
        camera_down_m = (v - float(intr.ppy)) * forward_m / float(intr.fy)

        offset_y_m = self.ring_left_from_camera_right * camera_right_m
        offset_z_m = self.ring_up_from_camera_down * camera_down_m

        stable_count = self.trackers["ring_gate"].update("ring_gate", center)
        radius_score = clamp((ring["radius"] - self.ring_min_radius_px) / 120.0, 0.0, 1.0)
        confidence = clamp(
            self.ring_confidence_floor +
            0.35 * clamp(ring["circularity"], 0.0, 1.0) +
            0.25 * radius_score,
            0.0,
            1.0
        )
        detected = (
            stable_count >= self.ring_min_stable_count and
            confidence >= self.ring_min_publish_confidence
        )

        self.publish_result({
            "target": "ring_gate",
            "detected": bool(detected),
            "reason": "ok" if detected else "unstable_or_low_confidence",
            "forward_m": float(forward_m),
            "offset_y_m": float(offset_y_m),
            "offset_z_m": float(offset_z_m),
            "confidence": float(confidence),
            "stable_count": int(stable_count),
            "ring_center_px": [float(u), float(v)],
            "image_center_px": [float(intr.ppx), float(intr.ppy)],
            "ring_radius_px": float(ring["radius"]),
            "ring_area_px": float(ring["area"]),
            "circularity": float(ring["circularity"]),
            "fill_ratio": float(ring.get("fill_ratio", 0.0)),
            "mapping_method": distance_method,
            "detector_method": ring["method"]
        })

        rospy.loginfo_throttle(
            0.3,
            "[RING] detected=%s forward=%.2f y=%.3f z=%.3f r=%.1f conf=%.2f stable=%d method=%s",
            str(bool(detected)),
            forward_m,
            offset_y_m,
            offset_z_m,
            ring["radius"],
            confidence,
            stable_count,
            distance_method
        )

        if self.show_debug:
            debug = color.copy()
            cv2.circle(debug, (int(u), int(v)), int(ring["radius"]), (0, 255, 0), 2)
            cv2.circle(debug, (int(u), int(v)), 5, (0, 0, 255), -1)
            cv2.imshow("stage1_front_ring", debug)
            cv2.waitKey(1)

    # =========================
    # Main loop
    # =========================

    def spin(self):
        rate = rospy.Rate(self.publish_rate_hz)
        disabled_read_interval = 1.0 / max(0.1, self.disabled_read_rate_hz)
        last_disabled_read = 0.0

        try:
            while not rospy.is_shutdown():
                self.frame_id += 1

                if not self.scan_enabled or self.scan_target == "none":
                    now = time.time()
                    if now - last_disabled_read > disabled_read_interval:
                        self.read_down_frame()
                        last_disabled_read = now
                    rate.sleep()
                    continue

                if self.scan_target in ["qr", "image_target", "special_target"]:
                    frame = self.read_down_frame()
                    if frame is None:
                        if self.scan_target != "qr":
                            self.publish_not_found(self.scan_target, "down_camera_no_frame")
                        rate.sleep()
                        continue

                    if self.scan_target == "qr":
                        self.process_qr(frame)
                    elif self.scan_target == "image_target":
                        self.process_image_target(frame)
                    elif self.scan_target == "special_target":
                        self.process_special_target(frame)

                elif self.scan_target == "ring_gate":
                    self.process_ring_gate()

                else:
                    rospy.logwarn_throttle(1.0, "Unknown scan_target: %s", self.scan_target)

                rate.sleep()

        finally:
            rospy.loginfo("Stopping stage1_vision_node...")
            if self.down_cap is not None:
                self.down_cap.release()
            if self.front_pipeline is not None:
                try:
                    self.front_pipeline.stop()
                except Exception:
                    pass
            self.stop_down_capture_writer()
            if self.show_debug:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        Stage1VisionNode().spin()
    except rospy.ROSInterruptException:
        pass
