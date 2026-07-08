#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image target ResNet TensorRT classification test node.

用途：
1. 打开下视/USB 相机后立即开始识别，不依赖 /uav/scan_enable。
2. 先用 OpenCV 找方形靶板外轮廓，再透视矫正成正方形图像。
3. 对矫正后的中心区域运行 ResNet18 TensorRT engine 分类模型。
4. 识别不到时低频打印简短信息；识别到并分类成功时按帧打印详细信息。
5. ResNet engine 默认使用相对路径 models/resnet18_4class_fp16.engine。
6. 类别文件默认使用相对路径 models/classes.json。

运行示例：
    python3 image_target_resnet_test_node.py --gui
    python3 image_target_resnet_test_node.py --no-gui
    python3 image_target_resnet_test_node.py --engine models/resnet18_4class_fp16.engine --classes models/classes.json --gui
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorrt as trt
except Exception:
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  # 初始化 CUDA 上下文
except Exception:
    cuda = None

try:
    import rospy
    from std_msgs.msg import String
except Exception:
    rospy = None
    String = None


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class StableCounter:
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


def ask_gui_if_needed(args):
    if args.gui is not None:
        return bool(args.gui)

    try:
        answer = input('是否打开图形界面？NoMachine/桌面环境输入 y，纯终端输入 n [y/N]: ')
        return answer.strip().lower() in ['y', 'yes', '1', 'true']
    except Exception:
        return False


def resolve_relative_path(path_text):
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_class_names(classes_path):
    with open(classes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        names = [str(x).strip().lower() for x in data]
    elif isinstance(data, dict):
        # 兼容 {"0": "beer", "1": "bicycle"} 这种写法。
        try:
            keys = sorted(data.keys(), key=lambda k: int(k))
        except Exception:
            keys = sorted(data.keys())
        names = [str(data[k]).strip().lower() for k in keys]
    else:
        raise RuntimeError('classes.json 格式不支持，应该是 list 或 dict。')

    if len(names) == 0:
        raise RuntimeError('classes.json 为空。')

    return names


class ResNetTRTClassifier:
    def __init__(self, engine_path, classes_path, input_size=224):
        if trt is None:
            raise RuntimeError('tensorrt 导入失败，请确认 Jetson 上 TensorRT Python 可用。')
        if cuda is None:
            raise RuntimeError('pycuda 导入失败，请先安装/配置 pycuda。')

        self.engine_path = Path(engine_path)
        self.classes_path = Path(classes_path)
        self.input_size = int(input_size)
        self.class_names = load_class_names(self.classes_path)

        if not self.engine_path.exists():
            raise RuntimeError('ResNet engine not found: %s' % str(self.engine_path))
        if not self.classes_path.exists():
            raise RuntimeError('classes.json not found: %s' % str(self.classes_path))

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(str(self.engine_path), 'rb') as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError('TensorRT engine 反序列化失败: %s' % str(self.engine_path))

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context 创建失败。')

        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_binding_idx = None
        self.output_binding_idx = None

        self._allocate_buffers()

        print('[RESNET_TRT] engine=%s' % str(self.engine_path), flush=True)
        print('[RESNET_TRT] classes=%s' % str(self.class_names), flush=True)
        print('[RESNET_TRT] input_size=%d mean=ImageNet std=ImageNet' % self.input_size, flush=True)

    def _binding_shape(self, binding_idx):
        shape = tuple(self.engine.get_binding_shape(binding_idx))
        if any(dim < 0 for dim in shape):
            # 动态输入兜底。当前导出的 ResNet 一般是固定 1x3x224x224。
            shape = (1, 3, self.input_size, self.input_size)
            self.context.set_binding_shape(binding_idx, shape)
            shape = tuple(self.context.get_binding_shape(binding_idx))
        return shape

    def _allocate_buffers(self):
        for idx in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(idx)
            shape = self._binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = int(trt.volume(shape))
            if size <= 0:
                raise RuntimeError('非法 binding shape: idx=%d shape=%s' % (idx, str(shape)))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            info = {
                'idx': idx,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem,
            }

            if is_input:
                self.inputs.append(info)
                self.input_binding_idx = idx
            else:
                self.outputs.append(info)
                self.output_binding_idx = idx

        if len(self.inputs) != 1:
            raise RuntimeError('当前脚本只支持 1 个输入，实际输入数=%d' % len(self.inputs))
        if len(self.outputs) < 1:
            raise RuntimeError('未找到输出 binding。')

    def preprocess(self, image_bgr):
        img = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        chw = np.transpose(img, (2, 0, 1))
        x = np.expand_dims(chw, axis=0)
        return np.ascontiguousarray(x)

    def infer(self, image_bgr):
        input_info = self.inputs[0]
        x = self.preprocess(image_bgr)
        x = x.astype(input_info['dtype'], copy=False)

        if x.size != input_info['host'].size:
            raise RuntimeError('输入尺寸不匹配：preprocess=%s, engine_input_shape=%s' % (
                str(x.shape), str(input_info['shape'])
            ))

        np.copyto(input_info['host'], x.ravel())

        cuda.memcpy_htod_async(input_info['device'], input_info['host'], self.stream)
        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError('TensorRT execute_async_v2 执行失败。')

        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)

        self.stream.synchronize()

        logits = np.array(self.outputs[0]['host'], dtype=np.float32).reshape(-1)
        if logits.size < len(self.class_names):
            raise RuntimeError('输出维度小于类别数：output=%d classes=%d' % (logits.size, len(self.class_names)))

        logits = logits[:len(self.class_names)]
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / max(float(np.sum(exp)), 1e-12)

        cls_id = int(np.argmax(probs))
        conf = float(probs[cls_id])
        name = self.class_names[cls_id]
        return name, conf, cls_id, probs


class ImageTargetResnetTester:
    def __init__(self, args):
        self.args = args
        self.show_debug = ask_gui_if_needed(args)

        self.cap = cv2.VideoCapture(args.camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)

        if not self.cap.isOpened():
            raise RuntimeError('Cannot open camera index %d' % args.camera_index)

        self.engine_path = resolve_relative_path(args.engine)
        self.classes_path = resolve_relative_path(args.classes)
        self.model = self.load_resnet_model()

        self.tracker = StableCounter(args.stable_jump_px)
        self.last_not_found_print = 0.0
        self.last_no_class_print = 0.0

        self.ros_pub = None
        if args.ros:
            if rospy is None:
                print('[WARN] rospy 不可用，跳过 ROS 发布。', flush=True)
            else:
                rospy.init_node('image_target_resnet_test_node', anonymous=True)
                self.ros_pub = rospy.Publisher(args.vision_result_topic, String, queue_size=10)

        print('[IMAGE_RESNET] camera=/dev/video%d %dx%d@%d gui=%s' % (
            args.camera_index,
            args.width,
            args.height,
            args.fps,
            str(self.show_debug)
        ), flush=True)
        print('[IMAGE_RESNET] engine=%s' % str(self.engine_path), flush=True)
        print('[IMAGE_RESNET] classes=%s' % str(self.classes_path), flush=True)

    def load_resnet_model(self):
        if not self.engine_path.exists():
            raise RuntimeError(
                'ResNet engine not found: %s\n'
                '默认相对路径是 models/resnet18_4class_fp16.engine。请把 engine 放到脚本同目录的 models/ 下，'
                '或用 --engine 指定相对/绝对路径。' % str(self.engine_path)
            )

        if not self.classes_path.exists():
            raise RuntimeError(
                'classes.json not found: %s\n'
                '默认相对路径是 models/classes.json。请把 classes.json 放到脚本同目录的 models/ 下，'
                '或用 --classes 指定相对/绝对路径。' % str(self.classes_path)
            )

        return ResNetTRTClassifier(
            engine_path=str(self.engine_path),
            classes_path=str(self.classes_path),
            input_size=self.args.input_size
        )

    def order_quad_points(self, pts):
        pts = np.array(pts, dtype=np.float32).reshape(4, 2)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        start = int(np.argmin(pts[:, 0] + pts[:, 1]))
        pts = np.roll(pts, -start, axis=0)
        if pts[1][0] < pts[3][0]:
            pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)
        return pts

    def quad_angles_deg(self, quad):
        angles = []
        for i in range(4):
            prev_p = quad[(i - 1) % 4]
            cur_p = quad[i]
            next_p = quad[(i + 1) % 4]
            v1 = prev_p - cur_p
            v2 = next_p - cur_p
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return []
            cos_angle = float(np.dot(v1, v2) / (n1 * n2))
            cos_angle = clamp(cos_angle, -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_angle)))
        return angles

    def find_square_board(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        candidates = []

        edges = cv2.Canny(blur, self.args.canny_low, self.args.canny_high)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

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
            if area < self.args.min_square_area_px:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue

            approx = cv2.approxPolyDP(contour, self.args.approx_ratio * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = self.order_quad_points(approx.reshape(4, 2))
            side_lengths = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
            min_side = min(side_lengths)
            max_side = max(side_lengths)
            if min_side <= 1e-6:
                continue

            aspect = max_side / min_side
            if aspect < self.args.square_aspect_min or aspect > self.args.square_aspect_max:
                continue

            x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
            rect_area = float(w * h)
            extent = area / rect_area if rect_area > 1e-6 else 0.0
            if extent < self.args.square_extent_min:
                continue

            angles = self.quad_angles_deg(quad)
            if len(angles) != 4:
                continue
            if min(angles) < self.args.square_angle_min_deg or max(angles) > self.args.square_angle_max_deg:
                continue

            center = np.mean(quad, axis=0)
            center_dist = float(np.linalg.norm(center - image_center))
            score = area - self.args.square_center_weight * center_dist * center_dist

            if score > best_score:
                best_score = score
                best = {
                    'quad': quad,
                    'center': center,
                    'area': area,
                    'aspect': aspect,
                    'extent': extent,
                    'angles': angles,
                    'score': score
                }

        return best

    def board_homography(self, quad, board_size_m):
        half = board_size_m * 0.5
        board_pts = np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            dtype=np.float32
        )
        h_img_to_board = cv2.getPerspectiveTransform(quad.astype(np.float32), board_pts)
        return h_img_to_board

    def perspective_point(self, h, point):
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, h)
        return dst.reshape(2)

    def compute_offsets_from_square(self, square):
        h_img_to_board = self.board_homography(square['quad'], self.args.board_size_m)
        release_px = np.array([self.args.release_u, self.args.release_v], dtype=np.float32)
        release_board = self.perspective_point(h_img_to_board, release_px)

        delta_right_m = -float(release_board[0])
        delta_down_m = -float(release_board[1])
        offset_x_m = self.args.body_forward_from_image_down * delta_down_m
        offset_y_m = self.args.body_left_from_image_right * delta_right_m

        target_center_px = np.mean(square['quad'], axis=0)
        offset_px = [
            float(target_center_px[0] - self.args.release_u),
            float(target_center_px[1] - self.args.release_v)
        ]
        return offset_x_m, offset_y_m, target_center_px, offset_px

    def warp_square_board(self, frame, square):
        quad = square['quad']
        size = int(self.args.warp_size)
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32
        )
        h = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(frame, h, (size, size))

    def center_crop(self, image):
        ratio = clamp(float(self.args.crop_ratio), 0.10, 1.0)
        h, w = image.shape[:2]
        crop_w = int(w * ratio)
        crop_h = int(h * ratio)
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        return image[y1:y1 + crop_h, x1:x1 + crop_w]

    def classify_image_target(self, board_warp):
        crop = self.center_crop(board_warp)
        try:
            class_name, conf, cls_id, probs = self.model.infer(crop)
        except Exception as e:
            return '', 0.0, 'resnet_trt_failed:%s' % str(e), -1
        return str(class_name).strip().lower(), float(conf), 'resnet_trt', int(cls_id)

    def maybe_publish_ros(self, data):
        if self.ros_pub is None:
            return
        data['stamp'] = time.time()
        self.ros_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def draw_debug(self, frame, square, label, conf, detected):
        debug = frame.copy()
        quad = square['quad'].astype(np.int32)
        center = square['center']
        cv2.polylines(debug, [quad], True, (0, 255, 0), 2)
        cv2.circle(debug, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        cv2.circle(debug, (int(self.args.release_u), int(self.args.release_v)), 5, (255, 0, 0), -1)
        text = '%s %.2f stable? %s' % (label, conf, str(detected))
        cv2.putText(debug, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow('image_target_resnet_test', debug)
        cv2.waitKey(1)

    def run(self):
        try:
            while True:
                if rospy is not None and self.args.ros and rospy.is_shutdown():
                    break

                ret, frame = self.cap.read()
                if not ret:
                    now = time.time()
                    if now - self.last_not_found_print > self.args.not_found_interval:
                        print('[IMAGE_RESNET] no camera frame', flush=True)
                        self.last_not_found_print = now
                    time.sleep(0.02)
                    continue

                if self.args.release_u < 0:
                    self.args.release_u = frame.shape[1] * 0.5
                if self.args.release_v < 0:
                    self.args.release_v = frame.shape[0] * 0.5

                square = self.find_square_board(frame)
                if square is None:
                    self.tracker.reset()
                    now = time.time()
                    if now - self.last_not_found_print > self.args.not_found_interval:
                        print('[IMAGE_RESNET] square not found', flush=True)
                        self.last_not_found_print = now
                    self.maybe_publish_ros({
                        'target': 'image_target',
                        'detected': False,
                        'reason': 'square_not_found',
                        'confidence': 0.0,
                        'stable_count': 0
                    })
                    if self.show_debug:
                        cv2.imshow('image_target_resnet_test', frame)
                        cv2.waitKey(1)
                    continue

                offset_x_m, offset_y_m, center_px, offset_px = self.compute_offsets_from_square(square)
                board_warp = self.warp_square_board(frame, square)
                class_name, resnet_conf, method, cls_id = self.classify_image_target(board_warp)

                if class_name == '':
                    self.tracker.reset()
                    now = time.time()
                    if now - self.last_no_class_print > self.args.not_found_interval:
                        print('[IMAGE_RESNET] square found but class not found: reason=%s conf=%.2f' % (method, resnet_conf), flush=True)
                        self.last_no_class_print = now
                    self.maybe_publish_ros({
                        'target': 'image_target',
                        'detected': False,
                        'reason': method,
                        'confidence': float(resnet_conf),
                        'stable_count': 0,
                        'target_center_px': [float(center_px[0]), float(center_px[1])],
                        'release_center_px': [float(self.args.release_u), float(self.args.release_v)],
                        'offset_px': offset_px,
                        'board_area_px': float(square['area']),
                        'board_aspect': float(square['aspect']),
                        'board_extent': float(square['extent']),
                        'mapping_method': 'square_homography_%dmm' % int(self.args.board_size_m * 1000.0),
                        'classifier_method': method
                    })
                    if self.show_debug:
                        self.draw_debug(frame, square, 'no_class', resnet_conf, False)
                    continue

                stable_count = self.tracker.update(class_name, center_px)
                confidence = clamp(0.15 + 0.85 * resnet_conf, 0.0, 1.0)
                detected = stable_count >= self.args.min_stable_count and confidence >= self.args.min_publish_confidence

                data = {
                    'target': 'image_target',
                    'detected': bool(detected),
                    'reason': 'ok' if detected else 'unstable_or_low_confidence',
                    'class_name': class_name,
                    'class_id': int(cls_id),
                    'offset_x_m': float(offset_x_m),
                    'offset_y_m': float(offset_y_m),
                    'confidence': float(confidence),
                    'resnet_confidence': float(resnet_conf),
                    'stable_count': int(stable_count),
                    'target_center_px': [float(center_px[0]), float(center_px[1])],
                    'release_center_px': [float(self.args.release_u), float(self.args.release_v)],
                    'offset_px': offset_px,
                    'board_area_px': float(square['area']),
                    'board_aspect': float(square['aspect']),
                    'board_extent': float(square['extent']),
                    'board_angles_deg': [float(a) for a in square['angles']],
                    'mapping_method': 'square_homography_%dmm' % int(self.args.board_size_m * 1000.0),
                    'classifier_method': method
                }
                self.maybe_publish_ros(data)

                print(
                    '[IMAGE_RESNET] detected=%s class=%s cls_id=%d conf=%.2f resnet=%.2f '
                    'offset=(%.3f, %.3f)m center=(%.1f, %.1f) area=%.0f aspect=%.2f extent=%.2f stable=%d method=%s' % (
                        str(bool(detected)),
                        class_name,
                        int(cls_id),
                        confidence,
                        resnet_conf,
                        offset_x_m,
                        offset_y_m,
                        center_px[0],
                        center_px[1],
                        square['area'],
                        square['aspect'],
                        square['extent'],
                        stable_count,
                        method
                    ),
                    flush=True
                )

                if self.show_debug:
                    self.draw_debug(frame, square, class_name, confidence, detected)

        except KeyboardInterrupt:
            print('\n[IMAGE_RESNET] stopped by user', flush=True)
        finally:
            self.cap.release()
            if self.show_debug:
                cv2.destroyAllWindows()


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Image target square + ResNet TensorRT classification test node.')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument('--gui', dest='gui', action='store_true', help='打开 OpenCV 图形窗口')
    gui_group.add_argument('--no-gui', dest='gui', action='store_false', help='不打开图形窗口，适合纯终端')
    parser.set_defaults(gui=None)

    parser.add_argument('--ros', action='store_true', help='可选：发布 /uav/vision_result，方便 rostopic echo')
    parser.add_argument('--vision-result-topic', default='/uav/vision_result')

    parser.add_argument('--engine', default='models/resnet18_4class_fp16.engine', help='ResNet TensorRT engine 路径；相对路径按脚本目录解析')
    parser.add_argument('--classes', default='models/classes.json', help='类别 json 路径；相对路径按脚本目录解析')
    parser.add_argument('--input-size', type=int, default=224, help='ResNet 输入尺寸，默认 224')
    parser.add_argument('--crop-ratio', type=float, default=0.50)
    parser.add_argument('--warp-size', type=int, default=416)

    parser.add_argument('--board-size-m', type=float, default=0.80)
    parser.add_argument('--release-u', type=float, default=-1.0)
    parser.add_argument('--release-v', type=float, default=-1.0)
    parser.add_argument('--body-forward-from-image-down', type=float, default=-1.0)
    parser.add_argument('--body-left-from-image-right', type=float, default=-1.0)

    parser.add_argument('--min-square-area-px', type=float, default=5000.0)
    parser.add_argument('--square-aspect-min', type=float, default=0.75)
    parser.add_argument('--square-aspect-max', type=float, default=1.30)
    parser.add_argument('--square-center-weight', type=float, default=0.15)
    parser.add_argument('--square-extent-min', type=float, default=0.55)
    parser.add_argument('--square-angle-min-deg', type=float, default=55.0)
    parser.add_argument('--square-angle-max-deg', type=float, default=125.0)
    parser.add_argument('--approx-ratio', type=float, default=0.035)
    parser.add_argument('--canny-low', type=float, default=60.0)
    parser.add_argument('--canny-high', type=float, default=160.0)

    parser.add_argument('--stable-jump-px', type=float, default=25.0)
    parser.add_argument('--min-stable-count', type=int, default=3)
    parser.add_argument('--min-publish-confidence', type=float, default=0.55)
    parser.add_argument('--not-found-interval', type=float, default=1.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    tester = ImageTargetResnetTester(args)
    tester.run()


if __name__ == '__main__':
    main()
