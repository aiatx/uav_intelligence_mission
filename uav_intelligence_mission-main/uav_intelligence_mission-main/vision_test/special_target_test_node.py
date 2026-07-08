#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特殊靶子识别独立测试脚本。

从当前 stage1_vision_node.py 的 special_target 逻辑拆出：
- 打开下视/USB 摄像头；
- 不等待 /uav/scan_enable，启动后直接识别；
- 使用方形靶板几何识别：Canny/自适应阈值 -> 轮廓 -> 四边形 -> 长宽比/饱满度/角度过滤；
- 输出 offset_x_m / offset_y_m、confidence、stable_count；
- 可选 --ros 发布 /uav/vision_result，格式与正式视觉节点保持一致。

说明：当前正式视觉节点里的 special_target 是几何方形识别，不默认使用 YOLO。
本脚本预留 --use-yolo，模型路径使用相对路径 models/special_target_detector.engine；不启用时不加载模型。
"""

# 使用说明：
#   1) 直接运行会询问是否打开图形界面，输入 y 开窗口，输入 n 只在终端打印。
#   2) 无 NoMachine/无 DISPLAY 的纯终端环境建议输入 n，或直接加 --no-gui。
#   3) 识别不到目标时低频打印简短信息；识别到候选/有效目标时逐帧打印详细信息。
#   4) 默认不需要 /uav/scan_enable，也不依赖 FSM；脚本启动后立即开始识别。
#   5) 需要同时向 ROS 话题发布测试结果时，加 --ros。


import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import rospy
    from std_msgs.msg import String
except Exception:
    rospy = None
    String = None


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class LowRatePrinter:
    def __init__(self, interval=1.0):
        self.interval = float(interval)
        self.last = 0.0

    def print(self, msg):
        now = time.time()
        if now - self.last >= self.interval:
            print(msg, flush=True)
            self.last = now


class StableCounter:
    def __init__(self, max_jump_px):
        self.max_jump_px = float(max_jump_px)
        self.last_center = None
        self.count = 0

    def reset(self):
        self.last_center = None
        self.count = 0

    def update(self, center):
        if center is None:
            self.reset()
            return 0
        center = np.array(center, dtype=np.float32)
        if self.last_center is None:
            self.last_center = center
            self.count = 1
            return self.count
        jump = float(np.linalg.norm(center - self.last_center))
        if jump <= self.max_jump_px:
            self.count += 1
        else:
            self.count = 1
        self.last_center = center
        return self.count


class SpecialTargetTester:
    def __init__(self, args, show_gui):
        self.args = args
        self.show_gui = bool(show_gui)
        self.not_found_printer = LowRatePrinter(args.not_found_log_interval)
        self.tracker = StableCounter(args.stable_jump_px)
        self.release_u = args.release_u if args.release_u >= 0 else args.width * 0.5
        self.release_v = args.release_v if args.release_v >= 0 else args.height * 0.5
        self.ros_pub = None
        self.yolo_model = None
        self.yolo_names = {}
        if args.ros:
            if rospy is None:
                print('[WARN] rospy 不可用，忽略 --ros。', flush=True)
            else:
                rospy.init_node('special_target_test_node', anonymous=True)
                self.ros_pub = rospy.Publisher('/uav/vision_result', String, queue_size=10)
                print('[ROS] publish: /uav/vision_result', flush=True)
        self.load_yolo_if_needed()

    def load_yolo_if_needed(self):
        if not self.args.use_yolo:
            return
        if YOLO is None:
            print('[WARN] ultralytics.YOLO 不可用，特殊靶继续使用几何识别。', flush=True)
            return
        script_dir = Path(__file__).resolve().parent
        model_path = Path(self.args.yolo_model)
        if not model_path.is_absolute():
            model_path = script_dir / model_path
        if not model_path.exists():
            print('[WARN] YOLO model not found: %s，特殊靶继续使用几何识别。' % str(model_path), flush=True)
            return
        self.yolo_model = YOLO(str(model_path), task=self.args.yolo_task)
        self.yolo_names = getattr(self.yolo_model, 'names', {}) or {}
        print('[YOLO] loaded relative model: %s' % str(model_path), flush=True)

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

    def quad_angle_degrees(self, quad):
        angles = []
        for i in range(4):
            p_prev = quad[(i - 1) % 4]
            p = quad[i]
            p_next = quad[(i + 1) % 4]
            v1 = p_prev - p
            v2 = p_next - p
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

        edges = cv2.Canny(blur, 60, 160)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

        best = None
        best_score = -1.0
        image_center = np.array([frame.shape[1] * 0.5, frame.shape[0] * 0.5], dtype=np.float32)

        for contour in candidates:
            area = float(cv2.contourArea(contour))
            if area < self.args.min_square_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
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

            angles = self.quad_angle_degrees(quad)
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
        board_pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
        h_img_to_board = cv2.getPerspectiveTransform(quad.astype(np.float32), board_pts)
        return h_img_to_board

    def perspective_point(self, h, point):
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, h)
        return dst.reshape(2)

    def compute_offsets(self, square):
        quad = square['quad']
        h_img_to_board = self.board_homography(quad, self.args.board_size_m)
        release_px = np.array([self.release_u, self.release_v], dtype=np.float32)
        release_board = self.perspective_point(h_img_to_board, release_px)
        delta_right_m = -float(release_board[0])
        delta_down_m = -float(release_board[1])
        offset_x_m = self.args.body_forward_from_image_down * delta_down_m
        offset_y_m = self.args.body_left_from_image_right * delta_right_m
        target_center_px = np.mean(quad, axis=0)
        offset_px = [float(target_center_px[0] - self.release_u), float(target_center_px[1] - self.release_v)]
        return offset_x_m, offset_y_m, target_center_px, offset_px

    def warp_square_board(self, frame, square, output_size=416):
        quad = square['quad']
        dst = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype=np.float32)
        h = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(frame, h, (output_size, output_size))

    def optional_yolo_classify(self, board_warp):
        if self.yolo_model is None:
            return '', 0.0, 'not_used'
        try:
            results = self.yolo_model.predict(board_warp, conf=self.args.yolo_conf, verbose=False)
        except Exception as e:
            return '', 0.0, 'predict_failed:%s' % str(e)
        if not results:
            return '', 0.0, 'empty_result'
        result = results[0]
        probs = getattr(result, 'probs', None)
        if probs is not None and getattr(probs, 'top1', None) is not None:
            cls_id = int(probs.top1)
            conf = float(probs.top1conf)
            name = self.yolo_names.get(cls_id, str(cls_id))
            return str(name).strip().lower(), conf, 'yolo_classify'
        boxes = getattr(result, 'boxes', None)
        if boxes is not None and len(boxes) > 0:
            best_box = max(boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            name = self.yolo_names.get(cls_id, str(cls_id))
            return str(name).strip().lower(), conf, 'yolo_detect'
        return '', 0.0, 'no_class'

    def make_result(self, square, frame):
        offset_x_m, offset_y_m, center_px, offset_px = self.compute_offsets(square)
        stable_count = self.tracker.update(center_px)
        area_score = clamp(square['area'] / (self.args.width * self.args.height * 0.25), 0.0, 1.0)
        aspect_score = clamp(1.0 - abs(1.0 - square['aspect']), 0.0, 1.0)
        extent_score = clamp((square['extent'] - self.args.square_extent_min) / max(1e-6, 1.0 - self.args.square_extent_min), 0.0, 1.0)
        confidence = clamp(0.30 + 0.30 * area_score + 0.25 * aspect_score + 0.15 * extent_score, 0.0, 1.0)
        yolo_name, yolo_conf, yolo_method = '', 0.0, 'not_used'
        if self.yolo_model is not None:
            board_warp = self.warp_square_board(frame, square)
            yolo_name, yolo_conf, yolo_method = self.optional_yolo_classify(board_warp)
            if yolo_name:
                confidence = clamp(0.5 * confidence + 0.5 * yolo_conf, 0.0, 1.0)
        detected = stable_count >= self.args.min_stable_count and confidence >= self.args.min_publish_confidence
        return {
            'target': 'special_target',
            'detected': bool(detected),
            'reason': 'ok' if detected else 'unstable_or_low_confidence',
            'offset_x_m': float(offset_x_m),
            'offset_y_m': float(offset_y_m),
            'confidence': float(confidence),
            'stable_count': int(stable_count),
            'target_center_px': [float(center_px[0]), float(center_px[1])],
            'release_center_px': [float(self.release_u), float(self.release_v)],
            'offset_px': offset_px,
            'board_area_px': float(square['area']),
            'board_aspect': float(square['aspect']),
            'board_extent': float(square['extent']),
            'board_angles_deg': [float(a) for a in square['angles']],
            'mapping_method': 'square_homography_%.0fmm' % (self.args.board_size_m * 1000.0),
            'detector_method': 'opencv_square',
            'yolo_class_name': yolo_name,
            'yolo_confidence': float(yolo_conf),
            'yolo_method': yolo_method,
        }

    def publish_ros(self, data):
        if self.ros_pub is not None:
            self.ros_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def draw(self, frame, square, result):
        if not self.show_gui:
            return
        debug = frame.copy()
        if square is not None:
            quad = square['quad'].astype(np.int32)
            cv2.polylines(debug, [quad], True, (0, 255, 0), 2)
            center = square['center']
            cv2.circle(debug, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        cv2.circle(debug, (int(self.release_u), int(self.release_v)), 5, (255, 0, 0), -1)
        if result:
            label = 'det=%s x=%.2f y=%.2f conf=%.2f st=%d' % (
                result['detected'], result['offset_x_m'], result['offset_y_m'], result['confidence'], result['stable_count'])
            cv2.putText(debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow('special_target_test', debug)
        cv2.waitKey(1)

    def open_camera(self):
        cap = cv2.VideoCapture(self.args.camera_index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.args.auto_exposure)
        cap.set(cv2.CAP_PROP_GAIN, self.args.gain)
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera index %d' % self.args.camera_index)
        print('[SPECIAL] camera opened: index=%d %dx%d@%d GUI=%s' % (
            self.args.camera_index, self.args.width, self.args.height, self.args.fps, str(self.show_gui)), flush=True)
        return cap

    def run(self):
        cap = self.open_camera()
        try:
            while True:
                if self.args.ros and rospy is not None and rospy.is_shutdown():
                    break
                ret, frame = cap.read()
                if not ret:
                    self.not_found_printer.print('[SPECIAL] no frame')
                    continue
                square = self.find_square_board(frame)
                if square is None:
                    self.tracker.reset()
                    self.not_found_printer.print('[SPECIAL] square not found')
                    self.draw(frame, None, None)
                    continue
                result = self.make_result(square, frame)
                print('[SPECIAL] det=%s x=%.3f y=%.3f conf=%.2f stable=%d area=%.0f aspect=%.2f extent=%.2f angles=%s' % (
                    result['detected'], result['offset_x_m'], result['offset_y_m'], result['confidence'], result['stable_count'],
                    result['board_area_px'], result['board_aspect'], result['board_extent'],
                    ','.join(['%.0f' % a for a in result['board_angles_deg']])), flush=True)
                self.publish_ros(result)
                self.draw(frame, square, result)
        finally:
            cap.release()
            if self.show_gui:
                cv2.destroyAllWindows()


def ask_gui(args):
    if args.gui:
        return True
    if args.no_gui:
        return False
    if os.environ.get('DISPLAY', '') == '':
        print('[GUI] 未检测到 DISPLAY，默认不开图形界面。需要强制打开可加 --gui。', flush=True)
        return False
    if not sys.stdin.isatty():
        return False
    ans = input('是否打开图形界面？NoMachine/桌面环境输入 y，纯终端输入 n [y/N]: ').strip().lower()
    return ans in ['y', 'yes', '1', 'true']


def parse_args():
    parser = argparse.ArgumentParser(description='Special target standalone tester')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--ros', action='store_true', help='发布 /uav/vision_result')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--auto-exposure', type=float, default=3.0)
    parser.add_argument('--gain', type=float, default=56.0)
    parser.add_argument('--board-size-m', type=float, default=0.80)
    parser.add_argument('--release-u', type=float, default=-1.0)
    parser.add_argument('--release-v', type=float, default=-1.0)
    parser.add_argument('--body-forward-from-image-down', type=float, default=-1.0)
    parser.add_argument('--body-left-from-image-right', type=float, default=-1.0)
    parser.add_argument('--min-square-area', type=float, default=5000.0)
    parser.add_argument('--square-aspect-min', type=float, default=0.75)
    parser.add_argument('--square-aspect-max', type=float, default=1.30)
    parser.add_argument('--square-center-weight', type=float, default=0.15)
    parser.add_argument('--square-extent-min', type=float, default=0.55)
    parser.add_argument('--square-angle-min-deg', type=float, default=55.0)
    parser.add_argument('--square-angle-max-deg', type=float, default=125.0)
    parser.add_argument('--stable-jump-px', type=float, default=25.0)
    parser.add_argument('--min-stable-count', type=int, default=3)
    parser.add_argument('--min-publish-confidence', type=float, default=0.60)
    parser.add_argument('--not-found-log-interval', type=float, default=1.0)
    parser.add_argument('--use-yolo', action='store_true', help='可选：启用相对路径 YOLO 模型辅助判断')
    parser.add_argument('--yolo-model', default='models/special_target_detector.engine')
    parser.add_argument('--yolo-task', default='classify')
    parser.add_argument('--yolo-conf', type=float, default=0.25)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    show_gui = ask_gui(args)
    SpecialTargetTester(args, show_gui).run()
