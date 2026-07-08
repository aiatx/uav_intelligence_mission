#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
圆环检测独立测试脚本。

从当前 stage1_vision_node.py 的 ring_gate 逻辑拆出：
- 打开前置 RealSense D435i；
- 不等待 /uav/scan_enable，启动后直接检测；
- 可选 OpenCV 图形窗口；
- 没检测到圆环时低频打印；检测到候选圆环时逐帧打印详细信息；
- 可选 --ros 发布 /uav/vision_result，格式与正式视觉节点保持一致。

圆环尺寸重要说明：
当前默认圆环【外径】为 1.20 m，即 --outer-diameter 1.20。
注意这里是外径 diameter，不是半径 radius。
如果之后换圆环：量外边缘到外边缘的实际外径 D，运行时传 --outer-diameter D。
如果只知道半径 R，要传 2R。
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

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

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


class RingGateTester:
    def __init__(self, args, show_gui):
        if rs is None:
            raise RuntimeError('pyrealsense2 import failed. 请确认 RealSense SDK / pyrealsense2 已安装。')
        self.args = args
        self.show_gui = bool(show_gui)
        self.not_found_printer = LowRatePrinter(args.not_found_log_interval)
        self.tracker = StableCounter(args.stable_jump_px)
        self.pipeline = None
        self.profile = None
        self.color_intrinsics = None
        self.ros_pub = None

        if args.ros:
            if rospy is None:
                print('[WARN] rospy 不可用，忽略 --ros。', flush=True)
            else:
                rospy.init_node('ring_gate_test_node', anonymous=True)
                self.ros_pub = rospy.Publisher('/uav/vision_result', String, queue_size=10)
                print('[ROS] publish: /uav/vision_result', flush=True)

    def open_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)
        config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)
        self.profile = self.pipeline.start(config)
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_profile.get_intrinsics()
        self.configure_camera()
        print('[RING] RealSense opened: %dx%d@%d' % (self.args.width, self.args.height, self.args.fps), flush=True)
        print('[RING] outer_diameter=%.3f m, GUI=%s' % (self.args.outer_diameter, str(self.show_gui)), flush=True)

    def configure_camera(self):
        try:
            color_sensor = self.profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if color_sensor.supports(rs.option.sharpness):
                color_sensor.set_option(rs.option.sharpness, 70)
            if color_sensor.supports(rs.option.contrast):
                color_sensor.set_option(rs.option.contrast, 55)
        except Exception as e:
            print('[WARN] front camera option setup failed, ignored: %s' % str(e), flush=True)

    def read_frames(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None, None
        color = np.asanyarray(color_frame.get_data())
        return color, depth_frame

    def find_ring_circle(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        image_center_x = frame.shape[1] * 0.5
        image_center_y = frame.shape[0] * 0.5

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.args.min_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            if circularity < self.args.min_circularity:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < self.args.min_radius or radius > self.args.max_radius:
                continue
            circle_area = math.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 1e-6 else 0.0
            if fill_ratio < self.args.fill_ratio_min or fill_ratio > self.args.fill_ratio_max:
                continue
            center_dist = math.sqrt((cx - image_center_x) ** 2 + (cy - image_center_y) ** 2)
            score = area * circularity - 0.05 * center_dist * center_dist
            if score > best_score:
                best_score = score
                best = {
                    'center': np.array([cx, cy], dtype=np.float32),
                    'radius': float(radius),
                    'area': area,
                    'circularity': circularity,
                    'fill_ratio': float(fill_ratio),
                    'score': float(score),
                    'method': 'contour_circle'
                }
        if best is not None:
            return best

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=80,
            param2=self.args.hough_param2,
            minRadius=int(self.args.min_radius),
            maxRadius=int(self.args.max_radius),
        )
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(np.float32)

        def hough_score(c):
            cx, cy, radius = float(c[0]), float(c[1]), float(c[2])
            center_dist = math.sqrt((cx - image_center_x) ** 2 + (cy - image_center_y) ** 2)
            return radius - self.args.hough_center_weight * center_dist * center_dist

        best_circle = max(circles, key=hough_score)
        return {
            'center': np.array([best_circle[0], best_circle[1]], dtype=np.float32),
            'radius': float(best_circle[2]),
            'area': float(math.pi * best_circle[2] * best_circle[2]),
            'circularity': 0.55,
            'fill_ratio': 1.0,
            'score': float(hough_score(best_circle)),
            'method': 'hough_circle'
        }

    def depth_at_pixel(self, depth_frame, u, v):
        if depth_frame is None or not self.args.use_depth:
            return 0.0
        values = []
        u = int(round(u))
        v = int(round(v))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x = u + dx
                y = v + dy
                if x < 0 or y < 0 or x >= self.args.width or y >= self.args.height:
                    continue
                d = depth_frame.get_distance(x, y)
                if d > 0.10:
                    values.append(d)
        if not values:
            return 0.0
        return float(np.median(values))

    def ring_distance_from_size(self, radius_px):
        fx = float(self.color_intrinsics.fx)
        diameter_px = max(1.0, 2.0 * float(radius_px))
        return self.args.outer_diameter * fx / diameter_px

    def make_result(self, ring, depth_frame):
        center = ring['center']
        u = float(center[0])
        v = float(center[1])
        depth_m = self.depth_at_pixel(depth_frame, u, v)
        size_m = self.ring_distance_from_size(ring['radius'])
        if depth_m > 0.10:
            forward_m = depth_m
            distance_method = 'depth'
        else:
            forward_m = size_m
            distance_method = 'known_outer_diameter'

        intr = self.color_intrinsics
        camera_right_m = (u - float(intr.ppx)) * forward_m / float(intr.fx)
        camera_down_m = (v - float(intr.ppy)) * forward_m / float(intr.fy)
        offset_y_m = self.args.left_from_camera_right * camera_right_m
        offset_z_m = self.args.up_from_camera_down * camera_down_m

        stable_count = self.tracker.update(center)
        radius_score = clamp((ring['radius'] - self.args.min_radius) / 120.0, 0.0, 1.0)
        confidence = clamp(
            self.args.confidence_floor +
            0.35 * clamp(ring['circularity'], 0.0, 1.0) +
            0.25 * radius_score,
            0.0,
            1.0
        )
        detected = stable_count >= self.args.min_stable_count and confidence >= self.args.min_publish_confidence
        return {
            'target': 'ring_gate',
            'detected': bool(detected),
            'reason': 'ok' if detected else 'unstable_or_low_confidence',
            'forward_m': float(forward_m),
            'offset_y_m': float(offset_y_m),
            'offset_z_m': float(offset_z_m),
            'confidence': float(confidence),
            'stable_count': int(stable_count),
            'ring_center_px': [float(u), float(v)],
            'image_center_px': [float(intr.ppx), float(intr.ppy)],
            'ring_radius_px': float(ring['radius']),
            'ring_area_px': float(ring['area']),
            'circularity': float(ring['circularity']),
            'fill_ratio': float(ring.get('fill_ratio', 0.0)),
            'mapping_method': distance_method,
            'detector_method': ring['method']
        }

    def publish_ros(self, data):
        if self.ros_pub is not None:
            self.ros_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def draw(self, frame, ring, result):
        if not self.show_gui:
            return
        debug = frame.copy()
        u, v = result['ring_center_px']
        cv2.circle(debug, (int(u), int(v)), int(result['ring_radius_px']), (0, 255, 0), 2)
        cv2.circle(debug, (int(u), int(v)), 5, (0, 0, 255), -1)
        label = 'det=%s f=%.2f y=%.2f z=%.2f conf=%.2f st=%d' % (
            result['detected'], result['forward_m'], result['offset_y_m'], result['offset_z_m'],
            result['confidence'], result['stable_count'])
        cv2.putText(debug, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow('ring_gate_test', debug)
        cv2.waitKey(1)

    def run(self):
        self.open_camera()
        try:
            while True:
                if self.args.ros and rospy is not None and rospy.is_shutdown():
                    break
                try:
                    color, depth_frame = self.read_frames()
                except Exception as e:
                    self.not_found_printer.print('[RING] no frame: %s' % str(e))
                    continue
                if color is None:
                    self.tracker.reset()
                    self.not_found_printer.print('[RING] no frame')
                    continue
                ring = self.find_ring_circle(color)
                if ring is None:
                    self.tracker.reset()
                    self.not_found_printer.print('[RING] not found')
                    if self.show_gui:
                        cv2.imshow('ring_gate_test', color)
                        cv2.waitKey(1)
                    continue
                result = self.make_result(ring, depth_frame)
                print('[RING] det=%s f=%.2f y=%.3f z=%.3f r=%.1f conf=%.2f stable=%d circ=%.2f fill=%.2f method=%s' % (
                    result['detected'], result['forward_m'], result['offset_y_m'], result['offset_z_m'],
                    result['ring_radius_px'], result['confidence'], result['stable_count'], result['circularity'],
                    result['fill_ratio'], result['detector_method']), flush=True)
                self.publish_ros(result)
                self.draw(color, ring, result)
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()
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
    parser = argparse.ArgumentParser(description='Ring gate standalone tester')
    parser.add_argument('--gui', action='store_true', help='强制打开 OpenCV 图形窗口')
    parser.add_argument('--no-gui', action='store_true', help='强制关闭 OpenCV 图形窗口')
    parser.add_argument('--ros', action='store_true', help='发布 /uav/vision_result')
    parser.add_argument('--width', type=int, default=848)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--outer-diameter', type=float, default=1.20, help='圆环外径，单位 m，默认 1.20')
    parser.add_argument('--use-depth', action='store_true', help='使用圆心附近深度；默认关闭')
    parser.add_argument('--min-radius', type=float, default=35.0)
    parser.add_argument('--max-radius', type=float, default=360.0)
    parser.add_argument('--min-area', type=float, default=800.0)
    parser.add_argument('--min-circularity', type=float, default=0.75)
    parser.add_argument('--confidence-floor', type=float, default=0.35)
    parser.add_argument('--min-stable-count', type=int, default=3)
    parser.add_argument('--min-publish-confidence', type=float, default=0.55)
    parser.add_argument('--fill-ratio-min', type=float, default=0.45)
    parser.add_argument('--fill-ratio-max', type=float, default=1.10)
    parser.add_argument('--hough-param2', type=float, default=50.0)
    parser.add_argument('--hough-center-weight', type=float, default=0.002)
    parser.add_argument('--stable-jump-px', type=float, default=22.0)
    parser.add_argument('--left-from-camera-right', type=float, default=-1.0)
    parser.add_argument('--up-from-camera-down', type=float, default=-1.0)
    parser.add_argument('--not-found-log-interval', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    show_gui = ask_gui(args)
    RingGateTester(args, show_gui).run()
