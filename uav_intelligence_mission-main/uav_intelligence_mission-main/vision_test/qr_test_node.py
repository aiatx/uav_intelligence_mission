#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维码识别独立测试脚本。

从当前 stage1_vision_node.py 的 QR 逻辑拆出：
- 打开下视/USB 摄像头；
- 不等待 /uav/scan_enable，启动后直接识别；
- OpenCV QRCodeDetector 优先，pyzbar 兜底；
- 识别内容仍默认要求任务格式：class1,class2,left/right；
- 可选 --accept-any 接受普通二维码文本，便于单独测试二维码库；
- 可选 --ros 发布 /uav/qr_text。
"""

# 使用说明：
#   1) 直接运行会询问是否打开图形界面，输入 y 开窗口，输入 n 只在终端打印。
#   2) 无 NoMachine/无 DISPLAY 的纯终端环境建议输入 n，或直接加 --no-gui。
#   3) 识别不到目标时低频打印简短信息；识别到候选/有效目标时逐帧打印详细信息。
#   4) 默认不需要 /uav/scan_enable，也不依赖 FSM；脚本启动后立即开始识别。
#   5) 需要同时向 ROS 话题发布测试结果时，加 --ros。


import argparse
import os
import re
import sys
import time

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None

try:
    import rospy
    from std_msgs.msg import String
except Exception:
    rospy = None
    String = None


class LowRatePrinter:
    def __init__(self, interval=1.0):
        self.interval = float(interval)
        self.last = 0.0

    def print(self, msg):
        now = time.time()
        if now - self.last >= self.interval:
            print(msg, flush=True)
            self.last = now


class QRTester:
    def __init__(self, args, show_gui):
        self.args = args
        self.show_gui = bool(show_gui)
        self.detector = cv2.QRCodeDetector()
        self.not_found_printer = LowRatePrinter(args.not_found_log_interval)
        self.ros_pub = None
        if args.ros:
            if rospy is None:
                print('[WARN] rospy 不可用，忽略 --ros。', flush=True)
            else:
                rospy.init_node('qr_test_node', anonymous=True)
                self.ros_pub = rospy.Publisher('/uav/qr_text', String, queue_size=10)
                print('[ROS] publish: /uav/qr_text', flush=True)

    def normalize_qr_text(self, text):
        text = str(text).strip()
        if self.args.accept_any:
            return text
        text = re.sub(r'\s+', '', text)
        parts = [p.strip().lower() for p in text.split(',')]
        if len(parts) != 3:
            return ''
        if parts[2] not in ['left', 'right']:
            return ''
        normalized = '{},{},{}'.format(parts[0], parts[1], parts[2])
        if not re.match(self.args.valid_regex, normalized):
            return ''
        return normalized

    def preprocess_candidates(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
        sharpen = cv2.addWeighted(gray_clahe, 1.6, blur, -0.6, 0)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        return [
            ('raw', frame),
            ('gray', gray),
            ('clahe', gray_clahe),
            ('sharpen', sharpen),
            ('adaptive', adaptive),
        ]

    def decode_opencv(self, frame):
        for method, img in self.preprocess_candidates(frame):
            try:
                data, points, _ = self.detector.detectAndDecode(img)
            except Exception:
                continue
            data = self.normalize_qr_text(data)
            if data and points is not None:
                pts = np.array(points, dtype=np.float32).reshape(-1, 2)
                return data, pts, 'opencv_' + method
        return '', None, ''

    def decode_pyzbar(self, frame):
        if pyzbar is None:
            return '', None, ''
        for method, img in self.preprocess_candidates(frame):
            decode_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            try:
                results = pyzbar.decode(decode_img)
            except Exception:
                continue
            if not results:
                continue
            best = max(results, key=lambda r: r.rect.width * r.rect.height)
            try:
                text = best.data.decode('utf-8', errors='replace')
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
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            return text, pts, 'pyzbar_' + method
        return '', None, ''

    def decode(self, frame):
        text, pts, method = self.decode_opencv(frame)
        if not text:
            text, pts, method = self.decode_pyzbar(frame)
        return text, pts, method

    def publish_ros(self, text):
        if self.ros_pub is not None:
            self.ros_pub.publish(String(data=text))

    def draw(self, frame, text, pts, method):
        if not self.show_gui:
            return
        debug = frame.copy()
        if pts is not None and len(pts) >= 4:
            cv2.polylines(debug, [pts.astype(np.int32)], True, (0, 255, 0), 2)
        if text:
            cv2.putText(debug, '%s [%s]' % (text, method), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('qr_test', debug)
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
        print('[QR] camera opened: index=%d %dx%d@%d GUI=%s' % (
            self.args.camera_index, self.args.width, self.args.height, self.args.fps, str(self.show_gui)), flush=True)
        if pyzbar is None:
            print('[WARN] pyzbar 不可用，将只使用 OpenCV QRCodeDetector。', flush=True)
        return cap

    def run(self):
        cap = self.open_camera()
        try:
            while True:
                if self.args.ros and rospy is not None and rospy.is_shutdown():
                    break
                ret, frame = cap.read()
                if not ret:
                    self.not_found_printer.print('[QR] no frame')
                    continue
                text, pts, method = self.decode(frame)
                if not text:
                    self.not_found_printer.print('[QR] not found')
                    self.draw(frame, '', None, '')
                    continue
                center = np.mean(pts, axis=0) if pts is not None else np.array([-1.0, -1.0])
                area = float(abs(cv2.contourArea(pts))) if pts is not None and len(pts) >= 4 else 0.0
                print('[QR] text=%s center=(%.1f, %.1f) area=%.0f method=%s' % (
                    text, center[0], center[1], area, method), flush=True)
                self.publish_ros(text)
                self.draw(frame, text, pts, method)
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
    parser = argparse.ArgumentParser(description='QR standalone tester')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--ros', action='store_true', help='发布 /uav/qr_text')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--auto-exposure', type=float, default=3.0)
    parser.add_argument('--gain', type=float, default=56.0)
    parser.add_argument('--accept-any', action='store_true', help='接受任意二维码文本，不强制任务格式')
    parser.add_argument('--valid-regex', default=r'^[a-zA-Z0-9_ -]+,[a-zA-Z0-9_ -]+,(left|right)$')
    parser.add_argument('--not-found-log-interval', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    show_gui = ask_gui(args)
    QRTester(args, show_gui).run()
