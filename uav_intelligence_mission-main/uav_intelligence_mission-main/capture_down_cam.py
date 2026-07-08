#!/usr/bin/env python3
import cv2
import os
import time
import argparse
import threading
from datetime import datetime


def keyboard_thread(state):
    while not state["quit"]:
        try:
            cmd = input()
        except EOFError:
            break

        cmd = cmd.strip().lower()

        if cmd == "q":
            state["quit"] = True
        elif cmd == "b":
            state["burst"] = True
        else:
            state["capture"] = True


def main():
    parser = argparse.ArgumentParser(description="SSH终端下视相机采图脚本：回车拍照，b连拍，q退出")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 /dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default="down_cam_dataset")
    parser.add_argument("--prefix", type=str, default="downcam")
    parser.add_argument("--gain", type=float, default=56)
    parser.add_argument("--auto-exposure", type=float, default=3)
    parser.add_argument("--burst-num", type=int, default=5, help="输入 b 后连拍张数")
    parser.add_argument("--burst-gap", type=float, default=0.2, help="连拍间隔，单位秒")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)

    # USB相机常用 MJPG，可以提高读取稳定性；如果无效，OpenCV会自动忽略
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 沿用你之前视觉节点里调过的参数
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, args.auto_exposure)
    cap.set(cv2.CAP_PROP_GAIN, args.gain)

    if not cap.isOpened():
        print(f"错误：无法打开 /dev/video{args.camera}")
        return

    print("下视相机已打开。")
    print(f"保存目录：{os.path.abspath(args.out)}")
    print("操作：")
    print("  直接回车：保存一张")
    print("  b + 回车：连拍")
    print("  q + 回车：退出")
    print("")
    print("当前设置：")
    print(f"  camera=/dev/video{args.camera}")
    print(f"  width={args.width}, height={args.height}, fps={args.fps}")
    print(f"  auto_exposure={args.auto_exposure}, gain={args.gain}")
    print("相机预热中...")

    last_frame = None

    # 预热，等自动曝光稳定
    for _ in range(40):
        ret, frame = cap.read()
        if ret:
            last_frame = frame
        time.sleep(0.03)

    print("预热完成，可以开始采图。")

    state = {
        "capture": False,
        "burst": False,
        "quit": False
    }

    t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    t.start()

    count = 0

    def save_frame(frame):
        nonlocal count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{args.prefix}_{timestamp}_{count:05d}.jpg"
        save_path = os.path.join(args.out, filename)

        ok = cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            count += 1
            print(f"[保存成功] {save_path}")
        else:
            print(f"[保存失败] {save_path}")

    try:
        while not state["quit"]:
            ret, frame = cap.read()
            if not ret:
                print("警告：未能读取相机画面")
                time.sleep(0.1)
                continue

            last_frame = frame

            if state["capture"]:
                state["capture"] = False
                save_frame(last_frame)

            if state["burst"]:
                state["burst"] = False
                print(f"开始连拍 {args.burst_num} 张...")
                for _ in range(args.burst_num):
                    ret, frame = cap.read()
                    if ret:
                        save_frame(frame)
                    time.sleep(args.burst_gap)

            time.sleep(0.005)

    finally:
        cap.release()
        print(f"采集结束，本次保存 {count} 张图片。")


if __name__ == "__main__":
    main()