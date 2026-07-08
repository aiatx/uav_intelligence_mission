#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易 UDP 地面站脚本 + 图片靶视觉结果显示

上行控制协议：
    start  -> CMD:START
    status -> CMD:STATUS
    L1     -> CMD:L1
    R0     -> CMD:R1 -> CMD:R2 -> CMD:R3

下行回包协议：
    ACK:<CMD>:...        普通命令确认
    ERR:<CMD>:...        普通命令错误
    STATUS:...           状态查询
    ACK:PING:...         通信检查

下行视觉协议：
    VISION:IMAGE:{json}

VISION:IMAGE 的 JSON 字段示例：
    {
      "target": "image_target",
      "detected": true,
      "class_name": "beer",
      "confidence": 0.92,
      "stable_count": 3,
      "reason": "ok",
      "offset_x_m": 0.01,
      "offset_y_m": -0.02,
      "classifier_method": "resnet_trt"
    }

说明：
    本脚本启动后会有后台接收线程。
    无人机端只要推送 VISION:IMAGE，终端会自动打印图片靶识别结果，
    不需要一直手动 status。
"""

import json
import queue
import socket
import threading
import time


UAV_IP = "192.168.151.102"
UAV_PORT = 8888
BOOT_PORT = UAV_PORT

RECV_TIMEOUT = 5.0
SOCKET_RECV_TIMEOUT = 0.2

# L0 / R0 拆分发送时，每个舵机指令之间的间隔
DROP_STEP_DELAY = 0.35


class ReceiverThread:
    def __init__(self, sock):
        self.sock = sock
        self.reply_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                data, from_addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                print("\n[UDP] 接收线程异常：{}".format(e), flush=True)
                continue

            text = data.decode("utf-8", errors="ignore").strip()
            if not text:
                continue

            if is_vision_message(text):
                print_vision_message(text, from_addr)
                continue

            self.reply_queue.put((text, from_addr, time.time()))


def is_vision_message(text):
    return text.startswith("VISION:IMAGE:")


def format_float(value, ndigits=2, default="NA"):
    try:
        return ("%%.%df" % ndigits) % float(value)
    except Exception:
        return default


def print_vision_message(text, from_addr=None):
    """打印无人机端推送的图片靶识别结果。"""
    try:
        payload_text = text.split(":", 2)[2]
        data = json.loads(payload_text)
    except Exception:
        print("\n[VISION][图片靶] 收到但解析失败：{}".format(text), flush=True)
        return

    detected = bool(data.get("detected", False))
    class_name = str(data.get("class_name", "") or "none")
    confidence = format_float(data.get("confidence", 0.0), 2)
    stable_count = data.get("stable_count", 0)
    reason = str(data.get("reason", ""))
    method = str(data.get("classifier_method", ""))
    offset_x = format_float(data.get("offset_x_m", 0.0), 3)
    offset_y = format_float(data.get("offset_y_m", 0.0), 3)
    area = format_float(data.get("board_area_px", 0.0), 0)

    if detected:
        status = "识别到"
    else:
        status = "未确认"

    src = ""
    if from_addr is not None:
        src = " from {}:{}".format(from_addr[0], from_addr[1])

    print(
        "\n[VISION][图片靶] {} | class={} | conf={} | stable={} | reason={} | offset=({}, {})m | area={} | method={}{}".format(
            status,
            class_name,
            confidence,
            stable_count,
            reason,
            offset_x,
            offset_y,
            area,
            method,
            src,
        ),
        flush=True,
    )


def reply_matches(msg, reply):
    """
    判断当前回包是否属于本次发送的命令。
    视觉消息由后台线程直接打印，不会进入这里。
    """
    msg = msg.strip().upper()
    reply = reply.strip().upper()

    if not msg.startswith("CMD:"):
        return True

    cmd = msg[4:].strip()

    if cmd == "PING":
        return reply.startswith("ACK:PING")

    if cmd == "STATUS":
        return reply.startswith("STATUS:")

    if cmd == "BOOT":
        return reply.startswith("ACK:BOOT") or reply.startswith("ERR:BOOT")

    return (
        reply.startswith("ACK:%s" % cmd) or
        reply.startswith("ERR:%s" % cmd) or
        reply.startswith("ERR:UNKNOWN_CMD:%s" % msg)
    )


def drain_reply_queue(rx):
    """清空已经迟到的普通命令回包；视觉消息不会在这个队列里。"""
    while True:
        try:
            rx.reply_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break


def wait_reply(rx, msg):
    deadline = time.time() + RECV_TIMEOUT

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise socket.timeout()

        try:
            reply, from_addr, _ = rx.reply_queue.get(timeout=remaining)
        except queue.Empty:
            raise socket.timeout()

        if reply_matches(msg, reply):
            return reply, from_addr

        print("收到迟到/错位普通回传，已丢弃：", reply)


def print_ping_result(reply):
    if reply.startswith("ACK:PING:NORMAL"):
        print("PING 结果：UDP 接收节点在线，FSM 状态机心跳正常。")
        return

    if reply.startswith("ACK:PING:RECEIVER_OK"):
        if "fsm=NO_HEARTBEAT" in reply:
            print("PING 结果：UDP 接收节点在线，但还没收到 FSM 状态机心跳。")
            return

        if "fsm=TIMEOUT" in reply:
            print("PING 结果：UDP 接收节点在线，但 FSM 状态机心跳超时。")
            return

        print("PING 结果：UDP 接收节点在线，FSM 状态未知。")
        return

    print("PING 结果：收到未知 PING 回传，请检查接收节点版本。")


def send_cmd(sock, rx, cmd, port=None):
    cmd = cmd.strip().upper()

    if not cmd:
        return

    drain_reply_queue(rx)

    if not cmd.startswith("CMD:"):
        msg = "CMD:" + cmd
    else:
        msg = cmd

    if port is None:
        port = UAV_PORT

    addr = (UAV_IP, port)

    print("\n发送：", msg)

    try:
        sock.sendto(msg.encode("utf-8"), addr)
    except Exception as e:
        print("发送失败：", e)
        return

    try:
        reply, from_addr = wait_reply(rx, msg)
        print("收到回传：", reply)
        print("来自：", from_addr[0], from_addr[1])

        if msg == "CMD:PING":
            print_ping_result(reply)

    except socket.timeout:
        print("等待回传超时。可能是无人机没收到、IP/端口不对，或无人机端节点没启动。")
    except Exception as e:
        print("接收回传失败：", e)


def send_drop_sequence(sock, rx, prefix):
    prefix = prefix.upper()

    if prefix not in ["L", "R"]:
        print("投放序列错误：prefix 必须是 L 或 R")
        return

    print("\n{}0：开始依次发送 {}1 / {}2 / {}3".format(prefix, prefix, prefix, prefix))

    for i in range(1, 4):
        cmd = "{}{}".format(prefix, i)
        send_cmd(sock, rx, cmd)
        time.sleep(DROP_STEP_DELAY)

    print("{}0：序列发送完成。".format(prefix))


def print_help():
    print("\n========== 简易 UDP 地面站 ==========")
    print("无人机地址：{}:{}".format(UAV_IP, UAV_PORT))
    print("回包超时：{} 秒".format(RECV_TIMEOUT))
    print("BOOT 地址：{}:{}".format(UAV_IP, BOOT_PORT))
    print("")
    print("基础命令：")
    print("  boot      发送 CMD:BOOT，一键启动实飞 launch")
    print("  ping      测试 UDP 通信，并注册地面站地址用于接收视觉结果")
    print("  status    查询无人机 FSM / MAVROS 状态")
    print("  start     发布 /uav/start=True，开始任务")
    print("  land      发布 /uav/land=True，普通 AUTO.LAND")
    print("  stop      发布 /uav/stop=True，急停 AUTO.LAND")
    print("  reset     发布 /uav/reset=True，复位 FSM")
    print("  disarm    发布 /uav/disarm=True")
    print("  loop      每 0.5 秒查询一次 STATUS")
    print("")
    print("视觉结果：")
    print("  无需输入命令。无人机端推送 VISION:IMAGE:{json} 后，这里会自动显示图片靶识别结果。")
    print("  第一次建议先输入 ping，让无人机端记录本地面站地址。")
    print("")
    print("投放机构命令：")
    print("  L1/L2/L3  1/2/3号舵机锁定")
    print("  R1/R2/R3  1/2/3号舵机释放")
    print("  L0        依次发送 CMD:L1 -> CMD:L2 -> CMD:L3")
    print("  R0        依次发送 CMD:R1 -> CMD:R2 -> CMD:R3")
    print("")
    print("  q         退出")
    print("====================================\n")


def status_loop(sock, rx):
    print("\n进入状态循环，每 0.5 秒查询一次 STATUS。按 Ctrl+C 退出状态循环。")

    try:
        while True:
            send_cmd(sock, rx, "STATUS")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n已退出状态循环。")


def is_single_drop_cmd(cmd):
    if len(cmd) != 2:
        return False
    if cmd[0] not in ["l", "r"]:
        return False
    if cmd[1] not in ["1", "2", "3"]:
        return False
    return True


def main():
    print_help()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(SOCKET_RECV_TIMEOUT)

    rx = ReceiverThread(sock)
    rx.start()

    try:
        while True:
            try:
                user_input = input("GROUND> ").strip()
            except KeyboardInterrupt:
                print("\n退出。")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ["q", "quit", "exit"]:
                print("退出。")
                break

            if cmd in ["h", "help", "?"]:
                print_help()
                continue

            if cmd == "loop":
                status_loop(sock, rx)
                continue

            if cmd == "boot":
                send_cmd(sock, rx, "BOOT", port=BOOT_PORT)
                continue

            if cmd in ["ping", "status", "start", "land", "stop", "reset", "disarm"]:
                send_cmd(sock, rx, cmd)
                continue

            if is_single_drop_cmd(cmd):
                send_cmd(sock, rx, cmd)
                continue

            if cmd == "l0":
                send_drop_sequence(sock, rx, "L")
                continue

            if cmd == "r0":
                send_drop_sequence(sock, rx, "R")
                continue

            print("未知命令：", user_input)
            print("输入 help 查看命令。")

    finally:
        rx.stop()
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
