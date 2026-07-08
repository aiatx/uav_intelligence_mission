#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UDP UAV Command Receiver + ESP32 Drop Serial Bridge

作用：
1. 在无人机端监听 UDP 指令；
2. 将电脑端发送的简单字符串协议转换成 FSM 的 ROS 控制话题；
3. 集成 ESP32 串口投放功能：
   - 地面站手动发送 CMD:L1/L2/L3/R1/R2/R3 时，直接通过串口发给 ESP32；
   - 地面站若发送 CMD:L0/R0，本节点也可拆成 L1~L3 / R1~R3 依次发给 ESP32；
   - 订阅 /uav/drop_cmd，将 FSM 的 image_drop_1 / special_drop / image_drop_2 映射成 R1 / R2 / R3；
4. 给电脑端返回 ACK / STATUS，方便简易地面站脚本使用。

默认监听：
    192.168.151.102:8888

默认 ESP32 串口：
    /dev/ttyUSB0 @ 115200

运行示例：
    rosrun <your_pkg> udp_uav_cmd_receiver_integrated.py \
        _bind_ip:=192.168.151.102 \
        _bind_port:=8888 \
        _esp32_port:=/dev/ttyUSB0 \
        _esp32_baud:=115200

地面站协议：
    输入 start -> 发送 CMD:START -> 本节点发布 /uav/start=True
    输入 L1    -> 发送 CMD:L1    -> 本节点串口发送 L1\n 给 ESP32
    输入 R0    -> 地面站通常拆成 CMD:R1/CMD:R2/CMD:R3；若没拆，本节点也会拆。

FSM 投放协议：
    FSM 发布 /uav/drop_cmd = image_drop_1 -> 本节点串口发送 R1\n
    FSM 发布 /uav/drop_cmd = special_drop  -> 本节点串口发送 R2\n
    FSM 发布 /uav/drop_cmd = image_drop_2 -> 本节点串口发送 R3\n
"""

import socket
import time
import threading
import os
import glob
import json

import rospy
from std_msgs.msg import Bool, String

try:
    import serial
    HAS_SERIAL = True
except Exception:
    serial = None
    HAS_SERIAL = False

try:
    from mavros_msgs.msg import State
    HAS_MAVROS_MSG = True
except Exception:
    HAS_MAVROS_MSG = False


class UdpUavCmdReceiver:
    def __init__(self):
        rospy.init_node("udp_uav_cmd_receiver", anonymous=False)

        # ---------- UDP 参数 ----------
        self.bind_ip = rospy.get_param("~bind_ip", "192.168.151.102")
        self.bind_port = int(rospy.get_param("~bind_port", 8888))

        # 可选安全口令。默认空，不校验。
        # 若设置 _auth_token:=abc123，则电脑端需要发送 CMD:START:abc123
        self.auth_token = str(rospy.get_param("~auth_token", "")).strip()
        self.fsm_heartbeat_timeout = float(rospy.get_param("~fsm_heartbeat_timeout", 2.0))

        # ---------- 视觉结果转发到地面站 ----------
        # 视觉节点会发布 /uav/vision_result，内容是 JSON 字符串。
        # 本节点只转发 target=image_target 的结果，方便地面站实时确认图片靶识别是否正确。
        self.vision_result_topic = str(rospy.get_param("~vision_result_topic", "/uav/vision_result"))
        self.vision_forward_enabled = bool(rospy.get_param("~vision_forward_enabled", True))
        self.vision_forward_target = str(rospy.get_param("~vision_forward_target", "image_target")).strip().lower()
        self.vision_forward_max_hz = float(rospy.get_param("~vision_forward_max_hz", 4.0))
        # 默认自动使用最近一次给无人机发 UDP 命令的地面站地址。
        # 如果想固定发给某台电脑，也可以传 _ground_station_ip:=... _ground_station_port:=...
        self.ground_station_ip = str(rospy.get_param("~ground_station_ip", "")).strip()
        self.ground_station_port = int(rospy.get_param("~ground_station_port", 0))
        self.last_client_addr = None
        self.last_vision_forward_time = 0.0
        self.last_vision_summary = "NONE"

        # ---------- ESP32 串口投放参数 ----------
        self.drop_serial_enabled = bool(rospy.get_param("~drop_serial_enabled", True))
        self.esp32_port = str(rospy.get_param("~esp32_port", "/dev/ttyUSB0"))
        self.esp32_baud = int(rospy.get_param("~esp32_baud", 115200))
        self.serial_timeout = float(rospy.get_param("~serial_timeout", 0.5))
        self.serial_read_wait = float(rospy.get_param("~serial_read_wait", 1.0))
        self.drop_step_delay = float(rospy.get_param("~drop_step_delay", 0.35))
        self.esp32_boot_wait = float(rospy.get_param("~esp32_boot_wait", 3.0))
        self.serial_reopen_interval = float(rospy.get_param("~serial_reopen_interval", 2.0))
        # 自动串口发现：如果 esp32_port 指向的 /dev/ttyUSB0 不存在，
        # 自动尝试 /dev/serial/by-id/*、/dev/ttyUSB*、/dev/ttyACM*。
        self.serial_auto_detect = bool(rospy.get_param("~serial_auto_detect", True))
        self.serial_candidates = []
        # 串口命令可靠性参数：每条 L/R 命令最多发送 serial_retry_count 次。
        # 只有读到 ESP32 回传 OK 才认为成功；读到 ERR 或超时未回 OK 会重发。
        self.serial_retry_count = int(rospy.get_param("~serial_retry_count", 3))
        self.serial_retry_delay = float(rospy.get_param("~serial_retry_delay", 0.2))

        self.esp32_ser = None
        self.serial_lock = threading.Lock()
        self.last_serial_open_try = 0.0
        self.last_serial_error = "NONE"
        self.last_serial_cmd = "NONE"
        self.last_serial_time = None
        self.last_serial_attempts = 0
        self.last_serial_reply = "NONE"
        self.last_drop_cmd = "NONE"
        self.last_drop_result = "NONE"

        # FSM drop_cmd -> ESP32 串口指令映射。
        # 用户指定三个靶子顺序：1、3、2，即 image_drop_1 -> R1，special_drop -> R2，image_drop_2 -> R3。
        self.drop_cmd_map = {
            "IMAGE_DROP_1": "R1",
            "SPECIAL_DROP": "R2",
            "IMAGE_DROP_2": "R3",
            # 兼容直接发布 ESP32 指令。
            "R1": "R1",
            "R2": "R2",
            "R3": "R3",
            "L1": "L1",
            "L2": "L2",
            "L3": "L3",
        }

        # ---------- FSM 控制话题 ----------
        self.start_pub = rospy.Publisher("/uav/start", Bool, queue_size=10)
        self.land_pub = rospy.Publisher("/uav/land", Bool, queue_size=10)
        self.stop_pub = rospy.Publisher("/uav/stop", Bool, queue_size=10)
        self.reset_pub = rospy.Publisher("/uav/reset", Bool, queue_size=10)
        self.disarm_pub = rospy.Publisher("/uav/disarm", Bool, queue_size=10)

        # ---------- 读取 FSM 状态，便于电脑端 STATUS 查询 ----------
        self.fsm_state = "UNKNOWN"
        self.safety_state = "UNKNOWN"
        self.land_status = "UNKNOWN"
        self.fsm_state_last_time = None

        rospy.Subscriber("/uav/fsm_state", String, self.fsm_state_cb, queue_size=10)
        rospy.Subscriber("/uav/safety_state", String, self.safety_state_cb, queue_size=10)
        rospy.Subscriber("/uav/land_status", String, self.land_status_cb, queue_size=10)

        # FSM 自动投放命令入口。
        rospy.Subscriber("/uav/drop_cmd", String, self.drop_cmd_cb, queue_size=10)

        # 订阅视觉节点结果，并把图片靶识别结果通过 UDP 推送给地面站。
        rospy.Subscriber(self.vision_result_topic, String, self.vision_result_cb, queue_size=20)

        # ---------- 可选读取 MAVROS 状态 ----------
        self.mavros_connected = "UNKNOWN"
        self.mavros_armed = "UNKNOWN"
        self.mavros_mode = "UNKNOWN"

        if HAS_MAVROS_MSG:
            rospy.Subscriber("/mavros/state", State, self.mavros_state_cb, queue_size=10)

        # ---------- UDP socket ----------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.settimeout(0.1)

        # 尝试打开 ESP32 串口。
        if self.drop_serial_enabled:
            self.open_esp32_serial(force=True)
        else:
            self.last_serial_error = "SERIAL_DISABLED"

        rospy.logwarn(
            "UDP UAV command receiver started. Listening on %s:%d",
            self.bind_ip,
            self.bind_port
        )

        rospy.logwarn(
            "Supported commands: CMD:PING, CMD:STATUS, CMD:START, CMD:LAND, CMD:STOP, CMD:RESET, CMD:DISARM, "
            "CMD:L1/L2/L3, CMD:R1/R2/R3, CMD:L0/R0"
        )

        rospy.logwarn(
            "Vision UDP forward: enabled=%s topic=%s target=%s max_hz=%.1f fixed_gs=%s:%d",
            str(self.vision_forward_enabled),
            self.vision_result_topic,
            self.vision_forward_target,
            self.vision_forward_max_hz,
            self.ground_station_ip if self.ground_station_ip else "AUTO_LAST_CLIENT",
            self.ground_station_port
        )

        rospy.logwarn(
            "Drop mapping: image_drop_1->R1, image_drop_2->R3, special_drop->R2. "
            "ESP32 serial=%s @ %d enabled=%s read_wait=%.2fs boot_wait=%.2fs retry=%d delay=%.2fs",
            self.esp32_port,
            self.esp32_baud,
            str(self.drop_serial_enabled),
            self.serial_read_wait,
            self.esp32_boot_wait,
            self.serial_retry_count,
            self.serial_retry_delay
        )

    # =========================
    # ROS 状态回调
    # =========================

    def fsm_state_cb(self, msg):
        self.fsm_state = msg.data
        self.fsm_state_last_time = rospy.Time.now()

    def safety_state_cb(self, msg):
        self.safety_state = msg.data

    def land_status_cb(self, msg):
        self.land_status = msg.data

    def mavros_state_cb(self, msg):
        self.mavros_connected = str(msg.connected)
        self.mavros_armed = str(msg.armed)
        self.mavros_mode = msg.mode

    def drop_cmd_cb(self, msg):
        """
        FSM 自动投放入口。
        FSM 满足图片靶/特殊靶视觉条件后发布 /uav/drop_cmd，
        本节点负责映射并通过串口发送给 ESP32。
        """
        raw_cmd = msg.data.strip()
        if raw_cmd == "":
            rospy.logwarn("/uav/drop_cmd empty, ignored.")
            return

        serial_cmd = self.map_drop_cmd_to_serial(raw_cmd)
        self.last_drop_cmd = raw_cmd

        if serial_cmd is None:
            self.last_drop_result = "UNMAPPED"
            rospy.logerr("/uav/drop_cmd unmapped: %s", raw_cmd)
            return

        ok, reason = self.send_serial_cmd(serial_cmd)
        self.last_drop_result = "%s:%s" % (serial_cmd, reason)

        if ok:
            rospy.logwarn("FSM drop_cmd %s -> ESP32 %s OK", raw_cmd, serial_cmd)
        else:
            rospy.logerr("FSM drop_cmd %s -> ESP32 %s FAILED: %s", raw_cmd, serial_cmd, reason)

    # =========================
    # UDP / ROS 工具
    # =========================

    def send_reply(self, text, addr):
        try:
            self.sock.sendto((text + "\n").encode("utf-8"), addr)
        except Exception as e:
            rospy.logwarn("Failed to send UDP reply to %s: %s", str(addr), str(e))

    def get_ground_station_addr(self):
        """获取视觉结果要推送到的地面站 UDP 地址。"""
        if self.ground_station_ip and self.ground_station_port > 0:
            return (self.ground_station_ip, self.ground_station_port)
        return self.last_client_addr

    def vision_result_cb(self, msg):
        """订阅 /uav/vision_result，并把图片靶识别结果推送给地面站。

        下行协议：
            VISION:IMAGE:{json}

        json 中保留视觉节点发布的关键信息，例如 detected、class_name、confidence、
        reason、stable_count、offset_x_m、offset_y_m、classifier_method 等。
        """
        if not self.vision_forward_enabled:
            return

        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "vision_result JSON parse failed: %s", str(e))
            return

        target = str(data.get("target", "")).strip().lower()
        if target != self.vision_forward_target:
            return

        if self.vision_forward_max_hz > 0:
            now = time.time()
            min_interval = 1.0 / max(0.1, self.vision_forward_max_hz)
            if now - self.last_vision_forward_time < min_interval:
                return
            self.last_vision_forward_time = now

        addr = self.get_ground_station_addr()
        if addr is None:
            rospy.logwarn_throttle(3.0, "No ground station UDP address yet. Send CMD:PING/STATUS/START from ground station first.")
            return

        payload = {
            "target": target,
            "detected": bool(data.get("detected", False)),
            "reason": data.get("reason", ""),
            "class_name": data.get("class_name", ""),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "stable_count": int(data.get("stable_count", 0) or 0),
            "offset_x_m": float(data.get("offset_x_m", 0.0) or 0.0),
            "offset_y_m": float(data.get("offset_y_m", 0.0) or 0.0),
            "target_center_px": data.get("target_center_px", []),
            "offset_px": data.get("offset_px", []),
            "board_area_px": float(data.get("board_area_px", 0.0) or 0.0),
            "board_aspect": float(data.get("board_aspect", 0.0) or 0.0),
            "classifier_method": data.get("classifier_method", data.get("detector_method", "")),
            "stamp": data.get("stamp", rospy.Time.now().to_sec()),
        }

        text = "VISION:IMAGE:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.send_reply(text, addr)

        cls = payload.get("class_name") or "none"
        self.last_vision_summary = "%s detected=%s class=%s conf=%.2f reason=%s" % (
            target,
            str(payload.get("detected")),
            cls,
            float(payload.get("confidence", 0.0)),
            str(payload.get("reason", ""))
        )

    def publish_bool_pulse(self, pub, topic_name):
        """
        连续发布几次 True，提高 UDP 单次命令转 ROS 话题的可靠性。
        不发布 False，因为 FSM 只关心 True 触发。
        """
        msg = Bool(data=True)

        for _ in range(3):
            pub.publish(msg)
            rospy.sleep(0.03)

        rospy.logwarn("Published %s=True", topic_name)

    def parse_command(self, raw_text):
        """
        支持格式：
            CMD:START
            CMD:START:token
            CMD:L1
            CMD:R0

        返回：
            command, error
        """
        text = raw_text.strip()

        if not text:
            return None, "EMPTY"

        parts = text.split(":")

        if len(parts) < 2:
            return None, "BAD_FORMAT"

        if parts[0].upper() != "CMD":
            return None, "BAD_PREFIX"

        cmd = parts[1].strip().upper()

        # 安全口令校验。默认 auth_token 为空，不校验。
        if self.auth_token:
            if len(parts) < 3:
                return None, "AUTH_REQUIRED"

            token = parts[2].strip()
            if token != self.auth_token:
                return None, "AUTH_FAILED"

        # 兼容一些别名，电脑端写起来更自由。
        alias = {
            "TAKEOFF": "START",
            "BEGIN": "START",
            "MISSION": "START",
            "RTL": "LAND",
            "EMERGENCY": "STOP",
            "E_STOP": "STOP",
            "KILL": "STOP",
        }

        cmd = alias.get(cmd, cmd)
        return cmd, None

    # =========================
    # ESP32 串口桥接
    # =========================

    def list_serial_candidates(self):
        """
        枚举可能的 ESP32 串口。
        优先顺序：
        1. 当前配置的 esp32_port，如果存在；
        2. /dev/serial/by-id/ 下的稳定设备名，优先 Espressif/CP210/CH340；
        3. /dev/ttyUSB*；
        4. /dev/ttyACM*。
        """
        candidates = []

        def add(path):
            if not path:
                return
            if path in candidates:
                return
            # /dev/serial/by-id 下是符号链接，exists 会跟随链接判断真实设备是否存在。
            if os.path.exists(path):
                candidates.append(path)

        add(self.esp32_port)

        by_id = sorted(glob.glob("/dev/serial/by-id/*"))
        preferred_keywords = [
            "ESPRESSIF",
            "USB_JTAG",
            "CP210",
            "CH340",
            "CH341",
            "WCH",
            "UART",
            "SERIAL",
        ]

        preferred = []
        normal = []
        for path in by_id:
            upper = os.path.basename(path).upper()
            if any(k in upper for k in preferred_keywords):
                preferred.append(path)
            else:
                normal.append(path)

        for path in preferred + normal:
            add(path)

        for path in sorted(glob.glob("/dev/ttyUSB*")):
            add(path)

        for path in sorted(glob.glob("/dev/ttyACM*")):
            add(path)

        self.serial_candidates = candidates
        return candidates

    def build_serial_missing_reason(self):
        """生成串口不存在时更容易看懂的错误信息。"""
        candidates = self.list_serial_candidates()
        if candidates:
            return "configured=%s missing; candidates=%s" % (
                self.esp32_port,
                ",".join(candidates)
            )
        return "configured=%s missing; no /dev/serial/by-id, /dev/ttyUSB*, /dev/ttyACM* found" % self.esp32_port

    def get_ports_to_try(self):
        """
        返回本次打开串口要尝试的端口列表。
        如果开启自动发现，即使配置的 /dev/ttyUSB0 不存在，也会自动尝试当前实际存在的端口。
        """
        ports = []

        def add_port(path):
            if path and path not in ports:
                ports.append(path)

        if os.path.exists(self.esp32_port):
            add_port(self.esp32_port)

        if self.serial_auto_detect:
            for path in self.list_serial_candidates():
                add_port(path)
        else:
            add_port(self.esp32_port)

        # 如果没有任何候选，也保留原配置，便于错误信息明确指出它不存在。
        if not ports:
            add_port(self.esp32_port)

        return ports

    def open_esp32_serial(self, force=False):
        """尝试打开 ESP32 串口。失败时不让整个 UDP 节点崩溃；开启自动发现时会尝试当前存在的 USB/ACM 端口。"""
        if not self.drop_serial_enabled:
            self.last_serial_error = "SERIAL_DISABLED"
            return False

        if not HAS_SERIAL:
            self.last_serial_error = "PY_SERIAL_NOT_INSTALLED"
            rospy.logerr("pyserial is not installed. Install with: sudo apt install python3-serial")
            return False

        now = time.time()
        if not force and now - self.last_serial_open_try < self.serial_reopen_interval:
            return self.esp32_ser is not None and getattr(self.esp32_ser, "is_open", False)

        self.last_serial_open_try = now

        with self.serial_lock:
            try:
                if self.esp32_ser is not None and self.esp32_ser.is_open:
                    return True
            except Exception:
                pass

            ports_to_try = self.get_ports_to_try()
            errors = []

            for port in ports_to_try:
                try:
                    rospy.logwarn("Opening ESP32 serial: %s @ %d", port, self.esp32_baud)
                    self.esp32_ser = serial.Serial(
                        port=port,
                        baudrate=self.esp32_baud,
                        timeout=self.serial_timeout,
                        write_timeout=self.serial_timeout
                    )

                    # 尽量避免某些 USB 转串口板在打开串口时被 DTR/RTS 反复复位。
                    try:
                        self.esp32_ser.setDTR(False)
                        self.esp32_ser.setRTS(False)
                    except Exception:
                        pass

                    # 串口打开可能导致 ESP32 复位，等待一下再清缓冲。
                    if self.esp32_boot_wait > 0:
                        rospy.sleep(self.esp32_boot_wait)

                    self.esp32_ser.reset_input_buffer()
                    self.esp32_ser.reset_output_buffer()

                    self.esp32_port = port
                    self.last_serial_error = "OK"
                    rospy.logwarn("ESP32 serial opened: %s", self.esp32_port)
                    return True

                except Exception as e:
                    self.esp32_ser = None
                    errors.append("%s:%s" % (port, str(e)))
                    rospy.logwarn("Open ESP32 serial failed on %s: %s", port, str(e))

            if errors:
                self.last_serial_error = " | ".join(errors)
            else:
                self.last_serial_error = self.build_serial_missing_reason()

            rospy.logerr("Open ESP32 serial failed: %s", self.last_serial_error)
            return False

    def ensure_serial_ready(self):
        """确保串口可用；不可用时尝试按间隔重连。"""
        try:
            if self.esp32_ser is not None and self.esp32_ser.is_open:
                return True
        except Exception:
            self.esp32_ser = None

        return self.open_esp32_serial(force=False)

    def read_esp32_lines(self, wait_time=None):
        """尝试读取 ESP32 串口回传。没有回传也不认为失败。"""
        if wait_time is None:
            wait_time = self.serial_read_wait

        lines = []
        if self.esp32_ser is None:
            return lines

        end_time = time.time() + wait_time

        while time.time() < end_time and not rospy.is_shutdown():
            try:
                line = self.esp32_ser.readline()
            except Exception:
                break

            if not line:
                continue

            text = line.decode("utf-8", errors="ignore").strip()
            if text:
                lines.append(text)

        return lines

    def evaluate_esp32_reply(self, cmd, lines):
        """
        判断 ESP32 是否真正执行了命令。
        约定：drop.ino 正常会回传类似 "L1 OK" / "R2 OK"；若回传 ERR 或没有 OK，则认为本次发送失败。
        """
        if not lines:
            return False, "NO_ACK"

        upper_lines = [str(x).strip().upper() for x in lines if str(x).strip()]
        if not upper_lines:
            return False, "NO_ACK"

        joined = " | ".join(upper_lines)

        if any("ERR" in line for line in upper_lines):
            return False, "ESP32_ERR:%s" % joined

        expected = "%s OK" % cmd
        if any(expected in line for line in upper_lines):
            return True, "OK_REPLY:%s" % joined

        # 兼容只回 OK 的固件；只要没有 ERR，且出现 OK，也认为成功。
        if any("OK" in line for line in upper_lines):
            return True, "OK_REPLY_GENERIC:%s" % joined

        return False, "NO_OK_REPLY:%s" % joined

    def send_serial_cmd(self, cmd):
        """给 ESP32 发送一条串口命令，例如 R1、L2；最多重发 serial_retry_count 次。"""
        cmd = cmd.strip().upper()

        if not cmd:
            return False, "EMPTY"

        if not self.is_single_drop_cmd(cmd):
            return False, "BAD_SERIAL_CMD"

        if not self.ensure_serial_ready():
            return False, "SERIAL_NOT_READY:%s" % self.last_serial_error

        data = (cmd + "\n").encode("utf-8")
        max_try = max(1, int(getattr(self, "serial_retry_count", 3)))
        retry_delay = max(0.0, float(getattr(self, "serial_retry_delay", 0.12)))

        all_lines = []
        last_reason = "NO_ATTEMPT"
        actual_attempts = 0

        for attempt in range(1, max_try + 1):
            actual_attempts = attempt

            with self.serial_lock:
                try:
                    # 清掉上一次命令或 ESP32 刚复位时残留的串口输出，避免把旧 ERR 当作本次回复。
                    try:
                        self.esp32_ser.reset_input_buffer()
                    except Exception:
                        pass

                    self.esp32_ser.write(data)
                    self.esp32_ser.flush()
                except Exception as e:
                    self.last_serial_error = str(e)
                    try:
                        if self.esp32_ser is not None:
                            self.esp32_ser.close()
                    except Exception:
                        pass
                    self.esp32_ser = None
                    self.last_serial_cmd = cmd
                    self.last_serial_time = rospy.Time.now()
                    self.last_serial_attempts = attempt
                    self.last_serial_reply = " | ".join(all_lines) if all_lines else "NONE"
                    return False, "SERIAL_WRITE_ERR:%s" % str(e)

                esp_lines = self.read_esp32_lines()

            if esp_lines:
                all_lines.extend(["try%d:%s" % (attempt, line) for line in esp_lines])
                rospy.logwarn("ESP32 reply for %s try %d/%d: %s", cmd, attempt, max_try, " | ".join(esp_lines))
            else:
                rospy.logwarn("ESP32 no reply for %s try %d/%d", cmd, attempt, max_try)

            ok, reason = self.evaluate_esp32_reply(cmd, esp_lines)
            last_reason = reason

            if ok:
                self.last_serial_cmd = cmd
                self.last_serial_time = rospy.Time.now()
                self.last_serial_error = "OK"
                self.last_serial_attempts = attempt
                self.last_serial_reply = " | ".join(all_lines) if all_lines else "NONE"
                return True, "SERIAL_OK:try=%d" % attempt

            if attempt < max_try:
                rospy.logwarn("ESP32 command %s failed on try %d/%d: %s, retry...", cmd, attempt, max_try, reason)
                rospy.sleep(retry_delay)

        self.last_serial_cmd = cmd
        self.last_serial_time = rospy.Time.now()
        self.last_serial_error = last_reason
        self.last_serial_attempts = actual_attempts
        self.last_serial_reply = " | ".join(all_lines) if all_lines else "NONE"
        return False, "SERIAL_NO_OK_AFTER_%d:%s" % (actual_attempts, last_reason)

    def is_single_drop_cmd(self, cmd):
        """单个舵机指令：L1/L2/L3/R1/R2/R3。"""
        if len(cmd) != 2:
            return False
        if cmd[0] not in ["L", "R"]:
            return False
        if cmd[1] not in ["1", "2", "3"]:
            return False
        return True

    def is_all_drop_cmd(self, cmd):
        """全部动作指令：L0/R0。"""
        return cmd in ["L0", "R0"]

    def handle_l0_r0(self, cmd):
        """
        L0/R0 不作为单条串口命令给 ESP32，而是拆成三条：
            L0 -> L1 L2 L3
            R0 -> R1 R2 R3
        """
        prefix = cmd[0]
        sent_list = []

        for i in range(1, 4):
            sub_cmd = "%s%d" % (prefix, i)
            ok, reason = self.send_serial_cmd(sub_cmd)
            sent_list.append("%s:%s" % (sub_cmd, reason))

            if not ok:
                return False, ";".join(sent_list)

            rospy.sleep(self.drop_step_delay)

        return True, ";".join(sent_list)

    def map_drop_cmd_to_serial(self, drop_cmd):
        """把 FSM 的 /uav/drop_cmd 映射成 ESP32 串口指令。"""
        key = drop_cmd.strip().upper()
        return self.drop_cmd_map.get(key, None)

    # =========================
    # STATUS / PING
    # =========================

    def serial_status_text(self):
        if not self.drop_serial_enabled:
            state = "DISABLED"
        elif self.esp32_ser is not None and getattr(self.esp32_ser, "is_open", False):
            state = "OK"
        else:
            state = "NOT_READY"

        return (
            "drop_serial=%s;"
            "esp32_port=%s;"
            "esp32_baud=%d;"
            "last_serial_cmd=%s;"
            "last_serial_error=%s;"
            "last_serial_attempts=%d;"
            "last_serial_reply=%s;"
            "serial_candidates=%s;"
            "last_drop_cmd=%s;"
            "last_drop_result=%s"
        ) % (
            state,
            self.esp32_port,
            self.esp32_baud,
            self.last_serial_cmd,
            self.last_serial_error,
            self.last_serial_attempts,
            self.last_serial_reply,
            ",".join(self.serial_candidates) if self.serial_candidates else "NONE",
            self.last_drop_cmd,
            self.last_drop_result
        )

    def build_status_text(self):
        return (
            "STATUS:"
            "fsm=%s;"
            "safety=%s;"
            "land=%s;"
            "connected=%s;"
            "armed=%s;"
            "mode=%s;"
            "%s;"
            "last_vision=%s"
        ) % (
            self.fsm_state,
            self.safety_state,
            self.land_status,
            self.mavros_connected,
            self.mavros_armed,
            self.mavros_mode,
            self.serial_status_text(),
            self.last_vision_summary
        )

    def build_ping_text(self):
        """
        PING 同时检查 UDP 接收节点、FSM 心跳和投放串口状态。
        """
        serial_part = self.serial_status_text()

        if self.fsm_state_last_time is None:
            return (
                "ACK:PING:RECEIVER_OK;"
                "receiver=OK;"
                "fsm=NO_HEARTBEAT;"
                "fsm_age=NA;"
                "state=%s;"
                "%s"
            ) % (self.fsm_state, serial_part)

        age = (rospy.Time.now() - self.fsm_state_last_time).to_sec()

        if age <= self.fsm_heartbeat_timeout:
            return (
                "ACK:PING:NORMAL;"
                "receiver=OK;"
                "fsm=NORMAL;"
                "fsm_age=%.2f;"
                "state=%s;"
                "%s"
            ) % (age, self.fsm_state, serial_part)

        return (
            "ACK:PING:RECEIVER_OK;"
            "receiver=OK;"
            "fsm=TIMEOUT;"
            "fsm_age=%.2f;"
            "state=%s;"
            "%s"
        ) % (age, self.fsm_state, serial_part)

    # =========================
    # UDP 命令处理
    # =========================

    def handle_command(self, raw_text, addr):
        cmd, error = self.parse_command(raw_text)

        if error:
            reply = "ERR:%s:%s" % (error, raw_text.strip())
            rospy.logwarn("UDP command rejected from %s: %s", str(addr), reply)
            self.send_reply(reply, addr)
            return

        # 记录最近一次地面站地址。视觉结果实时推送默认发给这个地址。
        self.last_client_addr = addr

        rospy.logwarn("UDP command received from %s: CMD:%s", str(addr), cmd)

        if cmd == "PING":
            self.send_reply(self.build_ping_text(), addr)
            return

        if cmd == "STATUS":
            self.send_reply(self.build_status_text(), addr)
            return

        # 地面站手动投放机构控制：直接转 ESP32 串口。
        if self.is_single_drop_cmd(cmd):
            ok, reason = self.send_serial_cmd(cmd)
            if ok:
                self.send_reply("ACK:%s:%s" % (cmd, reason), addr)
            else:
                self.send_reply("ERR:%s:%s" % (cmd, reason), addr)
            return

        # 兼容直接发送 L0/R0。地面站通常已经拆分，本节点再做一次兜底。
        if self.is_all_drop_cmd(cmd):
            ok, reason = self.handle_l0_r0(cmd)
            if ok:
                self.send_reply("ACK:%s:%s" % (cmd, reason), addr)
            else:
                self.send_reply("ERR:%s:%s" % (cmd, reason), addr)
            return

        if cmd == "START":
            self.publish_bool_pulse(self.start_pub, "/uav/start")
            self.send_reply("ACK:START:OK", addr)
            return

        if cmd == "LAND":
            self.publish_bool_pulse(self.land_pub, "/uav/land")
            self.send_reply("ACK:LAND:OK", addr)
            return

        if cmd == "STOP":
            self.publish_bool_pulse(self.stop_pub, "/uav/stop")
            self.send_reply("ACK:STOP:OK", addr)
            return

        if cmd == "RESET":
            self.publish_bool_pulse(self.reset_pub, "/uav/reset")
            self.send_reply("ACK:RESET:OK", addr)
            return

        if cmd == "DISARM":
            self.publish_bool_pulse(self.disarm_pub, "/uav/disarm")
            self.send_reply("ACK:DISARM:OK", addr)
            return

        self.send_reply("ERR:UNKNOWN_CMD:%s" % raw_text.strip(), addr)

    def spin(self):
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                rate.sleep()
                continue
            except Exception as e:
                rospy.logwarn("UDP recv error: %s", str(e))
                rate.sleep()
                continue

            try:
                text = data.decode("utf-8", errors="ignore").strip()
            except Exception:
                self.send_reply("ERR:DECODE_FAILED", addr)
                continue

            if text:
                self.handle_command(text, addr)

            rate.sleep()

        self.close()

    def close(self):
        try:
            if self.esp32_ser is not None:
                self.esp32_ser.close()
        except Exception:
            pass

        try:
            self.sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        node = UdpUavCmdReceiver()
        node.spin()
    except rospy.ROSInterruptException:
        pass
