#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
square_drop_udp_test.py

正方形飞行 + 舵机投放 + 简易 UDP 接收，一体化小测试脚本。

用途：
- 不依赖正式任务 udp_uav_cmd_receiver.py；
- 本脚本自己监听 UDP 8810；
- 本脚本自己自动寻找 ESP32 串口；
- 地面站可以直接发 START/LAND/STOP/L1/R1 等命令；
- 飞 0.8m x 0.8m 正方形，高度 0.5m；
- 除 (0,0) 顶点外，在另外三个顶点依次投放 R1/R2/R3。

默认 UDP：
    bind_ip   = 0.0.0.0
    bind_port = 8810

默认串口：
    esp32_port = /dev/ttyUSB0
    若不存在，会自动尝试：
        /dev/serial/by-id/*
        /dev/ttyUSB*
        /dev/ttyACM*

支持 UDP 指令：
    CMD:PING
    CMD:STATUS
    CMD:START 或 CMD:TAKEOFF      开始正方形飞行测试
    CMD:LAND                       普通降落
    CMD:STOP                       急停降落
    CMD:DISARM                     立即请求上锁
    CMD:L1/L2/L3                   锁定舵机
    CMD:R1/R2/R3                   释放舵机
    CMD:L0                         依次 L1 L2 L3
    CMD:R0                         依次 R1 R2 R3
"""

import os
import glob
import math
import time
import socket
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State

try:
    import serial
    HAS_SERIAL = True
except Exception:
    serial = None
    HAS_SERIAL = False


class SquareDropUdpTest:
    def __init__(self):
        rospy.init_node("square_drop_udp_test", anonymous=False)

        # ========== 参数 ==========
        self.bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        self.bind_port = int(rospy.get_param("~bind_port", 8810))

        self.target_altitude = float(rospy.get_param("~target_altitude", 0.5))
        self.square_side = float(rospy.get_param("~square_side", 0.8))

        # 判稳放松一些：只要进入这个半径就算到点。
        self.arrive_tolerance = float(rospy.get_param("~arrive_tolerance", 0.22))

        # 低速安全飞。
        self.speed = float(rospy.get_param("~speed", 0.18))
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))

        # 到点后保持时间，便于投放。
        self.vertex_hold_time = float(rospy.get_param("~vertex_hold_time", 0.7))
        self.drop_wait_time = float(rospy.get_param("~drop_wait_time", 0.8))

        # 降落速度。
        self.descend_speed = float(rospy.get_param("~descend_speed", 0.20))

        # ESP32 串口参数。
        self.drop_serial_enabled = bool(rospy.get_param("~drop_serial_enabled", True))
        self.esp32_port = str(rospy.get_param("~esp32_port", "/dev/ttyUSB0"))
        self.esp32_baud = int(rospy.get_param("~esp32_baud", 115200))
        self.serial_timeout = float(rospy.get_param("~serial_timeout", 0.5))
        self.serial_read_wait = float(rospy.get_param("~serial_read_wait", 1.0))
        self.serial_auto_detect = bool(rospy.get_param("~serial_auto_detect", True))
        self.serial_retry_count = int(rospy.get_param("~serial_retry_count", 3))
        self.serial_retry_delay = float(rospy.get_param("~serial_retry_delay", 0.2))
        self.drop_step_delay = float(rospy.get_param("~drop_step_delay", 0.35))
        self.esp32_boot_wait = float(rospy.get_param("~esp32_boot_wait", 3.0))

        # ========== 状态 ==========
        self.current_state = State()
        self.current_pos = PoseStamped()
        self.pose_ready = False

        self.start_requested = False
        self.mission_running = False
        self.land_requested = False
        self.stop_requested = False
        self.disarm_requested = False

        self.last_serial_cmd = "NONE"
        self.last_serial_result = "NONE"
        self.last_serial_reply = "NONE"
        self.last_serial_error = "NONE"
        self.last_serial_attempts = 0
        self.serial_candidates = []

        self.esp32_ser = None
        self.serial_lock = threading.Lock()

        # ========== ROS ==========
        rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=10)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pos_cb, queue_size=10)

        self.local_pos_pub = rospy.Publisher(
            "/mavros/setpoint_position/local",
            PoseStamped,
            queue_size=10
        )

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # ========== UDP ==========
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind((self.bind_ip, self.bind_port))
        self.udp_sock.settimeout(0.1)

        self.udp_thread = threading.Thread(target=self.udp_loop)
        self.udp_thread.daemon = True
        self.udp_thread.start()

        if self.drop_serial_enabled:
            self.open_esp32_serial()
        else:
            self.last_serial_error = "SERIAL_DISABLED"

        rospy.logwarn("square_drop_udp_test started.")
        rospy.logwarn("UDP listening on %s:%d", self.bind_ip, self.bind_port)
        rospy.logwarn(
            "Square: side=%.2fm altitude=%.2fm tolerance=%.2fm speed=%.2fm/s",
            self.square_side,
            self.target_altitude,
            self.arrive_tolerance,
            self.speed
        )
        rospy.logwarn(
            "Serial: port=%s baud=%d auto_detect=%s read_wait=%.2fs boot_wait=%.2fs retry=%d delay=%.2fs",
            self.esp32_port,
            self.esp32_baud,
            str(self.serial_auto_detect),
            self.serial_read_wait,
            self.esp32_boot_wait,
            self.serial_retry_count,
            self.serial_retry_delay
        )

    # ================= ROS 回调 =================

    def state_cb(self, msg):
        self.current_state = msg

    def pos_cb(self, msg):
        self.current_pos = msg
        self.pose_ready = True

    # ================= UDP =================

    def udp_loop(self):
        while not rospy.is_shutdown():
            try:
                data, addr = self.udp_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logwarn("UDP recv error: %s", str(e))
                continue

            text = data.decode("utf-8", errors="ignore").strip()
            if not text:
                continue

            reply = self.handle_udp_text(text)
            try:
                self.udp_sock.sendto((reply + "\n").encode("utf-8"), addr)
            except Exception as e:
                rospy.logwarn("UDP reply error: %s", str(e))

    def normalize_cmd(self, text):
        text = text.strip()
        if not text:
            return ""

        parts = text.split(":")
        if len(parts) >= 2 and parts[0].upper() == "CMD":
            return parts[1].strip().upper()

        return text.strip().upper()

    def handle_udp_text(self, text):
        cmd = self.normalize_cmd(text)
        rospy.logwarn("UDP CMD received: %s", cmd)

        alias = {
            "TAKEOFF": "START",
            "BEGIN": "START",
            "MISSION": "START",
            "E_STOP": "STOP",
            "KILL": "STOP",
            "RTL": "LAND",
        }
        cmd = alias.get(cmd, cmd)

        if cmd == "PING":
            return self.build_ping_text()

        if cmd == "STATUS":
            return self.build_status_text()

        if cmd == "START":
            if self.mission_running:
                return "ACK:START:MISSION_ALREADY_RUNNING"
            self.start_requested = True
            self.land_requested = False
            self.stop_requested = False
            return "ACK:START:REQUESTED"

        if cmd == "LAND":
            self.land_requested = True
            return "ACK:LAND:REQUESTED"

        if cmd == "STOP":
            self.stop_requested = True
            self.land_requested = True
            return "ACK:STOP:REQUESTED"

        if cmd == "DISARM":
            self.disarm_requested = True
            return "ACK:DISARM:REQUESTED"

        if self.is_single_drop_cmd(cmd):
            ok, reason = self.send_serial_cmd(cmd)
            if ok:
                return "ACK:%s:%s" % (cmd, reason)
            return "ERR:%s:%s" % (cmd, reason)

        if cmd in ["L0", "R0"]:
            ok, reason = self.handle_l0_r0(cmd)
            if ok:
                return "ACK:%s:%s" % (cmd, reason)
            return "ERR:%s:%s" % (cmd, reason)

        return "ERR:UNKNOWN_CMD:%s" % cmd

    def build_ping_text(self):
        return (
            "ACK:PING:SQUARE_TEST_OK;"
            "receiver=OK;"
            "mission_running=%s;"
            "armed=%s;"
            "mode=%s;"
            "%s"
        ) % (
            str(self.mission_running),
            str(self.current_state.armed),
            self.current_state.mode,
            self.serial_status_text()
        )

    def build_status_text(self):
        pos_text = "pose=NA"
        if self.pose_ready:
            p = self.current_pos.pose.position
            pos_text = "pose=%.2f,%.2f,%.2f" % (p.x, p.y, p.z)

        return (
            "STATUS:"
            "mission_running=%s;"
            "start_requested=%s;"
            "land_requested=%s;"
            "stop_requested=%s;"
            "connected=%s;"
            "armed=%s;"
            "mode=%s;"
            "%s;"
            "%s"
        ) % (
            str(self.mission_running),
            str(self.start_requested),
            str(self.land_requested),
            str(self.stop_requested),
            str(self.current_state.connected),
            str(self.current_state.armed),
            self.current_state.mode,
            pos_text,
            self.serial_status_text()
        )

    # ================= 串口 =================

    def list_serial_candidates(self):
        candidates = []

        def add(path):
            if not path:
                return
            if path in candidates:
                return
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

    def get_ports_to_try(self):
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

        if not ports:
            add_port(self.esp32_port)

        return ports

    def open_esp32_serial(self):
        if not self.drop_serial_enabled:
            self.last_serial_error = "SERIAL_DISABLED"
            return False

        if not HAS_SERIAL:
            self.last_serial_error = "PY_SERIAL_NOT_INSTALLED"
            rospy.logerr("pyserial not installed. Try: sudo apt install python3-serial")
            return False

        with self.serial_lock:
            try:
                if self.esp32_ser is not None and self.esp32_ser.is_open:
                    return True
            except Exception:
                pass

            errors = []
            ports_to_try = self.get_ports_to_try()

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
                    rospy.logwarn("Open serial failed on %s: %s", port, str(e))

            self.last_serial_error = " | ".join(errors) if errors else "NO_SERIAL_CANDIDATE"
            rospy.logerr("Open ESP32 serial failed: %s", self.last_serial_error)
            return False

    def ensure_serial_ready(self):
        try:
            if self.esp32_ser is not None and self.esp32_ser.is_open:
                return True
        except Exception:
            self.esp32_ser = None

        return self.open_esp32_serial()

    def read_esp32_lines(self, wait_time=None):
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

        if any("OK" in line for line in upper_lines):
            return True, "OK_REPLY_GENERIC:%s" % joined

        return False, "NO_OK_REPLY:%s" % joined

    def is_single_drop_cmd(self, cmd):
        return len(cmd) == 2 and cmd[0] in ["L", "R"] and cmd[1] in ["1", "2", "3"]

    def send_serial_cmd(self, cmd):
        cmd = cmd.strip().upper()

        if not self.is_single_drop_cmd(cmd):
            return False, "BAD_SERIAL_CMD"

        if not self.ensure_serial_ready():
            return False, "SERIAL_NOT_READY:%s" % self.last_serial_error

        data = (cmd + "\n").encode("utf-8")
        max_try = max(1, self.serial_retry_count)
        all_lines = []
        last_reason = "NO_ATTEMPT"

        for attempt in range(1, max_try + 1):
            with self.serial_lock:
                try:
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
                    self.last_serial_attempts = attempt
                    self.last_serial_reply = " | ".join(all_lines) if all_lines else "NONE"
                    return False, "SERIAL_WRITE_ERR:%s" % str(e)

                lines = self.read_esp32_lines()

            if lines:
                all_lines.extend(["try%d:%s" % (attempt, line) for line in lines])
                rospy.logwarn("ESP32 reply for %s try %d/%d: %s", cmd, attempt, max_try, " | ".join(lines))
            else:
                rospy.logwarn("ESP32 no reply for %s try %d/%d", cmd, attempt, max_try)

            ok, reason = self.evaluate_esp32_reply(cmd, lines)
            last_reason = reason

            if ok:
                self.last_serial_cmd = cmd
                self.last_serial_result = "SERIAL_OK:try=%d" % attempt
                self.last_serial_error = "OK"
                self.last_serial_attempts = attempt
                self.last_serial_reply = " | ".join(all_lines) if all_lines else "NONE"
                return True, self.last_serial_result

            if attempt < max_try:
                rospy.sleep(self.serial_retry_delay)

        self.last_serial_cmd = cmd
        self.last_serial_result = "SERIAL_NO_OK_AFTER_%d:%s" % (max_try, last_reason)
        self.last_serial_error = last_reason
        self.last_serial_attempts = max_try
        self.last_serial_reply = " | ".join(all_lines) if all_lines else "NONE"
        return False, self.last_serial_result

    def handle_l0_r0(self, cmd):
        prefix = cmd[0]
        results = []

        for i in range(1, 4):
            sub_cmd = "%s%d" % (prefix, i)
            ok, reason = self.send_serial_cmd(sub_cmd)
            results.append("%s:%s" % (sub_cmd, reason))
            if not ok:
                return False, ";".join(results)
            rospy.sleep(self.drop_step_delay)

        return True, ";".join(results)

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
            "last_serial_result=%s;"
            "last_serial_error=%s;"
            "last_serial_attempts=%d;"
            "last_serial_reply=%s;"
            "serial_candidates=%s"
        ) % (
            state,
            self.esp32_port,
            self.esp32_baud,
            self.last_serial_cmd,
            self.last_serial_result,
            self.last_serial_error,
            self.last_serial_attempts,
            self.last_serial_reply,
            ",".join(self.serial_candidates) if self.serial_candidates else "NONE",
        )

    # ================= 飞行控制 =================

    def get_distance(self, x1, y1, z1, x2, y2, z2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    def make_pose(self, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        # 固定 yaw = 0。
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def publish_pose(self, pose):
        pose.header.stamp = rospy.Time.now()
        self.local_pos_pub.publish(pose)

    def wait_fcu_and_pose(self, rate):
        rospy.logwarn("Waiting FCU connection and local pose...")

        while not rospy.is_shutdown():
            if self.current_state.connected and self.pose_ready:
                break
            rate.sleep()

        rospy.logwarn("FCU connected and pose ready.")

    def enter_offboard_and_arm(self, pose, rate):
        # 先发 setpoint 心跳。
        rospy.logwarn("Sending pre-offboard setpoints...")
        for _ in range(100):
            if rospy.is_shutdown():
                return False
            self.publish_pose(pose)
            rate.sleep()

        rospy.logwarn("Requesting OFFBOARD and arm...")
        last_req = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.stop_requested or self.land_requested:
                return False

            now = rospy.Time.now()

            if self.current_state.mode != "OFFBOARD" and (now - last_req) > rospy.Duration(2.0):
                try:
                    self.set_mode_client(base_mode=0, custom_mode="OFFBOARD")
                except Exception as e:
                    rospy.logwarn("Set OFFBOARD failed: %s", str(e))
                last_req = now

            elif not self.current_state.armed and (now - last_req) > rospy.Duration(2.0):
                try:
                    self.arming_client(True)
                except Exception as e:
                    rospy.logwarn("Arm failed: %s", str(e))
                last_req = now

            self.publish_pose(pose)

            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                rospy.logwarn("OFFBOARD and armed.")
                return True

            rate.sleep()

        return False

    def fly_to_waypoint(self, pose, wp, rate):
        target_x, target_y, target_z = wp
        dt = 1.0 / self.rate_hz
        step_dist = max(0.02, self.speed * dt)

        while not rospy.is_shutdown():
            if self.stop_requested or self.land_requested:
                return False

            dx = target_x - pose.pose.position.x
            dy = target_y - pose.pose.position.y
            dz = target_z - pose.pose.position.z
            cmd_dist = self.get_distance(
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                target_x,
                target_y,
                target_z
            )

            if cmd_dist > step_dist:
                pose.pose.position.x += dx / cmd_dist * step_dist
                pose.pose.position.y += dy / cmd_dist * step_dist
                pose.pose.position.z += dz / cmd_dist * step_dist
            else:
                pose.pose.position.x = target_x
                pose.pose.position.y = target_y
                pose.pose.position.z = target_z

            if self.pose_ready:
                p = self.current_pos.pose.position
                real_dist = self.get_distance(p.x, p.y, p.z, target_x, target_y, target_z)
            else:
                real_dist = 999.0

            self.publish_pose(pose)

            if real_dist < self.arrive_tolerance:
                rospy.logwarn(
                    "Arrived waypoint: x=%.2f y=%.2f z=%.2f real_dist=%.2f",
                    target_x,
                    target_y,
                    target_z,
                    real_dist
                )
                self.hold_pose(pose, self.vertex_hold_time, rate)
                return True

            rate.sleep()

        return False

    def hold_pose(self, pose, hold_time, rate):
        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start) < rospy.Duration(hold_time):
            if self.stop_requested or self.land_requested:
                return False
            self.publish_pose(pose)
            rate.sleep()
        return True

    def run_square_mission(self, rate):
        self.mission_running = True
        self.start_requested = False
        self.land_requested = False
        self.stop_requested = False

        start_pose = self.make_pose(
            self.current_pos.pose.position.x,
            self.current_pos.pose.position.y,
            max(self.current_pos.pose.position.z, 0.05)
        )

        if not self.enter_offboard_and_arm(start_pose, rate):
            self.mission_running = False
            return

        side = self.square_side
        z = self.target_altitude

        waypoints = [
            (0.0, 0.0, z, None),
            (side, 0.0, z, "R1"),
            (side, side, z, "R2"),
            (0.0, side, z, "R3"),
            (0.0, 0.0, z, None),
        ]

        pose = start_pose

        rospy.logwarn("========== Square drop test mission started ==========")

        for idx, item in enumerate(waypoints):
            wp = item[:3]
            drop_cmd = item[3]
            rospy.logwarn("Waypoint %d/%d -> x=%.2f y=%.2f z=%.2f drop=%s",
                          idx + 1, len(waypoints), wp[0], wp[1], wp[2], str(drop_cmd))

            ok = self.fly_to_waypoint(pose, wp, rate)
            if not ok:
                rospy.logwarn("Mission interrupted before waypoint %d.", idx + 1)
                break

            if drop_cmd is not None:
                rospy.logwarn("Drop at waypoint %d: %s", idx + 1, drop_cmd)
                ok, reason = self.send_serial_cmd(drop_cmd)
                if ok:
                    rospy.logwarn("Drop %s OK: %s", drop_cmd, reason)
                else:
                    rospy.logerr("Drop %s failed: %s", drop_cmd, reason)
                self.hold_pose(pose, self.drop_wait_time, rate)

        rospy.logwarn("========== Square mission finished/interrupted. Landing... ==========")
        self.land_requested = True
        self.precise_land(pose, rate)
        self.mission_running = False

    def precise_land(self, pose, rate):
        dt = 1.0 / self.rate_hz
        dz = max(0.02, self.descend_speed * dt)

        while not rospy.is_shutdown():
            if pose.pose.position.z > -0.05:
                pose.pose.position.z -= dz

            self.publish_pose(pose)

            if self.pose_ready and self.current_pos.pose.position.z < 0.10:
                rospy.logwarn("Touchdown detected by local z < 0.10m.")
                break

            rate.sleep()

        # 贴地保持 1s。
        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start) < rospy.Duration(1.0):
            self.publish_pose(pose)
            rate.sleep()

        self.request_disarm(rate)

    def request_auto_land(self):
        try:
            self.set_mode_client(base_mode=0, custom_mode="AUTO.LAND")
            rospy.logwarn("AUTO.LAND requested.")
        except Exception as e:
            rospy.logwarn("AUTO.LAND request failed: %s", str(e))

    def request_disarm(self, rate):
        rospy.logwarn("Requesting disarm...")
        while not rospy.is_shutdown() and self.current_state.armed:
            try:
                self.arming_client(False)
            except Exception as e:
                rospy.logwarn("Disarm failed: %s", str(e))
            rate.sleep()
        rospy.logwarn("Disarm done or vehicle already disarmed.")

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        self.wait_fcu_and_pose(rate)

        while not rospy.is_shutdown():
            if self.disarm_requested:
                self.disarm_requested = False
                self.request_disarm(rate)

            if self.stop_requested and self.current_state.armed:
                rospy.logerr("STOP requested. AUTO.LAND.")
                self.request_auto_land()
                self.stop_requested = False
                self.land_requested = False

            if self.land_requested and (not self.mission_running) and self.current_state.armed:
                rospy.logwarn("LAND requested while mission is not running. AUTO.LAND.")
                self.request_auto_land()
                self.land_requested = False

            if self.start_requested and not self.mission_running:
                self.run_square_mission(rate)

            rate.sleep()

        try:
            if self.esp32_ser is not None:
                self.esp32_ser.close()
        except Exception:
            pass

        try:
            self.udp_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        node = SquareDropUdpTest()
        node.spin()
    except rospy.ROSInterruptException:
        pass
