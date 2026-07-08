#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
无人机圆形绕障飞行程序：地面站控制 + 等待阶段持续 setpoint 版
============================================================

【文件版本】
2026-07-04-v5-低电压极限绕圈保护版

【这个版本解决的问题】
之前的地面站版脚本在等待地面站 TAKEOFF 指令时，可能没有持续发布
OFFBOARD 所需的 setpoint。这样如果你们提前用遥控器切到 OFFBOARD 并解锁，
飞控可能因为没有持续收到外部控制目标而无法保持 OFFBOARD，最终表现为：
地面站能发 TAKEOFF，机载端也能回 ACK，但无人机不真正起飞。

本版本改成：
1. 脚本启动并获得本地位置后，立刻开始持续发布“当前位置保持 setpoint”。
2. 等待 TAKEOFF 时也持续发布 setpoint，证明外部控制器在线。
3. 收到 TAKEOFF 后，再把高度目标从地面高度平滑抬高到起飞高度。
4. 收到 LAND 后，先发布当前位置、速度为 0、加速度为 0 的刹停目标，再降落上锁。
5. 起飞后默认持续绕圈，不再按固定圈数结束；当平均单节电压低于阈值时，先悬停刹停，再请求 AUTO.LAND。

【运行位置】
本文件运行在无人机机载电脑 Jetson / Ubuntu / ROS 环境中。

【通信方式】
Windows 地面站通过 UDP 发送普通文本指令：
- TAKEOFF / START：允许起飞并执行任务
- LAND：中断当前任务，刹停后降落上锁
- STATUS：查询当前飞控和脚本状态
- L1/L2/L3：起飞前锁定 1/2/3 号舵机
- R1/R2/R3：起飞前释放 1/2/3 号舵机
- L0/R0：起飞前依次锁定/释放 1/2/3 号舵机

注意：圆形飞行测试不包含自动投放环节。舵机命令只允许在未解锁、未起飞、任务未运行时执行，用于起飞前带配重状态调整。

【控制方式】
无人机轨迹控制使用 /mavros/setpoint_raw/local，消息类型为 PositionTarget：
- position：期望位置
- velocity：速度前馈
- acceleration_or_force：加速度前馈
- yaw：期望航向角

【重要安全说明】
1. 这个 LAND 是软件降落，不是硬件急停。
2. 实飞时必须保留遥控器接管、kill 开关和现场安全员。
3. 首次实飞建议先设置只起飞悬停，不自动绕圆。
4. 首次绕圆建议低速、大半径。
5. 电池低电压保护基于 /mavros/battery；实飞前必须确认该话题电压数据正确。
"""

# ============================================================
# 一、导入依赖库区
# ============================================================

import math
import os
import glob
import socket
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from sensor_msgs.msg import BatteryState
from mavros_msgs.srv import CommandBool, SetMode
from tf.transformations import euler_from_quaternion

try:
    import serial
    HAS_SERIAL = True
except Exception:
    serial = None
    HAS_SERIAL = False


# ============================================================
# 二、常用参数修改区
# ============================================================
# 你平时最常改的参数集中放在这里。
# 如果你不想在 rosrun 命令后面写 _radius:=...，就直接改这里。

# ------------------------------
# 2.1 地面站 UDP 通信参数
# ------------------------------
DEFAULT_UDP_BIND_IP = "0.0.0.0"       # 监听所有网卡；实机和仿真都建议这样写
DEFAULT_UDP_PORT = 7777               # 机载端监听端口；Windows 地面站要发到这个端口
DEFAULT_UDP_RECV_TIMEOUT = 0.2        # UDP 接收超时时间，单位 s；用于让线程能定期退出检查

# ------------------------------
# 2.1.1 起飞前舵机 / ESP32 串口参数
# ------------------------------
# 圆形绕障脚本不做自动投放，只允许在未起飞时通过地面站手动控制舵机。
DEFAULT_DROP_SERIAL_ENABLED = True
DEFAULT_ESP32_PORT = "/dev/ttyUSB0"
DEFAULT_ESP32_BAUD = 115200
DEFAULT_SERIAL_TIMEOUT = 0.5
DEFAULT_SERIAL_READ_WAIT = 1.0
DEFAULT_SERIAL_AUTO_DETECT = True
DEFAULT_SERIAL_RETRY_COUNT = 3
DEFAULT_SERIAL_RETRY_DELAY = 0.2
DEFAULT_DROP_STEP_DELAY = 0.35
DEFAULT_ESP32_BOOT_WAIT = 3.0

# ------------------------------
# 2.2 起飞和任务参数
# ------------------------------
DEFAULT_TAKEOFF_ALT = 1             # 起飞高度，单位 m
DEFAULT_START_CIRCLE_AFTER_TAKEOFF = True   # True：起飞后自动绕圆；False：只起飞悬停
DEFAULT_LAND_AFTER_MISSION = True            # True：任务结束自动降落；False：任务结束后悬停等 LAND

# ------------------------------
# 2.3 圆形绕障参数
# ------------------------------
DEFAULT_RADIUS = 0.9                  # 绕圆半径，单位 m；实飞首次建议 >= 1.2
DEFAULT_LOOPS = 3.0                   # 有限圈模式下的绕圆圈数；极限绕圈模式开启时，这个值只用于显示，不限制飞行
DEFAULT_CIRCLE_SPEED = 0.85            # 绕圆切向速度，单位 m/s；首次实飞建议 0.3~0.4
DEFAULT_CIRCLE_UNTIL_LOW_BATTERY = True  # True：一直绕圈，直到低电压或 LAND；False：按 DEFAULT_LOOPS 绕完
DEFAULT_CENTER_X = 1.5                # 圆心相对起飞点的 x 偏移，单位 m
DEFAULT_CENTER_Y = 0.0                # 圆心相对起飞点的 y 偏移，单位 m

# ------------------------------
# 2.4 轨迹和平滑参数
# ------------------------------
DEFAULT_RATE_HZ = 50.0                # setpoint 发布频率，单位 Hz；OFFBOARD 必须持续发布
DEFAULT_TAKEOFF_SPEED = 0.25          # 起飞阶段的上升速度，单位 m/s
DEFAULT_TRANSFER_SPEED = 0.4          # 转场速度，单位 m/s
DEFAULT_DESCEND_SPEED = 0.20          # 兜底下降速度，单位 m/s
DEFAULT_RAMP_TIME = 2.0               # 绕圆起步加速/结束减速时间，单位 s
DEFAULT_BRAKE_HOLD_TIME = 1.5         # 收到 LAND 后，先刹停悬停的时间，单位 s
DEFAULT_HOLD_AFTER_TAKEOFF_TIME = 2.0 # 起飞到目标高度后悬停时间，单位 s

# ------------------------------
# 2.4.1 到达判定参数
# ------------------------------
# 重要：不能只按“轨迹发布时间结束”判断到达，必须用实际 /mavros/local_position/pose 判断。
DEFAULT_XY_REACH_TOL = 0.15            # x/y 到达容差，单位 m
DEFAULT_Z_REACH_TOL = 0.10             # z 到达容差，单位 m
DEFAULT_REACH_STABLE_TIME = 0.6        # 连续满足到达条件的时间，单位 s
DEFAULT_REACH_TIMEOUT_MARGIN = 6.0     # 轨迹时间外额外等待时间，单位 s
DEFAULT_MIN_CIRCLE_ALT_RATIO = 0.85    # 实际高度至少达到目标高度的比例，才允许进入绕圆/转场

# ------------------------------
# 2.5 安全限制参数
# ------------------------------
DEFAULT_MAX_CENTRIPETAL_ACC = 1.0     # 实飞建议先保守，单位 m/s^2；确认稳定后再提高
DEFAULT_ENFORCE_ACC_LIMIT = True      # True：超过向心加速度限制就拒绝起飞
DEFAULT_OFFBOARD_ARM_TIMEOUT = 20.0   # 收到 TAKEOFF 后等待 OFFBOARD + ARM 的最长时间，单位 s
DEFAULT_AUTO_OFFBOARD = True          # True：收到 TAKEOFF 后脚本自动请求 OFFBOARD
DEFAULT_AUTO_ARM = True               # True：收到 TAKEOFF 后脚本自动请求解锁
DEFAULT_USE_AUTO_LAND = True          # True：优先使用 AUTO.LAND 降落
DEFAULT_AUTO_DISARM = True            # True：降落后自动上锁

# ------------------------------
# 2.6 电池低电压保护参数
# ------------------------------
DEFAULT_ENABLE_BATTERY_AUTO_LAND = True      # True：启用低电压自动降落保护
DEFAULT_LOW_CELL_VOLTAGE = 3.90              # 平均单节电压低于该值时触发保护，单位 V
DEFAULT_BATTERY_CELL_COUNT = 6               # 电池串数；FS-J310 说明书电池为 6S
DEFAULT_LOW_BATTERY_CONFIRM_TIME = 2.0       # 低电压持续超过该时间才触发，避免瞬间压降误判，单位 s
DEFAULT_REQUIRE_BATTERY_DATA_FOR_TAKEOFF = False  # True：没收到电池数据就拒绝起飞；SITL 测试建议 False


# ------------------------------
# 2.7 坐标和航向参数
# ------------------------------
DEFAULT_RELATIVE_FRAME = "body"       # body：圆心相对起飞时机头方向；local：直接使用 local 坐标
DEFAULT_CLOCKWISE = False             # False：逆时针；True：顺时针
DEFAULT_YAW_MODE = "fixed"            # fixed：机头固定；tangent：机头沿圆切线


# ============================================================
# 三、通用工具函数区
# ============================================================

def clamp(value, min_value, max_value):
    """把数值限制在指定范围内。"""
    return max(min_value, min(max_value, value))


def wrap_pi(angle):
    """把角度限制到 [-pi, pi]，避免 yaw 角过大。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def now_text():
    """生成简单的时间字符串，方便日志阅读。"""
    return time.strftime("%H:%M:%S")


# ============================================================
# 四、主飞行控制类
# ============================================================

class CircleObstacleGroundStationNode:
    """
    地面站控制的圆形绕障节点。

    这个类负责：
    1. 监听 Windows 地面站 UDP 指令。
    2. 等待 MAVROS 和本地位置。
    3. 等待 TAKEOFF 期间持续发布地面保持 setpoint。
    4. 收到 TAKEOFF 后完成 OFFBOARD、ARM、起飞、绕圆。
    5. 收到 LAND 后中断任务、刹停、降落、上锁。
    """

    # ------------------------------------------------------------
    # 4.1 初始化区
    # ------------------------------------------------------------

    def __init__(self):
        """初始化 ROS 节点、参数、发布器、订阅器、服务和 UDP 线程。"""
        rospy.init_node("circle_obstacle_gs_safe_v4")

        # ------------------------------
        # 读取 ROS 参数
        # ------------------------------
        self.udp_bind_ip = rospy.get_param("~udp_bind_ip", DEFAULT_UDP_BIND_IP)
        self.udp_port = int(rospy.get_param("~udp_port", DEFAULT_UDP_PORT))
        self.udp_recv_timeout = float(rospy.get_param("~udp_recv_timeout", DEFAULT_UDP_RECV_TIMEOUT))


        # 起飞前舵机 / ESP32 串口参数。
        self.drop_serial_enabled = bool(rospy.get_param("~drop_serial_enabled", DEFAULT_DROP_SERIAL_ENABLED))
        self.esp32_port = str(rospy.get_param("~esp32_port", DEFAULT_ESP32_PORT))
        self.esp32_baud = int(rospy.get_param("~esp32_baud", DEFAULT_ESP32_BAUD))
        self.serial_timeout = float(rospy.get_param("~serial_timeout", DEFAULT_SERIAL_TIMEOUT))
        self.serial_read_wait = float(rospy.get_param("~serial_read_wait", DEFAULT_SERIAL_READ_WAIT))
        self.serial_auto_detect = bool(rospy.get_param("~serial_auto_detect", DEFAULT_SERIAL_AUTO_DETECT))
        self.serial_retry_count = int(rospy.get_param("~serial_retry_count", DEFAULT_SERIAL_RETRY_COUNT))
        self.serial_retry_delay = float(rospy.get_param("~serial_retry_delay", DEFAULT_SERIAL_RETRY_DELAY))
        self.drop_step_delay = float(rospy.get_param("~drop_step_delay", DEFAULT_DROP_STEP_DELAY))
        self.esp32_boot_wait = float(rospy.get_param("~esp32_boot_wait", DEFAULT_ESP32_BOOT_WAIT))

        self.rate_hz = float(rospy.get_param("~rate_hz", DEFAULT_RATE_HZ))
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", DEFAULT_TAKEOFF_ALT))
        self.takeoff_speed = float(rospy.get_param("~takeoff_speed", DEFAULT_TAKEOFF_SPEED))
        self.transfer_speed = float(rospy.get_param("~transfer_speed", DEFAULT_TRANSFER_SPEED))
        self.descend_speed = float(rospy.get_param("~descend_speed", DEFAULT_DESCEND_SPEED))
        self.ramp_time = float(rospy.get_param("~ramp_time", DEFAULT_RAMP_TIME))
        self.brake_hold_time = float(rospy.get_param("~brake_hold_time", DEFAULT_BRAKE_HOLD_TIME))
        self.hold_after_takeoff_time = float(rospy.get_param("~hold_after_takeoff_time", DEFAULT_HOLD_AFTER_TAKEOFF_TIME))

        # 到达判定参数：用于防止“时间走完但飞机没到位”时提前进入下一阶段。
        self.xy_reach_tol = float(rospy.get_param("~xy_reach_tol", DEFAULT_XY_REACH_TOL))
        self.z_reach_tol = float(rospy.get_param("~z_reach_tol", DEFAULT_Z_REACH_TOL))
        self.reach_stable_time = float(rospy.get_param("~reach_stable_time", DEFAULT_REACH_STABLE_TIME))
        self.reach_timeout_margin = float(rospy.get_param("~reach_timeout_margin", DEFAULT_REACH_TIMEOUT_MARGIN))
        self.min_circle_alt_ratio = float(rospy.get_param("~min_circle_alt_ratio", DEFAULT_MIN_CIRCLE_ALT_RATIO))

        self.radius = float(rospy.get_param("~radius", DEFAULT_RADIUS))
        self.loops = float(rospy.get_param("~loops", DEFAULT_LOOPS))
        self.circle_speed = float(rospy.get_param("~circle_speed", DEFAULT_CIRCLE_SPEED))
        self.circle_until_low_battery = bool(rospy.get_param(
            "~circle_until_low_battery",
            DEFAULT_CIRCLE_UNTIL_LOW_BATTERY
        ))
        self.center_x = float(rospy.get_param("~center_x", DEFAULT_CENTER_X))
        self.center_y = float(rospy.get_param("~center_y", DEFAULT_CENTER_Y))

        self.start_circle_after_takeoff = bool(rospy.get_param(
            "~start_circle_after_takeoff",
            DEFAULT_START_CIRCLE_AFTER_TAKEOFF
        ))
        self.land_after_mission = bool(rospy.get_param(
            "~land_after_mission",
            DEFAULT_LAND_AFTER_MISSION
        ))

        self.max_centripetal_acc = float(rospy.get_param(
            "~max_centripetal_acc",
            DEFAULT_MAX_CENTRIPETAL_ACC
        ))
        self.enforce_acc_limit = bool(rospy.get_param(
            "~enforce_acc_limit",
            DEFAULT_ENFORCE_ACC_LIMIT
        ))
        self.offboard_arm_timeout = float(rospy.get_param(
            "~offboard_arm_timeout",
            DEFAULT_OFFBOARD_ARM_TIMEOUT
        ))

        self.auto_offboard = bool(rospy.get_param("~auto_offboard", DEFAULT_AUTO_OFFBOARD))
        self.auto_arm = bool(rospy.get_param("~auto_arm", DEFAULT_AUTO_ARM))
        self.use_auto_land = bool(rospy.get_param("~use_auto_land", DEFAULT_USE_AUTO_LAND))
        self.auto_disarm = bool(rospy.get_param("~auto_disarm", DEFAULT_AUTO_DISARM))

        # 电池低电压保护参数
        self.enable_battery_auto_land = bool(rospy.get_param(
            "~enable_battery_auto_land",
            DEFAULT_ENABLE_BATTERY_AUTO_LAND
        ))
        self.low_cell_voltage = float(rospy.get_param(
            "~low_cell_voltage",
            DEFAULT_LOW_CELL_VOLTAGE
        ))
        self.battery_cell_count = int(rospy.get_param(
            "~battery_cell_count",
            DEFAULT_BATTERY_CELL_COUNT
        ))
        self.low_battery_confirm_time = float(rospy.get_param(
            "~low_battery_confirm_time",
            DEFAULT_LOW_BATTERY_CONFIRM_TIME
        ))
        self.require_battery_data_for_takeoff = bool(rospy.get_param(
            "~require_battery_data_for_takeoff",
            DEFAULT_REQUIRE_BATTERY_DATA_FOR_TAKEOFF
        ))

        self.relative_frame = rospy.get_param("~relative_frame", DEFAULT_RELATIVE_FRAME)
        self.clockwise = bool(rospy.get_param("~clockwise", DEFAULT_CLOCKWISE))
        self.yaw_mode = rospy.get_param("~yaw_mode", DEFAULT_YAW_MODE)

        # ------------------------------
        # 飞控状态变量
        # ------------------------------
        self.state = State()
        self.pose = None
        self.last_pose_time = None

        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_yaw = 0.0
        self.home_ready = False

        # ------------------------------
        # 电池状态变量
        # ------------------------------
        self.battery_msg = None                 # 保存 /mavros/battery 原始消息
        self.last_battery_time = None           # 最近一次收到电池消息的时间
        self.battery_avg_cell_voltage = None    # 计算得到的平均单节电压
        self.low_battery_since = None           # 电压首次低于阈值的时间
        self.battery_land_already_reported = False  # 防止低电压提示刷屏
        self.completed_loops = 0.0              # 已经绕过的圈数，方便极限圈数统计

        # ------------------------------
        # 地面站命令状态变量
        # ------------------------------
        self.command_lock = threading.Lock()
        self.takeoff_requested = False
        self.land_requested = False
        self.mission_active = False
        self.last_ground_addr = None
        self.last_command_text = "无"


        # ------------------------------
        # 起飞前舵机 / ESP32 串口状态变量
        # ------------------------------
        self.esp32_ser = None
        self.serial_lock = threading.Lock()
        self.last_serial_cmd = "NONE"
        self.last_serial_result = "NONE"
        self.last_serial_reply = "NONE"
        self.last_serial_error = "NONE"
        self.last_serial_attempts = 0
        self.serial_candidates = []

        # ------------------------------
        # ROS 发布器
        # ------------------------------
        self.raw_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local",
            PositionTarget,
            queue_size=20
        )

        # ------------------------------
        # ROS 订阅器
        # ------------------------------
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mavros/battery", BatteryState, self.battery_cb)

        # ------------------------------
        # MAVROS 服务客户端
        # ------------------------------
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.rate = rospy.Rate(self.rate_hz)

        # ------------------------------
        # UDP 套接字和线程
        # ------------------------------
        self.udp_sock = None
        self.udp_thread = None
        self.udp_thread_stop = False


        # 脚本启动时尝试打开 ESP32 串口；如果此时没插好，后续收到 L/R 命令时还会自动重试。
        if self.drop_serial_enabled:
            self.open_esp32_serial()
        else:
            self.last_serial_error = "SERIAL_DISABLED"

    # ------------------------------------------------------------
    # 4.2 ROS 回调函数区
    # ------------------------------------------------------------

    def state_cb(self, msg):
        """接收 /mavros/state，保存飞控连接、解锁和模式信息。"""
        self.state = msg

    def pose_cb(self, msg):
        """接收 /mavros/local_position/pose，保存无人机当前位姿。"""
        self.pose = msg
        self.last_pose_time = rospy.Time.now()

    def battery_cb(self, msg):
        """
        接收 /mavros/battery，计算平均单节电压。

        计算逻辑：
        1. 如果 BatteryState.cell_voltage 数组有效，就直接对各节电压求平均。
        2. 如果 cell_voltage 为空，就用总电压 voltage 除以电池串数。
        3. FS-J310 说明书电池为 6S，所以默认 battery_cell_count=6。
        """
        self.battery_msg = msg
        self.last_battery_time = rospy.Time.now()

        valid_cells = [v for v in msg.cell_voltage if v is not None and v > 0.1]
        if valid_cells:
            self.battery_avg_cell_voltage = sum(valid_cells) / float(len(valid_cells))
        elif msg.voltage is not None and msg.voltage > 0.1 and self.battery_cell_count > 0:
            self.battery_avg_cell_voltage = msg.voltage / float(self.battery_cell_count)
        else:
            self.battery_avg_cell_voltage = None

    # ------------------------------------------------------------
    # 4.3 UDP 地面站通信区
    # ------------------------------------------------------------

    def start_udp_server(self):
        """启动 UDP 监听线程，用于接收 Windows 地面站指令。"""
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind((self.udp_bind_ip, self.udp_port))
        self.udp_sock.settimeout(self.udp_recv_timeout)

        self.udp_thread = threading.Thread(target=self.udp_loop, daemon=True)
        self.udp_thread.start()

        rospy.loginfo(
            "UDP 地面站监听已启动：%s:%d",
            self.udp_bind_ip,
            self.udp_port
        )

    def udp_loop(self):
        """UDP 后台线程：循环接收 TAKEOFF、LAND、STATUS 指令。"""
        while not rospy.is_shutdown() and not self.udp_thread_stop:
            try:
                data, addr = self.udp_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            raw_text = data.decode("utf-8", errors="ignore").strip()
            command = self.normalize_udp_command(raw_text)
            if not command:
                continue

            with self.command_lock:
                self.last_ground_addr = addr
                self.last_command_text = command

            rospy.loginfo("收到地面站指令：%s，来源：%s:%d", command, addr[0], addr[1])

            if command in ["TAKEOFF", "START", "BEGIN"]:
                self.handle_takeoff_command(addr)
            elif command == "LAND":
                self.handle_land_command(addr)
            elif command == "STATUS":
                self.send_status(addr)
            elif command == "PING":
                self.send_udp_message(addr, "ACK:PING:CIRCLE_TEST_OK;receiver=OK;%s" % self.serial_status_text())
            elif self.is_single_servo_cmd(command):
                ok, reason = self.handle_preflight_servo_cmd(command)
                if ok:
                    self.send_udp_message(addr, "ACK:%s:%s" % (command, reason))
                else:
                    self.send_udp_message(addr, "ERR:%s:%s" % (command, reason))
            elif command in ["L0", "R0"]:
                ok, reason = self.handle_l0_r0(command)
                if ok:
                    self.send_udp_message(addr, "ACK:%s:%s" % (command, reason))
                else:
                    self.send_udp_message(addr, "ERR:%s:%s" % (command, reason))
            else:
                self.send_udp_message(addr, "UNKNOWN：未知指令 %s" % command)


    def normalize_udp_command(self, text):
        """统一 UDP 命令格式，兼容 TAKEOFF 和 CMD:TAKEOFF 两种写法。"""
        text = text.strip()
        if not text:
            return ""

        parts = text.split(":")
        if len(parts) >= 2 and parts[0].strip().upper() == "CMD":
            return parts[1].strip().upper()

        return text.strip().upper()

    def is_single_servo_cmd(self, cmd):
        """判断是否为单个舵机命令：L1/L2/L3/R1/R2/R3。"""
        cmd = cmd.strip().upper()
        return len(cmd) == 2 and cmd[0] in ["L", "R"] and cmd[1] in ["1", "2", "3"]

    def is_servo_control_allowed(self):
        """圆形测试只允许起飞前控制舵机，飞行中禁止误触发。"""
        with self.command_lock:
            mission_active = self.mission_active
            takeoff_requested = self.takeoff_requested

        if mission_active or takeoff_requested or self.state.armed:
            return False
        return True

    def handle_preflight_servo_cmd(self, cmd):
        """处理起飞前舵机命令。"""
        if not self.is_servo_control_allowed():
            return False, "BLOCKED_NOT_PREFLIGHT:armed=%s,mission_active=%s,takeoff_requested=%s" % (
                self.state.armed,
                self.mission_active,
                self.takeoff_requested,
            )
        return self.send_serial_cmd(cmd)

    def handle_l0_r0(self, cmd):
        """起飞前依次执行 L1/L2/L3 或 R1/R2/R3。"""
        if not self.is_servo_control_allowed():
            return False, "BLOCKED_NOT_PREFLIGHT:armed=%s,mission_active=%s,takeoff_requested=%s" % (
                self.state.armed,
                self.mission_active,
                self.takeoff_requested,
            )

        prefix = cmd[0].upper()
        results = []

        for i in range(1, 4):
            sub_cmd = "%s%d" % (prefix, i)
            ok, reason = self.send_serial_cmd(sub_cmd)
            results.append("%s:%s" % (sub_cmd, reason))
            if not ok:
                return False, ";".join(results)
            rospy.sleep(self.drop_step_delay)

        return True, ";".join(results)

    # ------------------------------------------------------------
    # 4.3.1 起飞前舵机 / ESP32 串口控制区
    # ------------------------------------------------------------

    def list_serial_candidates(self):
        """列出可能的 ESP32 串口设备。"""
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
        """根据参数和自动识别结果生成待尝试串口列表。"""
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
        """打开 ESP32 串口；失败时保存错误，后续收到命令会自动重试。"""
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
                        write_timeout=self.serial_timeout,
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
        """确保串口可用；不可用时自动重新打开。"""
        try:
            if self.esp32_ser is not None and self.esp32_ser.is_open:
                return True
        except Exception:
            self.esp32_ser = None

        return self.open_esp32_serial()

    def read_esp32_lines(self, wait_time=None):
        """读取 ESP32 返回的若干行文本。"""
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
        """判断 ESP32 回包是否表示本次舵机命令成功。"""
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

    def send_serial_cmd(self, cmd):
        """向 ESP32 发送 L1/R1 等舵机命令。"""
        cmd = cmd.strip().upper()

        if not self.is_single_servo_cmd(cmd):
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

                    rospy.logwarn("ESP32 serial send: %s try %d/%d", cmd, attempt, max_try)
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

    def serial_status_text(self):
        """生成舵机串口状态文本，供 PING/STATUS 回传。"""
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

    def handle_takeoff_command(self, addr):
        """处理 TAKEOFF 指令，只设置标志位，不在 UDP 线程里直接飞行。"""
        with self.command_lock:
            if self.mission_active:
                self.send_udp_message(addr, "BUSY：当前任务正在执行，忽略重复 TAKEOFF")
                return
            self.takeoff_requested = True
            self.land_requested = False

        self.send_udp_message(addr, "ACK：已收到 TAKEOFF，主线程将开始起飞流程")

    def handle_land_command(self, addr):
        """处理 LAND 指令，设置降落标志。"""
        with self.command_lock:
            self.land_requested = True

        self.send_udp_message(addr, "ACK：已收到 LAND，将刹停并降落")

    def send_udp_message(self, addr, text):
        """向地面站发送一条文本回传。"""
        if addr is None or self.udp_sock is None:
            return
        try:
            msg = ("[%s] %s" % (now_text(), text)).encode("utf-8")
            self.udp_sock.sendto(msg, addr)
        except OSError:
            pass

    def send_status(self, addr=None):
        """向地面站回传当前状态，方便排查。"""
        if addr is None:
            with self.command_lock:
                addr = self.last_ground_addr

        pose_ok = self.is_pose_recent()
        with self.command_lock:
            takeoff_req = self.takeoff_requested
            land_req = self.land_requested
            mission_active = self.mission_active
            last_cmd = self.last_command_text

        battery_text = self.get_battery_status_text()
        serial_text = self.serial_status_text()
        text = (
            "STATUS：connected=%s, armed=%s, mode=%s, pose_ok=%s, "
            "home_ready=%s, takeoff_requested=%s, land_requested=%s, "
            "mission_active=%s, completed_loops=%.2f, %s, %s, last_command=%s"
            % (
                self.state.connected,
                self.state.armed,
                self.state.mode,
                pose_ok,
                self.home_ready,
                takeoff_req,
                land_req,
                mission_active,
                self.completed_loops,
                battery_text,
                serial_text,
                last_cmd,
            )
        )
        self.send_udp_message(addr, text)

    # ------------------------------------------------------------
    # 4.4 基础飞控工具区
    # ------------------------------------------------------------

    def get_yaw_from_pose(self, pose_msg):
        """从 PoseStamped 四元数中提取 yaw 航向角。"""
        q = pose_msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def is_pose_recent(self):
        """判断本地位置是否最近仍在更新。"""
        if self.pose is None or self.last_pose_time is None:
            return False
        age = (rospy.Time.now() - self.last_pose_time).to_sec()
        return age < 1.0

    def is_battery_recent(self):
        """判断电池话题是否最近仍在更新。"""
        if self.last_battery_time is None:
            return False
        age = (rospy.Time.now() - self.last_battery_time).to_sec()
        return age < 3.0

    def get_battery_status_text(self):
        """生成电池状态文本，供地面站 STATUS 回传使用。"""
        if self.battery_avg_cell_voltage is None:
            return "battery_avg=无数据, low_threshold=%.2fV" % self.low_cell_voltage
        return "battery_avg=%.3fV, low_threshold=%.2fV" % (
            self.battery_avg_cell_voltage,
            self.low_cell_voltage
        )

    def is_battery_low_confirmed(self):
        """
        判断是否已经满足低电压自动降落条件。

        低电压保护不是一低于阈值就立刻触发，而是要求连续低于阈值
        low_battery_confirm_time 秒，避免大油门瞬间压降导致误判。
        """
        if not self.enable_battery_auto_land:
            return False

        if self.battery_avg_cell_voltage is None:
            self.low_battery_since = None
            return False

        if self.battery_avg_cell_voltage >= self.low_cell_voltage:
            self.low_battery_since = None
            return False

        now = rospy.Time.now()
        if self.low_battery_since is None:
            self.low_battery_since = now
            return False

        low_duration = (now - self.low_battery_since).to_sec()
        return low_duration >= self.low_battery_confirm_time

    def request_land_if_battery_low(self):
        """
        如果平均单节电压持续低于阈值，就设置 LAND 请求。

        返回：
        True：已经触发低电压降落，需要中断当前轨迹。
        False：电池还没有触发保护。
        """
        if not self.is_battery_low_confirmed():
            return False

        with self.command_lock:
            self.land_requested = True

        if not self.battery_land_already_reported:
            self.battery_land_already_reported = True
            msg = (
                "WARN：平均单节电压 %.3fV 低于阈值 %.2fV，"
                "已完成 %.2f 圈，开始悬停并请求 AUTO.LAND"
                % (self.battery_avg_cell_voltage, self.low_cell_voltage, self.completed_loops)
            )
            rospy.logwarn(msg)
            self.send_udp_message(self.last_ground_addr, msg)

        return True

    def set_mode(self, mode):
        """调用 MAVROS 服务切换飞行模式。"""
        try:
            resp = self.mode_srv(base_mode=0, custom_mode=mode)
            return resp.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr("切换模式失败：%s", str(e))
            return False

    def arm(self, value=True):
        """调用 MAVROS 服务解锁或上锁。"""
        try:
            resp = self.arm_srv(value)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("解锁/上锁失败：%s", str(e))
            return False

    # ------------------------------------------------------------
    # 4.5 PositionTarget 生成和发布区
    # ------------------------------------------------------------

    def make_raw_target(self, x, y, z, vx, vy, vz, ax, ay, az, yaw):
        """
        构造 PositionTarget 控制目标。

        这里不忽略 position、velocity、acceleration、yaw，
        只忽略 yaw_rate。
        """
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = PositionTarget.IGNORE_YAW_RATE

        msg.position.x = x
        msg.position.y = y
        msg.position.z = z

        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = vz

        msg.acceleration_or_force.x = ax
        msg.acceleration_or_force.y = ay
        msg.acceleration_or_force.z = az

        msg.yaw = wrap_pi(yaw)
        msg.yaw_rate = 0.0
        return msg

    def publish_raw_target(self, x, y, z, vx=0.0, vy=0.0, vz=0.0,
                           ax=0.0, ay=0.0, az=0.0, yaw=None):
        """发布 PositionTarget 控制目标。"""
        if yaw is None:
            yaw = self.home_yaw
        msg = self.make_raw_target(x, y, z, vx, vy, vz, ax, ay, az, yaw)
        self.raw_pub.publish(msg)

    def publish_home_hold_setpoint(self):
        """
        发布“地面当前位置保持 setpoint”。

        这是本版本最关键的修正：
        等待 TAKEOFF 的时候也持续发这个 setpoint。
        它不会让飞机起飞，只是告诉飞控外部控制器在线，目标是保持 home 点。
        """
        if not self.home_ready:
            return
        self.publish_raw_target(
            self.home_x, self.home_y, self.home_z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            self.home_yaw
        )

    # ------------------------------------------------------------
    # 4.6 等待连接和 home 点记录区
    # ------------------------------------------------------------

    def wait_until_ready(self):
        """等待 MAVROS 连接和本地位置，并记录 home 点。"""
        rospy.loginfo("等待 MAVROS 连接飞控...")
        while not rospy.is_shutdown() and not self.state.connected:
            self.rate.sleep()

        rospy.loginfo("MAVROS 已连接，等待本地位置...")
        while not rospy.is_shutdown() and self.pose is None:
            self.rate.sleep()

        self.home_x = self.pose.pose.position.x
        self.home_y = self.pose.pose.position.y
        self.home_z = self.pose.pose.position.z
        self.home_yaw = self.get_yaw_from_pose(self.pose)
        self.home_ready = True

        rospy.loginfo(
            "Home 点已记录：x=%.3f, y=%.3f, z=%.3f, yaw=%.1f 度",
            self.home_x,
            self.home_y,
            self.home_z,
            math.degrees(self.home_yaw)
        )

        # 记录 home 后，先连续发布 2 秒地面保持 setpoint。
        # 这样后面无论手动切 OFFBOARD 还是脚本自动切 OFFBOARD，都有 setpoint 流。
        rospy.loginfo("开始发布地面保持 setpoint，等待地面站 TAKEOFF...")
        for _ in range(int(self.rate_hz * 2.0)):
            self.publish_home_hold_setpoint()
            self.rate.sleep()

    # ------------------------------------------------------------
    # 4.7 起飞前准备区
    # ------------------------------------------------------------

    def wait_for_offboard_and_arm(self):
        """
        收到 TAKEOFF 后，等待或请求 OFFBOARD + ARM。

        如果 auto_offboard=True，脚本会自动请求 OFFBOARD。
        如果 auto_arm=True，脚本会自动请求解锁。
        如果你们已经手动切好 OFFBOARD 和 ARM，函数会直接通过。
        等待期间会持续发布 home 保持 setpoint。
        """
        start_time = rospy.Time.now()
        last_mode_request = rospy.Time.now()
        last_arm_request = rospy.Time.now()

        self.send_status()
        self.send_udp_message(self.last_ground_addr, "STATUS：准备进入 OFFBOARD/ARM 阶段，持续发布保持 setpoint")

        while not rospy.is_shutdown():
            if self.is_land_requested():
                rospy.logwarn("等待 OFFBOARD/ARM 期间收到 LAND，取消起飞。")
                return False

            if self.state.mode == "OFFBOARD" and self.state.armed:
                rospy.loginfo("已经满足 OFFBOARD + ARMED，准备起飞。")
                self.send_udp_message(self.last_ground_addr, "STATUS：OFFBOARD + ARMED 已满足，开始起飞")
                return True

            now = rospy.Time.now()

            if self.auto_offboard and self.state.mode != "OFFBOARD":
                if (now - last_mode_request).to_sec() > 1.0:
                    ok = self.set_mode("OFFBOARD")
                    rospy.loginfo("请求 OFFBOARD：mode_sent=%s，当前 mode=%s", ok, self.state.mode)
                    self.send_udp_message(self.last_ground_addr, "STATUS：请求 OFFBOARD，mode_sent=%s，当前 mode=%s" % (ok, self.state.mode))
                    last_mode_request = now

            if self.auto_arm and not self.state.armed:
                if (now - last_arm_request).to_sec() > 1.0:
                    ok = self.arm(True)
                    rospy.loginfo("请求 ARM：success=%s，当前 armed=%s", ok, self.state.armed)
                    self.send_udp_message(self.last_ground_addr, "STATUS：请求 ARM，success=%s，当前 armed=%s" % (ok, self.state.armed))
                    last_arm_request = now

            elapsed = (now - start_time).to_sec()
            if elapsed > self.offboard_arm_timeout:
                rospy.logerr("等待 OFFBOARD/ARM 超时。")
                self.send_udp_message(
                    self.last_ground_addr,
                    "ERROR：等待 OFFBOARD/ARM 超时，当前 mode=%s, armed=%s" % (self.state.mode, self.state.armed)
                )
                return False

            # 等待期间继续发地面保持 setpoint。
            self.publish_home_hold_setpoint()
            self.rate.sleep()

    def check_circle_feasible(self):
        """检查绕圆速度和半径是否过于激进。"""
        if self.radius <= 0.05:
            rospy.logerr("绕圆半径过小，拒绝起飞。")
            return False
        if self.circle_speed <= 0.05:
            rospy.logerr("绕圆速度过小，拒绝起飞。")
            return False

        a_req = self.circle_speed * self.circle_speed / self.radius
        tilt_deg = math.degrees(math.atan2(a_req, 9.81))

        rospy.loginfo(
            "绕圆参数检查：r=%.2f m, v=%.2f m/s, a=%.2f m/s^2, 估计倾角=%.1f 度",
            self.radius,
            self.circle_speed,
            a_req,
            tilt_deg
        )

        if a_req > self.max_centripetal_acc:
            msg = "ERROR：绕圆参数过激，向心加速度 %.2f 超过限制 %.2f" % (
                a_req,
                self.max_centripetal_acc
            )
            rospy.logerr(msg)
            self.send_udp_message(self.last_ground_addr, msg)
            if self.enforce_acc_limit:
                return False
        return True

    # ------------------------------------------------------------
    # 4.8 平滑轨迹工具区
    # ------------------------------------------------------------

    def smoother(self, u):
        """五次多项式平滑函数，保证起点和终点速度、加速度较平滑。"""
        u = clamp(u, 0.0, 1.0)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        ds = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        dds = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
        return s, ds, dds

    def move_to(self, x, y, z, yaw=None, speed=None, hold_time=0.5):
        """
        平滑移动到指定位置。

        返回：
        True：正常到达
        False：中途收到 LAND 或 ROS 退出
        """
        if yaw is None:
            yaw = self.home_yaw
        if speed is None:
            speed = self.transfer_speed

        sx = self.pose.pose.position.x
        sy = self.pose.pose.position.y
        sz = self.pose.pose.position.z

        dx = x - sx
        dy = y - sy
        dz = z - sz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        duration = max(dist / max(speed, 0.05), 1.5)
        steps = max(int(duration * self.rate_hz), 1)

        rospy.loginfo("平滑移动到 x=%.2f, y=%.2f, z=%.2f，预计 %.2f 秒", x, y, z, duration)

        for i in range(steps + 1):
            if rospy.is_shutdown() or self.is_land_requested() or self.request_land_if_battery_low():
                return False

            u = float(i) / float(steps)
            s, ds, dds = self.smoother(u)

            px = sx + dx * s
            py = sy + dy * s
            pz = sz + dz * s

            vx = dx * ds / duration
            vy = dy * ds / duration
            vz = dz * ds / duration

            ax = dx * dds / (duration * duration)
            ay = dy * dds / (duration * duration)
            az = dz * dds / (duration * duration)

            self.publish_raw_target(px, py, pz, vx, vy, vz, ax, ay, az, yaw)
            self.rate.sleep()

        # 轨迹发布时间走完，只代表“目标点已经发布完”，不代表飞机真实到达。
        # 这里必须继续读取 /mavros/local_position/pose，确认真实位置进入容差并稳定一段时间。
        reach_timeout = max(duration + self.reach_timeout_margin, duration * 1.5)
        if not self.wait_until_reached(x, y, z, yaw, timeout=reach_timeout):
            return False

        return self.hold_position(x, y, z, yaw, hold_time)

    def hold_position(self, x, y, z, yaw=None, hold_time=1.0):
        """在指定位置悬停一段时间。"""
        if yaw is None:
            yaw = self.home_yaw

        steps = max(int(hold_time * self.rate_hz), 1)
        for _ in range(steps):
            if rospy.is_shutdown() or self.is_land_requested() or self.request_land_if_battery_low():
                return False
            self.publish_raw_target(x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw)
            self.rate.sleep()
        return True

    def current_rel_alt(self):
        """返回相对 home 点的当前高度，单位 m。"""
        if self.pose is None:
            return None
        return self.pose.pose.position.z - self.home_z

    def wait_until_reached(self, x, y, z, yaw=None, timeout=10.0):
        """
        等待无人机真实位置到达目标点。

        这个函数解决核心安全问题：
        move_to() 前半段只是发布了一条“应该怎么飞”的轨迹，
        但飞控、电机、定位、模式切换都可能让实际飞机滞后。
        所以进入下一阶段前，必须检查实际 pose 是否真的接近目标。
        """
        if yaw is None:
            yaw = self.home_yaw

        start_time = rospy.Time.now()
        reached_since = None
        last_report_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.is_land_requested() or self.request_land_if_battery_low():
                return False

            now = rospy.Time.now()
            elapsed = (now - start_time).to_sec()

            if not self.is_pose_recent():
                reached_since = None
                self.publish_raw_target(x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw)
                self.rate.sleep()
                continue

            px = self.pose.pose.position.x
            py = self.pose.pose.position.y
            pz = self.pose.pose.position.z

            ex = x - px
            ey = y - py
            ez = z - pz
            xy_err = math.sqrt(ex * ex + ey * ey)
            z_err = abs(ez)

            if xy_err <= self.xy_reach_tol and z_err <= self.z_reach_tol:
                if reached_since is None:
                    reached_since = now
                if (now - reached_since).to_sec() >= self.reach_stable_time:
                    rospy.loginfo(
                        "实际到达目标：xy_err=%.3f m, z_err=%.3f m, rel_alt=%.3f m",
                        xy_err, z_err, pz - self.home_z
                    )
                    return True
            else:
                reached_since = None

            if (now - last_report_time).to_sec() > 1.0:
                rospy.loginfo(
                    "等待实际到达：xy_err=%.3f m, z_err=%.3f m, rel_alt=%.3f m, timeout_left=%.1f s",
                    xy_err, z_err, pz - self.home_z, max(timeout - elapsed, 0.0)
                )
                last_report_time = now

            if elapsed > timeout:
                msg = (
                    "ERROR：实际位置未到达目标，禁止进入下一阶段；"
                    "xy_err=%.3f m, z_err=%.3f m, 当前相对高度=%.3f m"
                    % (xy_err, z_err, pz - self.home_z)
                )
                rospy.logerr(msg)
                self.send_udp_message(self.last_ground_addr, msg)
                with self.command_lock:
                    self.land_requested = True
                return False

            # 等待真实到达期间继续发布目标点，但速度/加速度前馈清零，避免继续加速。
            self.publish_raw_target(x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw)
            self.rate.sleep()

        return False

    # ------------------------------------------------------------
    # 4.9 圆形绕飞轨迹区
    # ------------------------------------------------------------

    def get_circle_center_abs(self):
        """计算圆心在 local 坐标系下的绝对坐标。"""
        if self.relative_frame == "body":
            c = math.cos(self.home_yaw)
            s = math.sin(self.home_yaw)
            dx_local = c * self.center_x - s * self.center_y
            dy_local = s * self.center_x + c * self.center_y
        else:
            dx_local = self.center_x
            dy_local = self.center_y
        return self.home_x + dx_local, self.home_y + dy_local

    def circle_speed_profile(self, t, ramp_t, cruise_t, omega):
        """生成圆周运动的角度、角速度、角加速度。"""
        if ramp_t <= 1e-6:
            return omega * t, omega, 0.0

        ramp_angle = 0.5 * omega * ramp_t

        if t < ramp_t:
            u = t / ramp_t
            s, ds, _ = self.smoother(u)
            integral_s = 2.5 * u**4 - 3.0 * u**5 + u**6
            theta_abs = omega * ramp_t * integral_s
            omega_abs = omega * s
            alpha_abs = omega * ds / ramp_t
            return theta_abs, omega_abs, alpha_abs

        if t < ramp_t + cruise_t:
            tc = t - ramp_t
            theta_abs = ramp_angle + omega * tc
            return theta_abs, omega, 0.0

        td = t - ramp_t - cruise_t
        u = clamp(td / ramp_t, 0.0, 1.0)
        s, ds, _ = self.smoother(u)
        integral_s = 2.5 * u**4 - 3.0 * u**5 + u**6
        theta_abs = ramp_angle + omega * cruise_t + omega * ramp_t * (u - integral_s)
        omega_abs = omega * (1.0 - s)
        alpha_abs = -omega * ds / ramp_t
        return theta_abs, omega_abs, alpha_abs

    def fly_circle(self):
        """
        执行圆形绕飞。

        本版本默认用于“极限圈数测试”：
        1. 起飞后先飞到圆起点。
        2. 按最开始设定的 radius、circle_speed、center_x、center_y 持续绕圈。
        3. 不再因为 loops 到达固定圈数就停止。
        4. 当平均单节电压持续低于 low_cell_voltage 时，返回 False，主流程会进入刹停和降落。

        如果你想恢复旧的“固定圈数”模式，可以运行时加：
        _circle_until_low_battery:=False
        """
        target_z = self.home_z + self.takeoff_alt

        # 二次保护：即使前面的起飞流程出错，也不允许低高度进入水平转场/绕圆。
        rel_alt = self.current_rel_alt()
        min_circle_alt = self.takeoff_alt * self.min_circle_alt_ratio
        if rel_alt is None or rel_alt < min_circle_alt:
            msg = (
                "ERROR：当前相对高度 %.3f m 未达到绕圆最低高度 %.3f m，禁止进入绕圆"
                % ((rel_alt if rel_alt is not None else -999.0), min_circle_alt)
            )
            rospy.logerr(msg)
            self.send_udp_message(self.last_ground_addr, msg)
            with self.command_lock:
                self.land_requested = True
            return False

        cx, cy = self.get_circle_center_abs()
        direction = -1.0 if self.clockwise else 1.0
        start_theta = 0.0

        start_x = cx + self.radius * math.cos(start_theta)
        start_y = cy + self.radius * math.sin(start_theta)

        rospy.loginfo("准备飞到圆起点：x=%.2f, y=%.2f, z=%.2f", start_x, start_y, target_z)
        if not self.move_to(start_x, start_y, target_z, self.home_yaw, self.transfer_speed, 1.0):
            return False

        if self.circle_until_low_battery:
            return self.fly_circle_until_low_battery(cx, cy, target_z, direction, start_theta)

        return self.fly_circle_fixed_loops(cx, cy, target_z, direction, start_theta)

    def fly_circle_until_low_battery(self, cx, cy, target_z, direction, start_theta):
        """
        持续绕圈，直到收到 LAND 或低电压保护触发。

        这个函数不会按 DEFAULT_LOOPS 自动结束，适合你们测试极限圈数。
        起步时仍然使用 ramp_time 平滑加速，之后一直按 circle_speed 匀速绕圈。
        """
        omega = self.circle_speed / self.radius
        ramp_t = max(self.ramp_time, 0.0)
        start_time = rospy.Time.now()
        last_report_loop = -1
        self.completed_loops = 0.0

        rospy.loginfo(
            "开始极限绕圈：半径=%.2f m，速度=%.2f m/s，低电压阈值=%.2f V/节",
            self.radius,
            self.circle_speed,
            self.low_cell_voltage
        )
        self.send_udp_message(
            self.last_ground_addr,
            "STATUS：开始极限绕圈，低于 %.2fV/节 后自动悬停并 AUTO.LAND" % self.low_cell_voltage
        )

        while not rospy.is_shutdown():
            if self.is_land_requested() or self.request_land_if_battery_low():
                return False

            elapsed = (rospy.Time.now() - start_time).to_sec()

            # ------------------------------
            # 角度规划
            # ------------------------------
            # ramp_t 内从 0 平滑加速到 omega；之后一直按 omega 匀速绕圈。
            if ramp_t > 1e-6 and elapsed < ramp_t:
                u = elapsed / ramp_t
                s, ds, _ = self.smoother(u)
                integral_s = 2.5 * u**4 - 3.0 * u**5 + u**6
                theta_abs = omega * ramp_t * integral_s
                omega_abs = omega * s
                alpha_abs = omega * ds / ramp_t
            else:
                ramp_angle = 0.5 * omega * ramp_t
                cruise_t = max(elapsed - ramp_t, 0.0)
                theta_abs = ramp_angle + omega * cruise_t
                omega_abs = omega
                alpha_abs = 0.0

            self.completed_loops = theta_abs / (2.0 * math.pi)

            # 每完成 1 圈，向终端和地面站汇报一次，方便统计极限圈数。
            current_loop_int = int(self.completed_loops)
            if current_loop_int > last_report_loop:
                last_report_loop = current_loop_int
                msg = "STATUS：已绕 %.2f 圈，%s" % (
                    self.completed_loops,
                    self.get_battery_status_text()
                )
                rospy.loginfo(msg)
                self.send_udp_message(self.last_ground_addr, msg)

            theta = start_theta + direction * theta_abs
            theta_dot = direction * omega_abs
            theta_ddot = direction * alpha_abs

            # ------------------------------
            # 圆轨迹位置、速度、加速度
            # ------------------------------
            x = cx + self.radius * math.cos(theta)
            y = cy + self.radius * math.sin(theta)
            z = target_z

            vx = -self.radius * math.sin(theta) * theta_dot
            vy = self.radius * math.cos(theta) * theta_dot
            vz = 0.0

            ax = -self.radius * math.cos(theta) * theta_dot * theta_dot - self.radius * math.sin(theta) * theta_ddot
            ay = -self.radius * math.sin(theta) * theta_dot * theta_dot + self.radius * math.cos(theta) * theta_ddot
            az = 0.0

            if self.yaw_mode == "tangent":
                yaw = theta + direction * math.pi / 2.0
            else:
                yaw = self.home_yaw

            self.publish_raw_target(x, y, z, vx, vy, vz, ax, ay, az, yaw)
            self.rate.sleep()

        return False

    def fly_circle_fixed_loops(self, cx, cy, target_z, direction, start_theta):
        """
        旧的固定圈数绕飞模式。

        只有当 circle_until_low_battery=False 时才会使用。
        """
        angle_total_abs = 2.0 * math.pi * abs(self.loops)
        omega = self.circle_speed / self.radius
        ramp_t = min(self.ramp_time, angle_total_abs / max(omega, 1e-6))
        cruise_angle = max(angle_total_abs - omega * ramp_t, 0.0)
        cruise_t = cruise_angle / max(omega, 1e-6)
        total_time = 2.0 * ramp_t + cruise_t

        rospy.loginfo("开始固定圈数绕圆：半径=%.2f, 圈数=%.2f, 速度=%.2f, 总时间=%.2f 秒",
                      self.radius, self.loops, self.circle_speed, total_time)
        self.send_udp_message(self.last_ground_addr, "STATUS：开始固定圈数绕圆，预计 %.1f 秒" % total_time)

        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.is_land_requested() or self.request_land_if_battery_low():
                return False

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= total_time:
                break

            theta_abs, omega_abs, alpha_abs = self.circle_speed_profile(elapsed, ramp_t, cruise_t, omega)
            self.completed_loops = theta_abs / (2.0 * math.pi)

            theta = start_theta + direction * theta_abs
            theta_dot = direction * omega_abs
            theta_ddot = direction * alpha_abs

            x = cx + self.radius * math.cos(theta)
            y = cy + self.radius * math.sin(theta)
            z = target_z

            vx = -self.radius * math.sin(theta) * theta_dot
            vy = self.radius * math.cos(theta) * theta_dot
            vz = 0.0

            ax = -self.radius * math.cos(theta) * theta_dot * theta_dot - self.radius * math.sin(theta) * theta_ddot
            ay = -self.radius * math.sin(theta) * theta_dot * theta_dot + self.radius * math.cos(theta) * theta_ddot
            az = 0.0

            if self.yaw_mode == "tangent":
                yaw = theta + direction * math.pi / 2.0
            else:
                yaw = self.home_yaw

            self.publish_raw_target(x, y, z, vx, vy, vz, ax, ay, az, yaw)
            self.rate.sleep()

        end_theta = start_theta + direction * angle_total_abs
        end_x = cx + self.radius * math.cos(end_theta)
        end_y = cy + self.radius * math.sin(end_theta)
        rospy.loginfo("固定圈数绕圆结束。")
        return self.hold_position(end_x, end_y, target_z, self.home_yaw, 1.0)

    # ------------------------------------------------------------
    # 4.10 LAND 和降落区
    # ------------------------------------------------------------

    def is_land_requested(self):
        """线程安全地读取 LAND 请求状态。"""
        with self.command_lock:
            return self.land_requested

    def brake_to_zero_then_land(self):
        """收到 LAND 后，先刹停，再自动降落上锁。"""
        rospy.logwarn("执行 LAND：先当前位置刹停，再降落。")
        self.send_udp_message(self.last_ground_addr, "STATUS：开始刹停，速度前馈和加速度前馈置零")

        if self.pose is None:
            rospy.logerr("没有当前位姿，无法安全刹停。")
            return

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        z = self.pose.pose.position.z
        yaw = self.get_yaw_from_pose(self.pose)

        self.hold_position_ignore_land(x, y, z, yaw, self.brake_hold_time)
        self.land_sequence(x, y, z, yaw)

    def hold_position_ignore_land(self, x, y, z, yaw, hold_time):
        """LAND 流程内部使用的悬停函数，不再被 LAND 标志打断。"""
        steps = max(int(hold_time * self.rate_hz), 1)
        for _ in range(steps):
            if rospy.is_shutdown():
                return
            self.publish_raw_target(x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw)
            self.rate.sleep()

    def land_sequence(self, x=None, y=None, z=None, yaw=None):
        """优先 AUTO.LAND，失败后使用 setpoint 兜底下降。"""
        if self.pose is None:
            return
        if x is None:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            z = self.pose.pose.position.z
            yaw = self.get_yaw_from_pose(self.pose)

        if self.use_auto_land:
            rospy.loginfo("请求 AUTO.LAND...")
            self.send_udp_message(self.last_ground_addr, "STATUS：请求 AUTO.LAND")
            ok = self.set_mode("AUTO.LAND")
            if ok:
                start_time = rospy.Time.now()
                while not rospy.is_shutdown():
                    if not self.state.armed:
                        rospy.loginfo("AUTO.LAND 完成，飞控已上锁。")
                        self.send_udp_message(self.last_ground_addr, "STATUS：AUTO.LAND 完成，已上锁")
                        return
                    rel_z = self.pose.pose.position.z - self.home_z
                    if rel_z < 0.15:
                        break
                    if (rospy.Time.now() - start_time).to_sec() > 20.0:
                        rospy.logwarn("AUTO.LAND 超时，进入兜底下降。")
                        break
                    self.rate.sleep()
            else:
                rospy.logwarn("AUTO.LAND 请求失败，进入兜底下降。")

        self.fallback_descend_and_disarm(x, y, z, yaw)

    def fallback_descend_and_disarm(self, x, y, current_z, yaw):
        """如果 AUTO.LAND 不成功，用 OFFBOARD setpoint 慢慢下降并上锁。"""
        rospy.logwarn("开始 OFFBOARD 兜底下降。")
        self.send_udp_message(self.last_ground_addr, "STATUS：开始 OFFBOARD 兜底下降")

        final_z = self.home_z + 0.08
        dz = final_z - current_z
        duration = max(abs(dz) / max(self.descend_speed, 0.05), 1.0)
        steps = max(int(duration * self.rate_hz), 1)

        for i in range(steps + 1):
            if rospy.is_shutdown():
                return
            u = float(i) / float(steps)
            z = current_z + dz * u
            self.publish_raw_target(x, y, z, 0.0, 0.0, dz / duration, 0.0, 0.0, 0.0, yaw)
            self.rate.sleep()

        if self.auto_disarm and self.state.armed:
            rospy.loginfo("兜底下降完成，执行上锁。")
            self.arm(False)
            self.send_udp_message(self.last_ground_addr, "STATUS：兜底下降完成，已请求上锁")

    # ------------------------------------------------------------
    # 4.11 主任务流程区
    # ------------------------------------------------------------

    def execute_takeoff_mission(self):
        """收到 TAKEOFF 后执行完整任务。"""
        with self.command_lock:
            self.mission_active = True

        try:
            # 起飞前电池检查：
            # - 如果要求必须有电池数据，但还没收到 /mavros/battery，则拒绝起飞。
            # - 如果起飞前平均单节电压已经低于阈值，也拒绝起飞。
            if self.require_battery_data_for_takeoff and self.battery_avg_cell_voltage is None:
                self.send_udp_message(self.last_ground_addr, "ERROR：未收到 /mavros/battery 电池数据，拒绝起飞")
                rospy.logerr("未收到 /mavros/battery 电池数据，拒绝起飞。")
                return

            if self.battery_avg_cell_voltage is not None and self.battery_avg_cell_voltage < self.low_cell_voltage:
                self.send_udp_message(
                    self.last_ground_addr,
                    "ERROR：起飞前平均单节电压 %.3fV 已低于 %.2fV，拒绝起飞"
                    % (self.battery_avg_cell_voltage, self.low_cell_voltage)
                )
                rospy.logerr("起飞前电池电压过低，拒绝起飞。")
                return

            if not self.check_circle_feasible():
                return

            if not self.wait_for_offboard_and_arm():
                return

            target_z = self.home_z + self.takeoff_alt
            rospy.loginfo("开始起飞到 %.2f m", target_z)
            self.send_udp_message(self.last_ground_addr, "STATUS：开始起飞到 %.2f m" % target_z)

            if not self.move_to(self.home_x, self.home_y, target_z, self.home_yaw,
                                self.takeoff_speed, self.hold_after_takeoff_time):
                return

            self.send_udp_message(self.last_ground_addr, "STATUS：起飞完成")

            if self.start_circle_after_takeoff:
                if not self.fly_circle():
                    return
                self.send_udp_message(self.last_ground_addr, "STATUS：绕圆任务完成")
            else:
                rospy.loginfo("配置为只起飞悬停，不自动绕圆。")
                self.send_udp_message(self.last_ground_addr, "STATUS：当前配置为只起飞悬停，等待 LAND")

            if self.land_after_mission:
                self.land_sequence()
            else:
                rospy.loginfo("任务完成后保持悬停，等待地面站 LAND。")
                while not rospy.is_shutdown() and not self.is_land_requested():
                    if self.pose is not None:
                        x = self.pose.pose.position.x
                        y = self.pose.pose.position.y
                        z = self.pose.pose.position.z
                        yaw = self.get_yaw_from_pose(self.pose)
                        self.publish_raw_target(x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw)
                    self.rate.sleep()

        finally:
            if self.is_land_requested():
                self.brake_to_zero_then_land()

            with self.command_lock:
                self.takeoff_requested = False
                self.land_requested = False
                self.mission_active = False

            self.send_udp_message(self.last_ground_addr, "STATUS：任务流程结束，回到等待 TAKEOFF 状态")

    def run(self):
        """程序主循环。"""
        self.start_udp_server()
        self.wait_until_ready()

        rospy.loginfo("进入等待 TAKEOFF 主循环。等待期间会持续发布 home 保持 setpoint。")
        self.send_udp_message(self.last_ground_addr, "STATUS：机载脚本已就绪，等待 TAKEOFF")

        while not rospy.is_shutdown():
            with self.command_lock:
                should_takeoff = self.takeoff_requested and not self.mission_active
                should_land = self.land_requested and not self.mission_active

            if should_takeoff:
                self.execute_takeoff_mission()
                continue

            if should_land:
                self.brake_to_zero_then_land()
                with self.command_lock:
                    self.land_requested = False
                continue

            # 等待 TAKEOFF 期间持续发布 home 保持 setpoint。
            self.publish_home_hold_setpoint()
            self.rate.sleep()


# ============================================================
# 五、程序入口区
# ============================================================

if __name__ == "__main__":
    try:
        node = CircleObstacleGroundStationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
