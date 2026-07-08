#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
微型无人机第一阶段 FSM：安全控制增强版
功能：
1. 从 YAML 读取任务航点；
2. 按 FSM 执行起飞、二维码点、绕障点、图片靶点、特殊靶点、圆环前等待点；
3. route/move 点：远距离只发速度 setpoint，接近目标后切位置锁点，避免最终位置强拉导致冲过头；
4. scan/drop 点：先用速度控制慢速靠近，进入捕获半径后位置锁点，停稳后再扫描/投放；
5. 二维码扫描仍然静态悬停；图片靶/特殊靶 ACTION 阶段支持视觉米制偏差闭环对准后投放；
6. 圆环阶段支持 ring_gate 视觉结果：搜索锁定圆环中心、动态生成 RING_PRE/RING_CENTER/RING_POST，并在 RING_PRE 进行左右/高度二次对准；
7. 静态识别 ACTION 阶段持续检查位置和速度；加速度只作为可选辅助门槛，避免实机噪声卡死；
8. 增加 /uav/start、/uav/stop、/uav/land、/uav/disarm、/uav/reset 安全控制话题；
9. /uav/stop 定义为“急停降落”：立即取消任务，切 AUTO.LAND，落地后自动 disarm，之后必须 reset 才能再次 start。
"""

import os
import math
import json
import copy
from dataclasses import dataclass

import yaml
import rospy

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from tf.transformations import euler_from_quaternion


# =========================
# 01. 数据结构定义区
# =========================

@dataclass
class Waypoint:
    """
    航点结构体。
    name: 航点名称。
    kind: 航点类型，route 或 scan。
    action: 到达并停稳后执行的动作。
    x, y, z: 相对起飞点坐标，单位 m。
    yaw_deg: 目标航向角，单位 deg。
    speed: 当前航点移动阶段的期望速度，单位 m/s；本版在 MOVE 阶段作为纯速度 setpoint 的上限使用。
    acc: 当前航点最大加速度，单位 m/s^2，用于限制速度指令变化，避免突然加减速。
    hold_time: scan 点停稳后的保持时间，单位 s。
    control_mode: 控制模式。auto 根据 kind 自动选择；position 为纯位置控制；fusion 为位置速度融合并停稳；fusion_route 为位置速度融合快速通过。
    dynamic_land: 是否使用二维码中的 left/right 动态替换该航点 y 坐标。
    """

    name: str
    kind: str
    action: str
    x: float
    y: float
    z: float
    yaw_deg: float
    speed: float
    acc: float
    hold_time: float = 0.5
    control_mode: str = "auto"
    dynamic_land: bool = False

    def yaw_rad(self):
        """将角度制 yaw 转成弧度制 yaw。"""
        return math.radians(self.yaw_deg)


# =========================
# 02. 通用数学函数区
# =========================

def norm3(vx, vy, vz):
    """计算三维向量模长。"""
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def clamp(value, min_value, max_value):
    """将 value 限制在 [min_value, max_value] 范围内。"""
    return max(min_value, min(max_value, value))


def wrap_pi(angle):
    """将角度限制到 [-pi, pi]。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def limit_vector_norm(vx, vy, vz, max_norm):
    """限制速度向量的最大模长。"""
    length = norm3(vx, vy, vz)

    if length < 1e-6:
        return 0.0, 0.0, 0.0

    if length <= max_norm:
        return vx, vy, vz

    scale = max_norm / length
    return vx * scale, vy * scale, vz * scale


def limit_vector_change(old_v, new_v, max_delta):
    """限制速度指令变化量，用于实现加速度约束。"""
    dx = new_v[0] - old_v[0]
    dy = new_v[1] - old_v[1]
    dz = new_v[2] - old_v[2]

    delta_len = norm3(dx, dy, dz)

    if delta_len < 1e-6:
        return new_v

    if delta_len <= max_delta:
        return new_v

    scale = max_delta / delta_len

    return [
        old_v[0] + dx * scale,
        old_v[1] + dy * scale,
        old_v[2] + dz * scale,
    ]


# =========================
# 03. FSM 主类
# =========================

class MicroUAVStage1FSM:
    def __init__(self):
        """初始化节点、读取 YAML、建立 ROS 通信。"""

        rospy.init_node("micro_uav_stage1_fsm")

        # ---------- YAML 路径 ----------
        self.mission_yaml = rospy.get_param(
            "~mission_yaml",
            os.path.expanduser("~/catkin_ws/src/uav_inventory/config/stage1_mission.yaml")
        )

        self.mission_cfg = self.load_yaml(self.mission_yaml)

        # ---------- 任务参数 ----------
        self.frame_mode = self.mission_cfg["mission"].get("frame", "relative_home")
        self.wait_real_qr = self.mission_cfg["mission"].get("wait_real_qr", False)
        self.wait_real_image = self.mission_cfg["mission"].get("wait_real_image", False)
        # 圆环视觉可以单独开关；默认跟随 wait_real_image，调航线时可设为 false。
        self.wait_real_ring = self.mission_cfg["mission"].get("wait_real_ring", self.wait_real_image)

        self.control_cfg = self.mission_cfg["mission"].get("control", {})
        self.action_cfg = self.mission_cfg["mission"].get("action", {})
        self.takeoff_cfg = self.mission_cfg["mission"].get("takeoff", {})
        self.landing_cfg = self.mission_cfg["mission"].get("landing", {})
        self.obstacle_cfg = self.mission_cfg["mission"].get("obstacle", {})
        self.qr_default_cfg = self.mission_cfg["mission"].get("qr_default", {})
        self.qr_default_enabled = bool(self.qr_default_cfg.get("enabled", True))
        self.qr_default_class_1 = str(self.qr_default_cfg.get("class_1", "")).strip().lower()
        self.qr_default_class_2 = str(self.qr_default_cfg.get("class_2", "")).strip().lower()
        self.qr_default_land_side = str(
            self.qr_default_cfg.get(
                "landing_side",
                self.landing_cfg.get("default_side", "right")
            )
        ).strip().lower()

        self.waypoints = self.parse_waypoints(self.mission_cfg)

        # ---------- 控制频率 ----------
        self.rate_hz = float(self.control_cfg.get("rate_hz", 30.0))

        # ---------- 圆形绕障参数 ----------
        # 障碍物中心坐标使用任务 YAML 中的相对起飞点坐标，单位 m。
        # 比赛地图中起飞点为 (1500,3000) mm，障碍物中心为 (5800,3000) mm，
        # 在当前代码坐标系下换算为 (4.30, 0.00) m。
        self.obstacle_mode = str(self.obstacle_cfg.get("mode", "waypoint_loop")).strip().lower()
        self.circle_center_x = float(self.obstacle_cfg.get("center_x", 4.30))
        self.circle_center_y = float(self.obstacle_cfg.get("center_y", 0.00))
        self.circle_radius = float(self.obstacle_cfg.get("radius", 0.90))
        self.circle_height = float(self.obstacle_cfg.get("height", 1.35))
        self.circle_speed = float(self.obstacle_cfg.get("circle_speed", 0.75))
        self.circle_ramp_time = float(self.obstacle_cfg.get("ramp_time", 2.00))
        self.circle_direction = str(self.obstacle_cfg.get("direction", "clockwise")).strip().lower()
        self.circle_start_angle_deg = float(self.obstacle_cfg.get("start_angle_deg", 90.0))
        self.circle_yaw_mode = str(self.obstacle_cfg.get("yaw_mode", "fixed")).strip().lower()
        self.circle_finish_pos_eps = float(self.obstacle_cfg.get("finish_pos_eps", 0.18))
        self.circle_finish_vel_th = float(self.obstacle_cfg.get("finish_vel_th", 0.25))
        self.circle_finish_stable_time = float(self.obstacle_cfg.get("finish_stable_time", 0.20))
        self.circle_finish_max_wait = float(self.obstacle_cfg.get("finish_max_wait", 1.50))
        # 圆形绕障结束后软退出条件：只作为噪声兜底，不能只靠超时直接进入下一个航点。
        self.circle_finish_soft_pos_eps = float(self.obstacle_cfg.get("finish_soft_pos_eps", 0.30))
        self.circle_finish_soft_vel_th = float(self.obstacle_cfg.get("finish_soft_vel_th", 0.35))

        # ---------- 判稳防卡死参数 ----------
        # 速度差分得到的加速度在实机上很容易被定位/速度噪声放大。
        # 因此默认只把加速度作为滤波后的辅助信息，不再让所有航点都被 acc 硬卡。
        self.acc_lpf_alpha = float(self.control_cfg.get("acc_lpf_alpha", 0.25))
        self.yaw_align_use_acc_gate = bool(self.control_cfg.get("yaw_align_use_acc_gate", False))
        self.yaw_align_soft_wait = float(self.control_cfg.get("yaw_align_soft_wait", 0.80))
        self.yaw_align_soft_vel = float(self.control_cfg.get("yaw_align_soft_vel", 0.25))
        self.brake_timeout = float(self.control_cfg.get("brake_timeout", 1.20))
        self.brake_soft_vel = float(self.control_cfg.get("brake_soft_vel", 0.25))
        self.action_use_acc_gate = bool(self.control_cfg.get("action_use_acc_gate", False))

        # ---------- yaw 控制参数 ----------
        self.yaw_eps = math.radians(float(self.control_cfg.get("yaw_eps_deg", 5.0)))
        # 快速通过点跳过 YAW_ALIGN 使用更严格的角度阈值。
        # 普通 YAW_ALIGN 仍可用 yaw_eps_deg=5°，但跳过原地对齐必须更接近目标 yaw，
        # 避免真机在机头仍有明显偏差时直接进入快速通过。
        self.skip_yaw_align_eps = math.radians(
            float(self.control_cfg.get("skip_yaw_align_eps_deg", 3.0))
        )
        self.yaw_break_eps = math.radians(float(self.control_cfg.get("yaw_break_eps_deg", 12.0)))

        # ---------- route 点判断参数 ----------
        self.route_pos_eps = float(self.control_cfg.get("route_pos_eps", 0.18))
        self.route_finish_vel = float(self.control_cfg.get("route_finish_vel", 0.18))
        self.route_finish_acc = float(self.control_cfg.get("route_finish_acc", 0.50))
        self.yaw_align_soft_vel = max(self.yaw_align_soft_vel, self.route_finish_vel)

        # ---------- 距离减速参数 ----------
        # 2026-05-23 修改：所有航点都采用“远处正常飞、近处线性降速、到点刹停”的折中方案。
        self.route_slow_radius = float(self.control_cfg.get("route_slow_radius", 1.00))
        self.obs_slow_radius = float(self.control_cfg.get("obs_slow_radius", 0.80))
        # 绕障中间点连续通过时使用更小的减速半径，避免每个折线点附近都明显降速。
        # YAML 中也可通过 obs_fast_slow_radius 覆盖；默认比普通绕障点更快，但仍保留一定减速。
        self.obs_fast_slow_radius = float(self.control_cfg.get("obs_fast_slow_radius", 0.45))
        self.scan_slow_radius = float(self.control_cfg.get("scan_slow_radius", 0.60))
        self.ring_slow_radius = float(self.control_cfg.get("ring_slow_radius", 0.80))
        self.land_slow_radius = float(self.control_cfg.get("land_slow_radius", 1.20))

        self.route_min_speed = float(self.control_cfg.get("route_min_speed", 0.12))
        self.scan_min_speed = float(self.control_cfg.get("scan_min_speed", 0.06))
        self.ring_min_speed = float(self.control_cfg.get("ring_min_speed", 0.08))

        # ---------- 速度控制转位置锁点的捕获半径 ----------
        # 2026-05-25 修改：MOVE 阶段不再发布“最终位置 + 速度前馈”。
        # 当距离目标点小于 capture_radius 后，切入 BRAKE，用位置锁点 + 0 速度等待真实停稳。
        self.route_capture_radius = float(
            self.control_cfg.get("route_capture_radius", max(0.35, self.route_pos_eps * 1.8))
        )
        self.scan_capture_radius = float(
            self.control_cfg.get("scan_capture_radius", max(0.25, float(self.control_cfg.get("scan_pos_eps", 0.08)) * 3.0))
        )
        self.ring_capture_radius = float(
            self.control_cfg.get("ring_capture_radius", max(0.18, self.route_pos_eps))
        )

        # ---------- scan 点判断参数 ----------
        self.scan_pos_eps = float(self.control_cfg.get("scan_pos_eps", 0.08))
        self.scan_kp = float(self.control_cfg.get("scan_kp", 0.70))
        self.scan_stable_vel = float(self.control_cfg.get("scan_stable_vel", 0.08))
        self.scan_stable_acc = float(self.control_cfg.get("scan_stable_acc", 0.25))
        self.scan_stable_time = float(self.control_cfg.get("scan_stable_time", 0.80))

        # ---------- 起飞参数 ----------
        self.takeoff_height = float(self.takeoff_cfg.get("z", 1.30))
        self.takeoff_yaw_deg = float(self.takeoff_cfg.get("yaw_deg", 0.0))
        self.takeoff_stable_time = float(self.takeoff_cfg.get("stable_time", 0.80))
        # 起飞只要求达到安全悬停状态，不能沿用 scan/drop 的 8cm 严格判据。
        self.takeoff_pos_eps = float(self.takeoff_cfg.get("pos_eps", 0.15))
        self.takeoff_stable_vel = float(self.takeoff_cfg.get("stable_vel", 0.18))
        self.takeoff_stable_acc = float(self.takeoff_cfg.get("stable_acc", 0.80))
        self.takeoff_max_wait = float(self.takeoff_cfg.get("max_wait", 5.0))
        self.takeoff_min_height = float(self.takeoff_cfg.get("min_height", 1.20))

        # ---------- 降落切 AUTO.LAND 前的准备保持参数 ----------
        # 目的：先用 OFFBOARD 锁住当前位置和固定 yaw，降低切 AUTO.LAND 瞬间 yaw 目标跳变导致的机身扭转。
        self.land_prepare_time = float(self.landing_cfg.get("prepare_time", 0.80))
        self.emergency_land_prepare_time = float(self.landing_cfg.get("emergency_prepare_time", 0.20))
        self.disarm_land_prepare_time = float(
            self.landing_cfg.get("disarm_prepare_time", self.land_prepare_time)
        )
        self.land_prepare_vel_th = float(self.landing_cfg.get("prepare_vel_th", 0.20))
        self.land_prepare_max_wait = float(
            self.landing_cfg.get("prepare_max_wait", max(1.20, self.land_prepare_time + 0.50))
        )
        self.emergency_land_prepare_max_wait = float(
            self.landing_cfg.get(
                "emergency_prepare_max_wait",
                max(0.30, self.emergency_land_prepare_time + 0.15)
            )
        )
        self.disarm_land_prepare_max_wait = float(
            self.landing_cfg.get("disarm_prepare_max_wait", self.land_prepare_max_wait)
        )

        # ---------- 动作参数 ----------
        self.qr_scan_timeout = float(self.action_cfg.get("qr_scan_timeout", 3.0))
        self.image_scan_time = float(self.action_cfg.get("image_scan_time", 1.5))
        # 真实视觉图片靶顺序动作参数：
        # 1) image_class_timeout：1.35m 高空等待图片类别的最大时间；
        # 2) image_high_aim_timeout：类别确认后，高空单独瞄准的最大时间；
        # 3) image_descend_timeout：锁定 XY 后下降到 image_drop_z 并投放的最大等待时间。
        # image_class_scan_time 保留为旧 YAML 兼容别名；若写了 image_class_timeout，则优先用新名字。
        self.image_class_timeout = float(
            self.action_cfg.get(
                "image_class_timeout",
                self.action_cfg.get("image_class_scan_time", self.image_scan_time)
            )
        )
        self.image_class_min_confidence = float(self.action_cfg.get("image_class_min_confidence", 0.55))
        self.image_class_min_stable_count = int(self.action_cfg.get("image_class_min_stable_count", 1))
        self.image_drop_z = float(self.action_cfg.get("image_drop_z", 0.70))
        self.image_descend_timeout = float(self.action_cfg.get("image_descend_timeout", 2.50))
        self.image_descend_pos_eps = float(self.action_cfg.get("image_descend_pos_eps", 0.12))
        # 图片靶低空满足投放条件后，连续稳定等待这段时间再真正投放。
        # 这样可以避免刚降到低空、速度瞬间达标就立刻投放。
        self.image_pre_drop_stable_time = float(
            self.action_cfg.get("image_pre_drop_stable_time", 0.30)
        )
        self.image_align_timeout_policy = str(
            self.action_cfg.get("image_align_timeout_policy", "force_drop")
        ).strip().lower()
        # 兜底策略：如果“剩余未扫描图片靶数量 == 剩余未投图片物块数量”，
        # 说明再继续等分类的意义不大，当前图片靶应直接进入“下降 + 视觉对准 + 投放”流程。
        # 这样可以避免前面已经排除/错过若干图片靶后，最后几个目标因为分类不稳定而白白跳过。
        self.image_force_drop_remaining_equal = bool(
            self.action_cfg.get("image_force_drop_remaining_equal", True)
        )

        self.drop_time = float(self.action_cfg.get("drop_time", 1.2))
        self.hold_after_action = float(self.action_cfg.get("hold_after_action", 0.3))
        # ACTION 软判稳只用于防止长期被轻微速度/定位抖动卡住；仍要求位置近、速度低。
        self.action_soft_timeout = float(self.action_cfg.get("action_soft_timeout", 1.50))
        self.action_soft_pos_eps = float(self.action_cfg.get("action_soft_pos_eps", 0.18))
        self.action_soft_vel = float(self.action_cfg.get("action_soft_vel", 0.20))

        # ---------- 图片靶 / 特殊靶视觉闭环对准参数 ----------
        # 视觉节点负责把像素偏差换算成米制偏差，FSM 只使用 offset_x_m / offset_y_m 做小范围移动。
        self.vision_result_timeout = float(self.action_cfg.get("vision_result_timeout", 0.50))
        self.image_align_timeout = float(self.action_cfg.get("image_align_timeout", 4.0))
        self.special_align_timeout = float(self.action_cfg.get("special_align_timeout", 4.0))
        self.align_xy_eps = float(self.action_cfg.get("align_xy_eps", 0.05))
        self.align_step_max = float(self.action_cfg.get("align_step_max", 0.15))
        self.align_gain = float(self.action_cfg.get("align_gain", 0.80))
        self.align_min_confidence = float(self.action_cfg.get("align_min_confidence", 0.75))
        self.align_min_stable_count = int(self.action_cfg.get("align_min_stable_count", 3))

        # ---------- 图片靶“高空识别 -> 高空瞄准 -> 下降投放”参数 ----------
        # 思路：图片靶先在较高高度只做类别识别；确认是目标后，再单独高空瞄准并锁定 XY；
        # 随后下降到 image_drop_z 直接投放，不再把低空识别/瞄准作为必要环节。
        self.image_scan_aim_enabled = bool(self.action_cfg.get("image_scan_aim_enabled", True))
        self.image_scan_aim_xy_eps = float(self.action_cfg.get("image_scan_aim_xy_eps", 0.08))
        self.image_scan_aim_min_confidence = float(
            self.action_cfg.get("image_scan_aim_min_confidence", self.image_class_min_confidence)
        )
        self.image_scan_aim_max_wait = float(self.action_cfg.get("image_scan_aim_max_wait", 1.20))
        # 新流程里 image_high_aim_timeout 是“确认类别之后”的单独高空瞄准最长等待时间。
        # 兼容旧 YAML：如果没有写 image_high_aim_timeout，就沿用 image_scan_aim_max_wait。
        self.image_high_aim_timeout = float(
            self.action_cfg.get("image_high_aim_timeout", self.image_scan_aim_max_wait)
        )
        self.image_scan_aim_required_for_drop = bool(
            self.action_cfg.get("image_scan_aim_required_for_drop", True)
        )
        self.image_scan_aim_max_shift = float(self.action_cfg.get("image_scan_aim_max_shift", 0.30))
        self.image_lock_xy_on_descend = bool(self.action_cfg.get("image_lock_xy_on_descend", True))
        self.image_low_align_max_shift = float(self.action_cfg.get("image_low_align_max_shift", 0.16))
        self.image_low_no_vision_force_drop = bool(
            self.action_cfg.get("image_low_no_vision_force_drop", True)
        )
        self.image_low_no_vision_force_wait = float(
            self.action_cfg.get("image_low_no_vision_force_wait", 0.70)
        )
        self.image_low_force_pos_eps = float(self.action_cfg.get("image_low_force_pos_eps", 0.18))
        self.image_low_force_vel = float(self.action_cfg.get("image_low_force_vel", 0.20))

        # 特殊靶专用参数：特殊靶只做靶心瞄准，不做类别识别。
        # 这里单独放宽特殊靶条件，避免影响图片靶分类/投放逻辑。
        self.special_align_xy_eps = float(
            self.action_cfg.get("special_align_xy_eps", self.align_xy_eps)
        )
        self.special_min_confidence = float(
            self.action_cfg.get("special_min_confidence", self.align_min_confidence)
        )
        self.special_min_stable_count = int(
            self.action_cfg.get("special_min_stable_count", self.align_min_stable_count)
        )
        self.special_force_drop_enabled = bool(
            self.action_cfg.get("special_force_drop_enabled", True)
        )
        self.special_force_drop_timeout = float(
            self.action_cfg.get("special_force_drop_timeout", self.special_align_timeout)
        )
        self.special_force_drop_allow_no_vision = bool(
            self.action_cfg.get("special_force_drop_allow_no_vision", True)
        )
        self.special_force_drop_pos_eps = float(
            self.action_cfg.get("special_force_drop_pos_eps", 0.22)
        )
        self.special_force_drop_vel = float(
            self.action_cfg.get("special_force_drop_vel", 0.25)
        )

        # ---------- 已知图片靶固定点微调 / 手控投放等待参数 ----------
        # 两个已知图片靶不再等待分类，只允许视觉做短时间小范围修正；
        # 实飞手控投放模式下，瞄准完成或超时后悬停 manual_image_hover_time 秒。
        self.fixed_image_align_min_wait = float(self.action_cfg.get("fixed_image_align_min_wait", 0.15))
        self.fixed_image_align_timeout = float(self.action_cfg.get("fixed_image_align_timeout", 0.75))
        self.fixed_image_force_drop_timeout = float(self.action_cfg.get("fixed_image_force_drop_timeout", 1.15))
        self.fixed_image_fallback_vel = float(self.action_cfg.get("fixed_image_fallback_vel", 0.22))
        self.fixed_image_align_max_shift = float(self.action_cfg.get("fixed_image_align_max_shift", 0.22))
        self.fixed_image_align_min_confidence = float(self.action_cfg.get("fixed_image_align_min_confidence", 0.45))
        self.fixed_image_align_min_stable_count = int(self.action_cfg.get("fixed_image_align_min_stable_count", 1))
        self.manual_image_hover_time = float(self.action_cfg.get("manual_image_hover_time", 3.0))
        self.manual_image_no_auto_drop = bool(self.action_cfg.get("manual_image_no_auto_drop", True))

        # ---------- 圆环视觉搜索 / 穿越参数 ----------
        # ring_gate 视觉节点输出当前机体系下的 forward_m、offset_y_m、offset_z_m。
        # FSM 根据这些米制偏差动态生成穿环三点，并在 RING_PRE 做二次对准。
        self.ring_search_timeout = float(self.action_cfg.get("ring_search_timeout", 5.0))
        self.ring_timeout_policy = str(self.action_cfg.get("ring_timeout_policy", "hold")).lower()
        self.ring_pre_distance = float(self.action_cfg.get("ring_pre_distance", 0.65))
        self.ring_post_distance = float(self.action_cfg.get("ring_post_distance", 0.80))
        # 圆环前视相机安装偏置补偿。
        # ring_align_bias_x：MAVROS local X 方向补偿，单位 m；正值表示动态穿环点整体向 local +X 偏。
        # ring_align_bias_z：MAVROS local Z 高度补偿，单位 m；正值表示动态穿环点整体升高。
        # 用法：如果圆环在相机画面居中时，机体中心实际偏向 -X 侧，就把 bias_x 调成正值；
        # 如果机体中心实际偏低，就把 bias_z 调成正值。先从 ±0.03~0.05 m 小步调。
        self.ring_align_bias_x = float(self.action_cfg.get("ring_align_bias_x", 0.0))
        self.ring_align_bias_z = float(self.action_cfg.get("ring_align_bias_z", 0.0))
        # 穿环后动态横移点：RING_POST 后保持同一 y，再沿 x 方向朝降落区移动一段距离。
        self.ring_exit_x_distance = float(self.action_cfg.get("ring_exit_x_distance", 0.0))
        self.ring_align_timeout = float(self.action_cfg.get("ring_align_timeout", 4.0))
        self.ring_yz_eps = float(self.action_cfg.get("ring_yz_eps", 0.07))
        self.ring_align_step_max = float(self.action_cfg.get("ring_align_step_max", 0.12))
        self.ring_align_gain = float(self.action_cfg.get("ring_align_gain", 0.75))
        self.ring_min_confidence = float(self.action_cfg.get("ring_min_confidence", 0.70))
        self.ring_min_stable_count = int(self.action_cfg.get("ring_min_stable_count", 3))
        # 远距离圆环搜索阶段单独增加“最短观察时间”和“更高稳定帧数”。
        # 这样 RING_SEARCH_START 不会一识别到圆环就立刻生成穿环点，
        # 可以多看几帧，降低远距离中心点被单帧抖动带偏的概率。
        self.ring_search_min_time = float(self.action_cfg.get("ring_search_min_time", 0.0))
        self.ring_search_min_stable_count = int(
            self.action_cfg.get("ring_search_min_stable_count", self.ring_min_stable_count)
        )
        self.ring_forward_min = float(self.action_cfg.get("ring_forward_min", 0.30))
        self.ring_forward_max = float(self.action_cfg.get("ring_forward_max", 3.50))
        self.ring_min_z = float(self.action_cfg.get("ring_min_z", 1.35))
        self.ring_max_z = float(self.action_cfg.get("ring_max_z", 1.85))

        # ---------- 是否自动切模式、自动解锁 ----------
        self.auto_set_mode = rospy.get_param("~auto_set_mode", False)
        self.auto_arm = rospy.get_param("~auto_arm", False)

        # ---------- 飞控状态 ----------
        self.current_state = State()
        self.extended_state = ExtendedState()
        self.extended_state_ok = False

        self.current_pose = None
        self.current_yaw = 0.0

        self.current_vel = [0.0, 0.0, 0.0]
        self.current_speed = 0.0
        self.current_acc_norm = 0.0

        self.last_vel = None
        self.last_vel_time = None

        # ---------- home 原点 ----------
        self.home_ready = False
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_yaw = 0.0

        # ---------- FSM 状态 ----------
        # WAIT_START：节点启动后的默认状态，必须收到 /uav/start=True 才开始执行任务。
        self.fsm_state = "WAIT_START"
        self.safety_state = "IDLE"
        self.land_status = "IDLE"
        self.last_land_status = ""

        self.nav_phase = "INIT"
        self.wp_index = 0

        self.locked_yaw = 0.0
        self.cmd_vel = [0.0, 0.0, 0.0]

        self.phase_start_time = rospy.Time.now()
        self.stable_start_time = None
        self.action_start_time = None
        # 2026-05-23 修改：静态扫描/投放 ACTION 阶段的连续停稳起点。
        # 只有无人机位置、速度、加速度都满足阈值时，才从该时间开始累计有效扫描时间。
        self.action_stable_start_time = None
        self.action_sent = False

        # ---------- 安全状态变量 ----------
        self.start_requested = False
        self.reset_required = False
        self.emergency_reason = ""
        self.land_reason = ""
        self.disarm_after_land = False

        self.last_mode_req = rospy.Time.now()
        self.last_land_req = rospy.Time.now()

        # ---------- 降落准备保持状态 ----------
        self.land_prepare_start_time = None
        self.land_prepare_kind = ""
        self.land_hold_target = None
        self.land_locked_yaw = None

        # ---------- 视觉结果缓存 ----------
        self.qr_text = ""
        self.qr_class_1 = ""
        self.qr_class_2 = ""
        self.qr_land_side = ""

        # ---------- 图片靶识别 / 投放状态 ----------
        self.current_image_class = ""
        self.target_class_1_done = False
        self.target_class_2_done = False
        self.image_drop_count = 0

        # ---------- 图片靶动作运行时状态 ----------
        # CLASS_SCAN: 1.35m 高空只识别类别；HIGH_AIM: 类别确认后在高空单独瞄准并锁定 XY；
        # DESCEND: 关闭视觉并下降到 image_drop_z 直接投放；DROP_WAIT: 投放后等待机构动作完成。
        self.image_action_phase = ""
        self.image_phase_start_time = None
        self.pending_drop_cmd = ""
        self.pending_image_class = ""
        # 图片靶高空瞄准锁定点。进入 DESCEND 前写入，DESCEND 阶段只围绕这个锁定点降高投放。
        self.image_locked_xy = None
        self.image_locked_xy_reason = ""
        # 图片靶低空投放前连续稳定计时起点；只在 DESCEND 阶段使用。
        self.image_pre_drop_stable_start_time = None

        # ---------- 图片靶 / 特殊靶视觉闭环缓存 ----------
        # vision_results[target] 保存 /uav/vision_result 的最近一次 JSON 结果。
        # action_hold_target 是 ACTION 阶段的动态悬停点，视觉对准时会在原航点附近小范围更新。
        self.vision_results = {}
        self.action_hold_target = None
        self.drop_sent_time = None

        # ---------- 圆环动态航点缓存 ----------
        # RING_SEARCH_START 识别到圆环后，会把 RING_PRE / RING_CENTER / RING_POST 替换为动态绝对坐标。
        self.dynamic_ring_points = {}
        self.ring_dynamic_ready = False
        self.ring_last_debug = {}

        # ---------- 圆形绕障运行时缓存 ----------
        self.circle_finish_start_time = None
        self.circle_finish_stable_start_time = None

        # ---------- ROS 发布 ----------
        self.raw_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local",
            PositionTarget,
            queue_size=20
        )

        self.fsm_state_pub = rospy.Publisher(
            "/uav/fsm_state",
            String,
            queue_size=10
        )

        self.safety_state_pub = rospy.Publisher(
            "/uav/safety_state",
            String,
            queue_size=10,
            latch=True
        )

        self.land_status_pub = rospy.Publisher(
            "/uav/land_status",
            String,
            queue_size=10,
            latch=True
        )

        self.scan_enable_pub = rospy.Publisher(
            "/uav/scan_enable",
            Bool,
            queue_size=10
        )

        self.scan_target_pub = rospy.Publisher(
            "/uav/scan_target",
            String,
            queue_size=10
        )

        self.drop_cmd_pub = rospy.Publisher(
            "/uav/drop_cmd",
            String,
            queue_size=10
        )

        # ---------- ROS 订阅 ----------
        self.state_sub = rospy.Subscriber(
            "/mavros/state",
            State,
            self.state_cb,
            queue_size=10
        )

        self.extended_state_sub = rospy.Subscriber(
            "/mavros/extended_state",
            ExtendedState,
            self.extended_state_cb,
            queue_size=10
        )

        self.pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.pose_cb,
            queue_size=10
        )

        self.vel_sub = rospy.Subscriber(
            "/mavros/local_position/velocity_local",
            TwistStamped,
            self.vel_cb,
            queue_size=10
        )

        self.qr_sub = rospy.Subscriber(
            "/uav/qr_text",
            String,
            self.qr_text_cb,
            queue_size=10
        )

        self.image_sub = rospy.Subscriber(
            "/uav/image_class",
            String,
            self.image_class_cb,
            queue_size=10
        )

        self.vision_result_sub = rospy.Subscriber(
            "/uav/vision_result",
            String,
            self.vision_result_cb,
            queue_size=10
        )

        self.start_sub = rospy.Subscriber(
            "/uav/start",
            Bool,
            self.start_cb,
            queue_size=5
        )

        self.stop_sub = rospy.Subscriber(
            "/uav/stop",
            Bool,
            self.stop_cb,
            queue_size=5
        )

        self.land_sub = rospy.Subscriber(
            "/uav/land",
            Bool,
            self.land_cb,
            queue_size=5
        )

        self.disarm_sub = rospy.Subscriber(
            "/uav/disarm",
            Bool,
            self.disarm_cb,
            queue_size=5
        )

        self.reset_sub = rospy.Subscriber(
            "/uav/reset",
            Bool,
            self.reset_cb,
            queue_size=5
        )

        # ---------- MAVROS 服务 ----------
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arming_client = rospy.ServiceProxy(
            "/mavros/cmd/arming",
            CommandBool
        )

        self.set_mode_client = rospy.ServiceProxy(
            "/mavros/set_mode",
            SetMode
        )

        self.publish_land_status("IDLE")
        self.publish_safety_state("IDLE")

        rospy.loginfo("micro_uav_stage1_fsm safety version started.")
        rospy.loginfo("Loaded %d waypoints from %s", len(self.waypoints), self.mission_yaml)

    # =========================
    # 04. YAML 读取区
    # =========================

    def load_yaml(self, yaml_path):
        """读取 YAML 任务文件。"""
        yaml_path = os.path.expanduser(yaml_path)

        if not os.path.exists(yaml_path):
            raise RuntimeError("Mission YAML not found: %s" % yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if cfg is None or "mission" not in cfg:
            raise RuntimeError("Invalid mission YAML: missing mission field.")

        return cfg

    def expand_obstacle_loop_waypoints(self, cfg, raw_points):
        """
        根据 YAML 中 mission.obstacle.loop_count 展开绕障模板点。
        用法：在一圈绕障模板点上写 obstacle_loop: true，FSM 会按 loop_count 重复这些点。
        非 obstacle_loop 点保持原顺序不变。
        """
        obstacle_cfg = cfg.get("mission", {}).get("obstacle", {})

        try:
            loop_count = int(obstacle_cfg.get("loop_count", 1))
        except Exception:
            loop_count = 1

        try:
            max_loop_count = int(obstacle_cfg.get("max_loop_count", 8))
        except Exception:
            max_loop_count = 8

        max_loop_count = max(1, max_loop_count)
        loop_count = clamp(loop_count, 1, max_loop_count)
        loop_count = int(loop_count)

        expanded = []
        template = []
        template_groups = 0

        def flush_template():
            nonlocal template, template_groups
            if len(template) == 0:
                return

            template_groups += 1
            for loop_idx in range(loop_count):
                for item in template:
                    new_item = copy.deepcopy(item)
                    base_name = str(new_item.get("name", "OBS_LOOP"))
                    if loop_count > 1:
                        new_item["name"] = "%s_L%02d" % (base_name, loop_idx + 1)
                    # 解析 Waypoint 时不需要这个临时标记。
                    new_item.pop("obstacle_loop", None)
                    expanded.append(new_item)

            template = []

        for item in raw_points:
            if bool(item.get("obstacle_loop", False)):
                template.append(item)
            else:
                flush_template()
                expanded.append(item)

        flush_template()

        if template_groups > 0:
            rospy.loginfo(
                "Obstacle loop expanded: loop_count=%d template_groups=%d raw_points=%d expanded_points=%d",
                loop_count,
                template_groups,
                len(raw_points),
                len(expanded)
            )

        return expanded

    def parse_waypoints(self, cfg):
        """从 YAML 中解析航点列表。"""
        result = []
        raw_points = self.expand_obstacle_loop_waypoints(
            cfg,
            cfg["mission"].get("waypoints", [])
        )

        for item in raw_points:
            wp = Waypoint(
                name=str(item.get("name", "UNKNOWN")),
                kind=str(item.get("kind", "route")),
                action=str(item.get("action", "none")),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                z=float(item.get("z", 1.30)),
                yaw_deg=float(item.get("yaw_deg", 0.0)),
                speed=float(item.get("speed", 0.4)),
                acc=float(item.get("acc", 0.3)),
                hold_time=float(item.get("hold_time", 0.5)),
                control_mode=str(item.get("control_mode", "auto")),
                dynamic_land=bool(item.get("dynamic_land", False))
            )

            if wp.kind not in ["route", "scan"]:
                raise RuntimeError("Waypoint %s has invalid kind: %s" % (wp.name, wp.kind))

            # 2026-05-18 修改：默认 route 标记为 position；2026-05-23 起 position 也会走速度受限的融合控制。
            if wp.control_mode == "auto":
                if wp.kind == "route":
                    wp.control_mode = "position"
                else:
                    wp.control_mode = "fusion"

            # 2026-06-02 修改：绕障中间点不再按 position 点逐个刹停。
            # OBS_ENTRY / OBS_EXIT 保持 position，用来稳定进入和退出绕障区；
            # 其余 OBS_* 普通 route 点自动改为 fusion_route，靠近后直接切下一个点，实现连续通过。
            if (
                wp.kind == "route" and
                wp.action == "none" and
                self.is_obstacle_fast_pass_waypoint(wp.name)
            ):
                wp.control_mode = "fusion_route"

            if wp.control_mode not in ["position", "fusion", "fusion_route"]:
                raise RuntimeError(
                    "Waypoint %s has invalid control_mode: %s" %
                    (wp.name, wp.control_mode)
                )

            result.append(wp)

        if len(result) == 0:
            raise RuntimeError("No waypoint found in mission YAML.")

        return result

    # =========================
    # 05. ROS 回调函数区
    # =========================

    def state_cb(self, msg):
        """更新飞控连接、模式和解锁状态，并检测正常任务中的异常退出。"""
        prev_armed = self.current_state.armed
        prev_mode = self.current_state.mode
        self.current_state = msg

        active_task = (
            self.fsm_state in ["WAIT_FCU", "TAKEOFF", "MISSION", "STAGE1_DONE"] and
            self.start_requested and
            not self.reset_required
        )

        # 飞行中突然上锁，禁止后续自动继续任务，避免再次解锁后冲向旧航点。
        if active_task and prev_armed and not msg.armed:
            self.force_wait_reset("armed changed True -> False during active task")
            return

        # 正常任务中突然退出 OFFBOARD，视作控制链路异常，直接急停降落。
        if active_task and prev_mode == "OFFBOARD" and msg.mode not in ["OFFBOARD", "AUTO.LAND"]:
            if msg.armed:
                self.request_emergency_land("mode changed OFFBOARD -> %s during active task" % msg.mode)
            else:
                self.force_wait_reset("mode changed OFFBOARD -> %s and vehicle is disarmed" % msg.mode)
            return

        # 如果外部已经切到 AUTO.LAND，就顺势进入本 FSM 的降落流程，不再回原航点。
        if active_task and msg.mode == "AUTO.LAND":
            self.request_normal_land("external AUTO.LAND detected")

    def extended_state_cb(self, msg):
        """更新 MAVROS 扩展状态，用 landed_state 判断是否落地。"""
        self.extended_state = msg
        self.extended_state_ok = True

    def pose_cb(self, msg):
        """更新当前位置和当前 yaw。"""
        self.current_pose = msg

        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        self.current_yaw = yaw

        if not self.home_ready:
            self.home_x = msg.pose.position.x
            self.home_y = msg.pose.position.y
            self.home_z = msg.pose.position.z
            self.home_yaw = yaw
            self.home_ready = True

            rospy.loginfo(
                "Home set: x=%.2f y=%.2f z=%.2f yaw=%.1f deg",
                self.home_x,
                self.home_y,
                self.home_z,
                math.degrees(self.home_yaw)
            )

    def vel_cb(self, msg):
        """更新当前速度，并通过速度差分估算加速度。"""
        now = msg.header.stamp

        if now.to_sec() <= 0.0:
            now = rospy.Time.now()

        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z

        self.current_vel = [vx, vy, vz]
        self.current_speed = norm3(vx, vy, vz)

        if self.last_vel is not None and self.last_vel_time is not None:
            dt = (now - self.last_vel_time).to_sec()

            if dt > 1e-3:
                ax = (vx - self.last_vel[0]) / dt
                ay = (vy - self.last_vel[1]) / dt
                az = (vz - self.last_vel[2]) / dt
                raw_acc_norm = norm3(ax, ay, az)

                # 速度差分会把测量抖动放大成“假加速度”。
                # 先低通再参与日志/可选判稳，避免单帧尖峰不断清零 stable 计时。
                alpha = clamp(getattr(self, "acc_lpf_alpha", 0.25), 0.0, 1.0)
                self.current_acc_norm = (
                    alpha * raw_acc_norm +
                    (1.0 - alpha) * self.current_acc_norm
                )

        self.last_vel = [vx, vy, vz]
        self.last_vel_time = now

    def qr_text_cb(self, msg):
        """接收二维码识别结果，格式示例：man,apple,left。"""
        self.qr_text = msg.data.strip()
        parts = [p.strip() for p in self.qr_text.split(",")]

        if len(parts) == 3:
            self.qr_class_1 = parts[0].lower()
            self.qr_class_2 = parts[1].lower()
            self.qr_land_side = parts[2].lower()

            rospy.loginfo(
                "QR parsed: class1=%s class2=%s land=%s",
                self.qr_class_1,
                self.qr_class_2,
                self.qr_land_side
            )

    def image_class_cb(self, msg):
        """接收当前图片靶类别，视觉节点建议发布 CIFAR-100 英文类别名。"""
        self.current_image_class = msg.data.strip().lower()

    def vision_result_cb(self, msg):
        """
        接收视觉节点统一结果。
        控制闭环只使用米制偏差 offset_x_m / offset_y_m；像素字段只用于视觉节点调试。
        期望 JSON 示例：
        {
          "target": "image_target",
          "detected": true,
          "class_name": "apple",
          "offset_x_m": 0.04,
          "offset_y_m": -0.03,
          "confidence": 0.90,
          "stable_count": 5
        }
        圆环示例：
        {
          "target": "ring_gate",
          "detected": true,
          "forward_m": 1.20,
          "offset_y_m": -0.05,
          "offset_z_m": 0.03,
          "confidence": 0.86,
          "stable_count": 4
        }
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "Invalid /uav/vision_result JSON: %s", str(e))
            return

        target = str(data.get("target", "")).strip().lower()

        if target not in ["image_target", "special_target", "ring_gate"]:
            return

        data["_recv_time"] = rospy.Time.now()
        self.vision_results[target] = data

        # 兼容旧的 /uav/image_class 逻辑：新视觉结果里带 class_name 时也更新 current_image_class。
        if target == "image_target":
            class_name = str(data.get("class_name", "")).strip().lower()
            if class_name != "":
                self.current_image_class = class_name

    def start_cb(self, msg):
        """收到 /uav/start=True 后，才允许进入 WAIT_FCU、起飞和任务。"""
        if not msg.data:
            return

        if self.reset_required or self.fsm_state == "WAIT_RESET":
            rospy.logerr("Start rejected: reset is required before a new mission.")
            self.publish_safety_state("START_REJECTED_WAIT_RESET")
            return

        if self.fsm_state in ["LANDING", "EMERGENCY_LAND", "DISARMING"]:
            rospy.logerr("Start rejected: landing or disarming is in progress.")
            self.publish_safety_state("START_REJECTED_LANDING")
            return

        if self.fsm_state in ["TAKEOFF", "MISSION"]:
            rospy.logwarn("Start ignored: mission is already running.")
            return

        rospy.loginfo("/uav/start received. Mission start requested.")
        self.clear_task_runtime(reset_wp=True)
        self.start_requested = True
        self.reset_required = False
        self.emergency_reason = ""
        self.land_reason = ""
        self.disarm_after_land = False
        self.publish_land_status("IDLE")
        self.publish_safety_state("START_REQUESTED")
        self.enter_fsm_state("WAIT_FCU")

    def stop_cb(self, msg):
        """收到 /uav/stop=True 后，立即取消任务并进入急停降落流程。"""
        if not msg.data:
            return

        self.request_emergency_land("/uav/stop received")

    def land_cb(self, msg):
        """收到 /uav/land=True 后，普通降落：取消任务、AUTO.LAND、落地后上锁。"""
        if not msg.data:
            return

        self.request_normal_land("/uav/land received")

    def disarm_cb(self, msg):
        """收到 /uav/disarm=True 后，若在空中则先降落，落地后再上锁。"""
        if not msg.data:
            return

        rospy.logwarn("/uav/disarm received. Safe disarm requested.")
        self.cancel_task_outputs()
        self.clear_task_runtime(reset_wp=True)
        self.start_requested = False
        self.disarm_after_land = True

        if self.current_state.armed and not self.is_landed():
            self.land_reason = "/uav/disarm received while airborne"
            self.publish_safety_state("DISARM_WAIT_LAND")
            self.publish_land_status("DISARM_WAIT_LAND")
            self.setup_landing_prepare("disarm")
            self.enter_fsm_state("DISARMING")
        else:
            self.publish_safety_state("DISARM_DIRECT")
            self.publish_land_status("DISARM_REQUESTED")
            self.enter_fsm_state("DISARMING")

    def reset_cb(self, msg):
        """收到 /uav/reset=True 后，只有无人机已上锁才允许回到 WAIT_START。"""
        if not msg.data:
            return

        if self.current_state.armed:
            rospy.logerr("Reset rejected: vehicle is still armed. Land and disarm first.")
            self.publish_safety_state("RESET_REJECTED_ARMED")
            self.publish_land_status("RESET_REJECTED_ARMED")
            return

        rospy.logwarn("/uav/reset accepted. FSM returns to WAIT_START.")
        self.cancel_task_outputs()
        self.clear_task_runtime(reset_wp=True)
        self.start_requested = False
        self.reset_required = False
        self.emergency_reason = ""
        self.land_reason = ""
        self.disarm_after_land = False
        self.publish_safety_state("IDLE")
        self.publish_land_status("IDLE")
        self.enter_fsm_state("WAIT_START")

    # =========================
    # 06. MAVROS 指令发布区
    # =========================

    def make_target_msg(self):
        """生成 PositionTarget 基础消息。"""
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        return msg

    def publish_velocity_yaw(self, vx, vy, vz, yaw):
        """
        发布速度 + yaw 指令。
        用于 route 点移动、原地转向、刹停、降落前清零旧 setpoint。
        """
        msg = self.make_target_msg()

        msg.type_mask = (
            PositionTarget.IGNORE_PX |
            PositionTarget.IGNORE_PY |
            PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = vz
        msg.yaw = yaw

        self.raw_pub.publish(msg)

    def publish_position_velocity_yaw(self, x, y, z, vx, vy, vz, yaw):
        """
        发布位置 + 速度 + yaw 融合指令。
        本版正常移动阶段不再使用它做限速；主要用于起飞、HOLD、ACTION 等锁点阶段。
        """
        msg = self.make_target_msg()

        msg.type_mask = (
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        msg.position.x = x
        msg.position.y = y
        msg.position.z = z

        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = vz

        msg.yaw = yaw

        self.raw_pub.publish(msg)

    def publish_position_velocity_accel_yaw(self, x, y, z, vx, vy, vz, ax, ay, az, yaw):
        """
        发布位置 + 速度前馈 + 加速度前馈 + yaw 指令。
        用于圆形绕障这类连续轨迹。位置约束圆轨迹，速度给出切向前馈，
        加速度给出向心/切向加速度前馈，三者必须来自同一条轨迹。
        """
        msg = self.make_target_msg()

        # 不忽略位置、速度、加速度，只忽略 yaw_rate。
        # 未设置 FORCE 位时，acceleration_or_force 按加速度前馈解释。
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

        msg.yaw = yaw

        self.raw_pub.publish(msg)

    def publish_position_yaw(self, x, y, z, yaw):
        """
        发布纯位置 + yaw 指令。
        保留为备用接口；正常任务中 HOLD/ACTION 更常使用位置 + 0 速度锁点。
        """
        msg = self.make_target_msg()

        msg.type_mask = (
            PositionTarget.IGNORE_VX |
            PositionTarget.IGNORE_VY |
            PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.yaw = yaw

        self.raw_pub.publish(msg)

    def publish_neutral_setpoint(self, yaw=None):
        """发布零速度指令，防止急停/降落前最后一帧旧目标继续推动飞机。"""
        if yaw is None:
            yaw = self.current_yaw

        self.publish_velocity_yaw(0.0, 0.0, 0.0, yaw)

    # =========================
    # 07. FSM 工具函数区
    # =========================

    def reset_image_action_runtime(self):
        """清理单个图片靶 ACTION 的两阶段运行时状态。"""
        self.image_action_phase = ""
        self.image_phase_start_time = None
        self.pending_drop_cmd = ""
        self.pending_image_class = ""
        self.image_locked_xy = None
        self.image_locked_xy_reason = ""
        self.image_pre_drop_stable_start_time = None

    def mission_z_to_abs(self, z):
        """把任务 YAML 中的高度 z 转成 MAVROS local 绝对高度。"""
        if self.frame_mode == "relative_home":
            return self.home_z + float(z)
        return float(z)

    def get_image_drop_abs_z(self):
        """读取图片靶低空投放高度的绝对 z。"""
        return self.mission_z_to_abs(self.image_drop_z)

    def apply_default_qr_result(self, reason):
        """二维码失败或占位扫描结束时，按 YAML 默认二维码结果继续任务。"""
        if not self.qr_default_enabled:
            rospy.logwarn("QR default disabled, continue with empty QR result. reason=%s", reason)
            return False

        class_1 = self.qr_default_class_1
        class_2 = self.qr_default_class_2
        land_side = self.qr_default_land_side

        if land_side not in ["left", "right"]:
            land_side = str(self.landing_cfg.get("default_side", "right")).strip().lower()

        if class_1 == "" or class_2 == "":
            rospy.logwarn(
                "QR default has empty class: class_1=%s class_2=%s. Image targets may not be dropped.",
                class_1,
                class_2
            )

        self.qr_class_1 = class_1
        self.qr_class_2 = class_2
        self.qr_land_side = land_side if land_side in ["left", "right"] else "right"
        self.qr_text = "default:%s,%s,%s" % (
            self.qr_class_1,
            self.qr_class_2,
            self.qr_land_side
        )

        rospy.logwarn(
            "QR default applied: reason=%s class1=%s class2=%s land=%s",
            reason,
            self.qr_class_1,
            self.qr_class_2,
            self.qr_land_side
        )
        return True

    def mark_image_drop_done(self, drop_cmd):
        """投放命令发出后，更新两个二维码目标的完成状态。"""
        if drop_cmd == "image_drop_1" and not self.target_class_1_done:
            self.target_class_1_done = True
            self.image_drop_count += 1
            return

        if drop_cmd == "image_drop_2" and not self.target_class_2_done:
            self.target_class_2_done = True
            self.image_drop_count += 1
            return

    def get_remaining_image_drop_count(self):
        """统计还需要投到图片靶上的物块数量。"""
        remaining = 0

        if self.qr_class_1.strip() != "" and not self.target_class_1_done:
            remaining += 1

        if self.qr_class_2.strip() != "" and not self.target_class_2_done:
            remaining += 1

        return remaining

    def get_remaining_image_scan_count(self):
        """统计从当前航点开始，后面还剩几个图片靶扫描航点。"""
        start_index = max(0, self.wp_index)
        remaining = 0

        for wp in self.waypoints[start_index:]:
            if wp.action == "image_scan_maybe_drop":
                remaining += 1

        return remaining

    def get_next_pending_image_drop_cmd(self):
        """选择下一个还没用过的图片靶投放命令。"""
        if self.qr_class_1.strip() != "" and not self.target_class_1_done:
            return "image_drop_1"

        if self.qr_class_2.strip() != "" and not self.target_class_2_done:
            return "image_drop_2"

        return ""

    def should_force_image_drop_by_remaining_count(self):
        """
        判断是否启用“剩余图片靶数 == 剩余投放物块数”的兜底投放逻辑。

        返回：
        should_force: 是否应强制把当前图片靶当作目标靶处理；
        drop_cmd: 本次使用的投放命令；
        remaining_scans: 剩余图片靶扫描航点数量；
        remaining_drops: 剩余图片靶投放物块数量。
        """
        remaining_scans = self.get_remaining_image_scan_count()
        remaining_drops = self.get_remaining_image_drop_count()

        if not self.image_force_drop_remaining_equal:
            return False, "", remaining_scans, remaining_drops

        if remaining_drops <= 0 or remaining_scans <= 0:
            return False, "", remaining_scans, remaining_drops

        if remaining_scans != remaining_drops:
            return False, "", remaining_scans, remaining_drops

        drop_cmd = self.get_next_pending_image_drop_cmd()
        if drop_cmd == "":
            return False, "", remaining_scans, remaining_drops

        return True, drop_cmd, remaining_scans, remaining_drops

    def lock_current_image_xy(self, tx, ty, reason):
        """锁定当前图片靶 ACTION 的 XY 目标点，后续下降/投放不再回到 YAML 原始 XY。"""
        if self.action_hold_target is not None:
            lock_x = float(self.action_hold_target[0])
            lock_y = float(self.action_hold_target[1])
        else:
            lock_x = float(tx)
            lock_y = float(ty)

        self.image_locked_xy = [lock_x, lock_y]
        self.image_locked_xy_reason = str(reason)

        rospy.loginfo(
            "[IMAGE_LOCK_XY] xy=(%.3f, %.3f) reason=%s",
            lock_x,
            lock_y,
            self.image_locked_xy_reason
        )

        return lock_x, lock_y

    def get_image_locked_xy_or_default(self, tx, ty):
        """读取图片靶已锁定 XY；若未锁定，则回退到当前航点 XY。"""
        if self.image_lock_xy_on_descend and self.image_locked_xy is not None:
            return float(self.image_locked_xy[0]), float(self.image_locked_xy[1])

        return float(tx), float(ty)

    def start_image_high_aim_for_drop(self, wp, tx, ty, tz, drop_cmd, img_class, reason):
        """
        类别已经确认后，进入“高空单独瞄准”阶段。
        这个阶段仍保持 image_target 视觉开启，但不再重新做投放决策，只利用 offset_x_m / offset_y_m 修正 XY。
        """
        self.pending_drop_cmd = drop_cmd
        self.pending_image_class = img_class
        self.image_action_phase = "HIGH_AIM"
        self.image_phase_start_time = rospy.Time.now()
        self.action_hold_target = [float(tx), float(ty), float(tz)]
        self.image_locked_xy = None
        self.image_locked_xy_reason = ""
        self.vision_results.pop("image_target", None)

        rospy.loginfo(
            "[IMAGE_START_HIGH_AIM] wp=%s class=%s cmd=%s reason=%s base_xy=(%.3f, %.3f) z=%.2f",
            wp.name,
            img_class,
            drop_cmd,
            reason,
            tx,
            ty,
            tz
        )

    def start_image_descend_for_drop(self, wp, tx, ty, drop_abs_z, drop_cmd, img_class, reason):
        """
        统一进入图片靶下降投放流程。
        本版进入 DESCEND 前必须已经完成“高空识别 + 高空瞄准锁 XY”；
        DESCEND 阶段关闭视觉，只降低高度到 image_drop_z，到位稳定后直接投放。
        """
        if self.image_lock_xy_on_descend and self.image_locked_xy is None:
            lock_x, lock_y = self.lock_current_image_xy(tx, ty, reason)
        else:
            lock_x, lock_y = self.get_image_locked_xy_or_default(tx, ty)

        self.pending_drop_cmd = drop_cmd
        self.pending_image_class = img_class
        self.image_action_phase = "DESCEND"
        self.image_phase_start_time = rospy.Time.now()
        self.image_pre_drop_stable_start_time = None
        self.action_hold_target = [lock_x, lock_y, drop_abs_z]
        self.vision_results.pop("image_target", None)
        # 一进入下降阶段就关闭视觉请求，避免拍到下降过程的图片，也避免低空结果干扰控制。
        self.disable_scan_request()

        rospy.loginfo(
            "[IMAGE_START_DESCEND] wp=%s class=%s cmd=%s reason=%s locked_xy=(%.3f, %.3f) descend_to_z=%.2f",
            wp.name,
            img_class,
            drop_cmd,
            reason,
            lock_x,
            lock_y,
            self.image_drop_z
        )

    def should_skip_waypoint_before_nav(self, wp):
        """在进入 YAW_ALIGN/MOVE 前跳过无需执行的航点，避免多余到点/hold。"""
        if (
            wp.action == "image_scan_maybe_drop" and
            self.target_class_1_done and
            self.target_class_2_done
        ):
            rospy.loginfo(
                "[SKIP_WP_BEFORE_NAV] both image targets dropped, skip %s before navigation.",
                wp.name
            )
            self.disable_scan_request()
            return True

        return False

    def current_xyz(self):
        """读取当前无人机位置。"""
        p = self.current_pose.pose.position
        return p.x, p.y, p.z

    def get_selected_land_side(self):
        """根据二维码结果选择降落方向；若 YAML 设置 force_side，则强制使用固定降落点。"""
        force_side = str(self.landing_cfg.get("force_side", "")).lower().strip()
        if force_side in ["left", "right"]:
            return force_side

        side = self.qr_land_side.lower().strip()

        if side not in ["left", "right"]:
            side = str(self.landing_cfg.get("default_side", "left")).lower()

        if side not in ["left", "right"]:
            side = "left"

        return side

    def get_dynamic_land_xy(self, wp):
        """
        根据二维码 left/right 动态生成降落相关航点坐标。

        设计意图：
        1. LAND_PRE_DYNAMIC：先飞到最终 H 点前方的预降速点，用来把从圆环后方回来的速度收下来；
        2. LAND_APPROACH_DYNAMIC：保留旧配置兼容，作用和 LAND_PRE_DYNAMIC 类似；
        3. LAND_FINAL_DYNAMIC：最终 H 点上方，悬停判稳后触发 mission_done / 降落。
        """
        side = self.get_selected_land_side()
        side_cfg = self.landing_cfg.get(side, {})

        land_x = float(side_cfg.get("x", wp.x))
        land_y = float(side_cfg.get("y", wp.y))

        if wp.name.startswith("LAND_PRE"):
            # 预降速点默认使用 landing.pre_x；未配置时复用旧的 approach_x。
            # 当前比赛坐标里最终 H 点 x 通常为 0.0，pre_x=1.0 表示先到 H 点前方 1m 处收速。
            pre_x = float(self.landing_cfg.get("pre_x", self.landing_cfg.get("approach_x", wp.x)))
            return pre_x, land_y

        if wp.name.startswith("LAND_APPROACH"):
            approach_x = float(self.landing_cfg.get("approach_x", wp.x))
            return approach_x, land_y

        return land_x, land_y

    def get_selected_land_abs_xy(self):
        """读取当前二维码选择的最终降落点，并转换成 MAVROS local 绝对坐标。"""
        side = self.get_selected_land_side()
        side_cfg = self.landing_cfg.get(side, {})

        land_x = float(side_cfg.get("x", 0.0))
        land_y = float(side_cfg.get("y", 0.0))

        if self.frame_mode == "relative_home":
            return self.home_x + land_x, self.home_y + land_y

        return land_x, land_y

    def get_abs_wp(self, wp):
        """将 YAML 中的相对航点转换成 MAVROS local 坐标。"""
        # 圆环搜索成功后，RING_PRE / RING_CENTER / RING_POST 使用动态绝对坐标，
        # 不再使用 YAML 中的默认固定点。RING_SEARCH_START 仍使用 YAML 固定搜索点。
        if wp.name in self.dynamic_ring_points:
            rx, ry, rz = self.dynamic_ring_points[wp.name]
            return rx, ry, rz, wp.yaw_rad()

        rel_x = wp.x
        rel_y = wp.y

        # 2026-05-18 修改：降落航点可根据二维码 left/right 动态选择上下 H 点。
        if wp.dynamic_land:
            rel_x, rel_y = self.get_dynamic_land_xy(wp)

        if self.frame_mode == "relative_home":
            return (
                self.home_x + rel_x,
                self.home_y + rel_y,
                self.home_z + wp.z,
                wp.yaw_rad()
            )

        return rel_x, rel_y, wp.z, wp.yaw_rad()

    def enter_fsm_state(self, new_state):
        """切换 FSM 大状态。"""
        self.fsm_state = new_state
        self.phase_start_time = rospy.Time.now()
        self.stable_start_time = None
        self.action_start_time = None
        self.action_stable_start_time = None
        self.action_sent = False
        self.action_hold_target = None
        self.drop_sent_time = None
        self.reset_image_action_runtime()
        self.cmd_vel = [0.0, 0.0, 0.0]

        rospy.loginfo("FSM -> %s", new_state)

    def enter_nav_phase(self, new_phase):
        """切换单个航点内部导航阶段。"""
        self.nav_phase = new_phase
        self.phase_start_time = rospy.Time.now()
        self.stable_start_time = None
        self.cmd_vel = [0.0, 0.0, 0.0]

        if new_phase == "INIT":
            # 每进入新航点，必须清空上一航点的 ACTION 运行时状态。
            # 原版本只清了 action_hold_target/drop_sent_time，没有清 action_start_time/action_sent。
            # 结果是第二个固定图片靶 image_drop_2 会继承第一个 image_drop_1 的 action_sent=True，
            # 但 drop_sent_time 又被清成 None，导致它既不会再次发送投放命令，也永远等不到 drop_wait_finished()。
            self.action_start_time = None
            self.action_stable_start_time = None
            self.action_sent = False
            self.action_hold_target = None
            self.drop_sent_time = None
            self.reset_image_action_runtime()
            self.circle_finish_start_time = None
            self.circle_finish_stable_start_time = None

        rospy.loginfo("NAV_PHASE -> %s", new_phase)

    def next_waypoint(self):
        """进入下一个航点。"""
        if self.wp_index < len(self.waypoints):
            rospy.loginfo("Waypoint done: %s", self.waypoints[self.wp_index].name)

        self.wp_index += 1
        self.enter_nav_phase("INIT")

        if self.wp_index >= len(self.waypoints):
            self.enter_fsm_state("STAGE1_DONE")

    def clear_task_runtime(self, reset_wp):
        """清理任务运行时状态，防止旧航点、旧速度和旧动作残留。"""
        if reset_wp:
            self.wp_index = 0
            self.qr_text = ""
            self.qr_class_1 = ""
            self.qr_class_2 = ""
            self.qr_land_side = ""
            self.current_image_class = ""
            self.target_class_1_done = False
            self.target_class_2_done = False
            self.image_drop_count = 0
            self.reset_image_action_runtime()
            self.vision_results = {}
            self.action_hold_target = None
            self.drop_sent_time = None
            self.dynamic_ring_points = {}
            self.ring_dynamic_ready = False
            self.ring_last_debug = {}
            self.circle_finish_start_time = None
            self.circle_finish_stable_start_time = None

        self.nav_phase = "INIT"
        self.locked_yaw = self.current_yaw
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.phase_start_time = rospy.Time.now()
        self.stable_start_time = None
        self.action_start_time = None
        self.action_stable_start_time = None
        self.action_sent = False
        self.action_hold_target = None
        self.drop_sent_time = None
        self.reset_image_action_runtime()
        self.clear_landing_prepare()

    def cancel_task_outputs(self):
        """关闭扫描输出，并清除本 FSM 的动作状态。"""
        self.scan_enable_pub.publish(Bool(data=False))
        self.scan_target_pub.publish(String(data="none"))
        self.action_sent = False
        self.action_start_time = None
        self.action_stable_start_time = None
        self.action_hold_target = None
        self.drop_sent_time = None
        self.reset_image_action_runtime()

    def clear_landing_prepare(self):
        """清理降落准备保持状态。"""
        self.land_prepare_start_time = None
        self.land_prepare_kind = ""
        self.land_hold_target = None
        self.land_locked_yaw = None

    def setup_landing_prepare(self, kind):
        """记录切 AUTO.LAND 前的保持点和固定 yaw。"""
        self.land_prepare_start_time = rospy.Time.now()
        self.land_prepare_kind = kind
        self.land_locked_yaw = self.current_yaw

        if self.current_pose is not None:
            cx, cy, cz = self.current_xyz()
            self.land_hold_target = [cx, cy, cz]
        else:
            self.land_hold_target = None

        rospy.logwarn(
            "Landing prepare locked: kind=%s yaw=%.1f deg hold=%s",
            kind,
            math.degrees(self.land_locked_yaw),
            str(self.land_hold_target)
        )

    def get_landing_prepare_limits(self):
        """根据普通降落、急停降落或空中上锁请求选择准备保持时间。"""
        if self.land_prepare_kind == "emergency":
            return self.emergency_land_prepare_time, self.emergency_land_prepare_max_wait

        if self.land_prepare_kind == "disarm":
            return self.disarm_land_prepare_time, self.disarm_land_prepare_max_wait

        return self.land_prepare_time, self.land_prepare_max_wait

    def publish_landing_prepare_setpoint(self):
        """切 AUTO.LAND 前保持当前位置和固定 yaw，避免 yaw setpoint 随估计值抖动。"""
        yaw = self.land_locked_yaw
        if yaw is None:
            yaw = self.current_yaw

        if self.land_hold_target is not None:
            self.publish_position_velocity_yaw(
                self.land_hold_target[0],
                self.land_hold_target[1],
                self.land_hold_target[2],
                0.0,
                0.0,
                0.0,
                yaw
            )
        else:
            self.publish_neutral_setpoint(yaw)

    def landing_prepare_ready(self):
        """判断切 AUTO.LAND 前的短暂停稳是否完成。"""
        if self.current_state.mode == "AUTO.LAND":
            return True

        if self.land_prepare_start_time is None:
            self.setup_landing_prepare("normal")

        self.publish_landing_prepare_setpoint()

        now = rospy.Time.now()
        elapsed = (now - self.land_prepare_start_time).to_sec()
        prepare_time, max_wait = self.get_landing_prepare_limits()
        speed_ready = self.current_speed < self.land_prepare_vel_th

        ready = (
            elapsed >= prepare_time and
            (speed_ready or elapsed >= max_wait)
        )

        if not ready:
            if self.land_prepare_kind == "emergency":
                status = "EMERGENCY_LAND_PREPARE"
            elif self.land_prepare_kind == "disarm":
                status = "DISARM_LAND_PREPARE"
            else:
                status = "LAND_PREPARE"

            self.publish_land_status(status)
            rospy.logwarn_throttle(
                0.3,
                "[LAND_PREPARE] kind=%s elapsed=%.2f/%.2f max=%.2f speed=%.2f<th=%.2f yaw=%.1f",
                self.land_prepare_kind,
                elapsed,
                prepare_time,
                max_wait,
                self.current_speed,
                self.land_prepare_vel_th,
                math.degrees(self.land_locked_yaw if self.land_locked_yaw is not None else self.current_yaw)
            )
            return False

        if not speed_ready:
            rospy.logwarn(
                "[LAND_PREPARE] max wait reached, continue AUTO.LAND: kind=%s elapsed=%.2f speed=%.2f",
                self.land_prepare_kind,
                elapsed,
                self.current_speed
            )

        return True

    def is_yaw_aligned(self, target_yaw):
        """判断 yaw 是否对齐。"""
        yaw_err = abs(wrap_pi(target_yaw - self.current_yaw))
        return yaw_err < self.yaw_eps

    def is_speed_acc_stable(self, vel_th, acc_th=None):
        """判断当前速度是否稳定；acc_th 为空或 <= 0 时不启用加速度硬门槛。"""
        if self.current_speed >= vel_th:
            return False

        if acc_th is None or acc_th <= 0.0:
            return True

        return self.current_acc_norm < acc_th

    def publish_fsm_state(self):
        """发布当前 FSM 状态，便于地面站显示。"""
        if self.wp_index < len(self.waypoints):
            wp_name = self.waypoints[self.wp_index].name
        else:
            wp_name = "NONE"

        msg = String()
        msg.data = "%s | safety=%s | land=%s | phase=%s | wp=%s" % (
            self.fsm_state,
            self.safety_state,
            self.land_status,
            self.nav_phase,
            wp_name
        )

        self.fsm_state_pub.publish(msg)

    def publish_safety_state(self, status):
        """发布安全状态。"""
        self.safety_state = status
        self.safety_state_pub.publish(String(data=status))

    def publish_land_status(self, status):
        """发布降落/上锁状态，状态变化时打印日志。"""
        self.land_status = status
        self.land_status_pub.publish(String(data=status))

        if status != self.last_land_status:
            rospy.logwarn("Landing status: %s", status)
            self.last_land_status = status

    def is_landed(self):
        """判断 PX4 是否已经确认落地。"""
        return (
            self.extended_state_ok and
            self.extended_state.landed_state == ExtendedState.LANDED_STATE_ON_GROUND
        )

    def force_wait_reset(self, reason):
        """进入 WAIT_RESET，禁止自动恢复旧任务。"""
        rospy.logerr("WAIT_RESET forced: %s", reason)
        self.cancel_task_outputs()
        self.clear_task_runtime(reset_wp=True)
        self.start_requested = False
        self.reset_required = True
        self.emergency_reason = reason
        self.disarm_after_land = False
        self.publish_safety_state("WAIT_RESET_FORCED")
        self.publish_land_status("WAIT_RESET")
        self.enter_fsm_state("WAIT_RESET")

    def request_emergency_land(self, reason):
        """急停降落：取消任务、清除目标、请求 AUTO.LAND，落地后自动上锁。"""
        if self.fsm_state in ["EMERGENCY_LAND", "DISARMING"]:
            return

        rospy.logerr("EMERGENCY LAND requested: %s", reason)
        self.cancel_task_outputs()
        self.clear_task_runtime(reset_wp=True)
        self.start_requested = False
        self.reset_required = True
        self.emergency_reason = reason
        self.disarm_after_land = True
        self.publish_safety_state("EMERGENCY_LAND")
        self.publish_land_status("EMERGENCY_LAND_REQUESTED")
        self.setup_landing_prepare("emergency")
        self.enter_fsm_state("EMERGENCY_LAND")

    def request_normal_land(self, reason):
        """普通降落：取消任务、请求 AUTO.LAND，落地后自动上锁。"""
        if self.fsm_state in ["LANDING", "EMERGENCY_LAND", "DISARMING"]:
            return

        rospy.logwarn("Normal landing requested: %s", reason)
        self.cancel_task_outputs()
        self.clear_task_runtime(reset_wp=True)
        self.start_requested = False
        self.reset_required = True
        self.land_reason = reason
        self.disarm_after_land = True
        self.publish_safety_state("LANDING")
        self.publish_land_status("LAND_REQUESTED")
        self.setup_landing_prepare("normal")
        self.enter_fsm_state("LANDING")

    # =========================
    # 08. OFFBOARD / LAND / DISARM 辅助区
    # =========================

    def try_set_offboard_and_arm(self):
        """根据参数决定是否自动切 OFFBOARD 和自动解锁。"""
        if not self.start_requested or self.reset_required:
            return

        if self.fsm_state not in ["WAIT_FCU", "TAKEOFF", "MISSION"]:
            return

        now = rospy.Time.now()

        if (now - self.last_mode_req).to_sec() < 2.0:
            return

        self.last_mode_req = now

        if self.auto_set_mode and self.current_state.mode != "OFFBOARD":
            try:
                response = self.set_mode_client(base_mode=0, custom_mode="OFFBOARD")
                if hasattr(response, "mode_sent") and not response.mode_sent:
                    rospy.logwarn("OFFBOARD request rejected.")
                else:
                    rospy.loginfo("Request OFFBOARD mode.")
            except rospy.ServiceException as e:
                rospy.logwarn("Set OFFBOARD failed: %s", str(e))

        if self.auto_arm and not self.current_state.armed:
            try:
                response = self.arming_client(True)
                if hasattr(response, "success") and not response.success:
                    rospy.logwarn("Arm request rejected.")
                else:
                    rospy.loginfo("Request arm.")
            except rospy.ServiceException as e:
                rospy.logwarn("Arm failed: %s", str(e))

    def try_set_auto_land(self):
        """请求 PX4 进入 AUTO.LAND 模式。"""
        if not self.current_state.armed:
            self.publish_land_status("DISARMED")
            self.clear_landing_prepare()
            self.enter_fsm_state("WAIT_RESET")
            return

        if self.current_state.mode == "AUTO.LAND":
            self.publish_land_status("WAIT_LANDED")
            return

        now = rospy.Time.now()

        if (now - self.last_land_req).to_sec() < 1.0:
            return

        try:
            response = self.set_mode_client(base_mode=0, custom_mode="AUTO.LAND")

            if hasattr(response, "mode_sent") and response.mode_sent:
                self.publish_land_status("AUTO_LAND_SENT")
                rospy.logwarn("AUTO.LAND mode requested.")
            else:
                self.publish_land_status("AUTO_LAND_REJECTED")
                rospy.logwarn("AUTO.LAND request rejected, will retry.")
        except rospy.ServiceException as e:
            self.publish_land_status("AUTO_LAND_FAILED")
            rospy.logwarn("Set AUTO.LAND failed: %s", str(e))

        self.last_land_req = now

    def try_disarm_after_landed(self):
        """等待 PX4 确认落地后，再调用 /mavros/cmd/arming false。"""
        if not self.current_state.armed:
            self.publish_land_status("DISARMED")
            self.reset_required = True
            self.start_requested = False
            self.disarm_after_land = False
            self.clear_landing_prepare()
            self.enter_fsm_state("WAIT_RESET")
            return

        if not self.is_landed():
            if not self.extended_state_ok:
                self.publish_land_status("WAIT_EXTENDED_STATE")
            else:
                self.publish_land_status("WAIT_LANDED")
            return

        now = rospy.Time.now()

        if (now - self.last_land_req).to_sec() < 1.0:
            return

        try:
            response = self.arming_client(False)

            if hasattr(response, "success") and response.success:
                self.publish_land_status("DISARMED")
                self.reset_required = True
                self.start_requested = False
                self.disarm_after_land = False
                self.clear_landing_prepare()
                self.enter_fsm_state("WAIT_RESET")
                rospy.logwarn("Disarm requested after PX4 confirmed landed.")
            else:
                self.publish_land_status("DISARM_DENIED")
                rospy.logwarn("Disarm denied, will retry after landed check.")
        except rospy.ServiceException as e:
            self.publish_land_status("DISARM_FAILED")
            rospy.logwarn("Disarm failed: %s", str(e))

        self.last_land_req = now

    def handle_landing(self):
        """统一处理 LANDING / EMERGENCY_LAND / DISARMING。"""
        self.cancel_task_outputs()

        if not self.current_state.armed:
            self.publish_land_status("DISARMED")
            self.reset_required = True
            self.clear_landing_prepare()
            self.enter_fsm_state("WAIT_RESET")
            return

        if not self.is_landed():
            if self.current_state.mode != "AUTO.LAND" and not self.landing_prepare_ready():
                return

            self.publish_landing_prepare_setpoint()
            self.try_set_auto_land()
            return

        self.publish_land_status("LANDED")
        self.try_disarm_after_landed()

    # =========================
    # 09. 起飞控制区
    # =========================

    def process_takeoff(self):
        """起飞到指定高度，并确认达到可继续执行任务的安全悬停状态。"""
        target_x = self.home_x
        target_y = self.home_y
        target_z = self.home_z + self.takeoff_height
        target_yaw = math.radians(self.takeoff_yaw_deg)

        self.publish_position_velocity_yaw(
            target_x,
            target_y,
            target_z,
            0.0,
            0.0,
            0.0,
            target_yaw
        )

        cx, cy, cz = self.current_xyz()

        xy_err = norm3(target_x - cx, target_y - cy, 0.0)
        z_err = abs(target_z - cz)
        pos_err = norm3(target_x - cx, target_y - cy, target_z - cz)
        height = cz - self.home_z
        takeoff_elapsed = (rospy.Time.now() - self.phase_start_time).to_sec()

        height_ok = height >= self.takeoff_min_height
        pos_ready = pos_err < self.takeoff_pos_eps
        speed_ready = self.current_speed < self.takeoff_stable_vel
        acc_ready = self.current_acc_norm < self.takeoff_stable_acc

        strict_stable = height_ok and pos_ready and speed_ready and acc_ready

        # 实机可能因为定位/速度抖动长期过不了严格 acc 门槛。
        # 只要高度安全、位置没有明显偏离、速度已经很低，超过 max_wait 后允许进入任务。
        soft_ready = (
            height_ok and
            xy_err < max(0.25, self.takeoff_pos_eps * 1.5) and
            z_err < max(0.22, self.takeoff_pos_eps * 1.5) and
            self.current_speed < max(0.25, self.takeoff_stable_vel * 1.3)
        )

        if strict_stable:
            if self.stable_start_time is None:
                self.stable_start_time = rospy.Time.now()

            stable_time = (rospy.Time.now() - self.stable_start_time).to_sec()

            if stable_time > self.takeoff_stable_time:
                self.enter_fsm_state("MISSION")
                self.enter_nav_phase("INIT")
        else:
            self.stable_start_time = None

            if takeoff_elapsed > self.takeoff_max_wait and soft_ready:
                rospy.logwarn(
                    "[TAKEOFF] strict stable timeout, continue by soft gate: "
                    "pos=%.2f xy=%.2f zerr=%.2f height=%.2f vel=%.2f acc=%.2f",
                    pos_err,
                    xy_err,
                    z_err,
                    height,
                    self.current_speed,
                    self.current_acc_norm
                )
                self.enter_fsm_state("MISSION")
                self.enter_nav_phase("INIT")

        rospy.loginfo_throttle(
            0.5,
            "[TAKEOFF] pos=%.2f xy=%.2f zerr=%.2f h=%.2f vel=%.2f acc=%.2f strict=%s soft=%s",
            pos_err,
            xy_err,
            z_err,
            height,
            self.current_speed,
            self.current_acc_norm,
            str(strict_stable),
            str(soft_ready)
        )

    # =========================
    # 10. 航点导航控制区
    # =========================

    def is_obstacle_fast_pass_waypoint(self, name):
        """
        判断是否为绕障中间连续通过点。
        只让 OBS_ENTRY / OBS_EXIT 保留刹停，其余 OBS_* route 点默认不断停。
        """
        name_upper = str(name).upper()

        if not name_upper.startswith("OBS"):
            return False

        return name_upper not in ["OBS_ENTRY", "OBS_EXIT"]

    def get_slow_radius(self, wp):
        """
        根据航点名称和控制模式选择减速半径。
        变量 slow_radius 的作用：当剩余距离小于该半径时，速度开始按距离线性下降。
        """
        name = wp.name.upper()

        if self.is_obstacle_fast_pass_waypoint(name):
            return self.obs_fast_slow_radius

        if name.startswith("RING"):
            return self.ring_slow_radius

        if name.startswith("OBS"):
            return self.obs_slow_radius

        if name.startswith("LAND"):
            return self.land_slow_radius

        if wp.kind == "scan" or name.startswith("IMG") or name.startswith("SPECIAL"):
            return self.scan_slow_radius

        return self.route_slow_radius

    def get_min_speed(self, wp):
        """
        根据航点类型选择接近目标点时的最低靠近速度。
        注意：一旦进入到点误差范围，会直接给 0 速度，不会继续保持最低速度向前冲。
        """
        name = wp.name.upper()

        if name.startswith("RING"):
            return self.ring_min_speed

        if wp.kind == "scan":
            return self.scan_min_speed

        return self.route_min_speed

    def calc_smooth_cmd_vel(self, wp, dx, dy, dz, dist, dt, arrive_eps):
        """
        计算平滑靠近目标点的速度指令。
        核心逻辑：
        1. 距离大于 slow_radius 时，按 wp.speed 正常飞；
        2. 距离小于 slow_radius 时，速度按 dist / slow_radius 线性下降；
        3. 距离进入 arrive_eps 后，目标速度直接变为 0，交给位置环和刹停阶段稳定；
        4. 最后用 wp.acc * dt 限制每一帧速度变化，避免指令突变。
        """
        if dist < 1e-6 or dist <= arrive_eps:
            desired_vel = [0.0, 0.0, 0.0]
        else:
            slow_radius = max(self.get_slow_radius(wp), arrive_eps + 0.05)
            max_speed = max(0.05, wp.speed)
            min_speed = clamp(self.get_min_speed(wp), 0.02, max_speed)

            if dist < slow_radius:
                target_speed = max_speed * dist / slow_radius
                target_speed = clamp(target_speed, min_speed, max_speed)
            else:
                target_speed = max_speed

            ux = dx / dist
            uy = dy / dist
            uz = dz / dist
            desired_vel = [ux * target_speed, uy * target_speed, uz * target_speed]

        max_delta = max(0.001, wp.acc * dt)
        self.cmd_vel = limit_vector_change(self.cmd_vel, desired_vel, max_delta)
        return self.cmd_vel

    def get_arrive_eps(self, wp):
        """
        根据航点类型选择真正的到点误差阈值。
        scan/drop 点需要更准，因此使用 scan_pos_eps；普通 route 点使用 route_pos_eps。
        """
        if wp.kind == "scan" or wp.control_mode == "fusion":
            return self.scan_pos_eps

        return self.route_pos_eps

    def get_capture_radius(self, wp):
        """
        速度控制切换到位置锁点的捕获半径。
        作用：远处只发速度，避免 PX4 位置环被最终目标强拉；近处再交给位置控制锁点。
        """
        name = wp.name.upper()

        if wp.control_mode == "fusion_route" or name.startswith("RING"):
            return max(self.ring_capture_radius, self.route_pos_eps)

        if wp.kind == "scan" or wp.control_mode == "fusion":
            return max(self.scan_capture_radius, self.scan_pos_eps)

        return max(self.route_capture_radius, self.route_pos_eps)

    def publish_velocity_approach(self, wp, dx, dy, dz, dist, dt, arrive_eps):
        """
        MOVE 阶段的纯速度靠近控制。
        关键点：这里只发布 velocity + yaw，不发布最终 position，避免 position + velocity 前馈导致冲过头。
        """
        self.calc_smooth_cmd_vel(wp, dx, dy, dz, dist, dt, arrive_eps)
        self.publish_velocity_yaw(
            self.cmd_vel[0],
            self.cmd_vel[1],
            self.cmd_vel[2],
            self.locked_yaw
        )
        return norm3(self.cmd_vel[0], self.cmd_vel[1], self.cmd_vel[2])

    def route_need_brake(self, wp):
        """
        判断普通 route 到点后是否需要先刹停再切下一个点。
        position 航点通常是绕障/回航转折点，默认刹停；fusion_route 用于圆环等连续通过点，默认不断停。
        """
        if wp.action != "none":
            return True

        return wp.control_mode == "position"

    def process_current_waypoint(self, dt):
        """处理当前航点，包括 yaw 对齐、移动、刹停、停稳、动作。"""
        if self.wp_index >= len(self.waypoints):
            self.enter_fsm_state("STAGE1_DONE")
            return

        wp = self.waypoints[self.wp_index]

        if self.should_skip_waypoint_before_nav(wp):
            self.next_waypoint()
            return

        tx, ty, tz, target_yaw = self.get_abs_wp(wp)
        cx, cy, cz = self.current_xyz()

        dx = tx - cx
        dy = ty - cy
        dz = tz - cz
        dist = norm3(dx, dy, dz)

        if self.nav_phase == "INIT":
            # 2026-05-23 修改：取消“默认朝向目标点/运动方向”的 yaw 策略。
            # 所有航点统一使用 YAML 中的 yaw_deg，避免普通点、绕障点、圆环点频繁转头。
            self.locked_yaw = target_yaw

            # 2026-06-06 提速修改：
            # 对于 action=none 的 fusion_route 快速通过点，如果当前 yaw 已经满足要求，
            # 直接进入 MOVE，不再先原地 YAW_ALIGN 停车。
            # 这样绕障中间点、下排快速通过点、图片靶 OVER/UP 等点才能真正连续通过。
            yaw_err = abs(wrap_pi(self.locked_yaw - self.current_yaw))
            can_skip_yaw_align = (
                wp.action == "none" and
                (
                    wp.control_mode == "fusion_route" or
                    wp.name.upper() == "RING_POST"
                ) and
                yaw_err < self.skip_yaw_align_eps
            )

            if can_skip_yaw_align:
                rospy.loginfo(
                    "[SKIP_YAW_ALIGN] wp=%s mode=%s yaw_err=%.1f deg<th=%.1f deg",
                    wp.name,
                    wp.control_mode,
                    math.degrees(yaw_err),
                    math.degrees(self.skip_yaw_align_eps)
                )
                self.enter_nav_phase("MOVE")
            else:
                self.enter_nav_phase("YAW_ALIGN")
            return

        if self.nav_phase == "YAW_ALIGN":
            self.publish_velocity_yaw(0.0, 0.0, 0.0, self.locked_yaw)

            yaw_ready = self.is_yaw_aligned(self.locked_yaw)
            # 每个航点都会经过 YAW_ALIGN。默认只看 yaw + 低速，
            # 不再让速度差分得到的 acc 把每个航点开始前都卡住。
            yaw_acc_th = self.route_finish_acc if self.yaw_align_use_acc_gate else None
            stable_ready = self.is_speed_acc_stable(
                self.route_finish_vel,
                yaw_acc_th
            )

            yaw_elapsed = (rospy.Time.now() - self.phase_start_time).to_sec()
            soft_yaw_ready = (
                yaw_ready and
                yaw_elapsed > self.yaw_align_soft_wait and
                self.current_speed < self.yaw_align_soft_vel
            )

            if yaw_ready and (stable_ready or soft_yaw_ready):
                if soft_yaw_ready and not stable_ready:
                    rospy.logwarn(
                        "[YAW_ALIGN] soft pass wp=%s vel=%.2f acc=%.2f",
                        wp.name,
                        self.current_speed,
                        self.current_acc_norm
                    )
                self.enter_nav_phase("MOVE")

            rospy.loginfo_throttle(
                0.5,
                "[YAW_ALIGN] wp=%s yaw_err=%.1f deg vel=%.2f acc=%.2f stable=%s soft=%s mode=%s",
                wp.name,
                math.degrees(abs(wrap_pi(self.locked_yaw - self.current_yaw))),
                self.current_speed,
                self.current_acc_norm,
                str(stable_ready),
                str(soft_yaw_ready),
                wp.control_mode
            )

            return

        if self.nav_phase == "MOVE":
            yaw_err = abs(wrap_pi(self.locked_yaw - self.current_yaw))

            if yaw_err > self.yaw_break_eps:
                self.enter_nav_phase("YAW_ALIGN")
                return

            if wp.control_mode == "position":
                self.process_route_move(wp, tx, ty, tz, dist, dx, dy, dz, dt)
                return

            if wp.control_mode == "fusion_route":
                self.process_fusion_route_move(wp, tx, ty, tz, dist, dx, dy, dz, dt)
                return

            if wp.control_mode == "fusion":
                self.process_scan_move(wp, tx, ty, tz, dist, dx, dy, dz, dt)
                return

        if self.nav_phase == "BRAKE":
            # 2026-05-25 修改：BRAKE 不再只发零速度。
            # 只发零速度可能会让飞机停在目标点后面；这里改成锁定目标位置 + 0 速度。
            self.publish_position_velocity_yaw(
                tx,
                ty,
                tz,
                0.0,
                0.0,
                0.0,
                self.locked_yaw
            )

            arrive_eps = self.get_arrive_eps(wp)

            if wp.kind == "scan" or wp.control_mode == "fusion":
                vel_th = self.scan_stable_vel
                acc_th = self.scan_stable_acc if self.action_use_acc_gate else None
            else:
                vel_th = self.route_finish_vel
                # 普通路线点只要求“到点 + 低速”，不再用加速度硬卡。
                acc_th = None

            pos_ready = dist < arrive_eps
            stable_ready = pos_ready and self.is_speed_acc_stable(vel_th, acc_th)

            brake_elapsed = (rospy.Time.now() - self.phase_start_time).to_sec()
            soft_route_ready = (
                wp.action == "none" and
                brake_elapsed > self.brake_timeout and
                dist < max(arrive_eps * 1.5, arrive_eps + 0.08) and
                self.current_speed < self.brake_soft_vel
            )

            # 对 scan/drop/ring 等带动作的点，不能只因为某一帧满足位置/速度阈值就开始 HOLD/ACTION。
            # 否则视觉刚进画面、机体还在轻微漂移时，image_class_timeout 等计时就已经开始了。
            # 这里要求连续稳定 scan_stable_time 秒后，才进入 HOLD，再由 hold_time 进入 ACTION。
            stable_duration = 0.0
            stable_enough_for_action = False

            if stable_ready:
                if self.stable_start_time is None:
                    self.stable_start_time = rospy.Time.now()

                stable_duration = (rospy.Time.now() - self.stable_start_time).to_sec()

                if wp.action == "none":
                    # 普通无动作航点仍然保持原来的快速通过/到点逻辑，不额外等 scan_stable_time。
                    stable_enough_for_action = True
                else:
                    stable_enough_for_action = stable_duration >= self.scan_stable_time
            else:
                self.stable_start_time = None

            if stable_enough_for_action or soft_route_ready:
                if soft_route_ready and not stable_ready:
                    rospy.logwarn(
                        "[BRAKE] soft pass wp=%s dist=%.2f vel=%.2f acc=%.2f",
                        wp.name,
                        dist,
                        self.current_speed,
                        self.current_acc_norm
                    )

                if wp.action == "none":
                    self.next_waypoint()
                else:
                    rospy.loginfo(
                        "[BRAKE_STABLE_DONE] wp=%s stable_for=%.2f/%.2f, enter HOLD then ACTION.",
                        wp.name,
                        stable_duration,
                        self.scan_stable_time
                    )
                    self.enter_nav_phase("HOLD")

            rospy.loginfo_throttle(
                0.5,
                "[POSITION_LOCK_BRAKE] wp=%s dist=%.2f eps=%.2f vel=%.2f acc=%.2f pos_ok=%s stable=%s stable_for=%.2f/%.2f soft=%s",
                wp.name,
                dist,
                arrive_eps,
                self.current_speed,
                self.current_acc_norm,
                str(pos_ready),
                str(stable_ready),
                stable_duration,
                self.scan_stable_time,
                str(soft_route_ready)
            )

            return

        if self.nav_phase == "HOLD":
            self.publish_position_velocity_yaw(
                tx,
                ty,
                tz,
                0.0,
                0.0,
                0.0,
                self.locked_yaw
            )

            hold_time = (rospy.Time.now() - self.phase_start_time).to_sec()

            if hold_time > wp.hold_time:
                self.enter_nav_phase("ACTION")

            return

        if self.nav_phase == "ACTION":
            self.process_action(wp, tx, ty, tz)
            return

    def process_route_move(self, wp, tx, ty, tz, dist, dx, dy, dz, dt):
        """
        普通 route 点移动。
        2026-05-25 修改：MOVE 阶段改为纯速度控制；进入捕获半径后切 BRAKE 位置锁点。
        """
        arrive_eps = self.get_arrive_eps(wp)
        capture_radius = self.get_capture_radius(wp)

        if dist <= capture_radius:
            self.enter_nav_phase("BRAKE")
            return

        cmd_speed = self.publish_velocity_approach(
            wp,
            dx,
            dy,
            dz,
            dist,
            dt,
            arrive_eps
        )

        rospy.loginfo_throttle(
            0.5,
            "[ROUTE_VEL_APPROACH] wp=%s dist=%.2f capture=%.2f cmd_v=%.2f real_v=%.2f acc=%.2f slow_r=%.2f",
            wp.name,
            dist,
            capture_radius,
            cmd_speed,
            self.current_speed,
            self.current_acc_norm,
            self.get_slow_radius(wp)
        )

    def process_fusion_route_move(self, wp, tx, ty, tz, pos_err, dx, dy, dz, dt):
        """
        复杂通过点控制，例如圆环前后直线通过。
        2026-05-25 修改：默认用纯速度通过，不再发送最终位置 + 速度前馈。
        """
        arrive_eps = self.get_arrive_eps(wp)
        capture_radius = self.get_capture_radius(wp)

        if pos_err <= arrive_eps:
            if self.route_need_brake(wp):
                self.enter_nav_phase("BRAKE")
            elif wp.action == "none":
                self.next_waypoint()
            else:
                self.enter_nav_phase("BRAKE")
            return

        # fusion_route 用于圆环这类连续通过点：不提前切位置锁点，只用速度控制靠近/通过。
        # 如果未来某个 fusion_route 需要停稳，只要给它配置 action 或改成 fusion 即可。
        cmd_speed = self.publish_velocity_approach(
            wp,
            dx,
            dy,
            dz,
            pos_err,
            dt,
            arrive_eps
        )

        rospy.loginfo_throttle(
            0.5,
            "[FUSION_ROUTE_VEL] wp=%s err=%.2f eps=%.2f cmd_v=%.2f real_v=%.2f slow_r=%.2f",
            wp.name,
            pos_err,
            arrive_eps,
            cmd_speed,
            self.current_speed,
            self.get_slow_radius(wp)
        )

    def process_scan_move(self, wp, tx, ty, tz, pos_err, dx, dy, dz, dt):
        """
        scan/drop 点移动。
        2026-05-25 修改：远处只发速度；接近扫描点后切 BRAKE，由位置锁点等待真实停稳。
        """
        arrive_eps = self.get_arrive_eps(wp)
        capture_radius = self.get_capture_radius(wp)

        if pos_err <= capture_radius:
            self.enter_nav_phase("BRAKE")
            return

        cmd_speed = self.publish_velocity_approach(
            wp,
            dx,
            dy,
            dz,
            pos_err,
            dt,
            arrive_eps
        )

        rospy.loginfo_throttle(
            0.5,
            "[SCAN_VEL_APPROACH] wp=%s err=%.2f capture=%.2f cmd_v=%.2f real_v=%.2f acc=%.2f slow_r=%.2f",
            wp.name,
            pos_err,
            capture_radius,
            cmd_speed,
            self.current_speed,
            self.current_acc_norm,
            self.get_slow_radius(wp)
        )

    # =========================
    # 11. 动作执行区
    # =========================

    def action_requires_static_hover(self, wp):
        """
        判断当前 ACTION 是否必须锁死在原航点静态悬停。
        图片靶 image_scan_maybe_drop 和特殊靶 special_drop 现在需要视觉闭环小范围修正，
        因此不再放进这个静态列表。
        """
        return wp.action in [
            "qr_scan",
            "image_drop_1",
            "image_drop_2",
        ]

    def check_action_static_stable(self, tx, ty, tz):
        """
        ACTION 阶段的持续停稳检查。
        返回 stable 和当前位置误差，供静态扫描/投放动作使用。
        """
        cx, cy, cz = self.current_xyz()
        pos_err = norm3(tx - cx, ty - cy, tz - cz)

        stable = (
            pos_err < self.scan_pos_eps and
            self.current_speed < self.scan_stable_vel
        )

        if self.action_use_acc_gate:
            stable = stable and self.current_acc_norm < self.scan_stable_acc

        return stable, pos_err

    def get_action_hold_target(self, tx, ty, tz):
        """读取 ACTION 阶段真正的悬停目标点；如果没有动态目标，则使用当前航点。"""
        if self.action_hold_target is None:
            self.action_hold_target = [tx, ty, tz]

        return (
            self.action_hold_target[0],
            self.action_hold_target[1],
            self.action_hold_target[2]
        )

    def get_latest_vision_result(self, target, min_confidence=None):
        """读取指定 target 的最新视觉结果，并检查超时、detected 和置信度。"""
        if target not in self.vision_results:
            return None, "no_result"

        data = self.vision_results[target]
        recv_time = data.get("_recv_time", None)

        if recv_time is None:
            return None, "no_recv_time"

        age = (rospy.Time.now() - recv_time).to_sec()

        if age > self.vision_result_timeout:
            return None, "result_timeout"

        detected_raw = data.get("detected", False)
        if isinstance(detected_raw, str):
            detected = detected_raw.strip().lower() in ["true", "1", "yes"]
        else:
            detected = bool(detected_raw)

        if not detected:
            return None, str(data.get("reason", "not_detected"))

        try:
            confidence = float(data.get("confidence", 1.0))
        except Exception:
            return None, "invalid_confidence"

        if min_confidence is None:
            min_confidence = self.align_min_confidence

        if confidence < min_confidence:
            return None, "low_confidence"

        return data, "ok"

    def get_vision_offset_xy(self, data):
        """从视觉结果中读取米制偏差。FSM 不使用像素偏差做控制。"""
        try:
            offset_x = float(data.get("offset_x_m"))
            offset_y = float(data.get("offset_y_m"))
            return True, offset_x, offset_y
        except Exception:
            return False, 0.0, 0.0

    def body_xy_to_local_xy(self, forward, left):
        """
        将机体系偏差转换到 MAVROS local 平面坐标。
        forward > 0 表示目标在机头前方；left > 0 表示目标在机体左侧。
        """
        c = math.cos(self.current_yaw)
        s = math.sin(self.current_yaw)

        dx = c * forward - s * left
        dy = s * forward + c * left

        return dx, dy

    def update_action_hold_target_by_vision(self, target_z, offset_x_m, offset_y_m):
        """
        根据视觉节点给出的米制偏差更新 ACTION 阶段悬停点。
        视觉偏差定义为：靶心相对投放点的位置。无人机应朝同方向小范围移动。
        """
        cx, cy, _ = self.current_xyz()
        dx_local, dy_local = self.body_xy_to_local_xy(offset_x_m, offset_y_m)

        dx_local, dy_local, _ = limit_vector_norm(
            dx_local,
            dy_local,
            0.0,
            self.align_step_max
        )

        self.action_hold_target = [
            cx + self.align_gain * dx_local,
            cy + self.align_gain * dy_local,
            target_z
        ]


    def update_action_hold_target_by_vision_bounded(self, base_x, base_y, target_z, offset_x_m, offset_y_m):
        """
        固定图片靶快速微调专用：根据视觉偏差更新悬停点，但限制最大偏移量。
        即使视觉方向或尺度有误差，也不会把无人机从已知靶心附近拉走太远。
        """
        cx, cy, _ = self.current_xyz()
        dx_local, dy_local = self.body_xy_to_local_xy(offset_x_m, offset_y_m)

        dx_local, dy_local, _ = limit_vector_norm(
            dx_local,
            dy_local,
            0.0,
            self.align_step_max
        )

        cand_x = cx + self.align_gain * dx_local
        cand_y = cy + self.align_gain * dy_local

        rel_x = cand_x - base_x
        rel_y = cand_y - base_y
        rel_norm = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        max_shift = max(0.0, self.fixed_image_align_max_shift)

        if max_shift > 1e-6 and rel_norm > max_shift:
            scale = max_shift / rel_norm
            cand_x = base_x + rel_x * scale
            cand_y = base_y + rel_y * scale

        self.action_hold_target = [cand_x, cand_y, target_z]

    def update_action_hold_target_by_vision_limited(self, base_x, base_y, target_z, offset_x_m, offset_y_m, max_shift):
        """
        通用限幅视觉微调：围绕 base_x/base_y 更新 ACTION 悬停点，并限制最大偏移。
        用于高空扫描锁定和低空锁点附近微调，避免视觉尺度错误把飞机拉离靶心。
        """
        cx, cy, _ = self.current_xyz()
        dx_local, dy_local = self.body_xy_to_local_xy(offset_x_m, offset_y_m)

        dx_local, dy_local, _ = limit_vector_norm(
            dx_local,
            dy_local,
            0.0,
            self.align_step_max
        )

        cand_x = cx + self.align_gain * dx_local
        cand_y = cy + self.align_gain * dy_local

        rel_x = cand_x - base_x
        rel_y = cand_y - base_y
        rel_norm = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        max_shift = max(0.0, float(max_shift))

        if max_shift > 1e-6 and rel_norm > max_shift:
            scale = max_shift / rel_norm
            cand_x = base_x + rel_x * scale
            cand_y = base_y + rel_y * scale

        self.action_hold_target = [cand_x, cand_y, target_z]

    def update_image_scan_aim_from_vision(self, wp, tx, ty, tz, data):
        """
        高空单独瞄准阶段根据视觉偏差修正图片靶 XY。
        返回：(aim_ready, offset_norm, reason)。aim_ready=True 表示 XY 偏差已经足够小，或未启用高空瞄准。
        """
        if not self.image_scan_aim_enabled:
            return True, 0.0, "disabled"

        if data is None:
            return False, 999.0, "no_vision"

        ok_offset, offset_x, offset_y = self.get_vision_offset_xy(data)
        if not ok_offset:
            return False, 999.0, "bad_offset"

        offset_norm = math.sqrt(offset_x * offset_x + offset_y * offset_y)

        if offset_norm > self.image_scan_aim_xy_eps:
            # 高空阶段围绕 YAML 靶心点限幅修正，逐步把下视相机中心移到图片靶圆心附近。
            self.update_action_hold_target_by_vision_limited(
                tx,
                ty,
                tz,
                offset_x,
                offset_y,
                self.image_scan_aim_max_shift
            )
            rospy.loginfo_throttle(
                0.4,
                "[IMAGE_SCAN_AIM] wp=%s offset=(%.3f, %.3f) norm=%.3f/%.3f target=(%.2f, %.2f, %.2f)",
                wp.name,
                offset_x,
                offset_y,
                offset_norm,
                self.image_scan_aim_xy_eps,
                self.action_hold_target[0],
                self.action_hold_target[1],
                self.action_hold_target[2]
            )
            return False, offset_norm, "aiming"

        return True, offset_norm, "aim_ready"

    def image_locked_force_drop_ready(self, wp, tx, ty, drop_abs_z, reason):
        """
        图片靶低空视觉丢失兜底：如果已锁定 XY，且飞机已稳定在锁点附近，则直接投放。
        用于解决低空相机离地太近，看不全图片靶圆环导致无法继续识别的问题。
        """
        lock_x, lock_y = self.get_image_locked_xy_or_default(tx, ty)
        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(lock_x, lock_y, drop_abs_z)
        cx, cy, cz = self.current_xyz()
        pos_err = norm3(hold_tx - cx, hold_ty - cy, hold_tz - cz)

        ready = (
            pos_err < self.image_low_force_pos_eps and
            self.current_speed < self.image_low_force_vel and
            self.pending_drop_cmd != ""
        )

        if ready:
            rospy.logwarn(
                "[IMAGE_LOCKED_FORCE_DROP] wp=%s cmd=%s reason=%s locked_xy=(%.3f, %.3f) pos_err=%.3f vel=%.3f",
                wp.name,
                self.pending_drop_cmd,
                reason,
                lock_x,
                lock_y,
                pos_err,
                self.current_speed
            )
            if self.send_drop_once(self.pending_drop_cmd):
                self.mark_image_drop_done(self.pending_drop_cmd)
            self.image_action_phase = "DROP_WAIT"
            return True

        rospy.loginfo_throttle(
            0.5,
            "[IMAGE_LOCKED_FORCE_WAIT] wp=%s reason=%s pos_err=%.3f/%.3f vel=%.3f/%.3f",
            wp.name,
            reason,
            pos_err,
            self.image_low_force_pos_eps,
            self.current_speed,
            self.image_low_force_vel
        )
        return False


    def get_ring_vision_offset(self, data):
        """
        读取圆环视觉结果。
        约定：
        forward_m > 0：圆环中心在当前机头前方多少米；也兼容 distance_m / range_m 字段。
        offset_y_m > 0：圆环中心在当前机体左侧。
        offset_z_m > 0：圆环中心在当前无人机上方。
        """
        try:
            if "forward_m" in data:
                forward_m = float(data.get("forward_m"))
            elif "distance_m" in data:
                forward_m = float(data.get("distance_m"))
            else:
                forward_m = float(data.get("range_m"))

            offset_y_m = float(data.get("offset_y_m"))
            offset_z_m = float(data.get("offset_z_m"))
        except Exception:
            return False, 0.0, 0.0, 0.0, "invalid_ring_offset"

        if forward_m < self.ring_forward_min or forward_m > self.ring_forward_max:
            return False, forward_m, offset_y_m, offset_z_m, "ring_forward_out_of_range"

        return True, forward_m, offset_y_m, offset_z_m, "ok"

    def get_ring_align_error_local(self, offset_y_m, offset_z_m):
        """
        把前视相机看到的圆环左右/高度偏差，换算成本机 local 坐标下的对准误差。

        这里加入 ring_align_bias_x / ring_align_bias_z，用来补偿前视相机识别中心
        和机体中心不重合的问题。
        - 未加偏置时：目标是让视觉 offset_y_m=0、offset_z_m=0。
        - 加偏置后：目标是让“视觉偏差 + 安装偏置补偿”接近 0。

        例如当前任务 yaw≈-90° 时，机体左右方向基本对应 local X：
        bias_x > 0 会让穿环轨迹整体向 local +X 偏；
        bias_z > 0 会让穿环轨迹整体升高。
        """
        dx_local, dy_local = self.body_xy_to_local_xy(0.0, offset_y_m)

        # X/Z 偏置只修正穿环平面内的横向和高度；
        # 不额外修正 local Y，避免在 RING_PRE 阶段追前后距离、越调越靠近圆环。
        err_x = dx_local + self.ring_align_bias_x
        err_y = dy_local
        err_z = offset_z_m + self.ring_align_bias_z

        return err_x, err_y, err_z

    def build_dynamic_ring_points_from_vision(self, forward_m, offset_y_m, offset_z_m, update_pre=True):
        """
        根据 ring_gate 视觉结果动态生成穿环三点。
        视觉结果是当前机体系；生成结果是 MAVROS local 绝对坐标。
        """
        cx, cy, cz = self.current_xyz()

        # 当前机体系 forward/left 转 local x/y。
        dx_local, dy_local = self.body_xy_to_local_xy(forward_m, offset_y_m)

        # 视觉测到的是“相机识别中心”相对圆环中心的偏差。
        # 前视相机不一定装在机体中心，因此这里允许用 X/Z 偏置把动态穿环轨迹整体微调。
        center_x = cx + dx_local + self.ring_align_bias_x
        center_y = cy + dy_local
        center_z = clamp(cz + offset_z_m + self.ring_align_bias_z, self.ring_min_z, self.ring_max_z)

        # 穿越方向按当前锁定 yaw 的机头方向生成，避免视觉轻微抖动影响穿越直线方向。
        pass_yaw = self.locked_yaw
        ux = math.cos(pass_yaw)
        uy = math.sin(pass_yaw)

        pre_x = center_x - ux * self.ring_pre_distance
        pre_y = center_y - uy * self.ring_pre_distance
        post_x = center_x + ux * self.ring_post_distance
        post_y = center_y + uy * self.ring_post_distance

        # 穿环后接续点：到圆环后方 post_distance 后，保持同一 y，沿 x 方向朝最终降落点移动 ring_exit_x_distance。
        # 这样避免 RING_POST 后立刻被固定 LAND_TOP_CORRIDOR 拉回，造成“穿过头又倒退”。
        exit_x = post_x
        exit_y = post_y
        if self.ring_exit_x_distance > 1e-6:
            land_abs_x, _ = self.get_selected_land_abs_xy()
            x_sign = -1.0 if post_x >= land_abs_x else 1.0
            exit_x = post_x + x_sign * self.ring_exit_x_distance

        if update_pre:
            self.dynamic_ring_points["RING_PRE"] = (pre_x, pre_y, center_z)

        self.dynamic_ring_points["RING_CENTER"] = (center_x, center_y, center_z)
        self.dynamic_ring_points["RING_POST"] = (post_x, post_y, center_z)
        self.dynamic_ring_points["RING_EXIT_X"] = (exit_x, exit_y, center_z)

        self.ring_dynamic_ready = True
        self.ring_last_debug = {
            "forward_m": forward_m,
            "offset_y_m": offset_y_m,
            "offset_z_m": offset_z_m,
            "ring_align_bias_x": self.ring_align_bias_x,
            "ring_align_bias_z": self.ring_align_bias_z,
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "pre_x": pre_x,
            "pre_y": pre_y,
            "post_x": post_x,
            "post_y": post_y,
            "exit_x": exit_x,
            "exit_y": exit_y,
        }

        rospy.loginfo(
            "Ring dynamic points updated: PRE=(%.2f, %.2f, %.2f) CENTER=(%.2f, %.2f, %.2f) POST=(%.2f, %.2f, %.2f) EXIT_X=(%.2f, %.2f, %.2f)",
            pre_x,
            pre_y,
            center_z,
            center_x,
            center_y,
            center_z,
            post_x,
            post_y,
            center_z,
            exit_x,
            exit_y,
            center_z
        )

    def update_ring_pre_align_target(self, offset_y_m, offset_z_m):
        """
        RING_PRE 二次对准：只根据横向和高度偏差修正，不追 forward_m，避免靠圆环过近。
        这里使用带 X/Z 安装偏置补偿后的 local 误差。
        """
        cx, cy, cz = self.current_xyz()
        err_x, err_y, err_z = self.get_ring_align_error_local(offset_y_m, offset_z_m)

        err_x, err_y, err_z = limit_vector_norm(
            err_x,
            err_y,
            err_z,
            self.ring_align_step_max
        )

        self.action_hold_target = [
            cx + self.ring_align_gain * err_x,
            cy + self.ring_align_gain * err_y,
            clamp(cz + self.ring_align_gain * err_z, self.ring_min_z, self.ring_max_z)
        ]

    def get_circle_loop_count(self):
        """读取并限制圆形绕障圈数。"""
        try:
            loop_count = int(self.obstacle_cfg.get("loop_count", 1))
        except Exception:
            loop_count = 1

        try:
            max_loop_count = int(self.obstacle_cfg.get("max_loop_count", 8))
        except Exception:
            max_loop_count = 8

        return int(clamp(loop_count, 1, max(1, max_loop_count)))

    def get_circle_direction_sign(self):
        """返回圆形绕障方向。当前代码坐标系下，clockwise 使用 theta 递减。"""
        if self.circle_direction in ["ccw", "counterclockwise", "anticlockwise", "逆时针"]:
            return 1.0

        return -1.0

    def get_circle_center_abs(self):
        """把 YAML 中的绕障圆心相对坐标转换成 MAVROS local 绝对坐标。"""
        if self.frame_mode == "relative_home":
            return self.home_x + self.circle_center_x, self.home_y + self.circle_center_y

        return self.circle_center_x, self.circle_center_y

    def circle_trapezoid_profile(self, elapsed, omega, ramp_time, total_angle):
        """
        生成平滑绕圆角度进度。
        返回：phi, phi_dot, phi_ddot, total_time, finished。
        phi 为已经走过的角度长度，单位 rad，始终为正；方向在外部乘 direction_sign。
        """
        omega = max(1e-4, abs(omega))
        ramp_time = max(0.0, ramp_time)
        total_angle = max(0.0, total_angle)

        if total_angle <= 1e-6:
            return 0.0, 0.0, 0.0, 0.0, True

        # 如果圈数很小或 ramp_time 太长，自动降级成三角速度曲线，避免恒速段为负。
        if ramp_time <= 1e-3 or total_angle <= omega * ramp_time:
            peak_omega = math.sqrt(total_angle * omega / max(ramp_time, total_angle / omega))
            peak_omega = max(1e-4, min(omega, peak_omega))
            accel_time = total_angle / peak_omega
            total_time = 2.0 * accel_time
            alpha = peak_omega / accel_time

            if elapsed < accel_time:
                phi = 0.5 * alpha * elapsed * elapsed
                phi_dot = alpha * elapsed
                phi_ddot = alpha
                return phi, phi_dot, phi_ddot, total_time, False

            if elapsed < total_time:
                tau = elapsed - accel_time
                phi = 0.5 * total_angle + peak_omega * tau - 0.5 * alpha * tau * tau
                phi_dot = max(0.0, peak_omega - alpha * tau)
                phi_ddot = -alpha
                return phi, phi_dot, phi_ddot, total_time, False

            return total_angle, 0.0, 0.0, total_time, True

        alpha = omega / ramp_time
        const_time = total_angle / omega - ramp_time
        total_time = 2.0 * ramp_time + const_time

        if elapsed < ramp_time:
            phi = 0.5 * alpha * elapsed * elapsed
            phi_dot = alpha * elapsed
            phi_ddot = alpha
            return phi, phi_dot, phi_ddot, total_time, False

        if elapsed < ramp_time + const_time:
            tau = elapsed - ramp_time
            phi = 0.5 * omega * ramp_time + omega * tau
            phi_dot = omega
            phi_ddot = 0.0
            return phi, phi_dot, phi_ddot, total_time, False

        if elapsed < total_time:
            tau = elapsed - ramp_time - const_time
            phi_before_down = 0.5 * omega * ramp_time + omega * const_time
            phi = phi_before_down + omega * tau - 0.5 * alpha * tau * tau
            phi_dot = max(0.0, omega - alpha * tau)
            phi_ddot = -alpha
            return phi, phi_dot, phi_ddot, total_time, False

        return total_angle, 0.0, 0.0, total_time, True

    def calc_circle_setpoint(self, elapsed):
        """根据当前绕圆时间计算位置、速度前馈、加速度前馈和调试信息。"""
        cx, cy = self.get_circle_center_abs()
        cz = self.mission_z_to_abs(self.circle_height)

        radius = max(0.10, self.circle_radius)
        speed = max(0.05, self.circle_speed)
        omega = speed / radius
        loop_count = self.get_circle_loop_count()
        total_angle = 2.0 * math.pi * loop_count
        direction_sign = self.get_circle_direction_sign()
        theta_start = math.radians(self.circle_start_angle_deg)

        phi, phi_dot_abs, phi_ddot_abs, total_time, finished = self.circle_trapezoid_profile(
            elapsed,
            omega,
            self.circle_ramp_time,
            total_angle
        )

        theta = theta_start + direction_sign * phi
        theta_dot = direction_sign * phi_dot_abs
        theta_ddot = direction_sign * phi_ddot_abs

        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        z = cz

        vx = -radius * math.sin(theta) * theta_dot
        vy = radius * math.cos(theta) * theta_dot
        vz = 0.0

        ax = -radius * math.cos(theta) * theta_dot * theta_dot - radius * math.sin(theta) * theta_ddot
        ay = -radius * math.sin(theta) * theta_dot * theta_dot + radius * math.cos(theta) * theta_ddot
        az = 0.0

        if self.circle_yaw_mode in ["tangent", "follow", "切线"] and abs(phi_dot_abs) > 1e-3:
            yaw = math.atan2(vy, vx)
        else:
            yaw = self.locked_yaw

        debug = {
            "theta": theta,
            "theta_dot": theta_dot,
            "theta_ddot": theta_ddot,
            "omega": omega,
            "loop_count": loop_count,
            "total_time": total_time,
            "finished": finished,
            "center_x": cx,
            "center_y": cy,
            "center_z": cz,
        }

        return x, y, z, vx, vy, vz, ax, ay, az, yaw, debug

    def process_circle_obstacle_action(self, wp):
        """执行圆形绕障：连续发布位置 + 速度前馈 + 加速度前馈。"""
        if self.action_start_time is None:
            self.action_start_time = rospy.Time.now()

        now = rospy.Time.now()
        elapsed = (now - self.action_start_time).to_sec()

        x, y, z, vx, vy, vz, ax, ay, az, yaw, debug = self.calc_circle_setpoint(elapsed)

        self.publish_position_velocity_accel_yaw(
            x, y, z,
            vx, vy, vz,
            ax, ay, az,
            yaw
        )

        if not debug["finished"]:
            # 实飞排查用：同时打印实际位置到圆心的半径，判断是目标圆有问题，还是飞机实际跟踪切内圈/外扩。
            cx_now, cy_now, _ = self.current_xyz()
            actual_r = math.sqrt(
                (cx_now - debug["center_x"]) * (cx_now - debug["center_x"]) +
                (cy_now - debug["center_y"]) * (cy_now - debug["center_y"])
            )
            radial_err = actual_r - self.circle_radius

            rospy.loginfo_throttle(
                0.5,
                "[CIRCLE_OBS] wp=%s t=%.2f/%.2f loop=%d center=(%.2f,%.2f) r_cmd=%.2f r_actual=%.2f radial_err=%.2f v=%.2f theta=%.1fdeg theta_dot=%.2f pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f) acc=(%.2f,%.2f)",
                wp.name,
                elapsed,
                debug["total_time"],
                debug["loop_count"],
                debug["center_x"],
                debug["center_y"],
                self.circle_radius,
                actual_r,
                radial_err,
                self.circle_speed,
                math.degrees(debug["theta"]),
                debug["theta_dot"],
                x, y, z,
                vx, vy,
                ax, ay
            )
            self.circle_finish_start_time = None
            self.circle_finish_stable_start_time = None
            return

        # 轨迹结束后，锁住圆起点/终点并确认速度降下来，再进入下一个航点，避免退出瞬间横向窜动。
        cx_now, cy_now, cz_now = self.current_xyz()
        pos_err = norm3(x - cx_now, y - cy_now, z - cz_now)
        stable = (
            pos_err < self.circle_finish_pos_eps and
            self.current_speed < self.circle_finish_vel_th
        )

        if self.circle_finish_start_time is None:
            self.circle_finish_start_time = now
            self.circle_finish_stable_start_time = None

        finish_wait = (now - self.circle_finish_start_time).to_sec()

        if stable:
            if self.circle_finish_stable_start_time is None:
                self.circle_finish_stable_start_time = now

            stable_time = (now - self.circle_finish_stable_start_time).to_sec()
        else:
            self.circle_finish_stable_start_time = None
            stable_time = 0.0

        # 严格退出：位置和速度都满足，并且连续稳定达到指定时间。
        strict_finish_ready = stable_time >= self.circle_finish_stable_time

        # 软退出：只在等待超过 finish_max_wait 后启用，且仍必须满足一个更宽松的位置/速度条件。
        # 注意：这里故意不能写成“只要超时就退出”，否则可能还没真正回到圆终点就切下一个航点。
        soft_finish_ready = (
            finish_wait >= self.circle_finish_max_wait and
            pos_err < self.circle_finish_soft_pos_eps and
            self.current_speed < self.circle_finish_soft_vel_th
        )

        if strict_finish_ready or soft_finish_ready:
            if soft_finish_ready and not strict_finish_ready:
                rospy.logwarn(
                    "[CIRCLE_OBS_FINISH_SOFT] wp=%s pos_err=%.2f<%.2f speed=%.2f<%.2f wait=%.2f/%.2f",
                    wp.name,
                    pos_err,
                    self.circle_finish_soft_pos_eps,
                    self.current_speed,
                    self.circle_finish_soft_vel_th,
                    finish_wait,
                    self.circle_finish_max_wait
                )
            else:
                rospy.loginfo(
                    "[CIRCLE_OBS_FINISH] wp=%s pos_err=%.2f speed=%.2f stable=%.2f",
                    wp.name,
                    pos_err,
                    self.current_speed,
                    stable_time
                )

            self.circle_finish_start_time = None
            self.circle_finish_stable_start_time = None
            self.disable_scan_request()
            self.next_waypoint()
            return

        rospy.loginfo_throttle(
            0.5,
            "[CIRCLE_OBS_FINISH_WAIT] wp=%s pos_err=%.2f strict_eps=%.2f soft_eps=%.2f speed=%.2f strict_th=%.2f soft_th=%.2f stable=%.2f/%.2f wait=%.2f/%.2f soft=%s",
            wp.name,
            pos_err,
            self.circle_finish_pos_eps,
            self.circle_finish_soft_pos_eps,
            self.current_speed,
            self.circle_finish_vel_th,
            self.circle_finish_soft_vel_th,
            stable_time,
            self.circle_finish_stable_time,
            finish_wait,
            self.circle_finish_max_wait,
            str(soft_finish_ready)
        )

    def send_drop_once(self, drop_cmd):
        """只发送一次投放命令，并记录投放开始时间。"""
        if self.action_sent:
            return False

        self.drop_cmd_pub.publish(String(data=drop_cmd))
        self.action_sent = True
        self.drop_sent_time = rospy.Time.now()
        rospy.loginfo("Drop command sent: %s", drop_cmd)
        return True

    def drop_wait_finished(self):
        """投放命令发出后，等待机构动作和额外保持时间结束。"""
        if self.drop_sent_time is None:
            return False

        return (rospy.Time.now() - self.drop_sent_time).to_sec() > (
            self.drop_time + self.hold_after_action
        )

    def disable_scan_request(self):
        """统一关闭视觉请求。"""
        self.scan_enable_pub.publish(Bool(data=False))
        self.scan_target_pub.publish(String(data="none"))

    def process_action(self, wp, tx, ty, tz):
        """
        到达 scan 点后的动作执行。
        二维码仍然使用静态悬停；图片靶和特殊靶使用视觉节点给出的米制偏差做小范围对准。
        """
        now = rospy.Time.now()

        if self.action_start_time is None:
            self.action_start_time = now
            self.action_stable_start_time = None
            self.action_sent = False
            self.drop_sent_time = None
            self.action_hold_target = [tx, ty, tz]

            if wp.action in [
                "image_scan_maybe_drop",
                "image_drop_1_align",
                "image_drop_2_align",
                "image_drop_1_align_manual_wait",
                "image_drop_2_align_manual_wait",
            ]:
                self.current_image_class = ""
                self.vision_results.pop("image_target", None)

            if wp.action == "image_scan_maybe_drop":
                self.reset_image_action_runtime()
                self.image_action_phase = "CLASS_SCAN"
                self.image_phase_start_time = now

            if wp.action == "special_drop":
                self.vision_results.pop("special_target", None)

            if wp.action == "ring_search":
                self.vision_results.pop("ring_gate", None)
                self.dynamic_ring_points = {}
                self.ring_dynamic_ready = False
                self.ring_last_debug = {}

            if wp.action == "ring_pre_align":
                self.vision_results.pop("ring_gate", None)

            if wp.action == "circle_obstacle":
                self.circle_finish_start_time = None
                self.circle_finish_stable_start_time = None

        # 圆形绕障是连续轨迹控制，必须保持 setpoint 流干净。
        # 注意：这里不能像二维码/图片靶/特殊靶动作那样，先发布“当前位置悬停、速度为 0”的 hold setpoint，
        # 再发布圆轨迹 setpoint。否则飞控会在同一个 ACTION 周期内先收到“停住”，又收到“绕圆”，
        # 位置/速度/加速度前馈会被静止目标污染，实飞容易出现绕圆半径收缩、跟踪不干净。
        if wp.action == "circle_obstacle":
            self.process_circle_obstacle_action(wp)
            return

        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(tx, ty, tz)

        # 其他 ACTION 阶段才发布当前悬停目标。视觉对准时 action_hold_target 会在原航点附近更新。
        self.publish_position_velocity_yaw(
            hold_tx,
            hold_ty,
            hold_tz,
            0.0,
            0.0,
            0.0,
            self.locked_yaw
        )

        if self.action_requires_static_hover(wp):
            stable, pos_err = self.check_action_static_stable(hold_tx, hold_ty, hold_tz)
            action_wait = (now - self.action_start_time).to_sec()
            soft_stable = (
                (not stable) and
                action_wait > self.action_soft_timeout and
                pos_err < self.action_soft_pos_eps and
                self.current_speed < self.action_soft_vel
            )

            if stable or soft_stable:
                if self.action_stable_start_time is None:
                    self.action_stable_start_time = now

                if soft_stable:
                    rospy.logwarn_throttle(
                        0.8,
                        "[ACTION] soft stable wp=%s err=%.2f vel=%.2f acc=%.2f",
                        wp.name,
                        pos_err,
                        self.current_speed,
                        self.current_acc_norm
                    )

                # 静态识别/投放动作只累计“可接受稳定时间”。
                # 严格稳定优先；若长期被噪声打断，软稳定也能让动作计时继续推进。
                action_elapsed = (now - self.action_stable_start_time).to_sec()
            else:
                self.action_stable_start_time = None

                # 如果还没有发出投放命令，不稳定时先不开启识别有效计时，也不进入投放逻辑。
                if not self.action_sent:
                    self.disable_scan_request()

                    rospy.loginfo_throttle(
                        0.5,
                        "[ACTION_WAIT_STABLE] wp=%s err=%.2f vel=%.2f acc=%.2f soft=%s",
                        wp.name,
                        pos_err,
                        self.current_speed,
                        self.current_acc_norm,
                        str(soft_stable)
                    )
                    return

                # 投放命令已经发出后，不再重置流程，只继续保持悬停等待动作结束。
                action_elapsed = (now - self.action_start_time).to_sec()
        else:
            action_elapsed = (now - self.action_start_time).to_sec()

        if wp.action == "none":
            self.next_waypoint()
            return

        if wp.action == "qr_scan":
            self.do_qr_scan_action(wp, action_elapsed)
            return

        if wp.action == "image_drop_1_align_manual_wait":
            self.do_fixed_image_align_manual_wait_action(
                wp, tx, ty, tz, "image_drop_1", action_elapsed
            )
            return

        if wp.action == "image_drop_2_align_manual_wait":
            self.do_fixed_image_align_manual_wait_action(
                wp, tx, ty, tz, "image_drop_2", action_elapsed
            )
            return

        if wp.action == "image_drop_1_align":
            self.do_fixed_image_align_drop_action(wp, tx, ty, tz, "image_drop_1", action_elapsed)
            return

        if wp.action == "image_drop_2_align":
            self.do_fixed_image_align_drop_action(wp, tx, ty, tz, "image_drop_2", action_elapsed)
            return

        if wp.action == "image_drop_1":
            self.do_image_drop_action(wp, "image_drop_1", action_elapsed)
            return

        if wp.action == "image_drop_2":
            self.do_image_drop_action(wp, "image_drop_2", action_elapsed)
            return

        if wp.action == "image_scan_maybe_drop":
            self.do_image_scan_maybe_drop_action(wp, tx, ty, tz, action_elapsed)
            return

        if wp.action == "special_drop":
            self.do_special_align_drop_action(wp, tx, ty, tz, action_elapsed)
            return

        if wp.action == "ring_search":
            self.do_ring_search_action(wp, tx, ty, tz, action_elapsed)
            return

        if wp.action == "ring_pre_align":
            self.do_ring_pre_align_action(wp, tx, ty, tz, action_elapsed)
            return

        if wp.action == "ring_wait":
            self.do_ring_wait_action(wp, action_elapsed)
            return

        if wp.action == "mission_done":
            self.do_mission_done_action(wp, action_elapsed)
            return

        rospy.logwarn("Unknown action %s at waypoint %s, skip.", wp.action, wp.name)
        self.next_waypoint()

    def do_qr_scan_action(self, wp, elapsed):
        """二维码扫描动作接口。识别失败时按 YAML 默认二维码结果继续任务。"""
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="qr"))

        if self.wait_real_qr:
            qr_ok = self.qr_class_1 != "" and self.qr_class_2 != "" and self.qr_land_side != ""

            if qr_ok:
                rospy.loginfo("QR scan success: %s", self.qr_text)
                self.disable_scan_request()
                self.next_waypoint()
                return

            if elapsed > self.qr_scan_timeout:
                self.apply_default_qr_result("qr_scan_timeout")
                self.disable_scan_request()
                self.next_waypoint()
                return
        else:
            if elapsed > self.qr_scan_timeout:
                self.apply_default_qr_result("qr_placeholder_done")
                self.disable_scan_request()
                self.next_waypoint()
                return

        rospy.loginfo_throttle(
            0.5,
            "[ACTION_QR] wp=%s elapsed=%.1f qr=%s default=(%s,%s,%s)",
            wp.name,
            elapsed,
            self.qr_text,
            self.qr_default_class_1,
            self.qr_default_class_2,
            self.qr_default_land_side
        )

    def do_image_drop_action(self, wp, drop_name, elapsed):
        """图片靶识别 + 投放动作接口。"""
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="image_target"))

        if not self.action_sent and elapsed > self.image_scan_time:
            self.send_drop_once(drop_name)

        if self.action_sent and self.drop_wait_finished():
            self.disable_scan_request()
            self.next_waypoint()
            return

        rospy.loginfo_throttle(
            0.5,
            "[ACTION_IMAGE_DROP] wp=%s cmd=%s elapsed=%.1f sent=%s",
            wp.name,
            drop_name,
            elapsed,
            str(self.action_sent)
        )

    def do_image_scan_maybe_drop_action(self, wp, tx, ty, tz, elapsed):
        """
        图片靶正式视觉顺序逻辑：
        1. CLASS_SCAN：在高空只识别类别，不在这个阶段改 XY；
        2. HIGH_AIM：确认当前图片靶需要投放后，再单独用 offset_x_m / offset_y_m 高空瞄准并锁定 XY；
        3. DESCEND：关闭视觉请求，锁定 XY 下降到 image_drop_z，到位稳定后直接投放；
        4. DROP_WAIT：等待投放机构动作完成后进入下一个航点。
        """
        if self.image_action_phase == "":
            self.image_action_phase = "CLASS_SCAN"
            self.image_phase_start_time = rospy.Time.now()

        if self.image_phase_start_time is None:
            self.image_phase_start_time = rospy.Time.now()

        phase_elapsed = (rospy.Time.now() - self.image_phase_start_time).to_sec()
        drop_abs_z = self.get_image_drop_abs_z()

        # 只有高空识别和高空瞄准阶段打开下视视觉。
        # 下降和投放等待阶段关闭视觉，避免采集到下降过程照片，也避免低空识别结果干扰控制。
        if self.image_action_phase in ["CLASS_SCAN", "HIGH_AIM"]:
            self.scan_enable_pub.publish(Bool(data=True))
            self.scan_target_pub.publish(String(data="image_target"))
        else:
            self.disable_scan_request()

        # 不接真实视觉时保留原来的航线调试占位逻辑：前两个图片靶各投一次。
        if not self.wait_real_image:
            if self.action_sent:
                if self.drop_wait_finished():
                    self.disable_scan_request()
                    self.next_waypoint()
                return

            drop_cmd = ""
            if elapsed > self.image_scan_time:
                if self.image_drop_count == 0:
                    drop_cmd = "image_drop_1"
                elif self.image_drop_count == 1:
                    drop_cmd = "image_drop_2"
                else:
                    self.disable_scan_request()
                    self.next_waypoint()
                    return

            if drop_cmd != "":
                if self.send_drop_once(drop_cmd):
                    self.mark_image_drop_done(drop_cmd)
            return

        # 兜底保护：如果已经投完，ACTION 内也直接跳过；正常情况下会在进入导航前跳过。
        if self.target_class_1_done and self.target_class_2_done:
            rospy.loginfo("Both QR image targets already dropped, skip %s.", wp.name)
            self.disable_scan_request()
            self.next_waypoint()
            return

        if self.image_action_phase == "DROP_WAIT" or self.action_sent:
            self.disable_scan_request()
            if self.drop_wait_finished():
                self.disable_scan_request()
                self.next_waypoint()
                return

            rospy.loginfo_throttle(
                0.5,
                "[IMAGE_DROP_WAIT] wp=%s cmd=%s elapsed=%.1f",
                wp.name,
                self.pending_drop_cmd,
                elapsed
            )
            return

        if self.image_action_phase == "CLASS_SCAN":
            # 第一阶段：高空只做类别识别，先不根据视觉偏差改 XY。
            self.action_hold_target = [float(tx), float(ty), float(tz)]

            force_drop, force_drop_cmd, remaining_scans, remaining_drops = (
                self.should_force_image_drop_by_remaining_count()
            )

            data, reason = self.get_latest_vision_result(
                "image_target",
                min_confidence=self.image_class_min_confidence
            )

            def start_force_high_aim(force_reason):
                img_class = self.current_image_class.strip().lower()
                if data is not None:
                    img_class = str(data.get("class_name", img_class)).strip().lower()
                if img_class == "":
                    img_class = "remaining_equal_unknown"

                rospy.logwarn(
                    "[IMAGE_REMAINING_EQUAL_FORCE_TO_AIM] wp=%s remaining_scans=%d remaining_drops=%d cmd=%s class=%s reason=%s",
                    wp.name,
                    remaining_scans,
                    remaining_drops,
                    force_drop_cmd,
                    img_class,
                    force_reason
                )
                self.start_image_high_aim_for_drop(
                    wp,
                    tx,
                    ty,
                    tz,
                    force_drop_cmd,
                    img_class,
                    "remaining_equal_%s" % force_reason
                )

            if data is None:
                # 剩余数量兜底仍然保留，但也先给高空识别一个完整等待窗口。
                # 到达分类超时后，再进入高空瞄准，而不是直接下降。
                if force_drop and phase_elapsed > self.image_class_timeout:
                    start_force_high_aim("class_timeout_%s" % reason)
                    return

                if phase_elapsed > self.image_class_timeout:
                    rospy.logwarn(
                        "[IMAGE_CLASS_TIMEOUT] wp=%s reason=%s timeout=%.2f qr=(%s,%s), skip.",
                        wp.name,
                        reason,
                        self.image_class_timeout,
                        self.qr_class_1,
                        self.qr_class_2
                    )
                    self.disable_scan_request()
                    self.next_waypoint()
                    return

                rospy.loginfo_throttle(
                    0.5,
                    "[IMAGE_CLASS_WAIT] wp=%s reason=%s phase_elapsed=%.2f/%.2f qr=(%s,%s)",
                    wp.name,
                    reason,
                    phase_elapsed,
                    self.image_class_timeout,
                    self.qr_class_1,
                    self.qr_class_2
                )
                return

            img_class = str(data.get("class_name", self.current_image_class)).strip().lower()
            self.current_image_class = img_class
            stable_count = int(data.get("stable_count", 1))
            confidence = float(data.get("confidence", 1.0))

            if img_class == "":
                if force_drop and phase_elapsed > self.image_class_timeout:
                    start_force_high_aim("empty_class_timeout")
                    return

                if phase_elapsed > self.image_class_timeout:
                    rospy.logwarn("[IMAGE_CLASS_EMPTY] wp=%s timeout, skip.", wp.name)
                    self.disable_scan_request()
                    self.next_waypoint()
                else:
                    rospy.loginfo_throttle(0.5, "[IMAGE_CLASS_EMPTY] wp=%s wait.", wp.name)
                return

            if stable_count < self.image_class_min_stable_count:
                if force_drop and phase_elapsed > self.image_class_timeout:
                    start_force_high_aim("unstable_class_timeout")
                    return

                if phase_elapsed > self.image_class_timeout:
                    rospy.logwarn(
                        "[IMAGE_CLASS_UNSTABLE_TIMEOUT] wp=%s class=%s stable=%d/%d, skip.",
                        wp.name,
                        img_class,
                        stable_count,
                        self.image_class_min_stable_count
                    )
                    self.disable_scan_request()
                    self.next_waypoint()
                else:
                    rospy.loginfo_throttle(
                        0.5,
                        "[IMAGE_CLASS_UNSTABLE] wp=%s class=%s conf=%.2f stable=%d/%d",
                        wp.name,
                        img_class,
                        confidence,
                        stable_count,
                        self.image_class_min_stable_count
                    )
                return

            qr_class_1 = self.qr_class_1.strip().lower()
            qr_class_2 = self.qr_class_2.strip().lower()
            drop_cmd = ""

            if img_class == qr_class_1 and not self.target_class_1_done:
                drop_cmd = "image_drop_1"
            elif img_class == qr_class_2 and not self.target_class_2_done:
                drop_cmd = "image_drop_2"

            if drop_cmd != "":
                rospy.loginfo(
                    "[IMAGE_CLASS_MATCH_TO_AIM] wp=%s class=%s cmd=%s conf=%.2f stable=%d/%d",
                    wp.name,
                    img_class,
                    drop_cmd,
                    confidence,
                    stable_count,
                    self.image_class_min_stable_count
                )
                self.start_image_high_aim_for_drop(
                    wp,
                    tx,
                    ty,
                    tz,
                    drop_cmd,
                    img_class,
                    "class_match"
                )
                return

            if force_drop:
                # 识别到了非目标类别，但剩余扫描点数等于剩余待投物块数时，仍按兜底逻辑处理。
                # 这样不会因为分类误判把最后必须投的靶跳过，但仍然先进入高空瞄准。
                start_force_high_aim("class_not_match_but_remaining_equal")
                return

            if phase_elapsed >= self.image_scan_time:
                rospy.loginfo(
                    "[IMAGE_CLASS_SKIP] wp=%s class=%s not needed/already dropped. qr=(%s,%s)",
                    wp.name,
                    img_class,
                    self.qr_class_1,
                    self.qr_class_2
                )
                self.disable_scan_request()
                self.next_waypoint()
                return

            rospy.loginfo_throttle(
                0.5,
                "[IMAGE_CLASS_CONFIRM_SKIP] wp=%s class=%s wait %.2f/%.2f qr=(%s,%s)",
                wp.name,
                img_class,
                phase_elapsed,
                self.image_scan_time,
                self.qr_class_1,
                self.qr_class_2
            )
            return

        if self.image_action_phase == "HIGH_AIM":
            # 第二阶段：类别已经确认，现在才开始用视觉偏差高空瞄准。
            if self.action_hold_target is None:
                self.action_hold_target = [float(tx), float(ty), float(tz)]
            else:
                self.action_hold_target[2] = float(tz)

            data, reason = self.get_latest_vision_result(
                "image_target",
                min_confidence=self.image_scan_aim_min_confidence
            )

            aim_ready = False
            aim_norm = 999.0
            aim_reason = reason

            if not self.image_scan_aim_enabled:
                aim_ready = True
                aim_norm = 0.0
                aim_reason = "disabled"
            elif data is not None:
                aim_ready, aim_norm, aim_reason = self.update_image_scan_aim_from_vision(
                    wp,
                    tx,
                    ty,
                    tz,
                    data
                )

            aim_timeout = phase_elapsed > self.image_high_aim_timeout

            if aim_ready or (not self.image_scan_aim_required_for_drop) or aim_timeout:
                if not aim_ready and self.image_scan_aim_required_for_drop:
                    rospy.logwarn(
                        "[IMAGE_HIGH_AIM_TIMEOUT] wp=%s class=%s cmd=%s reason=%s aim_norm=%.3f timeout=%.2f, lock current XY and descend.",
                        wp.name,
                        self.pending_image_class,
                        self.pending_drop_cmd,
                        aim_reason,
                        aim_norm,
                        self.image_high_aim_timeout
                    )
                else:
                    rospy.loginfo(
                        "[IMAGE_HIGH_AIM_READY] wp=%s class=%s cmd=%s reason=%s aim_norm=%.3f, lock XY and descend.",
                        wp.name,
                        self.pending_image_class,
                        self.pending_drop_cmd,
                        aim_reason,
                        aim_norm
                    )

                self.lock_current_image_xy(tx, ty, "high_aim_%s" % aim_reason)
                self.start_image_descend_for_drop(
                    wp,
                    tx,
                    ty,
                    drop_abs_z,
                    self.pending_drop_cmd,
                    self.pending_image_class,
                    "high_aim_done"
                )
                return

            rospy.loginfo_throttle(
                0.5,
                "[IMAGE_HIGH_AIM_WAIT] wp=%s class=%s cmd=%s reason=%s aim_norm=%.3f/%.3f elapsed=%.2f/%.2f target=(%.2f, %.2f, %.2f)",
                wp.name,
                self.pending_image_class,
                self.pending_drop_cmd,
                aim_reason,
                aim_norm,
                self.image_scan_aim_xy_eps,
                phase_elapsed,
                self.image_high_aim_timeout,
                self.action_hold_target[0],
                self.action_hold_target[1],
                self.action_hold_target[2]
            )
            return

        if self.image_action_phase == "DESCEND":
            # 第三阶段：下降过程中不再识别/瞄准，只锁定高空瞄准得到的 XY 降高。
            self.disable_scan_request()
            lock_x, lock_y = self.get_image_locked_xy_or_default(tx, ty)
            self.action_hold_target = [lock_x, lock_y, drop_abs_z]
            cx, cy, cz = self.current_xyz()
            pos_err = norm3(lock_x - cx, lock_y - cy, drop_abs_z - cz)
            z_err = abs(drop_abs_z - cz)
            descend_ready = (
                z_err < self.image_descend_pos_eps and
                pos_err < max(self.image_descend_pos_eps * 1.8, self.scan_pos_eps) and
                self.current_speed < max(self.scan_stable_vel, 0.18)
            )

            timeout_force_ready = (
                phase_elapsed > self.image_descend_timeout and
                pos_err < self.image_low_force_pos_eps and
                self.current_speed < self.image_low_force_vel and
                self.pending_drop_cmd != ""
            )

            pre_drop_ready = descend_ready or timeout_force_ready

            if pre_drop_ready:
                if self.image_pre_drop_stable_start_time is None:
                    self.image_pre_drop_stable_start_time = rospy.Time.now()

                pre_drop_stable_elapsed = (
                    rospy.Time.now() - self.image_pre_drop_stable_start_time
                ).to_sec()

                if pre_drop_stable_elapsed >= self.image_pre_drop_stable_time:
                    if timeout_force_ready and not descend_ready:
                        rospy.logwarn(
                            "[IMAGE_DESCEND_TIMEOUT_FORCE_DROP] wp=%s cmd=%s locked_xy=(%.3f, %.3f) pos_err=%.2f z_err=%.2f speed=%.2f stable=%.2f/%.2f",
                            wp.name,
                            self.pending_drop_cmd,
                            lock_x,
                            lock_y,
                            pos_err,
                            z_err,
                            self.current_speed,
                            pre_drop_stable_elapsed,
                            self.image_pre_drop_stable_time
                        )
                    else:
                        rospy.loginfo(
                            "[IMAGE_DESCEND_DROP] wp=%s cmd=%s locked_xy=(%.3f, %.3f) pos_err=%.2f z_err=%.2f speed=%.2f stable=%.2f/%.2f",
                            wp.name,
                            self.pending_drop_cmd,
                            lock_x,
                            lock_y,
                            pos_err,
                            z_err,
                            self.current_speed,
                            pre_drop_stable_elapsed,
                            self.image_pre_drop_stable_time
                        )

                    if self.send_drop_once(self.pending_drop_cmd):
                        self.mark_image_drop_done(self.pending_drop_cmd)
                    self.image_action_phase = "DROP_WAIT"
                    self.image_pre_drop_stable_start_time = None
                    return

                rospy.loginfo_throttle(
                    0.3,
                    "[IMAGE_PRE_DROP_STABLE_WAIT] wp=%s cmd=%s locked_xy=(%.3f, %.3f) pos_err=%.2f z_err=%.2f speed=%.2f stable=%.2f/%.2f",
                    wp.name,
                    self.pending_drop_cmd,
                    lock_x,
                    lock_y,
                    pos_err,
                    z_err,
                    self.current_speed,
                    pre_drop_stable_elapsed,
                    self.image_pre_drop_stable_time
                )
                return

            # 中途高度/位置/速度任一条件不满足，就清零重新累计，保证是“连续稳定”。
            self.image_pre_drop_stable_start_time = None

            rospy.loginfo_throttle(
                0.4,
                "[IMAGE_DESCEND_WAIT] wp=%s cmd=%s locked_xy=(%.3f, %.3f) pos_err=%.2f z_err=%.2f speed=%.2f elapsed=%.2f/%.2f",
                wp.name,
                self.pending_drop_cmd,
                lock_x,
                lock_y,
                pos_err,
                z_err,
                self.current_speed,
                phase_elapsed,
                self.image_descend_timeout
            )
            return

        rospy.logwarn("[IMAGE_UNKNOWN_PHASE] wp=%s phase=%s, skip.", wp.name, self.image_action_phase)
        self.disable_scan_request()
        self.next_waypoint()

    def do_fixed_image_align_drop_action(self, wp, tx, ty, tz, drop_name, elapsed):
        """
        已知图片靶固定点 + 短时间视觉微调 + 超时兜底自动投放。
        该函数保留为自动投放备用；当前手控投放模式主要使用 *_manual_wait 动作。
        """
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="image_target"))

        if self.action_sent:
            if self.drop_wait_finished():
                self.disable_scan_request()
                self.next_waypoint()
                return

            rospy.loginfo_throttle(
                0.5,
                "[FIXED_IMAGE_DROP_WAIT] wp=%s cmd=%s elapsed=%.2f",
                wp.name,
                drop_name,
                elapsed
            )
            return

        if elapsed < self.fixed_image_align_min_wait:
            rospy.loginfo_throttle(
                0.3,
                "[FIXED_IMAGE_ALIGN_START] wp=%s cmd=%s elapsed=%.2f",
                wp.name,
                drop_name,
                elapsed
            )
            return

        align_timeout = elapsed >= self.fixed_image_align_timeout
        force_timeout = elapsed >= self.fixed_image_force_drop_timeout
        if align_timeout and (self.current_speed < self.fixed_image_fallback_vel or force_timeout):
            rospy.logwarn(
                "[FIXED_IMAGE_FALLBACK_DROP] wp=%s cmd=%s elapsed=%.2f speed=%.2f force=%s",
                wp.name,
                drop_name,
                elapsed,
                self.current_speed,
                str(force_timeout)
            )
            self.send_drop_once(drop_name)
            return

        data, reason = self.get_latest_vision_result(
            "image_target",
            min_confidence=self.fixed_image_align_min_confidence
        )

        if data is None:
            rospy.loginfo_throttle(
                0.4,
                "[FIXED_IMAGE_WAIT_VISION] wp=%s cmd=%s reason=%s elapsed=%.2f timeout=%.2f",
                wp.name,
                drop_name,
                reason,
                elapsed,
                self.fixed_image_align_timeout
            )
            return

        ok_offset, offset_x, offset_y = self.get_vision_offset_xy(data)
        if not ok_offset:
            rospy.logwarn_throttle(
                0.4,
                "[FIXED_IMAGE_BAD_OFFSET] wp=%s cmd=%s elapsed=%.2f",
                wp.name,
                drop_name,
                elapsed
            )
            return

        offset_norm = math.sqrt(offset_x * offset_x + offset_y * offset_y)

        if offset_norm > self.align_xy_eps:
            self.update_action_hold_target_by_vision_bounded(tx, ty, tz, offset_x, offset_y)
            rospy.loginfo_throttle(
                0.3,
                "[FIXED_IMAGE_ALIGN] wp=%s cmd=%s offset=(%.3f, %.3f) norm=%.3f elapsed=%.2f",
                wp.name,
                drop_name,
                offset_x,
                offset_y,
                offset_norm,
                elapsed
            )
            return

        stable_count = int(data.get("stable_count", 1))
        confidence = float(data.get("confidence", 1.0))
        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(tx, ty, tz)
        hold_stable, pos_err = self.check_action_static_stable(hold_tx, hold_ty, hold_tz)

        ready_to_drop = (
            confidence >= self.fixed_image_align_min_confidence and
            stable_count >= self.fixed_image_align_min_stable_count and
            (hold_stable or elapsed >= self.fixed_image_align_timeout)
        )

        if ready_to_drop:
            rospy.loginfo(
                "[FIXED_IMAGE_ALIGNED_DROP] wp=%s cmd=%s offset=%.3f conf=%.2f stable=%d pos_err=%.3f elapsed=%.2f",
                wp.name,
                drop_name,
                offset_norm,
                confidence,
                stable_count,
                pos_err,
                elapsed
            )
            self.send_drop_once(drop_name)
            return

        rospy.loginfo_throttle(
            0.3,
            "[FIXED_IMAGE_READY_WAIT] wp=%s cmd=%s offset=%.3f conf=%.2f stable=%d/%d pos_err=%.3f hold_stable=%s elapsed=%.2f",
            wp.name,
            drop_name,
            offset_norm,
            confidence,
            stable_count,
            self.fixed_image_align_min_stable_count,
            pos_err,
            str(hold_stable),
            elapsed
        )

    def start_manual_image_hover_once(self, wp, label):
        """进入手控投放等待；默认不自动发布 drop_cmd，只悬停等待人工投放。"""
        if self.action_sent:
            return

        self.action_sent = True
        self.drop_sent_time = rospy.Time.now()

        if not self.manual_image_no_auto_drop:
            # 备用模式：如临时决定仍由代码触发投放，可在 YAML 中设为 false。
            self.drop_cmd_pub.publish(String(data=label))
            rospy.logwarn(
                "[MANUAL_IMAGE_WAIT_START_WITH_CMD] wp=%s cmd=%s hover=%.2fs",
                wp.name,
                label,
                self.manual_image_hover_time
            )
        else:
            rospy.logwarn(
                "[MANUAL_IMAGE_WAIT_START] wp=%s label=%s hover=%.2fs no_auto_drop=True",
                wp.name,
                label,
                self.manual_image_hover_time
            )

    def manual_image_hover_finished(self):
        """判断手控投放等待是否结束。"""
        if self.drop_sent_time is None:
            return False

        elapsed = (rospy.Time.now() - self.drop_sent_time).to_sec()
        return elapsed >= self.manual_image_hover_time

    def do_fixed_image_align_manual_wait_action(self, wp, tx, ty, tz, label, elapsed):
        """
        已知图片靶固定点 + 短时间视觉微调 + 手控投放等待。
        视觉能帮忙就微调；视觉来不及或对不准，超时后也进入手控投放等待。
        """
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="image_target"))

        # action_sent 在这个动作里表示“已经进入手控投放等待阶段”。
        if self.action_sent:
            wait_elapsed = 0.0
            if self.drop_sent_time is not None:
                wait_elapsed = (rospy.Time.now() - self.drop_sent_time).to_sec()

            if self.manual_image_hover_finished():
                rospy.logwarn(
                    "[MANUAL_IMAGE_WAIT_DONE] wp=%s label=%s wait=%.2fs",
                    wp.name,
                    label,
                    wait_elapsed
                )
                self.disable_scan_request()
                self.next_waypoint()
                return

            rospy.logwarn_throttle(
                0.5,
                "[MANUAL_IMAGE_WAITING] wp=%s label=%s wait=%.2f/%.2fs",
                wp.name,
                label,
                wait_elapsed,
                self.manual_image_hover_time
            )
            return

        if elapsed < self.fixed_image_align_min_wait:
            rospy.loginfo_throttle(
                0.3,
                "[MANUAL_IMAGE_ALIGN_START] wp=%s label=%s elapsed=%.2f",
                wp.name,
                label,
                elapsed
            )
            return

        # 超时兜底：视觉来不及或仍未对准时，也进入手控投放等待。
        align_timeout = elapsed >= self.fixed_image_align_timeout
        force_timeout = elapsed >= self.fixed_image_force_drop_timeout
        if align_timeout and (self.current_speed < self.fixed_image_fallback_vel or force_timeout):
            rospy.logwarn(
                "[MANUAL_IMAGE_FALLBACK_WAIT] wp=%s label=%s elapsed=%.2f speed=%.2f force=%s",
                wp.name,
                label,
                elapsed,
                self.current_speed,
                str(force_timeout)
            )
            self.start_manual_image_hover_once(wp, label)
            return

        data, reason = self.get_latest_vision_result(
            "image_target",
            min_confidence=self.fixed_image_align_min_confidence
        )

        if data is None:
            rospy.loginfo_throttle(
                0.4,
                "[MANUAL_IMAGE_WAIT_VISION] wp=%s label=%s reason=%s elapsed=%.2f timeout=%.2f",
                wp.name,
                label,
                reason,
                elapsed,
                self.fixed_image_align_timeout
            )
            return

        ok_offset, offset_x, offset_y = self.get_vision_offset_xy(data)
        if not ok_offset:
            rospy.logwarn_throttle(
                0.4,
                "[MANUAL_IMAGE_BAD_OFFSET] wp=%s label=%s elapsed=%.2f",
                wp.name,
                label,
                elapsed
            )
            return

        offset_norm = math.sqrt(offset_x * offset_x + offset_y * offset_y)

        if offset_norm > self.align_xy_eps:
            self.update_action_hold_target_by_vision_bounded(tx, ty, tz, offset_x, offset_y)
            rospy.loginfo_throttle(
                0.3,
                "[MANUAL_IMAGE_ALIGN] wp=%s label=%s offset=(%.3f, %.3f) norm=%.3f target=(%.2f, %.2f, %.2f) elapsed=%.2f",
                wp.name,
                label,
                offset_x,
                offset_y,
                offset_norm,
                self.action_hold_target[0],
                self.action_hold_target[1],
                self.action_hold_target[2],
                elapsed
            )
            return

        stable_count = int(data.get("stable_count", 1))
        confidence = float(data.get("confidence", 1.0))
        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(tx, ty, tz)
        hold_stable, pos_err = self.check_action_static_stable(hold_tx, hold_ty, hold_tz)

        ready_to_wait = (
            confidence >= self.fixed_image_align_min_confidence and
            stable_count >= self.fixed_image_align_min_stable_count and
            (hold_stable or elapsed >= self.fixed_image_align_timeout)
        )

        if ready_to_wait:
            rospy.logwarn(
                "[MANUAL_IMAGE_ALIGNED_WAIT] wp=%s label=%s offset=%.3f conf=%.2f stable=%d pos_err=%.3f elapsed=%.2f hover=%.2fs",
                wp.name,
                label,
                offset_norm,
                confidence,
                stable_count,
                pos_err,
                elapsed,
                self.manual_image_hover_time
            )
            self.start_manual_image_hover_once(wp, label)
            return

        rospy.loginfo_throttle(
            0.3,
            "[MANUAL_IMAGE_READY_WAIT] wp=%s label=%s offset=%.3f conf=%.2f stable=%d/%d pos_err=%.3f hold_stable=%s elapsed=%.2f",
            wp.name,
            label,
            offset_norm,
            confidence,
            stable_count,
            self.fixed_image_align_min_stable_count,
            pos_err,
            str(hold_stable),
            elapsed
        )

    def special_force_drop_ready(self, wp, tx, ty, tz, elapsed, reason):
        """
        特殊靶兜底投放判定。
        特殊靶位置是已知航点，视觉只用于小范围瞄准；如果瞄准失败，
        只要飞机已经基本停在低空投放点附近，就强制投一次，避免完全丢掉特殊靶动作分。
        """
        if not self.special_force_drop_enabled:
            return False

        if elapsed < self.special_force_drop_timeout:
            return False

        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(tx, ty, tz)
        cx, cy, cz = self.current_xyz()
        pos_err = norm3(hold_tx - cx, hold_ty - cy, hold_tz - cz)

        ready = (
            pos_err < self.special_force_drop_pos_eps and
            self.current_speed < self.special_force_drop_vel
        )

        if ready:
            self.send_drop_once("special_drop")
            rospy.logwarn(
                "[SPECIAL_FORCE_DROP] wp=%s reason=%s elapsed=%.2f pos_err=%.3f vel=%.3f",
                wp.name,
                reason,
                elapsed,
                pos_err,
                self.current_speed
            )
            return True

        rospy.logwarn_throttle(
            0.5,
            "[SPECIAL_FORCE_WAIT] reason=%s elapsed=%.2f pos_err=%.3f/%.3f vel=%.3f/%.3f",
            reason,
            elapsed,
            pos_err,
            self.special_force_drop_pos_eps,
            self.current_speed,
            self.special_force_drop_vel
        )
        return False

    def do_special_align_drop_action(self, wp, tx, ty, tz, elapsed):
        """
        特殊靶瞄准 + 投放。
        说明：特殊靶不需要做类别识别，只需要视觉节点给出 special_target 的靶心偏差，
        FSM 根据 offset_x_m / offset_y_m 做小范围对准；若瞄准超时，则执行兜底强制投放。
        """
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="special_target"))

        if self.action_sent:
            if self.drop_wait_finished():
                self.disable_scan_request()
                self.next_waypoint()
                return

            rospy.loginfo_throttle(
                0.5,
                "[SPECIAL_DROP_WAIT] wp=%s elapsed=%.1f",
                wp.name,
                elapsed
            )
            return

        # 调航线时如果不接真实视觉，特殊靶保留旧的直接投放占位逻辑。
        if not self.wait_real_image:
            self.send_drop_once("special_drop")
            return

        data, reason = self.get_latest_vision_result(
            "special_target",
            min_confidence=self.special_min_confidence
        )

        if data is None:
            rospy.loginfo_throttle(
                0.5,
                "[SPECIAL_WAIT_VISION] wp=%s reason=%s elapsed=%.1f",
                wp.name,
                reason,
                elapsed
            )
            # 特殊靶是已知定点：即使视觉一直没给出有效偏差，超时后也允许在低速、近点条件下强制投放。
            if self.special_force_drop_allow_no_vision:
                self.special_force_drop_ready(wp, tx, ty, tz, elapsed, "no_vision_%s" % reason)
            return

        ok_offset, offset_x, offset_y = self.get_vision_offset_xy(data)

        if not ok_offset:
            rospy.logwarn_throttle(
                0.5,
                "Special target %s has no valid offset_x_m/offset_y_m.",
                wp.name
            )
            if self.special_force_drop_allow_no_vision:
                self.special_force_drop_ready(wp, tx, ty, tz, elapsed, "invalid_offset")
            return

        offset_norm = math.sqrt(offset_x * offset_x + offset_y * offset_y)
        stable_count = int(data.get("stable_count", 1))
        confidence = float(data.get("confidence", 1.0))

        if offset_norm > self.special_align_xy_eps:
            self.update_action_hold_target_by_vision(tz, offset_x, offset_y)

            rospy.loginfo_throttle(
                0.5,
                "[SPECIAL_ALIGN] wp=%s offset=(%.3f, %.3f) norm=%.3f/%.3f conf=%.2f stable=%d target=(%.2f, %.2f, %.2f)",
                wp.name,
                offset_x,
                offset_y,
                offset_norm,
                self.special_align_xy_eps,
                confidence,
                stable_count,
                self.action_hold_target[0],
                self.action_hold_target[1],
                self.action_hold_target[2]
            )
            # 视觉有偏差但迟迟对不准时，也允许到达兜底时间后强制投放。
            self.special_force_drop_ready(wp, tx, ty, tz, elapsed, "align_timeout")
            return

        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(tx, ty, tz)
        hold_stable, pos_err = self.check_action_static_stable(hold_tx, hold_ty, hold_tz)

        # 特殊靶条件比图片靶放宽：只要求靶心偏差进入范围、置信度/稳定帧达标，且飞机基本停稳。
        # 如果长时间因定位噪声达不到严格 hold_stable，则下面的强制投放兜底会处理。
        ready_to_drop = (
            confidence >= self.special_min_confidence and
            stable_count >= self.special_min_stable_count and
            hold_stable
        )

        if ready_to_drop:
            self.send_drop_once("special_drop")
            rospy.loginfo(
                "[SPECIAL_DROP] aligned wp=%s offset=%.3f conf=%.2f stable=%d pos_err=%.3f",
                wp.name,
                offset_norm,
                confidence,
                stable_count,
                pos_err
            )
            return

        if self.special_force_drop_ready(wp, tx, ty, tz, elapsed, "ready_wait_timeout"):
            return

        rospy.loginfo_throttle(
            0.5,
            "[SPECIAL_READY_WAIT] wp=%s offset=%.3f/%.3f conf=%.2f/%.2f stable=%d/%d pos_err=%.3f hold_stable=%s elapsed=%.2f",
            wp.name,
            offset_norm,
            self.special_align_xy_eps,
            confidence,
            self.special_min_confidence,
            stable_count,
            self.special_min_stable_count,
            pos_err,
            str(hold_stable),
            elapsed
        )

    def do_drop_only_action(self, wp, drop_name, elapsed):
        """保留旧接口备用；当前 special_drop 已改用 do_special_align_drop_action。"""
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="special_target"))

        if not self.action_sent:
            self.send_drop_once(drop_name)

        if self.drop_wait_finished():
            self.disable_scan_request()
            self.next_waypoint()
            return

        rospy.loginfo_throttle(
            0.5,
            "[ACTION_DROP_ONLY] wp=%s cmd=%s elapsed=%.1f",
            wp.name,
            drop_name,
            elapsed
        )


    def do_ring_search_action(self, wp, tx, ty, tz, elapsed):
        """
        圆环搜索锁定：在 RING_SEARCH_START 停稳后请求视觉节点识别 ring_gate，
        根据 forward_m / offset_y_m / offset_z_m 动态生成 RING_PRE / RING_CENTER / RING_POST。
        """
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="ring_gate"))

        if not self.wait_real_ring:
            if elapsed > wp.hold_time:
                rospy.logwarn("wait_real_ring=false, keep YAML fixed ring points.")
                self.disable_scan_request()
                self.next_waypoint()
            return

        data, reason = self.get_latest_vision_result("ring_gate", min_confidence=self.ring_min_confidence)

        if data is None:
            if elapsed > self.ring_search_timeout:
                rospy.logwarn(
                    "Ring search timeout at %s, policy=%s, reason=%s",
                    wp.name,
                    self.ring_timeout_policy,
                    reason
                )

                if self.ring_timeout_policy in ["fixed", "continue"]:
                    rospy.logwarn("Ring timeout: continue with YAML fixed ring points.")
                    self.disable_scan_request()
                    self.next_waypoint()
                    return

                if self.ring_timeout_policy == "skip":
                    rospy.logwarn("Ring timeout: skip to next waypoint with current route.")
                    self.disable_scan_request()
                    self.next_waypoint()
                    return

                self.publish_safety_state("RING_SEARCH_HOLD")
                rospy.logwarn_throttle(1.0, "Ring search timeout, holding and waiting for valid ring vision.")
                return

            rospy.loginfo_throttle(
                0.5,
                "[RING_SEARCH_WAIT] wp=%s reason=%s elapsed=%.1f",
                wp.name,
                reason,
                elapsed
            )
            return

        ok, forward_m, offset_y_m, offset_z_m, reason = self.get_ring_vision_offset(data)
        if not ok:
            rospy.logwarn_throttle(0.5, "Invalid ring vision: %s", reason)
            return

        stable_count = int(data.get("stable_count", 1))
        confidence = float(data.get("confidence", 1.0))

        if elapsed < self.ring_search_min_time:
            rospy.loginfo_throttle(
                0.5,
                "[RING_SEARCH_MIN_TIME_WAIT] forward=%.2f y=%.2f z=%.2f conf=%.2f elapsed=%.1f/%.1f",
                forward_m,
                offset_y_m,
                offset_z_m,
                confidence,
                elapsed,
                self.ring_search_min_time
            )
            return

        if stable_count < self.ring_search_min_stable_count:
            rospy.loginfo_throttle(
                0.5,
                "[RING_SEARCH_STABLE_WAIT] forward=%.2f y=%.2f z=%.2f conf=%.2f stable=%d/%d",
                forward_m,
                offset_y_m,
                offset_z_m,
                confidence,
                stable_count,
                self.ring_search_min_stable_count
            )
            return

        self.build_dynamic_ring_points_from_vision(forward_m, offset_y_m, offset_z_m, update_pre=True)
        self.disable_scan_request()
        self.next_waypoint()

    def do_ring_pre_align_action(self, wp, tx, ty, tz, elapsed):
        """
        RING_PRE 二次对准：到动态预穿越点后，继续看圆环，修正左右和高度；
        偏差足够小后直接进入穿环，不再用近距离视觉刷新 RING_CENTER / RING_POST。
        """
        self.scan_enable_pub.publish(Bool(data=True))
        self.scan_target_pub.publish(String(data="ring_gate"))

        if not self.wait_real_ring:
            if elapsed > wp.hold_time:
                self.disable_scan_request()
                self.next_waypoint()
            return

        if elapsed > self.ring_align_timeout:
            rospy.logwarn("Ring pre-align timeout at %s, continue through current dynamic/fixed points.", wp.name)
            self.disable_scan_request()
            self.next_waypoint()
            return

        data, reason = self.get_latest_vision_result("ring_gate", min_confidence=self.ring_min_confidence)

        if data is None:
            rospy.loginfo_throttle(
                0.5,
                "[RING_PRE_WAIT_VISION] wp=%s reason=%s elapsed=%.1f",
                wp.name,
                reason,
                elapsed
            )
            return

        ok, forward_m, offset_y_m, offset_z_m, reason = self.get_ring_vision_offset(data)
        if not ok:
            rospy.logwarn_throttle(0.5, "Invalid ring pre-align vision: %s", reason)
            return

        err_x, err_y, err_z = self.get_ring_align_error_local(offset_y_m, offset_z_m)
        align_err = norm3(err_x, err_y, err_z)

        if align_err > self.ring_yz_eps:
            self.update_ring_pre_align_target(offset_y_m, offset_z_m)
            rospy.loginfo_throttle(
                0.5,
                "[RING_PRE_ALIGN] raw_y=%.3f raw_z=%.3f bias_x=%.3f bias_z=%.3f "
                "err_local=(%.3f, %.3f, %.3f) err=%.3f target=(%.2f, %.2f, %.2f)",
                offset_y_m,
                offset_z_m,
                self.ring_align_bias_x,
                self.ring_align_bias_z,
                err_x,
                err_y,
                err_z,
                align_err,
                self.action_hold_target[0],
                self.action_hold_target[1],
                self.action_hold_target[2]
            )
            return

        stable_count = int(data.get("stable_count", 1))
        confidence = float(data.get("confidence", 1.0))
        hold_tx, hold_ty, hold_tz = self.get_action_hold_target(tx, ty, tz)
        hold_stable, pos_err = self.check_action_static_stable(hold_tx, hold_ty, hold_tz)

        ready_to_pass = (
            confidence >= self.ring_min_confidence and
            stable_count >= self.ring_min_stable_count and
            hold_stable
        )

        if ready_to_pass:
            # 只把 RING_PRE 当作二次确认点，不再用近距离视觉刷新 CENTER/POST/EXIT_X。
            # 原因：RING_PRE 离圆环更近，前视画面更容易出现圆环截断、识别框变形，
            # 若此时重新估算圆环中心，可能把已经在远距离锁定好的穿环线重新拉偏或拉近。
            # 因此穿环中心和后方余量沿用 RING_SEARCH_START 远距离生成的动态点；
            # 视觉失败时则沿用 YAML 固定兜底点。
            self.disable_scan_request()
            rospy.loginfo(
                "Ring pre-align ready: forward=%.2f raw_y=%.3f raw_z=%.3f "
                "bias_x=%.3f bias_z=%.3f err=%.3f. "
                "Start passing gate without refreshing CENTER/POST from near vision.",
                forward_m,
                offset_y_m,
                offset_z_m,
                self.ring_align_bias_x,
                self.ring_align_bias_z,
                align_err
            )
            self.next_waypoint()
            return

        rospy.loginfo_throttle(
            0.5,
            "[RING_PRE_READY_WAIT] raw_y=%.3f raw_z=%.3f bias_x=%.3f bias_z=%.3f err=%.3f conf=%.2f stable=%d/%d pos_err=%.3f hold_stable=%s",
            offset_y_m,
            offset_z_m,
            self.ring_align_bias_x,
            self.ring_align_bias_z,
            align_err,
            confidence,
            stable_count,
            self.ring_min_stable_count,
            pos_err,
            str(hold_stable)
        )

    def do_mission_done_action(self, wp, elapsed):
        """完整航线结束后在降落点上方短暂停稳，然后按配置自动降落。"""
        self.scan_enable_pub.publish(Bool(data=False))

        if elapsed > wp.hold_time:
            auto_land = bool(self.landing_cfg.get("auto_land_after_mission", True))

            if auto_land:
                rospy.loginfo("Mission route finished. Auto landing from above landing point.")
                self.request_normal_land("mission_done waypoint reached")
            else:
                rospy.loginfo("Mission route finished. Holding above landing point.")
                self.enter_fsm_state("STAGE1_DONE")
            return

        rospy.loginfo_throttle(
            0.5,
            "[MISSION_DONE] wp=%s elapsed=%.1f land_side=%s",
            wp.name,
            elapsed,
            self.get_selected_land_side()
        )

    def do_ring_wait_action(self, wp, elapsed):
        """
        随机圆环前等待。
        第一阶段到这里结束，不直接穿环。
        """
        self.scan_enable_pub.publish(Bool(data=False))

        if elapsed > wp.hold_time:
            rospy.loginfo("Arrived fixed ring waiting point. Stage1 done.")
            self.enter_fsm_state("STAGE1_DONE")
            return

        rospy.loginfo_throttle(
            0.5,
            "[RING_WAIT] wp=%s elapsed=%.1f",
            wp.name,
            elapsed
        )

    # =========================
    # 12. 主循环区
    # =========================

    def spin(self):
        """FSM 主循环。"""
        rate = rospy.Rate(self.rate_hz)
        last_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - last_time).to_sec()
            last_time = now
            dt = clamp(dt, 1.0 / 100.0, 1.0 / 10.0)

            self.publish_fsm_state()

            # 安全状态优先级最高：急停、普通降落、上锁流程永远先于任务控制。
            if self.fsm_state in ["EMERGENCY_LAND", "LANDING", "DISARMING"]:
                self.handle_landing()
                rate.sleep()
                continue

            if self.fsm_state == "WAIT_RESET":
                self.cancel_task_outputs()
                self.publish_neutral_setpoint()
                rospy.logwarn_throttle(2.0, "WAIT_RESET: publish /uav/reset=True after vehicle is disarmed.")
                rate.sleep()
                continue

            # 等待定位数据时不进入任务；但如果已经收到 stop/land，前面的安全分支会优先处理。
            if self.current_pose is None or not self.home_ready:
                rate.sleep()
                continue

            if self.fsm_state == "WAIT_START":
                self.cancel_task_outputs()
                self.publish_neutral_setpoint()
                rospy.loginfo_throttle(2.0, "WAIT_START: publish /uav/start=True to start mission.")

            elif self.fsm_state == "WAIT_FCU":
                self.try_set_offboard_and_arm()
                self.publish_position_velocity_yaw(
                    self.home_x,
                    self.home_y,
                    self.home_z + self.takeoff_height,
                    0.0,
                    0.0,
                    0.0,
                    math.radians(self.takeoff_yaw_deg)
                )

                if self.current_state.connected:
                    self.enter_fsm_state("TAKEOFF")

            elif self.fsm_state == "TAKEOFF":
                self.try_set_offboard_and_arm()
                self.process_takeoff()

            elif self.fsm_state == "MISSION":
                self.try_set_offboard_and_arm()
                self.process_current_waypoint(dt)

            elif self.fsm_state == "STAGE1_DONE":
                self.process_stage1_done()

            else:
                rospy.logwarn_throttle(1.0, "Unknown FSM state: %s", self.fsm_state)

            rate.sleep()

    def process_stage1_done(self):
        """第一阶段结束后保持在最后一个点，等待 /uav/land 或人工接管。"""
        if len(self.waypoints) == 0:
            self.publish_velocity_yaw(0.0, 0.0, 0.0, self.current_yaw)
            return

        last_wp = self.waypoints[-1]
        tx, ty, tz, yaw = self.get_abs_wp(last_wp)

        self.publish_position_velocity_yaw(
            tx,
            ty,
            tz,
            0.0,
            0.0,
            0.0,
            yaw
        )

        rospy.loginfo_throttle(
            2.0,
            "STAGE1_DONE: holding at %s. Publish /uav/land=True to land.",
            last_wp.name
        )


# =========================
# 13. 程序入口
# =========================

if __name__ == "__main__":
    try:
        node = MicroUAVStage1FSM()
        node.spin()
    except rospy.ROSInterruptException:
        pass
