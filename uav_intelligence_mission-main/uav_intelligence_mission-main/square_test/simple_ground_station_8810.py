#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简易 UDP 地面站：正方形投放/舵机安全测试版

默认发送到：
    UAV_IP   = 192.168.151.102
    UAV_PORT = 8810

功能：
1. 起飞/开始：takeoff 或 start -> CMD:START
2. 降落：land -> CMD:LAND
3. 急停降落：stop -> CMD:STOP
4. 复位/上锁：reset / disarm
5. 舵机控制：
   L1/L2/L3：单个舵机锁定
   R1/R2/R3：单个舵机释放
   L0：依次发送 L1 -> L2 -> L3
   R0：依次发送 R1 -> R2 -> R3

注意：
- 无人机端 UDP 接收节点也必须监听 8810：
  rosrun uav_inventory udp_uav_cmd_receiver.py _bind_port:=8810
- 或者在 launch 中把 udp_uav_cmd_receiver 节点的 bind_port 改为 8810。
"""

import socket
import time
import argparse


DEFAULT_UAV_IP = "192.168.151.102"
DEFAULT_UAV_PORT = 8810

RECV_TIMEOUT = 5.0
DROP_STEP_DELAY = 0.35



def drain_socket(sock):
    """
    清空 UDP socket 中可能残留的上一条迟到回包。
    避免 L2 收到 L1 回包这种错位问题。
    """
    old_timeout = sock.gettimeout()

    try:
        sock.settimeout(0.0)

        while True:
            try:
                sock.recvfrom(4096)
            except BlockingIOError:
                break
            except socket.timeout:
                break
            except Exception:
                break
    finally:
        sock.settimeout(old_timeout)


def reply_matches(msg, reply):
    """
    判断当前回包是否属于本次发送的命令。
    不匹配的回包会被当作上一条命令的迟到回包丢弃。
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

    return (
        reply.startswith("ACK:%s" % cmd) or
        reply.startswith("ERR:%s" % cmd) or
        reply.startswith("ERR:UNKNOWN_CMD:%s" % cmd)
    )


def wait_reply(sock, msg):
    """
    等待本次命令对应的回包。
    若收到迟到/错位回包，则打印并丢弃，继续等本次回包。
    """
    deadline = time.time() + RECV_TIMEOUT

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise socket.timeout()

        old_timeout = sock.gettimeout()
        sock.settimeout(remaining)

        try:
            data, from_addr = sock.recvfrom(4096)
        finally:
            sock.settimeout(old_timeout)

        reply = data.decode("utf-8", errors="ignore").strip()

        if reply_matches(msg, reply):
            return reply, from_addr

        print("收到迟到/错位回传，已丢弃：", reply)


def print_ping_result(reply):
    if reply.startswith("ACK:PING:SQUARE_TEST_OK"):
        print("PING 结果：方形测试节点在线。")
        return

    if reply.startswith("ACK:PING:NORMAL"):
        print("PING 结果：UDP 接收节点在线，FSM 心跳正常。")
        return

    if reply.startswith("ACK:PING:RECEIVER_OK"):
        if "fsm=NO_HEARTBEAT" in reply:
            print("PING 结果：UDP 接收节点在线，但还没收到 FSM 心跳。")
            return
        if "fsm=TIMEOUT" in reply:
            print("PING 结果：UDP 接收节点在线，但 FSM 心跳超时。")
            return
        print("PING 结果：UDP 接收节点在线，FSM 状态未知。")
        return

    print("PING 结果：收到未知回传。")


def send_cmd(sock, ip, port, cmd):
    cmd = cmd.strip().upper()
    if not cmd:
        return None

    # 发送新命令前清空上一条迟到回包，避免 UDP 回包错位。
    drain_socket(sock)

    if not cmd.startswith("CMD:"):
        msg = "CMD:" + cmd
    else:
        msg = cmd

    addr = (ip, port)
    print("\n发送：", msg)

    try:
        sock.sendto(msg.encode("utf-8"), addr)
    except Exception as e:
        print("发送失败：", e)
        return None

    try:
        reply, from_addr = wait_reply(sock, msg)
        print("收到回传：", reply)
        print("来自：", from_addr[0], from_addr[1])

        if msg == "CMD:PING":
            print_ping_result(reply)

        return reply

    except socket.timeout:
        print("等待回传超时。请检查无人机 IP/端口、UDP 节点是否启动、是否监听 8810。")
        return None
    except Exception as e:
        print("接收回传失败：", e)
        return None


def is_single_servo_cmd(cmd):
    cmd = cmd.lower().strip()
    return len(cmd) == 2 and cmd[0] in ["l", "r"] and cmd[1] in ["1", "2", "3"]


def send_servo_sequence(sock, ip, port, prefix):
    prefix = prefix.upper()
    if prefix not in ["L", "R"]:
        print("序列错误：prefix 必须是 L 或 R")
        return

    print("\n{}0：开始依次发送 {}1 / {}2 / {}3".format(prefix, prefix, prefix, prefix))

    for i in range(1, 4):
        cmd = "{}{}".format(prefix, i)
        send_cmd(sock, ip, port, cmd)
        time.sleep(DROP_STEP_DELAY)

    print("{}0：序列发送完成。".format(prefix))


def print_help(ip, port):
    print("\n========== 简易 UDP 地面站：8810 安全测试版 ==========")
    print("无人机地址：{}:{}".format(ip, port))
    print("回包超时：{} 秒，自动丢弃上一条迟到回包".format(RECV_TIMEOUT))
    print("")
    print("飞行控制：")
    print("  takeoff/start    发送 CMD:START，开始/起飞")
    print("  land             发送 CMD:LAND，普通降落")
    print("  stop             发送 CMD:STOP，急停降落")
    print("  reset            发送 CMD:RESET，复位 FSM")
    print("  disarm           发送 CMD:DISARM，上锁")
    print("")
    print("状态查询：")
    print("  ping             测试 UDP/FSM/串口状态")
    print("  status           查询详细状态")
    print("")
    print("舵机控制：")
    print("  L1 L2 L3         锁定 1/2/3 号舵机")
    print("  R1 R2 R3         释放 1/2/3 号舵机")
    print("  L0               依次发送 L1 -> L2 -> L3")
    print("  R0               依次发送 R1 -> R2 -> R3")
    print("")
    print("其他：")
    print("  h/help/?         显示帮助")
    print("  q/quit/exit      退出")
    print("=====================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Simple UDP ground station for UAV drop test.")
    parser.add_argument("--ip", default=DEFAULT_UAV_IP, help="UAV IP address")
    parser.add_argument("--port", type=int, default=DEFAULT_UAV_PORT, help="UAV UDP port")
    args = parser.parse_args()

    uav_ip = args.ip
    uav_port = args.port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(RECV_TIMEOUT)

    print_help(uav_ip, uav_port)

    while True:
        try:
            user_input = input("GROUND-8810> ").strip()
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
            print_help(uav_ip, uav_port)
            continue

        if cmd in ["takeoff", "start", "begin"]:
            send_cmd(sock, uav_ip, uav_port, "START")
            continue

        if cmd in ["land", "stop", "reset", "disarm", "ping", "status"]:
            send_cmd(sock, uav_ip, uav_port, cmd)
            continue

        if is_single_servo_cmd(cmd):
            send_cmd(sock, uav_ip, uav_port, cmd)
            continue

        if cmd == "l0":
            send_servo_sequence(sock, uav_ip, uav_port, "L")
            continue

        if cmd == "r0":
            send_servo_sequence(sock, uav_ip, uav_port, "R")
            continue

        print("未知命令：", user_input)
        print("输入 help 查看命令。")

    sock.close()


if __name__ == "__main__":
    main()
