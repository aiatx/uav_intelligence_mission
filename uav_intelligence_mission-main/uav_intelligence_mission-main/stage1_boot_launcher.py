#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-1 boot launcher.

Usage:
    python3 stage1_boot_launcher.py

Behavior:
    1. Listen for UDP command CMD:BOOT.
    2. Reply ACK:BOOT:STARTING to the sender.
    3. Close the UDP socket.
    4. Run:
       roslaunch uav_inventory stage1_real_flight.launch

This script intentionally does not send periodic status heartbeat messages.
It is meant to be started by systemd or manually before the mission stack is up.
"""

import socket
import subprocess
import sys
import time


LOCAL_IP = "0.0.0.0"
LOCAL_PORT = 8888

ROS_SETUP = "/opt/ros/noetic/setup.bash"
CATKIN_SETUP = "/home/nvidia/catkin_ws/devel/setup.bash"
LAUNCH_PACKAGE = "uav_inventory"
LAUNCH_FILE = "stage1_real_flight.launch"

RECV_TIMEOUT = 1.0


def build_launch_command():
    return (
        "source {ros_setup} && "
        "source {catkin_setup} && "
        "roslaunch {package} {launch_file}"
    ).format(
        ros_setup=ROS_SETUP,
        catkin_setup=CATKIN_SETUP,
        package=LAUNCH_PACKAGE,
        launch_file=LAUNCH_FILE,
    )


def send_reply(sock, addr, text):
    try:
        sock.sendto((text + "\n").encode("utf-8"), addr)
    except Exception as exc:
        print("Failed to send UDP reply to {}: {}".format(addr, exc))


def run_roslaunch():
    launch_cmd = build_launch_command()
    print("Starting mission stack:")
    print("  {}".format(launch_cmd))

    # Keep this process alive while roslaunch is running. If roslaunch exits with
    # a non-zero code, this script exits with the same code so systemd
    # Restart=on-failure can bring the boot listener back.
    return subprocess.call(["bash", "-lc", launch_cmd])


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LOCAL_IP, LOCAL_PORT))
    sock.settimeout(RECV_TIMEOUT)

    print("Stage-1 boot launcher started.")
    print("Listening on {}:{} for CMD:BOOT".format(LOCAL_IP, LOCAL_PORT))

    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            continue
        except Exception as exc:
            print("UDP receive error: {}".format(exc))
            time.sleep(1.0)
            continue

        try:
            cmd = data.decode("utf-8", errors="ignore").strip().upper()
        except Exception:
            send_reply(sock, addr, "ERR:DECODE_FAILED")
            continue

        print("UDP command from {}: {}".format(addr, cmd))

        if cmd != "CMD:BOOT":
            send_reply(sock, addr, "ERR:UNKNOWN_CMD:{}".format(cmd))
            continue

        send_reply(sock, addr, "ACK:BOOT:STARTING")

        # Give the UDP packet a tiny moment to leave before closing the socket.
        time.sleep(0.1)
        sock.close()

        exit_code = run_roslaunch()
        print("roslaunch exited with code {}".format(exit_code))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
