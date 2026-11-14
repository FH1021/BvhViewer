# -*- coding: utf-8 -*-
"""
BVH 动作演示播放器（VSCode + Python + matplotlib）。

功能：
- 读入 BVH 文件，解析骨骼层级与逐帧动作；
- 三维骨架显示，循环播放；
- 播放 / 暂停按钮；
- 播放速度滑块（0.1x ~ 3.0x）；
- 帧进度滑块，可拖拽定位；
- “打开”按钮选择其他 BVH 文件；
- 快捷键：空格 = 播放/暂停，←/→ = 逐帧，+/- = 调速。

运行：
    python bvh_viewer.py demo.bvh
    python bvh_viewer.py                 # 未指定文件时弹出文件选择框
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

# 允许通过环境变量 MPLBACKEND 覆盖后端（例如无头测试用 Agg）；默认使用 TkAgg。
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 中文字体回退，避免界面中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.widgets import Button, Slider

from bvh_parser import (
    Skeleton,
    bone_connections,
    compute_forward_kinematics,
    parse_bvh,
)


def _resource_path(name: str) -> Path:
    """获取资源文件路径：兼容源码运行与 PyInstaller 打包后的单文件模式。"""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # PyInstaller 解压临时目录
    else:
        base = Path(__file__).parent
    return base / name


class BvhViewer:
    """BVH 三维播放器。"""

    def __init__(self, file_path: str | Path):
        self.skeleton: Skeleton | None = None
        self.world_pos: np.ndarray = np.zeros((0, 0, 3))
        self.connections: list[tuple[int, int]] = []

        self.playing = True
        self.speed = 1.0
        self.current = 0.0
        self._loading_slider = False  # 程序内部更新滑块时屏蔽回调

        # ---------- 界面布局 ----------
        self.fig = plt.figure(figsize=(10, 7.5))
        self.ax = self.fig.add_axes([0.04, 0.20, 0.92, 0.74], projection="3d")

        self.bone_lines: list = []
        self.joint_scatter = None

        # 播放 / 暂停
        self.ax_play = self.fig.add_axes([0.04, 0.05, 0.13, 0.07])
        self.btn_play = Button(self.ax_play, "暂停")

        # 打开文件
        self.ax_open = self.fig.add_axes([0.19, 0.05, 0.13, 0.07])
        self.btn_open = Button(self.ax_open, "打开 BVH")

        # 速度滑块
        self.ax_speed = self.fig.add_axes([0.38, 0.06, 0.24, 0.05])
        self.speed_slider = Slider(
            self.ax_speed, "速度", valmin=0.1, valmax=3.0, valinit=1.0,
            valstep=0.1, valfmt="%.1fx",
        )

        # 帧进度滑块
        self.ax_frame = self.fig.add_axes([0.68, 0.06, 0.28, 0.05])
        self.frame_slider = Slider(
            self.ax_frame, "帧", valmin=0, valmax=1, valinit=0, valstep=1,
        )

        self.status_text = self.fig.text(
            0.04, 0.005, "", fontsize=10, color="#333333"
        )

        # ---------- 事件绑定 ----------
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_open.on_clicked(self.open_file)
        self.speed_slider.on_changed(self.on_speed_change)
        self.frame_slider.on_changed(self.on_frame_change)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # ---------- 载入文件 ----------
        self.load(file_path)

        self._last_tick = time.perf_counter()
        # 30ms 刷新一次
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self.on_tick)
        self.timer.start()

    # ================= 数据载入 =================

    def load(self, file_path: str | Path) -> None:
        path = Path(file_path)
        print(f"[载入] {path}")
        self.skeleton = parse_bvh(path)
        self.world_pos = compute_forward_kinematics(self.skeleton)
        self.connections = bone_connections(self.skeleton)

        self.current = 0.0
        num_frames = self.skeleton.num_frames
        if num_frames == 0:
            raise ValueError("BVH 文件中没有动作数据（MOTION 段为空）")

        # 重建骨骼线条与关节点
        self.ax.clear()
        self.bone_lines = []
        for _ in self.connections:
            line, = self.ax.plot([], [], [], color="#4C78A8", lw=2, marker="o",
                                  markersize=4, markerfacecolor="#F58518")
            self.bone_lines.append(line)
        self.joint_scatter = self.ax.scatter([], [], [], c="#E45756", s=18)

        self.fit_view()
        self._rebuild_frame_slider(num_frames)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")
        self.ax.set_title(path.name, fontsize=12)
        self.draw_current_frame()
        self.update_status()

    def fit_view(self) -> None:
        """根据全部帧的坐标范围自动设置视角包围盒。"""
        if self.world_pos.size == 0:
            return
        lo = self.world_pos.reshape(-1, 3).min(axis=0)
        hi = self.world_pos.reshape(-1, 3).max(axis=0)
        center = (lo + hi) / 2
        span = np.maximum(hi - lo, 1e-3)
        pad = span * 0.15
        self.ax.set_xlim(center[0] - span[0] / 2 - pad[0], center[0] + span[0] / 2 + pad[0])
        self.ax.set_ylim(center[2] - span[2] / 2 - pad[2], center[2] + span[2] / 2 + pad[2])
        self.ax.set_zlim(center[1] - span[1] / 2 - pad[1], center[1] + span[1] / 2 + pad[1])
        self.ax.set_box_aspect((span[0], span[2], span[1]))

    def _rebuild_frame_slider(self, num_frames: int) -> None:
        self._loading_slider = True
        self.frame_slider.set_val(0)
        self.frame_slider.valmax = num_frames - 1
        self.frame_slider.ax.set_xlim(0, num_frames - 1)
        self._loading_slider = False

    # ================= 绘制 =================

    def draw_current_frame(self) -> None:
        if self.skeleton is None or self.world_pos.size == 0:
            return
        num_frames = self.skeleton.num_frames
        f = int(round(self.current)) % num_frames
        self.current = f

        positions = self.world_pos[f]
        # 骨骼连线：注意 matplotlib 3D 中 y 轴对应 BVH 的 Z，z 轴对应 BVH 的 Y
        for line, (child, parent) in zip(self.bone_lines, self.connections):
            c = positions[child]
            p = positions[parent]
            line.set_data_3d(
                [p[0], c[0]],
                [p[2], c[2]],
                [p[1], c[1]],
            )

        self.joint_scatter._offsets3d = (
            positions[:, 0].tolist(),
            positions[:, 2].tolist(),
            positions[:, 1].tolist(),
        )

        if not self._loading_slider:
            self._loading_slider = True
            self.frame_slider.set_val(f)
            self._loading_slider = False
        self.update_status()
        self.fig.canvas.draw_idle()

    # ================= 播放控制 =================

    def on_tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if not self.playing or self.skeleton is None:
            return
        # frame_time 为每帧秒数；按速度推进帧序号
        frame_time = self.skeleton.frame_time or 1 / 30
        self.current += self.speed * dt / frame_time
        num_frames = self.skeleton.num_frames
        if self.current >= num_frames:
            self.current -= num_frames
        self.draw_current_frame()

    def toggle_play(self, _event=None) -> None:
        self.playing = not self.playing
        self.btn_play.label.set_text("暂停" if self.playing else "播放")
        self._last_tick = time.perf_counter()
        self.update_status()
        self.fig.canvas.draw_idle()

    def on_speed_change(self, value: float) -> None:
        self.speed = float(value)
        self.update_status()

    def on_frame_change(self, value: float) -> None:
        if self._loading_slider or self.skeleton is None:
            return
        self.current = float(value)
        self.draw_current_frame()

    def open_file(self, _event=None) -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="选择 BVH 文件",
            filetypes=[("BVH 动作文件", "*.bvh"), ("所有文件", "*.*")],
        )
        root.destroy()
        if path:
            try:
                self.load(path)
            except Exception as exc:  # 载入失败时保留当前画面
                print(f"[错误] 无法载入 {path}: {exc}")

    def on_key(self, event) -> None:
        if event.key == " ":
            self.toggle_play()
        elif event.key == "right" and self.skeleton:
            self.current = (self.current + 1) % self.skeleton.num_frames
            self.draw_current_frame()
        elif event.key == "left" and self.skeleton:
            self.current = (self.current - 1) % self.skeleton.num_frames
            self.draw_current_frame()
        elif event.key in ("+", "="):
            self.speed = min(3.0, self.speed + 0.1)
            self.speed_slider.set_val(self.speed)
        elif event.key == "-":
            self.speed = max(0.1, self.speed - 0.1)
            self.speed_slider.set_val(self.speed)

    def update_status(self) -> None:
        if self.skeleton is None:
            return
        num_frames = self.skeleton.num_frames
        state = "播放中" if self.playing else "已暂停"
        self.status_text.set_text(
            f"帧: {int(self.current) + 1}/{num_frames}    速度: {self.speed:.1f}x    "
            f"{state}    帧时长: {self.skeleton.frame_time:.4f}s"
        )


def selftest(path: str) -> None:
    """无界面自检：解析并跑完所有帧的 FK，输出统计信息。"""
    skeleton = parse_bvh(path)
    world_pos = compute_forward_kinematics(skeleton)
    connections = bone_connections(skeleton)
    print(f"关节数: {len(skeleton.joints)}")
    print(f"骨骼连线数: {len(connections)}")
    print(f"帧数: {skeleton.num_frames}  通道数: {skeleton.num_channels}  帧时长: {skeleton.frame_time:.5f}s")
    print(f"世界坐标数组形状: {world_pos.shape}")
    print(f"坐标范围 X[{world_pos[:, :, 0].min():.3f}, {world_pos[:, :, 0].max():.3f}]  "
          f"Y[{world_pos[:, :, 1].min():.3f}, {world_pos[:, :, 1].max():.3f}]  "
          f"Z[{world_pos[:, :, 2].min():.3f}, {world_pos[:, :, 2].max():.3f}]")
    assert world_pos.shape[0] == skeleton.num_frames
    assert not np.isnan(world_pos).any(), "FK 结果存在 NaN"
    print("自检通过：解析与前向运动学计算正常。")


def main() -> None:
    parser = argparse.ArgumentParser(description="BVH 动作演示播放器")
    parser.add_argument("file", nargs="?", default=None, help="BVH 文件路径")
    parser.add_argument("--selftest", metavar="BVH", help="无界面自检模式")
    args = parser.parse_args()

    if args.selftest:
        selftest(args.selftest)
        return

    file_path = args.file
    if file_path is None:
        # 默认优先内置 / 同目录的 demo.bvh，否则弹窗选择
        demo = _resource_path("demo.bvh")
        if demo.exists():
            file_path = demo
        else:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="选择 BVH 文件",
                filetypes=[("BVH 动作文件", "*.bvh"), ("所有文件", "*.*")],
            )
            root.destroy()
            if not file_path:
                print("未选择文件，程序退出。")
                return

    viewer = BvhViewer(file_path)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 打包成 windowed exe 后没有控制台，弹出错误框便于定位问题
        if getattr(sys, "frozen", False):
            import traceback
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("BVH 播放器启动失败", traceback.format_exc())
        raise
