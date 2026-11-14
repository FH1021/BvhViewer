# -*- coding: utf-8 -*-
"""
BVH 文件解析与前向运动学（FK）计算模块。

职责：
1. 读取 BVH 文件中的 HIERARCHY（骨骼层级、偏移、通道）与 MOTION（逐帧动作数据）；
2. 依据通道顺序做欧拉旋转，沿骨骼链做前向运动学，输出每一帧各关节的世界坐标。

本模块不依赖第三方 BVH 库，仅使用 numpy。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# 坐标轴字母 -> 0/1/2
_AXIS = {"X": 0, "Y": 1, "Z": 2}


@dataclass
class Joint:
    """单个关节（End Site 也作为一个关节节点保存）。"""

    name: str
    parent: Optional["Joint"]
    offset: np.ndarray                      # 相对父关节的静止偏移 (x, y, z)
    channels: list[str] = field(default_factory=list)
    channel_indices: list[int] = field(default_factory=list)
    children: list["Joint"] = field(default_factory=list)
    is_end: bool = False
    index: int = -1                        # 在扁平关节列表中的序号

    def rotation_channels(self) -> list[str]:
        return [c for c in self.channels if c.endswith("rotation")]


@dataclass
class Skeleton:
    root: Joint
    joints: list[Joint] = field(default_factory=list)   # 深度优先扁平顺序
    frame_time: float = 0.0333333
    frames: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.frames.shape[1])


def _tokenize(text: str) -> list[str]:
    """把 BVH 文本切成词法单元，花括号单独成 token。"""
    text = text.replace("{", " { ").replace("}", " } ")
    return text.split()


def _euler_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """绕指定轴旋转 angle_deg 度的 4x4 齐次矩阵。"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        return np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
            dtype=float,
        )
    if axis == "Y":
        return np.array(
            [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
            dtype=float,
        )
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def parse_bvh(path: str | Path) -> Skeleton:
    """解析 BVH 文件，返回 Skeleton 对象。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = _tokenize(text)
    pos = 0

    def expect(expected: str) -> None:
        nonlocal pos
        if pos >= len(tokens) or tokens[pos].upper() != expected.upper():
            raise ValueError(f"BVH 解析错误：期望 '{expected}'，实际 '{tokens[pos] if pos < len(tokens) else 'EOF'}'")
        pos += 1

    def read_number() -> float:
        nonlocal pos
        value = float(tokens[pos])
        pos += 1
        return value

    skeleton = Skeleton(root=Joint(name="Root", parent=None, offset=np.zeros(3)))
    channel_counter = 0

    def parse_joint(parent: Optional[Joint], name: str, is_end: bool = False) -> Joint:
        nonlocal channel_counter, pos
        expect("{")
        expect("OFFSET")
        offset = np.array([read_number(), read_number(), read_number()])

        joint = Joint(name=name, parent=parent, offset=offset, is_end=is_end)

        if not is_end:
            expect("CHANNELS")
            n = int(read_number())
            for _ in range(n):
                joint.channels.append(tokens[pos])
                joint.channel_indices.append(channel_counter)
                channel_counter += 1
                pos += 1

        while True:
            if pos >= len(tokens):
                raise ValueError("BVH 解析错误：层级结构未闭合（缺少 '}'）")
            token = tokens[pos]
            if token.upper() == "JOINT":
                pos += 1
                child_name = tokens[pos]
                pos += 1
                joint.children.append(parse_joint(joint, child_name))
            elif token.upper() == "END":
                pos += 1  # END / End
                expect("SITE")
                expect("{")
                expect("OFFSET")
                end_offset = np.array([read_number(), read_number(), read_number()])
                expect("}")
                end_site = Joint(
                    name=f"{joint.name}_end",
                    parent=joint,
                    offset=end_offset,
                    is_end=True,
                )
                joint.children.append(end_site)
            elif token == "}":
                pos += 1
                break
            else:
                raise ValueError(f"BVH 解析错误：意外的 token '{token}'（关节 {joint.name}）")
        return joint

    # ---- HIERARCHY ----
    expect("HIERARCHY")
    expect("ROOT")
    root_name = tokens[pos]
    pos += 1
    skeleton.root = parse_joint(None, root_name)

    # 深度优先展开扁平关节列表
    def flatten(node: Joint) -> None:
        node.index = len(skeleton.joints)
        skeleton.joints.append(node)
        for child in node.children:
            flatten(child)

    flatten(skeleton.root)

    # ---- MOTION ----
    expect("MOTION")
    expect("Frames:")
    num_frames = int(read_number())
    expect("Frame")
    expect("Time:")
    skeleton.frame_time = read_number()

    motion_values: list[list[float]] = []
    while pos < len(tokens):
        row = [read_number() for _ in range(channel_counter)]
        motion_values.append(row)

    if len(motion_values) != num_frames:
        # 容错：以实际读到的行数为准，并给出提示
        print(f"[提示] 声明帧数 {num_frames}，实际读到 {len(motion_values)} 行动作数据")

    skeleton.frames = np.asarray(motion_values, dtype=float)
    return skeleton


def _local_transform(joint: Joint, values: np.ndarray) -> np.ndarray:
    """根据一帧的通道值计算关节的局部齐次变换矩阵。"""
    rotation = np.eye(4)
    # 含 position 通道的关节（通常是根关节）：位移通道为绝对位置，不再叠加 OFFSET；
    # 其余关节使用静止偏移 OFFSET。
    has_position = any(c.endswith("position") for c in joint.channels)
    translation = np.zeros(3) if has_position else joint.offset.copy()

    for channel, index in zip(joint.channels, joint.channel_indices):
        value = values[index]
        axis = channel[0]
        if channel.endswith("position"):
            translation[_AXIS[axis]] = value
        else:  # rotation：按 CHANNELS 中列出的顺序依次乘旋转矩阵
            rotation = rotation @ _euler_matrix(axis, value)

    local = np.eye(4)
    local[:3, :3] = rotation[:3, :3]
    local[:3, 3] = translation
    return local


def compute_forward_kinematics(skeleton: Skeleton) -> np.ndarray:
    """
    对每一帧做前向运动学。

    返回:
        world_pos: 形状 (帧数, 关节数, 3) 的数组，world_pos[f, j] 为第 f 帧第 j 个关节的世界坐标。
    """
    n_joints = len(skeleton.joints)
    num_frames = skeleton.num_frames
    world_pos = np.zeros((num_frames, n_joints, 3))

    for frame_idx in range(num_frames):
        values = skeleton.frames[frame_idx]
        world_matrices: list[np.ndarray] = [np.eye(4)] * n_joints
        for joint in skeleton.joints:
            local = _local_transform(joint, values)
            if joint.parent is None:
                world_matrices[joint.index] = local
            else:
                world_matrices[joint.index] = world_matrices[joint.parent.index] @ local
            world_pos[frame_idx, joint.index] = world_matrices[joint.index][:3, 3]

    return world_pos


def bone_connections(skeleton: Skeleton) -> list[tuple[int, int]]:
    """返回所有骨骼连线：(子关节序号, 父关节序号)。"""
    connections: list[tuple[int, int]] = []
    for joint in skeleton.joints:
        if joint.parent is not None:
            connections.append((joint.index, joint.parent.index))
    return connections
