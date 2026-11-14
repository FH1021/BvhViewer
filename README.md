# BvhViewer
# BVH 动作解析演示程序

在 VSCode + Python 环境下运行的 BVH 动作捕捉文件解析与三维播放演示。程序读取 BVH 文件，重建骨骼层级，通过前向运动学（FK）计算每一帧关节坐标，并在三维视图中播放动画，支持播放 / 暂停与速度调节。

## 一、文件清单

| 文件                  | 说明                                                   |
| ------------------- | ---------------------------------------------------- |
| `BvhViewer.exe`     | 打包好的单文件成品程序，双击即可运行，无需安装 Python（Windows x64）          |
| `bvh_parser.py`     | BVH 解析模块：解析 `HIERARCHY` 与 `MOTION` 段，建立骨骼树，完成前向运动学计算 |
| `bvh_viewer.py`     | 播放器主程序：三维骨架显示、播放/暂停、速度与帧进度控制、打开文件                    |
| `make_demo_bvh.py`  | 示例 BVH 生成脚本，运行后产出 `demo.bvh`                         |
| `make_extra_bvh.py` | 额外测试动作生成脚本，运行后产出 `walk_cycle.bvh`、`wave_hello.bvh`   |
| `demo.bvh`          | 自带的演示动作（60 帧，双臂摆动 + 身体起伏），已打包进 exe                   |
| `walk_cycle.bvh`    | 全身骨架原地步行循环（48 帧，含双腿、双臂）                              |
| `wave_hello.bvh`    | 上半身抬手挥手（40 帧，旋转通道顺序为 XYZ）                            |

## 二、环境依赖

- Python 3.10 及以上（开发环境为 Python 3.13）
- 依赖库：`numpy`、`matplotlib`
- 图形界面使用 Python 自带的 `tkinter`（官方 Windows 安装包默认包含）

如缺少依赖，在终端执行：

```bash
pip install numpy matplotlib
```

## 三、直接使用成品 EXE

项目根目录下的 `BvhViewer.exe` 已打包好全部依赖和内置演示动作：

1. 双击 `BvhViewer.exe` 即可启动，自动加载内置 `demo.bvh`；
2. 通过界面上的「打开 BVH」按钮载入其他动作文件，可选择同目录的 `walk_cycle.bvh`、`wave_hello.bvh` 测试；
3. 无需安装 Python 和任何第三方库；首次启动单文件 exe 需要几秒解压时间，属正常现象。

> 说明：exe 由 Windows x64 + Python 3.11 打包，仅适用于 64 位 Windows；拷贝到其他电脑时直接复制这一个文件即可。

## 四、在 VSCode 中运行源码

1. 用 VSCode 打开本文件夹（`文件 → 打开文件夹`）。
2. 按 `Ctrl + Shift + P` → `Python: Select Interpreter`，选择本机已安装的 Python 解释器。
3. 打开终端（`Ctrl + ~`），运行：

```bash
# 直接播放自带示例
python bvh_viewer.py demo.bvh

# 不带参数：自动尝试同目录的 demo.bvh，否则弹出文件选择框
python bvh_viewer.py

# 指定自己的 BVH 文件
python bvh_viewer.py 路径/到/你的动作.bvh
```

也可以在 VSCode 中打开 `bvh_viewer.py`，点击右上角运行按钮（`Run Python File`）。

## 五、界面控件与快捷键

| 控件 / 按键      | 功能                       |
| ------------ | ------------------------ |
| `暂停 / 播放` 按钮 | 暂停或继续动画                  |
| `打开 BVH` 按钮  | 通过文件选择框载入其他 BVH 文件       |
| `速度` 滑块      | 播放速度 0.1x ~ 3.0x，步进 0.1x |
| `帧` 滑块       | 跳转到指定帧，可拖拽定位             |
| `空格`         | 播放 / 暂停                  |
| `←` / `→`    | 上一帧 / 下一帧                |
| `-` / `+`    | 减速 / 加速 0.1x             |

动画循环播放；窗口内鼠标拖动可旋转视角，滚轮缩放。

## 六、实现原理简述

1. **词法与语法解析**：将 BVH 文件切分为词法单元，递归解析 `ROOT / JOINT / End Site` 层级，记录每个关节的 `OFFSET`（静止偏移）与 `CHANNELS`（通道及在动作数据中的序号）。
2. **动作数据**：`MOTION` 段读取总帧数、帧时长（`Frame Time`）与逐帧通道数值矩阵。
3. **前向运动学（FK）**：每一帧从根关节开始，按骨骼链向下累积齐次变换矩阵；旋转通道按 `CHANNELS` 中声明的轴顺序依次乘欧拉旋转矩阵，最终得到各关节的世界坐标。
4. **渲染**：每 30ms 刷新一次三维视图，根据当前帧更新骨骼连线与关节点；播放进度按 `速度 × 实际经过秒数 / 帧时长` 推进。
