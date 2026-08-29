"""场景可视化入口。开一个 Genesis 窗口看赛道、森林、动态障碍。

用法（在仓库根目录跑，或任意目录用绝对路径）：

    python GenesisDroneEnv/scene/view.py                    # 默认看 7 门赛道
    python GenesisDroneEnv/scene/view.py --scene forest     # 圆柱森林
    python GenesisDroneEnv/scene/view.py --scene dynamic    # 动态障碍（会动）
    python GenesisDroneEnv/scene/view.py --scene dynamic_forest
    python GenesisDroneEnv/scene/view.py --scene gate --seed 3
    python GenesisDroneEnv/scene/view.py --headless         # 不开窗口，跑数值验证

场景参数来源见同目录 README.md。关掉窗口即退出。
"""

import argparse
import sys
from pathlib import Path

# 复用两处已验证的代码：门加载器（assets/gate）和随机场景生成器（prototypes）。
# 这里不复制逻辑，保证单一来源。
WS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS_ROOT / "assets" / "gate"))
sys.path.insert(0, str(WS_ROOT / "prototypes" / "random_dynamic_scene"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import genesis as gs  # noqa: E402

from generate import (  # noqa: E402
    build_scene,
    sample_forest,
    sample_scene,
    sample_tree_forest,
    set_dynamic_poses,
)
from gate_track import add_isaac_track  # noqa: E402

# 静态森林用宽场景。来源：SANDO headless 原型的边界
# /home/tong/tongworkspace/rosworkspace/SANDO/prototypes/headless_paper_forest_teacher/run_prototype.py:153
# （窄走廊 y∈[-6,6] 放不下半径 1.0~1.5 m、净间隙 2 m 的树，42 棵就到顶了）
FOREST_BOUNDS = {"x": (-5.0, 110.0), "y": (-25.0, 25.0), "z": (0.5, 6.0)}

# 动态障碍用上游窄走廊。来源 SANDO run_sim.py 的默认采样区间
DYNAMIC_BOUNDS = {"x": (5.0, 100.0), "y": (-6.0, 6.0), "z": (0.5, 4.5)}

# 每个场景的相机机位（米）
CAMERAS = {
    "gate": {"pos": (14.0, -14.0, 9.0), "lookat": (2.0, 0.0, 1.2)},
    "forest": {"pos": (130.0, 45.0, 35.0), "lookat": (50.0, 0.0, 3.0)},
    "tree_forest": {"pos": (55.0, 45.0, 32.0), "lookat": (0.0, 0.0, 5.0)},
    "dynamic": {"pos": (110.0, 20.0, 20.0), "lookat": (50.0, 0.0, 2.0)},
    "dynamic_forest": {"pos": (130.0, 45.0, 35.0), "lookat": (50.0, 0.0, 3.0)},
}


def build(scene, name, seed):
    """往场景里加障碍。返回 (动态实体列表, 动态障碍参数列表)，两者按序一一对应。"""
    if name == "gate":
        add_isaac_track(scene)
        return [], []

    if name == "forest":
        trees = sample_forest(num_trees=200, bounds=FOREST_BOUNDS, seed=seed)
        build_scene(scene, trees)
        return [], []

    if name == "tree_forest":
        # 正常形态的森林：细树干 + 球状树冠，YOPO 参数，地图 60 x 60 m 居中
        trees = sample_tree_forest(seed=seed)
        build_scene(scene, trees)
        return [], []

    if name == "dynamic":
        obstacles = sample_scene(num_obstacles=100, bounds=DYNAMIC_BOUNDS, seed=seed)
        entities, dyn = build_scene(scene, obstacles)
        return dyn, [o for o in obstacles if o["dynamic"]]

    if name == "dynamic_forest":
        # 对应 SANDO 的 dynamic 难度思路：森林铺满 + 动态障碍沿中走廊飞
        trees = sample_forest(num_trees=150, bounds=FOREST_BOUNDS, seed=seed)
        movers = sample_scene(num_obstacles=60, bounds=DYNAMIC_BOUNDS, seed=seed + 1)
        build_scene(scene, trees)
        entities, dyn = build_scene(scene, movers)
        return dyn, [o for o in movers if o["dynamic"]]

    raise ValueError(f"未知场景: {name}")


def _pos(entity):
    v = entity.get_pos()
    if hasattr(v, "cpu"):
        v = v.cpu().numpy()
    return np.asarray(v, dtype=float).reshape(-1)[:3]


def main():
    parser = argparse.ArgumentParser(description="Genesis 场景可视化")
    parser.add_argument("--scene", default="gate", choices=list(CAMERAS), help="看哪个场景")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，同种子同场景")
    parser.add_argument("--headless", action="store_true", help="不开窗口，跑数值验证")
    parser.add_argument("--steps", type=int, default=300, help="headless 模式的仿真步数")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="error")

    scene = gs.Scene(show_viewer=not args.headless)
    scene.add_entity(gs.morphs.Plane())
    dyn, dyn_obs = build(scene, args.scene, args.seed)
    scene.build()

    if not args.headless:
        try:
            cam = CAMERAS[args.scene]
            scene.viewer.set_camera_pose(pos=cam["pos"], lookat=cam["lookat"])
        except Exception as exc:  # 相机设不上不影响看
            print(f"相机机位未生效: {exc}")
        print(f"场景 {args.scene}（seed={args.seed}），动态障碍 {len(dyn)} 个。关窗口退出。")
        t = 0.0
        try:
            while scene.viewer.is_alive():
                t += scene.dt
                if dyn:
                    set_dynamic_poses(dyn, dyn_obs, t)
                scene.step()
        except gs.GenesisException:
            pass  # 用户关掉窗口，is_alive 检测会滞后一步，这里兜住
        return

    # headless：验证场景能建起来、动态障碍会动
    print(f"[headless] 场景 {args.scene}（seed={args.seed}）建成，动态障碍 {len(dyn)} 个")
    if dyn:
        probe = range(min(3, len(dyn)))
        start = [_pos(dyn[i]) for i in probe]
        t = 0.0
        for _ in range(args.steps):
            t += scene.dt
            set_dynamic_poses(dyn, dyn_obs, t)
            scene.step()
        moved = [round(float(np.linalg.norm(_pos(dyn[i]) - s)), 3) for i, s in zip(probe, start)]
        print(f"[headless] {args.steps} 步后前 3 个动态障碍位移: {moved} m")


if __name__ == "__main__":
    main()
