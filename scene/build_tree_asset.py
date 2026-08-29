"""把 YOPO 的真实扫描树点云（tree.ply）转成 Genesis 可加载的树 mesh。

上游的树是点云（227969 个点，无面），Genesis 的 gs.morphs.Mesh 需要带面的
网格，所以要做一次表面重建：

    点云 → 体素占据网格 → 形态学闭运算（填内部空洞）→ 高斯平滑
         → marching cubes 抽等值面 → 简化 → 导出

用法：

    python GenesisDroneEnv/scene/build_tree_asset.py           # 默认 0.12 m 体素
    python GenesisDroneEnv/scene/build_tree_asset.py --voxel 0.15

产出：assets/trees/tree_lod.obj（树沿 +Z 生长，底面在 z=0，跟上游一致）

注意：树干在点云里只有薄薄一层表面点，重建后容易断，所以脚本里显式补一根
半径 0.25 m 的圆柱（剖面实测值），不依赖点云密度。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

WS_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PLY = WS_ROOT / "prototypes" / "YOPO_ref" / "tree.ply"
if not SOURCE_PLY.exists():
    SOURCE_PLY = Path(
        "/home/tong/tongworkspace/rosworkspace/SANDO/.refer/YOPO/Simulator/src/pointcloud/tree.ply"
    )
OUT_DIR = WS_ROOT / "assets" / "trees"

# 剖面实测（见 mygenesisdrone/docs/research/random_dynamic_scenes.md）
TRUNK_RADIUS = 0.25  # m，树干层 p95
TRUNK_TOP = 3.5  # m，干冠分界


def build_mesh(voxel: float) -> trimesh.Trimesh:
    import skimage.measure
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

    pc = trimesh.load(SOURCE_PLY)
    pts = np.asarray(pc.vertices)
    print(f"点云 {len(pts)} 点，范围 {np.round(pts.min(0), 2)} ~ {np.round(pts.max(0), 2)}")

    lo = pts.min(0) - voxel
    hi = pts.max(0) + voxel
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    grid = np.zeros(shape, dtype=bool)
    idx = np.floor((pts - lo) / voxel).astype(int)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    print(f"体素网格 {shape}（{grid.sum()} 占据）")

    # 显式补树干：圆柱 r=0.25，z 0~3.5，保证细干不被体素分辨率吃掉
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    wx = lo[0] + xx * voxel
    wy = lo[1] + yy * voxel
    wz = lo[2] + zz * voxel
    trunk = (wz <= TRUNK_TOP) & (((wx - cx) ** 2 + (wy - cy) ** 2) <= TRUNK_RADIUS**2)
    grid |= trunk

    # 闭运算填表面点之间的缝，再填实心
    st = np.ones((3, 3, 3), dtype=bool)
    grid = binary_closing(grid, structure=st, iterations=2)
    grid = binary_fill_holes(grid)

    # 平滑后抽等值面
    field = gaussian_filter(grid.astype(np.float32), sigma=1.2)
    verts, faces, _, _ = skimage.measure.marching_cubes(field, level=0.5)
    verts = verts * voxel + lo  # 回到米制世界坐标（Z-up，底面 z≈0）

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    print(f"重建完成：{len(mesh.vertices)} 顶点 / {len(mesh.faces)} 面，包围盒 {np.round(mesh.extents, 2)}")

    # 把树干轴移到 x=y=0、底面 z=0，放置时 pos 直接给树根坐标即可
    mesh.apply_translation([-cx, -cy, -mesh.bounds[0][2]])
    print(f"居中后包围盒 {np.round(mesh.bounds, 2)}")
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel", type=float, default=0.12, help="体素边长，米，越小越细")
    parser.add_argument("--target-faces", type=int, default=8000, help="简化目标面数")
    args = parser.parse_args()

    mesh = build_mesh(args.voxel)
    if len(mesh.faces) > args.target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=args.target_faces)
            print(f"简化到 {len(mesh.faces)} 面")
        except Exception as exc:
            print(f"简化跳过（{exc}），保留 {len(mesh.faces)} 面")

    mesh.export(OUT_DIR / "tree_lod.obj")
    print(f"已导出 {OUT_DIR / 'tree_lod.obj'}")


if __name__ == "__main__":
    main()
