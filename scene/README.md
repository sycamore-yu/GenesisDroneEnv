# 场景库

给无人机环境用的 Genesis 场景：竞速赛道、静态森林、动态障碍物。每个场景都能
开窗口直接看。

## 用法

```bash
# 在仓库根目录 /home/tong/tongworkspace/genesisworkspace 下跑
python GenesisDroneEnv/scene/view.py                    # 7 门竞速赛道
python GenesisDroneEnv/scene/view.py --scene forest     # 圆柱森林（SANDO 风格）
python GenesisDroneEnv/scene/view.py --scene tree_forest   # 正常森林（YOPO 风格，树干+树冠）
python GenesisDroneEnv/scene/view.py --scene dynamic    # 动态障碍（会动）
python GenesisDroneEnv/scene/view.py --scene dynamic_forest   # 圆柱森林 + 动态障碍

# 换随机种子（同种子同场景）
python GenesisDroneEnv/scene/view.py --scene forest --seed 3

# 不开窗口，只做数值验证
python GenesisDroneEnv/scene/view.py --scene dynamic --headless
```

关掉窗口即退出。相机机位已经按场景调好，窗口里可以自由旋转缩放。

## 假设

读这份文档前，我假设你：

- 知道怎么跑 Python 命令，知道 Genesis 场景的基本概念（`gs.Scene`、`add_entity`）。
- 不了解上游项目（YOPO / SANDO / isaac_drone_racer）的术语，下面都会解释。

**术语表**：

- **三叶结轨迹**：动态障碍物的运动方式。位置写成时间 `t` 的数学公式，
  好处是能直接算出任意未来时刻的位置，规划器可以拿来预测。
- **净间隙**：两个障碍物表面之间的空隙，不是中心距。
- **Git LFS**：只影响拿原始资产，本目录的脚本不涉及。

## 五种场景

| 场景 | 内容 | 参数来源 |
|---|---|---|
| `gate` | isaac_drone_racer 的 7 个竞速门，静止实体 | 上游 `drone_racer_env_cfg.py:41-47` |
| `forest` | **圆柱森林**：程序化圆柱，半径 1.0~1.5 m、高 6 m、净间隙 ≥ 2 m | SANDO `generate_random_forest.py:314-317` |
| `tree_forest` | **正常森林**：细树干 + 球状树冠，树形自然 | YOPO `maps.cpp:981-1046` + tree.ply 剖面实测 |
| `dynamic` | 100 个障碍（65% 会动），三叶结轨迹，峰值约 1.75 m/s | SANDO `run_sim.py:274-343` |
| `dynamic_forest` | 圆柱森林铺满 + 动态障碍沿中走廊飞 | 对应 SANDO 的 dynamic 难度思路 |

### 两种森林的区别

| | `forest`（圆柱森林） | `tree_forest`（正常森林） |
|---|---|---|
| 树的样子 | 一根 6 m 高的光秃圆柱 | **真实扫描树的 mesh**（从上游 tree.ply 点云重建，含细树干和自然树冠） |
| 来源 | SANDO 的参数化圆柱 | YOPO 的真实扫描树资产（tree.ply → marching cubes 重建，6000 面） |
| 地图 | x ∈ [-5, 110]、y ∈ [-25, 25] 长走廊 | 60 × 60 m，中心在原点 |
| 撒点 | 拒绝采样，树间净间隙 ≥ 2 m | 抖动网格（间距 4 m），**树冠会重叠，上游原样** |
| 数量 | 最多约 100 棵（受净间隙限制） | 固定 15×15 = 225 棵 |
| 碰撞 | 全部实体碰撞 | 树干圆柱 + 树冠保守球碰撞；树冠 mesh 纯视觉 |
| 适用 | 避障训练（几何简单、碰撞稳定） | 视觉场景、接近真实森林形态 |

`tree_forest` 每棵树三个实体：

1. **树干 `Cylinder`**（视觉 + 碰撞）：半径 0.125~0.25 m、高 1.75~3.5 m。
   飞行层里真正挡无人机的是它。
2. **树冠 `Mesh`**（**纯视觉**，`collision=False`）：6000 面的重建网格。
   树冠是凹形，225 棵做凸分解代价太高，不值得。
3. **树冠碰撞 `Sphere`**（**纯碰撞**，`visualization=False`）：半径 1.2~2.4 m
   的保守球，位于干冠分界上方。无人机撞树冠会被挡住，不会学到"能穿过树冠"。

树资产由 `build_tree_asset.py` 从上游点云一次性生成（点云 → 体素化 → 形态学
闭运算 → marching cubes → 简化），输出 `assets/trees/tree_lod.obj`。想换精度
改 `--voxel` / `--target-faces` 重跑即可。

跟 YOPO 原版一样，缩放 0.5~1.0 倍后树冠底最低到 1.35 m，**会伸进无人机飞行层**
（0.5~4 m）。

参数细节和三个上游项目的完整对比，见
`mygenesisdrone/docs/research/random_dynamic_scenes.md`。

## 动态障碍物是什么形状

**不是树，也不是行人、车辆。** 三个上游项目里的动态障碍物都是抽象几何体：

| 项目 | 形状 | 运动方式 | 速度 |
|---|---|---|---|
| SANDO / MIGHTY | 0.8 × 0.8 × 0.8 m 立方体 | 三叶结解析曲线 | 峰值约 1.75 m/s |
| YOPO_360 | 圆柱（宽 0.6~1.5 m、高 1.5~2.0 m） | 匀速直线 + 撞边界反弹 | 0.8~1.0 m/s |
| YOPO 本体 | — | **没有动态障碍物** | — |

没有社会力模型、没有随机游走、没有行人。要更真实的动态物（行人、车辆），
上游没有现成的，需要自己加运动模型——运动学驱动的接口已经就绪
（`set_dynamic_poses()`，换掉里面的位置公式即可）。

## 代码结构

```
scene/
├── view.py      唯一入口，全部场景都在这里
└── README.md    本文件
```

`view.py` 不复制生成逻辑，直接复用两处已验证的模块：

- 门加载器：`../assets/gate/gate_track.py`（赛道的 7 个门，含轴向修正）
- 场景生成器：`../../prototypes/random_dynamic_scene/generate.py`
  （森林采样、动态障碍采样、三叶结公式）

所以删掉 `prototypes/` 或 `assets/gate/` 会让本目录失效。如果以后要正式化，
把这两个模块挪进来并更新 import 即可。

## 已验证

Genesis 1.3.3 + CUDA，五种场景 headless 全部通过：

```
[headless] 场景 gate（seed=0）建成
[headless] 场景 forest（seed=0）建成
[headless] 场景 tree_forest（seed=0）建成          # 225 棵树 = 450 个实体
[headless] 场景 dynamic（seed=0）建成，动态障碍 65 个
           300 步后前 3 个动态障碍位移: [1.413, 1.322, 1.869] m
[headless] 场景 dynamic_forest（seed=0）建成，动态障碍 39 个
           300 步后前 3 个动态障碍位移: [0.993, 1.38, 1.678] m
```

## 已知行为（上游就这样，不是 bug）

- 动态障碍偶尔会穿地或互相穿透。上游 Gazebo 里障碍就是 `<static>true</static>`
  且没有 `<collision>`，运动学驱动、无碰撞反馈。要关掉 Genesis 侧的障碍间碰撞，
  用 `contype` / `conaffinity` 掩码。
- 森林树间距靠"半径和 + clearance"保证，但**没有保证起终点连通**——
  极端种子下可能出现树墙封死走廊。上游也没有做连通性检查。
  `dynamic` / `dynamic_forest` 场景里动态障碍走的走廊（y ∈ [-6, 6]）会相对稀疏。
