<div align="center">

# 🚁 Differentiable UAV Policy Learning in Genesis

**Analytic-gradient policy learning for quadrotor control with differentiable simulation**

[Wiki](.gitnexus/wiki/index.html) ·
[Roadmap](docs/roadmaps/autonomous-racing-and-avoidance.md)

</div>

---

## 📖 Overview

This repository extends **GenesisDroneEnv** with differentiable policy learning for quadrotors.

Instead of relying solely on model-free policy-gradient estimators, **APG** and **SHAC** exploit gradients through the Genesis dynamics to directly optimize UAV control policies.

Current status:

- ✅ Waypoint and track-following
- 🚧 Multi-gate drone racing
- 🧭 LiDAR-based dynamic obstacle avoidance

> **Status:** Research prototype / work in progress.

---

## ✨ Highlights

- **Differentiable UAV dynamics** — gradients propagate through Genesis from task losses to policy actions.
- **APG & SHAC** — analytic-gradient policy learning with full- and short-horizon differentiation.
- **Unified CTBR interface** — collective thrust and body-rate control across tracking and racing tasks.
- **Evaluation tools** — gradient checks, convergence plots, PPO/APG/SHAC comparison, and parallel-environment benchmarks.
- **Racing & avoidance extensions** — fixed multi-gate racing environments and a roadmap toward LiDAR-based dynamic avoidance.

---

## 🎬 Quick Start

### APG / SHAC Tracking

```bash
python scripts/train/track_diff_train.py --algo apg --updates 300
python scripts/train/track_diff_train.py --algo shac --updates 300
```

### Racing

```bash
python scripts/train/race_train.py --algo ppo
python scripts/train/race_train.py --algo apg
python scripts/train/race_train.py --algo shac
```

### Unified Entry Point

```bash
python scripts/train/diff_train.py track --algo apg
python scripts/train/diff_train.py race --algo shac
```

### Numerical Validation

```bash
python scripts/eval/track_diff_gradcheck.py
python scripts/eval/track_diff_terminal_value_gradcheck.py
python scripts/eval/track_diff_benchmark.py --algo apg --num-envs 1024
```

---

## 🔁 Differentiable Training Loop

```text
Policy πθ
   │
   ▼
Action
   │
   ▼
Genesis Differentiable Dynamics
   │
   ▼
State / Task Loss
   │
   └──────── simulation gradient ────────► πθ
```

The simulator computes gradients with respect to the executed actions, which are propagated back to the policy parameters through the actor computation graph.

**APG** differentiates directly through the rollout dynamics.

**SHAC** limits backpropagation to short horizons and uses a critic to represent long-horizon value.

---

## 🛣️ Roadmap

| Stage | Task                             | Status         |
| ----- | -------------------------------- | -------------- |
| I     | Waypoint / track following       | ✅ Validated    |
| II    | Fixed multi-gate racing          | 🚧 In progress |
| III   | Generalized racing               | 📋 Planned     |
| IV    | LiDAR dynamic obstacle avoidance | 📋 Planned     |

See [Roadmap](docs/roadmaps/autonomous-racing-and-avoidance.md) for details.

---

## 📚 References

The implementation and research roadmap are primarily related to the following work:

**[1] Analytic Policy Gradient**

N. Wiedemann, V. Wüest, A. Loquercio, M. Müller, D. Floreano, and D. Scaramuzza,
“Training Efficient Controllers via Analytic Policy Gradient,”
*IEEE International Conference on Robotics and Automation (ICRA)*, 2023.
arXiv:2209.13052.

**[2] Short-Horizon Actor-Critic**

J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg, and M. Macklin,
“Accelerated Policy Learning with Parallel Differentiable Simulation,”
*International Conference on Learning Representations (ICLR)*, 2022.
arXiv:2204.07137.

**[3] DiffAero**

X. Zhang, R. Wang, Y. Ren, J. Sun, H. Fang, J. Chen, and G. Wang,
“DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning,”
arXiv:2509.10247, 2025.

**[4] DiffRacing**

Y. Su, F. Yu, Y. Hu, X. Niu, L. Zhang, F. Sun, and D. Zou,
“Vector Field Augmented Differentiable Policy Learning for Vision-Based Drone Racing,”
arXiv:2603.08019, 2026.

**[5] Dynamic Obstacle Avoidance**

X. Fan, M. Lu, B. Xu, and P. Lu,
“Flying in Highly Dynamic Environments With End-to-End Learning Approach,”
*IEEE Robotics and Automation Letters*, vol. 10, no. 4, pp. 3851–3858, 2025.
DOI: 10.1109/LRA.2025.3547306.

**[6] Point-to-Motion**

B. Xu, Z. Yan, M. Lu, X. Fan, Y. Luo, Y. Lin, Z. Chen, Y. Chen, Q. Qiao, and P. Lu,
“Flow-Aided Flight Through Dynamic Clutters From Point to Motion,”
*IEEE Robotics and Automation Letters*, vol. 11, no. 1, pp. 218–225, 2026.
DOI: 10.1109/LRA.2025.3632608.

**[7] Point-Cloud UAV Navigation**

F. Gao, W. Wu, W. Gao, and S. Shen,
“Flying on Point Clouds: Online Trajectory Generation and Autonomous Navigation for Quadrotors in Cluttered Environments,”
*Journal of Field Robotics*, vol. 36, no. 4, pp. 710–733, 2019.
DOI: 10.1002/rob.21842.

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@inproceedings{wiedemann2023training,
  title     = {Training Efficient Controllers via Analytic Policy Gradient},
  author    = {Wiedemann, Nina and W{\"u}est, Valentin and Loquercio, Antonio
               and M{\"u}ller, Matthias and Floreano, Dario and Scaramuzza, Davide},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023}
}

@inproceedings{xu2022accelerated,
  title     = {Accelerated Policy Learning with Parallel Differentiable Simulation},
  author    = {Xu, Jie and Makoviychuk, Viktor and Narang, Yashraj and Ramos, Fabio
               and Matusik, Wojciech and Garg, Animesh and Macklin, Miles},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}

@misc{zhang2025diffaero,
  title         = {DiffAero: A GPU-Accelerated Differentiable Simulation Framework
                   for Efficient Quadrotor Policy Learning},
  author        = {Zhang, Xinhong and Wang, Runqing and Ren, Yunfan and Sun, Jian
                   and Fang, Hao and Chen, Jie and Wang, Gang},
  year          = {2025},
  eprint        = {2509.10247},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@misc{su2026diffracing,
  title         = {Vector Field Augmented Differentiable Policy Learning
                   for Vision-Based Drone Racing},
  author        = {Su, Yang and Yu, Feng and Hu, Yu and Niu, Xinze
                   and Zhang, Linzuo and Sun, Fangyu and Zou, Danping},
  year          = {2026},
  eprint        = {2603.08019},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{fan2025dynamic,
  title   = {Flying in Highly Dynamic Environments With End-to-End Learning Approach},
  author  = {Fan, Xiyu and Lu, Minghao and Xu, Bowen and Lu, Peng},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {10},
  number  = {4},
  pages   = {3851--3858},
  year    = {2025},
  doi     = {10.1109/LRA.2025.3547306}
}

@article{xu2026p2m,
  title   = {Flow-Aided Flight Through Dynamic Clutters From Point to Motion},
  author  = {Xu, Bowen and Yan, Zexuan and Lu, Minghao and Fan, Xiyu
             and Luo, Yi and Lin, Youshen and Chen, Zhiqiang and Chen, Yeke
             and Qiao, Qiyuan and Lu, Peng},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {11},
  number  = {1},
  pages   = {218--225},
  year    = {2026},
  doi     = {10.1109/LRA.2025.3632608}
}

@article{gao2019pointclouds,
  title   = {Flying on Point Clouds: Online Trajectory Generation and
             Autonomous Navigation for Quadrotors in Cluttered Environments},
  author  = {Gao, Fei and Wu, William and Gao, Wenliang and Shen, Shaojie},
  journal = {Journal of Field Robotics},
  volume  = {36},
  number  = {4},
  pages   = {710--733},
  year    = {2019},
  doi     = {10.1002/rob.21842}
}
```

</details>

---

## 🙏 Acknowledgements

This project builds on **Genesis** and the upstream **GenesisDroneEnv** environment.

The APG and SHAC implementations are informed by [1–3]. Adapted third-party components retain their original licenses; see `THIRD_PARTY_NOTICES`.