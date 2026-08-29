# 选择性移植 DiffAero 算法核心

APG、SHAC、TD-λ、必要缓冲区及 Actor/Critic 核心从 BSD-3-Clause 许可的 DiffAero 选择性移植，以降低重新实现算法公式和边界的错误风险；移植文件保留原版权、许可和修改说明。删除 OmegaConf、导出器、RNN、多智能体和 DiffAero Runner 等无关部分，Genesis 的 `alive` 吸收状态、可微窗口、`scene.backward()` 和完整回合重置由本项目实现。DLO-Lab 只作为窗口生命周期参考，不移植其绳索手工梯度链。
