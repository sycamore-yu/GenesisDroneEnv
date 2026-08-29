# 可微环境采用 DiffAero 风格返回契约

可微环境的 `step()` 返回 `obs, (loss, reward), done, extras`：`loss` 保留 Genesis 梯度并由 APG/SHAC 最小化，`reward` 是独立连续分数且立即切断梯度，`done` 只表示新终止或完整回合截断事件，持续存活状态位于 `extras["alive"]`。该契约不伪装成 RSL-RL 兼容接口，直接对应选择性移植的 DiffAero 算法语义，并取代 ADR-0004。

`reward` 默认采用 `1 - Σ(reward_weight[i] * continuous_penalty[i])`。它覆盖位置、目标速度、倾斜、偏航、三轴角速度、相邻动作变化和连续安全边界；默认权重与对应损失相同，但配置字段独立。固定坠毁损失不进入 `reward`，到达和坠毁继续作为独立任务指标。
