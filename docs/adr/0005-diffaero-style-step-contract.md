# 可微环境采用 DiffAero 风格返回契约

可微环境的 `step()` 返回 `obs, (physics_loss, policy_loss, reward), done, extras`：`physics_loss` 只包含依赖 Genesis 状态的任务损失，
由 `Scene.backward()` 反向传播；`policy_loss` 只包含直接依赖策略输出的正则项，由普通 PyTorch 反向传播。两者的梯度在同一次
策略更新中相加。`reward` 是独立连续分数且立即切断梯度，`done` 只表示新终止或完整回合截断事件，持续存活状态位于
`extras["alive"]`。该契约避免 `Scene.backward(retain_graph=True)` 保留直接策略图，并取代原单一 `loss` 契约。

`reward` 默认采用 `1 - Σ(reward_weight[i] * continuous_penalty[i])`。它覆盖位置、目标速度、倾斜、偏航、三轴角速度、相邻动作变化
和连续安全边界；默认权重与对应损失相同，但配置字段独立。固定坠毁损失不进入 `reward`，到达和坠毁继续作为独立任务指标。
