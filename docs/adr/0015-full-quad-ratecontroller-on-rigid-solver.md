# 竞速完整四旋翼使用 RateController 外力和 Genesis 刚体求解器

DiffAero 上游提交 291ea14 的 Racing + quad + PPO 路径用 RateController 把策略动作变成总推力和机体系力矩，再用自己的 RK4 积分。Genesis 复现保留这条控制语义，但用 RigidSolver（刚体求解器）做状态积分，不复制 RK4。当前竞速直接改现有 `RaceEnv`，不新增独立兼容环境。
