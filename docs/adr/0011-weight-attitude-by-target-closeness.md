# 倾斜和偏航损失随航点接近度增强

复用位置损失中的 `closeness = exp(-distance)`，定义 `position_loss = 1 - closeness`。几何倾斜损失 `1 - body_z·world_z` 和世界零偏航损失 `1 - cos(yaw)` 都乘以 `closeness`。远处允许无人机倾斜和转向以加速，接近航点时逐步恢复水平姿态与世界零偏航；不增加接近半径参数，也不重复计算指数。
