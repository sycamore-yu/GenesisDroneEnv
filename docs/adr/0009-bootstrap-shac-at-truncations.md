# SHAC 在截断边界继续自举

SHAC 的可微窗口结束和 1500 步时间上限都属于截断，不是坠毁。

演员窗口目标：

$$
L_\pi=\sum_{t=0}^{H-1}\gamma^t L_t+\gamma^H V_\phi(s_H)
$$

其中 $V_\phi(s_H)$ 对 $s_H$ 保持梯度，评论家参数在演员更新时冻结；只有真实坠毁将 $V(s_H)$ 清零。评论家 TD-lambda 目标在截断边界同样自举。该语义沿用 DiffAero，避免学习“短窗口之后世界消失”的错误有限视野。可用 `use_terminal_value: false` 复现缺少终点价值梯度的旧演员目标。
