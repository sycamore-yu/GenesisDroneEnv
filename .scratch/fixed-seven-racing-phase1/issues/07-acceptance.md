# 07 — 固定七门可靠性验收

**What to build（实施目标）：** 完成第一阶段正式实验，并生成可复现结果。只有三种算法都通过可靠性门槛，后续赛道阶段才可以开始。

**Blocked by（前置工单）：** 06 — 三算法统一评价与真实碰撞验证。

**Status（状态）：** ready-for-human（需要人工）

- [x] 每种算法训练入口支持 3 个独立训练种子。
- [x] 每个模型使用同一 100 回合评价集合。
- [ ] 每个算法完整七门成功率至少 90%。
- [ ] 每个算法碰撞率不高于 5%。
- [x] 记录配置、检查点、随机种子、代码版本和全部评价指标。
- [ ] APG（解析策略梯度）报告 32、64、96 步窗口结果，并据此选择第一阶段默认窗口。

## Comments（讨论）

验收脚本 `scripts/eval/race_accept.py` 已记录种子、代码版本和门槛。`--smoke` 只检查记录格式。完整 3 种子训练和 90% / 5% 数字需要 GPU（图形处理器）长时间运行，本会话没有跑完。

请在本机运行：

```
python scripts/train/race_train.py --algo ppo --seed 1
python scripts/train/race_train.py --algo apg --horizon 32 --seed 1
python scripts/eval/race_eval.py --algo ppo --checkpoint <ckpt> --states config/race/eval_states.pt --enable-gate-contact
python scripts/eval/race_accept.py
```
