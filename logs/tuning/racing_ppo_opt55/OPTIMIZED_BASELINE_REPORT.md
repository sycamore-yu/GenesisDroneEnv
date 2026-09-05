# Genesis optimized PPO baseline report

Assumptions: you know Racing (竞速), PPO (近端策略优化), det = deterministic fixed-state mean passed gates (固定初始状态、确定性策略平均过门数), DiffAero is the benchmark (对照目标) only.

RaceEnv / full_quad / CTBR / eval states were not changed in this phase.

Eval protocol: 100 fixed states, seed 20250830, checksum `8c975c09876e669b...`.

---

## Four answers

### 1. Original RSL-RL 5.5 baseline?

Stock Gaussian PPO, 1024 envs, rollout 16, [256,128] ELU, LR 3e-4, desired_kl 0.01, entropy 0.01, init_std 0.223, adaptive KL, max_grad_norm 1.

| updates | det | sto |
|--------:|----:|----:|
| 200 | 1.38 | 1.01 |
| 500 | 4.02 | 3.89 |
| 1000 | 5.17 | 17.41 |

Logs: `logs/race/rslrl55_gaussian_8gate_*`, `logs/race/rslrl55_gaussian1000_8gate_*`.

### 2. What did Optuna find?

Stage 1 (16 trials, seed0, 200 updates, objective = fixed-state **det**):

Best trial was Stage1 trial2 (det@200 = 4.41), but Stage2 extension of that config collapsed (det@499 = 2.68, sto = 12.9, std = 5.13).

The durable winner after Stage2 was **Stage1 trial3**:

| param | baseline | optimized (trial3) |
|-------|----------|--------------------|
| learning_rate | 3e-4 | 2.110721108616641e-4 |
| desired_kl | 0.01 | 0.015906451881642775 |
| entropy_coef | 0.01 | 0.00875174422525385 |
| init_std | 0.2231301601 | 0.4567092003128319 |

Network / rollout / gamma / lambda / value_loss / max_grad_norm unchanged. Adaptive schedule kept.

Frozen into: `config/race/ppo_full_quad.yaml`.

### 3. How much did optimized baseline improve?

**seed0 (trial3 config)**

| updates | baseline det | optimized det | Δ |
|--------:|-------------:|--------------:|--:|
| 200 | 1.38 | 2.86 | +1.48 |
| 500 | 4.02 | 7.54 | +3.52 |
| 1000 | 5.17 | 14.45 | +9.28 |

seed0 peak mid-run: **det@400 = 15.37** (sto 18.33, std 1.63).

**3-seed confirmation (seeds 0/1/2), same config**

| updates | mean det | std across seeds | individuals |
|--------:|---------:|-----------------:|-------------|
| 200 | 4.48 | 1.38 | 2.86 / 4.36 / 6.23 |
| 400 | 9.93 | 3.85 | 15.37 / 6.96 / 7.45 |
| 500 | **8.47** | **1.76** | 7.54 / 10.94 / 6.94 |

Paths:

- seed0 1000: `logs/tuning/racing_ppo_opt55/stage2/stage1_3_1000/`
- seed1 500: `logs/tuning/racing_ppo_opt55/stage2/stage1_3_seed1_500/`
- seed2 500: `logs/tuning/racing_ppo_opt55/stage2/stage1_3_seed2_500/`
- Stage1 study: `logs/tuning/racing_ppo_opt55/`

### 4. DiffAero same-protocol order?

DiffAero actor on DiffAero (same protocol): **≈ 8.13 det gates**.

| result | vs DiffAero 8.13 |
|--------|------------------|
| optimized 3-seed mean det@500 = 8.47 | **same order / slightly above** |
| seed0 det@400 = 15.37 | above |
| seed0 det@999 = 14.45 | above, but std = 9.37 (not deployable as-is) |

Do **not** use different-protocol “22 gates”.

---

## Formal baseline choice

**Use trial3 PPO hyperparams (now in `ppo_full_quad.yaml`).**

Prefer deployment / reporting checkpoint around **400–500 updates**, not blindly the last 1000 step:

- At ~500: multi-seed mean det ≈ 8.5, std across seeds moderate, action std ~2.
- At 999 (seed0): det still high, but policy std → 9.4 and sto−det gap → +15. Reject as the “stable long run” artifact.

Keep RSL-RL 5.5 stock Gaussian. No custom PPO fork. No LayerNorm / tanh / orthogonal revive.

---

## Bottleneck after this phase

Evidence class: **optimizer / exploration** was the Stage1–2 lever that moved det from 5.17 → DiffAero-order.

Remaining issue if training past ~500 without care: **std / exploration blow-up** (long-run log-std growth), not proven as network-size or 8-gate credit-assignment failure yet.

Stop further PPO grid search. PPO is frozen for APG / SHAC / BaseEnv work unless a new measured failure appears.

---

## Environment

Main conda `genesis`: `rsl-rl-lib==5.5.0` (snapshots under `logs/race/rslrl55_migration/`).
