# GenesisDroneEnv Racing PPO: RSL-RL 3.1.3 → 5.5.0 staged report

Assumptions (假设，事先说明): you can read a technical log. You already know Racing (竞速任务), PPO (近端策略优化), and DiffAero (对照仓库). This file records measurements, not a claim that Genesis equals DiffAero 22 gates.

All numbers point to real log paths. Different RSL-RL major versions, eval protocols, and physics commits are not treated as one-variable experiments.

---

## 0. Protection snapshot

Do not overwrite 3.1.3 evidence.

| Item | Value | Path |
| Git commit at record | `23b8e1b5f9521b7408f8bd8fc7da920dcc9fd38c` | `logs/race/rslrl55_migration/env_snapshot_3.1.3.json` |
| Branch (3.1.3 record) | `main` | same |
| Migration branch | `rsl-rl-5.5-migration` | git |
| Python | 3.11.16 | both envs |
| torch | 2.13.0+cu130 | both |
| CUDA | 13.0, RTX 4090 | both |
| Genesis | 1.3.3 | both |
| tensordict | 0.14.0 | both |
| rsl-rl-lib (kept) | 3.1.3 in conda `genesis` | snapshot |
| rsl-rl-lib (new) | 5.5.0 in conda `genesis-rslrl55` | `logs/race/rslrl55_migration/env_snapshot_5.5.0.json` |

`pip --dry-run` planned only `rsl-rl-lib==5.5.0`. No torch/CUDA replace. `pip check`: no broken requirements. Genesis GPU smoke: `cuda:0`.

---

## 1. 3.1.3 → 5.5.0 migration changes

RSL-RL 5.5 native stack:

`OnPolicyRunner → PPO → actor MLPModel + Distribution, critic MLPModel, RolloutStorage`

Project stack:

`RaceEnv / GenesisTaskEnv → RslRlAdapter → RSL-RL 5.5`

Changed for contract, not a PPO copy:

- `config/race/ppo.yaml`, `config/race/ppo_full_quad.yaml`: `actor` / `critic` / `obs_groups` / `GaussianDistribution`
- `config/track_rl/rl_env.yaml`, `config/se3_controller_eval/rl_env.yaml`: same 5.5 keys (required after upgrade)
- `genesis_drones/experiment/builder.py`: `build_ppo_actor()` via `resolve_class`, load `actor_state_dict`
- `genesis_drones/experiment/train_loop.py`: `runner.alg.save()`
- `genesis_drones/adapters/policy.py`: `actor(obs, stochastic_output=...)`
- `genesis_drones/adapters/rsl_rl_adapter.py`: TensorDict `batch_size=[num_envs]`
- `genesis_drones/evaluation/race.py`, `scripts/eval/race_eval.py`: 5.5 actor fields

Kept:

- rollout 16, gamma 0.99, lambda 0.95, clip 0.2, epochs 4, mini-batches 8
- entropy 0.01, value_loss_coef 2, adaptive KL 0.01, max_grad_norm 1
- MLP `[256,128]`, ELU, obs_normalization false
- Gaussian state-independent log std, `init_std=0.2231301601`
- LR `3e-4` from the 3.1.3 Optuna/one-gate verified value, not DiffAero yaml `0.0026`

Did not copy `PPO.update`. Did not edit site-packages.

---

## 2. Old vs new contract

| | RSL-RL 3.1.3 | RSL-RL 5.5.0 |
| obs groups | `policy` | `actor` / `critic` sets; TensorDict keys still `policy` |
| checkpoint | combined `model_state_dict` | `actor_state_dict` + `critic_state_dict` |
| det action | `act_inference` | `actor(..., stochastic_output=False)` |
| grad clip | joint actor+critic | separate `max_grad_norm` on each |
| timeout bootstrap | `r += gamma * V` on `time_outs` using **current** transition value | **same** (`tests/test_rsl_rl55_contract.py`) |

Timeout parity is **not** solved by the upgrade. Confirmed in 5.5 `PPO.process_env_step`.

Tests: `tests/test_rsl_rl55_contract.py` (shapes, yaml, timeout, separate clip, LayerNorm/orthogonal loaders). `tests/test_tanh_gaussian.py` (8 passed).

---

## 3. RSL-RL 5.5 stock baseline (migration baseline)

Not a single-mechanism claim. Whole RSL-RL major version changed.

Log: `logs/race/rslrl55_gaussian_8gate_2026-09-03_08-42-21/`  
Replicated 1000-update run (same seed/config): `logs/race/rslrl55_gaussian1000_8gate_2026-09-03_09-33-49/`

200-update training rollout (TensorBoard, not the fixed-state protocol):

| update | n_passed_gates | reward | std |
| 49 | 0.21 | -3.60 | 0.23 |
| 99 | 0.41 | 2.16 | 0.27 |
| 149 | 0.79 | 13.12 | 0.30 |
| 199 | 1.13 | 25.38 | 0.34 |

KL last ≈ 0.013, LR last ≈ 5.8e-4, actor/critic post-clip grad ≈ 1.0.

---

## 4. One-gate result

Question: can RSL-RL 5.5 stock PPO learn one-gate?

**Yes.**

Path: `logs/race/rslrl55_onegate_2026-09-03_08-37-27/onegate_eval.json`

- 80 updates, 1024 envs, LR 3e-4, `max_grad_norm=1`
- deterministic pass: **1.0** (64/64)
- stochastic pass: **0.797**
- KL last 0.012, LR last 2.56e-4, std last 0.258
- actor/critic post-clip grad ≈ 1.0

Contrast: 3.1.3 stock one-gate = 0 gates (`logs/race/diag_onegate_ppo_2026-09-03_05-17-07`). 3.1.3 needed `max_grad_norm=100` (`logs/race/diag_onegate_grad100_2026-09-03_05-55-38`) to get mean_det_gates=1.0.

5.5 separate clipping removes the 3.1.3 joint-clip one-gate failure. That does **not** by itself explain 8-gate vs DiffAero.

---

## 5. 8-gate 0/50/100/150/200/1000 curve

Protocol: 100 fixed initial states, checksum `8c975c09876e669b...` (same family as prior 8-gate evals).  
DiffAero eval in source uses `test=True` → `tanh(mean)` = **deterministic**. Fair DiffAero compare is **deterministic gates**.

### 5.5 Gaussian stock, fixed-state eval

Path: `logs/race/rslrl55_gaussian1000_8gate_2026-09-03_09-33-49/det_sto_eval.json`  
(200-update file matches 0–200: `logs/race/rslrl55_gaussian_8gate_2026-09-03_08-42-21/det_sto_eval.json`)

| update | det gates | sto gates | DiffAero (cited) | old Genesis SplitClip (cited, training gates) |
| 0 | 0.00 | 0.00 | — | — |
| 50 | 0.00 | 0.05 | — | — |
| 100 | 0.30 | 0.29 | ≈ 1.20 | ≈ 0.39 |
| 150 | 0.83 | 0.89 | ≈ 1.68 | ≈ 0.31 |
| 200 | 1.38 | 1.01 | ≈ 4.79 | ≈ 1.13 |
| 500 | 4.02 | 3.89 | — | — |
| 999 | 5.17 | 17.41 | — | — |

Training rollout at 999: 16.68 gates (`Episode/n_passed_gates` in the same TensorBoard). Close to stochastic fixed-state 17.41. Far from deterministic 5.17.

---

## 6. Deterministic vs stochastic (same protocol)

Same checkpoint, same 100 states, same eval loop.

At **200**: det 1.38 ≥ sto 1.01. This does **not** support “policy depends on exploration noise”.

At **1000**: sto 17.41 >> det 5.17. This **does** support that claim **at long training**, together with clipping (section 7). Do not mix training-rollout gates with deterministic eval to make this claim.

Collision / survival / return / length at 999 (Gaussian):

| | det | sto |
| gates | 5.17 | 17.41 |
| collision | 0.86 | 0.80 |
| survival (truncated, includes OOB `|x|,|y|>5` or `z>7`) | 0.14 | 0.20 |
| episode time s | 4.96 | 14.81 |
| return | 36.6 | 238.8 |

---

## 7. Action saturation / std / entropy

Gaussian 1000, same eval JSON:

| update | std | det `|a|>1` | sto `|a|>1` | det env-clip mean | sto env-clip mean |
| 0 | 0.223 | 0.014 | 0.037 | 0.001 | 0.005 |
| 200 | 0.338 | 0.033 | 0.069 | 0.013 | 0.018 |
| 500 | 1.090 | 0.225 | 0.508 | 0.198 | 0.478 |
| 999 | 2.685 | 0.622 | 0.792 | 1.940 | 2.688 |

Std 0.22 → 2.68 stays inside DiffAero `LOG_STD` range `[-5, 2]` → `[0.0067, 7.39]`. This is **not** a “std bound repair” result.

Entropy (TB): -0.32 → 1.30 (200) → 9.04 (999). KL last still ≈ 0.014. Adaptive LR last 1.3e-3.

---

## 8. Distribution experiments

Trigger at 200: **no** (sto not >> det; `|a|>1` small).  
Trigger at 1000: **yes**. Implemented local `TanhGaussianDistribution` via `class_name` (`genesis_drones/algorithms/tanh_gaussian.py`). No site-packages edit. No PPO copy.

Math used:

- sample: `a = tanh(z)`, `z ~ Normal(mu, sigma)`
- det: `tanh(mu)`
- log_prob: base log_prob minus `log(1-a^2)` Jacobian
- KL for adaptive LR: KL of the **pre-tanh** Gaussians (equals KL of the squashed laws because tanh is a diffeomorphism)
- entropy: DiffAero-style **base Gaussian entropy**

Tests: `tests/test_tanh_gaussian.py`.

### Tanh 200

`logs/race/rslrl55_tanh_8gate_2026-09-03_10-23-13/det_sto_eval.json`

| update | det | sto | `|a|>1` |
| 100 | 0.14 | 0.08 | 0 |
| 150 | 0.31 | 0.58 | 0 |
| 199 | 0.87 | 0.83 | 0 |

Hard clipping gone. det ≈ sto. **det 0.87 < Gaussian 1.38**. Hypothesis rejected as a 200-update improvement.

### Tanh 1000 (crashed)

`logs/race/rslrl55_tanh1000_8gate_2026-09-03_10-34-42/`  
Crash at iteration 417: `RuntimeError: normal expects all elements of std >= 0.0` (NaN std). Training gates ≈ 1.4. Last checkpoints evaluated:

| update | det | sto | `|a|>1` |
| 200 | 0.87 | 0.88 | 0 |
| 350 | 1.23 | 1.12 | 0 |
| 400 | 1.33 | 1.30 | 0 |

Gaussian at 500 already has det **4.02**. Tanh at 400 is still **1.33**. Tanh does not close the long-horizon gap on this stack. A numerical guard was added after the crash (`nan_to_num` + `std.clamp_min(1e-6)`). That guard was **not** used in the crashed run. No second 1000 after the crash.

---

## 9. Std parameterization experiments

Not run as a “std_max=0.5/1.0” knob. Std at 1000 is inside DiffAero’s legal range. Official 5.5 `std_range` is clamp, not DiffAero’s tanh map. No evidence that a bound repair is the 200-update gap.

---

## 10. Network experiments

Observation scale first: `logs/race/rslrl55_migration/obs_stats.json`  
max |obs| = 10.37, fraction `|x|>10` = 0.3%, `|x|>100` = 0. Skipped `obs_normalization`.

DiffAero MLP source: LayerNorm + orthogonal hidden (`gain=√2`) + last layer `gain=0.01` + ELU. Hidden sizes already `[256,128]`.

### LayerNorm only

`logs/race/rslrl55_layernorm_8gate_2026-09-03_09-01-17/det_sto_eval.json`

199 det **0.77** vs Gaussian **1.38**. Rejected.

### Orthogonal hidden init only

`logs/race/rslrl55_orthogonal_8gate_2026-09-03_09-18-36/det_sto_eval.json`

199 det **1.58** vs Gaussian **1.38**. Within noise. Not a clear positive. No 1000.

Last-layer gain 0.01 not tested (initial `|mu|` already moderate). Combined LN+ortho+tanh not tested (would not be one variable).

---

## 11. Critic / value diagnostics

5.5 clips actor and critic separately at 1. Post-clip actor/critic grad ≈ 1.0 on 200-update runs (probe hooks `optimizer.step`, after clip). Orthogonal 200: actor post-clip 0.92 (sometimes under 1).

Value loss (Gaussian TB): 66 → 9.9 (200) → 35 (999) as returns grew. Critic is fitting, not frozen. No critic-no-clip experiment: no pre-clip grad or explained-variance series showing critic clip as the bottleneck. Joint-clip hack not restored.

---

## 12. Unified DiffAero comparison

| | DiffAero | Genesis 5.5 Gaussian | notes |
| eval action | `test=True` → tanh(mean) | det = raw mean; env `clamp(-1,1)` | not identical semantics |
| 100 det gates | ≈ 1.20 | 0.30 | |
| 150 det gates | ≈ 1.68 | 0.83 | |
| 200 det gates | ≈ 4.79 | 1.38 | |
| 1000 det gates | not measured here | 5.17 | Genesis 1000 ≈ DiffAero 200 |
| 1000 sto gates | not DiffAero eval | 17.41 | matches training ~16.7 |
| best DiffAero actor on DiffAero | 8.13 gates | — | `logs/race/diag_diffaero_actor_on_diffaero.json` |
| same actor on Genesis | 3.71 gates | — | `logs/race/diag_diffaero_actor_on_genesis.json` |
| open-loop replay | 12 vs Genesis 2 gates | — | `logs/race/diag_openloop_replay.json` |

3.1.3 adaptkl det eval stayed 0.08–0.42 (`logs/race/ppo_full_quad_adaptkl_seed0/det_eval.json`). 5.5 stock is a real step up from that 3.1.3 det curve. It is not a strict one-variable vs SplitClip because the RSL-RL major version changed.

---

## 13. Excluded root causes (for the 200-update 8-gate gap vs DiffAero 4.79)

| Hypothesis | Why excluded |
| 5.5 cannot learn Racing at all | one-gate det 1.0; 8-gate 200 det 1.38 rising |
| 3.1.3 joint clip still blocking 5.5 | 5.5 clips separately; one-gate works at clip=1 |
| obs scale needs EmpiricalNormalization | obs mostly O(1), max 10.4 |
| policy depends on noise at 200 | det ≥ sto at 200 |
| Gaussian hard-clip at 200 | `|a|>1` 3–7% |
| LayerNorm missing | 200 det 0.77, worse |
| orthogonal hidden init missing | 200 det 1.58, not a clear gain |
| tanh squash as 200 fix | 200 det 0.87, worse; 400 det 1.33 vs Gaussian 500 det 4.02 |
| std outside DiffAero legal range | 0.22–2.68 ⊂ [0.0067, 7.39] |
| critic joint clip | gone in 5.5; critic value loss falls 66→9.9 by 200 |

---

## 14. Remaining main gap

1. **Sample inefficiency vs DiffAero at the 200-update mark:** det 1.38 vs 4.79. Genesis reaches DiffAero’s 200 det (~4.8) only near update 500–1000.
2. **Deterministic vs stochastic split after std grows:** at 1000, env hard-clip turns Gaussian noise into a de-facto squash; det 5.17 vs sto 17.41. Tanh removes clip but did not beat Gaussian det, then NaN-crashed.
3. **Simulator gap still real for transferred DiffAero policies:** 8.13 → 3.71 gates; open-loop 12 → 2. This does **not** cap Genesis-trained PPO (det 5.17 > transfer 3.71; sto 17 > 3.71).

Official 5.5 Racing PPO baseline: **stock Gaussian MLP, LR 3e-4, clip=1, 1024 envs, rollout 16**. Use `config/race/ppo_full_quad.yaml`. Long run: `logs/race/rslrl55_gaussian1000_8gate_2026-09-03_09-33-49/`.

---

## 15. Staged class: **A (PARITY-ENOUGH)** with a time-scale caveat

**A**, not “Genesis = DiffAero 22 gates”.

Evidence:

- 5.5 stock PPO learns one-gate and 8-gate without joint-clip hacks.
- 1000-update **deterministic** gates 5.17 sit at the same order as DiffAero’s cited **200-update** 4.79.
- 1000-update **stochastic / training** gates ~17 sit at the same order as the 22-gate DiffAero success target.
- LayerNorm / orthogonal / tanh did not produce a larger one-variable jump than “train longer on stock 5.5”.

Counter-evidence:

- At the **same 200 updates**, det 1.38 vs 4.79. This is not A at the DiffAero-200 checkpoint.
- DiffAero 1000 deterministic curve was not measured here. If DiffAero det is already ~22 at 1000, Genesis det 5.17 is still behind.
- Gaussian 1000 det/sto gap and clipping are PPO-semantics leftovers, not simulator.

Already excluded: see §13.

Not excluded:

- DiffAero LR 0.0026 vs 3e-4 (source-driven, one variable)
- LayerNorm **plus** orthogonal **plus** last-layer 0.01 together
- DiffAero tanh-mapped log_std **plus** a numerically stable tanh Gaussian
- rollout length / reset / curriculum (C-style), after the 200-update deficit
- integrator/dynamics mismatch for **transferred** policies (D), already measured

This is **not** C as a hard 8-gate failure: 1000-update 8-gate learns.  
This is **not** B as a successful single-variable fix: tanh did not raise det.  
This is **not** D as “PPO tuning is pointless”: Genesis-trained PPO beats the transferred DiffAero actor on Genesis.

---

## 16. Next highest-value work

1. Keep `rslrl55_gaussian1000_8gate_2026-09-03_09-33-49` as the official 5.5 baseline. Compare future changes to **its** det/sto JSON, not to 3.1.3.
2. Measure DiffAero **1000-update deterministic** gates on the same 100 states / checksum if you need a same-horizon DiffAero number.
3. If you still want tanh: do not rerun the crashed recipe. Combine Jacobian squash with DiffAero’s tanh log_std map, plus the NaN guard, then 200 then 1000. Expect this to be a **stability** job, not a free score.
4. One source-driven LR probe (`0.0026` vs `3e-4`) at 200 updates, nothing else changed.
5. Do not change DiffAero reward / gate / termination to chase 22 gates on Genesis. Simulator transfer/replay already show dynamics divergence for copied trajectories.

---

## Failed / weaker runs kept

| run | path | 200 det |
| Gaussian 200 | `logs/race/rslrl55_gaussian_8gate_2026-09-03_08-42-21` | 1.38 |
| Gaussian 1000 | `logs/race/rslrl55_gaussian1000_8gate_2026-09-03_09-33-49` | 1.38 / 5.17@999 |
| LayerNorm | `logs/race/rslrl55_layernorm_8gate_2026-09-03_09-01-17` | 0.77 |
| Orthogonal | `logs/race/rslrl55_orthogonal_8gate_2026-09-03_09-18-36` | 1.58 |
| Tanh 200 | `logs/race/rslrl55_tanh_8gate_2026-09-03_10-23-13` | 0.87 |
| Tanh 1000 crash | `logs/race/rslrl55_tanh1000_8gate_2026-09-03_10-34-42` | 0.87; 1.33@400; NaN@417 |
