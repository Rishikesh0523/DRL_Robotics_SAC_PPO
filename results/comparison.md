# SAC vs PPO vs P-controller: 2-D arena results

## Training runs

| run | algo | seed | steps | wall-clock [min] | steps/s | best-model success | best-model return |
|---|---|---|---|---|---|---|---|
| ppo_dr_seed0 | PPO | 0 | 2,000,000 | 46.9 | 711 | 99.0% | 37.16 |
| ppo_dr_seed1 | PPO | 1 | 2,000,000 | 47.0 | 710 | 93.0% | 33.47 |
| ppo_dr_seed2 | PPO | 2 | 2,000,000 | 46.9 | 710 | 98.0% | 36.76 |
| sac_dr_seed0 | SAC | 0 | 300,000 | 63.1 | 79 | 98.0% | 36.68 |
| sac_dr_seed1 | SAC | 1 | 300,000 | 63.5 | 79 | 98.0% | 35.63 |
| sac_dr_seed2 | SAC | 2 | 300,000 | 63.3 | 79 | 100.0% | 37.47 |
| sac_dr_seed3 | SAC | 3 | 300,000 | 40.7 | 123 | 98.0% | 35.86 |

## Evaluation (deterministic policy, random start/goal unless noted)

*dynamics*: nominal = ideal simulator dynamics; perturbed = domain-randomised dynamics (speed loss, lateral slip, delays, jittered control period) that mimic the Gazebo robot.

| policy | dynamics | episodes | success | collision | timeout | mean return | steps to goal | time to goal [s] | path eff. | SPL | mean abs w [rad/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pctrl | nominal | 200 | 61.5% | 38.5% | 0.0% | 16.91 | 43.76 | 5.25 | 0.99 | 0.61 | 0.43 |
| pctrl_avoid | nominal | 200 | 97.0% | 0.0% | 3.0% | 35.20 | 60.39 | 7.25 | 0.95 | 0.92 | 0.33 |
| pctrl_avoid (fixed goal 2.5,2.5) | nominal | 50 | 86.0% | 0.0% | 14.0% | 32.48 | 68.67 | 8.24 | 0.93 | 0.80 | 0.27 |
| pctrl_avoid | perturbed | 200 | 95.5% | 0.0% | 4.5% | 33.44 | 81.28 | 9.75 | 0.92 | 0.87 | 0.29 |
| ppo_best | nominal | 200 | 96.0% | 3.5% | 0.5% | 35.66 | 35.54 | 4.26 | 0.87 | 0.83 | 0.79 |
| ppo_best (fixed goal 2.5,2.5) | nominal | 50 | 96.0% | 4.0% | 0.0% | 38.00 | 43.67 | 5.24 | 0.84 | 0.81 | 0.79 |
| ppo_best | perturbed | 200 | 95.0% | 4.5% | 0.5% | 34.51 | 48.39 | 5.81 | 0.82 | 0.78 | 0.88 |
| ppo_dr_seed0 | nominal | 200 | 98.5% | 1.5% | 0.0% | 36.87 | 35.16 | 4.22 | 0.86 | 0.85 | 0.90 |
| ppo_dr_seed0 | perturbed | 200 | 96.5% | 3.5% | 0.0% | 35.52 | 43.98 | 5.28 | 0.83 | 0.80 | 0.95 |
| ppo_dr_seed1 | nominal | 200 | 96.0% | 3.5% | 0.5% | 35.66 | 35.54 | 4.26 | 0.87 | 0.83 | 0.79 |
| ppo_dr_seed1 | perturbed | 200 | 95.0% | 4.5% | 0.5% | 34.51 | 48.39 | 5.81 | 0.82 | 0.78 | 0.88 |
| ppo_dr_seed2 | nominal | 200 | 98.5% | 1.5% | 0.0% | 37.18 | 34.47 | 4.14 | 0.86 | 0.85 | 0.76 |
| ppo_dr_seed2 | perturbed | 200 | 96.5% | 3.5% | 0.0% | 35.69 | 43.92 | 5.27 | 0.82 | 0.79 | 0.84 |
| ppo_nominal_seed0_best | nominal | 200 | 98.5% | 1.5% | 0.0% | 37.40 | 31.76 | 3.81 | 0.91 | 0.90 | 0.78 |
| ppo_nominal_seed0_best | perturbed | 200 | 90.5% | 9.5% | 0.0% | 32.06 | 50.10 | 6.01 | 0.74 | 0.67 | 0.81 |
| random | nominal | 200 | 1.0% | 52.5% | 46.5% | -28.76 | 198.50 | 23.82 | 0.13 | 0.00 | 0.75 |
| sac_best | nominal | 200 | 99.5% | 0.5% | 0.0% | 37.22 | 40.28 | 4.83 | 0.85 | 0.85 | 0.75 |
| sac_best (fixed goal 2.5,2.5) | nominal | 50 | 100.0% | 0.0% | 0.0% | 40.16 | 46.34 | 5.56 | 0.82 | 0.82 | 0.73 |
| sac_best | perturbed | 200 | 99.0% | 1.0% | 0.0% | 36.61 | 48.01 | 5.76 | 0.84 | 0.83 | 0.83 |
| sac_dr_seed0 | nominal | 200 | 100.0% | 0.0% | 0.0% | 37.38 | 42.80 | 5.14 | 0.82 | 0.82 | 0.77 |
| sac_dr_seed0 | perturbed | 200 | 100.0% | 0.0% | 0.0% | 37.02 | 49.97 | 6.00 | 0.83 | 0.83 | 0.86 |
| sac_dr_seed1 | nominal | 200 | 99.5% | 0.5% | 0.0% | 37.22 | 40.28 | 4.83 | 0.85 | 0.85 | 0.75 |
| sac_dr_seed1 | perturbed | 200 | 99.0% | 1.0% | 0.0% | 36.61 | 48.01 | 5.76 | 0.84 | 0.83 | 0.83 |
| sac_dr_seed2 | nominal | 200 | 99.5% | 0.5% | 0.0% | 37.51 | 36.69 | 4.40 | 0.87 | 0.87 | 0.77 |
| sac_dr_seed2 | perturbed | 200 | 99.5% | 0.5% | 0.0% | 37.14 | 44.58 | 5.35 | 0.87 | 0.86 | 0.86 |
| sac_dr_seed3 | nominal | 200 | 99.5% | 0.5% | 0.0% | 37.03 | 40.77 | 4.89 | 0.85 | 0.84 | 0.76 |
| sac_dr_seed3 | perturbed | 200 | 97.5% | 2.5% | 0.0% | 35.76 | 48.98 | 5.88 | 0.84 | 0.82 | 0.83 |

Metric definitions: *success* = within 0.30 m of the goal; *collision* = LiDAR min range < 0.20 m or body contact; *timeout* = 300 steps (36 s); *path eff.* = shortest/actual path on successful episodes; *SPL* = success weighted by path length (0 on failure); one step = 0.12 s.
