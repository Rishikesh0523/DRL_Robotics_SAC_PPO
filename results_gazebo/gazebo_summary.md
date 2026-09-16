# Gazebo evaluation (drl_inference_node.py)

| policy | episodes | success | collision | timeout | steps_to_goal | time_to_goal_s | path_eff | mean_start_dist |
|---|---|---|---|---|---|---|---|---|
| pctrl | 20 | 40% | 25% | 35% | 137.6 | 16.5 | 0.51 | 3.79 |
| pctrl_avoid | 20 | 55% | 0% | 45% | 106.0 | 12.7 | 0.55 | 3.79 |
| ppo_dr | 20 | 65% | 30% | 5% | 95.2 | 11.4 | 0.44 | 3.79 |
| ppo_dr_fixedgoal | 10 | 90% | 10% | 0% | 80.0 | 9.6 | 0.45 | 3.79 |
| ppo_dr_speed0.7 | 20 | 75% | 15% | 10% | 63.4 | 7.6 | 0.71 | 3.79 |
| ppo_nominal | 20 | 5% | 75% | 20% | 152.0 | 18.2 | 0.06 | 3.79 |
| sac_dr | 20 | 90% | 10% | 0% | 44.2 | 5.3 | 0.69 | 3.79 |
| sac_dr_fixedgoal | 10 | 90% | 10% | 0% | 58.2 | 7.0 | 0.61 | 3.79 |
