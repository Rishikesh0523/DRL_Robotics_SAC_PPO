#!/usr/bin/env bash
# Launch the SAC vs PPO experiment (3 seeds each) in the background.
#
#   bash scripts/run_all_training.sh                 # defaults below (domain randomisation ON)
#   DR=0 bash scripts/run_all_training.sh            # nominal dynamics (ablation)
#   SAC_STEPS=300000 PPO_STEPS=2000000 SEEDS="0 1 2" SAC_DEVICE=cuda bash scripts/run_all_training.sh
#
# Logs: runs/<tag>/stdout.log   Monitor: tail -f runs/*/stdout.log   or   tensorboard --logdir runs
set -euo pipefail
cd "$(dirname "$0")/.."
SAC_STEPS=${SAC_STEPS:-300000}
PPO_STEPS=${PPO_STEPS:-2000000}
SEEDS=${SEEDS:-"0 1 2"}
DR=${DR:-1}
SAC_DEVICE=${SAC_DEVICE:-cpu}
SAC_TRAIN_FREQ=${SAC_TRAIN_FREQ:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

suffix=$([ "$DR" = 1 ] && echo "_dr" || echo "_nominal")
drflag=$([ "$DR" = 1 ] && echo "--dr" || echo "")
mkdir -p runs
for seed in $SEEDS; do
  tag="sac${suffix}_seed${seed}"; mkdir -p "runs/$tag"
  nohup python3 train.py --algo sac $drflag --sac-train-freq "$SAC_TRAIN_FREQ" --steps "$SAC_STEPS" --seed "$seed" \
      --tag "$tag" --device "$SAC_DEVICE" --threads 1 --eval-freq 10000 > "runs/$tag/stdout.log" 2>&1 &
  echo "started $tag (pid $!)"
  tag="ppo${suffix}_seed${seed}"; mkdir -p "runs/$tag"
  nohup python3 train.py --algo ppo $drflag --steps "$PPO_STEPS" --seed "$seed" \
      --tag "$tag" --device cpu --threads 1 --eval-freq 20000 > "runs/$tag/stdout.log" 2>&1 &
  echo "started $tag (pid $!)"
done
echo "all runs launched: $(date)"
