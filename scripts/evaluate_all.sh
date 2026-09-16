#!/usr/bin/env bash
# Evaluate every trained run + baselines in the 2-D simulator and build the comparison report.
#   bash scripts/evaluate_all.sh [EPISODES]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-200}
mkdir -p results
# baselines
python3 evaluate.py --policy pctrl       --episodes "$N" --out results/eval_pctrl
python3 evaluate.py --policy pctrl_avoid --episodes "$N" --out results/eval_pctrl_avoid
python3 evaluate.py --policy pctrl_avoid --episodes "$N" --dr --out results/eval_pctrl_avoid_DRdyn
python3 evaluate.py --policy random      --episodes "$N" --out results/eval_random
# trained runs (best model of each run), nominal and perturbed dynamics
for d in runs/*/ runs_nominal/*/; do
  [ -f "$d/best_model.zip" ] || continue
  tag=$(basename "$d")
  # the nominal ablation runs were trained with the measured-velocity feature
  extra=""; case "$tag" in *nominal*) extra='--env-cfg {"vel_obs":"measured"}';; esac
  python3 evaluate.py --model "$d/best_model.zip" --episodes "$N" $extra --out "results/eval_${tag}"
  python3 evaluate.py --model "$d/best_model.zip" --episodes "$N" --dr $extra --out "results/eval_${tag}_DRdyn"
done
# the two shipped models: also fixed goal (2.5, 2.5) as in the report
for m in sac ppo; do
  [ -f "models/${m}_best.zip" ] || continue
  python3 evaluate.py --model "models/${m}_best.zip" --episodes "$N" --out "results/eval_${m}_best"
  python3 evaluate.py --model "models/${m}_best.zip" --episodes "$N" --dr --out "results/eval_${m}_best_DRdyn"
  python3 evaluate.py --model "models/${m}_best.zip" --episodes 50 --fixed-goal --out "results/eval_${m}_best_fixedgoal" \
      --gif "results/${m}_demo.gif" --gif-episodes 3
done
python3 compare.py --runs runs --evals results --out results
echo "report: results/comparison.md"
