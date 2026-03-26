#!/bin/bash

# Navigate to the project root just to be safe
cd ~/dev/navio

echo "=================================================="
echo " RUNNING RELATIVE POSE ERROR (RPE) EVALUATION"
echo "=================================================="
python3 scripts/evaluate_rpe.py \
    rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    results/estimated_trajectory.txt \
    --verbose

echo ""
echo "=================================================="
echo " RUNNING ABSOLUTE TRAJECTORY ERROR (ATE) EVALUATION"
echo "=================================================="
python3 scripts/evaluate_ate.py \
    rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    results/estimated_trajectory.txt \
    --verbose \
    --plot results/ate_plot.png  # <-- Added a bonus flag to draw a graph!
    
echo "=================================================="
echo " Evaluation Complete! (Check results/ directory for ate_plot.png)"