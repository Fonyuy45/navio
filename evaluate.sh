#!/bin/bash
cd ~/dev/navio
echo "=================================================="
echo " RUNNING RELATIVE POSE ERROR (RPE) EVALUATION"
echo "=================================================="
python3 scripts/evaluate_rpe.py \
    rgbd_dataset_freiburg2_desk/groundtruth.txt \
    results_fr2/estimated_trajectory.txt \
    --verbose
echo ""
echo "=================================================="
echo " RUNNING ABSOLUTE TRAJECTORY ERROR (ATE) EVALUATION"
echo "=================================================="
python3 scripts/evaluate_ate.py \
    rgbd_dataset_freiburg2_desk/groundtruth.txt \
    results_fr2/estimated_trajectory.txt \
    --verbose \
    --plot results_fr2/ate_plot.png
echo "=================================================="
echo " Evaluation Complete! (Check results_fr2/ directory for ate_plot.png)"