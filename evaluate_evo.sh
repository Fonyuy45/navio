#!/bin/bash
cd ~/dev/navio
echo "=================================================="
echo "PREPARING RESULTS DIRECTORY"
echo "=================================================="
mkdir -p results_v2
rm -f results_fr2/*.png
echo " Results folder ready!"
echo ""
echo "=================================================="
echo "CONFIGURING PLOT STYLES AND COLORS"
echo "=================================================="
evo_config set plot_reference_color green > /dev/null
evo_config set plot_seaborn_palette bright > /dev/null
evo_config set plot_seaborn_style white > /dev/null
evo_config set plot_linewidth 2.0 > /dev/null
echo "=================================================="
echo "GENERATING CLEAN TRAJECTORY PLOT (evo_traj)"
echo "=================================================="
evo_traj tum results_fr2/estimated_trajectory.txt \
    --ref rgbd_dataset_freiburg2_desk/groundtruth.txt \
    --plot \
    --plot_mode xy \
    --align \
    --save_plot results_fr2/clean_trajectory.png
echo ""
echo "=================================================="
echo "GENERATING ERROR HEATMAP PLOT (evo_ape)"
echo "=================================================="
evo_ape tum rgbd_dataset_freiburg2_desk/groundtruth.txt \
    results_fr2/estimated_trajectory.txt \
    --plot \
    --plot_mode xy \
    --align \
    --save_plot results_fr2/error_heatmap.png
rm -f ./*.png
echo ""
echo "=================================================="
echo "Evaluation Complete! Open the 'results_fr2' directory to see plots."