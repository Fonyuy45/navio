#!/bin/bash

# ==============================================================================
# NAVIO EVALUATION SCRIPT (using evo)
# ==============================================================================

cd ~/dev/navio

echo "=================================================="
echo "PREPARING RESULTS DIRECTORY"
echo "=================================================="
# Create a 'results' folder if it doesn't already exist
mkdir -p results

# Delete any old plots in the folder so evo doesn't pause and ask to overwrite
rm -f results/*.png
echo " Results folder ready!"
echo ""


echo "=================================================="
echo "CONFIGURING PLOT STYLES AND COLORS"
echo "=================================================="
evo_config set plot_reference_color darkgreen > /dev/null
evo_config set plot_seaborn_palette deep > /dev/null


# 1. Pure black for the ground truth line
evo_config set plot_reference_color green > /dev/null

# 2. 'bright' palette forces pure, highly saturated primary colors (pure blue)
evo_config set plot_seaborn_palette bright > /dev/null

# 3. Clean white background with no grid
#evo_config set plot_seaborn_style whitegrid > /dev/null
evo_config set plot_seaborn_style white > /dev/null

# 4. Turn ON the error lines between the two trajectories! set to true or false to toggle
#evo_config set plot_pose_correspondences false > /dev/null

# 5. Keep the lines thick and readable
evo_config set plot_linewidth 2.0 > /dev/null


echo "=================================================="
echo "GENERATING CLEAN TRAJECTORY PLOT (evo_traj)"
echo "=================================================="
# Added 'results/' before the filename 
evo_traj tum results/estimated_trajectory.txt \
    --ref rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    --plot \
    --plot_mode xy \
    --align \
    --save_plot results/clean_trajectory.png

echo ""
echo "=================================================="
echo "GENERATING ERROR HEATMAP PLOT (evo_ape)"
echo "=================================================="
# Notice we added 'results/' before the filename too
evo_ape tum rgbd_dataset_freiburg1_xyz/groundtruth.txt results/estimated_trajectory.txt \
    --plot \
    --plot_mode xy \
    --align \
    --save_plot results/error_heatmap.png

#  finds any .png in the CURRENT folder (navio) and deletes them.
# It does NOT touch the files inside the /results folder.
rm -f ./*.png

echo ""
echo "=================================================="
echo "Evaluation Complete! Open the 'results' directory to see plots."