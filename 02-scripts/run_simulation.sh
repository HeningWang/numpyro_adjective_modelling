#!/bin/bash

# Set the locale to ensure decimal points are interpreted correctly
export LC_NUMERIC="en_US.UTF-8"

# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
nobj_list=$(seq 2 4 30)                      # 2, 6, 10, 14, 18, 22, 26, 30  (8 values)
speaker_list=("incremental_speaker" "global_speaker")
color_semvalue_list=$(seq 0.9 0.02 0.99)     # 0.90, 0.92, 0.94, 0.96, 0.98  (5-6 values)
k_list=$(seq 0.1 0.2 0.99)                   # 0.1, 0.3, 0.5, 0.7, 0.9       (5 values)
size_distribution_list=("normal")
sd_spread_list=(2.0 7.75 15.0)               # blurred / baseline / sharp     (3 values)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
sample_size=1000
alpha=1.0
bias=0.0
wf=0.5          # perceptual blur: cognitive constant, NOT swept
world_length=2

# Total iterations: 8 (nobj) * 2 (speaker) * 6 (color_semvalue) * 5 (k) * 3 (sd_spread) = 1440
# Total contexts: 1440 * 1000 = 1,440,000  (Monte Carlo SE < 0.016)
n_total=1440
counter=0

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
for speaker in "${speaker_list[@]}"; do
    for nobj in $nobj_list; do
        for color_semvalue in $color_semvalue_list; do
            for k in $k_list; do
                for sd_spread in "${sd_spread_list[@]}"; do
                    ((counter++))
                    python 01-simulation-random-states.py \
                        --nobj              $nobj \
                        --sample_size       $sample_size \
                        --color_semvalue    $color_semvalue \
                        --wf                $wf \
                        --k                 $k \
                        --speaker           $speaker \
                        --alpha             $alpha \
                        --bias              $bias \
                        --world_length      $world_length \
                        --size_distribution normal \
                        --sd_spread         $sd_spread
                    echo -ne "Progress: $counter / $n_total \r"
                done
            done
        done
    done
done

echo -e "\nSimulation complete!"
