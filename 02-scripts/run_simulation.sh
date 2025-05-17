#!/bin/bash

# Set the locale to ensure decimal points are interpreted correctly
export LC_NUMERIC="en_US.UTF-8"

# Define values for arguments
nobj_list=$(seq 2 4 30)  # Generates a sequence from 2 to 30 with a step of 4, 7 values in total
speaker_list=("incremental_speaker" "global_speaker")  # List of speaker models, two values in total
color_semvalue_list=$(seq 0.9 0.02 0.99)  # Generates a sequence from 0.90 to 0.99 with a step of 0.02, six values in total
k_list=$(seq 0.1 0.2 0.99)  # Generates a sequence from 0.05 to 0.99 with a step of 0.2, five values in total
wf_list=(0.5)  # Generates a sequence from 0.2 to 1.0 with a step of 0.2, five values in total
size_distribution_list=("normal")  # List of size distributions, three values in total
# Total iterations: 5 (nobj) * 2 (speaker) * 6 (color_semvalue) * 4 (k) * 5 (wf) * 3 (size_distribution) = 1800 * 10000 (sample_size) = 18,000,000

# Define other arguments with default values
sample_size=10000
form_semvalue=0.98
alpha=1.0
bias=0.0
wf=0.5
world_length=2
size_distribution="normal"
# 10000 * 7 nobj * 2 speaker * 6 color_semvalue * 5 k = 4200000
n_total=42000000
counter=0
# Loop over speaker_list, nobj_list, color_semvalue_list, k_list, and wf_list to run the Python script for each combination
for speaker in "${speaker_list[@]}"; do 
    for nobj in $nobj_list; do
        for color_semvalue in $color_semvalue_list; do
            for k in $k_list; do
                        ((counter++))
                        python 01-simulation-random-states.py --nobj $nobj \
                            --sample_size $sample_size \
                            --color_semvalue $color_semvalue \
                            --form_semvalue $form_semvalue \
                            --wf $wf \
                            --k $k \
                            --speaker $speaker \
                            --alpha $alpha \
                            --bias $bias \
                            --world_length $world_length \
                            --size_distribution $size_distribution
                        echo -ne "Progress: $counter / $n_total \r"
            done
        done
    done
done

echo -e "\nSimulation complete!"