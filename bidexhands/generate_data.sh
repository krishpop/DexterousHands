#!/bin/bash

experts=$(ls logs/**/*/*seed40*/model_6500.pt)

for expert in $experts; do
    echo $expert
done

datasets=$(ls logs/**/*.hdf5)

for dataset in $datasets; do
    echo $dataset
done

# Create an associative array to store unique parent directories
declare -A unique_parents

# Process datasets and store unique parent directories
for dataset in $datasets; do
    parent_dir=$(dirname "$dataset")
    unique_parents["$parent_dir"]=1
done

# Filter experts based on unique parent directories
filtered_experts=()
for expert in $experts; do
    expert_parent=$(dirname "$(dirname "$(dirname "$expert")")")
    if [[ ! ${unique_parents["$expert_parent"]} ]]; then
        filtered_experts+=("$expert")
    fi
done

# Print filtered experts
echo "Filtered experts:"
for expert in "${filtered_experts[@]}"; do
    echo "$expert"
done

# generate data with filtered experts
for expert in "${filtered_experts[@]}"; do
    echo "Generating data with $expert"
    # get task name from expert path
    task=$(basename "$(dirname "$(dirname "$(dirname "$expert")")")")
    python train.py --algo ppo --datatype ppo_collect --model_dir $expert --task $task --num_envs 64 --num_rollouts 500
    python train.py --algo ppo --datatype ppo_collect --model_dir $expert --task $task --num_envs 64 --num_rollouts 500 --success_only
done