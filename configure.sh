#!/bin/bash

# Output file
output_file="config/runner_definition.mk"
exp_run="config/exp_run"
mkdir -p config

read -p "Pleaser enter your preferred python command (python/python3/python3.11) " python_cmd
echo -e "DEFAULTPYTHON=$python_cmd" > "$output_file"

if command -v srun &> /dev/null; then
    # Prompt the user for any extra flags
    read -p "Please enter any extra flags for srun: " extra_flags

    # If srun is available, set the RUN variable accordingly
    echo "RUN=srun $extra_flags --time 20:00 --mem 32G --cpus-per-task=32 --ntasks=1" >> "$output_file"
    echo "srun $extra_flags --time 20:00 --mem 32G --cpus-per-task=32 --ntasks=1" > "$exp_run"

    # Prompt the user for their email
    read -p "Please enter your email: " email


    # Write the extra target into output_file
    echo "run_all:" >> "$output_file"
    echo -e "\t-bash run.sh" >> "$output_file"
    echo -e "\tsrun  --job-name \"QMSR jobs complete\" --mail-type END --mail-user $email $extra_flags ls -a"  >> "$output_file"
else
    # If srun is not available, set the RUN variable to an empty string
    echo 'RUN=' >> "$output_file"
    echo "" > "$exp_run"
fi
