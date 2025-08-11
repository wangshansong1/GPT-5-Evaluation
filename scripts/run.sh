#!/bin/bash
set -e

models=${1:-"gpt-4o-2024-11-20"}
datasets=${2:-"medqa"}
tasks=${3:-"text"}
output_dir=${4:-"dev"}

method=${5:-"zero_shot"}
prompting_type=${6:-"cot"}

temperature=${7:-0}

IFS=","

for model in $models; do
    for dataset in $datasets; do
        for task in $tasks; do
            date +"%Y-%m-%d %H:%M:%S"
            echo "Model: ${model}"
            echo "Dataset: ${dataset}"
            echo "Task: ${task}"
            echo "Output: ${output_dir}"
            log_dir="outputs/${output_dir}/${model}/${dataset}/${method}/${prompting_type}/logs"
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
            fi
            log_file="${log_dir}/run-${model}-${dataset}-${task}.log"

            cp "${BASH_SOURCE[0]}" "${log_dir}/run.sh"
            cp main.py "${log_dir}/main.py"
            cp utils.py "${log_dir}/utils.py"
            cp model/api_agent.py "${log_dir}/api_agent.py"
            cp config/prompt_templates.py "${log_dir}/prompt_templates.py"
            python main.py --model "${model}" --dataset "${dataset}" --task "${task}" --output-dir "${output_dir}" --method "${method}" --prompting-type "${prompting_type}" --temperature "${temperature}"
        done
    done
done

# bash scripts/run.sh