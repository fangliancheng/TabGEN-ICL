#!/bin/bash

datanames=(adult magic shoppers default california)

model_names=(gpt-4o-mini) 

num_synthetic=3000

output_formats=(json)

seeds=(0 1 2)

for seed in "${seeds[@]}"
do
  for model_name in "${model_names[@]}"
  do
    for dataname in "${datanames[@]}"
    do
      for output_format in "${output_formats[@]}"   
      do
        time=$(date +%Y%m%d_%H%M%S)
        exp_name=new_${output_format}_${seed}_${time}  
        dir=logs/${dataname}_exp_${exp_name}/
        if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Directory created: $dir"
        fi
        model_filename="${model_name}_${num_synthetic}_temp1.0_${exp_name}"
        echo "Running script for dataname: $dataname with LLM: $model_name"
        nohup python -u run_llm_generator.py  --output_format $output_format --num_synthetic $num_synthetic --model $model_name --dataname $dataname --exp_name $exp_name > logs/${dataname}_exp_${exp_name}/generation_log_${model_filename}.log 2>&1 
        echo "Evaluating mle..."
        nohup python -u eval/eval_mle.py --dataname $dataname --model $model_filename > logs/${dataname}_exp_${exp_name}/eval_mle_${dataname}_${model_filename}.log 2>&1 
        echo "Evaluating dcr..."
        nohup python -u eval/eval_dcr.py --dataname $dataname --model $model_filename > logs/${dataname}_exp_${exp_name}/eval_dcr_${dataname}_${model_filename}.log 2>&1 
        echo "Evaluating density..."
        nohup python -u eval/eval_density.py --dataname $dataname --model $model_filename > logs/${dataname}_exp_${exp_name}/eval_density_${dataname}_${model_filename}.log 2>&1 
        echo "Evaluating detection..."
        nohup python -u eval/eval_detection.py --dataname $dataname --model $model_filename > logs/${dataname}_exp_${exp_name}/eval_detection_${dataname}_${model_filename}.log 2>&1 
        echo "Evaluating Jensen-Shannon Divergence..."
        nohup python -u eval/eval_jsd.py --dataname $dataname --model $model_filename > logs/${dataname}_exp_${exp_name}/eval_jsd_${dataname}_${model_filename}.log 2>&1 
        echo "Evaluating Wasserstein Distance..."
        nohup python -u eval/eval_wd.py --dataname $dataname --model $model_filename > logs/${dataname}_exp_${exp_name}/eval_wasserstein_${dataname}_${model_filename}.log 2>&1 
      done
    done 
  done
done

