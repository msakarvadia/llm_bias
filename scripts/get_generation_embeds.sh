#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --qos=regular
#SBATCH --account=m1266
#SBATCH --constraint=gpu
#SBATCH --gpus=4


cd /pscratch/sd/m/mansisak/llm_bias
module load conda
conda activate env/

cd /pscratch/sd/m/mansisak/llm_bias/scratch_work
python save_model_logits.py --config_path ../configs/reddit_synthetic_llama2_7b.yaml
