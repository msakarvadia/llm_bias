#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:00:00
#SBATCH --qos=regular
#SBATCH --account=m1266
#SBATCH --constraint=gpu
#SBATCH --gpus=4


cd /pscratch/sd/m/mansisak/llm_bias
module load conda
conda activate env/

cd /pscratch/sd/m/mansisak/llm_bias/src/train/
python save_generation_embeds.py --config_path ../../configs/reddit_synthetic_llama2_7b.yaml
