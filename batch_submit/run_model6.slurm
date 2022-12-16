#!/bin/bash

#SBATCH --job run_mdl6
#SBATCH --partition dept_gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1

# #SBATCH --ntasks-per-node=4

# #SBATCH --exclude g001, g011, g019

#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --mail-type=ALL

source ~/.bashrc

job_dir="${USER}_${SLURM_JOB_ID}.dcb.private.net"

mkdir /scr/$job_dir

rsync -a ${SLURM_SUBMIT_DIR} /scr/${job_dir}

trap "echo 'copying files'; rsync -avz * ${SLURM_SUBMIT_DIR}" EXIT

module load anaconda
eval "$(conda shell.bash hook)"

source activate dehaze
python3 ./run_model.py --model LightDehazeNet_Attn_Conv --output_dir lightdehazenet_attn_conv --epochs 40 --learning_rate 0.02