#!/bin/bash

#SBATCH --job run_s
#SBATCH --partition any_cpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1

# #SBATCH --ntasks-per-node=4

# #SBATCH --exclude g001, g011, g019

#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --mail-type=ALL

sbatch run_model1.slurm
sbatch run_model2.slurm
sbatch run_model3.slurm
sbatch run_model4.slurm
sbatch run_model5.slurm
sbatch run_model6.slurm
sbatch run_model7.slurm
