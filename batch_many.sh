#!/bin/bash
#
# Example with an array of experiments
#  The --array line says that we will execute 4 experiments (numbered 0,1,2,3).
#   You can specify ranges or comma-separated lists on this line
#  For each experiment, the SLURM_ARRAY_TASK_ID will be set to the experiment number
#   In this case, this ID is used to set the name of the stdout/stderr file names
#   and is passed as an argument to the python program
#
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --exclusive
#SBATCH --partition=gpu
# memory in MB
#SBATCH --mem=15000
# The %j is translated into the job number
#SBATCH --output=results/music_%j_stdout.txt
#SBATCH --error=results/music_%j_stderr.txt
#SBATCH --time=06:00:00
#SBATCH --job-name=music
#SBATCH --mail-user=Jacob.e.Sturges-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/MusicGenerator
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

module load cuDNN/8.2.1.32-CUDA-11.3.1

source /home/cs504319/miniconda3/bin/activate

# Change this line to start an instance of your experiment
python train1.py --exp_index 0



