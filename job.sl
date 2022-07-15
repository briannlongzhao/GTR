#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --gres=gpu:a100:2

ulimit -s unlimited
nvidia-smi
conda init
source ~/.bashrc
conda activate gtr

# Set $TMPDIR if on Discovery
echo HOSTNAME=$HOSTNAME
if [[ $HOSTNAME =~ "discovery" ]] || [[ $HOSTNAME == [a-z][0-9][0-9]-[0-9][0-9] ]]
then
	export TMPDIR=/scratch1/briannlz
fi

# Prepare datasets in $TMPDIR
echo TMPDIR=$TMPDIR
./datasets/prep_data.sh

# Train
python train_net.py --num-gpus 2 --config-file configs/GTR_MOT_FPN.yaml

# Evaluate
#python train_net.py --config-file configs/GTR_MOT_FPN.yaml --eval-only MODEL.WEIGHTS models/GTR_MOT_FPN.pth