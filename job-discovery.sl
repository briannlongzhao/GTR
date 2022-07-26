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

# Set TMPDIR if on Discovery
if [[ $HOSTNAME =~ "discovery" || $HOSTNAME =~ "hpc" || $HOSTNAME =~ [a-z][0-9][0-9]-[0-9][0-9] ]]
then
    export TMPDIR=/scratch1/briannlz
else
    echo "Host not Discovery, exit"
    exit
fi

# Prepare datasets in $TMPDIR
./datasets/prep_data.sh

# Train
python train_net.py --num-gpus 2 --config-file configs/GTR_MOT_FPN.yaml

# Evaluate only
#python train_net.py --config-file configs/GTR_MOT_FPN.yaml --eval-only MODEL.WEIGHTS models/GTR_MOT_FPN.pth

# Copy output from $TMPDIR back to home directory
if [[ -v TMPDIR ]]
then
    echo Copy output from TMPDIR=$TMPDIR to home
    rm -r output
    cp -r $TMPDIR/GTR/output ./output
fi
echo Done
