#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=large_gpu
#SBATCH --account=rpixel
#SBATCH --qos=premium_memory
#SBATCH --mem=128GB
#SBATCH --time=10:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --gres=gpu:4

nvidia-smi
conda init
source ~/.bashrc
conda activate gtr

# Assert running on Turing
if ! [[ $HOSTNAME =~ "turing" || $HOSTNAME =~ "vista" ]]
then
    echo "Host not Turing, exit"
    exit
fi

# Prepare datasets in $TMPDIR
echo TMPDIR=$TMPDIR
./datasets/prep_data.sh

# Train
python train_net.py --num-gpus 4 --config-file configs/GTR_MOTFull_FPN.yaml

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
