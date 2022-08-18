#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=large_gpu
#SBATCH --account=rpixel
#SBATCH --qos=premium_memory
#SBATCH --mem=128GB
#SBATCH --time=5-00:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --open-mode=truncate
#SBATCH --gres=gpu:4

nvidia-smi
conda init
source ~/.bashrc
conda activate gtr

# Assert running on Turing
echo "HOSTNAME=$HOSTNAME"
if ! [[ $HOSTNAME =~ "turing" || $HOSTNAME =~ "vista" ]]
then
    echo "Error: Host not Turing"
    exit
fi
if ! [[ -v TMPDIR ]]
then
    echo "Error: TMPDIR not set"
    exit
fi

# Prepare datasets in $TMPDIR
./datasets/prep_data.sh BDD100K

# Login wandb
wandb login --relogin da75e98d29ae627bc5e000d68b033fda0155fc79

# Train
python train_net.py --num-gpus 8 --config-file configs/GTR_BDD_DR2101_C2.yaml --visualize

# Evaluate only
#python train_net.py --config-file configs/GTR_BDD_DR2102_C2.yaml --eval-only --visualize MODEL.WEIGHTS models/GTR_MOT_FPN.pth

# Copy output from $TMPDIR back to home directory
if [[ -v TMPDIR ]]; then
    echo Copy output from TMPDIR=$TMPDIR to home
    if ! [[ -d output/ ]]; then
        mkdir output/
    fi
    cp -r $TMPDIR/GTR/output/* output/
fi
echo Done
