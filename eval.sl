#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=ALL
#SBATCH --account=rpixel
#SBATCH --qos=premium_memory
#SBATCH --mem=32GB
#SBATCH --time=5:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --open-mode=truncate
#SBATCH --gres=gpu:1

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

# Evaluate only
python train_net.py --config-file configs/GTR_BDD_DR2101_C2.yaml --eval-only --visualize OUTPUT_DIR "./output/${SLURM_JOBID}/GTR_BDD/auto" MODEL.WEIGHTS models/best.pth

# Copy output from $TMPDIR back to home directory
#if [[ -v TMPDIR ]]; then
#    echo Copy output from TMPDIR=$TMPDIR to home
#    if ! [[ -d output/ ]]; then
#        mkdir output/
#        if ! [[ -d output/baseline/ ]]; then
#            mkdir output/baseline/
#        fi
#    fi
#    cp -r $TMPDIR/GTR/output/* output/
#fi
echo Done
