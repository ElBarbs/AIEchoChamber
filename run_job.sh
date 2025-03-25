#!/bin/bash
#SBATCH --account=lbarbier
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20000M
#SBATCH --time=00:05:00

module load python/3.12.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
cp /home/lbarbier/projects/def-vigliens/lbarbier/AIEchoChamber $SLURM_TMPDIR/AIEchoChamber
cd $SLURM_TMPDIR/AIEchoChamber
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install compel-2.0.3-py3-none-any.whl
python main.py --input "flowers.jpg"
cp $SLURM_TMPDIR/AIEchoChamber/output /home/lbarbier/projects/def-vigliens/lbarbier/AIEchoChamber/output