#!/bin/bash
#SBATCH --account=def-vigliens
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20000M
#SBATCH --time=00:05:00

module load python/3.12.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
mkdir $SLURM_TMPDIR/AIEchoChamber
cp -r /project/def-vigliens/lbarbier/AIEchoChamber/files $SLURM_TMPDIR/AIEchoChamber
cp -r /project/def-vigliens/lbarbier/AIEchoChamber/huggingface $SLURM_TMPDIR/AIEchoChamber
wait
cd $SLURM_TMPDIR/AIEchoChamber/files
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install compel-2.0.3-py3-none-any.whl
python main.py --input "flowers.jpg"
cp -r $SLURM_TMPDIR/AIEchoChamber/files/output /project/def-vigliens/lbarbier/AIEchoChamber/output