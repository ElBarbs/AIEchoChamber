#!/bin/bash
#SBATCH --account=def-vigliens
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=40:00:00

module load python-build-bundle/2025a
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
mkdir $SLURM_TMPDIR/AIEchoChamber
cp -r /project/def-vigliens/lbarbier/AIEchoChamber/files $SLURM_TMPDIR/AIEchoChamber
cp -r /project/def-vigliens/lbarbier/AIEchoChamber/huggingface $SLURM_TMPDIR/AIEchoChamber
wait
cd $SLURM_TMPDIR/AIEchoChamber/files
pip install --no-index --upgrade pip
pip install transformers==4.50.3 torchvision==0.21.0 pillow==11.1.0
pip install --no-index -r requirements.txt
pip install --no-index compel-2.0.3-py3-none-any.whl
python main.py --input "friends.jpg" --iterations 2000
wait
cp -r $SLURM_TMPDIR/AIEchoChamber/files/output /project/def-vigliens/lbarbier/AIEchoChamber/output