RUN=true
TIME=1
TSUBAME_GROUP=tga-k
WORKSPACE=/home/3/um02143/workspace/GOAL
CONDA_ENV=goal

# Args
tag=SOREC_val
dataset=SOREC_val.json
ckpt=GOAL_ViT_base16_DCI.pth 

batch_size=16
epochs=10
dataset=atasets/DCI_train_del_org.json
output_dir=exps/finetune_out_DCI/long_batch16_base16



echo '#!/bin/bash
#$ -l gpu_1=1
#$ -l h_rt='${TIME}':00:00
#$ -j y
#$ -o '${WORKSPACE}'/logs/'${tag}'.o
#$ -e /'${WORKSPACE}'/logs/'${tag}'.e


# Environment Setup
. /etc/profile.d/modules.sh
source ~/.bashrc
cd '${WORKSPACE}'
# module load cuda/12.1.0 cudnn/9.0.0 intel/2024.0.2 nccl/2.20.5 # To avoid torch_squim errors
conda activate '${CONDA_ENV}'

python goal_loss_finetuning.py --batch_size '${batch_size}' --epochs '${epochs}'
' > logs/${tag}.sh


chmod u+x logs/${tag}.sh
if $RUN; then
  qsub -g ${TSUBAME_GROUP} logs/${tag}.sh
fi
