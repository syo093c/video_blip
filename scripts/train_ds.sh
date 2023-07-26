#!/bin/bash
#$ -l rt_F=2
#$ -l h_rt=2:00:00
#$ -j y
#$ -N finetune_blip
#$ -o logs/
#$ -cwd

export JOBMODE=1

if [ "${JOBMODE}" == 0 ]; then
    echo "this is interactive mode"
    export WANDB_NAME="BLIP-interactive-"$(date "+%Y-%m-%d-%H:%M")
else
    echo "this is batch job mode"
    source /etc/profile.d/modules.sh
    module load python/3.11 cuda/11.7 cudnn/8.6 hpcx-mt/2.12
    source ../abci_llm/.venv/bin/activate
    export WANDB_NAME="BLIP-batch-"$(date "+%Y-%m-%d-%H:%M")
fi

cat ${SGE_JOB_HOSTLIST} | awk '{print $0, "slots=4"}' > hostfile

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CONFIG="config/config_finetune.yaml"
export MODEL="Salesforce/blip2-opt-2.7b"
#export MODEL="Salesforce/blip2-opt-6.7b"

export MASTER_ADDR=$(cat ${SGE_JOB_HOSTLIST} | head -n 1)
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="abci-llm-hackathon"
export PYTHON_FILE="src/finetune_blip_video.py"
export DATA_DIR="/data/bddx/v_and_l_data_224x224_video"

deepspeed --hostfile hostfile --launcher OpenMPI --no_ssh_check --master_addr=$MASTER_ADDR $PYTHON_FILE --model_name $MODEL --config_file $CONFIG --data_dir $DATA_DIR
