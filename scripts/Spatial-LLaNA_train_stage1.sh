#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -A IscrC_AMALLM
#SBATCH --nodes=4
#SBATCH --cpus-per-task=32
#SBATCH --error=llana_13b_shapenerf_objanerf_stage1_AUGMENTED_nf2seq_spatial516.err
#SBATCH --output=llana_13b_shapenerf_objanerf_stage1_AUGMENTED_nf2seq_spatial516.out
#SBATCH -J debug

# Print SLURM environment variables for debugging
echo "NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"

# Set MASTER_ADDR and MASTER_PORT
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# load coding environment
cd /leonardo_scratch/fast/IscrB_U3DCAP
source .cuda_12_venv/bin/activate
module load cuda/12.1
module load openmpi/4.1.6--gcc--12.2.0
cd dev/Spatial-LLaNA

# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')
datetime=$(date '+%d-%m-%Y_%H:%M')

#model_name_or_path=/leonardo_scratch/fast/IscrB_U3DCAP/.cache/huggingface/hub/models--andreamaduzzi--LLaNA-13B_init/snapshots/688d456b762d5d313ce035a356a44fd351f23d1d
model_name_or_path=/leonardo_work/IscrB_U3DCAP/results/LLaNA-13B_init
#root=data/objanerf_text
root=data/shapenerf_objanerf_text
data_folder=vecs_nf2seq_spatial516_AUGMENTED
anno_folder=texts
output_dir=outputs/LLaNA_13B_train_stage1_shapenerf_objanerf_AUGMENTED_nf2seq_spatial516/${filename}_${datetime}
export WANDB_MODE=offline


while true; do
    PORT=$((10000 + RANDOM % 20000))
    ss -tuln | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT

export WANDB_MODE=offline
export MASTER_ADDR=$head_node
export WORLD_SIZE=$((SLURM_NNODES * 4))  # 4 GPUs per node
export NCCL_NET=IB  # TODO: understand if it helps or not








### CODE TO RELEASE
master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')
datetime=$(date '+%d-%m-%Y_%H:%M')

model_name_or_path=andreamaduzzi/LLaNA-13B_init
root=data/spatial_llana_dataset
data_folder=vecs
anno_folder=texts
output_dir=outputs/Spatial-LLaNA-13B_train_stage1_${datetime}

torchrun --nnodes=2 --nproc_per_node=4 --master_port=$master_port spatial_llana/train/train_spatial_llana.py \
    --model_name_or_path $model_name_or_path \
    --root $root \
    --data_folder $data_folder \
    --anno_folder $anno_folder \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 4096 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm True \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --report_to wandb \
    --run_name Spatial-LLaNA-13B_train_stage1_${datetime} \