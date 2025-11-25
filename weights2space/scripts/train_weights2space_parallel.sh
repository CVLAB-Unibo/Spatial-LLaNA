master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=11111
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_NET=IB


cd weights2space/

#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
export WANDB_MODE=offline

start_time=$(date +%s)
mpirun -np $SLURM_NTASKS -x MASTER_ADDR -x MASTER_PORT -x PATH -bind-to none -map-by slot python ../weights2space/train/train_weights2space_parallel.py --launcher "mpi"
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "MPI job execution time: $duration seconds"