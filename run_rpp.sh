while getopts "p:m:d:" opt; do
  case $opt in
    p) path=$OPTARG ;;
    m) model=$OPTARG ;;
    d) dataset=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_MODEL="${path}"
export PROJECT_NAME='Graph-R1'
export EXPERIMENT_NAME="${model}_${dataset}_rpp"
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
set -x

ray stop
ray start --head

python3 -m verl.trainer.main_ppo \
    data.train_files=datasets/"${dataset}"/processed/train.parquet \
    data.val_files=datasets/"${dataset}"/processed/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_start_length=4096 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=3 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    tool.env='search' $@ 