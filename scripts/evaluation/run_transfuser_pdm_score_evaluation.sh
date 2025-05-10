TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_transfuser_agent/first/checkpoints/best/best_epoch.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
export CUDA_VISIBLE_DEVICES=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
metric_cache_path=$CACHE_PATH \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent


# If you want to run one stage only simulation, plesae uncomment and keep only the following lines:
# TRAIN_TEST_SPLIT=navtest
# CHECKPOINT=/path/to/transfuser.ckpt

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# agent=transfuser_agent \
# worker=single_machine_thread_pool \
# agent.checkpoint_path=$CHECKPOINT \
# experiment_name=transfuser_agent \
# traffic_agents_policy=non_reactive \
