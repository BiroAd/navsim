TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_vision_llm_agent/2025.05.09.11.01.19/checkpoints/epochs/epoch7.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
export CUDA_VISIBLE_DEVICES=0

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=vision_llm_agent \
agent.checkpoint_path=$CHECKPOINT \
metric_cache_path=$CACHE_PATH \
worker=ray_distributed \
metric_cache_path=$CACHE_PATH \
experiment_name=vision_llm_agent_eval \
> vision_llm_eval_output1.log 2>&1 &