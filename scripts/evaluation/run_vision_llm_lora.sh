TRAIN_TEST_SPLIT=warmup_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
export CUDA_VISIBLE_DEVICES=0

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=vision_llm_lora_agent \
agent.checkpoint_path=\'/root/workdir/biroa/navsim/exp/training_vision_llm_lora_agent/first_phase/checkpoints/best/best-epoch=09-val_loss=0.00.ckpt\' \
metric_cache_path=$CACHE_PATH \
worker=sequential \
metric_cache_path=$CACHE_PATH \
experiment_name=vision_llm_lora_agent_eval \
> vision_llm_lora_eval_output7.log 2>&1 &