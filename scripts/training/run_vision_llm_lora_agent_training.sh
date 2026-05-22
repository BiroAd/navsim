TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=vision_llm_lora_agent \
agent.checkpoint_path=\'/root/workdir/biroa/navsim/exp/training_vision_llm_lora_agent/second_phase_16/checkpoints/best/best-epoch=02-val_loss=0.00.ckpt\' \
experiment_name=training_vision_llm_lora_agent \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH \
> vision_lora_training_output11.log 2>&1 &
