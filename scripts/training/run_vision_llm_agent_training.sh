TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache
export CUDA_VISIBLE_DEVICES=1

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=vision_llm_agent \
experiment_name=training_vision_llm_agent \
train_test_split=$TRAIN_TEST_SPLIT \
> vision_llm_training_output0.log 2>&1 &