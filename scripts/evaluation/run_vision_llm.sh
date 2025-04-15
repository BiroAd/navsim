TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_vision_llm_agent/only_status/checkpoints/best/best.ckpt

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=vision_llm_agent \
agent.checkpoint_path=$CHECKPOINT \
worker=single_machine_thread_pool \
experiment_name=vision_llm_agent_eval \
> vision_llm_eval_output2.log 2>&1 &