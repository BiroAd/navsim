TRAIN_TEST_SPLIT=navtrain
export CUDA_VISIBLE_DEVICES=3

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
experiment_name=training_ego_mlp_agent \
trainer.params.max_epochs=5 \
train_test_split=$TRAIN_TEST_SPLIT \
> mlp_training_output.log 2>&1 &
