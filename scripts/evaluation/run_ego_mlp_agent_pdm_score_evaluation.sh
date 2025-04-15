TRAIN_TEST_SPLIT=navtest
#CHECKPOINT=/path/to/ego_status_mlp.ckpt
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_ego_mlp_agent/first/checkpoints/best/best_epoch.ckpt
export CUDA_VISIBLE_DEVICES=3

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent
