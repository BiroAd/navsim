TRAIN_TEST_SPLIT=navtest
CHECKPOINT="/root/workdir/biroa/navsim/exp/training_transfuser_agent/first/checkpoints/best/best_epoch.ckpt"

export CUDA_VISIBLE_DEVICES=3

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
#worker=single_machine_thread_pool \
worker=sequential \
experiment_name=transfuser_agent_eval