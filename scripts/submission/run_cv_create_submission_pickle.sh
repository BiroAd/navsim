TEAM_NAME="ContiBME"
AUTHORS="test"
EMAIL="biro.adam@edu.bme.hu" 
INSTITUTION="Budapest_University_of_Technology_and_Economics"
COUNTRY="Hungary"
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_vision_llm_agent/best_currently/checkpoints/best/best.ckpt

TRAIN_TEST_SPLIT=warmup_two_stage
export CUDA_VISIBLE_DEVICES=0

nohup python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=vision_llm_agent \
agent.checkpoint_path=$CHECKPOINT \
worker=ray_distributed \
experiment_name=submission_cv_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
> vision_llm_submission.log 2>&1 &
