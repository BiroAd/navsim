TEAM_NAME="MUST_SET"
AUTHORS="MUST_SET"
EMAIL="MUST_SET"
INSTITUTION="MUST_SET"
COUNTRY="MUST_SET"

TRAIN_TEST_SPLIT=warmup_two_stage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=submission_cv_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
