TRAIN_TEST_SPLIT=private_test_hard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
