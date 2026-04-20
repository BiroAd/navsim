TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_transfuser_agent/first/checkpoints/best/best_epoch.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles
export CUDA_VISIBLE_DEVICES=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
metric_cache_path=$CACHE_PATH \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH
