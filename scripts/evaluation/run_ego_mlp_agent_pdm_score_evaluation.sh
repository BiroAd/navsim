TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=/root/workdir/biroa/navsim/exp/training_ego_mlp_agent/first/checkpoints/best/best_epoch.ckpt
CACHE_PATH=YOUR_PATH_TO_METRIC_CACHE
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles
export CUDA_VISIBLE_DEVICES=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH


# If you want to run one stage only simulation, plesae uncomment and keep only the following lines:

# TRAIN_TEST_SPLIT=navtest
# CHECKPOINT=/path/to/ego_status_mlp.ckpt

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# agent=ego_status_mlp_agent \
# agent.checkpoint_path=$CHECKPOINT \
# experiment_name=ego_mlp_agent \
# traffic_agents_policy=non_reactive \
