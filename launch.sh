#!/bin/bash

echo launch experiment of Instant NGP
echo GPU $1
echo snapshot dir: $2
echo screenshot dir: $3

SCENE=lego
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=chair
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=drums
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=ficus
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=hotdog
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=materials
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=mic
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility

SCENE=ship
CKPT_DIR=$2/$SCENE
mkdir -p $CKPT_DIR
SCREENSHOT_DIR=$3/$SCENE
mkdir -p $SCREENSHOT_DIR

# TRAINING
echo run $SCENE training
python ./scripts/run.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps 20000 --save_snapshot $CKPT_DIR/$SCENE-20000.msgpack --nerf_compatibility

# TESTING
echo run $SCENE testing
python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $CKPT_DIR/$SCENE-20000.msgpack --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility