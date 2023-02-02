#!/bin/bash
PHASE=train
if [ "$5" = "test" ]; then
  PHASE=$5
fi

echo launch experiment of Instant NGP
echo GPU $1
echo snapshot dir: $2
echo screenshot dir: $3
echo experiment name: $4
echo running phase: $PHASE

export CUDA_VISIBLE_DEVICES=$1


# TRAINING
for SCENE in chair drums ficus hotdog lego materials mic ship
do
  echo $SCENE
  for v in 1 3 5 10 20 30 40 50 60 70 80 90 100
  do
    CKPT_DIR=$2/$4/$SCENE/$v
    echo $CKPT_DIR
    mkdir -p $CKPT_DIR

    if [ "$PHASE" = "train" ]; then
      RUNS=10
      if [ $v -gt 50 ]; then
        RUNS=5
        if [ $v -gt 90 ]; then
          RUNS=1
        fi
      fi
      echo $v $RUNS

      RUN=1
      while [ "$RUN" -le $RUNS ]; do
        STEPS=$((v * 1000))
        #SNAPSHOT=$CKPT_DIR/$SCENE-$STEPS-views$v.msgpack
        echo The $RUN run scene:$SCENE
        #python ./scripts/run_views_test.py --scene ~/datasets/nerf_synthetic/$SCENE/transforms_train.json --mode nerf --n_steps $STEPS --train_dir $CKPT_DIR --nerf_compatibility --eval_view_quantity True --views $v
        RUN=$(( RUN + 1 ))
      done
    fi

    if [ "$PHASE" = "test" ]; then
      for c in $(ls $CKPT_DIR); do
        RUNDIR=$CKPT_DIR/$c
        TRAIN_TRANSFORM=$RUNDIR/$(ls $RUNDIR | grep .json)
        TEST_TRANSFORM=~/datasets/nerf_synthetic/$SCENE/transforms_test.json
        SNAPSHOT=$RUNDIR/$(ls $RUNDIR | grep .msgpack)
        
        if [ -f "$TRAIN_TRANSFORM" ]; then
          # test on train data
          SCREENSHOT_DIR=$RUNDIR/screenshot_on_train
          mkdir -p $SCREENSHOT_DIR 

          echo "python scripts/run_views_test.py --mode nerf --load_snapshot $SNAPSHOT --screenshot_dir $SCREENSHOT_DIR --test_transforms $TRAIN_TRANSFORM --nerf_compatibility"

          # test on test data
          SCREENSHOT_DIR=$RUNDIR/screenshot_on_test
          mkdir -p $SCREENSHOT_DIR

          #python scripts/run_views_test.py --mode nerf --load_snapshot $SNAPSHOT --screenshot_dir $SCREENSHOT_DIR --test_transforms $TEST_TRANSFORM --nerf_compatibility
        fi


      done
    fi

  #   # TESTING
  #   SCREENSHOT_DIR=$3/$SCENE/$v
  #   mkdir -p $SCREENSHOT_DIR
  #   echo run $SCENE testing
  #   python scripts/run.py --mode nerf --scene ~/datasets/nerf_synthetic/$SCENE/ --load_snapshot $SNAPSHOT --screenshot_dir $SCREENSHOT_DIR --test_transforms ~/datasets/nerf_synthetic/$SCENE/transforms_test.json --nerf_compatibility
  done
done
