GPU=1
TRAIN_DIR=~/runtime/instantngp
SCREENSHOT_DIR=~/runtime/instantngp
EXP=test_sparse_views
TARGETDIR=$TRAIN_DIR/$EXP

mkdir -p $TARGETDIR
echo GPU: $GPU
echo TRAIN_DIR: $TRAIN_DIR
echo SCREENSHOT_DIR: $SCREENSHOT_DIR
echo EXPERIMENT: $EXP

if [ -d $TARGETDIR ]; then
nohup sh launch_view_quantity.sh $GPU $TRAIN_DIR $SCREENSHOT_DIR $EXP > $TARGETDIR/nohup.out &
fi