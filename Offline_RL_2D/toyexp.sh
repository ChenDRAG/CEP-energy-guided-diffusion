#!/bin/bash
echo "STARTED"
counted=0
selected_gpu=0

TXTLOGDIR="./toy/"
if [ ! -d $TXTLOGDIR ]; then
  mkdir $TXTLOGDIR
fi

# alpha represents the beta hyperparameter in the paper

main() {
    TASK="rings"
    seed=0
    setting="mse_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=9 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "mse" > $txtname 2>&1 &

    TASK="checkerboard"
    seed=0
    setting="mse_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=1 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "mse" > $txtname 2>&1 &

    TASK="2spirals"
    seed=0
    setting="mse_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=2 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "mse" > $txtname 2>&1 &



    TASK="rings"
    seed=0
    setting="CEP_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=3 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "CEP" > $txtname 2>&1 &

    TASK="checkerboard"
    seed=0
    setting="CEP_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=4 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "CEP" > $txtname 2>&1 &

    TASK="2spirals"
    seed=0
    setting="CEP_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=5 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "CEP" > $txtname 2>&1 &



    TASK="rings"
    seed=0
    setting="emse_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=6 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "emse" > $txtname 2>&1 &

    TASK="checkerboard"
    seed=0
    setting="emse_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=7 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "emse" > $txtname 2>&1 &

    TASK="2spirals"
    seed=0
    setting="emse_alpha20"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp$TASK${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=8 python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 20 --s 1.0 --method "emse" > $txtname 2>&1 &

}


main "$@"; exit