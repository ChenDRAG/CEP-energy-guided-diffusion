## Toy2D experiments

For reproducing toy 2D experiments in the paper, run
```.bash
$ TASK="rings"; setting="CEP_alpha3"; seed=0; python3 -u bandit_toy.py --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 3 --method "CEP"
```

Checkpoints will be stored in the `./models` folder. Visualization scripts are provided in `draw_toy.ipynb`. You can also download pretrained checkpoints from this url link (TBD).

## D4RL experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed.

### Running
To pretrain the behavior model, run

```.bash
$ TASK="walker2d-medium-expert-v2"; seed=0; setting="reproduce"; python3 -u train_behavior.py --expid $TASK${seed}${setting} --env $TASK --seed ${seed}
```

The pretrained behavior model will be stored in the `./models_rl/`. Once we have the pretrained checkpoint at `/path/to/pretrained/ckpt.pth` (TBD, add links for pretrained bahvior model), we can train the critic model:

```.bash
$ TASK="walker2d-medium-expert-v2"; seed=0; setting="reproduce"; python3 -u train_critic.py --actor_load_path /path/to/pretrained/ckpt.pth --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 3 --q_alpha 1 --method "CEP"
```