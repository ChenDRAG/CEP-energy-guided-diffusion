import os
import gym
import d4rl
import scipy
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model_dql import ScoreNet, MlpScoreNet
from utils import get_args, pallaral_eval_policy, simple_eval_policy, pallaral_eval_policy_resample
# from dataset.dataset import Diffusion_buffer
from dataset.dataset import DQL_buffer as Diffusion_buffer
LOAD_FAKE=False

def train_critic(args, score_model, data_loader, start_epoch=0):
    total_steps= 1000000
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    n_epochs = 100
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 4 if "antmaze" not in args.env else 4
    save_interval = 20 if "antmaze" not in args.env else 20

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}
            if (epoch < 50):
                loss = score_model.q[0].update_q0_mine(data)
            loss2 = score_model.q[0].update_qt(data)
            avg_loss += loss2
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            study_scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] if "antmaze" in args.env else [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

            for guidance_scale in study_scales:
                score_model.q[0].guidance_scale = guidance_scale

                envs = args.eval_func(score_model.select_actions)
                mean = np.mean([envs[i].dbag_return for i in range(args.seed_per_evaluation)])
                std = np.std([envs[i].dbag_return for i in range(args.seed_per_evaluation)])

                args.writer.add_scalar("eval/rew{}".format(guidance_scale), mean, global_step=epoch)
                args.writer.add_scalar("eval/std{}".format(guidance_scale), std, global_step=epoch)
            score_model.q[0].guidance_scale = 1.0

            bm, bs = simple_eval_policy(score_model.q[0].gaussian_policy.forward, args.env, args.seed, eval_episodes=20)
            args.writer.add_scalar("eval_baseline/rew", bm, global_step=epoch)
            args.writer.add_scalar("eval_baseline/std", bs, global_step=epoch)
            
        if (epoch % save_interval == (save_interval - 1)) or epoch==0:
            torch.save(score_model.q[0].state_dict(), os.path.join("./models", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("critic/loss", avg_loss / num_items, global_step=epoch)
        data_p = [0, 10, 25, 50, 75, 90, 100]
        args.writer.add_scalars("target/mean", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].all_mean, data_p))}, global_step=epoch)
        args.writer.add_scalars("target/std", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].all_std, data_p))}, global_step=epoch)
        args.writer.add_scalars("target/debug", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].debug_used, data_p))}, global_step=epoch)

def critic(args):
    M=16
    if "antmaze" in args.env:
        M=32

    for dir in ["./models", "./Benchmark_logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", str(args.expid))):
        os.makedirs(os.path.join("./models", str(args.expid)))

    writer = SummaryWriter("./Benchmark_logs2/" + str(args.expid))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, track_obs=False, select_per_state=args.select_per_state, diffusion_steps=args.diffusion_steps)
    args.eval_func_resample = functools.partial(pallaral_eval_policy_resample, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, track_obs=False, select_per_state=args.select_per_state, diffusion_steps=args.diffusion_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.actor_type == "large":
        score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    elif args.actor_type == "small":
        score_model= MlpScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)
    # if args.actor_loadpath is not specifided, should be determined by expid and args.actor_load_epoch
    if args.actor_load_setting is None:
        args.actor_loadpath = os.path.join("./models", str(args.expid), "ckpt{}.pth".format(args.actor_load_epoch))
    else:
        args.actor_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "ckpt{}.pth".format(args.actor_load_epoch))
    print("loading actor...")
    ckpt = torch.load(args.actor_loadpath, map_location=args.device)
    score_model.load_state_dict(ckpt)
    
    dataset = Diffusion_buffer(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    score_model.q[0].guidance_scale = 0.0
    if os.path.exists(args.actor_loadpath+ "actions{}_raw.npy".format(args.diffusion_steps)) and LOAD_FAKE:
        dataset.fake_actions = torch.Tensor(np.load(args.actor_loadpath+ "actions{}_raw.npy".format(args.diffusion_steps)).astype(np.float32)).to("cuda")
    else:
        allstates = dataset.states[:].cpu().numpy()
        all_resuls = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 256 + 1)):
            all_resuls.append(score_model.sample(states, sample_per_state=M, diffusion_steps=args.diffusion_steps))
        returns = np.concatenate([res[0] for res in all_resuls]) # <bz, M, 1>    
        actions = np.concatenate([res[1] for res in all_resuls])
        dataset.fake_actions = torch.Tensor(actions.astype(np.float32)).to("cuda")
        if LOAD_FAKE:
            np.save(args.actor_loadpath+ "actions{}_raw.npy".format(args.diffusion_steps), actions)

    # fake next action
    if os.path.exists(args.actor_loadpath+ "next_actions{}_raw.npy".format(args.diffusion_steps)) and LOAD_FAKE:
        dataset.fake_next_actions = torch.Tensor(np.load(args.actor_loadpath+ "next_actions{}_raw.npy".format(args.diffusion_steps)).astype(np.float32)).to("cuda")
    else:
        allstates = dataset.next_states[:].cpu().numpy()
        all_resuls = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 256 + 1)):
            all_resuls.append(score_model.sample(states, sample_per_state=M, diffusion_steps=args.diffusion_steps))
        returns = np.concatenate([res[0] for res in all_resuls]) # <bz, M, 1>    
        actions = np.concatenate([res[1] for res in all_resuls])
        dataset.fake_next_actions = torch.Tensor(actions.astype(np.float32)).to("cuda")
        if LOAD_FAKE:
            np.save(args.actor_loadpath+ "next_actions{}_raw.npy".format(args.diffusion_steps), actions)

    print("training critic")
    train_critic(args, score_model, data_loader, start_epoch=0)
    print("finished")

if __name__ == "__main__":
    args = get_args()
    if "antmaze" not in args.env:
        args.sample_per_epoch=1000000
        args.seed_per_evaluation = 10
    else:
        args.sample_per_epoch=1000000
        args.seed_per_evaluation = 100
    args.batch_size = 256
    critic(args)