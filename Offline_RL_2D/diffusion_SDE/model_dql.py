import torch
import torch.nn as nn
import gym
import d4rl
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from diffusion_SDE import dpm_solver_pytorch
from diffusion_SDE import schedule
from scipy.special import softmax
MAX_BZ_SIZE = 1024
DEFAULT_DEVICE = "cuda"

def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim),SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)

class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))
    
class TripleQ(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)
        self.q3 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_), self.q3(as_)

    def forward(self, action, condition=None):
        q1, q2, q3 = self.both(action, condition)
        return torch.min(q1, torch.min(q2, q3))

class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)

class GaussPolicy(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.net = mlp([state_dim, 256, 256, action_dim], output_activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)

class GuidanceQt(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [action_dim+32+state_dim, 256, 256, 256, 256, 1]
        self.qt = mlp(dims, activation=SiLU)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))
        
    def forward(self, action, t, condition=None):
        embed = self.embed(t)
        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)

class Critic_Guide(nn.Module):
    def __init__(self, adim, sdim) -> None:
        super().__init__()
        # is sdim is 0  means unconditional guidance
        self.conditional_sampling = False if sdim==0 else True
        self.q0 = None
        self.qt = None

    def forward(self, a, condition=None):
        return self.q0(a, condition)

    def calculate_guidance(self, a, t, condition=None):
        raise NotImplementedError
    
    def calculateQ(self, a, condition=None):
        return self(a, condition)
    
    def update_q0(self, data):
        raise NotImplementedError
    
    def update_qt(self, data):
        raise NotImplementedError

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Expectile_Critic_Guide(Critic_Guide):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__(adim, sdim)
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.q0 = TwinQ(adim, sdim).to(DEFAULT_DEVICE)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = ValueFunction(sdim).to(DEFAULT_DEVICE)
        self.qt = GuidanceQt(adim, sdim).to(DEFAULT_DEVICE)
        self.gaussian_policy = GaussPolicy(adim, sdim).to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4)
        self.policy_optimizer = torch.optim.Adam(self.gaussian_policy.parameters(), lr=3e-4)
        self.q_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.q_optimizer, T_max=200, eta_min=0.)
        self.qt_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.qt_optimizer, T_max=200, eta_min=0.)
        self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=200, eta_min=0.)
        self.tau = 0.9 if "maze" in args.env else 0.7
        self.discount = 0.99
        self.grad_norm = 0.0
        
        self.args = args
        self.alpha = args.alpha
        self.guidance_scale = args.s
    
    def change_lr(self):
        self.q_lr_scheduler.step()
        self.qt_lr_scheduler.step()
        self.policy_lr_scheduler.step()

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            guidance =  self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def update_q0_mine(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]

        fake_a = data['fake_a']
        fake_a_ = data['fake_a_']
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            # target_q = 
            # next_v = self.vf(s_).detach()
            softmax = nn.Softmax(dim=1)
            next_energy = self.q0_target(fake_a_ , torch.stack([s_]*fake_a_.shape[1] ,axis=1)).detach().squeeze() # <bz, 16>            
            next_v = torch.sum(softmax(self.args.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True)

            energy = self.q0_target(fake_a , torch.stack([s]*fake_a.shape[1] ,axis=1)).detach().squeeze() # <bz, 16>
            v = torch.sum(softmax(self.args.q_alpha * energy) * energy, dim=-1, keepdim=True)

            adv = target_q - v
        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.grad_norm > 0.1:
            assert False
            self.critic_grad_norms = nn.utils.clip_grad_norm_(self.q0.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.q_optimizer.step()
        
        # Update target
        update_target(self.q0, self.q0_target, 0.005)

        # Update policy
        exp_adv = torch.exp(self.alpha * adv.detach()).clamp(max=100.0).squeeze() # should be 90-150
        policy_out = self.gaussian_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)
        policy_loss = torch.mean(exp_adv * bc_losses)
        # self.policy_optimizer.zero_grad()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def update_qt(self, data):
        # input  many s <bz, S>  anction <bz, M, A>, energy?  None or <bz, M, 1>
        s = data['s']
        a = data['a']
        fake_a = data['fake_a']
        energy = self.q0_target(fake_a , torch.stack([s]*fake_a.shape[1] ,axis=1)).detach().squeeze()

        self.all_mean = torch.mean(energy, dim=-1).detach().cpu().squeeze().numpy()
        self.all_std = torch.std(energy, dim=-1).detach().cpu().squeeze().numpy()
        
        if self.args.method == "mse":
            random_t = torch.rand(a.shape[0], device=DEFAULT_DEVICE) * (1. - 1e-3) + 1e-3
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            # calculate sample based baselines
            # sample_based_baseline = torch.max(energy, dim=-1, keepdim=True)[0]  #<bz , 1>
            sample_based_baseline = 0.0
            self.debug_used = (self.q0_target(a, s).detach() * self.alpha - sample_based_baseline * self.alpha).detach().cpu().squeeze().numpy()
            loss = torch.mean((self.qt(perturbed_a, random_t, s) - self.q0_target(a, s).detach() * self.alpha + sample_based_baseline * self.alpha)**2)
        elif self.args.method == "emse":
            random_t = torch.rand(a.shape[0], device=DEFAULT_DEVICE) * (1. - 1e-3) + 1e-3
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            # calculate sample based baselines
            # sample_based_baseline = (torch.logsumexp(energy*self.alpha, dim=-1, keepdim=True)- np.log(energy.shape[1])) /self.alpha   #<bz , 1>
            sample_based_baseline = torch.max(energy, dim=-1, keepdim=True)[0]  #<bz , 1>
            self.debug_used = (self.q0_target(a, s).detach() * self.alpha - sample_based_baseline * self.alpha).detach().cpu().squeeze().numpy()
            def unlinear_func(value, alpha, clip=False):
                if clip:
                    return torch.exp(torch.clamp(value*alpha, -100, 4.5))
                else:
                    return torch.exp(value*alpha)
            loss = torch.mean((unlinear_func(self.qt(perturbed_a, random_t, s), 1.0, clip=True) - unlinear_func(self.q0_target(a, s).detach() - sample_based_baseline, self.alpha, clip=True))**2)
        elif self.args.method == "cross_entrophy":
            logsoftmax = nn.LogSoftmax(dim=1)
            softmax = nn.Softmax(dim=1)
            
            x0_data_energy = energy * self.alpha
            # random_t = torch.rand((fake_a.shape[0], fake_a.shape[1]), device=DEFAULT_DEVICE) * (1. - 1e-3) + 1e-3
            random_t = torch.rand((fake_a.shape[0], ), device=DEFAULT_DEVICE) * (1. - 1e-3) + 1e-3
            random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
            z = torch.randn_like(fake_a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
            xt_model_energy = self.qt(perturbed_fake_a, random_t, torch.stack([s]*fake_a.shape[1] ,axis=1)).squeeze()
            p_label = softmax(x0_data_energy)
            self.debug_used = torch.flatten(p_label).detach().cpu().numpy()
            loss = -torch.mean(torch.sum(p_label * logsoftmax(xt_model_energy), axis=-1))  #  <bz,M>
        else:
            raise NotImplementedError

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()
        
class MlpQnet(nn.Module):
    def __init__(self, sdim, adim) -> None:
        super().__init__()
        self.sort = nn.Sequential(nn.Linear(sdim+adim, 256), SiLU(), nn.Linear(256, 256), SiLU(),nn.Linear(256, 1))
    def forward(self, s, a):
        return self.sort(torch.cat([s,a], axis=-1))

class MUlQnet(nn.Module):
    def __init__(self, sdim, adim) -> None:
        super().__init__()
        self.sort = nn.Sequential(nn.Linear(sdim+adim, 512), SiLU(), nn.Linear(512, 256), SiLU(),nn.Linear(256, 256), SiLU(),nn.Linear(256, 1))
    def forward(self, s, a):
        return self.sort(torch.cat([s,a], axis=-1))
    
class ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device=args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True)
        # self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.q = []
        self.q.append(Expectile_Critic_Guide(adim=output_dim, sdim=input_dim-output_dim, args=args))
        self.args = args

    def forward_dmp_wrapper_fn(self, x, t):
        score = self(x, t)
        result = - (score + self.q[0].calculate_guidance(x, t, self.condition)) * self.marginal_prob_std(t)[1][..., None]
        return result
    
    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def calculateQ(self, s,a,t=None):
        if s is None:
            if self.condition.shape[0] == a.shape[0]:
                s = self.condition
            elif self.condition.shape[0] == 1:
                s = torch.cat([self.condition]*a.shape[0])
            else:
                assert False
        return self.q[0](a,s)
    
    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def _select(self, returns, actions, alpha, num=1, replace=True):
        # returns: (n, 4) actions (n, NA)
        returns = returns[:, 0]
        returns = returns * alpha
        index = np.random.choice(actions.shape[0], p=softmax(returns), size=num, replace=replace)
        selected_returns = returns[index]
        selected_returns = np.exp(selected_returns - np.max(selected_returns))
        self.weighted = selected_returns / np.sum(selected_returns)
        return  actions[index]
    
    def select_actions(self, states, sample_per_state=32, select_per_state=1, alpha=100, replace=False, weighted_mean=False, diffusion_steps=25):
        returns, actions = self.sample(states, sample_per_state, diffusion_steps)
        if isinstance(select_per_state, int):
            select_per_state = [select_per_state] * actions.shape[0]
        if (isinstance(alpha, int) or isinstance(alpha, float)):
            alpha = [alpha] * actions.shape[0]
        if (isinstance(replace, int) or isinstance(replace, float) or isinstance(replace, bool)):
            replace = [replace] * actions.shape[0]
        if (isinstance(weighted_mean, int) or isinstance(weighted_mean, float) or isinstance(weighted_mean, bool)):
            weighted_mean = [weighted_mean] * actions.shape[0]
        # select `select_per_sample` data from 32 data, ideally should be 1.
        # Selection should happen according to `alpha`
        # replace defines whether to put back data
        out_actions = []
        for i in range(actions.shape[0]):
            raw_actions = self._select(returns[i], actions[i], alpha=alpha[i], num=select_per_state[i], replace=replace[i])
            out_actions.append(np.average(raw_actions, weights=self.weighted if weighted_mean[i] else None, axis=0))
        return out_actions
    
    def select_action_debug(self, states, sample_per_state=50):
        states = torch.Tensor(states).unsqueeze(0).to("cuda")

    def sample(self, states, sample_per_state=16, diffusion_steps=15):
        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2)
            returns = self.calculateQ(states, torch.FloatTensor(results).to(self.device)).reshape(num_states, sample_per_state, 1).to("cpu").detach().numpy()
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim).copy()
            self.condition = None
        self.train()
        return returns, actions


class ScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, 32), SiLU())
        self.sort_t = nn.Sequential(
                        nn.Linear(64, 128),                        
                        SiLU(),
                        nn.Linear(128, 128),
                    )
        self.down_block1 = Residual_Block(output_dim, 512)
        self.down_block2 = Residual_Block(512, 256)
        self.down_block3 = Residual_Block(256, 128)
        self.middle1 = Residual_Block(128, 128)
        self.up_block3 = Residual_Block(256, 256)
        self.up_block2 = Residual_Block(512, 512)
        self.last = nn.Linear(1024, output_dim)
        
    def forward(self, x, t, condition=None):
        embed = self.embed(t)
        
        if condition is not None:
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        embed = self.sort_t(embed)
        d1 = self.down_block1(x, embed)
        d2 = self.down_block2(d1, embed)
        d3 = self.down_block3(d2, embed)
        u3 = self.middle1(d3, embed)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed)
        u0 = torch.cat([d1, u1], dim=-1)
        h = self.last(u0)
        self.h = h
        # Normalize output
        return h / self.marginal_prob_std(t)[1][..., None]
    

class MlpScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.dense1 = Dense(embed_dim, 32)
        self.dense2 = Dense(input_dim, 256 - 32)
        self.block1 = nn.Sequential(
            nn.Linear(256, 512),
            SiLU(),
            nn.Linear(512, 256),
            SiLU(),
            nn.Linear(256, 256),
            SiLU(),
            nn.Linear(256, 256),
        )
        self.decoder = Dense(256, output_dim)

    def forward(self, x, t, condition=None):
        if condition is not None:
            x = torch.cat([condition, x])
        else:
            if self.condition.shape[0] == x.shape[0]:
                x = torch.cat([self.condition, x],axis=-1)
            elif self.condition.shape[0] == 1:
                self.condition
                x = torch.cat([torch.cat([self.condition]*x.shape[0]), x],axis=-1)
            else:
                assert False
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))
        # Encoding path
        h = torch.cat((self.dense2(x), self.dense1(embed)),dim=-1)
        
        h = self.block1(h)
        h = self.decoder(self.act(h))
        # Normalize output
        h = h / self.marginal_prob_std(t)[1][..., None]
        return h