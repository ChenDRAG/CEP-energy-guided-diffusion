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
from scipy.special import softmax
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import torch
from torch.distributions import Independent, Normal
MAX_BZ_SIZE = 1024
DISCOUNT = 0.99


# Dataset iterator
def inf_train_gen(data, batch_size=200):
    print(data)
    if data == "swissroll":
        print(data)
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.sum(data**2, axis=-1,keepdims=True) / 9.0
    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        center_dist = X[:,0]**2 + X[:,1]**2
        energy = np.zeros_like(center_dist)

        energy[(center_dist >=8.5)] = 0.667 
        energy[(center_dist >=5.0) & (center_dist <8.5)] = 0.333 
        energy[(center_dist >=2.0) & (center_dist <5.0)] = 1.0 
        energy[(center_dist <2.0)] = 0.0

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), energy[:,None]


    elif data == "moons":
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data.astype(np.float32), (y > 0.5).astype(np.float32)[:,None]

    elif data == "8gaussians":
        scale = 4.
        centers = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                    (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        
        
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        indexes = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            indexes.append(idx)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.array(indexes, dtype="float32")[:,None] / 7.0

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x, np.clip((1-np.concatenate([n,n]) / 10),0,1)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        points = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        points_x = points[:,0]
        judger = ((points_x > 0) & (points_x <= 2)) | ((points_x <= -2))
        return points, judger.astype(np.float32)[:,None]

    elif data == "line":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        assert False

class Toy_dataset(torch.utils.data.Dataset):
    def __init__(self, name, datanum=1000000):
        assert name in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
        self.datanum =datanum
        self.name = name
        self.datas, self.energy = inf_train_gen(name, batch_size=datanum)
        self.datas = torch.Tensor(self.datas).to("cuda")
        self.energy = torch.Tensor(self.energy).to("cuda")
        self.datadim = 2
      
    def __getitem__(self, index):
        return {"a": self.datas[index], "e": self.energy[index]}

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.datanum


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

class DQL_buffer(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        data = d4rl.qlearning_dataset(gym.make(args.env))
        self.device = args.device
        self.states = torch.from_numpy(data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(data['next_observations']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float().to(self.device)
        self.is_finished = torch.from_numpy(data['terminals']).view(-1, 1).float().to(self.device)


        reward_tune = "iql_antmaze" if "antmaze" in args.env else "iql_locomotion"
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            min_ret, max_ret = return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        print("dql dataloard loaded")
        
        self.len = self.states.shape[0]
        # self.fake_len = int(np.maximum(np.round(args.sample_per_epoch / self.len), 1)) * self.len
        self.fake_len = self.len
        print(self.len, "data loaded", self.fake_len, "data faked")

    def __getitem__(self, index):
        data = {'s': self.states[index % self.len],
                'a': self.actions[index % self.len],
                'r': self.rewards[index % self.len],
                's_':self.next_states[index % self.len],
                'd': self.is_finished[index % self.len],
                'fake_a':self.fake_actions[index % self.len],  # self.fake_actions <D, 16, A>
                'fake_a_':self.fake_next_actions[index % self.len],  # self.fake_next_actions <D, 16, A>
            }
        return data

    def __add__(self, other):
        pass
    def __len__(self):
        return self.fake_len



class Diffusion_buffer(torch.utils.data.Dataset):
    # discarded implementation, only used for training RL behavior model
    def __init__(self, args):
        self.args=args
        self.normalise_return = args.normalise_return
        data = self._load_data(args)
        
        returns = data["returns"]
        # self.raw_returns = [returns]
        # self.raw_values = []
        self.returns_mean = np.mean(returns).astype(np.float32)
        self.returns_std = np.maximum(np.std(returns), 0.1).astype(np.float32)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            if "maze" in args.env:
                data["rewards"] -= 1
                print("reward minused by 1")
            else:
                max_min = np.max(data["returns"]) - np.min(data["returns"])
                self.max_min = max_min
                data["rewards"] = (data["rewards"]).astype(np.float32) / max_min * 1000.0
                print("returns and rewards normalised at mean {}, std {}".format(self.returns_mean, max_min / 1000.0))
        else:
            print("no normalization return and reward")

        self.actions = data["actions"].to("cuda").astype(np.float32)
        self.states = data["states"].to("cuda").astype(np.float32)
        self.next_states = data["next_states"].to("cuda").astype(np.float32)
        self.rewards = data["rewards"].to("cuda").astype(np.float32)
        self.done = data["done"].to("cuda").astype(np.float32)
        self.is_finished = data["is_finished"].to("cuda").astype(np.float32)
        
        self.len = self.states.shape[0]
        self.fake_len = int(np.maximum(np.round(args.sample_per_epoch / self.len), 1)) * self.len
        print(self.len, "data loaded", self.fake_len, "data faked")


    def __getitem__(self, index):
        data = {'s': self.states[index % self.len],
                'a': self.actions[index % self.len],
                'r': self.rewards[index % self.len],
                's_':self.next_states[index % self.len],
                'd': self.is_finished[index % self.len],
                'fake_a':self.fake_actions[index % self.len],  # self.fake_actions <D, 16, A>
                'fake_a_':self.fake_next_actions[index % self.len],  # self.fake_next_actions <D, 16, A>
            }
        return data

    def __add__(self, other):
        pass
    def __len__(self):
        return self.fake_len
    
    def _load_data(self, args):
        if "hopper" in args.env or "walker" in args.env or "halfcheetah" in args.env or "pen" in args.env or "hammer" in args.env or "door" in args.env or "relocate" in args.env or "kitchen" in args.env:
            t = gym.make(args.env)
            dataset = t.get_dataset()
            if not dataset["terminals"][-1]:
                dataset["timeouts"][-1] = True
            data = {}
            assert not np.any(dataset["terminals"] & dataset["timeouts"])
            data["states"] = dataset["observations"]
            data["actions"] = dataset["actions"]
            data["next_states"] = dataset["next_observations"]
            data["rewards"] = dataset["rewards"][:, None]
            data["done"] = (dataset["terminals"] | dataset["timeouts"])[:, None]
            data["is_finished"] = dataset["terminals"][:, None]
            assert data["done"][-1, 0]
            data["returns"] = np.zeros((data["states"].shape[0], 1))
            last = 0
            for i in range(data["returns"].shape[0] - 1, -1, -1):
                last = data["rewards"][i, 0] + DISCOUNT * last * (1. - data["done"][i, 0])
                data["returns"][i, 0] = last
        elif "maze" in args.env:
            t = gym.make(args.env)
            dataset = t.get_dataset()
            if dataset["terminals"][-1] == False:
                dataset["timeouts"][-1] = True
            assert "observations_next" not in dataset and "next_observations" not in dataset 
            assert dataset["timeouts"].shape[0] == dataset["observations"].shape[0]
            data = {}
            if "maze2d" in args.env:
                assert np.sum(dataset["terminals"]) == 0
                assert np.max(dataset["observations"][1:,0] - dataset["observations"][:-1,0]) < 1.0
                data["states"] = dataset["observations"][:-1, :]
                data["next_states"] = dataset["observations"][1:, :] 
                data["done"] = np.zeros((data["states"].shape[0], 1))
                data["done"][-1,0]  = 1
                data["is_finished"] = np.zeros((data["states"].shape[0], 1))
                data["actions"] = dataset["actions"][:-1, :]
                data["rewards"] = dataset["rewards"][:-1, None]
                data["returns"] = np.zeros((data["states"].shape[0], 1))
                last = 0
                for i in range(data["returns"].shape[0] - 1, -1, -1):
                    last = data["rewards"][i, 0] + DISCOUNT * last * (1. - data["done"][i, 0])
                    data["returns"][i, 0] = last
            elif "antmaze" in args.env:
                if args.env == "antmaze-medium-play-v2":
                    # solve the little bug within this dataset
                    rewid = np.where(dataset["rewards"]>0.001)[0]
                    positions = dataset["observations"][rewid, :2]
                    badid = rewid[~np.all(positions > 19.0, axis=1)]
                    print("{} badid detected".format(badid.shape[0]))
                    dataset["rewards"][badid] = 0.0
                    dataset["terminals"][badid] = 0
                assert set(np.where(np.abs(dataset["observations"][1:,0] - dataset["observations"][:-1,0]) > 1.)[0]).issubset(set(np.where(dataset["timeouts"])[0]))
                assert np.all(np.where(dataset["rewards"])[0] == np.where(dataset["terminals"])[0])
                doneid = dataset["terminals"] | dataset["timeouts"]
                start_id = np.where(doneid)[0]+1
                assert start_id[-1] == doneid.shape[0]
                assert start_id[0] != 0
                start_id = [0] + [i for i in start_id]
                data = {"states":[], "next_states":[], "done":[], "is_finished":[], "rewards":[], "actions":[]}
                for i in range(len(start_id) - 1):
                    if start_id[i+1] - start_id[i] < 5:
                        continue
                    if dataset["terminals"][start_id[i+1]-1]:
                        data["states"].append(dataset["observations"][start_id[i]: start_id[i+1]])
                        next_states = np.zeros_like(data["states"][-1])
                        next_states[:-1] = data["states"][-1][1:]
                        data["next_states"].append(next_states)
                        data["actions"].append(dataset["actions"][start_id[i]: start_id[i+1]])
                        data["rewards"].append(dataset["rewards"][start_id[i]: start_id[i+1], None])
                        done = np.zeros((data["states"][-1].shape[0], 1))
                        done[-1, 0] = 1
                        data["done"].append(done)
                        data["is_finished"].append(done)
                    elif dataset["timeouts"][start_id[i+1]-1]:
                        data["states"].append(dataset["observations"][start_id[i]: start_id[i+1]-1])
                        data["next_states"].append(dataset["observations"][start_id[i]+1: start_id[i+1]])
                        data["actions"].append(dataset["actions"][start_id[i]: start_id[i+1]-1])
                        data["rewards"].append(dataset["rewards"][start_id[i]: start_id[i+1]-1, None])
                        done = np.zeros((data["states"][-1].shape[0], 1))
                        done[-1, 0] = 1
                        data["done"].append(done)
                        data["is_finished"].append(np.zeros_like(data["rewards"][-1]))
                    else:
                        assert False
                for k in ["states", "next_states", "done", "is_finished", "rewards", "actions"]:
                    data[k] = np.concatenate(data[k])
                    size = data[k].shape[0]
                print("data num {}".format(size))
                for k in ["states", "next_states", "done", "is_finished", "rewards", "actions"]:
                    assert data[k].shape[0] == size
                    assert data[k].ndim == 2
                    # bootstrap by 0 ignore is_finished
                data["returns"] = np.zeros((data["states"].shape[0], 1))
                last = 0
                for i in range(data["returns"].shape[0] - 1, -1, -1):
                    last = data["rewards"][i, 0] + DISCOUNT * last * (1. - data["done"][i, 0])
                    data["returns"][i, 0] = last
        elif "Toy" in args.env or "toy" in args.env :
            if "single" in args.env:
                data = np.load("./dataset/toy-singleside-v0.npy", allow_pickle=True).item()
            else:
                data = np.load("./dataset/toy-v0.npy", allow_pickle=True).item()
            data["returns"] = np.zeros((data["states"].shape[0], 1))
            last = 0
            for i in range(data["returns"].shape[0] - 1, -1, -1):
                last = data["rewards"][i, 0] + DISCOUNT * last * (1. - data["done"][i, 0])
                data["returns"][i, 0] = last
        else:
            raise NotImplementedError
        
        
        return data
