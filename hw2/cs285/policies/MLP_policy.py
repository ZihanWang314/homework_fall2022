import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        if not self.discrete:
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
            
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
        
        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ptu.device = 'cuda'
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        
        # TODO return the action that the policy prescribes
        observation = torch.tensor(observation).float().to(ptu.device)
        if self.discrete:
            pred = self(observation)
            pred = pred.softmax(-1)
            actions = torch.rand(pred.shape[0]).to(ptu.device) > pred[:, 0]
            return actions.long().cpu().detach().numpy()
        else:
            mu = self(observation)
            m = distributions.Normal(mu, self.logstd.exp())
            actions = m.sample()
            return actions.cpu().detach().numpy()
                

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        return self.mean_net(observation)



#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()
        

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        if q_values is not None:
            q_values = ptu.from_numpy(q_values)
        # TODO: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        if len(actions.shape) == 1: # discrete action space:
            action_onehot = nn.functional.one_hot(actions.long()).bool()
            prob = self(observations).softmax(-1)[action_onehot]
            log_prob = torch.log(prob)
            loss = - (log_prob * advantages).mean()
        else:
            mu = self(observations)
            std = self.logstd.exp()
            prob = torch.exp(-  (1/2) * (((actions - mu) / std) ** 2))
            log_prob = torch.log(prob).sum(dim=-1)
            loss = - (log_prob * advantages).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.nn_baseline:
            ## TODO: update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.

            ## Note: You will need to convert the targets into a tensor using
                ## ptu.from_numpy before using it in the loss
            q_values = q_values - q_values.mean() / (q_values.var() + 1e-16).sqrt()
            loss_baseline = self.baseline_loss(q_values, self.baseline(observations).squeeze())
            loss_baseline.backward()
            self.baseline_optimizer.step()
            self.baseline_optimizer.zero_grad()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())

# seed = 1
# 63 epoches 1e-2 2000: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 2000 -lr 1e-2 -rtg --exp_name q2_b2000_r1e-2
# 52 epoches 2e-2 2000: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 2000 -lr 2e-2 -rtg --exp_name q2_b2000_r2e-2
# diverge 3e-2 2000: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 2000 -lr 3e-2 -rtg --exp_name q2_b2000_r3e-2
# 60 epoches 2e-2 1000: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 2e-2 -rtg --exp_name q2_b1000_r2e-2
# 57 epoches 2e-2 500: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 2e-2 -rtg --exp_name q2_b500_r2e-2
# 43 epoches 2e-2 300: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 300 -lr 2e-2 -rtg --exp_name q2_b300_r2e-2
# 18 epoches 3e-2 3000: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 3000 -lr 3e-2 -rtg --exp_name q2_b3000_r3e-2
# 28 epoches 5e-2 3000: python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 3000 -lr 5e-2 -rtg --exp_name q2_b3000_r5e-2
    