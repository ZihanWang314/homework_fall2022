from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.infrastructure import sac_utils
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()), 
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.loss = torch.nn.MSELoss()

    def _compute_q_targets(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        entropy_scale = self.actor.alpha
        next_actions = self.actor.get_action(next_ob_no)
        next_actions = ptu.from_numpy(next_actions)
        next_log_pis = self.actor(next_ob_no).log_prob(next_actions).sum(-1)
        next_q_values = torch.stack(self.critic_target(next_ob_no, next_actions)).min(0)[0]
        next_values = next_q_values - entropy_scale * next_log_pis
        q_targets = re_n + self.gamma * (1 - terminal_n) * next_values
        q_targets = q_targets.detach()
        return q_targets

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        # ob_no = ptu.from_numpy(ob_no)
        # ac_na = ptu.from_numpy(ac_na).to(torch.long)
        # next_ac_na = self.actor.get_action(next_ob_no, sample=False)
        # next_ob_no = ptu.from_numpy(next_ob_no)
        # next_ac_na = ptu.from_numpy(next_ac_na)
        # reward_n = ptu.from_numpy(re_n)
        # terminal_n = ptu.from_numpy(terminal_n)

        # q_t_values_1, q_t_values_2 = self.critic(ob_no, ac_na)        

        # q_tp1_1, q_tp1_2 = self.critic_target(next_ob_no, next_ac_na)


        # target_1 = reward_n + self.gamma * q_tp1_1 * (1 - terminal_n)
        # target_2 = reward_n + self.gamma * q_tp1_2 * (1 - terminal_n)
        # target_1 = target_1.detach()
        # target_2 = target_2.detach()
        # assert q_t_values_1.shape == target_1.shape
        # critic_loss = self.loss(q_t_values_1, target_1) / 2 + self.loss(q_t_values_2, target_2) / 2
        # print(target_1.mean(), q_t_values_1.mean())

        # critic_loss.backward()
        # self.critic.optimizer.step()
        # self.critic.optimizer.zero_grad()

        # return critic_loss
        ob_no, ac_na, next_ob_no, re_n, terminal_n = map(ptu.from_numpy, [ob_no, ac_na, next_ob_no, re_n, terminal_n])
        q_targets = self._compute_q_targets(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        q_values_1, q_values_2 = self.critic(ob_no, ac_na)
        loss_1, loss_2 = self.loss(q_values_1, q_targets), self.loss(q_values_2, q_targets)
        loss = (loss_1 + loss_2) / 2
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()
        return loss.detach()


    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if self.training_step % self.agent_params['critic_target_update_frequency'] == 0:
            sac_utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        if self.training_step % self.agent_params['actor_update_frequency'] == 0:
            for i in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)
        else:
            actor_loss, alpha_loss, alpha = 0, 0, 0
        self.training_step += 1
        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
