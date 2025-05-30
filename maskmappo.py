import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from torchrl.envs.utils import step_mdp

# class to store solution trajectory data
class MemoryBuffer:
    def __init__(self):
        self.initial_states = []
        self.actions = []
        self.actions_masks = []
        self.log_probs = []
        self.rewards = []
        self.values = []
    
    def clear_memory(self):
        del self.initial_states[:]
        del self.actions[:]
        del self.actions_masks[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]

# actual policy, with actor and critic
class ActorCritic(nn.Module):
    def __init__(self, actor_in, actor_out, critic_in, hidden):
        super(ActorCritic, self).__init__()
        # actor / policy network
        self.actor = nn.Sequential(
            nn.Linear(in_features=actor_in, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=actor_out),
        )
        self.softmaxer = nn.Softmax()
        # critic / value network
        self.critic = nn.Sequential(
            nn.Linear(in_features=critic_in, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=1),
        )
    
    def act(self, real_obs, action_mask):
        # action probs
        probs = self.actor(real_obs)
        masked_probs = probs.clone()
        masked_probs[~action_mask] = -1e10
        
        probs = self.softmaxer(probs)
        masked_probs = self.softmaxer(masked_probs)
        
        dist = Categorical(probs)
        masked_dist = Categorical(masked_probs)
        # get action and log prob of action
        action = masked_dist.sample()
        log_prob = dist.log_prob(action)
        # value
        value = self.critic(real_obs)
        return action.detach(), log_prob.detach(), value.detach()
    
    def evaluate(self, real_obs, actions_masks, actions):
        # get action log prob
        probs = self.actor(real_obs)
        probs = self.softmaxer(probs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        # value
        values = self.critic(real_obs)
        return log_probs, values, entropy
    
    def act_real(self, real_obs, action_mask):
        with torch.no_grad():
            # action probs
            probs = self.actor(real_obs)
            probs[~action_mask] = -1e10
            probs = self.softmaxer(probs)
            action = torch.argmax(probs)
            return action

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        # memory buffer
        self.memory = MemoryBuffer()
    
    def select_action(self, state, policy):
        with torch.no_grad():
            real_obs = torch.reshape(state["real_obs"].clone(), (-1,))
            action_mask = state["agents", "action_mask"][self.agent_id].clone()
            # choose action by using initial unchanged "old" policy
        action, log_prob, value = policy.act(real_obs, action_mask)
        self.memory.initial_states.append(real_obs)
        self.memory.actions.append(action)
        self.memory.actions_masks.append(action_mask)
        self.memory.log_probs.append(log_prob)
        self.memory.values.append(value)
        return action.item()
    
    def make_action(self, state, policy):
        with torch.no_grad():
            real_obs = torch.reshape(state["real_obs"].clone(), (-1,))
            action_mask = state["agents", "action_mask"][self.agent_id].clone()
            action = policy.act_real(real_obs, action_mask)
            return action
    
    
         
class MultiAgent:
    def __init__(self, n_agents,
                 gamma, eps_clip, update_epochs,
                 value_coef, entropy_coef,
                 actor_in, actor_out, actor_lr,
                 critic_in, critic_lr,
                 hidden):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        # networks
        self.policy = ActorCritic(actor_in, actor_out, critic_in, hidden)
        self.policy_old = ActorCritic(actor_in, actor_out, critic_in, hidden)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # value loss function
        self.critic_loss_func = nn.MSELoss()
        # coefs for loss function
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        # optimizer
        self.optimizer = optim.Adam([
                        {"params": self.policy.actor.parameters(), "lr": actor_lr},
                        {"params": self.policy.critic.parameters(), "lr": critic_lr}
                    ])
        # agents
        self.agents = []
        for i in range(n_agents):
            agent = Agent(i)
            self.agents.append(agent)
    
    def train(self, env):
        # start task
        state = env.reset()
        for agent in self.agents: agent.memory.clear_memory()
        done = False
        # get batch of data during completing the task
        while not done:
            actions = []
            # agents choose actions based on their policies
            for agent in self.agents:
                action = agent.select_action(state, self.policy_old)
                actions.append(action)
            actions = torch.tensor(actions, dtype=torch.int32)
            # send actions to environment
            state["agents", "action"] = actions
            step = env.step(state)
            # claim rewards
            rewards = step["next", "agents", "reward"]
            for agent in self.agents:
                agent.memory.rewards.append(rewards[agent.agent_id])
            state = step_mdp(step)
            done = state["done"]
        # update agents policy
        total_losses = []
        value_losses = []
        for agent in self.agents:
            future_rewards = []
            discounted_reward = 0
            # calculate advantage
            for reward in reversed(agent.memory.rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                future_rewards.insert(0, discounted_reward)
            future_rewards = torch.tensor(future_rewards, dtype=torch.float32)
            future_rewards = (future_rewards - future_rewards.mean()) / (future_rewards.std() + 1e7)
            old_initial_states = torch.squeeze(torch.stack(agent.memory.initial_states, dim=0)).detach()
            old_actions = torch.squeeze(torch.stack(agent.memory.actions, dim=0)).detach()
            old_actions_masks = torch.squeeze(torch.stack(agent.memory.actions_masks, dim=0)).detach()
            old_log_probs = torch.squeeze(torch.stack(agent.memory.log_probs, dim=0)).detach()
            old_values = torch.squeeze(torch.stack(agent.memory.values, dim=0)).detach()
            
            advantages = future_rewards.detach() - old_values.detach()
            
            total_loss = 0
            value_loss = 0
            for i in range(self.update_epochs):
                log_probs, values, entropy = self.policy.evaluate(old_initial_states, old_actions_masks, old_actions)
                values = torch.squeeze(values)
                ratios = torch.exp(log_probs - old_log_probs.detach())
                # surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                # loss
                value_loss_v = self.critic_loss_func(values, future_rewards)
                loss = -torch.min(surr1, surr2) + self.value_coef * value_loss_v - self.entropy_coef * entropy
                total_loss = loss.mean().detach()
                value_loss = value_loss_v.mean().detach()
                # gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            total_losses.append(total_loss)
            value_losses.append(value_loss)
            self.policy_old.load_state_dict(self.policy.state_dict())
            agent.memory.clear_memory()
        return sum(total_losses) / len(total_losses), sum(value_losses) / len(value_losses)
    
    def test(self, env):
        state = env.reset()
        done = False
        while not done:
            actions = []
            # agents choose actions
            for agent in self.agents:
                action = agent.make_action(state, self.policy_old)
                actions.append(action)
            actions = torch.tensor(actions, dtype=torch.int32)
            # send actions to environment
            state["agents", "action"] = actions
            step = env.step(state)
            # claim rewards
            rewards = step["next", "agents", "reward"]
            for agent in self.agents:
                agent.memory.rewards.append(rewards[agent.agent_id])
            state = step_mdp(step)
            done = state["done"]
        return env.last_time_step

# from jsp_env_ma import MAJSPEnv
# from pathlib import Path

# instance_path = Path(__file__).parent.absolute() / "instances/train/t1"
# env = MAJSPEnv({"instance_path": instance_path})

# # HYPERPARAMS
# # general
# train_epochs = 10
# # ppo loss
# gamma = 0.99
# eps_clip = 0.2
# update_epochs = 20
# entropy_coef = 0.01
# value_coef = 0.5
# # networks
# actor_in = env.jobs * 7
# actor_out = env.jobs + 1
# critic_in = actor_in
# actor_lr = 0.0003
# critic_lr = 0.001
# hidden = 64

# # task solver
# solver = MultiAgent(env.machines, gamma, eps_clip, update_epochs,
#                     value_coef, entropy_coef,
#                     actor_in, actor_out, actor_lr,
#                     critic_in, critic_lr,
#                     hidden)

# losses = []
# value_losses = []
# results = []

# for i in range(train_epochs):
#     loss, value_loss = solver.train(env)
#     losses.append(loss)
#     value_losses.append(value_loss)
#     print("epoch", i, "completed")
#     results.append(solver.test(env))
