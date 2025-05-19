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
    def __init__(self, agent_id, actor_in, actor_out, critic_in, hidden):
        super(ActorCritic, self).__init__()
        self.agent_id = agent_id
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
        probs[~action_mask] = -1e10
        probs = self.softmaxer(probs)
        dist = Categorical(probs)
        # get action and log prob of action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # value
        value = self.critic(real_obs)
        return action.detach(), log_prob.detach(), value.detach()
    
    def evaluate(self, real_obs, actions_masks, actions):
        # get action log prob
        probs = self.actor(real_obs)
        probs[~actions_masks] = -1e10
        probs = self.softmaxer(probs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        # value
        values = self.critic(real_obs)
        return log_probs, values
    
    def act_real(self, real_obs, action_mask):
        with torch.no_grad():
            # action probs
            probs = self.actor(real_obs)
            probs[~action_mask] = -1e10
            probs = self.softmaxer(probs)
            action = torch.argmax(probs)
            return action

class Agent:
    def __init__(self, agent_id, gamma, eps_clip, update_epochs,
                 actor_in, actor_out, actor_lr,
                 critic_in, critic_lr,
                 hidden, device="cpu"):
        self.agent_id = agent_id
        # policy
        self.policy = ActorCritic(agent_id, actor_in, actor_out, critic_in, hidden).to(device)
        self.optimizer = optim.Adam([
                        {"params": self.policy.actor.parameters(), "lr": actor_lr},
                        {"params": self.policy.critic.parameters(), "lr": critic_lr}
                    ])
        self.critic_loss_func = nn.MSELoss()
        self.policy_old = ActorCritic(agent_id, actor_in, actor_out, critic_in, hidden).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # update policy variables
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        # memory buffer
        self.memory = MemoryBuffer()
    
    def select_action(self, state):
        with torch.no_grad():
            real_obs = torch.reshape(state["real_obs"].clone(), (-1,))
            action_mask = state["agents", "action_mask"][self.agent_id].clone()
            # choose action by using initial unchanged "old" policy
            action, log_prob, value = self.policy_old.act(real_obs, action_mask)
        self.memory.initial_states.append(real_obs)
        self.memory.actions.append(action)
        self.memory.actions_masks.append(action_mask)
        self.memory.log_probs.append(log_prob)
        self.memory.values.append(value)
        return action.item()
    
    def make_action(self, state):
        with torch.no_grad():
            real_obs = torch.reshape(state["real_obs"].clone(), (-1,))
            action_mask = state["agents", "action_mask"][self.agent_id].clone()
            action = self.policy_old.act_real(real_obs, action_mask)
            return action
    
    
         
class MultiAgent:
    def __init__(self, n_agents,
                 gamma, eps_clip, update_epochs,
                 actor_in, actor_out, actor_lr,
                 critic_in, critic_lr,
                 hidden, device="cpu"):
        self.device = device
        self.agents = []
        for i in range(n_agents):
            agent = Agent(i, gamma, eps_clip, update_epochs,
                          actor_in, actor_out, actor_lr,
                          critic_in, critic_lr,
                          hidden, device)
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
                action = agent.select_action(state)
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
        # update agents policies
        # total_losses = []
        for agent in self.agents:
            future_rewards = []
            discounted_reward = 0
            for reward in reversed(agent.memory.rewards):
                discounted_reward = reward + agent.gamma * discounted_reward
                future_rewards.insert(0, discounted_reward)
            future_rewards = torch.tensor(future_rewards, dtype=torch.float32).to(self.device)
            old_initial_states = torch.squeeze(torch.stack(agent.memory.initial_states, dim=0)).detach().to(self.device)
            old_actions = torch.squeeze(torch.stack(agent.memory.actions, dim=0)).detach().to(self.device)
            old_actions_masks = torch.squeeze(torch.stack(agent.memory.actions_masks, dim=0)).detach().to(self.device)
            old_log_probs = torch.squeeze(torch.stack(agent.memory.log_probs, dim=0)).detach().to(self.device)
            old_values = torch.squeeze(torch.stack(agent.memory.values, dim=0)).detach().to(self.device)
            
            advantages = future_rewards.detach() - old_values.detach()
            
            # total_loss = 0
            for _ in range(agent.update_epochs):
                log_probs, values = agent.policy.evaluate(old_initial_states, old_actions_masks, old_actions)
                values = torch.squeeze(values)
                ratios = torch.exp(log_probs - old_log_probs.detach())
                # surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-agent.eps_clip, 1+agent.eps_clip) * advantages
                # loss
                loss = -torch.min(surr1, surr2) + 0.5 * agent.critic_loss_func(values, future_rewards)
                # total_loss += loss.mean().detach()
                # gradient step
                agent.optimizer.zero_grad()
                loss.mean().backward()
                agent.optimizer.step()
            # total_losses.append(total_loss)
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            agent.memory.clear_memory()
        # return total_losses
    
    def test(self, env):
        state = env.reset()
        done = False
        while not done:
            actions = []
            # agents choose actions
            for agent in self.agents:
                action = agent.make_action(state)
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
