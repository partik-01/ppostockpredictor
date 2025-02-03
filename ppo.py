# ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from collections import deque
import random
from stock_trading_env import StockTradingEnv

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (
            torch.FloatTensor(self.states),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.probs),
            torch.FloatTensor(self.vals),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones)
        )

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared features extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, action_dim),
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state):
        features = self.features(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return torch.softmax(action_logits, dim=-1), value

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2,
                 value_clip=0.2, entropy_coef=0.01, max_grad_norm=0.5,
                 batch_size=256, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        self.memory = PPOMemory(batch_size=batch_size)
        self.replay_buffer = ReplayBuffer()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.to(self.device)
        
        # Running mean/std for observation normalization
        self.running_mean = np.zeros(state_dim)
        self.running_std = np.ones(state_dim)
        self.count = 0

        # Add reward normalization
        self.return_rms = RunningMeanStd()
        
        # Add observation normalization
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        
        # Add learning rate annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,  # Adjust based on your total episodes
            eta_min=1e-6
        )
    

    def normalize_reward(self, reward):
        """Normalize rewards using running statistics"""
        self.return_rms.update(np.array([reward]))
        return reward / (self.return_rms.std + 1e-8)

    def normalize_observation(self, obs):
        """Normalize observations using running statistics"""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / (self.obs_rms.std + 1e-8)
    


    def normalize_observation(self, obs):
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_std = np.sqrt(self.running_std**2 + delta * delta2 / self.count)
        return (obs - self.running_mean) / (self.running_std + 1e-8)

    def choose_action(self, state):
        state = self.normalize_observation(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def learn(self):
        states, actions, old_log_probs, old_values, rewards, dones = self.memory.generate_batches()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Calculate advantages using GAE
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = old_values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            action_probs, values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            values = values.squeeze()
            value_clipped = old_values + torch.clamp(values - old_values, 
                                                   -self.value_clip, self.value_clip)
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            # Total loss
            total_loss = actor_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()
        self.memory.clear_memory()

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)
        
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop

def train_ppo(env, agent, num_episodes=1000, max_steps=1000, eval_frequency=10):
    early_stopping = EarlyStopping(patience=50)
    best_reward = float('-inf')
    episode_rewards = []
    eval_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(max_steps):
            action, log_prob, value = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            # Store in replay buffer and PPO memory
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.memory.store_memory(obs, action, log_prob, value, reward, done)
            
            obs = next_obs
            episode_reward += reward
            
            if len(agent.memory.states) >= agent.memory.batch_size:
                agent.learn()
            
            if done:
                break
        
        # Final update
        if len(agent.memory.states) > 0:
            agent.learn()
        
        episode_rewards.append(episode_reward)
        
        # Evaluation phase
        if episode % eval_frequency == 0:
            eval_reward = evaluate_agent(env, agent)
            eval_rewards.append(eval_reward)
            
            # Early stopping check
            if early_stopping(eval_reward):
                print(f"Early stopping triggered at episode {episode}")
                break
            
            # Logging
            if eval_reward > best_reward:
                best_reward = eval_reward
                # Save best model here if needed
            
            avg_reward = np.mean(episode_rewards[-10:])
            print(f'Episode {episode}:')
            print(f'  Training Reward: {episode_reward:.2f}')
            print(f'  Eval Reward: {eval_reward:.2f}')
            print(f'  Portfolio Value: {info["portfolio_value"]:.2f}')
            print(f'  10-Episode Average: {avg_reward:.2f}')
            print(f'  Best Reward: {best_reward:.2f}')
            print(f'  Shares Held: {info["shares_held"]}')
            print(f'  Current Price: {info["current_price"]:.2f}')
            print(f'  Learning Rate: {agent.scheduler.get_last_lr()[0]:.6f}\n')
    
    return episode_rewards, eval_rewards

def evaluate_agent(env, agent, n_episodes=5):
    total_reward = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _, _ = agent.choose_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def main():
    # Load and preprocess data
    df = pd.read_csv('NULB.csv')
    env = StockTradingEnv(df)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=1e-4,
                    batch_size=256,
                    n_epochs=10)
    
    train_rewards, eval_rewards = train_ppo(env, agent)

    # Save the trained model
    torch.save({
        'model_state_dict': agent.actor_critic.state_dict(),
        'running_mean': agent.running_mean,
        'running_std': agent.running_std
    }, 'ppo_model.pth')
    
    # Plot training progress
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Training Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_rewards)
    plt.title('Evaluation Progress')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Evaluation Reward')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()