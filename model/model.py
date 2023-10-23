import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class DDPG():
    def __init__(
            self,
            model,
            learning_rate = [1e-4, 2e-4],
            reward_decay = 0.98,
            memory_size = 5000,
            batch_size = 64,
            tau = 0.01,
            epsilon_params = [1.0, 0.5, 0.00001], # init var / final var / decay
            criterion = nn.MSELoss()
    ) -> None:
        # initialize parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.criterion = criterion
        self.epsilon_params = epsilon_params
        self.epsilon = self.epsilon_params[0]
        self._build_net(model[0], model[1])
        self.init_memory()
    
    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(self.device)
        self.critic_target.eval()
        
    def save_load_model(self, op, path):
        anet_path = path + "ddpg_anet.pt"
        cnet_path = path + "ddpg_cnet.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif op == "load":
            self.critic.load_state_dict(torch.load(cnet_path, map_location=self.device))
            self.critic_target.load_state_dict(torch.load(cnet_path, map_location=self.device))
            self.actor.load_state_dict(torch.load(anet_path, map_location=self.device))

    def choose_action(self, s, eval=False):
        s_ts = torch.FloatTensor(np.expand_dims(s,0)).to(self.device)
        action = self.actor(s_ts)
        action = action.cpu().detach().numpy()[0]
        
        if eval == False:
            action += np.random.normal(0, self.epsilon, action.shape)
        else:
            action += np.random.normal(0, self.epsilon_params[1], action.shape)
        
        action = np.clip(action, -1, 1)
        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s":[], "a":[], "r":[], "sn":[], "end":[]}

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            self.memory["end"][index] = end

        self.memory_counter += 1

    def soft_update(self):
        # Store sample to replay buffer
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        s_batch = [self.memory["s"][index] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        sn_batch = [self.memory["sn"][index] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]

        # Construct torch tensor
        s_ts = torch.FloatTensor(np.array(s_batch)).to(self.device)
        a_ts = torch.FloatTensor(np.array(a_batch)).to(self.device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(self.device).view(self.batch_size, 1)
        sn_ts = torch.FloatTensor(np.array(sn_batch)).to(self.device)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(self.device).view(self.batch_size, 1)
        
        # TD-target
        with torch.no_grad():
            a_next = self.actor(sn_ts)
            q_next_target = self.critic_target(sn_ts, a_next)
            q_target = r_ts + end_ts * self.gamma * q_next_target
        
        # Critic loss
        q_eval = self.critic(s_ts, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        a_curr = self.actor(s_ts)
        q_current = self.critic(s_ts, a_curr)
        self.actor_loss = -q_current.mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()
        
        # Update target network and noise variance
        self.soft_update()
        if self.epsilon > self.epsilon_params[1]:
            self.epsilon -= self.epsilon_params[2]
        else:
            self.epsilon = self.epsilon_params[1]
        
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())
    

class SAC():
    def __init__(
        self,
        model,
        n_actions,
        learning_rate = [1e-4, 2e-4],
        reward_decay = 0.98,
        replace_target_iter = 300,
        memory_size = 5000,
        batch_size = 64,
        tau = 0.01,
        alpha = 0.5,
        auto_entropy_tuning = True,
        criterion = nn.MSELoss()
    ):
        # initialize parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.criterion = criterion
        self._build_net(model[0], model[1])
        self.init_memory()

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(self.device)
        self.critic_target.eval()

        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.Tensor(self.n_actions).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0001)
    
    def save_load_model(self, op, path):
        anet_path = path + "sac_anet.pt"
        cnet_path = path + "sac_cnet.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif op == "load":
            self.critic.load_state_dict(torch.load(cnet_path, map_location=self.device))
            self.critic_target.load_state_dict(torch.load(cnet_path, map_location=self.device))
            self.actor.load_state_dict(torch.load(anet_path, map_location=self.device))

    def choose_action(self, s, eval=False):
        s_ts = torch.FloatTensor(np.expand_dims(s,0)).to(self.device)
        if eval == False:
            action, _, _ = self.actor.sample(s_ts)
        else:
            _, _, action = self.actor.sample(s_ts)
        
        action = action.cpu().detach().numpy()[0]
        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s":[], "a":[], "r":[], "sn":[], "end":[]}

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            self.memory["end"][index] = end

        self.memory_counter += 1

    def soft_update(self):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        s_batch = [self.memory["s"][index] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        sn_batch = [self.memory["sn"][index] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]

        # Construct torch tensor
        s_ts = torch.FloatTensor(np.array(s_batch)).to(self.device)
        a_ts = torch.FloatTensor(np.array(a_batch)).to(self.device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(self.device).view(self.batch_size, 1)
        sn_ts = torch.FloatTensor(np.array(sn_batch)).to(self.device)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(self.device).view(self.batch_size, 1)
        
        # TD-target
        with torch.no_grad():
            a_next, logpi_next, _ = self.actor.sample(sn_ts)
            q_next_target = self.critic_target(sn_ts, a_next) - self.alpha * logpi_next
            q_target = r_ts + end_ts * self.gamma * q_next_target
        
        # Critic loss
        q_eval = self.critic(s_ts, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        a_curr, logpi_curr, _ = self.actor.sample(s_ts)
        q_current = self.critic(s_ts, a_curr)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        
        # Adaptive entropy adjustment
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())
        
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        mu = torch.tanh(self.fc4(h_fc3))
        return mu
    
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class PolicyNetGaussian(nn.Module):
    def __init__(self):
        super(PolicyNetGaussian, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_mean = nn.Linear(512, 2)
        self.fc4_logstd = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        a_mean = self.fc4_mean(h_fc3)
        a_logstd = self.fc4_logstd(h_fc3)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return a_mean, a_logstd
    
    def sample(self, s):
        a_mean, a_logstd = self.forward(s)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)
    
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512+2, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
    
    def forward(self, s, a):
        h_fc1 = F.relu(self.fc1(s))
        h_fc1_a = torch.cat((h_fc1, a), 1)
        h_fc2 = F.relu(self.fc2(h_fc1_a))
        h_fc3 = F.relu(self.fc3(h_fc2))
        q_out = self.fc4(h_fc3)
        return q_out