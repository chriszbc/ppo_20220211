import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
from scipy import stats
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class CrossFire:
    def __init__(self):
        self.edges, _ = self.load_edges()
        self.attackedAreas = 0

    # load two-way topology
    def load_edges(self):
        file = 'network-brand.txt'
        edges = {}
        B = {}
        topo = []  # topo[i] concludes the nodes connected to node i
        with open(file) as f:
            datas = f.readlines()

        for d in datas:
            d = d.split()
            d = [int(x) for x in d]
            i, j, b = d[0], d[1], d[2]
            if i in edges:
                edges[i].append([j, b])
            else:
                edges[i] = [[j, b]]
            if j in edges:
                edges[j].append([i, b])
            else:
                edges[j] = [[i, b]]

        i = 0
        keys = list(edges.keys())
        keys.sort()
        for key in keys:
            while (i != key):
                i = i + 1
                topo.append([])
            topo.append([j[0] for j in edges[key]])
            B[key] = [j[1] for j in edges[key]]
            i = i + 1

        print('===Loaded topology with ' + str(len(topo)) + ' nodes===')
        return topo, B

    # randomly select a start node and return the largest scale of its linked nodes
    def selectNodes(self, num):
        import random

        edges = self.edges
        all_nodes = set()
        attack_all = [[]]
        start = random.choice(list(range(len(edges))))
        all_nodes.add(start)

        area = 0
        attack_all[area] = set()
        attack_all[area].add(start)

        while (sum([len(a) for a in attack_all]) < num):
            new_nodes = set()
            for s in attack_all[area]:
                for n in edges[s]:
                    new_nodes.add(n)
            attacks = list(new_nodes)
            for a in attack_all:
                attacks = attacks + list(a)
            if len(set(attacks)) <= num:
                attack_all[area] = new_nodes
                all_nodes = set(list(new_nodes) + list(all_nodes))
            else:
                area = area + 1
                attack_all.append(set())
                new_start = random.choice(list(range(len(edges))))
                attack_all[area].add(new_start)
                all_nodes.add(new_start)

        self.attackedAreas = area
        return list(all_nodes)

class Env:
    def __init__(self):
        self.action_set = []
        self.action_total = []
        self.state = []
        self.state_size = 10
        self.matrix = [[0 for i in range(100)] for i in range(100)]
        self.degree_list = [0 for i in range(100)]
        self.degree_change_list = [0 for i in range(100)]
        self.posibility_list = [0 for i in range(100)]
        self.temp_posibility_list = [0 for i in range(100)]
        self.degree_change_list = [0 for i in range(100)]
        flag = 0
        for i in range(self.state_size):
            self.state.append([i, 0, 0])  # [服务编号，状态，状态保持时间]

    def action_process2(self):
        action_total = []
        actionset = []

        for r in range(7):
            contents = []
            uni = []
            if r == 0:
                my_file = open('results_123/action_set_1.txt', 'r+')
                service_num = 1
            if r == 1:
                my_file = open('results_123/action_set_2.txt', 'r+')
                service_num = 1
            if r == 2:
                my_file = open('results_123/action_set_3.txt', 'r+')
                service_num = 1
            if r == 3:
                my_file = open('results_123/action_set_12.txt', 'r+')
                service_num = 2
            if r == 4:
                my_file = open('results_123/action_set_13.txt', 'r+')
                service_num = 2
            if r == 5:
                my_file = open('results_123/action_set_23.txt', 'r+')
                service_num = 2
            if r == 6:
                my_file = open('results_123/action_set_123.txt', 'r+')
                service_num = 3

            for i in range(20000):
                content = my_file.readline()
                # print(content)
                line = content.strip()
                node = line.split(' ')

                if node[0] == "Mapping":
                    break

                j = int(i / service_num)
                if i % service_num == 0:
                    contents.append([])
                    # contents[j].append(node)
                for n in node:
                    contents[j].append(int(n))

                for c in contents:
                    if c not in uni:
                        uni.append(c)

            action_total.append(uni)
        self.action_total = action_total

        for a in action_total:
            actionset.extend(a)
        self.action_set = actionset
        return actionset

    def RequestArrive(self):
        state = [0, 0, 0, 0, 0, 0, 0]
        p = [0.101, 0.101, 0.101, 0.0253, 0.0253, 0.0253,  0.0127]
        p = [0.258, 0.258, 0.258, 0.064, 0.064, 0.064, 0.032]
        p = [0.156, 0.156, 0.156, 0.118, 0.118, 0.118, 0.178]
        r = random.random()
        if r <= 0.156:
            state_num = 0
        elif r > 0.156 and r <= 0.312:
            state_num = 1
        elif r > 0.312 and r <= 0.468:
            state_num = 2
        elif r > 0.468 and r <= 0.586:
            state_num = 3
        elif r > 0.586 and r <= 0.704:
            state_num = 4
        elif r > 0.704 and r <= 0.822:
            state_num = 5
        elif r > 0.822 and r <= 1.0:
            state_num = 6


        state[state_num] = 1
        return state, state_num


    def importFigure(self):  # 载入网络拓扑

        i = 0
        j = 0
        my_file = open('network.txt', 'r')
        content = my_file.readline()
        while (content):
            # content = my_file.readline()
            nodes = content.split()
            self.matrix[int(nodes[0])][int(nodes[1])] = 1
            self.matrix[int(nodes[1])][int(nodes[0])] = 1
            content = my_file.readline()
        # print("matrix", self.matrix)
        return self.matrix

    def calculateDegree(self):  # 计算节点的度 （节点所连边的个数）

        self.matrix = self.importFigure()
        # print("matrix: ", self.matrix)
        global degree_list
        i = 0
        j = 0
        degree = 0
        while (i < 100):
            while (j < 100):
                if (self.matrix[i][j] == 1):
                    degree = degree + 1
                j = j + 1
                self.degree_list[i] = degree
            i = i + 1
            j = 0
            degree = 0
        # print("degree list: ", self.degree_list)
        return self.degree_list

    def degreeChange(self):
        i = 0
        while (i < 100):
            if (self.degree_list[i] >= 7):
                self.degree_change_list[i] = self.degree_list[i] * 10000
            if (self.degree_list[i] < 7 and self.degree_list[i] >= 6):
                self.degree_change_list[i] = self.degree_list[i] * 1000
            if (self.degree_list[i] < 6):
                self.degree_change_list[i] = self.degree_list[i] / 10000
            i = i + 1

    def buildingModel(self):
        sum = 0
        i = 0
        while (i < 100):
            sum = sum + self.degree_change_list[i]
            i = i + 1
        i = 0
        while (i < 100):
            self.posibility_list[i] = float(self.degree_change_list[i] / sum)
            i = i + 1

    def renewPossibilities(self, p):  # 根据权重分配已经消失的概率,消失的是0.8，0.8根据权重分配到每个元素中，权重计算为0.05/0.2,0.05/0.2,0.02/0.2,0.08/0.2
        temp = self.posibility_list[p]
        for i in range(self.posibility_list.__len__()):
            # 先计算权重,如果不等于P，则计算公式为posibility_list[i] = posibility_list[i] + posibility_list[p]*(posibility_list[i]/(1-posibility_list[p])
            if (i == p):
                self.posibility_list[i] = 0
            else:
                self.posibility_list[i] = self.posibility_list[i] + \
                                          temp * (self.posibility_list[i] / (1 - temp))

    def random_pick(self, some_list):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, self.posibility_list):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    def selectNodes(self):  # 选取被攻击的点

        pickingNodes = [0 for i in range(7)]
        selecting = 0
        selecting_list = [0 for i in range(100)]
        i = 0
        while (i < 100):
            selecting_list[i] = i
            i = i + 1
        selecting = 0
        x = 0

        for i in range(100):
            self.temp_posibility_list[i] = self.posibility_list[i]

        while (selecting < 7):
            x = self.random_pick(selecting_list)
            pickingNodes[selecting] = x
            p = selecting_list.index(x)
            # selecting_list.pop(p)
            self.renewPossibilities(p)
            selecting = selecting + 1

        for i in range(100):
            self.posibility_list[i] = self.temp_posibility_list[i]

        print('11111', pickingNodes)

        return pickingNodes

    def envfeedback(self, action_num, state):  # 这里算出reward, 并return 出来
        reward = 0
        value = 0
        self.importFigure()
        self.calculateDegree()
        self.degreeChange()
        self.buildingModel()
        test_nodeD = self.selectNodes()

        # attacker = CrossFire()
        # test_nodeC = attacker.selectNodes(7)

        index = []
        hit = 0

        test_node2 = [0, 58, 7, 6, 3, 2, 10]

        for t in test_nodeD:
            if t in self.action_set[action_num]:
                hit += 1
            #     reward = 0
            # else:
            #     reward = 10

        reward = 1 - hit/len(self.action_set[action_num])
        reward_ = reward * reward

        if state == 0 and action_num > 37:
                reward = value
        elif state == 1:
            if action_num < 38 or action_num > 81:
                reward = value
        elif state == 2:
            if action_num < 82 or action_num > 117:
                reward = value
        elif state == 3:
            if action_num < 118 or action_num > 167:
                reward = value
        elif state == 4:
            if action_num < 168 or action_num > 217:
                reward = value
        elif state == 5:
            if action_num < 218 or action_num > 267:
                reward = value
        elif state == 6:
            if action_num < 268:
                reward = value

        return reward, reward_



class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):

        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, epslion, state_num):
        print('eps:', epslion)
        if epslion > random.random():
            print('111111')
            if state_num == 0:
                action = random.randint(0, 37)
            elif state_num == 1:
                action = random.randint(38, 81)
            elif state_num == 2:
                action = random.randint(82, 117)
            elif state_num == 3:
                action = random.randint(118, 167)
            elif state_num == 4:
                action = random.randint(168, 217)
            elif state_num == 5:
                action = random.randint(218, 267)
            elif state_num == 6:
                action = random.randint(268, 317)
            action = torch.tensor(action)
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)
        else:
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state)
            print(action_probs)
            dist = Categorical(action_probs)
            # print('dist', dist)
            action = dist.sample()
        print('act', action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        # print(action_probs)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            # print('discount', discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            # print('rewards', rewards)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # print("loss", loss.view(1, -1))

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    state_dim = 7
    action_dim = 4
    action_dim = 318  # 4
    # render = False
    solved_reward = 200  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 2000  # max training episodes
    max_timesteps = 10  # max timesteps in one episode
    n_latent_var = 512  # number of variables in hidden layer
    update_timestep = 10  # update policy every n timesteps
    lr = 0.005
    betas = (0.9, 0.999)  # parameter in optimizer
    gamma = 0.9  # discount factor
    K_epochs = 16  # update policy using 1 trajectory for K epochs
    eps_clip = 1  # clip parameter for PPO
    random_seed = 123
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)

    env = Env()
    env.action_process2()  # generate the action set
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)
    # logging variables
    running_reward = 0
    running_reward_ = 0
    reward_sum = 0
    count_sum = 0
    avg_length = 0
    timestep = 0
    epslion = 0.99
    min_eps = 0.1
    count = 0
    count_ = 0
    a_times = [0 for i in range(318)]
    max_count = [4, 4, 4, 2, 2, 2, 1]
    state_count = [0 for i in range(7)]  # state 计数器
    training_time = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        print("#############episdoe", i_episode, '################')
        if epslion > min_eps or i_episode < 1000:
            epslion *= 0.996

        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:

            state1, state_num = env.RequestArrive()  # state为服务请求 possion 到达状态  处理 state 输入
            state = np.array(state1)
            print('---------state:', state,'----------------')

            if 1 not in state1:  # 无服务时不选状态
                continue
            print('state_count', state_count)
            if state_count[state_num] != 0:
                state_count[state_num] += 1
                if state_count[state_num] > max_count[state_num]:
                    state_count[state_num] = 0
                continue

            action = ppo.policy_old.act(state, memory, epslion, state_num)
            state_count[state_num] += 1
            print('action:', action)
            a_times[action] += 1
            reward, reward_ = env.envfeedback(action, state_num)
            training_time += 1
            done = False
            print('reward', reward)
            print('reward_', reward_)
            print(a_times)
            print(state_count)
            # Saving reward and is_terminal:
            memory.rewards.append(reward_)
            memory.is_terminals.append(done)

            # update if its time
            if training_time % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            count += 1
            if reward != 0:
                running_reward_ += reward
                reward_sum += reward_
                count_sum += 1
                count_ += 1
        if count_sum != 0:
            # file_name = 'ppo_reward5.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write(str(reward_sum/count_sum))
            #     file_obj.write("\n")
            reward_sum = 0
            count_sum = 0

        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = running_reward / count

            # file_name = 'ppo_AMP3.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write("reward: ")
            #     file_obj.write(str(running_reward))
            #     file_obj.write('     ')
            #     file_obj.write(str(running_reward_/count_))
            #     file_obj.write("\n")
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            count = 0
            running_reward_ = 0
            count_ = 0
        print(training_time)

if __name__ == '__main__':
