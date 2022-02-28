import random

class env:
    def __init__(self, action_set):
        self.state = []
        self.reward = 0
        self.action_set = action_set

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

    def envfeedback(self, attack_node, action_num, state):  # 这里算出reward, 并return 出来
        reward = 0
        value = 0

        # attacker = CrossFire()
        # test_nodeC = attacker.selectNodes(7)

        index = []
        hit = 0

        test_node2 = [0, 58, 7, 6, 3, 2, 10]

        for t in attack_node:
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