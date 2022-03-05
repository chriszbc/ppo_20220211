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
        print("111111", action_num)
        service_flows = self.action_set[action_num]
        service_node = []

        if state == 0:
            service_node.extend(service_flows[0])
        elif state == 1:
            service_node.extend(service_flows[1])
        elif state == 2:
            service_node.extend(service_flows[2])
        elif state == 3:
            service_node.extend(service_flows[0])
            service_node.extend(service_flows[1])
        elif state == 4:
            service_node.extend(service_flows[0])
            service_node.extend(service_flows[2])
        elif state == 5:
            service_node.extend(service_flows[1])
            service_node.extend(service_flows[2])
        elif state == 6:
            service_node.extend(service_flows[0])
            service_node.extend(service_flows[1])
            service_node.extend(service_flows[2])

        print('state_num: ', state)
        print("service_node: ", service_node)
        print("attack_node: ", attack_node)

        hit = 0

        for t in service_node:
            if t in attack_node:
                hit += 1
            #     reward = 0
            # else:
            #     reward = 10
        print("hit: ", hit)

        reward = 1 - hit/len(service_node)
        reward_ = reward * reward * 10

        return reward, reward_