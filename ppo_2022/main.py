import ppo_2022.attack_model as attack_model
import ppo_2022.Environment as Env
import ppo_2022.ppo_mod as ppo_mod
import ppo_2022.utils as utils
import numpy as np

EP_MAX = 1000  # 最大步数
EP_LEN = 128
GAMMA = 0.9  # 折扣因子
A_LR = 0.0001  # A网络的学习速率
C_LR = 0.0002  # c网络的学学习速率
BATCH = 16  # 缓冲池长度
A_UPDATE_STEPS = 10  #
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1  # 状态维度和动作维度
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

def main():
    # env = gym.make('Pendulum-v0').unwrapped

    # 初始化部分
    action_set, function_set = utils.process_action_set()  # 动作集
    ppo = ppo_mod.PPO()
    env = Env.env(action_set)
    a_model_2 = attack_model.attack_model2()
    all_ep_r = []
    times = 0

    # 循环部分
    for ep in range(EP_MAX):
        # s = env.reset()  # 状态初始化
        buffer_s, buffer_a, buffer_r = [], [], []  # 缓存区
        ep_r = 0  # 初始化回合
        reward_sum = 0

        for t in range(EP_LEN):  # 在规定的回合长度内

            state, state_num = env.RequestArrive()  # state为服务请求 possion 到达状态  处理 state 输入
            s = np.array(state)

            a = ppo.choose_action(s)  # 输出action，不需要处理嘛？？？
            action_num = int(a)*100+249
            print("action: ", action_num)

            print('---------state:', state, '----------------')

            if 1 not in state:  # 无服务时不选状态
                continue
            a_node = a_model_2.attack_model_2(times)  # 被攻击的点， 为1-12代号
            a_node_test = [2, 6, 10]
            print("attack_node: ", a_node)
            times += 1

            r, r_ = env.envfeedback(attack_node=a_node, action_num=action_num, state=state_num)  # envfeedback要处理

            reward_sum += r_

            buffer_s.append(s)  # 把这些参量加到缓存区
            buffer_a.append(a)
            buffer_r.append(r_)  # 规范奖励，发现有用的东西

            # s = s_
            ep_r += r

            # 更新PPO
            # 如果buffer收集一个batch或者episode完了
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:  # TODO 这里具体再解释一下
                v_s_ = ppo.get_v(s)  # 计算 discounted reward
                discounted_r = []
                for r in buffer_r[::-1]:  # print(a[::-1]) ### 取从后向前（相反）的元素[1 2 3 4 5]-->[ 5 4 3 2 1 ]
                    v_s_ = r + GAMMA * v_s_  # 状态价值计算
                    discounted_r.append(v_s_)
                discounted_r.reverse()  # 先反方向加入再逆转

                # 清空 buffer
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)  # 更新PPO TODO 这些定义具体是干什么用的呢？
                print("22222222222222222222222")

        file_name = 'data/ppo_reward_fw3.txt'
        with open(file_name, 'a') as file_obj:
            file_obj.write(str(reward_sum))
            file_obj.write('\n')


        # if ep == 0:
        #     all_ep_r.append(ep_r)
        # else:
        #     all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        # print(
        #     'Ep: %i' % ep,
        #     "|Ep_r: %i" % ep_r,
        #     ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        # )
    # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    # plt.xlabel('Episode')  # 回合数
    # plt.ylabel('Moving averaged episode reward')  # 平均回报
    # plt.show()

if __name__ == '__main__':
    main()