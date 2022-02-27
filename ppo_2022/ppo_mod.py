# PPO主要通过限制新旧策略的比率，那些远离旧策略的改变不会发生

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import gym

# 定义一些超级参量
EP_MAX = 1000  # 最大步数
EP_LEN = 200
GAMMA = 0.9  # 折扣因子
A_LR = 0.0001  # A网络的学习速率
C_LR = 0.0002  # c网络的学学习速率
BATCH = 32  # 缓冲池长度
A_UPDATE_STEPS = 10  #
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1  # 状态维度和动作维度
# 作者一共提出了两种方法，一种是Adaptive KL Penalty Coefficient, 另一种是Clipped Surrogate Objective,结果证明，clip的这个方法更好
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL惩罚方法
    dict(name='clip', epsilon=0.2),  # clip方法，发现这个更好
][1]  # 选择优化的方法


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        # 状态变量的结构，多少个不知道，但是有S_DIM这么多维
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        # https://blog.csdn.net/yangfengling1023/article/details/81774580/
        # 全连接层，曾加了一个层，全连接层执行操作 outputs = activation(inputs.kernel+bias) 如果执行结果不想进行激活操作，则设置activation=None
        # self.s:输入该网络层的数据  100：输出的维度大小，改变inputs的最后一维    tf.nn.relu6：激活函数，即神经网络的非线性变化
        with tf.variable_scope('critic'):  # tf.variable_scope：变量作用域
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)  # 输出一个float32的数
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')  # discounted reward
            self.advantage = self.tfdc_r - self.v  # 相当于td_error
            # c网络的loss也就是最小化advantage，先平方再取平均
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            # critic 网络的优化器
            self.ctrain_op = tf.train.AdadeltaOptimizer(C_LR).minimize(self.closs)

        # actor
        # 建立了两个actor网络
        # actor有两个actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        # 输入时state，输出是描述动作分布的mu和sigma
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            # tf.squeeze函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
            # axis可以用来指定要删掉的为1的维度，此处要注意指定的维度必须确保其是1，否则会报错
            # 这里pi.sample是指从pi中取了样，会根据这个样本选择动作，网络优化之后会选择更好的样本出来
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # 这里应该是只删掉了0维的1  TODO 不是很明白
            # print("pi.sample(1)", pi.sample(1))
            # print("self.sample_op", self.sample_op)
        with tf.variable_scope('update_oldpi'):
            # .assign的意思是增加新的一列
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # 动作占位， td_error占位
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):  # 下面这些应该是在实现loss函数
            with tf.variable_scope('surrogate'):
                # surrogate目标函数：
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            # 如果选择了KL惩罚方法,这种方法稍微复杂
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                # tf.distributions.kl_divergence KL散度，也就是两个分布的相对熵，体现的是两个分布的相似程度，熵越小越相似
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            # 如果选择的是clip方法，这个比较简单
            else:  # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))
        # a网络优化器
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # 指定文件来保存图，tensorboard的步骤
        tf.summary.FileWriter("log/", self.sess.graph)
        # 运行会话
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):  # 更新函数
        # print("fuction: update")
        # 先运行俩会话
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        if METHOD['name'] == 'kl_pen':  # TODO 不懂
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            # 这一块是在计算惩罚项的期望，并对系数进行自适应调整
            # 更新后的系数用于下一次策略更新。在这个方案中，我们偶尔会看到KL差异与target显著不同，但是这种情况很少，
            # 因为B会迅速调整。参数1.5和2是启发式选择的，但算法对它们不是很敏感。B的初值是另一个超参数，但在实际应用中
            # 并不重要，因为算法可以快速调整它。
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution
        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def choose_action(self, s):
        # print("fuction: choose_action")
        # print('state: ', s, "s.shape", s.shape)
        s = s[np.newaxis, :]
        # print('s[np.newaxis, :]: ', s, s.shape)
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        print("action",a)
        print("np.clip(a, -2, 2)", np.clip(a, -2, 2))
        return np.clip(a, -2, 2)
        # np.clip: 是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。所有比a_min小的数都会强制变为a_min；

    def get_v(self, s):
        # print("fuction: get_v")
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def _build_anet(self, name, trainable):
        # print("fuction: _build_anet")
        with tf.variable_scope(name):
            # tf.nn.relu    激活函数
            #
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            # tf.nn.tanh 计算l1的双曲正切值  会把值压缩到（-1，1）之间     # 平均值
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # tf.nn.softplus  这个函数的作用是计算激活函数softplus，即log( exp(l1) + 1)。
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)  # 标准差
            # 该函数定义了一个正态分布。  mu是平均值   sigma是标准差
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            print("norm_dist", norm_dist)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        print("I am params", params)
        return norm_dist, params


def main():
    env = gym.make('Pendulum-v0').unwrapped
    ppo = PPO()
    all_ep_r = []
    for ep in range(EP_MAX):
        s = env.reset()  # 状态初始化
        buffer_s, buffer_a, buffer_r = [], [], []  # 缓存区
        ep_r = 0  # 初始化回合
        for t in range(EP_LEN):  # 在规定的回合长度内
            env.render()  # 环境渲染
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)  # 执行动作获取需要的参量
            buffer_s.append(s)  # 把这些参量加到缓存区
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)  # 规范奖励，发现有用的东西
            s = s_
            ep_r += r

            # 更新PPO
            # 如果buffer收集一个batch或者episode完了
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:  # TODO 这里具体再解释一下
                v_s_ = ppo.get_v(s_)  # 计算 discounted reward
                discounted_r = []
                for r in buffer_r[::-1]:  # print(a[::-1]) ### 取从后向前（相反）的元素[1 2 3 4 5]-->[ 5 4 3 2 1 ]
                    v_s_ = r + GAMMA * v_s_  # 状态价值计算
                    discounted_r.append(v_s_)
                discounted_r.reverse()  # 先反方向加入再逆转

                # 清空 buffer
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)  # 更新PPO TODO 这些定义具体是干什么用的呢？
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')  # 回合数
    plt.ylabel('Moving averaged episode reward')  # 平均回报
    plt.show()


if __name__ == '__main__':
    main()

