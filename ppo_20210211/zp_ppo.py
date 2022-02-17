#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
import tensorflow as tf
import numpy as np
import gym
import ns3gym
import z3
import math
import random
from time import time
import numpy as np
from tqdm import trange

car_dis = 150  # the distance of cars
phi = 10  # the threshold of change

file = open("ax.txt", 'w')
file1 = open("attacked_rsu.txt", 'w')
file2 = open("score.txt", 'w')
file3 = open("state.txt", 'w')
def get_matrix(D,s,L,H):    #L is chang,H is gao  ,ue is heng zuo biao ,rsu is zong zuo biao
   re=[]
   for j in range(0,H): 
    temp=[]
    for i in range(L*j,L*j+L):
        temp.append(D[i+s])
    re.append(temp)
   return re    

def Dis(M ,N ,C, New_Link):
    sum = 0
    for i in range(M): 
        for j in range(N):
            sum += abs(C[i][j] - New_Link[i][j])
    return sum

def get_attack_rsu(A):
    re=[]
    for j in range(0,48):
     re.append(0)
     for i in range(50,60):
         if(A[i][j]==1):
              re[j]=1
              break
    return re

# C是链接矩阵（T时隙，ue数量*rsu数量），A是RSU受攻击与否（T时隙，rsu数量*1）
# D_C是每一个车辆的链接表，描述每个车辆在某个时刻对其他车的观测情况（ue数*ue数，当前时隙）
# N是rsu数量，T是当前时隙（int）
record=[] #这个矩阵是一个ue数量*ue数量的表，表中每一个元素是一个1*2的数组，分别代表m和n元素（记录的时隙之前）
record=np.zeros((60, 60,2))
Aij=[]
X=[]
for i in range(60):
    X_1=[]
    for j in range(60):
        X_1.append(20)
    X.append(X_1)
D_C=[]
def mark(C,A,D_C,X,T,M,N):
  #  file3.write(str(record[0][10][0])+str("  ")+str(record[0][10][1])+str("\n"))
    #通过链接矩阵C和受攻击矩阵A获得当前时隙下哪些rsu受到攻击，哪些车辆在链接
    ue_line=np.matmul(C,A)#得到一个由01组成的列向量（ue数量*1，0是该车链接的rsu没有被攻击，1相反）
    #用来临时储存本时隙的链接情况
    temp_rec=[]

    # 把D_C的每一行和ue_line点乘， 如果是1表明该ue既被观察到了同时也在rsu被攻击的时候链接上了，如果是0则可能是为观测或者正常
    for i in range(M):
        temp_rec_1=[]
        for j in range(M):
            temp_rec_1.append(np.dot(D_C[i][j],np.transpose(ue_line[j])))
        temp_rec.append(temp_rec_1)

    #把未观测的值设置成-1，这时临时储存矩阵由-1（未观测），0（观测正常），1（观测异常）组成
    for i in range(M):
        for j in range(N):
            if D_C[i][j]==0:
                temp_rec[i][j]=-1

    # 更新评分表，重新计算m和n
    for i in range(M):
        for j in range(M):
            if temp_rec[i][j]==-1:
                continue
            else:
                record[i][j][1]=record[i][j][1]+1
                if temp_rec[i][j]==1:
                    record[i][j][0]=record[i][j][0]+1

    #print(temp_rec)

    #设置常量参数
    a=70 #a是调整范围的变量
    A0=0.5 #A0是没有交互的初始值

    # 计算当前时隙下Aij（k），并加入评分表（全局变量）
    Aij_temp=np.zeros((M,M))

    for i in range(M):
        for j in range(M):
            phi = np.arctan(record[i][j][1] - a) / np.pi + 0.5
          #  file3.write(str(X[i][j])+" ")
            if record[i][j][1] == 0:
               # file3.write("??????????????????????????????????????????")
                Aij_temp[i][j] = A0
            else:
                Aij_temp[i][j] = phi * record[i][j][0] / record[i][j][1] + (1 - phi) * A0 #
    file3.write("\n")
    temp=[]
    for i in range(0,M):
        k=0
        for j in range(0,M):
           k=k+Aij_temp[j][i]
        k=k/60  
        temp.append(k)
    Aij.append(temp)


def connect_same(i,j,A):
     for k in range(0,48):
          if A[i][k]==1 and A[j][k]==1:
             return True
     return False

# M是ue数量，N是rsu数量，C是链接矩阵，D是距离矩阵，都是M*N的
def remake(M ,N ,C, D):

    # 第一步 找到最大负载的rsu
    # 临时储存点乘之后的矩阵
    temp_max = []

    for i in range(M):
        temp_max_line = []
        for j in range(N):
            temp_max_line.append(C[i][j] * D[i][j])
        temp_max.append(temp_max_line)

    # 储存最大值的索引
    max_index = []

    # 将M*N转置成为N*M
    temp_max = np.transpose(temp_max).tolist()

    # 找出每一行（对应每个rsu）的最大的索引（对应某辆ue）
    for i in range(N):
        max_index.append(temp_max[i].index(max(temp_max[i])))

    #用矩阵相乘求和
    one_matrix=np.ones((M,1))
    temp_sum=np.matmul(temp_max,one_matrix).tolist() #二维矩阵,n*1

    #算出减去最大值之后的和
    temp_sum_after=[]
    for i in range(N):
        temp_sum_after.append(temp_sum[i][0]-temp_max[i][max_index[i]])

    #求出需要切除部分链接的rsu序号
    temp_rsu=temp_sum_after.index(max(temp_sum_after))

    #得到减去最大值之后的矩阵
    temp_minus_after=temp_sum
    temp_minus_after[temp_rsu][0]=temp_minus_after[temp_rsu][0]-temp_max[temp_rsu][max_index[temp_rsu]]

    # 第二步 求解合适的新的链接矩阵

    temp_after_add=[]

    # 把所有的rsu都加上该ue距离各个rsu的值，找到最小的rsu的序号
    for i in range(N):
        temp_after_add.append(temp_minus_after[i][0]+D[max_index[temp_rsu]][i])

    min_index=temp_after_add.index(min(temp_after_add))

    return_list=[]

    #返回值是一个数组，三个值分别为需要断开的ue的值，之前链接的rsu的值，新链接的rsu的值
    return_list.append(max_index[temp_rsu])
    return_list.append(temp_rsu)
    return_list.append(min_index)

    return return_list



__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=0,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
EP_MAX = 1
EP_LEN = 1000
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 20
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 10 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = np.array(s)[np.newaxis,:]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -10, 10)


    def get_v(self, s):
        #if s.ndim < 2:
        s = np.array(s)[np.newaxis,:]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 20 # seconds
stepTime = 0.1  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123}
debug = False
env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

ppo = PPO()
all_ep_r = []
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)
try:
 for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        np.set_printoptions(threshold=np.inf)
        a = ppo.choose_action(s)
        b=[]
        for i in a:
            b.append(int(i)+10)
        print("cycle: ",t," action:  ",b)
        s_, r, done, info = env.step(b)
        print("---obs, reward, done, info: ", s_, r, done, info)

        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    file.close()
    file1.close()
    file2.close()
    file3.close()
    env.close()
    print("Done")
