import numpy as np
from fitter import Fitter
import random
from scipy import integrate
from scipy import optimize
from scipy.stats import uniform, norm, expon

# 找Adphist的分布
def distribution_finder(adp_hist):
    thre = 7000 # 如果比它小，就不服从任何分布
    f = Fitter(np.array(adp_hist), distributions=['expon', 'uniform', 'norm'])
    f.fit()
    error = f.summary()['sumsquare_error'][0] # 返回最小的sum square error
    print('error : %s' % (error))
    if error < thre:
        best_dic = f.get_best(method = 'sumsquare_error')
        dis_type = list(best_dic.keys())[0] # 分布种类
        para = list(f.fitted_param[dis_type]) # 最佳分布的参数
    else:
        dis_type = 'none'
        para = []
    print('distibution : %s' % (dis_type))
    return dis_type, para

# 从equation 1中获取自适应时间T
def eqation_1(c_adp, c_att, adp_hist, dis_type, para):
    if para:
        para_0 = round(para[0])
        para_1 = round(para[1])
    if dis_type == 'uniform': # 均匀分布
        def H(x):
            return uniform.cdf(x,loc=para_0,scale=para_1)
        def G(x):
            return x * uniform.pdf(x,loc=para_0,scale=para_1)
    elif dis_type == 'norm':
        def H(x):
            return norm.cdf(x,loc=para_0,scale=para_1)
        def G(x):
            return x * norm.pdf(x,loc=para_0,scale=para_1)
    elif dis_type == 'expon':
        def H(x):
            return expon.cdf(x,loc=para_0,scale=para_1)
        def G(x):
            return x * expon.pdf(x,scale=para_1)
    else: # Adphist does not fit any closed form distribution
        total = 0
        if adp_hist:
            for each in adp_hist:
                total += each
            return total / len(adp_hist)
        else:
            return 4

    def Li(t):
        return (c_adp + c_att * H(t)) / (integrate.quad(G, 0, t)[0]  + t * (1 - H(t)))
    res = optimize.fmin_cg(Li, 1)
    return res[0]

# algorithm 2 : 求最佳适应时间
def decide_adaption_time(c_adp, c_att, adp_hist):
    if adp_hist:
        dis_type, para = distribution_finder(adp_hist)  # Fit a distribution H(t) using Adphist
    else:
        dis_type, para = 'none', []
    T = round(eqation_1(c_adp, c_att, adp_hist, dis_type, para))  # Derive adaptation time T
    return T

adp_hist = []
att_thre = 0.2 # 攻击比例阈值
c_adp, c_att = 5, 5 # 适应成本，攻击成本
node_num = 10 # 节点数

# algorithm 1 : 主循环
while True:
    T = decide_adaption_time(c_adp, c_att, adp_hist)
    print('Adaption time: %s' % (T))
    t_elapsed = 1
    while t_elapsed < T:
        # temp = [1,2,3,4,5] # 攻击者列表
        # att_num = len(temp)
        att_num = random.randint(0,10)
        ratio = att_num / node_num
        if ratio > att_thre:
            break
        else:
            t_elapsed += 1
    
    link_no = random.randint(0, 300)
    # TODO : make adaptions
    print('Making adaption to route %s' % (link_no))

    adp_hist.append(min(t_elapsed - 1, T))  # update Adphist
